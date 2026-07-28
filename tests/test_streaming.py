"""Tests for app/streaming.py and the `stream: true` request path.

Two things are being pinned here, and only one of them is the SSE format.

The other is the *property that makes buffer-then-stream correct*: the cascade
is unchanged, so a request that escalates still escalates, and the client never
sees a token from a tier whose gate rejected it. A streaming implementation
that got the framing right and leaked a discarded 0.5B answer would pass a
naive format test and be wrong.
"""

from __future__ import annotations

import json
import math

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditLog
from app.budgets import BudgetLedger, TenantRegistry
from app.cache.exact import ExactCache
from app.cache.semantic import SemanticCache
from app.cache.singleflight import SingleFlight
from app.config import Settings
from app.main import app
from app.providers.base import Completion, ProviderUnavailable
from app.ratelimit import RedisRateLimiter
from app.resilience import BreakerRegistry, RetryPolicy
from app.schemas import ChatCompletionResponse, HlrMetadata
from app.security import ApiKeyAuth, RequestLimits
from app.streaming import DONE, split_content, sse_stream
from app.telemetry import Telemetry
from tests.test_gateway import StubFrontier, StubOllama

# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "",
    "one",
    "a b c",
    "  leading spaces",
    "trailing spaces   ",
    "line one\nline two\n\nline four",
    "  \n\t mixed \n whitespace \t\t ",
    "a" * 200,
    "short " * 50,
    "unicode: café — naïve — 日本語 — 🎉 emoji",
    "```python\ndef f(x):\n    return x + 1\n```",
])
@pytest.mark.parametrize("chunk_chars", [1, 5, 24, 1000])
def test_chunking_is_lossless(text, chunk_chars):
    """The single property that matters: deltas concatenate back to the source.

    A chunker that drops a newline produces text that looks fine in a terminal
    and is broken in rendered markdown — exactly the bug that would ship.
    """
    assert "".join(split_content(text, chunk_chars)) == text


def test_no_chunk_exceeds_the_budget_unless_one_word_does():
    text = "the quick brown fox jumps over the lazy dog"
    for chunk in split_content(text, 12):
        assert len(chunk) <= 12 + 1  # +1 for the trailing space of the last word


def test_a_word_longer_than_the_budget_is_not_split():
    """Cutting a word in half would corrupt multi-byte characters and produce
    garbage in any client that renders incrementally."""
    long_word = "supercalifragilisticexpialidocious"
    chunks = split_content(f"hi {long_word} bye", 5)
    assert any(long_word in c for c in chunks)
    assert "".join(chunks) == f"hi {long_word} bye"


def test_zero_chunk_chars_sends_one_chunk():
    assert split_content("a b c", 0) == ["a b c"]


def test_empty_content_produces_no_content_chunks():
    assert split_content("", 24) == []


# --------------------------------------------------------------------------
# SSE framing
# --------------------------------------------------------------------------


def _response(content="Paris is the capital.", tier="local_small") -> ChatCompletionResponse:
    return ChatCompletionResponse.build(
        content=content, model="qwen2.5:0.5b", tokens_in=10, tokens_out=5,
        hlr=HlrMetadata(trace_id="t123", tier=tier, path=[tier], reason="because"),
    )


def _frames(text: str) -> list[dict]:
    """Parse an SSE body into payload dicts, excluding the [DONE] sentinel."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: ") and block != "data: [DONE]":
            out.append(json.loads(block[6:]))
    return out


def test_stream_starts_with_a_role_chunk_and_ends_with_done():
    events = list(sse_stream(_response()))
    assert events[-1] == DONE
    first = json.loads(events[0][6:])
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert first["object"] == "chat.completion.chunk"


def test_deltas_reassemble_into_the_full_answer():
    text = "Paris is the capital of France."
    events = "".join(sse_stream(_response(content=text)))
    assembled = "".join(
        f["choices"][0]["delta"].get("content", "")
        for f in _frames(events) if f["choices"]
    )
    assert assembled == text


def test_x_hlr_rides_on_the_terminal_chunk():
    """Invariant 1 holds for streaming too — but only the last chunk can carry
    it, because `path` is not final until the cascade is."""
    frames = _frames("".join(sse_stream(_response())))
    carriers = [f for f in frames if "x_hlr" in f]
    assert len(carriers) == 1
    assert carriers[-1] is frames[-1]
    assert carriers[0]["choices"][0]["finish_reason"] == "stop"
    assert carriers[0]["x_hlr"]["tier"] == "local_small"


def test_every_chunk_shares_one_id_and_model():
    frames = _frames("".join(sse_stream(_response())))
    assert len({f["id"] for f in frames}) == 1
    assert {f["model"] for f in frames} == {"qwen2.5:0.5b"}


def test_usage_chunk_only_when_asked():
    without = _frames("".join(sse_stream(_response())))
    assert not any("usage" in f for f in without)

    with_usage = _frames("".join(sse_stream(_response(), include_usage=True)))
    last = with_usage[-1]
    assert last["choices"] == []          # OpenAI's shape for the usage frame
    assert last["usage"]["total_tokens"] == 15


def test_empty_answer_still_produces_a_valid_stream():
    events = list(sse_stream(_response(content="")))
    assert events[-1] == DONE
    frames = _frames("".join(events))
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------------------
# through the gateway
# --------------------------------------------------------------------------


@pytest.fixture
def stream_client():
    def _make(stub, *, frontier=None, cache=None, settings=None,
              tenants=None, budget_redis=None):
        c = TestClient(app)
        c.app.state.settings = settings or Settings(_env_file=None)
        c.app.state.ollama = stub
        c.app.state.http = None
        c.app.state.redis = None
        c.app.state.exact_cache = cache if cache is not None else ExactCache(None)
        c.app.state.telemetry = Telemetry(None)
        c.app.state.frontier = frontier or StubFrontier()
        c.app.state.semantic_cache = SemanticCache(None, enabled=False)
        c.app.state.singleflight = SingleFlight(None)
        c.app.state.auth = ApiKeyAuth(set())
        c.app.state.rate_limiter = RedisRateLimiter(None, limit=0)
        c.app.state.limits = RequestLimits()
        c.app.state.breakers = BreakerRegistry(failure_threshold=10**6)
        c.app.state.retry_policy = RetryPolicy(attempts=1)
        # Disabled: the request path must be exercised without any test
        # writing files. tests/test_audit.py covers the enabled path.
        c.app.state.audit = AuditLog("unused", enabled=False)
        c.app.state.calibrators = {}
        # No tenants and no default cap: budgets are opt-in, so the
        # shipped default must be what most tests exercise.
        c.app.state.tenants = tenants or TenantRegistry()
        c.app.state.budgets = BudgetLedger(budget_redis, c.app.state.tenants)
        return c
    return _make


def _post(c, content="q", **extra):
    return c.post("/v1/chat/completions",
        json={"messages": [{"role": "user", "content": content}], **extra})


def test_stream_true_returns_sse_not_json(stream_client):
    c = stream_client(StubOllama())
    r = _post(c, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.endswith(DONE)


def test_streamed_and_buffered_answers_are_identical(stream_client):
    """The only difference must be framing. Same tier, same text, same cost."""
    plain = _post(stream_client(StubOllama()), "capital?").json()
    streamed = _frames(_post(stream_client(StubOllama()), "capital?", stream=True).text)

    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in streamed if f["choices"])
    assert text == plain["choices"][0]["message"]["content"]
    assert streamed[-1]["x_hlr"]["tier"] == plain["x_hlr"]["tier"]
    assert streamed[-1]["x_hlr"]["cost_saved_usd"] == plain["x_hlr"]["cost_saved_usd"]


def test_hlr_headers_survive_streaming(stream_client):
    """A property of buffering: the routing decision is final before the first
    byte, so it can ride in headers. A true streaming proxy could not do this."""
    r = _post(stream_client(StubOllama()), stream=True)
    assert r.headers["X-HLR-Tier"] == "local_small"
    assert "X-HLR-Saved-Usd" in r.headers
    assert r.headers["X-Accel-Buffering"] == "no"


def test_a_streamed_request_that_escalates_never_leaks_the_rejected_answer(stream_client):
    """The load-bearing test for D8.

    The 0.5B answers with low confidence and is escalated. Its text must not
    appear anywhere in the stream — that is the entire reason the answer is
    buffered before the first byte instead of streamed as it is generated.
    """
    weak = Completion(
        content="WRONG_LOW_CONFIDENCE_ANSWER", model="qwen2.5:0.5b",
        tokens_in=10, tokens_out=4,
        # geo_mean 0.2 — under both raw thresholds, so both gates escalate.
        logprobs=[{"token": "t", "logprob": math.log(0.2),
                   "top_logprobs": [{"token": "t", "logprob": math.log(0.2)}]}] * 4,
        latency_ms=100.0,
    )
    c = stream_client(
        StubOllama(completion=weak),
        frontier=StubFrontier(content="[the frontier answer]"),
    )
    r = _post(c, "q", stream=True)

    assert "WRONG_LOW_CONFIDENCE_ANSWER" not in r.text
    frames = _frames(r.text)
    text = "".join(f["choices"][0]["delta"].get("content", "")
                   for f in frames if f["choices"])
    assert text == "[the frontier answer]"
    assert frames[-1]["x_hlr"]["tier"] == "frontier"
    # ...and the metadata still admits both local tiers were paid for.
    assert frames[-1]["x_hlr"]["path"] == ["local_small", "local_large", "frontier"]


def test_a_failure_is_a_normal_http_error_not_a_broken_stream(stream_client):
    """Nothing is sent until everything has succeeded, so an error is still a
    503 with a JSON body — never a half-written stream."""
    c = stream_client(
        StubOllama(error=ProviderUnavailable("unreachable", fix="ollama serve &")),
        frontier=StubFrontier(error=ProviderUnavailable("frontier down")),
    )
    r = _post(c, "q", stream=True)
    assert r.status_code == 503
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["code"] == "all_tiers_down"


def test_cached_answers_stream_too(stream_client):
    """The cache path and the cascade path go through one response helper, so
    they cannot drift on this."""
    class FakeRedis:
        def __init__(self):
            self.store = {}

        async def get(self, k):
            return self.store.get(k)

        async def set(self, k, v, ex=None):
            self.store[k] = v
            return True

    cache = ExactCache(FakeRedis())
    c1 = stream_client(StubOllama(), cache=cache)
    _post(c1, "capital?")                                    # populate

    c2 = stream_client(StubOllama(), cache=cache)
    r = _post(c2, "capital?", stream=True)                   # served from cache
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = _frames(r.text)
    assert frames[-1]["x_hlr"]["tier"] == "cache_exact"
    assert r.headers["X-HLR-Tier"] == "cache_exact"


def test_stream_options_include_usage_through_the_gateway(stream_client):
    c = stream_client(StubOllama())
    r = _post(c, "q", stream=True, stream_options={"include_usage": True})
    frames = _frames(r.text)
    assert frames[-1]["usage"]["prompt_tokens"] == 41


def test_unknown_stream_options_keys_are_tolerated(stream_client):
    """OpenAI adds fields; a gateway that 422s on them breaks real clients."""
    c = stream_client(StubOllama())
    r = _post(c, "q", stream=True, stream_options={"include_usage": False, "future_key": 1})
    assert r.status_code == 200


def test_chunk_size_is_configurable(stream_client):
    one_shot = Settings(_env_file=None, stream_chunk_chars=0)
    c = stream_client(StubOllama(), settings=one_shot)
    frames = _frames(_post(c, "q", stream=True).text)
    content_frames = [f for f in frames
                      if f["choices"] and "content" in f["choices"][0]["delta"]]
    assert len(content_frames) == 1


# --------------------------------------------------------------------------
# integration — the real OpenAI SDK, a real model
#
# Every hermetic test above verifies the bytes we *believe* the contract to
# be. This one verifies the contract itself, by handing the gateway to the
# client whose expectations actually matter. It is the only test that could
# catch a framing detail we got confidently and consistently wrong.
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_real_openai_sdk_can_stream_from_the_gateway():
    """Needs a running gateway and a live Ollama.

    Point it elsewhere with `GATEWAY_URL=http://localhost:8020 pytest -m
    integration` — useful when the default port holds an older build, which is
    exactly how this test first "failed".
    """
    import os

    openai = pytest.importorskip("openai")

    base = os.getenv("GATEWAY_URL", "http://localhost:8000").rstrip("/")
    client = openai.OpenAI(base_url=f"{base}/v1", api_key="unused")
    stream = client.chat.completions.create(
        model="gateway",
        messages=[{"role": "user", "content": "Name three primary colours."}],
        max_tokens=30,
        stream=True,
        stream_options={"include_usage": True},
    )

    text, usage, roles = "", None, 0
    for chunk in stream:
        assert chunk.object == "chat.completion.chunk"
        if chunk.choices and chunk.choices[0].delta.role:
            roles += 1
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
        if chunk.usage:
            usage = chunk.usage

    assert text.strip(), "the SDK reassembled an empty answer"
    assert roles == 1, "exactly one role chunk, as the SDK expects"
    assert usage is not None and usage.total_tokens > 0
