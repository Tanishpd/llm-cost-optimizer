"""Tests for app/providers/ollama.py.

Unit tests use httpx.MockTransport — no network, no running Ollama, so they run
in CI. The `integration` marker guards the handful that need a live server:

    pytest -m "not integration"    # unit only
    pytest -m integration          # needs `ollama serve` + models pulled
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.providers.base import Completion, Provider, ProviderTimeout, ProviderUnavailable
from app.providers.ollama import OllamaProvider

# A real /api/chat response captured from qwen2.5:0.5b (M0, 2026-07-28).
REAL_CHAT_RESPONSE = {
    "model": "qwen2.5:0.5b",
    "created_at": "2026-07-28T09:14:51.1Z",
    "message": {"role": "assistant", "content": "Paris"},
    "done": True,
    "done_reason": "stop",
    "logprobs": [
        {
            "token": "Paris",
            "logprob": -0.027870038524270058,
            "bytes": [80, 97, 114, 105, 115],
            "top_logprobs": [
                {"token": "Paris", "logprob": -0.027870038524270058},
                {"token": " Paris", "logprob": -4.606834411621094},
            ],
        }
    ],
    "total_duration": 2175459583,
    "load_duration": 1957589125,
    "prompt_eval_count": 41,
    "prompt_eval_duration": 180478000,
    "eval_count": 2,
    "eval_duration": 21569000,
}


def make_provider(handler) -> OllamaProvider:
    transport = httpx.MockTransport(handler)
    return OllamaProvider(httpx.AsyncClient(transport=transport), base_url="http://ollama:11434")


# --------------------------------------------------------------------------
# protocol conformance
# --------------------------------------------------------------------------


def test_ollama_satisfies_the_provider_protocol():
    """The seam that keeps backends swappable (invariant 8)."""
    p = make_provider(lambda r: httpx.Response(200, json={}))
    assert isinstance(p, Provider)


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_parses_a_real_response():
    p = make_provider(lambda r: httpx.Response(200, json=REAL_CHAT_RESPONSE))
    c = await p.chat(model="qwen2.5:0.5b", messages=[{"role": "user", "content": "capital?"}])
    assert c.content == "Paris"
    assert c.model == "qwen2.5:0.5b"
    assert c.tokens_in == 41
    assert c.tokens_out == 2
    assert len(c.logprobs) == 1
    assert c.finish_reason == "stop"


@pytest.mark.asyncio
async def test_eval_count_may_exceed_logprob_entries():
    """The M0 wrinkle: the stopping token is counted in eval_count but produces
    no logprob entry. Anything asserting equality would be wrong."""
    p = make_provider(lambda r: httpx.Response(200, json=REAL_CHAT_RESPONSE))
    c = await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert c.tokens_out == 2
    assert len(c.logprobs) == 1
    assert c.tokens_out != len(c.logprobs)


@pytest.mark.asyncio
async def test_chat_always_disables_streaming():
    """The gate cannot decide until the whole answer exists, and the
    non-streaming path is what accumulates all chunks' logprobs."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=REAL_CHAT_RESPONSE)

    await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert seen["stream"] is False


@pytest.mark.asyncio
async def test_logprobs_only_requested_when_asked():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=REAL_CHAT_RESPONSE)

    p = make_provider(handler)
    await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert "logprobs" not in payloads[0]

    await p.chat(
        model="m", messages=[{"role": "user", "content": "q"}], logprobs=True, top_logprobs=5
    )
    assert payloads[1]["logprobs"] is True
    assert payloads[1]["top_logprobs"] == 5


@pytest.mark.asyncio
async def test_top_logprobs_clamped_to_server_bound():
    """Ollama 400s outside 0-20. Clamp rather than let a caller trigger it."""
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=REAL_CHAT_RESPONSE)

    await make_provider(handler).chat(
        model="m", messages=[{"role": "user", "content": "q"}], logprobs=True, top_logprobs=99
    )
    assert payloads[0]["top_logprobs"] == 20


@pytest.mark.asyncio
async def test_sampling_params_are_forwarded():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=REAL_CHAT_RESPONSE)

    await make_provider(handler).chat(
        model="m",
        messages=[{"role": "user", "content": "q"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=128,
    )
    opts = payloads[0]["options"]
    assert opts["temperature"] == 0.7
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 128


@pytest.mark.asyncio
async def test_load_duration_tracked_separately_from_latency():
    """Cold-start time must not silently pollute latency percentiles."""
    p = make_provider(lambda r: httpx.Response(200, json=REAL_CHAT_RESPONSE))
    c = await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert c.load_ms == pytest.approx(1957.6, abs=0.5)  # 1957589125 ns
    assert c.was_cold_load is True


@pytest.mark.asyncio
async def test_warm_load_is_not_flagged_cold():
    warm = dict(REAL_CHAT_RESPONSE, load_duration=1_000_000)  # 1 ms
    p = make_provider(lambda r: httpx.Response(200, json=warm))
    c = await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert c.was_cold_load is False


# --------------------------------------------------------------------------
# failure modes — invariant 4: degrade, don't fail
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_model_names_the_pull_command():
    """A missing model should skip its tier with an actionable reason, not
    take the cascade down."""
    p = make_provider(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(ProviderUnavailable) as exc:
        await p.chat(model="qwen2.5:3b", messages=[{"role": "user", "content": "q"}])
    assert "qwen2.5:3b" in str(exc.value)
    assert exc.value.fix == "ollama pull qwen2.5:3b"


@pytest.mark.asyncio
async def test_unreachable_server_names_the_fix():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    p = make_provider(handler)
    with pytest.raises(ProviderUnavailable) as exc:
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert exc.value.fix == "Run: ollama serve &"


@pytest.mark.asyncio
async def test_timeout_is_distinct_from_unavailable():
    """Callers treat these differently: a timeout may be worth retrying at the
    next tier, an unreachable server means skip every local tier."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    with pytest.raises(ProviderTimeout):
        await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])


@pytest.mark.asyncio
async def test_server_error_surfaces_detail():
    p = make_provider(lambda r: httpx.Response(500, text="internal boom"))
    with pytest.raises(ProviderUnavailable, match="500"):
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])


@pytest.mark.asyncio
async def test_health_never_raises():
    """/healthz must report, not explode — and the detail should be useful
    enough to act on, which means naming the URL that failed."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    ok, detail = await make_provider(handler).health()
    assert ok is False
    assert "unreachable" in detail
    assert "http://ollama:11434" in detail  # says *which* endpoint, not just "error"


@pytest.mark.asyncio
async def test_health_reports_version_when_up():
    ok, detail = await make_provider(
        lambda r: httpx.Response(200, json={"version": "0.30.10"})
    ).health()
    assert ok is True
    assert detail == "v0.30.10"


@pytest.mark.asyncio
async def test_warm_never_raises():
    """A failed warm must not stop the gateway starting."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert await make_provider(handler).warm("m") == 0.0


# --------------------------------------------------------------------------
# embeddings — the /api/embed vs /api/embeddings shape trap
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_uses_the_current_endpoint_and_shape():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    vectors = await make_provider(handler).embed(model="nomic-embed-text", texts="hello")
    assert paths == ["/api/embed"]  # not the deprecated /api/embeddings
    assert vectors == [[0.1, 0.2, 0.3]]  # always list-of-vectors


@pytest.mark.asyncio
async def test_embed_batches():
    p = make_provider(lambda r: httpx.Response(200, json={"embeddings": [[0.1], [0.2]]}))
    assert len(await p.embed(model="m", texts=["a", "b"])) == 2


@pytest.mark.asyncio
async def test_legacy_singular_shape_is_rejected_loudly():
    """The deprecated endpoint returns {"embedding": [...]} — flat, singular.
    Silently accepting it would produce a wrong-shaped vector index."""
    p = make_provider(lambda r: httpx.Response(200, json={"embedding": [0.1, 0.2]}))
    with pytest.raises(ProviderUnavailable, match="unexpected /api/embed response shape"):
        await p.embed(model="m", texts="hello")


# --------------------------------------------------------------------------
# server introspection
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_normalizes_latest_suffix():
    """`nomic-embed-text` and `nomic-embed-text:latest` should both match, so
    callers don't have to guess which form was pulled."""
    payload = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:3b"}]}
    names = await make_provider(lambda r: httpx.Response(200, json=payload)).list_models()
    assert {"nomic-embed-text", "nomic-embed-text:latest", "qwen2.5:3b"} <= names


@pytest.mark.asyncio
async def test_version():
    p = make_provider(lambda r: httpx.Response(200, json={"version": "0.30.10"}))
    assert await p.version() == "0.30.10"


# --------------------------------------------------------------------------
# integration — needs a live Ollama
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_chat_with_logprobs():
    async with httpx.AsyncClient(timeout=120.0) as client:
        p = OllamaProvider(client)
        c = await p.chat(
            model="qwen2.5:0.5b",
            messages=[{"role": "user", "content": "What is the capital of France? One word."}],
            logprobs=True,
            top_logprobs=5,
            max_tokens=10,
        )
    assert c.content.strip()
    assert c.logprobs, "logprobs missing — see RESEARCH.md §2"
    assert all({"token", "logprob"} <= set(e) for e in c.logprobs)
    assert c.tokens_in > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_embed_dimensions():
    async with httpx.AsyncClient(timeout=120.0) as client:
        vectors = await OllamaProvider(client).embed(model="nomic-embed-text", texts="hello")
    assert len(vectors) == 1
    assert len(vectors[0]) == 768  # documented dimension
