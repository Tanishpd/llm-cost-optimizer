"""Tests for the frontier provider and the mock service.

The mock's job is to be *faithful* — if it diverges from a real provider's
envelope, the gateway's cost math is wrong in a way nothing else would catch.
So these tests check the envelope as hard as the client.
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.providers.base import Provider, ProviderTimeout, ProviderUnavailable
from app.providers.frontier import FrontierProvider
from mock_frontier.main import app as mock_app

OPENAI_RESPONSE = {
    "id": "chatcmpl-mock-abc123",
    "object": "chat.completion",
    "created": 1785155391,
    "model": "gpt-4o-mock",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "A frontier answer."},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ],
    "usage": {"prompt_tokens": 41, "completion_tokens": 156, "total_tokens": 197},
}


def make_provider(handler) -> FrontierProvider:
    return FrontierProvider(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        base_url="http://frontier:8001/v1",
        api_key="test-key",
    )


# --------------------------------------------------------------------------
# provider
# --------------------------------------------------------------------------


def test_frontier_satisfies_the_provider_protocol():
    assert isinstance(make_provider(lambda r: httpx.Response(200, json=OPENAI_RESPONSE)), Provider)


async def test_parses_the_openai_envelope():
    p = make_provider(lambda r: httpx.Response(200, json=OPENAI_RESPONSE))
    c = await p.chat(model="gpt-4o-mock", messages=[{"role": "user", "content": "q"}])
    assert c.content == "A frontier answer."
    assert c.model == "gpt-4o-mock"
    assert c.tokens_in == 41
    assert c.tokens_out == 156
    assert c.finish_reason == "stop"


async def test_no_logprobs_are_requested_or_returned():
    """The last tier has nothing above it, so there is nothing for a gate to
    decide — and most providers do not expose them anyway."""
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=OPENAI_RESPONSE)

    c = await make_provider(handler).chat(
        model="m", messages=[{"role": "user", "content": "q"}], logprobs=True, top_logprobs=5
    )
    assert "logprobs" not in sent
    assert c.logprobs == []


async def test_bearer_token_is_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json=OPENAI_RESPONSE)

    await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert seen["auth"] == "Bearer test-key"


async def test_posts_to_the_chat_completions_path():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=OPENAI_RESPONSE)

    await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert seen["url"] == "http://frontier:8001/v1/chat/completions"


async def test_never_streams():
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=OPENAI_RESPONSE)

    await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert sent["stream"] is False


# --------------------------------------------------------------------------
# failures — each names what to do about it
# --------------------------------------------------------------------------


async def test_bad_key_is_actionable():
    p = make_provider(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(ProviderUnavailable) as exc:
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert exc.value.fix == "Check FRONTIER_API_KEY"


async def test_rate_limit_is_distinguished():
    p = make_provider(lambda r: httpx.Response(429, json={"error": {"message": "slow down"}}))
    with pytest.raises(ProviderUnavailable, match="429"):
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])


async def test_unreachable_names_how_to_start_the_mock():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderUnavailable) as exc:
        await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert "mock_frontier" in exc.value.fix


async def test_timeout_is_its_own_type():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderTimeout):
        await make_provider(handler).chat(model="m", messages=[{"role": "user", "content": "q"}])


async def test_error_message_is_extracted_from_the_envelope():
    p = make_provider(
        lambda r: httpx.Response(500, json={"error": {"message": "upstream exploded"}})
    )
    with pytest.raises(ProviderUnavailable, match="upstream exploded"):
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])


async def test_non_json_error_falls_back_to_text():
    """A provider having an outage may not return JSON at all."""
    p = make_provider(lambda r: httpx.Response(502, text="<html>Bad Gateway</html>"))
    with pytest.raises(ProviderUnavailable, match="Bad Gateway"):
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])


async def test_empty_choices_is_an_error_not_an_empty_answer():
    """Serving "" as a successful answer would be worse than failing."""
    p = make_provider(lambda r: httpx.Response(200, json={"choices": [], "usage": {}}))
    with pytest.raises(ProviderUnavailable, match="no choices"):
        await p.chat(model="m", messages=[{"role": "user", "content": "q"}])


async def test_health_never_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    ok, detail = await make_provider(handler).health()
    assert ok is False and "ConnectError" in detail


# --------------------------------------------------------------------------
# the mock service — faithfulness is the whole requirement
# --------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    return TestClient(mock_app)


def test_mock_returns_a_complete_openai_envelope(mock_client):
    r = mock_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o-mock", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer k"},
    )
    assert r.status_code == 200
    body = r.json()
    assert {"id", "object", "created", "model", "choices", "usage"} <= set(body)
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    choice = body["choices"][0]
    assert {"index", "message", "finish_reason"} <= set(choice)
    assert choice["message"]["role"] == "assistant"


def test_mock_usage_block_is_truthful(mock_client):
    """The gateway's entire cost math derives from this — a round fabricated
    number here would make every savings figure meaningless."""
    r = mock_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "x" * 400}]},
        headers={"Authorization": "Bearer k"},
    )
    usage = r.json()["usage"]
    assert usage["prompt_tokens"] > 50  # scales with the actual prompt
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_mock_respects_max_tokens(mock_client):
    r = mock_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 5},
        headers={"Authorization": "Bearer k"},
    )
    assert r.json()["usage"]["completion_tokens"] <= 5


def test_mock_content_admits_it_is_canned(mock_client):
    """A reader must not mistake mock output for a model answer — that is how
    fake numbers end up in a README."""
    r = mock_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "q"}]},
        headers={"Authorization": "Bearer k"},
    )
    assert "mock" in r.json()["choices"][0]["message"]["content"].lower()


def test_mock_requires_authorization(mock_client):
    """A real provider rejects a missing key; the mock does too, so the
    gateway's auth wiring is exercised rather than assumed."""
    r = mock_client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}
    )
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_mock_rejects_streaming(mock_client):
    r = mock_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "q"}], "stream": True},
        headers={"Authorization": "Bearer k"},
    )
    assert r.status_code == 400


def test_mock_error_envelope_matches_openai(mock_client):
    r = mock_client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code in (401, 422)


def test_mock_healthz_reports_its_config(mock_client):
    cfg = mock_client.get("/healthz").json()["config"]
    assert {"latency_ms", "jitter_ms", "failure_rate"} <= set(cfg)


# --------------------------------------------------------------------------
# provider against the real mock service
# --------------------------------------------------------------------------


async def test_provider_parses_the_real_mock():
    """End to end through both halves, so a divergence between the client's
    expectations and the mock's envelope cannot hide."""
    transport = httpx.ASGITransport(app=mock_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        provider = FrontierProvider(client, base_url="http://mock/v1", api_key="k")
        completion = await provider.chat(
            model="gpt-4o-mock", messages=[{"role": "user", "content": "What is 2+2?"}]
        )
    assert completion.content
    assert completion.tokens_in > 0
    assert completion.tokens_out > 0
    assert completion.logprobs == []


async def test_a_429_carries_the_providers_retry_hint(mock_client):
    """The body is where providers say how long to wait — Gemini's 429 reads
    "Please retry in 23.7s". Discarding it forced blind exponential backoff,
    which cost one eval run ~15 minutes on a single call (30+60+120+240+480s)
    waiting out a limit that clears in under a minute."""
    provider = make_provider(lambda r: httpx.Response(
        429, json={"error": {"message": "Quota exceeded. Please retry in 23.7s."}}))

    with pytest.raises(ProviderUnavailable) as exc:
        await provider.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert "retry in 23.7s" in str(exc.value)


async def test_the_retry_hint_survives_a_verbose_429_body(mock_client):
    """Gemini buries the delay after ~430 characters of boilerplate.

    Clipping the body before searching it discards the one field a caller
    needs — which is what happened, and it killed a run at the same prompt
    twice with 15 minutes of blind backoff each time.
    """
    import httpx
    import pytest

    from app.providers.base import ProviderUnavailable

    verbose = ("You exceeded your current quota, please check your plan and billing "
               "details. For more information on this error, head to: "
               "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your "
               "current usage, head to: https://ai.dev/rate-limit. "
               "* Quota exceeded for metric: "
               "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
               "limit: 20, model: gemini-3-flash\nPlease retry in 23.695367191s.")
    assert verbose.index("Please retry in") > 300, "fixture no longer reproduces the bug"

    provider = make_provider(lambda r: httpx.Response(
        429, json={"error": {"code": 429, "message": verbose}}))

    with pytest.raises(ProviderUnavailable) as exc:
        await provider.chat(model="m", messages=[{"role": "user", "content": "q"}])
    assert "retry in 23.695367191s" in str(exc.value)
