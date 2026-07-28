"""Tests for app/main.py — the M1 gateway skeleton.

Uses FastAPI's TestClient with a stubbed provider swapped into app.state, so
these run with no Ollama and no network. The `integration` marker covers the
handful that need a live server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.providers.base import Completion, ProviderTimeout, ProviderUnavailable

REAL_LOGPROBS = [
    {"token": "Paris", "logprob": -0.0279, "bytes": [80], "top_logprobs": [
        {"token": "Paris", "logprob": -0.0279}, {"token": " Paris", "logprob": -4.61}]},
]


class StubOllama:
    """Minimal stand-in. Each test decides what the provider does."""

    name = "ollama"

    def __init__(self, *, completion=None, error=None, models=None, healthy=True):
        self._completion = completion
        self._error = error
        self._models = models if models is not None else {
            "qwen2.5:0.5b", "qwen2.5:3b", "nomic-embed-text"
        }
        self._healthy = healthy
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._completion or Completion(
            content="Paris", model=kwargs["model"], tokens_in=41, tokens_out=2,
            logprobs=REAL_LOGPROBS, latency_ms=120.0,
        )

    async def health(self):
        return (True, "v0.30.10") if self._healthy else (False, "unreachable at http://x")

    async def list_models(self):
        return self._models

    async def warm(self, model):
        return 0.0


@pytest.fixture
def client_factory():
    """Build a TestClient with a stubbed provider, bypassing the real lifespan."""

    def _make(stub: StubOllama, settings: Settings | None = None) -> TestClient:
        c = TestClient(app)
        c.app.state.settings = settings or Settings(_env_file=None)
        c.app.state.ollama = stub
        c.app.state.http = None
        return c

    return _make


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_returns_openai_shaped_response(client_factory):
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "capital?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Paris"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"] == {"prompt_tokens": 41, "completion_tokens": 2, "total_tokens": 43}


def test_response_names_the_serving_model_not_gateway(client_factory):
    """Self-describing in logs: you asked "gateway", you get told what ran."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.json()["model"] == "qwen2.5:3b"


def test_every_response_carries_x_hlr(client_factory):
    """Invariant 1."""
    c = client_factory(StubOllama())
    hlr = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()["x_hlr"]
    for key in (
        "trace_id", "tier", "path", "gates", "confidence", "confidence_mode", "reason",
        "router", "cache", "latency_ms", "tokens_in", "tokens_out", "cost_usd",
        "baseline_cost_usd", "cost_saved_usd", "degraded",
    ):
        assert key in hlr, f"missing {key}"


def test_local_tier_is_free_but_baseline_is_not(client_factory):
    """The savings claim, end to end: 41 in / 2 out priced at frontier rates."""
    c = client_factory(StubOllama())
    hlr = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()["x_hlr"]
    assert hlr["cost_usd"] == 0.0
    assert hlr["baseline_cost_usd"] > 0
    assert hlr["cost_saved_usd"] == hlr["baseline_cost_usd"]
    # 41/1e6*2.50 + 2/1e6*10.00 = 0.0001225
    assert hlr["baseline_cost_usd"] == pytest.approx(0.000123, abs=1e-6)


def test_latency_breakdown_is_reported(client_factory):
    """Keys present for the stages that ran, plus `total`.

    `total` is wall-clock rounded to 0.1 ms, so against an instant stub it can
    legitimately be 0.0 — the assertion is non-negativity, not truthiness. The
    per-tier figure comes from the provider and is asserted exactly.
    """
    c = client_factory(StubOllama())
    latency = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()["x_hlr"]["latency_ms"]
    assert latency["local_large"] == pytest.approx(120.0)  # from the stub
    assert "total" in latency
    assert latency["total"] >= 0.0
    assert "model_load" not in latency  # warm stub, so no cold-load key


def test_cold_load_recorded_separately_from_tier_latency(client_factory):
    """Cold-start must not distort the tier's latency stats."""
    cold = Completion(content="x", model="qwen2.5:3b", tokens_in=1, tokens_out=1,
                      latency_ms=2100.0, load_ms=1957.6)
    c = client_factory(StubOllama(completion=cold))
    latency = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()["x_hlr"]["latency_ms"]
    assert latency["model_load"] == pytest.approx(1957.6)
    assert "local_large" in latency


def test_headers_expose_tier_and_cost(client_factory):
    """So a proxy or log pipeline can route on them without parsing the body."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.headers["X-HLR-Tier"] == "local_large"
    assert "X-HLR-Trace-Id" in r.headers
    assert float(r.headers["X-HLR-Cost-Usd"]) == 0.0
    assert float(r.headers["X-HLR-Saved-Usd"]) > 0


def test_logprobs_are_requested(client_factory):
    """M4's gates depend on them; M1 must already be asking."""
    stub = StubOllama()
    c = client_factory(stub)
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert stub.calls[0]["logprobs"] is True
    assert stub.calls[0]["top_logprobs"] == 5


def test_sampling_params_forwarded(client_factory):
    stub = StubOllama()
    c = client_factory(stub)
    c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "q"}],
        "temperature": 0.7, "top_p": 0.9, "max_tokens": 64,
    })
    assert stub.calls[0]["temperature"] == 0.7
    assert stub.calls[0]["top_p"] == 0.9
    assert stub.calls[0]["max_tokens"] == 64


def test_trace_id_is_honoured_when_supplied(client_factory):
    c = client_factory(StubOllama())
    r = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "q"}]},
        headers={"X-HLR-Trace-Id": "abc123"},
    )
    assert r.headers["X-HLR-Trace-Id"] == "abc123"
    assert r.json()["x_hlr"]["trace_id"] == "abc123"


# --------------------------------------------------------------------------
# reserved models
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("gateway", "qwen2.5:3b"),
    ("gateway:small", "qwen2.5:0.5b"),
    ("gateway:large", "qwen2.5:3b"),
])
def test_reserved_models_select_the_tier(client_factory, model, expected):
    stub = StubOllama()
    c = client_factory(stub)
    c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "q"}]})
    assert stub.calls[0]["model"] == expected


def test_frontier_tier_says_it_is_not_built_yet(client_factory):
    """Honest 501 rather than pretending or 500ing."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"model": "gateway:frontier", "messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "tier_not_implemented"
    assert "M4" in r.json()["error"]["message"]


# --------------------------------------------------------------------------
# errors — and they carry x_hlr too
# --------------------------------------------------------------------------


def test_streaming_is_refused_with_a_reason(client_factory):
    """D8: the gate needs the whole response, so streaming would mean streaming
    tokens that may be discarded."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "stream": True})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_streaming"
    assert "D8" in r.json()["error"]["message"]


def test_error_responses_carry_x_hlr(client_factory):
    """A failure should be as traceable as a success."""
    c = client_factory(StubOllama())
    body = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}], "stream": True}).json()
    assert "x_hlr" in body
    assert body["x_hlr"]["trace_id"]
    assert body["x_hlr"]["tier"] is None


def test_ollama_down_returns_503_naming_the_fix(client_factory):
    c = client_factory(StubOllama(error=ProviderUnavailable("unreachable", fix="Run: ollama serve &")))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "all_tiers_down"
    assert "ollama serve" in r.json()["error"]["message"]
    assert "ollama_unavailable" in r.json()["x_hlr"]["degraded"]


def test_timeout_returns_504_not_503(client_factory):
    """Distinct from unreachable: the backend accepted but didn't answer."""
    c = client_factory(StubOllama(error=ProviderTimeout("too slow")))
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "upstream_timeout"


def test_empty_messages_rejected(client_factory):
    c = client_factory(StubOllama())
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 422


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_healthz_200_when_ollama_up_even_though_redis_pending(client_factory):
    """Invariant 4: degraded is not unhealthy. Redis isn't wired until M2, and
    that must not make the gateway look broken."""
    c = client_factory(StubOllama())
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["dependencies"]["ollama"]["ok"] is True


def test_healthz_503_only_when_no_model_tier_reachable(client_factory):
    c = client_factory(StubOllama(healthy=False))
    r = c.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "unhealthy"


def test_healthz_reports_missing_models(client_factory):
    c = client_factory(StubOllama(models={"qwen2.5:0.5b"}))
    cfg = c.get("/healthz").json()["config"]
    assert "qwen2.5:3b" in cfg["missing_models"]
    assert "nomic-embed-text" in cfg["missing_models"]


def test_healthz_reports_active_thresholds(client_factory):
    c = client_factory(StubOllama(), Settings(_env_file=None, confidence_mode="calibrated"))
    cfg = c.get("/healthz").json()["config"]
    assert cfg["confidence_mode"] == "calibrated"
    assert cfg["thresholds"] == {"small": 0.92, "large": 0.85}


def test_root_describes_current_milestone(client_factory):
    c = client_factory(StubOllama())
    assert "M1" in c.get("/").json()["milestone"]


# --------------------------------------------------------------------------
# integration — real server, real model
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_live_end_to_end():
    """Full stack: real lifespan, real Ollama, real cost accounting."""
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "model": "gateway:small",
            "messages": [{"role": "user", "content": "What is the capital of France? One word."}],
            "max_tokens": 10,
        })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].strip()
    hlr = body["x_hlr"]
    assert hlr["tier"] == "local_small"
    assert hlr["path"] == ["local_small"]
    assert hlr["tokens_in"] > 0
    assert hlr["cost_usd"] == 0.0
    assert hlr["cost_saved_usd"] > 0
    assert hlr["latency_ms"]["total"] > 0
