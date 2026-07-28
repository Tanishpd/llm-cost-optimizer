"""Tests for app/main.py — the M1 gateway skeleton.

Uses FastAPI's TestClient with a stubbed provider swapped into app.state, so
these run with no Ollama and no network. The `integration` marker covers the
handful that need a live server.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.cache.exact import CachedAnswer, ExactCache
from app.cache.keys import build_key
from app.config import Settings
from app.main import app
from app.providers.base import Completion, ProviderTimeout, ProviderUnavailable
from app.telemetry import Telemetry

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
    """Build a TestClient with stubbed dependencies, bypassing the real lifespan.

    Defaults to a cache backed by nothing (`ExactCache(None)`), which is the
    Redis-unavailable path — so most gateway tests implicitly assert the
    gateway still works with caching disabled.
    """

    def _make(
        stub: StubOllama,
        settings: Settings | None = None,
        cache: ExactCache | None = None,
        telemetry: Telemetry | None = None,
    ) -> TestClient:
        c = TestClient(app)
        c.app.state.settings = settings or Settings(_env_file=None)
        c.app.state.ollama = stub
        c.app.state.http = None
        c.app.state.redis = None
        c.app.state.exact_cache = cache if cache is not None else ExactCache(None)
        # Defaults to no-op telemetry — most gateway tests are about the
        # request path, and telemetry must never affect it.
        c.app.state.telemetry = telemetry if telemetry is not None else Telemetry(None)
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
    assert r.json()["model"] == "qwen2.5:0.5b"  # short prompt -> router picks small


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
    assert latency["local_small"] == pytest.approx(120.0)  # from the stub
    assert "total" in latency
    assert latency["total"] >= 0.0
    assert "model_load" not in latency  # warm stub, so no cold-load key


def test_cold_load_recorded_separately_from_tier_latency(client_factory):
    """Cold-start must not distort the tier's latency stats."""
    cold = Completion(content="x", model="qwen2.5:0.5b", tokens_in=1, tokens_out=1,
                      latency_ms=2100.0, load_ms=1957.6)
    c = client_factory(StubOllama(completion=cold))
    latency = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]}).json()["x_hlr"]["latency_ms"]
    assert latency["model_load"] == pytest.approx(1957.6)
    assert "local_small" in latency


def test_headers_expose_tier_and_cost(client_factory):
    """So a proxy or log pipeline can route on them without parsing the body."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
    assert r.headers["X-HLR-Tier"] == "local_small"
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
    ("gateway", "qwen2.5:0.5b"),  # short prompt -> router picks small
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


def test_healthz_200_when_ollama_up_but_redis_down(client_factory):
    """Invariant 4: degraded is not unhealthy. Redis being unreachable disables
    caching; it must not make the gateway look broken."""
    c = client_factory(StubOllama())  # cache defaults to ExactCache(None)
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["dependencies"]["ollama"]["ok"] is True
    assert body["dependencies"]["redis"]["ok"] is False
    assert "still serving" in body["dependencies"]["redis"]["impact"]


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
    assert "M3" in c.get("/").json()["milestone"]


# --------------------------------------------------------------------------
# T0 exact cache, through the full request path
# --------------------------------------------------------------------------


@pytest.fixture
def live_cache():
    """A working in-memory cache, reusing the fake from the cache tests."""
    from tests.test_exact_cache import FakeRedis

    return ExactCache(FakeRedis(), ttl_local=3600, ttl_frontier=86400)


def _ask(client, prompt="What is the capital of France?", **extra):
    return client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": prompt}], **extra}
    )


def test_second_identical_request_is_a_cache_hit(client_factory, live_cache):
    """The M2 demo: same prompt twice, second one never touches the model."""
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)

    first = _ask(c).json()
    assert first["x_hlr"]["tier"] == "local_small"
    assert first["x_hlr"]["cache"]["exact_hit"] is False

    second = _ask(c).json()
    assert second["x_hlr"]["tier"] == "cache_exact"
    assert second["x_hlr"]["path"] == ["cache_exact"]
    assert second["x_hlr"]["cache"]["exact_hit"] is True
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    assert len(stub.calls) == 1, "the model was called twice — cache did not serve"


def test_cache_hit_reports_the_same_savings_as_the_original(client_factory, live_cache):
    """A hit costs nothing but still credits the baseline, so the savings
    number doesn't collapse when traffic is cacheable (D13)."""
    c = client_factory(StubOllama(), cache=live_cache)
    first = _ask(c).json()["x_hlr"]
    second = _ask(c).json()["x_hlr"]
    assert second["cost_usd"] == 0.0
    assert second["baseline_cost_usd"] == first["baseline_cost_usd"]
    assert second["cost_saved_usd"] == first["cost_saved_usd"]


def test_cache_hit_names_the_originating_tier(client_factory, live_cache):
    c = client_factory(StubOllama(), cache=live_cache)
    _ask(c)
    reason = _ask(c).json()["x_hlr"]["reason"]
    assert "exact cache hit" in reason
    assert "local_small" in reason  # where the answer actually came from


def test_different_sampling_params_miss(client_factory, live_cache):
    """Invariant 5, end to end: temperature 0 and 1.0 are different questions."""
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)
    _ask(c, temperature=0.0)
    r = _ask(c, temperature=1.0).json()
    assert r["x_hlr"]["tier"] == "local_small"
    assert len(stub.calls) == 2


def test_different_tier_misses(client_factory, live_cache):
    """An answer from the 0.5B must not be served for a 3B request."""
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)
    c.post("/v1/chat/completions", json={"model": "gateway:small",
                                         "messages": [{"role": "user", "content": "q"}]})
    r = c.post("/v1/chat/completions", json={"model": "gateway:large",
                                             "messages": [{"role": "user", "content": "q"}]})
    assert r.json()["x_hlr"]["tier"] == "local_large"
    assert len(stub.calls) == 2


def test_whitespace_variants_still_hit(client_factory, live_cache):
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)
    _ask(c, "What is  2+2?")
    r = _ask(c, "What is 2+2?")
    assert r.json()["x_hlr"]["tier"] == "cache_exact"
    assert len(stub.calls) == 1


def test_nocache_bypasses_both_read_and_write(client_factory, live_cache):
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)
    for _ in range(2):
        r = c.post("/v1/chat/completions", json={"model": "gateway:nocache",
                                                 "messages": [{"role": "user", "content": "q"}]})
        assert r.json()["x_hlr"]["tier"] == "local_small"
        assert r.json()["x_hlr"]["cache"]["exact_checked"] is False
    assert len(stub.calls) == 2


def test_gateway_serves_normally_when_redis_is_down(client_factory):
    """Invariant 4: no cache, no problem — just no savings from caching."""
    stub = StubOllama()
    c = client_factory(stub)  # ExactCache(None)
    for _ in range(2):
        r = _ask(c)
        assert r.status_code == 200
        assert r.json()["x_hlr"]["tier"] == "local_small"
    assert len(stub.calls) == 2


def test_cache_lookup_latency_is_recorded(client_factory, live_cache):
    c = client_factory(StubOllama(), cache=live_cache)
    _ask(c)
    latency = _ask(c).json()["x_hlr"]["latency_ms"]
    assert "cache_exact" in latency
    assert "local_small" not in latency  # the model never ran


def test_flush_endpoint_clears_the_cache(client_factory, live_cache):
    stub = StubOllama()
    c = client_factory(stub, cache=live_cache)
    _ask(c)
    assert _ask(c).json()["x_hlr"]["tier"] == "cache_exact"

    assert c.delete("/cache").json()["flushed"] == 1
    assert _ask(c).json()["x_hlr"]["tier"] == "local_small"
    assert len(stub.calls) == 2


def test_cache_hit_headers(client_factory, live_cache):
    c = client_factory(StubOllama(), cache=live_cache)
    _ask(c)
    r = _ask(c)
    assert r.headers["X-HLR-Tier"] == "cache_exact"
    assert float(r.headers["X-HLR-Cost-Usd"]) == 0.0
    assert float(r.headers["X-HLR-Saved-Usd"]) > 0


# --------------------------------------------------------------------------
# /metrics reflects real traffic
# --------------------------------------------------------------------------


@pytest.fixture
def live_telemetry():
    from tests.test_telemetry import FakeRedis as TelemetryFakeRedis

    return Telemetry(TelemetryFakeRedis())


def test_metrics_counts_served_tiers(client_factory, live_cache, live_telemetry):
    c = client_factory(StubOllama(), cache=live_cache, telemetry=live_telemetry)
    _ask(c)          # miss  -> local_large
    _ask(c)          # hit   -> cache_exact
    _ask(c)          # hit   -> cache_exact

    m = c.get("/metrics").json()
    assert m["requests"] == 3
    assert m["tiers"]["served"] == {
        "cache_exact": 2, "cache_semantic": 0, "local_small": 1, "local_large": 0, "frontier": 0
    }
    assert sum(m["tiers"]["served"].values()) == m["requests"]  # identity


def test_metrics_hit_rate_matches_traffic(client_factory, live_cache, live_telemetry):
    c = client_factory(StubOllama(), cache=live_cache, telemetry=live_telemetry)
    _ask(c)
    _ask(c)
    m = c.get("/metrics").json()
    assert m["cache"]["hit_rate"] == pytest.approx(0.5)
    assert m["cache"]["exact_hit_rate"] == pytest.approx(0.5)


def test_metrics_savings_reconcile(client_factory, live_cache, live_telemetry):
    """The headline number, checkable rather than trusted."""
    c = client_factory(StubOllama(), cache=live_cache, telemetry=live_telemetry)
    for _ in range(3):
        _ask(c)
    cost = c.get("/metrics").json()["cost"]
    assert cost["actual_usd"] == 0.0  # all local or cached
    assert cost["baseline_usd"] > 0
    assert cost["saved_usd"] == pytest.approx(cost["baseline_usd"] - cost["actual_usd"], abs=1e-9)
    assert cost["saved_pct"] == pytest.approx(100.0)


def test_metrics_recent_log(client_factory, live_cache, live_telemetry):
    c = client_factory(StubOllama(), cache=live_cache, telemetry=live_telemetry)
    _ask(c)
    _ask(c)
    recent = c.get("/metrics/recent").json()["requests"]
    assert len(recent) == 2
    assert recent[0]["tier"] == "cache_exact"  # newest first
    assert recent[1]["tier"] == "local_small"


def test_metrics_reset(client_factory, live_cache, live_telemetry):
    c = client_factory(StubOllama(), cache=live_cache, telemetry=live_telemetry)
    _ask(c)
    c.delete("/metrics")
    assert c.get("/metrics").json()["requests"] == 0


def test_metrics_without_redis_still_renders(client_factory):
    """A dashboard must show "no data", not error."""
    c = client_factory(StubOllama())  # Telemetry(None)
    m = c.get("/metrics").json()
    assert m["requests"] == 0
    assert m["degraded"] == ["redis_unavailable"]
    assert "cost" in m and "tiers" in m


def test_telemetry_failure_does_not_fail_the_request(client_factory, live_cache):
    """A served answer must not be lost because recording it failed."""
    from tests.test_telemetry import FakeRedis as TelemetryFakeRedis

    broken = Telemetry(TelemetryFakeRedis(fail_on={"execute"}))
    c = client_factory(StubOllama(), cache=live_cache, telemetry=broken)
    r = _ask(c)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "Paris"


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


# --------------------------------------------------------------------------
# M3: the router picks the tier
# --------------------------------------------------------------------------


def test_router_sends_short_factual_to_small(client_factory):
    stub = StubOllama()
    c = client_factory(stub)
    r = _ask(c, "What is the capital of France?")
    assert r.json()["x_hlr"]["tier"] == "local_small"
    assert stub.calls[0]["model"] == "qwen2.5:0.5b"


def test_router_sends_reasoning_with_code_to_large(client_factory):
    stub = StubOllama()
    c = client_factory(stub)
    r = _ask(c, "Explain why this deadlocks, step by step:\n```py\nlock.acquire()\nlock.acquire()\n```")
    assert r.json()["x_hlr"]["tier"] == "local_large"
    assert stub.calls[0]["model"] == "qwen2.5:3b"


def test_long_summarization_stays_local(client_factory):
    """The expensive failure mode this guard exists to prevent: long != hard."""
    stub = StubOllama()
    c = client_factory(stub)
    r = _ask(c, "Summarize the following article.\n\n" + ("some prose. " * 900))
    hlr = r.json()["x_hlr"]
    assert hlr["tier"] != "frontier"
    assert hlr["router"]["features"]["n_tokens"] > 600  # genuinely long


def test_router_record_is_in_x_hlr(client_factory):
    c = client_factory(StubOllama())
    router = _ask(c).json()["x_hlr"]["router"]
    assert router["entry_tier"] == "local_small"
    assert isinstance(router["score"], float)
    assert len(router["reason"]) > 20
    assert "n_tokens" in router["features"]


def test_route_latency_is_recorded(client_factory):
    c = client_factory(StubOllama())
    assert "route" in _ask(c).json()["x_hlr"]["latency_ms"]


def test_forced_tier_overrides_router_and_says_so(client_factory):
    """The escape hatch should be visible in the metadata, not silent."""
    c = client_factory(StubOllama())
    r = c.post("/v1/chat/completions", json={
        "model": "gateway:large", "messages": [{"role": "user", "content": "2+2?"}]})
    router = r.json()["x_hlr"]["router"]
    assert router["entry_tier"] == "local_large"
    assert "forced" in router["reason"]
    assert "local_small" in router["reason"]  # what the router would have chosen


def test_frontier_entry_returns_501_naming_the_router_reason(client_factory):
    """Until M4: honest refusal, and it explains why it was headed there."""
    c = client_factory(StubOllama())
    r = _ask(c, "Prove that every finite integral domain is a field. Walk me through "
                "step by step, explain why each lemma holds, discuss trade-offs of the "
                "alternative proofs. What if the ring were infinite? How would you "
                "generalize?\n```\nR = Z/pZ\n```")
    assert r.status_code == 501
    assert "frontier" in r.json()["error"]["message"]
    assert "hardness signals" in r.json()["error"]["message"]


# --------------------------------------------------------------------------
# POST /route
# --------------------------------------------------------------------------


def test_route_endpoint_calls_no_model(client_factory):
    stub = StubOllama()
    c = client_factory(stub)
    r = c.post("/route", json={"messages": [{"role": "user", "content": "What is 2+2?"}]})
    assert r.status_code == 200
    assert stub.calls == [], "the router endpoint called a model"


def test_route_endpoint_returns_decision_and_features(client_factory):
    c = client_factory(StubOllama())
    body = c.post("/route", json={
        "messages": [{"role": "user", "content": "What is the capital of France?"}]}).json()
    assert body["entry_tier"] == "local_small"
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["reason"]) > 20
    assert body["features"]["n_tokens"] > 0
    assert body["cache_preview"]["exact_hit"] is False


def test_route_endpoint_previews_a_cache_hit(client_factory, live_cache):
    c = client_factory(StubOllama(), cache=live_cache)
    _ask(c, "What is the capital of France?")  # populate
    body = c.post("/route", json={
        "messages": [{"role": "user", "content": "What is the capital of France?"}]}).json()
    assert body["cache_preview"]["exact_checked"] is True
    assert body["cache_preview"]["exact_hit"] is True


def test_route_endpoint_agrees_with_the_chat_path(client_factory):
    """The preview must not diverge from what a real request would do."""
    c = client_factory(StubOllama())
    prompt = "Explain why this deadlocks, step by step:\n```py\nx\n```"
    routed = c.post("/route", json={"messages": [{"role": "user", "content": prompt}]}).json()
    served = _ask(c, prompt).json()["x_hlr"]
    assert routed["entry_tier"] == served["tier"] == served["router"]["entry_tier"]
    assert routed["score"] == pytest.approx(served["router"]["score"])
