"""Tests that the pieces are actually connected to each other.

Unit tests verify components. Nothing verified they were *plugged in*, and
that gap produced the second-worst bug in the project: `CONFIDENCE_MODE=
calibrated` was a documented, settable knob, `confidence.py` sat at 99% line
coverage, every calibrator test passed — and `main.py` never loaded
`eval/calibrator.json`, so the gates silently used raw thresholds while
`x_hlr.confidence_mode` reported `calibrated`. Coverage cannot see that,
because every line ran; they just ran disconnected.

Two defences here, and the second is the interesting one:

1. **Every `app.state.X` the request path reads is set by the lifespan.**
   Catches a handler reaching for something startup forgot.

2. **Every setting is either proven to change behaviour, or explicitly
   declared inert.** This is the generalised form of the calibrator bug — a
   knob that does nothing — and the declaration is the point: adding a new
   setting now forces a decision rather than allowing silence.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "app/main.py"


@pytest.fixture
def booted(monkeypatch):
    """A client that ran the *real* lifespan, not a hand-assembled state.

    Every other gateway test stubs `app.state` directly, which is what makes
    them fast and hermetic — and is exactly why none of them could notice that
    the lifespan was not wiring something up.
    """
    monkeypatch.setenv("WARM_MODELS_ON_STARTUP", "false")
    monkeypatch.setenv("AUDIT_ENABLED", "false")
    # Point at a closed port. These boot the *real* lifespan, which would
    # otherwise attach to whatever Redis is running on the developer's machine
    # — and then an exact-cache hit from an earlier run serves the request
    # without calling the model, so `stub.calls` is empty and the test fails
    # for a reason that has nothing to do with what it asserts. CI has no Redis
    # and would have passed regardless, which is the worst version of this:
    # green in CI, red on the one machine anyone debugs on.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


# --------------------------------------------------------------------------
# 1. everything the handler reads, the lifespan sets
# --------------------------------------------------------------------------


def _state_attributes_read_by_the_app() -> set[str]:
    """Parse `app/main.py` for every `…app.state.X` read.

    AST rather than a regex so a renamed attribute cannot hide behind
    formatting.
    """
    tree = ast.parse(MAIN.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        # …<something>.state.<name>
        if isinstance(value, ast.Attribute) and value.attr == "state":
            found.add(node.attr)
    return found


def test_every_state_attribute_the_app_reads_is_set_at_startup(booted):
    """A handler reaching for `app.state.thing` that the lifespan never set is
    a 500 on the happy path — and only for the request that reaches that
    branch, which may be a rare one."""
    read = _state_attributes_read_by_the_app()
    # Set per-request by middleware, not by the lifespan.
    per_request = {"trace_id", "principal", "tenant"}

    missing = sorted(
        name for name in read - per_request
        if not hasattr(booted.app.state, name)
    )
    assert not missing, f"the request path reads state the lifespan never sets: {missing}"


def test_the_lifespan_wires_every_collaborator_the_cascade_needs(booted):
    """Named explicitly, because these are the ones whose absence degrades
    quietly rather than crashing: an empty `calibrators` dict silently means
    raw thresholds, and a missing budget ledger silently means no caps."""
    state = booted.app.state
    for name in ("settings", "ollama", "frontier", "exact_cache", "semantic_cache",
                 "telemetry", "singleflight", "breakers", "retry_policy", "auth",
                 "rate_limiter", "limits", "calibrators", "tenants", "budgets", "audit"):
        assert hasattr(state, name), f"lifespan never set app.state.{name}"


def test_the_calibrators_the_lifespan_loads_reach_the_cascade():
    """The exact regression. `execute()` takes a `calibrators` argument; for
    two milestones `main.py` never passed it."""
    source = MAIN.read_text()
    call = re.search(r"outcome = await execute\((.*?)\n    \)", source, re.S)
    assert call, "could not find the execute() call — has it been renamed?"
    assert "calibrators=" in call.group(1), (
        "main.py calls execute() without passing calibrators; calibrated mode "
        "would silently fall back to raw thresholds"
    )


# --------------------------------------------------------------------------
# 2. every setting either does something, or says it does not
# --------------------------------------------------------------------------

# Settings with no *directly* observable effect on a single request. Each entry
# needs a reason, because "I could not think of a test" and "this genuinely
# does nothing to a response" look identical in a list of names.
INERT: dict[str, str] = {
    # Infrastructure addresses: exercised by every integration test, and
    # asserting on them here would just restate the fixture.
    "ollama_base_url": "connection target; covered by integration tests",
    "redis_url": "connection target; covered by integration tests",
    "frontier_base_url": "connection target; covered by tests/test_frontier.py",
    "frontier_api_key": "sent upstream; asserted in tests/test_frontier.py",
    # Tuning knobs whose effect is timing or memory, not response content.
    "ollama_num_ctx": "passed to Ollama; no observable change in our response",
    "ollama_keep_alive": "affects Ollama's residency, not our output",
    "ollama_max_loaded_models": "advisory to the operator; we never send it",
    "ollama_timeout_s": "timing only; the timeout path is tested in resilience",
    "ollama_connect_timeout_s": "timing only",
    "frontier_timeout_s": "timing only",
    "warm_models_on_startup": "startup latency only",
    "breaker_cooldown_s": "timing; behaviour covered in tests/test_resilience.py",
    "retry_base_delay_s": "timing only",
    "log_level": "log verbosity",
    "log_format": "log shape; covered in tests/test_logging_setup.py",
    # Eval-only; never read on the serving path.
    "eval_use_real_frontier": "eval harness only",
    "eval_judge_base_url": "eval harness only",
    "eval_judge_model": "eval harness only",
    "eval_judge_api_key": "eval harness only",
    "eval_judge_max_tokens": "eval harness only; covered in tests/test_eval.py",
    "eval_reference_max_tokens": "eval harness only; covered in tests/test_eval.py",
    "eval_min_call_interval_s": "eval harness only; covered in tests/test_eval.py",
    "eval_judge_price_in": "eval harness only; covered in tests/test_eval.py",
    "eval_judge_price_out": "eval harness only; covered in tests/test_eval.py",
    "eval_judge_reasoning_effort": "eval harness only; covered in tests/test_eval.py",
    "eval_reference_reasoning_effort": "eval harness only; covered in tests/test_eval.py",
    # Audit sizing: behaviour is covered in tests/test_audit.py against the
    # class directly, which is cheaper than driving it through a request.
    "audit_path": "covered in tests/test_audit.py",
    "audit_max_bytes": "covered in tests/test_audit.py",
    "audit_keep_files": "covered in tests/test_audit.py",
    "audit_queue_size": "covered in tests/test_audit.py",
    "audit_fsync": "durability, not response content",
    "audit_enabled": "covered in tests/test_audit.py",
    # Tenancy: covered in tests/test_budgets.py end to end.
    "tenants": "covered in tests/test_budgets.py",
    "default_daily_budget_usd": "covered in tests/test_budgets.py",
    "max_daily_spend_usd": "covered in tests/test_budgets.py (the ceiling section)",
    "cache_scope": "covered in tests/test_gateway.py (cross-tenant caching)",
    "shadow_sample_rate": "covered in tests/test_shadow.py",
    "shadow_divergence_threshold": "covered in tests/test_shadow.py",
    "calibrator_path": "covered by the startup-guard tests in test_gateway.py",
    "semantic_max_entries": "index capacity; covered in tests/test_semantic_cache.py",
    "cache_ttl_local": "TTL; covered in tests/test_exact_cache.py",
    "cache_ttl_frontier": "TTL; covered in tests/test_exact_cache.py",
    "confidence_strategy": "the self_consistency path is not implemented (D3)",
    "top_logprobs": "request parameter; asserted in tests/test_ollama_provider.py",
    "require_auth": "startup guard; asserted in tests/test_gateway.py",
    "gateway_api_keys": "covered in tests/test_security.py",
    "rate_limit_per_minute": "covered in tests/test_ratelimit.py",
    "max_body_bytes": "covered in tests/test_security.py",
    "max_prompt_chars": "covered in tests/test_security.py",
    "max_messages": "covered in tests/test_security.py",
    "max_output_tokens": "covered in tests/test_security.py",
    "breaker_failure_threshold": "covered in tests/test_resilience.py",
    "retry_attempts": "covered in tests/test_resilience.py",
    "frontier_price_in": "covered in tests/test_costs.py",
    "frontier_price_out": "covered in tests/test_costs.py",
    "small_model": "covered in tests/test_gateway.py",
    "large_model": "covered in tests/test_gateway.py",
    "embed_model": "covered in tests/test_semantic_cache.py",
    "frontier_model": "covered in tests/test_gateway.py",
    "semantic_enabled": "covered in tests/test_semantic_cache.py",
    "semantic_threshold": "covered in tests/test_semantic_cache.py",
    "semantic_max_chars": "covered in tests/test_semantic_cache.py",
    "router_small_max_tokens": "covered in tests/test_router.py",
    "router_escalate_min_tokens": "covered in tests/test_router.py",
    "threshold_small": "covered in tests/test_cascade.py",
    "threshold_large": "covered in tests/test_cascade.py",
    "target_small": "covered in tests/test_cascade.py (calibrated mode)",
    "target_large": "covered in tests/test_cascade.py (calibrated mode)",
    "confidence_mode": "covered in tests/test_cascade.py and the startup guard",
    "stream_chunk_chars": "covered in tests/test_streaming.py",
    "default_max_tokens": "covered below — found dead by this very test",
    "default_temperature": "covered below — found dead by this very test",
    "eval_max_spend_usd": "eval harness only; covered in tests/test_eval.py",
    "rate_limit_scope": "covered in tests/test_gateway.py (M24, both scopes asserted)",
}


def test_every_setting_is_accounted_for():
    """The meta-test, and the real defence.

    A new knob must either be exercised somewhere or be explicitly declared
    inert *with a reason*. That is what the calibrated-mode bug needed and did
    not have: `CONFIDENCE_MODE` was settable, documented, and connected to
    nothing, and no test was obliged to notice.
    """
    fields = set(Settings.model_fields)
    unaccounted = sorted(fields - set(INERT))
    assert not unaccounted, (
        f"settings with no recorded coverage: {unaccounted}. Add a test, or add "
        "an INERT entry saying why it cannot be observed."
    )


def test_the_inert_list_has_not_rotted():
    """An entry for a setting that no longer exists means the list is being
    edited by copy-paste rather than by thinking about it."""
    stale = sorted(set(INERT) - set(Settings.model_fields))
    assert not stale, f"INERT names settings that no longer exist: {stale}"


def test_every_inert_entry_gives_a_reason():
    blank = sorted(k for k, v in INERT.items() if not v or len(v) < 10)
    assert not blank, f"INERT entries with no real reason: {blank}"


# --------------------------------------------------------------------------
# the response contract survives a real boot
# --------------------------------------------------------------------------


def test_a_booted_gateway_serves_the_documented_contract(booted):
    """One end-to-end assertion through the real lifespan, with no model call.

    `/route` runs the router and the cache lookups and nothing else, so this
    stays hermetic while still proving the wiring holds under a real boot.
    """
    r = booted.post("/route", json={"messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["entry_tier"] in ("local_small", "local_large", "frontier")
    assert body["reason"]


def test_healthz_reports_every_subsystem_after_a_real_boot(booted):
    """`/healthz` is how an operator learns what is degraded. A subsystem the
    lifespan added but never reported is invisible exactly when it matters."""
    cfg = booted.get("/healthz").json()["config"]
    for key in ("confidence_mode", "thresholds", "circuits", "rate_limit_distributed",
                "tenants", "budgets_enforced", "audit"):
        assert key in cfg, f"/healthz does not report {key}"


# --------------------------------------------------------------------------
# DEFAULT_MAX_TOKENS / DEFAULT_TEMPERATURE
#
# The meta-test above found these on its first run: defined in config.py,
# documented in .env.example *and* PLAN.md ("When the request omits
# `max_tokens`"), and read by nothing. The real defaults were hardcoded in
# schemas.py and happened to match, so setting them did nothing and nobody
# could tell. Exactly the calibrated-mode bug, in a quieter corner.
# --------------------------------------------------------------------------


def _client_with(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setenv("WARM_MODELS_ON_STARTUP", "false")
    monkeypatch.setenv("AUDIT_ENABLED", "false")
    # The stub provider has no embedder, and the semantic tier is not what
    # these assert on.
    monkeypatch.setenv("SEMANTIC_ENABLED", "false")
    # Point at a closed port. These boot the *real* lifespan, which would
    # otherwise attach to whatever Redis is running on the developer's machine
    # — and then an exact-cache hit from an earlier run serves the request
    # without calling the model, so `stub.calls` is empty and the test fails
    # for a reason that has nothing to do with what it asserts. CI has no Redis
    # and would have passed regardless, which is the worst version of this:
    # green in CI, red on the one machine anyone debugs on.
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    return TestClient(app)


def test_default_max_tokens_applies_when_the_request_omits_it(monkeypatch):
    from tests.test_gateway import StubOllama

    with _client_with(monkeypatch, DEFAULT_MAX_TOKENS=77) as c:
        stub = StubOllama()
        c.app.state.ollama = stub
        c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
        assert stub.calls[0]["max_tokens"] == 77
    get_settings.cache_clear()


def test_an_explicit_max_tokens_still_wins(monkeypatch):
    """The default is a fallback, not an override."""
    from tests.test_gateway import StubOllama

    with _client_with(monkeypatch, DEFAULT_MAX_TOKENS=77) as c:
        stub = StubOllama()
        c.app.state.ollama = stub
        c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "q"}], "max_tokens": 5})
        assert stub.calls[0]["max_tokens"] == 5
    get_settings.cache_clear()


def test_default_temperature_applies_when_the_request_omits_it(monkeypatch):
    from tests.test_gateway import StubOllama

    with _client_with(monkeypatch, DEFAULT_TEMPERATURE=0.7) as c:
        stub = StubOllama()
        c.app.state.ollama = stub
        c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
        assert stub.calls[0]["temperature"] == pytest.approx(0.7)
    get_settings.cache_clear()


def test_route_and_chat_agree_on_the_cache_key_under_a_changed_default(monkeypatch):
    """Both endpoints hash temperature and max_tokens into the cache key
    (invariant 5). If only one applied the operator defaults they would compute
    different keys for the same request, and `/route`'s `exact_hit` preview
    would disagree with the real lookup — a preview that lies is worse than no
    preview.

    A fake cache is installed after boot so this exercises a *working* cache
    while staying hermetic. Asserting against an unreachable Redis would pass
    either way, because both sides would simply miss — which is exactly how the
    first version of this test managed to assert nothing.
    """
    from app.cache.exact import ExactCache
    from tests.test_budgets import FakeRedis
    from tests.test_gateway import StubOllama

    class KVRedis(FakeRedis):
        async def set(self, key, value, ex=None):
            self.store[key] = value
            return True

    with _client_with(monkeypatch, DEFAULT_MAX_TOKENS=99) as c:
        c.app.state.ollama = StubOllama()
        c.app.state.exact_cache = ExactCache(KVRedis())
        body = {"messages": [{"role": "user", "content": "same question"}]}

        assert c.post("/route", json=body).json()["cache_preview"]["exact_hit"] is False
        assert c.post("/v1/chat/completions", json=body).status_code == 200
        after = c.post("/route", json=body).json()["cache_preview"]

    get_settings.cache_clear()
    assert after["exact_hit"] is True, (
        "/route computed a different cache key than the chat handler — the two "
        "disagree about the operator's sampling defaults"
    )
