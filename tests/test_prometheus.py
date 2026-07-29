"""Tests for app/prometheus.py and the negotiated `/metrics`.

The format is picky in ways that fail *silently*: a redeclared TYPE makes
Prometheus drop the series, an unescaped quote truncates a line, a missing
trailing newline can lose the last sample. None of that raises anywhere near
the mistake, so it is asserted here.

The other half is the negotiation. `/metrics` has always served JSON for the
dashboard, and `/metrics` is also the name the ecosystem reserves for
exposition format. Both callers have to keep working.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import prometheus
from app.main import app
from app.telemetry import _empty_metrics


def sample_metrics() -> dict:
    m = _empty_metrics()
    m["requests"] = 42
    m["tiers"]["served"] = {"local_small": 30, "frontier": 12}
    m["tiers"]["attempted"] = {"local_small": 42, "local_large": 12, "frontier": 12}
    m["cost"] = {"actual_usd": 0.13, "baseline_usd": 1.2,
                 "saved_usd": 1.07, "saved_pct": 89.2}
    m["latency_ms"] = {"total": {"p50": 28.0, "p95": 2069.0, "n": 42}}
    m["cache"]["guard_rejections"] = {"number_mismatch": 3}
    return m


def parse(text: str) -> dict[str, list[str]]:
    """Group non-comment lines by metric name."""
    out: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        out.setdefault(name, []).append(line)
    return out


# --------------------------------------------------------------------------
# format correctness
# --------------------------------------------------------------------------


def test_every_metric_has_exactly_one_help_and_type():
    """Prometheus *drops* a series whose TYPE is redeclared. A duplicate here
    is a metric that silently vanishes, not one that errors."""
    text = prometheus.render(sample_metrics())
    helps = [ln.split()[2] for ln in text.splitlines() if ln.startswith("# HELP")]
    types = [ln.split()[2] for ln in text.splitlines() if ln.startswith("# TYPE")]
    assert len(helps) == len(set(helps)), "duplicate HELP lines"
    assert len(types) == len(set(types)), "duplicate TYPE lines"
    assert set(helps) == set(types)


def test_every_sample_belongs_to_a_declared_metric():
    text = prometheus.render(sample_metrics())
    declared = {ln.split()[2] for ln in text.splitlines() if ln.startswith("# TYPE")}
    assert set(parse(text)) <= declared


def test_counters_are_named_total_and_gauges_are_not():
    """Convention, and tooling leans on it: `rate()` over something that is not
    monotonic is a meaningless number."""
    text = prometheus.render(sample_metrics())
    for line in text.splitlines():
        if not line.startswith("# TYPE"):
            continue
        _, _, name, kind = line.split()
        if kind == "counter":
            assert name.endswith("_total"), f"counter not named _total: {name}"
        else:
            assert not name.endswith("_total"), f"gauge named _total: {name}"


def test_output_ends_with_a_newline():
    """Required by the format. Without it the last sample can be dropped."""
    assert prometheus.render(sample_metrics()).endswith("\n")


def test_a_redeclared_type_is_refused_rather_than_emitted():
    r = prometheus.Registry()
    r.add("thing_total", 1, help="x", type="counter")
    with pytest.raises(ValueError, match="both counter and gauge"):
        r.add("thing_total", 2, help="x", type="gauge")


def test_invalid_names_are_refused():
    r = prometheus.Registry()
    with pytest.raises(ValueError, match="metric name"):
        r.add("has-a-dash", 1, help="x")
    with pytest.raises(ValueError, match="label name"):
        r.add("fine", 1, help="x", labels={"has-a-dash": "v"})


@pytest.mark.parametrize("raw,expected", [
    ('quote"inside', 'quote\\"inside'),
    ("back\\slash", "back\\\\slash"),
    ("new\nline", "new\\nline"),
    ("plain", "plain"),
])
def test_label_values_are_escaped(raw, expected):
    """An unescaped quote truncates the line and corrupts everything after it
    on that series."""
    assert prometheus.escape_label_value(raw) == expected


def test_a_hostile_label_value_still_produces_parseable_output():
    """Tier names come from config, and `reason` strings from degradations —
    neither is guaranteed to be tame."""
    r = prometheus.Registry()
    r.add("thing", 1, help="x", labels={"tier": 'a"b\\c\nd'})
    line = [ln for ln in r.render().splitlines() if not ln.startswith("#")][0]
    assert line.count('"') == 2 + line.count('\\"')
    assert "\n" not in line.replace("\\n", "")


def test_special_floats_use_prometheus_spelling():
    r = prometheus.Registry()
    r.add("a", float("inf"), help="x")
    r.add("b", float("-inf"), help="x")
    r.add("c", float("nan"), help="x")
    text = r.render()
    assert "hlr_a +Inf" in text and "hlr_b -Inf" in text and "hlr_c NaN" in text


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------


def test_the_numbers_that_matter_are_exported():
    text = prometheus.render(sample_metrics())
    for name in ("hlr_requests_total", "hlr_cost_usd_total", "hlr_cost_saved_usd_total",
                 "hlr_requests_by_tier_total", "hlr_tier_attempted_total",
                 "hlr_latency_ms", "hlr_cache_guard_rejections_total"):
        assert name in text, f"{name} not exported"


def test_attempted_exceeds_served_and_both_are_visible():
    """The gap between them is the escalation cost — the thing this gateway
    exists to manage. Exporting only `served` would hide it."""
    parsed = parse(prometheus.render(sample_metrics()))
    served = {ln for ln in parsed["hlr_requests_by_tier_total"]}
    attempted = {ln for ln in parsed["hlr_tier_attempted_total"]}
    assert any('tier="local_small"} 30' in ln for ln in served)
    assert any('tier="local_small"} 42' in ln for ln in attempted)


def test_health_and_breaker_state_are_numeric():
    """Prometheus cannot alert on the string "open"."""
    text = prometheus.render(
        sample_metrics(),
        health={"status": "degraded",
                "dependencies": {"redis": {"ok": False}, "ollama": {"ok": True}},
                "config": {"circuits": {"frontier": "open", "local_small": "closed"}}})
    assert 'hlr_circuit_state{tier="frontier"} 2' in text
    assert 'hlr_circuit_state{tier="local_small"} 0' in text
    assert 'hlr_dependency_up{dependency="redis"} 0' in text


def test_an_unknown_breaker_state_is_minus_one_not_a_crash():
    text = prometheus.render(sample_metrics(),
                             health={"config": {"circuits": {"x": "melted"}}})
    assert 'hlr_circuit_state{tier="x"} -1' in text


def test_tenant_spend_and_the_enforcement_flag_are_exported():
    """`budget_enforced 0` is the alertable one: a cap exists and is not being
    applied, which means spend is uncapped right now."""
    text = prometheus.render(
        sample_metrics(),
        tenants={"usage": {"acme": {"spent_usd": 4.2, "limit_usd": 5.0, "enforced": False}}})
    assert 'hlr_tenant_spend_usd{tenant="acme"} 4.2' in text
    assert 'hlr_tenant_budget_usd{tenant="acme"} 5' in text
    assert 'hlr_tenant_budget_enforced{tenant="acme"} 0' in text


def test_an_uncapped_tenant_emits_no_budget_series():
    """A missing series is how Prometheus spells "no budget". Emitting 0 would
    read as a cap of zero dollars."""
    text = prometheus.render(
        sample_metrics(),
        tenants={"usage": {"free": {"spent_usd": 1.0, "limit_usd": None, "enforced": False}}})
    assert "hlr_tenant_budget_usd" not in text
    assert 'hlr_tenant_spend_usd{tenant="free"} 1' in text


def test_it_renders_the_redis_is_down_payload():
    """The payload someone will actually be staring at during an incident."""
    text = prometheus.render(_empty_metrics(degraded=["redis_unavailable"]))
    assert 'hlr_degraded{reason="redis_unavailable"} 1' in text
    assert "hlr_requests_total 0" in text


def test_a_cold_gateway_renders_without_dividing_by_zero():
    assert prometheus.render(_empty_metrics()).endswith("\n")


# --------------------------------------------------------------------------
# negotiation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("accept,expected", [
    ("text/plain;version=0.0.4;q=0.5,*/*;q=0.1", True),      # Prometheus
    ("application/openmetrics-text;version=1.0.0", True),
    ("text/plain", True),
    ("application/json", False),
    ("*/*", False),                                           # curl's default
    ("text/html,application/xhtml+xml", False),               # a browser
    (None, False),
    ("", False),
])
def test_negotiation(accept, expected):
    """`*/*` must keep getting JSON: curl sends it by default, and every
    existing caller would break the day they stopped sending an explicit
    Accept."""
    assert prometheus.wants_prometheus(accept) is expected


@pytest.fixture
def client(monkeypatch):
    from app.config import get_settings
    monkeypatch.setenv("WARM_MODELS_ON_STARTUP", "false")
    monkeypatch.setenv("AUDIT_ENABLED", "false")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_metrics_still_returns_json_by_default(client):
    r = client.get("/metrics")
    assert r.headers["content-type"].startswith("application/json")
    assert "requests" in r.json()


def test_metrics_returns_exposition_when_a_scraper_asks(client):
    r = client.get("/metrics", headers={"Accept": "text/plain;version=0.0.4,*/*;q=0.1"})
    assert r.headers["content-type"].startswith("text/plain")
    assert "# TYPE hlr_requests_total counter" in r.text


def test_the_explicit_endpoint_never_negotiates(client):
    """For the human with curl who will not think about Accept headers."""
    r = client.get("/metrics/prometheus")
    assert r.headers["content-type"].startswith("text/plain")
    assert "hlr_requests_total" in r.text


def test_the_scrape_reports_the_same_health_as_healthz(client):
    """Two health checks that could disagree would be worse than one."""
    health = client.get("/healthz").json()
    scrape = client.get("/metrics/prometheus").text
    for dep, info in health["dependencies"].items():
        assert f'hlr_dependency_up{{dependency="{dep}"}} {1 if info["ok"] else 0}' in scrape


def test_metrics_is_still_authenticated(client):
    """It exposes cost data. A scraper can send a bearer token; making the
    endpoint public so scraping is easier would be the wrong trade."""
    from app.security import PUBLIC_PATHS
    assert "/metrics" not in PUBLIC_PATHS
    assert "/metrics/prometheus" not in PUBLIC_PATHS
