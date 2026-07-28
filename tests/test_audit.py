"""Tests for app/audit.py — the durable routing record.

The claims worth testing are the ones that would be embarrassing to get wrong
in a thing called an *audit trail*:

- it does not lose records it accepted,
- it does not silently lose the ones it rejected,
- it never blocks the request path,
- it survives a restart, which is the entire reason it exists rather than the
  capped Redis stream telemetry already had.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.audit import AuditLog, audit_entry
from app.main import app
from app.telemetry import RequestRecord


def record(trace_id="t1", **kw) -> RequestRecord:
    base = dict(
        trace_id=trace_id, tier="local_small", path=["local_small"],
        reason="because", latency_ms={"total": 1.0}, tokens_in=10, tokens_out=5,
        cost_micro_usd=0, baseline_micro_usd=100, created_at=time.time(),
    )
    base.update(kw)
    return RequestRecord(**base)


async def drained(log: AuditLog) -> None:
    """Wait for the writer to catch up, without sleeping a fixed amount."""
    for _ in range(200):
        if log._queue.empty():
            await asyncio.sleep(0.01)      # let the in-flight batch land
            return
        await asyncio.sleep(0.005)
    raise AssertionError("audit queue never drained")


def lines(path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


async def test_a_record_lands_on_disk_as_one_json_line(tmp_path):
    log = AuditLog(tmp_path, flush_interval_s=0.01)
    await log.start()
    try:
        log.write(audit_entry(record(), tenant="acme", model="qwen2.5:0.5b"))
        await drained(log)
    finally:
        await log.stop()

    written = lines(tmp_path / log._path.name)
    assert len(written) == 1
    assert written[0]["trace_id"] == "t1"
    assert written[0]["tenant"] == "acme"
    assert written[0]["model"] == "qwen2.5:0.5b"


async def test_every_accepted_record_is_written(tmp_path):
    log = AuditLog(tmp_path, flush_interval_s=0.01)
    await log.start()
    try:
        for i in range(500):
            assert log.write(audit_entry(record(f"t{i}"), tenant="acme"))
        await drained(log)
    finally:
        await log.stop()

    written = lines(log._path)
    assert len(written) == 500
    assert {w["trace_id"] for w in written} == {f"t{i}" for i in range(500)}


async def test_shutdown_drains_what_is_queued(tmp_path):
    """Records accepted but not yet written must not vanish on a clean stop —
    losing them is exactly the failure a durable trail exists to prevent."""
    log = AuditLog(tmp_path, flush_interval_s=10.0)
    await log.start()
    for i in range(50):
        log.write(audit_entry(record(f"t{i}"), tenant="acme"))
    await log.stop()                        # no drain wait: stop() must do it

    assert len(lines(log._path)) == 50


async def test_records_survive_a_restart(tmp_path):
    """The whole point. Telemetry's capped stream cannot do this."""
    first = AuditLog(tmp_path, flush_interval_s=0.01)
    await first.start()
    first.write(audit_entry(record("before"), tenant="acme"))
    await first.stop()

    second = AuditLog(tmp_path, flush_interval_s=0.01)
    await second.start()
    second.write(audit_entry(record("after"), tenant="acme"))
    await drained(second)
    await second.stop()

    # Same PID in a test process, so both runs append to one file.
    assert [w["trace_id"] for w in lines(second._path)] == ["before", "after"]


async def test_write_never_blocks_even_when_the_queue_is_full(tmp_path):
    """`write()` is sync on purpose: a caller must not be able to await a disk.

    Blocking here would violate invariant 3 under exactly the conditions
    (sustained load) where it would hurt most.
    """
    log = AuditLog(tmp_path, queue_size=5, flush_interval_s=100.0)
    await log.start()
    try:
        started = time.perf_counter()
        for i in range(1000):
            log.write(audit_entry(record(f"t{i}"), tenant="acme"))
        elapsed = time.perf_counter() - started
    finally:
        await log.stop()

    assert elapsed < 1.0, "write() blocked"
    assert log.stats.dropped > 0


async def test_drops_are_counted_not_silent(tmp_path):
    """An audit trail with unreported holes invites conclusions drawn from
    absence, which is worse than having no trail at all."""
    log = AuditLog(tmp_path, queue_size=2, flush_interval_s=100.0)
    await log.start()
    try:
        accepted = sum(log.write(audit_entry(record(f"t{i}"), tenant="a")) for i in range(50))
    finally:
        await log.stop()

    assert log.stats.dropped == 50 - accepted
    assert log.stats.dropped > 0


# --------------------------------------------------------------------------
# rotation
# --------------------------------------------------------------------------


async def test_rotation_bounds_disk_use(tmp_path):
    log = AuditLog(tmp_path, max_bytes=2_000, keep_files=3, flush_interval_s=0.01)
    await log.start()
    try:
        for i in range(400):
            log.write(audit_entry(record(f"t{i}"), tenant="acme"))
        await drained(log)
    finally:
        await log.stop()

    files = sorted(p.name for p in tmp_path.glob("audit-*"))
    assert log.stats.rotations > 0
    # Live file plus at most `keep` rotated siblings.
    assert len(files) <= 4, files
    assert not (tmp_path / f"{log._path.name}.4").exists(), "kept more than asked"


async def test_rotation_drops_the_oldest_and_keeps_the_newest(tmp_path):
    """Rotation must discard from the far end. Dropping the newest would make
    the trail useless for the question it exists to answer.

    Read across every file, not just the live one: rotation happens *after* a
    write, so immediately afterwards the newest batch is in `.1` and the live
    file is empty. That is normal, and a test that only looked at the live file
    would fail on correct behaviour.
    """
    log = AuditLog(tmp_path, max_bytes=1_500, keep_files=2, flush_interval_s=0.01)
    await log.start()
    try:
        # Drain between writes so each lands in its own batch. Rotation is
        # per-batch (see `_write_batch`), so queueing all 300 at once produces
        # exactly one rotation and discards nothing — which is correct
        # behaviour, and would make this test assert nothing.
        for i in range(300):
            log.write(audit_entry(record(f"t{i}"), tenant="acme"))
            await drained(log)
    finally:
        await log.stop()

    seen = set()
    for f in tmp_path.glob("audit-*"):
        seen |= {e["trace_id"] for e in lines(f)}

    assert log.stats.rotations > 0, "the test did not actually rotate"
    assert "t299" in seen, "the newest record was rotated away"
    assert "t0" not in seen, "nothing was actually discarded"


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------


async def test_disabled_writes_nothing_and_creates_no_directory(tmp_path):
    """A dev checkout must not sprout an audit directory just by importing."""
    target = tmp_path / "never"
    log = AuditLog(target, enabled=False)
    await log.start()
    assert log.write({"a": 1}) is False
    await log.stop()
    assert not target.exists()


async def test_an_unwritable_directory_disables_auditing_rather_than_failing(tmp_path):
    """Invariant 4. But loudly: someone asked for a trail and is not getting
    one, so `enabled` flips and the error is recorded."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")

    log = AuditLog(blocker / "sub", flush_interval_s=0.01)
    await log.start()
    assert log.enabled is False
    assert log.stats.last_error
    assert log.write({"a": 1}) is False
    await log.stop()


# --------------------------------------------------------------------------
# the entry shape
# --------------------------------------------------------------------------


def test_the_entry_carries_the_routing_story_not_just_the_outcome():
    """"Why did this cost $0.04" needs the path and the gates, not the tier."""
    entry = audit_entry(
        record(tier="frontier",
               path=["local_small", "local_large", "frontier"],
               gates=[{"gate": "small", "verdict": "escalate"},
                      {"gate": "large", "verdict": "escalate"}],
               cost_micro_usd=1660),
        tenant="acme", model="gpt-4o-mock", principal="abc123",
    )
    assert entry["path"] == ["local_small", "local_large", "frontier"]
    assert len(entry["gates"]) == 2
    assert entry["cost_micro_usd"] == 1660
    assert entry["tenant"] == "acme"
    assert entry["principal"] == "abc123"
    assert entry["ts"] > 0


def test_the_entry_never_carries_a_raw_key():
    """`principal` is a digest. A credential in a file that exists to be read
    later, by more people than saw the request, would be a real leak."""
    entry = audit_entry(record(), tenant="acme", principal="abc123")
    assert "sk-" not in json.dumps(entry)
    assert entry["principal"] == "abc123"


def test_entries_are_json_serializable_even_with_odd_values():
    """`default=str` in the writer, but the entry itself should not rely on it
    for anything routine."""
    json.dumps(audit_entry(record(confidence=0.9876), tenant="acme"))


# --------------------------------------------------------------------------
# through the gateway
# --------------------------------------------------------------------------


@pytest.fixture
def audited_client(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("AUDIT_PATH", str(tmp_path))
    monkeypatch.setenv("AUDIT_ENABLED", "true")
    monkeypatch.setenv("WARM_MODELS_ON_STARTUP", "false")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")   # unreachable
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.mark.integration
def test_a_real_request_is_audited_end_to_end(audited_client):
    """Needs a live Ollama. Proves the wiring, not just the module."""
    tmp_path = audited_client
    with TestClient(app) as c:
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+2? Number only."}],
            "max_tokens": 8,
        })
        assert r.status_code == 200
        trace = r.json()["x_hlr"]["trace_id"]

        health = c.get("/healthz").json()
        assert health["config"]["audit"]["written"] >= 0

    files = list(tmp_path.glob("audit-*.jsonl"))
    assert files, "no audit file written"
    entries = [json.loads(ln) for ln in files[0].read_text().splitlines() if ln.strip()]
    assert any(e["trace_id"] == trace for e in entries)


def test_refusals_are_audited(tmp_path):
    """A 402 is a spend decision. Telemetry counts served traffic only, which
    is right for a rate/cost aggregate and wrong for a record of what the
    gateway decided — so refusals go to the audit trail even though they never
    reach telemetry."""
    from decimal import Decimal

    from app.budgets import BudgetLedger, Tenant, TenantRegistry, window_key
    from app.cache.exact import ExactCache
    from app.cache.semantic import SemanticCache
    from app.cache.singleflight import SingleFlight
    from app.config import Settings
    from app.costs import usd_to_micro
    from app.ratelimit import RedisRateLimiter
    from app.resilience import BreakerRegistry, RetryPolicy
    from app.security import ApiKeyAuth, RequestLimits
    from app.telemetry import Telemetry
    from tests.test_budgets import FakeRedis
    from tests.test_gateway import StubFrontier, StubOllama

    reg = TenantRegistry([
        Tenant("acme", keys=frozenset({"sk-acme"}), daily_budget_usd=Decimal("1.00"))])
    redis = FakeRedis()
    redis.store[window_key("acme")] = usd_to_micro(Decimal("1.00"))

    captured: list[dict] = []

    class CapturingAudit(AuditLog):
        def write(self, record):
            captured.append(record)
            return True

    c = TestClient(app)
    c.app.state.settings = Settings(_env_file=None)
    c.app.state.ollama = StubOllama()
    c.app.state.http = None
    c.app.state.redis = None
    c.app.state.exact_cache = ExactCache(None)
    c.app.state.telemetry = Telemetry(None)
    c.app.state.frontier = StubFrontier()
    c.app.state.semantic_cache = SemanticCache(None, enabled=False)
    c.app.state.singleflight = SingleFlight(None)
    c.app.state.tenants = reg
    c.app.state.budgets = BudgetLedger(redis, reg)
    c.app.state.auth = ApiKeyAuth({"sk-acme"}, tenant_for=lambda k: reg.resolve(k).name)
    c.app.state.rate_limiter = RedisRateLimiter(None, limit=0)
    c.app.state.limits = RequestLimits()
    c.app.state.breakers = BreakerRegistry(failure_threshold=10**6)
    c.app.state.retry_policy = RetryPolicy(attempts=1)
    c.app.state.audit = CapturingAudit(tmp_path, enabled=True)

    r = c.post("/v1/chat/completions",
               json={"model": "gateway:frontier",
                     "messages": [{"role": "user", "content": "q"}]},
               headers={"X-API-Key": "sk-acme"})

    assert r.status_code == 402
    refusals = [e for e in captured if e.get("outcome") == "refused"]
    assert len(refusals) == 1
    assert refusals[0]["status"] == 402
    assert refusals[0]["code"] == "budget_exhausted"
    assert refusals[0]["tenant"] == "acme"
    assert refusals[0]["cost_micro_usd"] == 0
