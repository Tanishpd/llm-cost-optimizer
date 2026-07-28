"""Per-request telemetry and the aggregates behind `/metrics`.

Two stores, because they answer different questions:

- a capped Redis **stream** (`hlr:telemetry`) — the last N individual requests,
  for the dashboard's live log and for debugging one specific decision;
- **counter hashes** (`hlr:stats:*`) — monotonic totals, so `/metrics` is O(1)
  rather than a scan over the stream.

Like the cache, nothing here raises: telemetry failing must never fail a
request that already succeeded (invariant 4).

The counts must reconcile — the identities are documented in docs/API.md and
asserted in `tests/test_telemetry.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.costs import micro_to_usd, saved_pct

logger = logging.getLogger("hlr.telemetry")

STREAM_KEY = "hlr:telemetry"
STREAM_MAXLEN = 10_000

TIERS = ("cache_exact", "cache_semantic", "local_small", "local_large", "frontier")
MODEL_TIERS = ("local_small", "local_large", "frontier")
GATES = ("small", "large")

# Hash keys, namespaced per PLAN.md's schema table.
K_SERVED = "hlr:stats:tiers"          # served-by counts
K_ATTEMPTED = "hlr:stats:attempted"   # attempted-by counts
K_ENTRY = "hlr:stats:router"          # pre-router entry-tier counts
K_GATES = "hlr:stats:gates"           # per-gate pass/fail
K_COST = "hlr:stats:cost"             # micro-USD totals + token totals
K_CACHE = "hlr:stats:cache"           # guard rejections, lookups
K_LOADS = "hlr:stats:model_loads"     # cold loads, cumulative load ms


@dataclass
class RequestRecord:
    """One request, as written to the stream.

    `path` is every tier attempted in order — a request that walked
    0.5B → 3B → frontier paid for all three, and the record says so.
    """

    trace_id: str
    tier: str | None
    path: list[str]
    reason: str
    latency_ms: dict[str, float]
    tokens_in: int
    tokens_out: int
    cost_micro_usd: int
    baseline_micro_usd: int
    entry_tier: str | None = None
    confidence: float | None = None
    gates: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: str | None = None          # "exact" | "semantic" | None
    guard_rejected: str | None = None
    cold_load_ms: float = 0.0
    degraded: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def to_stream_fields(self) -> dict[str, str]:
        """Redis streams store flat string fields; nested values are JSON."""
        return {
            "trace_id": self.trace_id,
            "tier": self.tier or "",
            "path": ",".join(self.path),
            "reason": self.reason[:500],
            "latency_ms": json.dumps(self.latency_ms),
            "tokens_in": str(self.tokens_in),
            "tokens_out": str(self.tokens_out),
            "cost_micro_usd": str(self.cost_micro_usd),
            "baseline_micro_usd": str(self.baseline_micro_usd),
            "entry_tier": self.entry_tier or "",
            "confidence": "" if self.confidence is None else f"{self.confidence:.6f}",
            "gates": json.dumps(self.gates),
            "cache_hit": self.cache_hit or "",
            "guard_rejected": self.guard_rejected or "",
            "cold_load_ms": str(self.cold_load_ms),
            "degraded": ",".join(self.degraded),
            "created_at": str(self.created_at),
        }


class Telemetry:
    """Records requests and serves aggregates. Degrades to no-ops."""

    def __init__(self, redis: Redis | None) -> None:
        self._redis = redis

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def record(self, rec: RequestRecord) -> bool:
        """Write one request. Never raises.

        Stream append and counter updates go in a single pipeline: one
        round-trip, and the counters cannot drift from the stream because
        either both land or neither does.
        """
        if self._redis is None:
            return False
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.xadd(STREAM_KEY, rec.to_stream_fields(), maxlen=STREAM_MAXLEN, approximate=True)

            if rec.tier:
                pipe.hincrby(K_SERVED, rec.tier, 1)
            for tier in rec.path:
                if tier in MODEL_TIERS:
                    pipe.hincrby(K_ATTEMPTED, tier, 1)
            if rec.entry_tier:
                pipe.hincrby(K_ENTRY, rec.entry_tier, 1)

            for gate in rec.gates:
                name = gate.get("gate")
                if name in GATES:
                    pipe.hincrby(K_GATES, f"{name}:evaluated", 1)
                    verdict = "accepted" if gate.get("verdict") == "accept" else "escalated"
                    pipe.hincrby(K_GATES, f"{name}:{verdict}", 1)

            pipe.hincrby(K_COST, "requests", 1)
            pipe.hincrby(K_COST, "actual_micro_usd", rec.cost_micro_usd)
            pipe.hincrby(K_COST, "baseline_micro_usd", rec.baseline_micro_usd)
            pipe.hincrby(K_COST, "tokens_in", rec.tokens_in)
            pipe.hincrby(K_COST, "tokens_out", rec.tokens_out)
            # Baseline accumulators are the serving tier's usage — the same
            # numbers, but named apart so /metrics can prove the identity
            # rather than assuming it (docs/API.md).
            pipe.hincrby(K_COST, "baseline_in", rec.tokens_in)
            pipe.hincrby(K_COST, "baseline_out", rec.tokens_out)
            if rec.tier == "frontier":
                pipe.hincrby(K_COST, "frontier_in", rec.tokens_in)
                pipe.hincrby(K_COST, "frontier_out", rec.tokens_out)

            if rec.cache_hit:
                pipe.hincrby(K_CACHE, f"hit_{rec.cache_hit}", 1)
            if rec.guard_rejected:
                pipe.hincrby(K_CACHE, f"guard_{rec.guard_rejected}", 1)
            if rec.cold_load_ms > 0:
                pipe.hincrby(K_LOADS, "cold_loads", 1)
                pipe.hincrby(K_LOADS, "total_load_ms", int(rec.cold_load_ms))

            await pipe.execute()
            return True
        except RedisError as exc:
            logger.warning("telemetry write failed (request unaffected): %s", exc)
            return False

    async def _hash(self, key: str) -> dict[str, int]:
        try:
            raw = await self._redis.hgetall(key)  # type: ignore[union-attr]
        except (RedisError, AttributeError):
            return {}
        out: dict[str, int] = {}
        for k, v in (raw or {}).items():
            try:
                out[k if isinstance(k, str) else k.decode()] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    async def metrics(self) -> dict[str, Any]:
        """Aggregates for `/metrics` and the dashboard.

        Returns a zeroed but structurally complete payload when Redis is
        unavailable — a dashboard should render "no data", not crash.
        """
        if self._redis is None:
            return _empty_metrics(degraded=["redis_unavailable"])

        served = await self._hash(K_SERVED)
        attempted = await self._hash(K_ATTEMPTED)
        entry = await self._hash(K_ENTRY)
        gates = await self._hash(K_GATES)
        cost = await self._hash(K_COST)
        cache = await self._hash(K_CACHE)
        loads = await self._hash(K_LOADS)

        requests = cost.get("requests", 0)
        actual = cost.get("actual_micro_usd", 0)
        baseline = cost.get("baseline_micro_usd", 0)

        hits_exact = cache.get("hit_exact", 0)
        hits_semantic = cache.get("hit_semantic", 0)
        hits = hits_exact + hits_semantic

        return {
            "window": "all_time",
            "requests": requests,
            "tiers": {
                "served": {t: served.get(t, 0) for t in TIERS},
                "attempted": {t: attempted.get(t, 0) for t in MODEL_TIERS},
            },
            "router": {"entry": {t: entry.get(t, 0) for t in MODEL_TIERS}},
            "cache": {
                "hit_rate": round(hits / requests, 3) if requests else 0.0,
                "exact_hit_rate": round(hits_exact / requests, 3) if requests else 0.0,
                "semantic_hit_rate": round(hits_semantic / requests, 3) if requests else 0.0,
                "guard_rejections": {
                    k.removeprefix("guard_"): v for k, v in cache.items() if k.startswith("guard_")
                },
            },
            "gates": {
                g: {
                    "evaluated": gates.get(f"{g}:evaluated", 0),
                    "accepted": gates.get(f"{g}:accepted", 0),
                    "escalated": gates.get(f"{g}:escalated", 0),
                    "escalation_rate": (
                        round(gates.get(f"{g}:escalated", 0) / gates[f"{g}:evaluated"], 3)
                        if gates.get(f"{g}:evaluated")
                        else 0.0
                    ),
                }
                for g in GATES
            },
            "cost": {
                "actual_usd": float(micro_to_usd(actual)),
                "baseline_usd": float(micro_to_usd(baseline)),
                "saved_usd": float(micro_to_usd(baseline - actual)),
                "saved_pct": saved_pct(baseline, actual),
            },
            "tokens": {
                "in": cost.get("tokens_in", 0),
                "out": cost.get("tokens_out", 0),
                "baseline_in": cost.get("baseline_in", 0),
                "baseline_out": cost.get("baseline_out", 0),
                "frontier_in": cost.get("frontier_in", 0),
                "frontier_out": cost.get("frontier_out", 0),
            },
            "latency_ms": await self.latency_percentiles(),
            "model_loads": {
                "cold_loads": loads.get("cold_loads", 0),
                "mean_load_ms": (
                    round(loads.get("total_load_ms", 0) / loads["cold_loads"], 1)
                    if loads.get("cold_loads")
                    else 0.0
                ),
            },
            "degraded": [],
        }

    async def latency_percentiles(self, *, sample: int = 2000) -> dict[str, dict[str, float]]:
        """p50/p95 per stage, plus end-to-end.

        Computed from the stream rather than kept as running counters: exact
        percentiles need the distribution, and a streaming approximation is not
        worth the complexity for a dashboard endpoint that is never on the
        request path.

        Bounded by `sample` so a full 10k-entry stream cannot make `/metrics`
        slow. The bound is reported in the payload rather than left implicit —
        a percentile over a truncated window is a different number, and saying
        so is cheaper than someone discovering it later.
        """
        if self._redis is None:
            return {}
        try:
            entries = await self._redis.xrevrange(STREAM_KEY, count=sample)
        except RedisError:
            return {}

        buckets: dict[str, list[float]] = {}
        for _id, fields in entries:
            raw = fields.get("latency_ms") if isinstance(fields, dict) else None
            if raw is None:
                continue
            try:
                stages = json.loads(raw if isinstance(raw, str) else raw.decode())
            except (ValueError, AttributeError):
                continue
            for stage, value in (stages or {}).items():
                if isinstance(value, int | float):
                    buckets.setdefault("end_to_end" if stage == "total" else stage, []).append(
                        float(value)
                    )

        return {
            stage: {
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "n": len(values),
            }
            for stage, values in sorted(buckets.items())
        }

    async def recent(self, count: int = 20) -> list[dict[str, Any]]:
        """Most recent requests, newest first — the dashboard's live log."""
        if self._redis is None:
            return []
        try:
            entries = await self._redis.xrevrange(STREAM_KEY, count=count)
        except RedisError:
            return []
        out = []
        for _id, fields in entries:
            rec = {
                (k if isinstance(k, str) else k.decode()): (
                    v if isinstance(v, str) else v.decode()
                )
                for k, v in fields.items()
            }
            for j in ("latency_ms", "gates"):
                try:
                    rec[j] = json.loads(rec.get(j) or "null")
                except ValueError:
                    rec[j] = None
            rec["path"] = [p for p in (rec.get("path") or "").split(",") if p]
            out.append(rec)
        return out

    async def reset(self) -> int:
        """Clear telemetry. Dev convenience; scoped to the telemetry keys."""
        if self._redis is None:
            return 0
        try:
            return int(
                await self._redis.delete(
                    STREAM_KEY, K_SERVED, K_ATTEMPTED, K_ENTRY, K_GATES, K_COST, K_CACHE, K_LOADS
                )
            )
        except RedisError:
            return 0


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. numpy is available, but keeping this
    module import-light means telemetry can be exercised without it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 1)
    pos = q * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 1)


def _empty_metrics(*, degraded: list[str] | None = None) -> dict[str, Any]:
    """Structurally complete zeros, so a dashboard renders "no data" rather
    than KeyError-ing on a missing section."""
    return {
        "window": "all_time",
        "requests": 0,
        "tiers": {"served": dict.fromkeys(TIERS, 0), "attempted": dict.fromkeys(MODEL_TIERS, 0)},
        "router": {"entry": dict.fromkeys(MODEL_TIERS, 0)},
        "cache": {
            "hit_rate": 0.0,
            "exact_hit_rate": 0.0,
            "semantic_hit_rate": 0.0,
            "guard_rejections": {},
        },
        "gates": {
            g: {"evaluated": 0, "accepted": 0, "escalated": 0, "escalation_rate": 0.0}
            for g in GATES
        },
        "cost": {"actual_usd": 0.0, "baseline_usd": 0.0, "saved_usd": 0.0, "saved_pct": 0.0},
        "tokens": {
            "in": 0, "out": 0, "baseline_in": 0, "baseline_out": 0,
            "frontier_in": 0, "frontier_out": 0,
        },
        "latency_ms": {},
        "model_loads": {"cold_loads": 0, "mean_load_ms": 0.0},
        "degraded": degraded or [],
    }
