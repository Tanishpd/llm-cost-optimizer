"""Prometheus exposition format for the metrics `/metrics` already computes.

`/metrics` returning JSON under a name the ecosystem reserves for something
specific is a trap: the first thing anyone deploying this does is point a
scraper at `/metrics` and get a parse error. So `/metrics` now negotiates —
`Accept: text/plain` (which every Prometheus scraper sends) gets exposition
format, anything else keeps the JSON the dashboard consumes — and
`/metrics/prometheus` always serves exposition for people who would rather be
explicit than rely on a header.

Pure functions over the metrics dict. No I/O, no HTTP, no Redis — the same
property that makes `confidence.py` easy to test exhaustively.

**Naming follows the conventions, because tooling depends on them.** Counters
end in `_total`, units are in the name (`_usd`, `_ms`, `_bytes`), and a metric
name is emitted with exactly one `# TYPE` line. Prometheus silently drops a
series whose type is redeclared, so a duplicate here is a metric that vanishes
rather than one that errors.

**Cost is exposed as a counter in USD**, not micro-USD. Micro-USD is the right
internal representation because integers sum exactly (`costs.py`), but a
dashboard asking "what did today cost" wants dollars, and `rate()` over a
float counter is well-defined.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Prometheus' own grammar. Anything not matching is a series that silently
# fails to parse at the far end, so it is enforced here rather than hoped for.
_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

PREFIX = "hlr"


# Every control character except tab and line feed — LF has a defined escape,
# tab is harmless inside a quoted value, and everything else is either an
# accident or an injection.
#
# The range is wider than C0 on purpose. Python's `str.splitlines()` — and
# several exposition parsers — break on \v \f \x1c \x1d \x1e \x85 \u2028 and
# \u2029 as well as \r\n. Property testing found \r first and then \x85 (NEL,
# a C1 control) after the first fix, which is the argument for enumerating the
# whole class rather than the character that happened to fail.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]")


def escape_label_value(value: str) -> str:
    """Make an arbitrary string safe to sit inside `label="..."`.

    The exposition format defines exactly three escapes — backslash,
    double-quote and line feed — so those are escaped. Every *other* control
    character is replaced, not escaped, because there is no defined sequence
    for them and emitting one raw is how a single sample becomes two lines.

    Found by property testing: a lone `\r` in a tenant name split the sample
    across two lines, producing `hlr_thing{label="` and `"} 1` — two fragments,
    neither of which parses. Tenant names come from operator JSON and tier
    names from config; neither is guaranteed tame.
    """
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    return _CONTROL_RE.sub("\ufffd", escaped)


def _format_value(value: float | int) -> str:
    """Prometheus spells these three differently from Python."""
    if isinstance(value, bool):
        return "1" if value else "0"
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


class Registry:
    """Accumulates samples and renders them, grouped so each name gets exactly
    one HELP/TYPE header."""

    def __init__(self) -> None:
        self._meta: dict[str, tuple[str, str]] = {}
        self._samples: dict[str, list[tuple[dict[str, str], float]]] = {}

    def add(
        self,
        name: str,
        value: float | int,
        *,
        help: str,
        type: str = "gauge",
        labels: dict[str, str] | None = None,
    ) -> None:
        full = f"{PREFIX}_{name}"
        if not _NAME_RE.match(full):
            raise ValueError(f"invalid Prometheus metric name: {full!r}")
        for key in (labels or {}):
            if not _LABEL_RE.match(key):
                raise ValueError(f"invalid Prometheus label name: {key!r}")

        if full in self._meta and self._meta[full][1] != type:
            # Prometheus drops a series whose type is redeclared, so this would
            # be a metric that silently disappears rather than one that errors.
            raise ValueError(f"{full} declared as both {self._meta[full][1]} and {type}")
        self._meta[full] = (help, type)
        self._samples.setdefault(full, []).append((labels or {}, float(value)))

    def render(self) -> str:
        out: list[str] = []
        for name in sorted(self._samples):
            help_text, kind = self._meta[name]
            out.append(f"# HELP {name} {help_text}")
            out.append(f"# TYPE {name} {kind}")
            for labels, value in self._samples[name]:
                if labels:
                    rendered = ",".join(
                        f'{k}="{escape_label_value(v)}"' for k, v in sorted(labels.items())
                    )
                    out.append(f"{name}{{{rendered}}} {_format_value(value)}")
                else:
                    out.append(f"{name} {_format_value(value)}")
        # A trailing newline is required by the format, not cosmetic.
        return "\n".join(out) + "\n"


def render(
    metrics: dict[str, Any],
    *,
    health: dict[str, Any] | None = None,
    tenants: dict[str, Any] | None = None,
) -> str:
    """Render `/metrics`' payload (plus optional health and tenant usage).

    Takes the dicts rather than the app so this stays pure — and so the same
    renderer can be exercised against the "Redis is down" payload, which is
    exactly when someone will be looking at a dashboard.
    """
    r = Registry()

    r.add("requests_total", metrics.get("requests", 0),
          help="Requests served since startup.", type="counter")

    tiers = metrics.get("tiers") or {}
    for tier, count in (tiers.get("served") or {}).items():
        r.add("requests_by_tier_total", count, type="counter", labels={"tier": tier},
              help="Requests served, by the tier that answered.")
    for tier, count in (tiers.get("attempted") or {}).items():
        # Attempted exceeds served for any tier that escalated: a request that
        # walked 0.5B -> 3B -> frontier was attempted by three and served by
        # one. The gap between these two IS the escalation cost.
        r.add("tier_attempted_total", count, type="counter", labels={"tier": tier},
              help="Times a tier was called, including calls whose answer was discarded.")

    for tier, count in ((metrics.get("router") or {}).get("entry") or {}).items():
        r.add("router_entry_total", count, type="counter", labels={"tier": tier},
              help="Pre-router entry-tier choices.")

    for gate, stats in (metrics.get("gates") or {}).items():
        r.add("gate_evaluated_total", stats.get("evaluated", 0), type="counter",
              labels={"gate": gate}, help="Confidence-gate evaluations.")
        r.add("gate_escalated_total", stats.get("escalated", 0), type="counter",
              labels={"gate": gate}, help="Gate verdicts that escalated to the next tier.")
        r.add("gate_escalation_ratio", stats.get("escalation_rate", 0.0),
              labels={"gate": gate},
              help="Escalations over evaluations. A gauge, not a counter — it is a ratio.")

    cost = metrics.get("cost") or {}
    r.add("cost_usd_total", cost.get("actual_usd", 0.0), type="counter",
          help="Money actually spent on the frontier tier, USD.")
    r.add("baseline_cost_usd_total", cost.get("baseline_usd", 0.0), type="counter",
          help="What this traffic would have cost with no gateway, USD.")
    r.add("cost_saved_usd_total", cost.get("saved_usd", 0.0), type="counter",
          help="baseline_cost_usd_total minus cost_usd_total.")
    r.add("cost_saved_ratio", (cost.get("saved_pct", 0.0) or 0.0) / 100.0,
          help="Fraction of the baseline avoided. 0..1.")

    for direction, key in (("in", "in"), ("out", "out")):
        r.add("tokens_total", (metrics.get("tokens") or {}).get(key, 0), type="counter",
              labels={"direction": direction}, help="Tokens processed.")
    for direction in ("in", "out"):
        r.add("frontier_tokens_total",
              (metrics.get("tokens") or {}).get(f"frontier_{direction}", 0), type="counter",
              labels={"direction": direction}, help="Tokens the frontier billed for.")

    cache = metrics.get("cache") or {}
    r.add("cache_hit_ratio", cache.get("hit_rate", 0.0),
          help="Cache hits over requests, both tiers. 0..1.")
    for kind in ("exact", "semantic"):
        r.add("cache_hit_ratio_by_tier", cache.get(f"{kind}_hit_rate", 0.0),
              labels={"tier": kind}, help="Cache hit rate for one cache tier. 0..1.")
    for guard, count in (cache.get("guard_rejections") or {}).items():
        # Rising guard rejections mean the semantic cache is refusing hits it
        # is finding. Over-conservatism should be visible, not invisible (D15).
        r.add("cache_guard_rejections_total", count, type="counter",
              labels={"guard": guard}, help="Semantic hits rejected by a false-hit guard.")

    for stage, quantiles in (metrics.get("latency_ms") or {}).items():
        for q in ("p50", "p95"):
            if q in quantiles:
                r.add("latency_ms", quantiles[q], labels={"stage": stage, "quantile": q[1:]},
                      help="Per-stage latency, milliseconds. Computed over a bounded "
                           "sample of recent requests, not all time.")

    loads = metrics.get("model_loads") or {}
    r.add("model_cold_loads_total", loads.get("cold_loads", 0), type="counter",
          help="Model loads from cold. Each costs seconds on the request that triggers it.")

    for name in metrics.get("degraded") or []:
        r.add("degraded", 1, labels={"reason": name},
              help="Active degradations. Absent when healthy.")

    if tenants:
        for name, row in (tenants.get("usage") or {}).items():
            r.add("tenant_spend_usd", row.get("spent_usd", 0.0), labels={"tenant": name},
                  help="Spend today (UTC day). Resets daily — a gauge, not a counter.")
            if row.get("limit_usd") is not None:
                r.add("tenant_budget_usd", row["limit_usd"], labels={"tenant": name},
                      help="Configured daily cap, USD. Absent for uncapped tenants.")
            r.add("tenant_budget_enforced", 1 if row.get("enforced") else 0,
                  labels={"tenant": name},
                  help="0 when a cap exists but cannot be applied — i.e. spend is uncapped.")

    shadow = metrics.get("shadow") or {}
    if shadow.get("completed") is not None:
        r.add("shadow_sampled_total", shadow.get("sampled", 0), type="counter",
              help="Requests selected for a shadow comparison.")
        r.add("shadow_completed_total", shadow.get("completed", 0), type="counter",
              help="Shadow comparisons that produced a similarity score.")
        r.add("shadow_divergent_total", shadow.get("divergent", 0), type="counter",
              help="Comparisons where the gate's answer did not resemble the frontier's. "
                   "The false-cheap signal.")
        r.add("shadow_failed_total", shadow.get("failed", 0), type="counter",
              help="Shadow calls that errored. Costs a measurement, never a request.")
        r.add("shadow_skipped_budget_total", shadow.get("skipped_budget", 0), type="counter",
              help="Comparisons skipped because a budget or the ceiling was exhausted.")
        r.add("shadow_cost_usd_total", shadow.get("cost_usd", 0.0), type="counter",
              help="Money spent on measurement rather than on answers.")
        r.add("shadow_mean_similarity", shadow.get("mean_similarity", 0.0),
              help="Mean token overlap with the frontier. A tripwire, not a quality "
                   "score — watch the trend, not the value.")
        r.add("shadow_divergence_ratio", shadow.get("divergence_rate", 0.0),
              help="Divergent over completed. Rising means the gate is accepting "
                   "answers it should escalate.")

    if health:
        r.add("up", 1 if health.get("status") != "unhealthy" else 0,
              help="1 unless no model tier is reachable at all.")
        for dep, info in (health.get("dependencies") or {}).items():
            r.add("dependency_up", 1 if info.get("ok") else 0, labels={"dependency": dep},
                  help="1 when a dependency is reachable.")
        config = health.get("config") or {}
        for tier, state in (config.get("circuits") or {}).items():
            # Numeric because Prometheus cannot alert on a string.
            code = {"closed": 0, "half_open": 1, "open": 2}.get(str(state), -1)
            r.add("circuit_state", code, labels={"tier": tier},
                  help="Circuit breaker: 0 closed, 1 half-open, 2 open, -1 unknown.")
        audit = config.get("audit") or {}
        if "dropped" in audit:
            r.add("audit_dropped_total", audit["dropped"], type="counter",
                  help="Audit records dropped because the writer fell behind. "
                       "Non-zero means the trail has holes.")
            r.add("audit_written_total", audit.get("written", 0), type="counter",
                  help="Audit records written to disk.")
        if "budgets_enforced" in config:
            r.add("budgets_enforced", 1 if config["budgets_enforced"] else 0,
                  help="0 when tenant caps are configured but cannot be applied.")

    return r.render()


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def wants_prometheus(accept: str | None) -> bool:
    """Does this client want exposition format rather than JSON?

    Prometheus sends `Accept: application/openmetrics-text;...,text/plain;...`.
    Browsers send `text/html,...` and `*/*` — and a bare `*/*` (curl's default)
    must keep getting JSON, or every existing caller breaks the moment they
    stop sending an explicit Accept.
    """
    if not accept:
        return False
    lowered = accept.lower()
    if "application/json" in lowered:
        return False
    return "text/plain" in lowered or "openmetrics" in lowered
