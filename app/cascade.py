"""The tier chain, as ordered data.

The obvious implementation — nested `if/else` across five tiers, two gates and
two cache short-circuits — grows combinatorially and becomes untestable. As
data, each tier is independently testable and the ordering is inspectable at
runtime, which is exactly what `x_hlr.path` reports back to the caller
(DECISIONS.md D14).

Adding or reordering a tier is a change to a list literal. Never reintroduce
branching here.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.confidence import GateVerdict, evaluate_gate
from app.providers.base import Completion, ProviderError, ProviderTimeout, ProviderUnavailable

TierName = Literal["local_small", "local_large", "frontier"]

# A tier runner: given the request, produce a Completion or raise ProviderError.
Runner = Callable[[str, list[dict[str, str]]], Awaitable[Completion]]


@dataclass(frozen=True)
class Tier:
    """One rung of the ladder.

    `gate_id` is None for the last tier — there is nowhere left to escalate, so
    evaluating a gate there would be theatre.
    """

    name: TierName
    model: str
    gate_id: Literal["small", "large"] | None
    threshold: float = 0.0
    billed: bool = False


@dataclass
class CascadeResult:
    """What the executor produces. Maps directly onto the `x_hlr` block."""

    completion: Completion | None
    tier: TierName | None
    path: list[str] = field(default_factory=list)
    gates: list[GateVerdict] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float | None = None
    # Set when every tier failed — the caller turns this into a 503.
    error: ProviderError | None = None

    @property
    def served(self) -> bool:
        return self.completion is not None


def build_tiers(
    *,
    small_model: str,
    large_model: str,
    frontier_model: str,
    threshold_small: float,
    threshold_large: float,
) -> list[Tier]:
    """The chain, cheapest first. This list *is* the routing policy."""
    return [
        Tier("local_small", small_model, gate_id="small", threshold=threshold_small),
        Tier("local_large", large_model, gate_id="large", threshold=threshold_large),
        Tier("frontier", frontier_model, gate_id=None, billed=True),
    ]


def entry_index(tiers: list[Tier], entry_tier: str) -> int:
    """Where the router said to start. Unknown names start at the top rather
    than raising — a routing bug should cost money, not availability."""
    for i, tier in enumerate(tiers):
        if tier.name == entry_tier:
            return i
    return 0


async def execute(
    tiers: list[Tier],
    *,
    start: int,
    messages: list[dict[str, str]],
    runners: dict[str, Runner],
    confidence_mode: Literal["raw", "calibrated"] = "raw",
    calibrators: dict[str, Any] | None = None,
) -> CascadeResult:
    """Walk the chain from `start` until a tier's answer is accepted.

    A tier is accepted when it has no gate (the last one), or when its gate
    says so. Otherwise the walk continues to the next tier.

    Two failure behaviours worth stating plainly:

    - A tier that is **unavailable** (model not pulled, service down) is
      skipped, and the walk continues. One missing model must not take the
      whole cascade down (invariant 4).
    - If the walk runs out of tiers while holding a **below-threshold answer**,
      that answer is returned anyway, with the shortfall stated in `reason`.
      Returning a weak answer and admitting it beats returning nothing.
    """
    result = CascadeResult(completion=None, tier=None)
    calibrators = calibrators or {}
    best_below_threshold: tuple[Tier, Completion, GateVerdict] | None = None

    for tier in tiers[start:]:
        runner = runners.get(tier.name)
        if runner is None:
            result.degraded.append(f"{tier.name}_not_configured")
            continue

        result.path.append(tier.name)
        started = time.perf_counter()
        try:
            completion = await runner(tier.model, messages)
        except ProviderUnavailable as exc:
            result.latency_ms[tier.name] = round((time.perf_counter() - started) * 1000, 1)
            result.degraded.append(f"{tier.name}_unavailable")
            result.error = exc
            continue
        except ProviderTimeout as exc:
            result.latency_ms[tier.name] = round((time.perf_counter() - started) * 1000, 1)
            result.degraded.append(f"{tier.name}_timeout")
            result.error = exc
            continue

        result.latency_ms[tier.name] = completion.latency_ms or round(
            (time.perf_counter() - started) * 1000, 1
        )
        if completion.was_cold_load:
            result.latency_ms.setdefault("model_load", completion.load_ms)

        # Last tier: nothing above it, so nothing to decide.
        if tier.gate_id is None:
            result.completion = completion
            result.tier = tier.name
            result.reason = _served_reason(tier, result)
            return result

        verdict = evaluate_gate(
            tier.gate_id,
            completion.logprobs,
            completion.content,
            tier.threshold,
            mode=confidence_mode,
            calibrator=calibrators.get(tier.gate_id),
        )
        result.gates.append(verdict)

        if verdict.verdict == "accept":
            result.completion = completion
            result.tier = tier.name
            result.confidence = verdict.confidence
            result.reason = _served_reason(tier, result, verdict)
            return result

        # Keep the best rejected answer in case nothing better materializes.
        if best_below_threshold is None or verdict.confidence > best_below_threshold[2].confidence:
            best_below_threshold = (tier, completion, verdict)

    # Ran out of tiers.
    if best_below_threshold is not None:
        tier, completion, verdict = best_below_threshold
        result.completion = completion
        result.tier = tier.name
        result.confidence = verdict.confidence
        result.degraded.append("escalation_exhausted")
        result.reason = (
            f"no tier cleared its gate; returning the best available answer from "
            f"{tier.name} at confidence {verdict.confidence:.2f} "
            f"(needed {verdict.threshold:.2f}) — escalation was unavailable"
        )
        return result

    result.reason = "no tier could serve this request: " + ", ".join(result.degraded)
    return result


def _served_reason(
    tier: Tier, result: CascadeResult, verdict: GateVerdict | None = None
) -> str:
    """Human-readable, naming the path taken. Invariant 2 — this reaches the
    dashboard and is how anyone judges whether the routing was sensible."""
    escalations = [g for g in result.gates if g.verdict == "escalate"]
    if not escalations:
        if verdict is not None:
            return f"served by {tier.name}: {verdict.reason}"
        return f"served by {tier.name} (final tier, no gate above it)"

    hops = "; ".join(f"{g.gate} gate escalated ({g.reason})" for g in escalations)
    if verdict is not None:
        return f"served by {tier.name} after {len(escalations)} escalation(s) — {hops} — {verdict.reason}"
    return f"served by {tier.name} after {len(escalations)} escalation(s) — {hops}"
