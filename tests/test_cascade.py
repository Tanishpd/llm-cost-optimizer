"""Tests for app/cascade.py — the escalation logic.

This is where the project's central claim finally executes: a model's own
logprobs decide whether its answer stands. The tests are built around real
logprob shapes rather than invented confidences.
"""

from __future__ import annotations

import math

import pytest

from app.cascade import CascadeResult, Tier, build_tiers, entry_index, execute
from app.providers.base import Completion, ProviderTimeout, ProviderUnavailable

TIERS = build_tiers(
    small_model="qwen2.5:0.5b",
    large_model="qwen2.5:3b",
    frontier_model="gpt-4o-mock",
    threshold_small=0.55,
    threshold_large=0.62,
)
MESSAGES = [{"role": "user", "content": "q"}]


def logprobs_for(prob: float, n: int = 20) -> list[dict]:
    """n tokens each at `prob`, so geo_mean == prob exactly."""
    lp = math.log(prob)
    return [{"token": "t", "logprob": lp, "top_logprobs": [
        {"token": "t", "logprob": lp}, {"token": "u", "logprob": lp - 2}]} for _ in range(n)]


def completion(prob: float, *, model="m", content="an answer", n=20) -> Completion:
    return Completion(
        content=content, model=model, tokens_in=10, tokens_out=n,
        logprobs=logprobs_for(prob, n), latency_ms=100.0,
    )


def frontier_completion(content="frontier answer") -> Completion:
    """The frontier returns no logprobs — it is the last tier, so there is
    nothing above it for a gate to decide (see providers/frontier.py)."""
    return Completion(
        content=content, model="gpt-4o-mock", tokens_in=41, tokens_out=91,
        logprobs=[], latency_ms=1800.0,
    )


def runners_returning(*by_tier: tuple[str, Completion | Exception]):
    """Build a runner map. Each entry is (tier_name, completion-or-exception)."""
    mapping = dict(by_tier)
    calls: list[str] = []

    def make(name):
        async def run(model, messages):
            calls.append(name)
            value = mapping[name]
            if isinstance(value, Exception):
                raise value
            return value
        return run

    return {name: make(name) for name in mapping}, calls


# --------------------------------------------------------------------------
# the chain is data
# --------------------------------------------------------------------------


def test_chain_is_cheapest_first():
    assert [t.name for t in TIERS] == ["local_small", "local_large", "frontier"]


def test_only_the_last_tier_bills():
    assert [t.billed for t in TIERS] == [False, False, True]


def test_last_tier_has_no_gate():
    """Nothing above it to escalate to, so a gate there would be theatre."""
    assert TIERS[-1].gate_id is None
    assert all(t.gate_id is not None for t in TIERS[:-1])


def test_entry_index_finds_each_tier():
    assert entry_index(TIERS, "local_small") == 0
    assert entry_index(TIERS, "local_large") == 1
    assert entry_index(TIERS, "frontier") == 2


def test_unknown_entry_starts_at_the_top():
    """A routing bug should cost money, not availability."""
    assert entry_index(TIERS, "nonsense") == 0


# --------------------------------------------------------------------------
# accepting and escalating
# --------------------------------------------------------------------------


async def test_confident_first_tier_stops_the_walk():
    runners, calls = runners_returning(("local_small", completion(0.95)))
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_small"
    assert r.path == ["local_small"]
    assert calls == ["local_small"]  # nothing else was called
    assert r.gates[0].verdict == "accept"


async def test_unconfident_first_tier_escalates():
    """The core mechanic: the 0.5B's own logprobs send it up a tier."""
    runners, calls = runners_returning(
        ("local_small", completion(0.40)),   # below 0.55
        ("local_large", completion(0.80)),   # above 0.62
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert r.path == ["local_small", "local_large"]
    assert [g.verdict for g in r.gates] == ["escalate", "accept"]
    assert calls == ["local_small", "local_large"]


async def test_both_gates_escalate_to_frontier():
    runners, calls = runners_returning(
        ("local_small", completion(0.40)),
        ("local_large", completion(0.50)),  # below 0.62
        ("frontier", frontier_completion()),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "frontier"
    assert r.path == ["local_small", "local_large", "frontier"]
    assert len(r.gates) == 2
    assert all(g.verdict == "escalate" for g in r.gates)


async def test_path_records_every_tier_paid_for():
    """A request that walked all three paid for all three."""
    runners, _ = runners_returning(
        ("local_small", completion(0.30)),
        ("local_large", completion(0.40)),
        ("frontier", frontier_completion()),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert len(r.path) == 3
    assert set(r.latency_ms) >= {"local_small", "local_large", "frontier"}


async def test_starting_mid_chain_skips_cheaper_tiers():
    runners, calls = runners_returning(
        ("local_small", completion(0.95)),
        ("local_large", completion(0.95)),
    )
    r = await execute(TIERS, start=1, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert calls == ["local_large"]
    assert "local_small" not in r.path


async def test_frontier_entry_calls_nothing_local():
    runners, calls = runners_returning(("frontier", frontier_completion()))
    r = await execute(TIERS, start=2, messages=MESSAGES, runners=runners)
    assert r.tier == "frontier"
    assert calls == ["frontier"]
    assert r.gates == []


async def test_hedge_escalates_despite_high_confidence():
    """A model can be fluently, confidently uncertain — the veto is independent
    of the score (RESEARCH.md §3)."""
    hedged = completion(0.98, content="I don't know the answer to that.")
    runners, calls = runners_returning(
        ("local_small", hedged), ("local_large", completion(0.90))
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert r.gates[0].verdict == "escalate"
    assert r.gates[0].hedged is True


async def test_missing_logprobs_force_escalation():
    """An unmeasurable answer must not be scored as confident."""
    blind = Completion(content="x", model="m", tokens_in=1, tokens_out=1, logprobs=[])
    runners, _ = runners_returning(("local_small", blind), ("local_large", completion(0.90)))
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert r.gates[0].confidence == 0.0


# --------------------------------------------------------------------------
# degradation — invariant 4
# --------------------------------------------------------------------------


async def test_unavailable_tier_is_skipped_not_fatal():
    """One missing model must not take the cascade down."""
    runners, calls = runners_returning(
        ("local_small", ProviderUnavailable("not pulled", fix="ollama pull ...")),
        ("local_large", completion(0.90)),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert "local_small_unavailable" in r.degraded
    assert calls == ["local_small", "local_large"]


async def test_timeout_is_recorded_distinctly():
    runners, _ = runners_returning(
        ("local_small", ProviderTimeout("slow")), ("local_large", completion(0.90))
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert "local_small_timeout" in r.degraded


async def test_unconfigured_tier_is_skipped():
    runners, _ = runners_returning(("local_large", completion(0.90)))
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"
    assert "local_small_not_configured" in r.degraded


async def test_exhausted_escalation_returns_the_best_weak_answer():
    """Escalation impossible: return the below-threshold answer and say so.
    Returning nothing would be worse."""
    runners, _ = runners_returning(
        ("local_small", completion(0.40)),
        ("local_large", completion(0.50)),
        ("frontier", ProviderUnavailable("frontier down")),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.tier == "local_large"  # the higher-confidence of the two rejects
    assert r.confidence == pytest.approx(0.50, abs=0.01)
    assert "escalation_exhausted" in r.degraded
    assert "best available" in r.reason
    assert "needed" in r.reason  # states the shortfall


async def test_best_weak_answer_is_the_highest_confidence_one():
    runners, _ = runners_returning(
        ("local_small", completion(0.52, content="small answer")),
        ("local_large", completion(0.45, content="large answer")),
        ("frontier", ProviderUnavailable("down")),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.completion.content == "small answer"  # 0.52 > 0.45
    assert r.tier == "local_small"


async def test_all_tiers_down_serves_nothing():
    runners, _ = runners_returning(
        ("local_small", ProviderUnavailable("a")),
        ("local_large", ProviderUnavailable("b")),
        ("frontier", ProviderUnavailable("c")),
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert r.completion is None
    assert r.tier is None
    assert r.error is not None
    assert "no tier could serve" in r.reason


# --------------------------------------------------------------------------
# reasons — invariant 2
# --------------------------------------------------------------------------


async def test_reason_names_the_escalations_that_happened():
    runners, _ = runners_returning(
        ("local_small", completion(0.40)), ("local_large", completion(0.80))
    )
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert "escalation" in r.reason
    assert "small gate escalated" in r.reason


async def test_reason_is_prose_not_an_enum():
    runners, _ = runners_returning(("local_small", completion(0.95)))
    r = await execute(TIERS, start=0, messages=MESSAGES, runners=runners)
    assert len(r.reason) > 25 and " " in r.reason


async def test_direct_serve_reason_mentions_no_escalation():
    runners, _ = runners_returning(("frontier", frontier_completion()))
    r = await execute(TIERS, start=2, messages=MESSAGES, runners=runners)
    assert "no gate above it" in r.reason


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------


async def test_each_gate_uses_its_own_threshold():
    """0.58 clears the small gate (0.55) but not the large one (0.62)."""
    runners_small, _ = runners_returning(("local_small", completion(0.58)))
    r_small = await execute(TIERS, start=0, messages=MESSAGES, runners=runners_small)
    assert r_small.tier == "local_small"

    runners_large, _ = runners_returning(
        ("local_large", completion(0.58)), ("frontier", frontier_completion())
    )
    r_large = await execute(TIERS, start=1, messages=MESSAGES, runners=runners_large)
    assert r_large.tier == "frontier"


def test_result_served_property():
    assert CascadeResult(completion=None, tier=None).served is False
    assert CascadeResult(completion=completion(0.9), tier="local_small").served is True
