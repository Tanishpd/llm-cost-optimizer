"""Unit tests for app/costs.py.

The arithmetic here produces the project's headline number, so the tests are
deliberately fussy about exactness and about the three-way distinction between
billed / baseline / throughput tokens.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.costs import (
    CostBreakdown,
    TokenUsage,
    compute_cost,
    micro_to_usd,
    ns_to_ms,
    price_micro_usd,
    saved_pct,
)

PRICE_IN = Decimal("2.50")
PRICE_OUT = Decimal("10.00")


# --------------------------------------------------------------------------
# usage parsing
# --------------------------------------------------------------------------


def test_usage_from_ollama_uses_documented_field_names():
    resp = {"prompt_eval_count": 41, "eval_count": 156, "total_duration": 123}
    u = TokenUsage.from_ollama(resp)
    assert (u.tokens_in, u.tokens_out, u.total) == (41, 156, 197)


def test_usage_from_ollama_tolerates_missing_counts():
    """Ollama omits counts in some error/short-circuit responses."""
    assert TokenUsage.from_ollama({}) == TokenUsage(0, 0)
    assert TokenUsage.from_ollama({"eval_count": None}).tokens_out == 0


def test_usage_from_openai_uses_usage_block():
    resp = {"usage": {"prompt_tokens": 41, "completion_tokens": 156, "total_tokens": 197}}
    u = TokenUsage.from_openai(resp)
    assert (u.tokens_in, u.tokens_out) == (41, 156)


def test_usage_from_openai_tolerates_missing_usage():
    assert TokenUsage.from_openai({}) == TokenUsage(0, 0)


# --------------------------------------------------------------------------
# pricing — must match the worked examples in docs/API.md
# --------------------------------------------------------------------------


def test_price_matches_api_doc_worked_example():
    """docs/API.md: 41 in / 156 out at 2.50/10.00 -> 0.0016625 USD -> 0.00166.

    Exactly 1662.5 micro-USD, which lands on a rounding boundary. ROUND_HALF_UP
    (away from zero) gives 1663; banker's rounding would give 1662. Pinning the
    half-up choice deliberately — money should not round to even.
    """
    micro = price_micro_usd(TokenUsage(41, 156), PRICE_IN, PRICE_OUT)
    assert micro == 1663
    assert float(micro_to_usd(micro)) == pytest.approx(0.001663, abs=5e-7)
    # Rounds to the 0.00166 quoted in the doc at 5 dp either way.
    assert round(float(micro_to_usd(micro)), 5) == 0.00166


def test_price_matches_api_doc_aggregate_actual():
    """docs/API.md /metrics: frontier 3104 in / 7812 out -> 0.0859."""
    micro = price_micro_usd(TokenUsage(3104, 7812), PRICE_IN, PRICE_OUT)
    assert float(micro_to_usd(micro)) == pytest.approx(0.0859, abs=5e-5)


def test_price_matches_api_doc_aggregate_baseline():
    """docs/API.md /metrics: baseline 61430 in / 155210 out -> 1.7057."""
    micro = price_micro_usd(TokenUsage(61430, 155210), PRICE_IN, PRICE_OUT)
    assert float(micro_to_usd(micro)) == pytest.approx(1.7057, abs=5e-5)


def test_price_is_zero_for_zero_tokens():
    assert price_micro_usd(TokenUsage(0, 0), PRICE_IN, PRICE_OUT) == 0


def test_price_returns_int_so_accumulation_is_exact():
    """The reason money is micro-USD ints: summing floats drifts. 10k requests
    at a fractional micro-USD each must land on an exact total."""
    per_request = price_micro_usd(TokenUsage(41, 156), PRICE_IN, PRICE_OUT)
    assert isinstance(per_request, int)
    total = sum(per_request for _ in range(10_000))
    assert total == per_request * 10_000  # no drift, by construction


# --------------------------------------------------------------------------
# cost breakdown — the billed/baseline distinction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["cache_exact", "cache_semantic", "local_small", "local_large"])
def test_non_frontier_tiers_bill_zero_but_still_have_a_baseline(tier):
    """The whole savings claim: local work is free, but the baseline still
    prices it as if it had gone to the frontier."""
    c = compute_cost(
        served_by=tier, usage=TokenUsage(41, 156), price_in_per_m=PRICE_IN,
            price_out_per_m=PRICE_OUT
    )
    assert c.cost_micro_usd == 0
    assert c.baseline_micro_usd > 0
    assert c.saved_micro_usd == c.baseline_micro_usd


def test_frontier_tier_bills_the_baseline_so_nothing_is_saved():
    c = compute_cost(
        served_by="frontier",
        usage=TokenUsage(41, 156),
        price_in_per_m=PRICE_IN,
        price_out_per_m=PRICE_OUT,
    )
    assert c.cost_micro_usd == c.baseline_micro_usd
    assert c.saved_micro_usd == 0


def test_cache_hits_are_credited_at_full_frontier_price():
    """A documented, disclosed assumption (D13): the baseline charges every
    request including cache hits. This test pins the behaviour so it can't
    change silently."""
    c = compute_cost(
        served_by="cache_exact",
        usage=TokenUsage(41, 156),
        price_in_per_m=PRICE_IN,
        price_out_per_m=PRICE_OUT,
    )
    assert c.saved_micro_usd == price_micro_usd(TokenUsage(41, 156), PRICE_IN, PRICE_OUT)


def test_breakdown_serializes_to_api_field_names():
    c = compute_cost(
        served_by="local_small",
        usage=TokenUsage(41, 156),
        price_in_per_m=PRICE_IN,
        price_out_per_m=PRICE_OUT,
    )
    d = c.as_dict()
    assert set(d) == {
        "tokens_in", "tokens_out", "cost_usd", "baseline_cost_usd", "cost_saved_usd",
    }
    assert d["cost_usd"] == 0.0
    assert d["cost_saved_usd"] == d["baseline_cost_usd"]


def test_saved_equals_baseline_minus_cost_identity():
    """docs/API.md reconciliation identity."""
    c = CostBreakdown(tokens_in=1, tokens_out=1, cost_micro_usd=859_000,
        baseline_micro_usd=1_705_700)
    assert c.saved_micro_usd == 1_705_700 - 859_000


# --------------------------------------------------------------------------
# percentages and durations
# --------------------------------------------------------------------------


def test_saved_pct_matches_api_doc_example():
    """1.7057 baseline, 0.0859 actual -> 95.0%."""
    assert saved_pct(1_705_700, 85_900) == pytest.approx(95.0, abs=0.05)


def test_saved_pct_is_zero_when_nothing_would_have_been_spent():
    """Avoids a divide-by-zero on an empty window."""
    assert saved_pct(0, 0) == 0.0
    assert saved_pct(-1, 0) == 0.0


def test_saved_pct_is_100_when_everything_was_free():
    assert saved_pct(1_000_000, 0) == 100.0


def test_ns_to_ms_converts_ollama_durations():
    """Ollama reports nanoseconds: total_duration 2175459583 -> ~2175.5 ms."""
    assert ns_to_ms(2_175_459_583) == pytest.approx(2175.5, abs=0.1)
    assert ns_to_ms(21_569_000) == pytest.approx(21.6, abs=0.1)


def test_ns_to_ms_handles_missing_durations():
    assert ns_to_ms(None) == 0.0
    assert ns_to_ms(0) == 0.0


# --------------------------------------------------------------------------
# found by mutation testing
# --------------------------------------------------------------------------


def test_ns_to_ms_uses_the_right_divisor_and_precision():
    """Two mutants survived here: `/1_000_000` -> `/1000001`, and
    `round(..., 1)` -> `round(..., 2)`.

    Nothing asserted either, because every existing test only checked that the
    number was "about right". Latency is reported to the caller in `x_hlr` and
    charted on the dashboard; a silently rescaled or re-rounded figure is the
    kind of wrong that gets believed.
    """
    assert ns_to_ms(1_000_000) == 1.0
    assert ns_to_ms(1_500_000_000) == 1500.0
    # One decimal place, exactly — not two.
    assert ns_to_ms(1_234_567) == 1.2
    assert ns_to_ms(0) == 0.0
    # The divisor is only separable at scale: a one-in-a-million error needs
    # ~1e11 ns before it moves the first decimal place. Asserting at small
    # values cannot tell 1_000_000 from 1_000_001, which is why the first
    # version of this test still let that mutant live.
    assert ns_to_ms(100_000_000_000) == 100_000.0
