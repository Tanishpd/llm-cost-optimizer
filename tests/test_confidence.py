"""Unit tests for app/confidence.py.

No I/O, no network — the module under test is pure functions, which is the
whole reason it was written that way.

The fixtures at the bottom are *real* logprob arrays captured from
qwen2.5:0.5b on 2026-07-28 (see docs/RESEARCH.md §2), not invented ones.
"""

from __future__ import annotations

import math

import pytest

from app.confidence import (
    EMPTY_METRICS,
    LOW_PROB,
    ConfidenceMetrics,
    LogisticCalibrator,
    evaluate_gate,
    extract_metrics,
    find_hedge,
    geo_mean,
    normalized_entropy,
    percentile,
    score,
    sequence_probability,
)


def lp(logprob: float, token: str = "x", *, top: list[float] | None = None) -> dict:
    """Build one logprobs entry in Ollama's verified shape."""
    entry: dict = {"token": token, "logprob": logprob, "bytes": [120]}
    if top is not None:
        entry["top_logprobs"] = [
            {"token": f"t{i}", "logprob": v, "bytes": [116]} for i, v in enumerate(top)
        ]
    return entry


# --------------------------------------------------------------------------
# geo_mean: the load-bearing claim is length-invariance
# --------------------------------------------------------------------------


def test_geo_mean_is_length_invariant():
    """The property the whole design depends on: same per-token confidence,
    wildly different lengths, same score."""
    per_token = math.log(0.9)
    for n in (1, 10, 100, 1000):
        assert geo_mean([per_token] * n) == pytest.approx(0.9, abs=1e-9)


def test_geo_mean_equals_inverse_perplexity():
    logprobs = [math.log(0.9), math.log(0.5), math.log(0.7)]
    perplexity = math.exp(-sum(logprobs) / len(logprobs))
    assert geo_mean(logprobs) == pytest.approx(1.0 / perplexity)


def test_sequence_probability_collapses_where_geo_mean_does_not():
    """Why ∏p_i is unusable: at a healthy p=0.9, a 100-token answer scores
    2.7e-5 — any fixed threshold rejects it. geo_mean stays at 0.9."""
    logprobs = [math.log(0.9)] * 100
    assert sequence_probability(logprobs) == pytest.approx(2.656e-5, rel=1e-3)
    assert sequence_probability(logprobs) < 0.85  # would be rejected
    assert geo_mean(logprobs) == pytest.approx(0.9)  # correctly accepted


def test_geo_mean_bounds():
    assert geo_mean([]) == 0.0
    assert geo_mean([0.0]) == pytest.approx(1.0)  # log(1.0) == 0
    assert 0.0 < geo_mean([math.log(0.01)]) < 1.0


# --------------------------------------------------------------------------
# percentile / entropy
# --------------------------------------------------------------------------


def test_percentile_matches_linear_interpolation():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(values, 0.0) == pytest.approx(0.1)
    assert percentile(values, 0.5) == pytest.approx(0.3)
    assert percentile(values, 1.0) == pytest.approx(0.5)
    assert percentile(values, 0.25) == pytest.approx(0.2)


def test_percentile_edge_cases():
    assert percentile([], 0.5) == 0.0
    assert percentile([0.42], 0.1) == pytest.approx(0.42)


def test_normalized_entropy_extremes():
    # All mass on one token -> 0. Uniform over k -> 1.
    peaked = [math.log(0.999), math.log(0.001)]
    assert normalized_entropy(peaked) < 0.02
    uniform = [math.log(0.25)] * 4
    assert normalized_entropy(uniform) == pytest.approx(1.0, abs=1e-9)


def test_normalized_entropy_needs_two_alternatives():
    assert normalized_entropy([math.log(0.9)]) == 0.0
    assert normalized_entropy([]) == 0.0


# --------------------------------------------------------------------------
# extract_metrics
# --------------------------------------------------------------------------


def test_extract_metrics_empty_forces_escalation():
    """Missing logprobs must not silently score as confident."""
    m = extract_metrics([])
    assert m == EMPTY_METRICS
    assert m.geo_mean == 0.0
    assert m.frac_low == 1.0


def test_extract_metrics_computes_every_feature():
    logprobs = [
        lp(math.log(0.9), top=[math.log(0.9), math.log(0.05)]),
        lp(math.log(0.3), top=[math.log(0.3), math.log(0.25)]),
        lp(math.log(0.6), top=[math.log(0.6), math.log(0.2)]),
    ]
    m = extract_metrics(logprobs)
    assert m.n_tokens == 3
    assert m.geo_mean == pytest.approx((0.9 * 0.3 * 0.6) ** (1 / 3))
    assert m.min_prob == pytest.approx(0.3)
    assert m.frac_low == pytest.approx(1 / 3)  # only 0.3 is below 0.5
    assert m.mean_margin > 0
    assert 0.0 <= m.mean_norm_entropy <= 1.0


def test_frac_low_boundary_is_exclusive():
    """A token at exactly LOW_PROB is not 'low'."""
    m = extract_metrics([lp(math.log(LOW_PROB))])
    assert m.frac_low == 0.0


def test_feature_vector_order_is_stable():
    """The calibrator is fitted against this order — reordering silently
    invalidates eval/calibrator.json."""
    m = ConfidenceMetrics(0.6, 0.1, 0.2, 1.5, 0.4, 0.25, 42)
    assert m.feature_vector() == [0.6, 0.1, 0.2, 1.5, 0.4, 0.25, 42.0]


def test_missing_top_logprobs_degrades_gracefully():
    """top_logprobs=0 is legal; margin/entropy just go to zero."""
    m = extract_metrics([lp(math.log(0.8)), lp(math.log(0.7))])
    assert m.mean_margin == 0.0
    assert m.mean_norm_entropy == 0.0
    assert m.geo_mean == pytest.approx((0.8 * 0.7) ** 0.5)


# --------------------------------------------------------------------------
# hedge detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I don't know the answer to that.",
        "I do not know.",
        "I'm not sure, but possibly Paris.",
        "As an AI, I cannot browse the web.",
        "I cannot provide that information.",
        "There is insufficient information to answer.",
        "That's unclear from the context given.",
    ],
)
def test_find_hedge_detects_refusals(text):
    assert find_hedge(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "Paris.",
        "The capital of France is Paris.",
        "Rainbows form because light refracts through water droplets.",
        "I know the answer is 42.",
    ],
)
def test_find_hedge_ignores_confident_answers(text):
    assert find_hedge(text) is None


# --------------------------------------------------------------------------
# gate decisions
# --------------------------------------------------------------------------


def test_gate_accepts_confident_answer():
    logprobs = [lp(math.log(0.95)) for _ in range(20)]
    v = evaluate_gate("small", logprobs, "The capital is Paris.", threshold=0.55)
    assert v.verdict == "accept"
    assert v.confidence == pytest.approx(0.95)
    assert "0.95" in v.reason


def test_gate_escalates_below_threshold():
    logprobs = [lp(math.log(0.40)) for _ in range(20)]
    v = evaluate_gate("small", logprobs, "Some uncertain answer here.", threshold=0.55)
    assert v.verdict == "escalate"
    assert v.confidence == pytest.approx(0.40)


def test_gate_escalates_on_hedge_despite_high_confidence():
    """A model can be fluently, confidently uncertain. Logprobs miss that."""
    logprobs = [lp(math.log(0.98)) for _ in range(20)]
    v = evaluate_gate("small", logprobs, "I don't know the answer.", threshold=0.55)
    assert v.verdict == "escalate"
    assert v.hedged is True
    assert v.confidence > 0.9  # score was high; the veto overrode it
    assert "hedged" in v.reason


def test_gate_escalates_when_logprobs_missing():
    v = evaluate_gate("small", [], "Some answer.", threshold=0.55)
    assert v.verdict == "escalate"
    assert v.confidence == 0.0
    assert "no logprobs" in v.reason


def test_gate_reason_is_human_readable():
    """Invariant 2: reason is prose, not an enum."""
    v = evaluate_gate("large", [lp(math.log(0.3))] * 5, "answer", threshold=0.62)
    assert " " in v.reason and v.reason.lower() == v.reason.lower()
    assert v.reason not in {"escalate", "ESCALATE", "below_threshold"}


def test_gate_verdict_serializes_to_api_shape():
    v = evaluate_gate("small", [lp(math.log(0.9), top=[math.log(0.9), math.log(0.05)])] * 4,
                      "ok", threshold=0.55)
    d = v.as_dict()
    assert set(d) == {"gate", "confidence", "threshold", "verdict", "reason", "metrics"}
    assert set(d["metrics"]) == {
        "geo_mean", "min_prob", "p10_prob", "mean_margin",
        "mean_norm_entropy", "frac_low", "n_tokens",
    }


def test_gate_threshold_boundary_accepts():
    """confidence == threshold accepts; the check is `< threshold`."""
    logprobs = [lp(math.log(0.55))] * 10
    v = evaluate_gate("small", logprobs, "answer", threshold=0.55)
    assert v.verdict == "accept"


# --------------------------------------------------------------------------
# calibrator
# --------------------------------------------------------------------------


def test_calibrated_mode_uses_the_calibrator():
    m = ConfidenceMetrics(0.6, 0.1, 0.2, 1.5, 0.4, 0.25, 42)
    cal = LogisticCalibrator(weights=[0.0] * 7, bias=0.0)
    assert score(m, mode="calibrated", calibrator=cal) == pytest.approx(0.5)
    assert score(m, mode="raw") == pytest.approx(0.6)


def test_calibrated_mode_falls_back_when_unfitted():
    """No calibrator yet must not break serving."""
    m = ConfidenceMetrics(0.6, 0.1, 0.2, 1.5, 0.4, 0.25, 42)
    assert score(m, mode="calibrated", calibrator=None) == pytest.approx(0.6)


def test_calibrator_rejects_wrong_feature_count():
    cal = LogisticCalibrator(weights=[1.0, 2.0], bias=0.0)
    with pytest.raises(ValueError, match="expects 2 features"):
        cal.predict_proba([1.0, 2.0, 3.0])


def test_calibrator_sigmoid_is_numerically_stable():
    cal = LogisticCalibrator(weights=[1.0], bias=0.0)
    assert cal.predict_proba([1000.0]) == pytest.approx(1.0)
    assert cal.predict_proba([-1000.0]) == pytest.approx(0.0)
    assert 0.0 <= cal.predict_proba([-800.0]) <= 1.0


# --------------------------------------------------------------------------
# Real captured data — qwen2.5:0.5b, temperature=0, 2026-07-28
# --------------------------------------------------------------------------

CAPTURED_PARIS = [
    {
        "token": "Paris",
        "logprob": -0.027870038524270058,
        "bytes": [80, 97, 114, 105, 115],
        "top_logprobs": [
            {"token": "Paris", "logprob": -0.027870038524270058},
            {"token": " Paris", "logprob": -4.606834411621094},
            {"token": "L", "logprob": -5.052446365356445},
        ],
    }
]


def test_real_ollama_response_parses():
    """Guards against the response shape drifting under us."""
    m = extract_metrics(CAPTURED_PARIS)
    assert m.n_tokens == 1
    assert m.geo_mean == pytest.approx(0.9725, abs=1e-3)
    assert m.frac_low == 0.0
    assert m.mean_margin == pytest.approx(4.579, abs=1e-2)


def test_real_easy_prompt_clears_the_raw_threshold():
    """qwen2.5:0.5b answering 'capital of France' measured geo_mean 0.9722 —
    comfortably over THRESHOLD_SMALL=0.55."""
    v = evaluate_gate("small", CAPTURED_PARIS, "Paris", threshold=0.55)
    assert v.verdict == "accept"
