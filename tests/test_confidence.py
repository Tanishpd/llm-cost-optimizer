"""Unit tests for app/confidence.py.

No network — the module under test is pure functions, which is the whole reason
it was written that way. The two tests that read `eval/calibrator.json` are the
exception and are deliberate: that artifact is not exercised by serving (the
gateway ships `raw`), so without them nothing would notice it rotting.

The fixtures at the bottom are *real* logprob arrays captured from
qwen2.5:0.5b on 2026-07-28 (see docs/RESEARCH.md §2), not invented ones.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.confidence import (
    EMPTY_METRICS,
    FEATURE_NAMES,
    LOW_PROB,
    ConfidenceMetrics,
    LogisticCalibrator,
    calibrators_from_dict,
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
# calibrator (de)serialization
#
# The fit standardizes features before solving, so the weights live in
# standardized space. Applying them to raw features produces a number in
# [0, 1] that looks like a probability and is not one — the failure mode this
# section exists to prevent, since nothing downstream could detect it.
# --------------------------------------------------------------------------


def test_standardization_is_applied_when_present():
    raw = [0.7, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0]
    mean = [0.7, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0]
    scale = [0.1] * 7

    cal = LogisticCalibrator(weights=[5.0] * 7, bias=0.0, mean=mean, scale=scale)
    # Feature vector sits exactly on the fitted mean -> standardized to all
    # zeros -> sigmoid(bias) = 0.5, whatever the weights are.
    assert cal.predict_proba(raw) == pytest.approx(0.5)

    # Without mean/scale the same weights see raw magnitudes and saturate.
    uncal = LogisticCalibrator(weights=[5.0] * 7, bias=0.0)
    assert uncal.predict_proba(raw) == pytest.approx(1.0)


def test_standardization_shifts_the_score_in_the_right_direction():
    mean = [0.7, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0]
    scale = [0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 10.0]
    cal = LogisticCalibrator(weights=[1.0, 0, 0, 0, 0, 0, 0], bias=0.0,
                             mean=mean, scale=scale)

    above = cal.predict_proba([0.9, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0])
    below = cal.predict_proba([0.5, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0])
    assert above > 0.5 > below


def test_mismatched_mean_length_raises_rather_than_scoring():
    cal = LogisticCalibrator(weights=[1.0] * 7, bias=0.0, mean=[0.0] * 3, scale=[1.0] * 7)
    with pytest.raises(ValueError, match="mean/scale"):
        cal.predict_proba([0.5] * 7)


def test_calibrators_load_from_the_eval_json_shape():
    payload = {
        "calibrators": {
            "small": {
                "weights": [0.1] * 7, "bias": -0.5,
                "mean": [0.0] * 7, "scale": [1.0] * 7,
                "feature_names": list(FEATURE_NAMES),
                "fitted_temperature": 0.0, "n_train": 67,
            },
            "large": {"weights": [0.2] * 7, "bias": 0.5},
        }
    }
    cals = calibrators_from_dict(payload)

    assert set(cals) == {"small", "large"}
    assert cals["small"].n_train == 67
    assert cals["small"].scale == [1.0] * 7
    # A fit saved without mean/scale is taken at face value, not invented.
    assert cals["large"].mean == []


def test_the_real_calibrator_file_loads():
    """eval/calibrator.json is a real artifact and must stay loadable.

    It is not used for serving (the gateway ships `raw`), which is exactly why
    it needs a test — nothing else would notice it rotting.
    """
    path = Path(__file__).resolve().parent.parent / "eval" / "calibrator.json"
    cals = calibrators_from_dict(json.loads(path.read_text()))

    assert set(cals) == {"small", "large"}
    for gate, cal in cals.items():
        assert len(cal.weights) == len(FEATURE_NAMES), gate
        assert len(cal.mean) == len(FEATURE_NAMES), gate
        assert len(cal.scale) == len(FEATURE_NAMES), gate
        p = cal.predict_proba([0.7, 0.3, 0.5, 4.0, 0.25, 0.15, 30.0])
        assert 0.0 <= p <= 1.0, gate


def test_calibrators_from_dict_refuses_a_reordered_feature_vector():
    """A silently reordered vector scores plausible nonsense. Fail loudly."""
    scrambled = list(FEATURE_NAMES)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    payload = {"calibrators": {"small": {
        "weights": [0.1] * 7, "bias": 0.0, "feature_names": scrambled}}}

    with pytest.raises(ValueError, match="was fitted on"):
        calibrators_from_dict(payload)


def test_calibrators_from_dict_refuses_a_wrong_weight_count():
    payload = {"calibrators": {"small": {"weights": [0.1] * 5, "bias": 0.0}}}
    with pytest.raises(ValueError, match="5 weights"):
        calibrators_from_dict(payload)


def test_empty_payload_is_not_an_error():
    """A checkout that has never run eval must still start."""
    assert calibrators_from_dict({}) == {}
    assert calibrators_from_dict({"calibrators": None}) == {}


def test_serving_calibrator_agrees_with_the_eval_one():
    """app/confidence and eval/calibration must score identically.

    They are separate classes on purpose — eval may use numpy, serving may not
    — but a divergence would mean the validated ECE describes a model that is
    not the one making routing decisions.
    """
    from eval.calibration import Calibrator as EvalCalibrator

    path = Path(__file__).resolve().parent.parent / "eval" / "calibrator.json"
    raw = json.loads(path.read_text())["calibrators"]["small"]
    features = [0.62, 0.21, 0.44, 3.1, 0.31, 0.22, 48.0]

    serving = calibrators_from_dict({"calibrators": {"small": raw}})["small"]
    reference = EvalCalibrator(
        weights=raw["weights"], bias=raw["bias"],
        mean=raw["mean"], scale=raw["scale"],
    )
    assert serving.predict_proba(features) == pytest.approx(
        reference.predict_proba(features), abs=1e-12
    )


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


def test_sequence_probability_of_nothing_is_zero_not_one():
    """Mutant: the empty case returned 1.0 and no test noticed.

    This function is deliberately unused — it exists as an executable
    counter-example to the approach the brief implies (RESEARCH.md §3). A
    counter-example whose behaviour is unpinned teaches the wrong thing as
    easily as the right one, and 1.0 would read as "certain" for no evidence
    at all.
    """
    assert sequence_probability([]) == 0.0
