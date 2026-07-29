"""Tests for the eval harness: dataset, grading, calibration.

The calibration tests matter most. They exist to make sure the validation half
actually *fails* when it should — a validator that always passes is worse than
no validator, because it launders a guess into a measured-looking number.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.calibration import (
    MAX_ECE,
    Calibrator,
    bootstrap_ece_ci,
    brier_score,
    cross_val_predictions,
    expected_calibration_error,
    fit_calibrator,
    reliability_bins,
    should_flip,
    validate,
)
from eval.dataset import DATASET, EvalPrompt, auto_gradable, by_category, judge_gradable
from eval.grading import grade, grade_auto, judge_prompt, parse_judge

# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_dataset_is_the_documented_size():
    assert 100 <= len(DATASET) <= 140


def test_every_id_is_unique():
    ids = [p.id for p in DATASET]
    assert len(ids) == len(set(ids))


def test_all_six_categories_are_represented():
    assert set(by_category()) == {
        "easy_factual", "arithmetic", "obscure_factual", "summarization", "reasoning", "code"
    }


def test_auto_subset_is_large_enough_to_fit_on():
    """The free path has to be usable on its own, or the "no spend" story is
    theatre."""
    assert len(auto_gradable()) >= 50


def test_auto_prompts_all_carry_expected_answers():
    for p in auto_gradable():
        assert p.expected, f"{p.id} is auto-graded but has no expected answer"


def test_judge_prompts_carry_no_expected_answers():
    for p in judge_gradable():
        assert not p.expected, f"{p.id} is judge-graded but has expected answers"


def test_difficulty_has_range():
    """A set of only easy questions fits a calibrator that is confident about
    everything."""
    cats = by_category()
    assert cats["easy_factual"] >= 20
    assert cats["obscure_factual"] >= 10
    assert cats["reasoning"] >= 15


def test_near_miss_pairs_are_present():
    """Australia/Austria and similar traps, so the eval measures the failure
    the guards exist for."""
    prompts = " ".join(p.prompt for p in DATASET)
    assert "Australia" in prompts and "Austria" in prompts


# --------------------------------------------------------------------------
# auto grading
# --------------------------------------------------------------------------


def test_correct_answer_passes():
    p = EvalPrompt("t", "easy_factual", "Capital of France?", "auto", ("Paris",))
    assert grade_auto(p, "Paris").acceptable is True
    assert grade_auto(p, "The capital is Paris.").acceptable is True


def test_wrong_answer_fails():
    p = EvalPrompt("t", "easy_factual", "Capital of France?", "auto", ("Paris",))
    assert grade_auto(p, "London").acceptable is False


def test_empty_answer_fails():
    p = EvalPrompt("t", "easy_factual", "q", "auto", ("Paris",))
    assert grade_auto(p, "   ").acceptable is False


def test_matching_is_token_bounded_not_substring():
    """`"4" in "42"` is True, which would mark a wrong arithmetic answer
    correct. This is the bug that would quietly inflate every accuracy number."""
    p = EvalPrompt("t", "arithmetic", "2+2?", "auto", ("4",))
    assert grade_auto(p, "42").acceptable is False
    assert grade_auto(p, "The answer is 4.").acceptable is True


def test_accents_are_folded():
    p = EvalPrompt("t", "easy_factual", "q", "auto", ("Brasilia",))
    assert grade_auto(p, "Brasília").acceptable is True


def test_case_and_punctuation_are_ignored():
    p = EvalPrompt("t", "easy_factual", "q", "auto", ("Paris",))
    assert grade_auto(p, "paris!").acceptable is True


def test_forbidden_answers_are_checked_first():
    """A model that names a plausible-but-wrong answer has failed even if it
    also mentions something correct. The M0 probe did exactly this: "Treaty of
    Kadesh, 1956, Egypt and Israel" mentions Egypt correctly alongside a
    fabricated date and the wrong counterparty."""
    p = EvalPrompt(
        "t", "obscure_factual", "q", "auto",
        expected=("1259", "Hittites"), forbidden=("1956",),
    )
    result = grade_auto(p, "It was signed in 1956 between Egypt and the Hittites.")
    assert result.acceptable is False
    assert "1956" in result.reason


def test_multiple_accepted_forms():
    p = EvalPrompt("t", "easy_factual", "q", "auto", ("7", "seven"))
    assert grade_auto(p, "seven").acceptable is True
    assert grade_auto(p, "7").acceptable is True


def test_grade_reports_which_mode_was_used():
    """A report must be able to say what kind of evidence a number rests on."""
    p = EvalPrompt("t", "easy_factual", "q", "auto", ("Paris",))
    assert grade_auto(p, "Paris").mode == "auto"


# --------------------------------------------------------------------------
# judge grading
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reply,expected", [
    ("ACCEPTABLE", True),
    ("acceptable", True),
    ("The answer is ACCEPTABLE.", True),
    ("UNACCEPTABLE", False),
    ("unacceptable", False),
])
def test_judge_verdicts_are_parsed(reply, expected):
    g = parse_judge(reply)
    assert g is not None and g.acceptable is expected


def test_unacceptable_is_not_read_as_acceptable():
    """"UNACCEPTABLE" contains "ACCEPTABLE" as a substring — checking the
    positive case first would invert every negative verdict."""
    g = parse_judge("UNACCEPTABLE")
    assert g is not None and g.acceptable is False


def test_unparseable_verdict_is_not_a_label():
    """An off-rubric reply must not silently become a label."""
    assert parse_judge("I think it's pretty good?") is None
    assert parse_judge("") is None


def test_judge_prompt_includes_all_three_parts():
    text = judge_prompt("Q?", "reference text", "candidate text")
    assert "Q?" in text and "reference text" in text and "candidate text" in text


def test_judge_prompt_without_reply_is_ungradable():
    p = EvalPrompt("t", "reasoning", "q", "judge")
    assert grade(p, "some answer") is None


# --------------------------------------------------------------------------
# calibration: fitting
# --------------------------------------------------------------------------


def make_separable(n=80, seed=0):
    """A feature that genuinely predicts the label, plus noise features."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < signal).astype(float)  # P(y=1) == signal
    X = np.column_stack([signal] + [rng.normal(0, 1, n) for _ in range(6)])
    return X, labels


def test_calibrator_outputs_probabilities():
    X, y = make_separable()
    cal = fit_calibrator(X, y)
    probs = cal.predict_many(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_calibrator_learns_the_signal():
    X, y = make_separable(n=200)
    cal = fit_calibrator(X, y)
    probs = cal.predict_many(X)
    # Higher signal should mean higher predicted probability.
    assert np.corrcoef(X[:, 0], probs)[0, 1] > 0.5


def test_sigmoid_is_numerically_stable():
    cal = Calibrator(weights=[1.0], bias=0.0)
    assert cal.predict_proba([1000.0]) == pytest.approx(1.0)
    assert cal.predict_proba([-1000.0]) == pytest.approx(0.0)


def test_constant_feature_does_not_produce_nan():
    """A zero-variance feature would divide by zero during standardization and
    poison every prediction with NaN."""
    n = 60
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(n)] + [rng.normal(size=n) for _ in range(6)])
    y = (rng.uniform(size=n) < 0.5).astype(float)
    probs = fit_calibrator(X, y).predict_many(X)
    assert np.all(np.isfinite(probs))


def test_regularization_prevents_saturation():
    """Unregularized, a separable fit drives weights to infinity and outputs
    only 0.0 and 1.0 — perfectly confident and completely uncalibrated."""
    n = 40
    X = np.column_stack([np.linspace(-3, 3, n)] + [np.zeros(n) for _ in range(6)])
    y = (X[:, 0] > 0).astype(float)  # perfectly separable
    probs = fit_calibrator(X, y, l2=1.0).predict_many(X)
    assert 0.0 < probs.min() and probs.max() < 1.0


# --------------------------------------------------------------------------
# calibration: validation — must FAIL when it should
# --------------------------------------------------------------------------


def test_perfect_predictions_have_zero_ece():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(probs, labels, n_bins=2) == pytest.approx(0.0)


def test_confidently_wrong_predictions_have_high_ece():
    probs = np.array([0.95] * 10)
    labels = np.zeros(10)
    assert expected_calibration_error(probs, labels, n_bins=2) > 0.9


def test_bins_are_equal_count_not_equal_width():
    """Equal-width bins leave most bins nearly empty when predictions cluster,
    and an empty bin is noise indistinguishable from miscalibration."""
    probs = np.concatenate([np.full(18, 0.9), np.array([0.1, 0.2])])
    labels = np.ones(20)
    counts = [b["count"] for b in reliability_bins(probs, labels, n_bins=4)]
    assert max(counts) - min(counts) <= 1


def test_brier_rewards_calibrated_probabilities():
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(np.array([0.9, 0.1, 0.9, 0.1]), labels) < brier_score(
        np.array([0.5, 0.5, 0.5, 0.5]), labels
    )


def test_bootstrap_ci_is_conservative():
    """ECE is biased *upward* under bootstrap resampling: duplicated points
    look miscalibrated, so the resampled distribution sits above the point
    estimate and the interval does not bracket it.

    That is the useful direction for a flip rule — a pessimistic upper bound
    makes `calibrated` mode harder to justify, never easier. Asserted as the
    property the rule actually depends on.
    """
    rng = np.random.default_rng(3)
    probs = rng.uniform(size=100)
    labels = (rng.uniform(size=100) < probs).astype(float)
    point = expected_calibration_error(probs, labels)
    lo, hi = bootstrap_ece_ci(probs, labels)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi >= point, "the upper bound must not be optimistic about calibration"


def test_validation_rejects_random_predictions():
    """The important negative case: noise must not pass."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 7))
    y = (rng.uniform(size=60) < 0.5).astype(float)  # label independent of X
    report = validate("small", X, y)
    assert report.passed is False
    assert report.verdict


def test_validation_rejects_too_little_data():
    X = np.random.default_rng(0).normal(size=(6, 7))
    y = np.array([1.0, 0, 1, 0, 1, 0])
    report = validate("small", X, y)
    assert report.passed is False
    assert "insufficient data" in report.verdict


def test_validation_rejects_single_class_data():
    X = np.random.default_rng(0).normal(size=(40, 7))
    y = np.ones(40)  # every answer acceptable — nothing to learn
    assert validate("small", X, y).passed is False


def test_out_of_fold_predictions_cover_every_row():
    X, y = make_separable(n=60)
    oof = cross_val_predictions(X, y)
    assert len(oof) == len(y)
    assert np.all((oof >= 0) & (oof <= 1))


def test_should_flip_requires_every_gate_to_pass():
    """One trustworthy gate and one untrustworthy gate is not a calibrated
    system — shipping it would put a measured number and a guess behind the
    same 0.85 label."""
    rng = np.random.default_rng(11)
    X_good, y_good = make_separable(n=200, seed=5)
    good = validate("large", X_good, y_good)
    bad = validate("small", rng.normal(size=(60, 7)), (rng.uniform(size=60) < 0.5).astype(float))

    assert should_flip({"large": good}) is good.passed
    assert should_flip({"small": bad, "large": good}) is False
    assert should_flip({}) is False


def test_validation_report_is_serializable():
    X, y = make_separable(n=60)
    d = validate("small", X, y).as_dict()
    for key in ("gate", "n", "ece", "ece_ci_high", "brier", "brier_baseline", "passed", "verdict"):
        assert key in d


def test_thresholds_are_the_documented_ones():
    assert MAX_ECE == 0.10


# --------------------------------------------------------------------------
# report: the sweep metrics
# --------------------------------------------------------------------------


def test_sweep_counts_false_cheap_and_false_expensive():
    """The two error types are not symmetric: false-cheap serves a wrong
    answer, false-expensive just wastes money."""
    from eval.report import sweep

    scores = np.array([0.9, 0.9, 0.2, 0.2])
    labels = np.array([1.0, 0.0, 1.0, 0.0])  # one right and one wrong on each side
    row = sweep(scores, labels, [0.5])[0]
    assert row.accepted == 2
    assert row.escalated == 2
    assert row.false_cheap == 1       # accepted the 0.9-scoring wrong answer
    assert row.false_expensive == 1   # escalated the 0.2-scoring right answer


def test_sweep_at_zero_threshold_accepts_everything():
    from eval.report import sweep

    scores, labels = np.array([0.1, 0.9]), np.array([0.0, 1.0])
    row = sweep(scores, labels, [0.0])[0]
    assert row.escalated == 0
    assert row.false_cheap == 1
    assert row.false_expensive == 0


def test_sweep_at_high_threshold_escalates_everything():
    """Escalating everything gives perfect quality and zero savings — which is
    why escalation rate alone is a useless headline."""
    from eval.report import sweep

    scores, labels = np.array([0.1, 0.9]), np.array([0.0, 1.0])
    row = sweep(scores, labels, [1.1])[0]
    assert row.escalated == 2
    assert row.false_cheap == 0
    assert row.quality_preserved == 1.0


def test_quality_preserved_assumes_escalation_fixes_things():
    """A documented optimism: reasonable for the frontier, generous for gate
    small escalating to the 3B."""
    from eval.report import sweep

    scores, labels = np.array([0.1]), np.array([0.0])  # wrong, and escalated
    assert sweep(scores, labels, [0.5])[0].quality_preserved == 1.0


# --------------------------------------------------------------------------
# retrying a rate-limited provider
#
# A free-tier key rate-limits, and until `with_retry` existed a single 429
# aborted the whole run. Progress was never lost — results are checkpointed
# after every prompt — but finishing meant restarting the script by hand
# however many times the provider decided to push back.
# --------------------------------------------------------------------------


async def test_a_rate_limited_call_is_retried_until_it_succeeds():
    from app.providers.base import ProviderUnavailable
    from eval.run_eval import with_retry

    attempts = {"n": 0}

    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderUnavailable("frontier rate-limited the request (429)")
        return "the answer"

    assert await with_retry(flaky, what="probe", base_delay=0.001) == "the answer"
    assert attempts["n"] == 3


async def test_a_timeout_is_retried_too():
    """A judge call that hangs is far more likely to be a slow provider than a
    broken prompt."""
    from app.providers.base import ProviderTimeout
    from eval.run_eval import with_retry

    attempts = {"n": 0}

    async def slow():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ProviderTimeout("timed out")
        return "ok"

    assert await with_retry(slow, what="probe", base_delay=0.001) == "ok"


async def test_it_gives_up_rather_than_looping_forever():
    """A provider that is genuinely down must surface, not spin."""
    from app.providers.base import ProviderUnavailable
    from eval.run_eval import with_retry

    attempts = {"n": 0}

    async def dead():
        attempts["n"] += 1
        raise ProviderUnavailable("down")

    with pytest.raises(ProviderUnavailable):
        await with_retry(dead, what="probe", attempts=3, base_delay=0.001)
    assert attempts["n"] == 3


async def test_a_successful_call_is_not_delayed():
    """The common case must cost nothing — no sleep on the happy path."""
    import time

    from eval.run_eval import with_retry

    async def fine():
        return "ok"

    started = time.perf_counter()
    assert await with_retry(fine, what="probe", base_delay=10.0) == "ok"
    assert time.perf_counter() - started < 0.5


def test_the_judge_token_budget_has_headroom_for_a_reasoning_model():
    """A one-word verdict needs more than a one-word budget.

    Reasoning models spend hidden thinking tokens against `max_tokens` and
    those never appear in `completion_tokens`. At the old hardcoded 10,
    gemini-3-flash-preview returned an empty string, every verdict parsed as
    None, and the run would have produced zero labels after paying for 129
    calls — with no error anywhere. Measured: 10 fails, 64 works.
    """
    from app.config import Settings

    assert Settings(_env_file=None).eval_judge_max_tokens >= 64


def test_an_unparseable_verdict_is_not_silently_dropped():
    """`parse_judge` returning None must be *counted*, not ignored — that is
    what made the failure invisible."""
    from eval.grading import parse_judge

    assert parse_judge("") is None
    assert parse_judge("ACCEPTABLE") is not None


async def test_the_retry_honours_the_providers_own_delay():
    """Gemini's 429 body says "Please retry in 23.7s". Believing it beats
    guessing — blind exponential backoff either waits far too long or retries
    straight back into the same closed window."""
    import time

    from app.providers.base import ProviderUnavailable
    from eval.run_eval import with_retry

    attempts = {"n": 0}

    async def limited():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ProviderUnavailable("429: Please retry in 0.05s. RESOURCE_EXHAUSTED")
        return "ok"

    started = time.perf_counter()
    assert await with_retry(limited, what="probe", base_delay=60.0) == "ok"
    # It waited the hinted 0.05s (+1s margin), not the 60s default.
    assert time.perf_counter() - started < 5.0


async def test_pacing_holds_a_minimum_gap():
    """Staying under a rate limit is faster than retrying through one."""
    import time

    import eval.run_eval as re_mod

    re_mod._last_call_at = 0.0
    await re_mod.paced(0.0)          # disabled -> no delay
    started = time.perf_counter()
    await re_mod.paced(0.15)
    await re_mod.paced(0.15)
    assert time.perf_counter() - started >= 0.15


@pytest.mark.parametrize("reply,expected", [
    ("ACCEPTABLE", True),
    ("UNACCEPTABLE", False),
    ("ACCEPT", True),                  # gemini-2.5-flash's actual wording
    ("UNACCEPT", False),
    ("NOT ACCEPTABLE", False),
    ("  acceptable  ", True),
    ("Verdict: UNACCEPTABLE", False),
])
def test_the_judge_parser_accepts_the_wordings_models_actually_use(reply, expected):
    """Written against one model's phrasing, it returned None for another's —
    gemini-2.5-flash discriminated perfectly and scored nothing."""
    from eval.grading import parse_judge

    g = parse_judge(reply)
    assert g is not None, f"{reply!r} produced no label"
    assert g.acceptable is expected


@pytest.mark.parametrize("reply", ["", "maybe", "I think it's fine", "42"])
def test_an_unrecognised_verdict_is_still_refused(reply):
    """Tolerance must not become guessing. No label beats a wrong one."""
    from eval.grading import parse_judge

    assert parse_judge(reply) is None


def test_rejection_is_never_read_as_acceptance():
    """Every affirmative token is a substring of its negation. Checking
    positives first would invert every rejection — the worst bug a grader can
    have, and invisible in the output."""
    from eval.grading import parse_judge

    for reply in ("UNACCEPTABLE", "UNACCEPT", "NOT ACCEPTABLE", "unacceptable"):
        assert parse_judge(reply).acceptable is False, reply
