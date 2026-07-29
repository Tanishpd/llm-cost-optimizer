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


async def test_a_per_day_quota_stops_immediately_instead_of_backing_off():
    """A minute limit is worth waiting out; a day limit is not.

    Retrying a per-day quota burns the whole backoff ladder
    (30+60+120+240+480s) to rediscover something that will not change for
    hours — and then crashes, losing the fit for data already on disk.
    """
    import time

    from app.providers.base import ProviderUnavailable
    from eval.run_eval import DailyQuotaExhausted, with_retry

    calls = {"n": 0}

    async def daily_capped():
        calls["n"] += 1
        raise ProviderUnavailable(
            'frontier rate-limited the request (429): {"quotaId": '
            '"GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}')

    started = time.perf_counter()
    with pytest.raises(DailyQuotaExhausted):
        await with_retry(daily_capped, what="probe", base_delay=30.0)

    assert calls["n"] == 1, "it retried a quota that will not clear for hours"
    assert time.perf_counter() - started < 1.0, "it backed off anyway"


async def test_a_per_minute_limit_is_still_retried():
    """The distinction has to cut both ways, or the fix trades one failure for
    another."""
    from app.providers.base import ProviderUnavailable
    from eval.run_eval import with_retry

    calls = {"n": 0}

    async def minute_capped():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ProviderUnavailable("429: Please retry in 0.05s. RESOURCE_EXHAUSTED")
        return "ok"

    assert await with_retry(minute_capped, what="probe", base_delay=0.01) == "ok"
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# paid-call accounting
#
# The harness tracked tokens meticulously for the free local tiers and not at
# all for the API calls that cost money, so pricing a run meant estimating from
# character counts after the fact. These tests cover the accounting and the two
# things it is easy to get wrong: hidden thinking tokens, and pricing the judge
# at the reference model's rate.
# --------------------------------------------------------------------------


class _FakeCompletion:
    def __init__(self, tokens_in, tokens_out, usage=None):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.raw = {"usage": usage or {}}


def test_usage_keeps_the_providers_own_block():
    """Named fields would drop whatever a provider reports beyond the standard
    two — which is exactly where the billed-but-invisible tokens live."""
    from eval.run_eval import usage_of

    u = usage_of("judge", "gemini-2.5-flash", _FakeCompletion(
        100, 8, {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108,
                 "completion_tokens_details": {"reasoning_tokens": 412}}))
    assert (u.tokens_in, u.tokens_out) == (100, 8)
    assert u.detail["completion_tokens_details"]["reasoning_tokens"] == 412
    # The two standard counts are already first-class; keeping them in `detail`
    # too would double-count them in any sum over the block.
    assert "prompt_tokens" not in u.detail
    assert "completion_tokens" not in u.detail


@pytest.mark.parametrize("usage,expected", [
    ({"completion_tokens_details": {"reasoning_tokens": 412}}, 412),   # Gemini/OpenAI
    ({"thoughts_token_count": 77}, 77),                                # Gemini native
    ({"usageMetadata": {"thoughts_token_count": 5}}, 5),               # nested deeper
    ({}, 0),
    ({"completion_tokens_details": {"reasoning_tokens": 0}}, 0),
])
def test_thinking_tokens_are_found_wherever_the_provider_hid_them(usage, expected):
    """This project has already lost a run to hidden thinking: the judge was
    called with max_tokens=10, thinking ate the whole budget, and every verdict
    came back empty. Finding them is how the cost report can say so."""
    from eval.run_eval import thinking_tokens, usage_of

    assert thinking_tokens(usage_of("judge", "m", _FakeCompletion(1, 1, usage))) == expected


def test_cost_report_prices_the_judge_separately_from_the_reference():
    """The judge is deliberately a different model (D11), so it has different
    rates. One price for both is simply a wrong number."""
    from decimal import Decimal

    from eval.run_eval import ApiUsage, PromptResult, cost_report

    r = PromptResult(id="x", category="c", grade_mode="judge", prompt="p")
    r.usage = [ApiUsage("reference", "big", 1_000_000, 1_000_000),
               ApiUsage("judge", "small", 1_000_000, 1_000_000)]
    out = cost_report({"x": r}, prices={
        "reference": (Decimal("2.00"), Decimal("10.00")),
        "judge": (Decimal("0.30"), Decimal("2.50")),
    })
    assert "$12.0000" in out          # reference: 2 + 10
    assert "$2.8000" in out           # judge:     0.30 + 2.50
    assert "$14.8000" in out          # and the total is their sum


def test_cost_report_gives_a_range_when_no_total_settles_it():
    """A named `reasoning_tokens` with no `total_tokens` is genuinely
    ambiguous: OpenAI counts it inside `completion_tokens`, others do not.
    Picking one and being quietly wrong by 2x is worse than showing both."""
    from decimal import Decimal

    from eval.run_eval import ApiUsage, PromptResult, cost_report

    r = PromptResult(id="x", category="c", grade_mode="judge", prompt="p")
    r.usage = [ApiUsage("judge", "m", 0, 1_000_000,
                        {"completion_tokens_details": {"reasoning_tokens": 1_000_000}})]
    out = cost_report({"x": r}, prices={"judge": (Decimal("0"), Decimal("10.00"))})
    assert "$10.0000 to $20.0000" in out
    assert "thinking 1,000,000" in out


def test_a_total_tokens_gap_is_charged_exactly():
    """The shape Gemini's OpenAI layer actually returns, measured 2026-07-29:
    no `reasoning_tokens` key anywhere, thinking visible only as the gap in
    `total_tokens`. Billing on `completion_tokens` alone understates it, and
    nothing in the response says so."""
    from decimal import Decimal

    from eval.run_eval import (
        ApiUsage,
        PromptResult,
        billable_out,
        cost_report,
        thinking_tokens,
    )

    u = ApiUsage("judge", "gemini-2.5-flash", 9, 2, tokens_total=34)
    assert thinking_tokens(u) == 23, "9 + 2 + 23 = 34"
    assert billable_out(u) == 25, "the 23 hidden tokens are billed as output"

    r = PromptResult(id="x", category="c", grade_mode="judge", prompt="p")
    r.usage = [ApiUsage("judge", "m", 0, 1_000_000, tokens_total=2_000_000)]
    out = cost_report({"x": r}, prices={"judge": (Decimal("0"), Decimal("10.00"))})
    assert "$20.0000" in out and " to " not in out, "a total makes it exact, not a range"


def test_reasoning_effort_none_leaves_no_gap_to_charge():
    """The measured effect of turning thinking off: prompt + completion ==
    total exactly, so there is nothing hidden and nothing to estimate."""
    from eval.run_eval import ApiUsage, billable_out, thinking_tokens

    u = ApiUsage("judge", "gemini-2.5-flash", 9, 2, tokens_total=11)
    assert thinking_tokens(u) == 0
    assert billable_out(u) == 2


def test_cost_report_says_so_rather_than_reporting_zero():
    """A run cached before accounting existed cost money; reporting $0.00 for
    it would be a confident lie."""
    from eval.run_eval import PromptResult, cost_report

    r = PromptResult(id="x", category="c", grade_mode="judge", prompt="p")
    assert "no paid-call usage recorded" in cost_report({"x": r}, prices={})


def test_usage_survives_a_cache_round_trip(tmp_path, monkeypatch):
    from eval import run_eval as ev

    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path)
    r = ev.PromptResult(id="x", category="c", grade_mode="judge", prompt="p")
    r.usage = [ev.ApiUsage("judge", "m", 3, 4, {"completion_tokens_details":
                                                {"reasoning_tokens": 9}})]
    ev.save_cached("t", {"x": r})
    back = ev.load_cached("t")["x"].usage
    assert [(u.kind, u.tokens_in, u.tokens_out) for u in back] == [("judge", 3, 4)]
    assert ev.thinking_tokens(back[0]) == 9


def test_a_cache_without_usage_still_loads(tmp_path, monkeypatch):
    """Every run collected before this existed."""
    import json

    from eval import run_eval as ev

    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path)
    (tmp_path / "old.json").write_text(json.dumps({
        "x": {"id": "x", "category": "c", "grade_mode": "auto", "prompt": "p",
              "tiers": {}, "reference": "", "labels": {}, "label_source": {},
              "label_reason": {}}}))
    assert ev.load_cached("old")["x"].usage == []


# --------------------------------------------------------------------------
# token budget
# --------------------------------------------------------------------------


def _cached_at(budget):
    from eval.run_eval import PromptResult, TierResult

    r = PromptResult(id="x", category="c", grade_mode="auto", prompt="p")
    r.tiers["local_small"] = TierResult("local_small", "m", "answer", {"n_tokens": 5},
                                        1, 5, 1.0, max_tokens=budget)
    return {"x": r}


def test_resuming_across_a_budget_change_is_refused():
    """`n_tokens` is a calibrator feature, so a cache mixing two budgets fits a
    distribution no configuration ever produced — and both halves look like
    valid rows, so nothing downstream can notice."""
    from eval.run_eval import BudgetMismatch, check_budget

    with pytest.raises(BudgetMismatch) as exc:
        check_budget(_cached_at(120), 512, "full")
    assert "120" in str(exc.value) and "512" in str(exc.value)
    assert "rm eval/runs/full.json" in str(exc.value), "it must say how to proceed"


def test_a_matching_budget_resumes_normally():
    from eval.run_eval import check_budget

    check_budget(_cached_at(512), 512, "full")


def test_a_cache_predating_the_field_is_unknown_not_mismatched():
    """Refusing to load every historical run would be worse than saying so."""
    from eval.run_eval import check_budget

    check_budget(_cached_at(0), 512, "full")


def test_the_eval_calibrates_at_the_length_production_serves():
    """The default used to be 120 while the gateway served 512, so the
    calibrator was fitted to answers a quarter the length of the ones it would
    score — and the judge compared those against full-length references."""
    from app.config import Settings
    from eval.run_eval import build_parser

    parser = build_parser()
    assert parser.parse_args([]).max_tokens == 0, "0 means 'take the setting'"
    # The resolution `main` performs. 512, not the old hardcoded 120.
    resolved = parser.parse_args([]).max_tokens or Settings().default_max_tokens
    assert resolved == Settings().default_max_tokens == 512
    assert parser.parse_args(["--max-tokens", "64"]).max_tokens == 64, "still overridable"


async def test_the_paid_path_records_what_it_spent(tmp_path, monkeypatch):
    """End-to-end through the real `collect`, with the providers stubbed.

    The unit tests above cover the accounting functions; this covers the wiring
    to them. Both call sites — reference and judge — have to append usage, and
    a missed one shows up here as a short list rather than as a plausible-
    looking total that is quietly missing a third of the calls.
    """
    from app.providers.base import Completion
    from eval import run_eval as ev
    from eval.dataset import DATASET

    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path)

    class FakeOllama:
        def __init__(self, client): pass
        async def chat(self, **kw):
            return Completion(content="a local answer", model=kw["model"],
                              tokens_in=5, tokens_out=3,
                              logprobs=[{"logprob": -0.2}] * 3)

    class FakeApi:
        def __init__(self, reply): self.reply, self.calls = reply, 0
        async def chat(self, **kw):
            self.calls += 1
            return Completion(
                content=self.reply, model=kw["model"], tokens_in=200, tokens_out=9,
                raw={"usage": {"prompt_tokens": 200, "completion_tokens": 9,
                               "completion_tokens_details": {"reasoning_tokens": 40}}})

    monkeypatch.setattr(ev, "OllamaProvider", FakeOllama)
    frontier, judge = FakeApi("the reference answer"), FakeApi("ACCEPTABLE")
    prompt = next(p for p in DATASET if p.grade != "auto")

    results = await ev.collect(
        [prompt], tag="t", small_model="s", large_model="l", max_tokens=512,
        use_real_frontier=True, frontier=frontier, judge=judge,
        judge_model="judge-m", frontier_model="ref-m")

    r = results[prompt.id]
    assert r.labels == {"local_small": True, "local_large": True}
    # 1 reference + 1 judge per local tier. A missing call site shows up here.
    assert [u.kind for u in r.usage] == ["reference", "judge", "judge"]
    assert frontier.calls == 1 and judge.calls == 2
    assert all(ev.thinking_tokens(u) == 40 for u in r.usage)

    from decimal import Decimal
    out = ev.cost_report(results, prices={
        "reference": (Decimal("2.50"), Decimal("10.00")),
        "judge": (Decimal("0.30"), Decimal("2.50"))})
    assert "reference" in out and "judge" in out and "thinking" in out
    # And it survives the round trip the checkpoint performs after every prompt.
    assert [u.kind for u in ev.load_cached("t")[prompt.id].usage] == \
        ["reference", "judge", "judge"]


# --------------------------------------------------------------------------
# reasoning_effort passthrough
# --------------------------------------------------------------------------


def _capture_post(monkeypatch):
    """A FrontierProvider whose POST body we can inspect."""

    from app.providers.frontier import FrontierProvider

    sent = {}

    class FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "ACCEPTABLE"},
                                 "finish_reason": "stop"}],
                    "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    class FakeClient:
        async def post(self, url, **kw):
            sent.update(kw.get("json") or {})
            return FakeResponse()

    return FrontierProvider(FakeClient(), base_url="https://x/v1", api_key="k"), sent


async def test_reasoning_effort_is_sent_when_set(monkeypatch):
    provider, sent = _capture_post(monkeypatch)
    await provider.chat(model="gemini-2.5-flash", messages=[{"role": "user", "content": "q"}],
                        reasoning_effort="none")
    assert sent["reasoning_effort"] == "none"


async def test_reasoning_effort_is_absent_when_unset(monkeypatch):
    """The field is standard; its accepted values are not. Anthropic — this
    project's default judge — does not use it, so sending it unconditionally
    would 400 against the documented configuration."""
    provider, sent = _capture_post(monkeypatch)
    await provider.chat(model="claude-sonnet-5", messages=[{"role": "user", "content": "q"}])
    assert "reasoning_effort" not in sent


async def test_a_rejected_reasoning_effort_says_what_to_do(monkeypatch):
    """Which values a model accepts varies per model within one provider, so
    this 400 arrives mid-run with an opaque body."""

    from app.providers.base import ProviderUnavailable
    from app.providers.frontier import FrontierProvider

    class Rejecting:
        status_code = 400
        text = '{"error": {"message": "reasoning_effort not supported"}}'
        def json(self):
            return {"error": {"message": "reasoning_effort not supported"}}

    class FakeClient:
        async def post(self, url, **kw):
            return Rejecting()

    provider = FrontierProvider(FakeClient(), base_url="https://x/v1", api_key="k")
    with pytest.raises(ProviderUnavailable) as exc:
        await provider.chat(model="gemini-3-flash-preview",
                            messages=[{"role": "user", "content": "q"}],
                            reasoning_effort="none")
    assert "none" in exc.value.fix and "gemini-2.5-flash" in exc.value.fix


async def test_the_eval_passes_reasoning_effort_to_the_judge_only_when_configured(
        tmp_path, monkeypatch):
    """Wiring, not just capability — the setting has to reach the call."""
    from app.providers.base import Completion
    from eval import run_eval as ev
    from eval.dataset import DATASET

    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path)

    class FakeOllama:
        def __init__(self, client): pass
        async def chat(self, **kw):
            return Completion(content="a", model="m", tokens_in=1, tokens_out=1,
                              logprobs=[{"logprob": -0.2}])

    class Recorder:
        def __init__(self, reply): self.reply, self.efforts = reply, []
        async def chat(self, **kw):
            self.efforts.append(kw.get("reasoning_effort"))
            return Completion(content=self.reply, model="m", tokens_in=1, tokens_out=1,
                              raw={"usage": {}})

    monkeypatch.setattr(ev, "OllamaProvider", FakeOllama)
    frontier, judge = Recorder("ref"), Recorder("ACCEPTABLE")
    p = next(x for x in DATASET if x.grade != "auto")
    await ev.collect([p], tag="t", small_model="s", large_model="l", max_tokens=512,
                     use_real_frontier=True, frontier=frontier, judge=judge,
                     judge_model="j", frontier_model="f",
                     judge_reasoning_effort="none")

    assert judge.efforts == ["none", "none"], "the judge must get it"
    assert frontier.efforts == [""], "the reference must not — it is the gold standard"


# --------------------------------------------------------------------------
# the eval's own spend ceiling
#
# MAX_DAILY_SPEND_USD lives in BudgetLedger, on the gateway's request path.
# run_eval calls the provider directly and never touches it, so the one
# component that deliberately spends real money had no cap on it at all.
# --------------------------------------------------------------------------


def _tracker(ceiling="5.00"):
    from decimal import Decimal

    from eval.run_eval import SpendTracker

    return SpendTracker(Decimal(ceiling), {
        "reference": (Decimal("0.50"), Decimal("3.00")),
        "judge": (Decimal("0.30"), Decimal("2.50")),
    })


def test_the_ceiling_ships_armed():
    """D27: a business policy may default to off, a safety limit may not."""
    from decimal import Decimal

    from app.config import Settings

    assert Settings(_env_file=None).eval_max_spend_usd == Decimal("5.00")
    assert _tracker().armed is True


def test_it_can_be_disarmed_explicitly():
    assert _tracker("0").armed is False
    _tracker("0").check()          # never raises, however much is spent


def test_spend_accumulates_from_provider_reported_tokens():
    from eval.run_eval import ApiUsage

    t = _tracker()
    t.add(ApiUsage("reference", "m", 1_000_000, 0))          # 1M in  @ $0.50
    t.add(ApiUsage("judge", "m", 0, 1_000_000))              # 1M out @ $2.50
    assert float(t.spent) == pytest.approx(3.00)


def test_thinking_tokens_count_toward_the_ceiling():
    """They are billed as output. A ceiling that ignored them would be wrong by
    36% on the measured run, in the direction that costs money."""
    from eval.run_eval import ApiUsage

    t = _tracker()
    t.add(ApiUsage("reference", "m", 0, 1_000, tokens_total=101_000))
    # billable output is total - input = 101,000, not the 1,000 visible.
    assert float(t.spent) == pytest.approx(101_000 / 1e6 * 3.00)


def test_it_binds_before_the_call_not_after():
    """A ceiling that notices only once it has been crossed is a report."""
    from eval.run_eval import ApiUsage, SpendCeilingReached

    t = _tracker("1.00")
    t.check()                                                 # nothing spent yet
    t.add(ApiUsage("judge", "m", 0, 1_000_000))               # $2.50, over
    with pytest.raises(SpendCeilingReached, match="EVAL_MAX_SPEND_USD"):
        t.check()


def test_a_normal_run_never_approaches_it():
    """The measured full run is $0.1139 against a $5.00 ceiling — ~45x
    headroom, so this must not fire in ordinary use."""
    import json
    from decimal import Decimal
    from pathlib import Path

    from eval.run_eval import SpendTracker, load_cached

    root = Path(__file__).resolve().parent.parent
    prices = {"reference": (Decimal("0.50"), Decimal("3.00")),
              "judge": (Decimal("0.30"), Decimal("2.50"))}
    t = SpendTracker(json.loads((root / "eval/calibrator.json").read_text())
                     and Decimal("5.00"), prices)
    for r in load_cached("full").values():
        for u in r.usage:
            t.add(u)
    assert float(t.spent) == pytest.approx(0.1139, abs=0.001)
    t.check()   # must not raise on a real, complete run


async def test_the_ceiling_actually_stops_a_run(tmp_path, monkeypatch):
    """Through the real `collect`, not just the tracker.

    The tracker being correct is worth nothing if nothing consults it — the
    same class of gap as a setting wired to no code, which is why
    tests/test_wiring.py exists.
    """
    from decimal import Decimal

    from app.providers.base import Completion
    from eval import run_eval as ev
    from eval.dataset import DATASET

    monkeypatch.setattr(ev, "RUNS_DIR", tmp_path)

    class FakeOllama:
        def __init__(self, client): pass
        async def chat(self, **kw):
            return Completion(content="a", model="m", tokens_in=1, tokens_out=1,
                              logprobs=[{"logprob": -0.2}])

    class Pricey:
        def __init__(self, reply): self.reply, self.calls = reply, 0
        async def chat(self, **kw):
            self.calls += 1
            # 1M output tokens per call: one call blows a $1 ceiling.
            return Completion(content=self.reply, model="m", tokens_in=0,
                              tokens_out=1_000_000, raw={"usage": {}})

    monkeypatch.setattr(ev, "OllamaProvider", FakeOllama)
    frontier, judge = Pricey("ref"), Pricey("ACCEPTABLE")
    judged = [p for p in DATASET if p.grade != "auto"][:4]
    spend = ev.SpendTracker(Decimal("1.00"), {
        "reference": (Decimal("0"), Decimal("3.00")),
        "judge": (Decimal("0"), Decimal("2.50"))})

    with pytest.raises(ev.SpendCeilingReached):
        await ev.collect(judged, tag="t", small_model="s", large_model="l",
                         max_tokens=512, use_real_frontier=True, frontier=frontier,
                         judge=judge, judge_model="j", frontier_model="f", spend=spend)

    assert frontier.calls < len(judged), "it kept spending after the ceiling bound"
    # And whatever was collected before it bound is on disk, not discarded.
    assert ev.load_cached("t"), "progress was not checkpointed"
