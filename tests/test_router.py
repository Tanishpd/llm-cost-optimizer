"""Tests for the pre-router.

The load-bearing property is the *asymmetry*: entering too low is cheap
(the gates catch it), entering too high spends money with no gate above it.
Most of these tests are about the frontier guard.
"""

from __future__ import annotations

import pytest

from app.router.classifier import Classifier, Prediction
from app.router.heuristics import (
    classify_imperative,
    estimate_context_chars,
    estimate_tokens,
    extract_features,
)
from app.router.policy import (
    SCORE_FRONTIER,
    RouteDecision,
    count_hardness_signals,
    decide,
    score_features,
)

# --------------------------------------------------------------------------
# token estimation
# --------------------------------------------------------------------------


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") >= 1
    assert estimate_tokens("word " * 100) > estimate_tokens("word " * 10)


def test_estimate_tokens_handles_dense_text():
    """Code and CJK have few words but many tokens — the char floor catches
    what the word factor would under-count."""
    dense = "x" * 400  # one "word", 400 chars
    assert estimate_tokens(dense) >= 100


def test_estimate_tokens_biases_high():
    """Under-estimating makes long prompts look cheap, which is the direction
    that costs quality."""
    text = " ".join(["word"] * 100)  # 100 words
    assert estimate_tokens(text) >= 100


# --------------------------------------------------------------------------
# imperative classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prompt", [
    "Summarize this article.",
    "Translate to French: hello",
    "Rewrite the following more concisely",
    "Extract the dates from this text",
])
def test_transformation_verbs_classify_simple(prompt):
    assert classify_imperative(prompt) == "simple"


@pytest.mark.parametrize("prompt", [
    "Prove that every finite integral domain is a field.",
    "Design a rate limiter for a distributed system",
    "Debug this race condition",
    "Compare eventual and strong consistency",
])
def test_reasoning_verbs_classify_hard(prompt):
    assert classify_imperative(prompt) == "hard"


def test_only_the_opening_is_inspected():
    """"Summarize this article about proving theorems" is a summarization task.
    Scanning the whole prompt would misread it as a proof."""
    assert classify_imperative("Summarize this article about proving theorems") == "simple"


def test_bare_question_classifies_as_question():
    assert classify_imperative("What is the capital of France?") == "question"


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------


def test_code_fence_detected():
    assert extract_features("Fix this:\n```py\nx=1\n```").has_code_fence is True
    assert extract_features("no code here").has_code_fence is False


def test_reasoning_markers_counted():
    f = extract_features("Explain why this works, step by step, with trade-offs.")
    assert f.reasoning_markers >= 3


def test_arithmetic_detected():
    assert extract_features("What is 45 * 92?").has_arithmetic is True
    assert extract_features("Convert 50 percent to a fraction").has_arithmetic is True
    assert extract_features("Tell me about history").has_arithmetic is False


def test_entity_density():
    dense = extract_features("Compare Paris and Berlin and Madrid and Lisbon")
    sparse = extract_features("compare the cities we discussed")
    assert dense.entity_density > sparse.entity_density


def test_context_chars_counts_supplied_material():
    prompt = "Summarize:\n```\n" + ("data line\n" * 50) + "```"
    f = extract_features(prompt)
    assert f.context_chars > 0
    assert f.context_chars / f.n_chars > 0.5


def test_quoted_blocks_count_as_context():
    prompt = "Summarize this:\n" + "\n".join("> a quoted line of some length" for _ in range(10))
    assert estimate_context_chars(prompt) > 0


def test_features_serialize_for_x_hlr():
    d = extract_features("q").as_dict()
    for key in (
        "n_tokens", "n_chars", "has_code_fence", "imperative_class",
        "reasoning_markers", "has_arithmetic", "question_count",
        "entity_density", "context_chars", "is_multi_turn",
    ):
        assert key in d


# --------------------------------------------------------------------------
# routing decisions
# --------------------------------------------------------------------------


def test_short_factual_enters_at_small():
    d = decide("What is the capital of France?")
    assert d.entry_tier == "local_small"


def test_trivial_arithmetic_enters_at_small():
    assert decide("What is 2+2?").entry_tier == "local_small"


def test_reasoning_prompt_enters_at_large():
    d = decide("Explain why a rainbow has the colour order it does, step by step.")
    assert d.entry_tier in ("local_large", "frontier")
    assert d.score > 0


def test_hard_multi_signal_prompt_reaches_frontier():
    d = decide(
        "Prove that every finite integral domain is a field. Walk me through the "
        "reasoning step by step, explain why each lemma holds, and discuss the "
        "trade-offs of the alternative proofs. What if the ring were infinite? "
        "How would you generalize?\n```\nR = Z/pZ\n```"
    )
    assert d.entry_tier == "frontier"
    assert count_hardness_signals(d.features) >= 2


def test_decision_is_a_dataclass_not_a_string():
    """CLAUDE.md conventions."""
    assert isinstance(decide("q"), RouteDecision)


def test_every_decision_carries_a_human_readable_reason():
    """Invariant 2: prose, not an enum. It reaches the dashboard."""
    for prompt in ("What is 2+2?", "Summarize this.", "Prove Fermat's little theorem"):
        reason = decide(prompt).reason
        assert " " in reason and len(reason) > 20
        assert reason not in {"local_small", "local_large", "frontier"}


def test_reason_names_the_actual_signals():
    d = decide("Explain why this fails, step by step.\n```\ncode\n```")
    assert "reasoning marker" in d.reason or "code block" in d.reason


# --------------------------------------------------------------------------
# the frontier guard — the ungated path
# --------------------------------------------------------------------------


def test_length_alone_never_reaches_frontier():
    """The canonical cheap workload: summarization is long AND cheap. Sending
    it to the frontier on length would be the expensive failure mode."""
    long_summarization = "Summarize the following article.\n\n" + ("some prose. " * 900)
    d = decide(long_summarization)
    assert d.entry_tier != "frontier"
    assert d.features.n_tokens > 600  # genuinely long


def test_very_long_context_heavy_prompt_stays_local():
    prompt = "Summarize:\n```\n" + ("log line here\n" * 800) + "```"
    assert decide(prompt).entry_tier != "frontier"


def test_high_score_with_one_signal_is_held_back():
    """The guard: score can clear the frontier bar, but one hardness signal is
    not enough to reach an ungated tier."""
    features = extract_features("Prove it. " + ("padding words " * 500))
    score, _ = score_features(features)
    if score >= SCORE_FRONTIER and count_hardness_signals(features) < 2:
        assert decide("Prove it. " + ("padding words " * 500)).entry_tier != "frontier"


def test_guard_firing_is_explained_in_the_reason():
    """When the guard holds a prompt back, the reason should say so — the
    difference between a considered decision and an accident."""
    prompt = "Derive it.\n" + ("filler text " * 700)
    d = decide(prompt)
    if d.entry_tier == "local_large" and d.score >= SCORE_FRONTIER:
        assert "hardness signal" in d.reason


def test_length_is_not_a_hardness_signal():
    short_hard = extract_features("Prove this. Explain why. Compare approaches. Trade-offs?")
    long_easy = extract_features("Summarize. " + ("word " * 2000))
    assert count_hardness_signals(short_hard) > count_hardness_signals(long_easy)


# --------------------------------------------------------------------------
# scoring behaviour
# --------------------------------------------------------------------------


def test_score_is_bounded():
    for prompt in ("", "q", "Prove " * 200, "Summarize " * 200):
        score, _ = score_features(extract_features(prompt))
        assert 0.0 <= score <= 1.0


def test_transformation_verb_pulls_score_down():
    """The discount is only observable above the floor: the score clamps at 0,
    and a prompt with no positive signals is already there. Uses a prompt with
    real hardness signals so the pull can be seen."""
    body = " Explain why it happens, step by step, with trade-offs.\n```\nx = 1\n```"
    plain, _ = score_features(extract_features("Analyze inflation." + body))
    simple, _ = score_features(extract_features("Summarize inflation." + body))
    assert plain > 0  # guard: the comparison is meaningful
    assert simple < plain


def test_transformation_verb_can_reach_the_floor():
    """A cheap prompt with no hardness signals scores 0 either way — both route
    to local_small, which is the correct outcome even though the discount is
    invisible in the number."""
    simple, _ = score_features(extract_features("Summarize the causes of inflation"))
    assert simple == 0.0
    assert decide("Summarize the causes of inflation").entry_tier == "local_small"


def test_length_contributes_on_a_curve_not_a_cliff():
    """A one-token difference must not flip the tier — that is arbitrary and
    impossible to explain in a reason string."""
    a, _ = score_features(extract_features("word " * 100))
    b, _ = score_features(extract_features("word " * 101))
    assert abs(a - b) < 0.05


def test_signals_are_reported():
    _, signals = score_features(extract_features("Prove this step by step.\n```\nx\n```"))
    assert signals
    assert any("code" in s for s in signals)


# --------------------------------------------------------------------------
# classifier seam
# --------------------------------------------------------------------------


class StubClassifier:
    name = "stub-v1"

    def __init__(self, tier="frontier", confidence=0.9):
        self._tier, self._confidence = tier, confidence

    def predict(self, features, prompt):
        return Prediction(
            entry_tier=self._tier, score=0.99, confidence=self._confidence,
            reason=f"stub says {self._tier}",
        )


def test_stub_satisfies_the_protocol():
    assert isinstance(StubClassifier(), Classifier)


def test_confident_classifier_overrides_the_heuristic():
    d = decide("What is 2+2?", classifier=StubClassifier(tier="frontier", confidence=0.9))
    assert d.entry_tier == "frontier"
    assert d.classifier == "stub-v1"


def test_unsure_classifier_defers_to_the_heuristic():
    """An unsure learned model is worse than a legible rule."""
    d = decide("What is 2+2?", classifier=StubClassifier(tier="frontier", confidence=0.3))
    assert d.entry_tier == "local_small"
    assert d.classifier == "heuristic-v1"


# --------------------------------------------------------------------------
# configurability
# --------------------------------------------------------------------------


def test_thresholds_are_configurable():
    prompt = "Explain this concept in some reasonable detail please. " * 3
    tight = decide(prompt, small_max_tokens=5, escalate_min_tokens=50)
    loose = decide(prompt, small_max_tokens=500, escalate_min_tokens=5000)
    assert tight.score >= loose.score


def test_multi_turn_raises_the_score():
    single, _ = score_features(extract_features("What about it?", is_multi_turn=False))
    multi, _ = score_features(extract_features("What about it?", is_multi_turn=True))
    assert multi > single
