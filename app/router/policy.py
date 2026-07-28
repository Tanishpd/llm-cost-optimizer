"""Entry-tier policy — features in, `RouteDecision` out.

Returns a dataclass, never a bare string (CLAUDE.md conventions), and always
carries a **human-readable reason**. That string is a product requirement, not
a debug aid: it appears in `x_hlr.router.reason` and on the dashboard, and it is
how anyone judges whether the routing was sensible.

The asymmetry that shapes everything here (DECISIONS.md D10):

    too-low entry  -> costs latency; the gates still catch a weak answer
    too-high entry -> costs money, and nothing gates the frontier

So the score is compared against two thresholds, but frontier entry additionally
requires **two independent hardness signals**. Length alone can never send a
prompt to the frontier — summarization is long *and* cheap, and it is this
project's canonical cheap workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.router.classifier import Classifier
from app.router.heuristics import PromptFeatures, extract_features

EntryTier = Literal["local_small", "local_large", "frontier"]

# Score thresholds. Tuned by hand for M3; M6 replaces them with measurements.
SCORE_LARGE = 0.35     # above this, skip the 0.5B
SCORE_FRONTIER = 0.75  # above this AND >=2 hardness signals, skip both local tiers

# Feature weights. Positive pushes toward a more capable tier.
W_HARD_VERB = 0.35
W_REASONING_MARKER = 0.18   # per marker, capped
W_CODE_FENCE = 0.15
W_ARITHMETIC = 0.10
W_MULTI_TURN = 0.12
W_LONG_PROMPT = 0.20        # applied on a curve, see _length_score
W_MANY_QUESTIONS = 0.08
W_SIMPLE_VERB = -0.45       # strong pull downward: transformation tasks are cheap
W_CONTEXT_HEAVY = -0.20     # mostly supplied material -> transformation, not reasoning


@dataclass(frozen=True)
class RouteDecision:
    """The pre-router's output."""

    entry_tier: EntryTier
    score: float
    reason: str
    features: PromptFeatures
    signals: list[str] = field(default_factory=list)
    classifier: str = "heuristic-v1"

    def as_dict(self) -> dict:
        return {
            "entry_tier": self.entry_tier,
            "score": round(self.score, 4),
            "reason": self.reason,
            "features": self.features.as_dict(),
            "signals": self.signals,
            "classifier": self.classifier,
        }


def _length_score(n_tokens: int, small_max: int, escalate_min: int) -> float:
    """Length contribution, on a curve rather than a cliff.

    Below `small_max` it contributes nothing; it ramps to 1.0 at
    `escalate_min` and saturates there. A cliff would make a one-token
    difference flip the tier, which is both arbitrary and hard to explain in a
    `reason` string.
    """
    if n_tokens <= small_max:
        return 0.0
    if n_tokens >= escalate_min:
        return 1.0
    return (n_tokens - small_max) / max(escalate_min - small_max, 1)


def score_features(
    features: PromptFeatures, *, small_max: int = 60, escalate_min: int = 600
) -> tuple[float, list[str]]:
    """Weighted difficulty score in [0, 1], plus the signals that produced it.

    The signal list is what makes the reason string honest — it names the actual
    contributors rather than restating the number.
    """
    score = 0.0
    signals: list[str] = []

    if features.imperative_class == "hard":
        score += W_HARD_VERB
        signals.append("hard-verb opening")
    elif features.imperative_class == "simple":
        score += W_SIMPLE_VERB
        signals.append("transformation verb (cheap)")

    if features.reasoning_markers:
        capped = min(features.reasoning_markers, 3)
        score += W_REASONING_MARKER * capped
        signals.append(f"{features.reasoning_markers} reasoning marker(s)")

    if features.has_code_fence:
        score += W_CODE_FENCE
        signals.append("code block")

    if features.has_arithmetic:
        score += W_ARITHMETIC
        signals.append("arithmetic")

    if features.is_multi_turn:
        score += W_MULTI_TURN
        signals.append("multi-turn context")

    if features.question_count > 2:
        score += W_MANY_QUESTIONS
        signals.append(f"{features.question_count} questions")

    length = _length_score(features.n_tokens, small_max, escalate_min)
    if length > 0:
        score += W_LONG_PROMPT * length
        signals.append(f"~{features.n_tokens} tokens")

    # A prompt that is mostly supplied material is usually a transformation
    # task: long, but the answer is largely in the prompt.
    if features.n_chars > 0 and features.context_chars / features.n_chars > 0.5:
        score += W_CONTEXT_HEAVY
        signals.append("mostly supplied context")

    return max(0.0, min(1.0, score)), signals


def count_hardness_signals(features: PromptFeatures) -> int:
    """Independent indicators that this genuinely needs a frontier model.

    Length is **not** among them, on purpose. This is the guard that stops the
    one ungated path being reached on a single weak signal.
    """
    return sum(
        (
            features.imperative_class == "hard",
            features.reasoning_markers >= 2,
            features.has_code_fence,
            features.question_count > 2,
            features.is_multi_turn and features.n_tokens > 200,
        )
    )


def decide(
    prompt: str,
    *,
    system_prompt: str = "",
    is_multi_turn: bool = False,
    small_max_tokens: int = 60,
    escalate_min_tokens: int = 600,
    classifier: Classifier | None = None,
) -> RouteDecision:
    """Choose where to enter the cascade. No model is called.

    A `classifier` (M6+) overrides the heuristic when it is confident; below
    0.6 confidence the heuristic wins, because an unsure learned model is worse
    than a legible rule.
    """
    features = extract_features(
        prompt, system_prompt=system_prompt, is_multi_turn=is_multi_turn
    )
    score, signals = score_features(
        features, small_max=small_max_tokens, escalate_min=escalate_min_tokens
    )

    if classifier is not None:
        prediction = classifier.predict(features, prompt)
        if prediction.confidence >= 0.6:
            return RouteDecision(
                entry_tier=prediction.entry_tier,  # type: ignore[arg-type]
                score=prediction.score,
                reason=prediction.reason,
                features=features,
                signals=signals,
                classifier=classifier.name,
            )

    hardness = count_hardness_signals(features)

    if score >= SCORE_FRONTIER and hardness >= 2:
        tier: EntryTier = "frontier"
        reason = (
            f"entering at frontier: score {score:.2f} with {hardness} independent "
            f"hardness signals ({', '.join(signals)}); skipping both local tiers"
        )
    elif score >= SCORE_LARGE:
        tier = "local_large"
        why = ", ".join(signals) if signals else "no strong cheap-task signals"
        reason = f"entering at local_large: score {score:.2f} ({why})"
        if score >= SCORE_FRONTIER and hardness < 2:
            # The guard firing is worth saying out loud — it is the difference
            # between a considered decision and an accident.
            reason += (
                f"; score alone would reach frontier but only {hardness} hardness "
                "signal(s), and length never escalates on its own"
            )
    else:
        tier = "local_small"
        why = ", ".join(signals) if signals else f"short prompt, ~{features.n_tokens} tokens"
        reason = f"entering at local_small: score {score:.2f} ({why})"

    return RouteDecision(
        entry_tier=tier, score=score, reason=reason, features=features, signals=signals
    )
