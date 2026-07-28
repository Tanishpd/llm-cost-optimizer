"""Confidence scoring from token log-probabilities.

Pure functions over Ollama's ``logprobs`` array — no I/O, no HTTP, no Redis.
That is deliberate: this is the module the whole cascade's correctness rests on,
so it is the one that must be exhaustively unit-testable in isolation.

The central point, derived in docs/RESEARCH.md §3: **do not use sequence
probability** (``∏ p_i``). It decays exponentially with length — a 100-token
answer at a healthy p=0.9 per token scores 2.7e-5, so any fixed threshold
rejects everything longer than a few tokens. It measures length, not certainty.

Use :func:`geo_mean` instead — the geometric mean of token probabilities,
equivalently 1/perplexity — which is length-invariant.

Ollama returns natural-log probabilities. See docs/RESEARCH.md §2 for the
verified response shape.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

# A token whose probability is below this counts as "low" for `frac_low`.
# 0.5 is the natural cut: the model was less than even-odds on this token.
LOW_PROB = 0.5

# The fixed feature order every calibrator is fitted against. It must match
# `ConfidenceMetrics.feature_vector()` and `eval/calibration.FEATURE_NAMES`;
# `calibrators_from_dict` refuses to load a fit that disagrees, because a
# reordered vector produces plausible-looking nonsense rather than an error.
FEATURE_NAMES: tuple[str, ...] = (
    "geo_mean",
    "min_prob",
    "p10_prob",
    "mean_margin",
    "mean_norm_entropy",
    "frac_low",
    "n_tokens",
)

# Phrases that mean "escalate regardless of score". A model can be fluently,
# confidently uncertain — high geo_mean while explicitly refusing — so these
# are checked independently of the metrics. Cheap and high-precision.
HEDGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi (?:do not|don't) know\b",
        r"\bi(?:'m| am) not sure\b",
        r"\bi(?:'m| am) unable to\b",
        r"\bas an ai\b",
        r"\bi cannot (?:provide|answer|determine)\b",
        r"\bcan(?:no|')t (?:provide|answer|determine)\b",
        r"\binsufficient information\b",
        r"\bnot enough (?:information|context)\b",
        r"\bunclear from the (?:context|question)\b",
    )
)

# Below this many tokens, a non-trivial answer is suspiciously terse.
MIN_ANSWER_TOKENS = 3


@dataclass(frozen=True)
class ConfidenceMetrics:
    """The feature vector, plus the raw headline score.

    Field order here defines the calibrator's feature order — see
    :meth:`feature_vector`. Do not reorder without refitting
    ``eval/calibrator.json``.
    """

    geo_mean: float
    min_prob: float
    p10_prob: float
    mean_margin: float
    mean_norm_entropy: float
    frac_low: float
    n_tokens: int

    def feature_vector(self) -> list[float]:
        """The 7 features, in the fixed order the calibrators are fitted on."""
        return [
            self.geo_mean,
            self.min_prob,
            self.p10_prob,
            self.mean_margin,
            self.mean_norm_entropy,
            self.frac_low,
            float(self.n_tokens),
        ]

    def as_dict(self) -> dict[str, float | int]:
        """Shape emitted in ``x_hlr.gates[].metrics`` (see docs/API.md)."""
        return {
            "geo_mean": round(self.geo_mean, 6),
            "min_prob": round(self.min_prob, 6),
            "p10_prob": round(self.p10_prob, 6),
            "mean_margin": round(self.mean_margin, 6),
            "mean_norm_entropy": round(self.mean_norm_entropy, 6),
            "frac_low": round(self.frac_low, 6),
            "n_tokens": self.n_tokens,
        }


@dataclass(frozen=True)
class GateVerdict:
    """One gate's decision, as it appears in ``x_hlr.gates[]``."""

    gate: str
    confidence: float
    threshold: float
    verdict: Literal["accept", "escalate"]
    metrics: ConfidenceMetrics
    reason: str
    hedged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "confidence": round(self.confidence, 6),
            "threshold": self.threshold,
            "verdict": self.verdict,
            "reason": self.reason,
            "metrics": self.metrics.as_dict(),
        }


EMPTY_METRICS = ConfidenceMetrics(
    geo_mean=0.0,
    min_prob=0.0,
    p10_prob=0.0,
    mean_margin=0.0,
    mean_norm_entropy=1.0,
    frac_low=1.0,
    n_tokens=0,
)


def geo_mean(logprobs: list[float]) -> float:
    """Geometric mean of token probabilities = ``exp(mean(logprob))`` = 1/perplexity.

    Length-invariant, unlike ``∏ p_i``. Range (0, 1].

    >>> round(geo_mean([math.log(0.9)] * 100), 4)
    0.9
    """
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs) / len(logprobs))


def sequence_probability(logprobs: list[float]) -> float:
    """``∏ p_i``. **Not** a usable confidence metric — provided so tests can
    demonstrate the length-decay problem this module exists to avoid."""
    if not logprobs:
        return 0.0
    return math.exp(sum(logprobs))


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in [0, 1].

    numpy would do this, but keeping this module dependency-free means the
    confidence math can be unit-tested with nothing installed.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def normalized_entropy(top_logprobs: list[float]) -> float:
    """Entropy of the top-k distribution, renormalized and scaled to [0, 1].

    Ollama returns only the top ``k`` alternatives (k = ``TOP_LOGPROBS``), not the
    full vocabulary, so this is entropy over the *renormalized* top-k — a lower
    bound on true entropy, but a consistent one, which is what a calibrator needs.

    0.0 = all mass on one token. 1.0 = uniform across the k alternatives.
    """
    if len(top_logprobs) < 2:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    probs = [p / total for p in probs]
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return h / math.log(len(probs))


def extract_metrics(logprobs: list[dict[str, Any]]) -> ConfidenceMetrics:
    """Compute the feature vector from Ollama's ``logprobs`` array.

    Each entry is ``{"token": str, "logprob": float, "bytes": [int],
    "top_logprobs": [{"token", "logprob", "bytes"}, ...]}`` — verified shape,
    docs/RESEARCH.md §2.

    An empty array (logprobs unsupported or not requested) yields
    :data:`EMPTY_METRICS`, whose values force escalation rather than silently
    scoring 1.0.
    """
    if not logprobs:
        return EMPTY_METRICS

    sampled = [float(e["logprob"]) for e in logprobs]
    probs = [math.exp(lp) for lp in sampled]

    margins: list[float] = []
    entropies: list[float] = []
    for entry in logprobs:
        top = entry.get("top_logprobs") or []
        if len(top) >= 2:
            margins.append(float(top[0]["logprob"]) - float(top[1]["logprob"]))
        if top:
            entropies.append(normalized_entropy([float(t["logprob"]) for t in top]))

    return ConfidenceMetrics(
        geo_mean=geo_mean(sampled),
        min_prob=min(probs),
        p10_prob=percentile(probs, 0.10),
        mean_margin=(sum(margins) / len(margins)) if margins else 0.0,
        mean_norm_entropy=(sum(entropies) / len(entropies)) if entropies else 0.0,
        frac_low=sum(1 for p in probs if p < LOW_PROB) / len(probs),
        n_tokens=len(sampled),
    )


def find_hedge(text: str) -> str | None:
    """Return the hedge phrase found in ``text``, or None.

    Independent of the metrics: a model can be confidently fluent while
    explicitly refusing, and logprobs will not catch that.
    """
    for pattern in HEDGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


class Calibrator(Protocol):
    """Maps a feature vector to P(this tier's answer is as good as the frontier's).

    Fitted per gate in M6 — gate ``small`` and gate ``large`` are different
    decisions and must not share parameters (docs/DECISIONS.md D5).
    """

    def predict_proba(self, features: list[float]) -> float: ...


@dataclass(frozen=True)
class LogisticCalibrator:
    """Logistic map ``sigmoid(w·x + b)``, fitted offline by ``eval/run_eval.py``.

    Logistic rather than isotonic: at N≈120 labels isotonic overfits badly
    (docs/RESEARCH.md §3).
    """

    weights: list[float]
    bias: float
    # The fit standardizes features first (n_tokens spans 1..600, geo_mean
    # 0..1 — un-standardized, the fit is dominated by whichever feature has
    # the largest units). The weights are therefore in standardized space and
    # are MEANINGLESS applied to raw features, so these travel with them.
    # Empty means "already raw", which is only true of a hand-built calibrator.
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    # Recorded so a request at a different temperature can be flagged as
    # out-of-calibration rather than silently scored — logprob distributions
    # shift with temperature.
    fitted_temperature: float = 0.0
    n_train: int = 0
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict_proba(self, features: list[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError(
                f"calibrator expects {len(self.weights)} features, got {len(features)}"
            )
        x = features
        if self.mean and self.scale:
            if len(self.mean) != len(x) or len(self.scale) != len(x):
                raise ValueError("calibrator mean/scale do not match the feature vector")
            x = [(v - m) / s for v, m, s in zip(x, self.mean, self.scale, strict=True)]
        z = sum(w * v for w, v in zip(self.weights, x, strict=True)) + self.bias
        # Numerically stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


def calibrators_from_dict(data: dict[str, Any]) -> dict[str, LogisticCalibrator]:
    """Deserialize the contents of `eval/calibrator.json`, keyed by gate id.

    Takes the already-parsed dict rather than a path, so this module keeps its
    no-I/O property; `app/main.py` does the read in the lifespan.

    Raises rather than skipping on a malformed or mismatched fit. A calibrator
    silently treated as absent would send serving back to raw thresholds while
    `x_hlr.confidence_mode` kept reporting `calibrated` — the exact class of
    confidently-wrong metadata this project is organized against.
    """
    out: dict[str, LogisticCalibrator] = {}
    for gate, raw in (data.get("calibrators") or {}).items():
        names = tuple(raw.get("feature_names") or FEATURE_NAMES)
        if names != FEATURE_NAMES:
            raise ValueError(
                f"calibrator {gate!r} was fitted on {names}, but this build computes "
                f"{FEATURE_NAMES}. Refit with eval/run_eval.py."
            )
        weights = [float(w) for w in raw["weights"]]
        if len(weights) != len(FEATURE_NAMES):
            raise ValueError(
                f"calibrator {gate!r} has {len(weights)} weights for "
                f"{len(FEATURE_NAMES)} features"
            )
        out[gate] = LogisticCalibrator(
            weights=weights,
            bias=float(raw["bias"]),
            mean=[float(v) for v in raw.get("mean") or []],
            scale=[float(v) for v in raw.get("scale") or []],
            fitted_temperature=float(raw.get("fitted_temperature") or 0.0),
            n_train=int(raw.get("n_train") or 0),
        )
    return out


def score(
    metrics: ConfidenceMetrics,
    *,
    mode: Literal["raw", "calibrated"] = "raw",
    calibrator: Calibrator | None = None,
) -> float:
    """The single confidence number, per mode.

    ``raw`` returns ``geo_mean`` — honest, available immediately, and **not**
    comparable to the spec's 85%.

    ``calibrated`` returns P(answer is good enough), which is what 0.85 is meant
    to mean. Falls back to raw if no calibrator has been fitted yet, rather than
    failing — a gateway with no calibrator should still serve.
    """
    if mode == "calibrated" and calibrator is not None:
        return calibrator.predict_proba(metrics.feature_vector())
    return metrics.geo_mean


def evaluate_gate(
    gate: str,
    logprobs: list[dict[str, Any]],
    text: str,
    threshold: float,
    *,
    mode: Literal["raw", "calibrated"] = "raw",
    calibrator: Calibrator | None = None,
) -> GateVerdict:
    """Decide whether this tier's answer stands, or escalates to the next.

    Escalates when any of these hold:

    1. No logprobs came back — cannot judge, so do not gamble.
    2. A hedge phrase is present — overrides the score entirely.
    3. The answer is suspiciously short.
    4. ``score < threshold``.

    Every branch produces a human-readable ``reason``; that string reaches the
    caller in ``x_hlr`` and the dashboard, and is a product requirement rather
    than a debug aid (CLAUDE.md invariant 2).
    """
    metrics = extract_metrics(logprobs)

    if not logprobs:
        return GateVerdict(
            gate=gate,
            confidence=0.0,
            threshold=threshold,
            verdict="escalate",
            metrics=metrics,
            reason="no logprobs returned; cannot assess confidence, escalating",
        )

    hedge = find_hedge(text)
    if hedge is not None:
        conf = score(metrics, mode=mode, calibrator=calibrator)
        return GateVerdict(
            gate=gate,
            confidence=conf,
            threshold=threshold,
            verdict="escalate",
            metrics=metrics,
            reason=f"model hedged ({hedge!r}) despite confidence {conf:.2f}",
            hedged=True,
        )

    if metrics.n_tokens < MIN_ANSWER_TOKENS:
        conf = score(metrics, mode=mode, calibrator=calibrator)
        if conf < threshold:
            return GateVerdict(
                gate=gate,
                confidence=conf,
                threshold=threshold,
                verdict="escalate",
                metrics=metrics,
                reason=(
                    f"answer only {metrics.n_tokens} token(s) and "
                    f"confidence {conf:.2f} < {threshold:.2f}"
                ),
            )

    conf = score(metrics, mode=mode, calibrator=calibrator)
    if conf < threshold:
        return GateVerdict(
            gate=gate,
            confidence=conf,
            threshold=threshold,
            verdict="escalate",
            metrics=metrics,
            reason=(
                f"confidence {conf:.2f} below {threshold:.2f} "
                f"({metrics.frac_low:.0%} of tokens under {LOW_PROB})"
            ),
        )

    return GateVerdict(
        gate=gate,
        confidence=conf,
        threshold=threshold,
        verdict="accept",
        metrics=metrics,
        reason=f"confidence {conf:.2f} at or above {threshold:.2f}",
    )
