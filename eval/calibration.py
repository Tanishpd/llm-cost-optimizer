"""Fitting and — more importantly — *validating* the confidence calibrators.

Fitting a model does not make it calibrated. A logistic regression will happily
produce numbers in [0,1] that have no relationship to observed frequency, and
calling those "probabilities" is how an 85% threshold becomes decoration.

So this module's centre of gravity is the validation half: reliability computed
on **pooled out-of-fold predictions**, an ECE with a bootstrap confidence
interval, and a hard pass/fail rule. `should_flip()` returns False by default,
and staying in `raw` mode is a legitimate — and honest — outcome.

numpy only. scikit-learn would do the logistic fit in one line, but at N≈60–120
a hand-rolled IRLS is a few lines, keeps the eval runnable without the extra
~180 MB of dependency, and makes the regularization explicit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FEATURE_NAMES = (
    "geo_mean", "min_prob", "p10_prob", "mean_margin",
    "mean_norm_entropy", "frac_low", "n_tokens",
)

# The flip rule (docs/PLAN.md M6). Deliberately quantitative: "looks close to
# the diagonal" is not a criterion at this sample size.
MAX_ECE = 0.10
MAX_ECE_CI_UPPER = 0.20


@dataclass
class Calibrator:
    """Logistic map from the feature vector to P(answer is acceptable)."""

    weights: list[float]
    bias: float
    feature_names: tuple[str, ...] = FEATURE_NAMES
    # Feature standardization, stored so inference reproduces training exactly.
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    fitted_temperature: float = 0.0
    n_train: int = 0

    def predict_proba(self, features: list[float]) -> float:
        x = np.asarray(features, dtype=float)
        if self.mean and self.scale:
            x = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        z = float(np.dot(self.weights, x) + self.bias)
        # Numerically stable sigmoid.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def predict_many(self, matrix: np.ndarray) -> np.ndarray:
        return np.array([self.predict_proba(list(row)) for row in matrix])


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean, unit-scale. A constant feature gets scale 1 rather than 0 —
    dividing by its zero standard deviation would produce NaNs that silently
    poison every prediction."""
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    return (X - mean) / scale, mean, scale


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, iters: int = 200
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by IRLS/Newton steps.

    The L2 term is not optional at this sample size: with ~60 rows and 7
    features, an unregularized fit separates the data and drives weights to
    infinity, producing a model that outputs only 0.0 and 1.0 — perfectly
    confident and completely uncalibrated.
    """
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])  # bias as a column
    w = np.zeros(d + 1)
    # Do not penalize the bias — it should be free to match the base rate.
    penalty = np.eye(d + 1) * l2
    penalty[-1, -1] = 0.0

    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        # Floor the variance so a saturated fit cannot make the Hessian singular.
        W = np.clip(p * (1 - p), 1e-6, None)
        gradient = Xb.T @ (y - p) - penalty @ w
        hessian = -(Xb.T * W) @ Xb - penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        w_new = w - step
        if not np.all(np.isfinite(w_new)) or np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new if np.all(np.isfinite(w_new)) else w
            break
        w = w_new

    return w[:-1], float(w[-1])


def fit_calibrator(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, temperature: float = 0.0
) -> Calibrator:
    Xs, mean, scale = _standardize(X)
    weights, bias = fit_logistic(Xs, y, l2=l2)
    return Calibrator(
        weights=[float(v) for v in weights],
        bias=bias,
        mean=[float(v) for v in mean],
        scale=[float(v) for v in scale],
        fitted_temperature=temperature,
        n_train=len(y),
    )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    return float(np.mean((probs - labels) ** 2))


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 5
) -> list[dict[str, float]]:
    """Equal-**count** bins, not equal-width.

    Equal-width bins leave most bins nearly empty when predictions cluster,
    and an empty bin contributes noise indistinguishable from miscalibration.
    """
    n = len(probs)
    if n == 0:
        return []
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(probs)
    out = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        out.append({
            "count": int(len(chunk)),
            "mean_predicted": float(np.mean(probs[chunk])),
            "observed_rate": float(np.mean(labels[chunk])),
            "gap": float(abs(np.mean(probs[chunk]) - np.mean(labels[chunk]))),
        })
    return out


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 5
) -> float:
    """Count-weighted mean gap between predicted and observed frequency."""
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    if not bins:
        return 1.0
    total = sum(b["count"] for b in bins)
    return float(sum(b["gap"] * b["count"] for b in bins) / total)


def bootstrap_ece_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 5,
    resamples: int = 1000,
    seed: int = 1337,
) -> tuple[float, float]:
    """95% percentile CI for ECE.

    The point estimate alone is not decidable at N≈60: a "good" ECE can easily
    be noise. The interval is what the flip rule keys on.

    Note the interval does **not** bracket the point estimate. ECE is biased
    upward under resampling — duplicated points look miscalibrated — so the
    bootstrap distribution sits above it. That bias is in the useful direction
    here: a pessimistic upper bound makes `calibrated` mode harder to justify,
    never easier.
    """
    if len(probs) == 0:
        return (1.0, 1.0)
    rng = np.random.default_rng(seed)
    n = len(probs)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        values.append(expected_calibration_error(probs[idx], labels[idx], n_bins=n_bins))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def cross_val_predictions(
    X: np.ndarray, y: np.ndarray, *, folds: int = 5, l2: float = 1.0, seed: int = 1337
) -> np.ndarray:
    """Out-of-fold predictions for every row.

    Reliability is computed on these **pooled** predictions rather than
    per-fold: at this sample size a single fold holds ~12 rows, which is far
    too few to bin meaningfully.
    """
    n = len(y)
    if n < folds * 2:
        folds = max(2, n // 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    oof = np.zeros(n)
    for fold in np.array_split(order, folds):
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        if len(np.unique(y[mask])) < 2:
            # A fold whose training half is single-class cannot fit anything;
            # fall back to the base rate rather than fabricating a model.
            oof[fold] = float(np.mean(y[mask])) if mask.any() else 0.5
            continue
        cal = fit_calibrator(X[mask], y[mask], l2=l2)
        oof[fold] = cal.predict_many(X[fold])
    return oof


@dataclass
class ValidationReport:
    """Everything needed to decide whether the calibrated mode is trustworthy."""

    gate: str
    n: int
    positive_rate: float
    ece: float
    ece_ci_low: float
    ece_ci_high: float
    brier: float
    brier_baseline: float
    bins: list[dict[str, float]]
    passed: bool
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def validate(gate: str, X: np.ndarray, y: np.ndarray, *, folds: int = 5,
    l2: float = 1.0) -> ValidationReport:
    """Fit under cross-validation and judge the result honestly."""
    n = len(y)
    if n < 10 or len(np.unique(y)) < 2:
        return ValidationReport(
            gate=gate, n=n, positive_rate=float(np.mean(y)) if n else 0.0,
            ece=1.0, ece_ci_low=1.0, ece_ci_high=1.0, brier=1.0, brier_baseline=1.0,
            bins=[], passed=False,
            verdict=f"insufficient data: n={n}, classes={len(np.unique(y)) if n else 0}",
        )

    oof = cross_val_predictions(X, y, folds=folds, l2=l2)
    ece = expected_calibration_error(oof, y)
    lo, hi = bootstrap_ece_ci(oof, y)
    brier = brier_score(oof, y)
    # The honest comparison: does the model beat always predicting the base rate?
    base_rate = float(np.mean(y))
    brier_base = brier_score(np.full_like(oof, base_rate), y)

    passed = ece <= MAX_ECE and hi < MAX_ECE_CI_UPPER and brier < brier_base
    reasons = []
    if ece > MAX_ECE:
        reasons.append(f"ECE {ece:.3f} > {MAX_ECE}")
    if hi >= MAX_ECE_CI_UPPER:
        reasons.append(f"ECE 95% CI upper {hi:.3f} >= {MAX_ECE_CI_UPPER}")
    if brier >= brier_base:
        reasons.append(f"Brier {brier:.3f} no better than base rate {brier_base:.3f}")
    verdict = "calibration is trustworthy" if passed else "; ".join(reasons)

    return ValidationReport(
        gate=gate, n=n, positive_rate=base_rate, ece=ece, ece_ci_low=lo, ece_ci_high=hi,
        brier=brier, brier_baseline=brier_base,
        bins=reliability_bins(oof, y), passed=passed, verdict=verdict,
    )


def should_flip(reports: dict[str, ValidationReport]) -> bool:
    """Is `CONFIDENCE_MODE=calibrated` justified?

    Every gate must pass. One trustworthy gate and one untrustworthy gate is
    not a calibrated system, and shipping it would put a measured number and a
    guess behind the same 0.85 label.
    """
    return bool(reports) and all(r.passed for r in reports.values())


def save(
    path: Path,
    calibrators: dict[str, Calibrator],
    reports: dict[str, ValidationReport],
    provenance: dict[str, object] | None = None,
) -> None:
    """Persist fits *and* their validation, so nobody can load a calibrator
    without also being able to see whether it earned its place.

    `provenance` records *what was calibrated*. A fit is only meaningful for
    the models and answer length it was fitted on: these weights map
    qwen2.5:0.5b's logprob statistics onto P(acceptable), and applying them to
    a different model yields a number in [0, 1] that looks exactly like a
    probability and is not one. Nothing downstream can detect that, which is
    why the gateway refuses to start on a mismatch rather than warning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "calibrators": {g: asdict(c) for g, c in calibrators.items()},
        "validation": {g: r.as_dict() for g, r in reports.items()},
        "flip_recommended": should_flip(reports),
        "thresholds": {"max_ece": MAX_ECE, "max_ece_ci_upper": MAX_ECE_CI_UPPER},
        "provenance": provenance or {},
    }, indent=2))


def load(path: Path) -> dict[str, Calibrator]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for gate, raw in (data.get("calibrators") or {}).items():
        raw.pop("feature_names", None)
        out[gate] = Calibrator(**raw, feature_names=FEATURE_NAMES)
    return out
