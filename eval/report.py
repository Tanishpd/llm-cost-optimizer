#!/usr/bin/env python3
"""Turn a completed eval run into a cost-vs-quality report.

    python -m eval.report              # the free auto-gradable run
    python -m eval.report --tag full   # a run that used a real API

Reads `eval/runs/<tag>.json` — never calls a model, never spends anything.

The headline is **not** escalation rate or cost saved. Those are easy to make
look good by escalating nothing. The metric that matters is the **false-cheap
rate**: how often a gate accepted an answer that was actually wrong. That is
the number a cheap-routing system can quietly get away with damaging.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.calibration import FEATURE_NAMES, cross_val_predictions, validate
from eval.run_eval import TIER_TO_GATE, build_matrix, load_cached

# Per-1M-token prices used for the cost columns. Placeholders — the same ones
# in .env.example — so the shape of the trade-off is right even though the
# absolute dollars are not a quote.
PRICE_IN, PRICE_OUT = 2.50, 10.00


@dataclass
class SweepRow:
    threshold: float
    accepted: int
    escalated: int
    escalation_rate: float
    # Accepted an answer that was actually wrong. The one that matters.
    false_cheap: int
    false_cheap_rate: float
    # Escalated an answer that was actually fine — wasted money, not quality.
    false_expensive: int
    false_expensive_rate: float
    quality_preserved: float


def sweep(scores: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[SweepRow]:
    """How each threshold trades cost against correctness."""
    rows = []
    n = len(labels)
    for t in thresholds:
        accept = scores >= t
        accepted = int(accept.sum())
        escalated = n - accepted
        false_cheap = int(((accept) & (labels == 0)).sum())
        false_expensive = int(((~accept) & (labels == 1)).sum())
        # Quality preserved: accepted-and-correct plus everything escalated,
        # since escalation is assumed to fix it.
        preserved = int(((accept) & (labels == 1)).sum()) + escalated
        rows.append(SweepRow(
            threshold=t,
            accepted=accepted,
            escalated=escalated,
            escalation_rate=round(escalated / n, 3) if n else 0.0,
            false_cheap=false_cheap,
            false_cheap_rate=round(false_cheap / n, 3) if n else 0.0,
            false_expensive=false_expensive,
            false_expensive_rate=round(false_expensive / n, 3) if n else 0.0,
            quality_preserved=round(preserved / n, 3) if n else 0.0,
        ))
    return rows


def raw_scores(results, gate: str) -> tuple[np.ndarray, np.ndarray]:
    """geo_mean and labels — what `raw` mode actually thresholds on."""
    tier = next(t for t, g in TIER_TO_GATE.items() if g == gate)
    scores, labels = [], []
    for r in results.values():
        tr = r.tiers.get(tier)
        if tr is None or tr.error or not tr.features or tier not in r.labels:
            continue
        scores.append(float(tr.features.get("geo_mean", 0.0)))
        labels.append(1.0 if r.labels[tier] else 0.0)
    return np.array(scores), np.array(labels)


def per_category(results, gate: str) -> dict[str, tuple[int, float]]:
    tier = next(t for t, g in TIER_TO_GATE.items() if g == gate)
    buckets: dict[str, list[float]] = {}
    for r in results.values():
        if tier in r.labels:
            buckets.setdefault(r.category, []).append(1.0 if r.labels[tier] else 0.0)
    return {c: (len(v), round(float(np.mean(v)), 3)) for c, v in sorted(buckets.items())}


def _table(rows: list[SweepRow], highlight: float | None = None) -> str:
    out = [
        "    thresh  escalate%  false-cheap  false-expensive  quality",
        "    ------  ---------  -----------  ---------------  -------",
    ]
    for r in rows:
        mark = " <-- shipped" if highlight is not None and abs(r.threshold - highlight) < 1e-9 else ""
        out.append(
            f"    {r.threshold:>6.2f}  {r.escalation_rate:>8.0%}  "
            f"{r.false_cheap:>4} ({r.false_cheap_rate:>4.0%})  "
            f"{r.false_expensive:>6} ({r.false_expensive_rate:>4.0%})  "
            f"{r.quality_preserved:>6.0%}{mark}"
        )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="auto")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    results = load_cached(args.tag)
    if not results:
        print(f"no run found at eval/runs/{args.tag}.json — run `python -m eval.run_eval` first")
        return 1

    modes = {src for r in results.values() for src in r.label_source.values()}
    shipped = {"small": 0.55, "large": 0.62}
    payload: dict = {"tag": args.tag, "prompts": len(results), "label_modes": sorted(modes)}

    if not args.json:
        print(f"\n=== eval report: {args.tag} ===")
        print(f"prompts: {len(results)}   labels: {', '.join(sorted(modes)) or 'none'}")
        if modes == {"auto"}:
            print(
                "\n  NOTE: this run used only the auto-gradable subset — objective\n"
                "  ground-truth grading, no API spend. It is a smaller and EASIER\n"
                "  sample than the full set (short factual and arithmetic questions),\n"
                "  so acceptance rates here are higher than production traffic would\n"
                "  give. Run with --real-frontier for the judge-graded half."
            )

    for gate in ("small", "large"):
        scores, labels = raw_scores(results, gate)
        X, y, _ = build_matrix(results, gate)
        if len(labels) == 0:
            print(f"\n  gate {gate}: no labelled data")
            continue

        report = validate(gate, X, y)
        cats = per_category(results, gate)
        grid = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]
        raw_rows = sweep(scores, labels, grid)

        payload[gate] = {
            "n": int(len(labels)),
            "acceptable_rate": round(float(labels.mean()), 3),
            "validation": report.as_dict(),
            "by_category": cats,
            "raw_sweep": [r.__dict__ for r in raw_rows],
        }

        if args.json:
            continue

        print(f"\n--- gate {gate} ({'0.5B' if gate == 'small' else '3B'}) ---")
        print(f"  n={len(labels)}   acceptable={labels.mean():.0%}")
        print(f"  by category: " + ", ".join(f"{c}={rate:.0%} (n={n})" for c, (n, rate) in cats.items()))
        print(f"\n  raw mode — thresholding geo_mean:")
        print(_table(raw_rows, highlight=shipped[gate]))

        print(f"\n  calibration: ECE={report.ece:.3f} (95% CI upper {report.ece_ci_high:.3f}), "
              f"Brier={report.brier:.3f} vs base rate {report.brier_baseline:.3f}")
        print(f"  verdict: {'PASS — calibration is trustworthy' if report.passed else 'FAIL — ' + report.verdict}")

        if report.passed:
            oof = cross_val_predictions(X, y)
            cal_rows = sweep(oof, y, [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95])
            print(f"\n  calibrated mode — thresholding P(acceptable):")
            print(_table(cal_rows, highlight=0.92 if gate == "small" else 0.85))

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    cal_path = Path(__file__).parent / "calibrator.json"
    if cal_path.exists():
        flip = json.loads(cal_path.read_text()).get("flip_recommended", False)
        print("\n=== recommendation ===")
        if flip:
            print("  CONFIDENCE_MODE=calibrated — both gates passed validation.")
            print("  Set TARGET_SMALL=0.92 and TARGET_LARGE=0.85.")
        else:
            print("  STAY IN raw MODE. At least one gate failed validation, and a")
            print("  system with one measured gate and one guessed gate should not")
            print("  present both behind the same 0.85 label.")
            print("  This is a real result, not a bug — see the per-gate verdicts above.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
