#!/usr/bin/env python3
"""Run the eval: every prompt through every local tier, then fit and validate.

    python -m eval.run_eval                    # free: auto-gradable subset only
    python -m eval.run_eval --real-frontier    # adds reference answers + judge ($$$)
    python -m eval.run_eval --limit 20         # quick smoke run

**Every model response is cached to `eval/runs/`.** Re-running reuses it, so
re-fitting a calibrator never re-spends and never re-waits. Delete the file to
force a fresh run.

The default path costs nothing and still produces a real calibration, because
the auto-gradable prompts are graded against ground truth rather than a model's
opinion. It is a smaller and easier sample than the full set — `report.py` says
so rather than letting a partial result read as the whole story.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from app.confidence import extract_metrics
from app.providers.base import ProviderError
from app.providers.frontier import FrontierProvider
from app.providers.ollama import OllamaProvider
from eval.calibration import FEATURE_NAMES, fit_calibrator, save, validate
from eval.dataset import DATASET, EvalPrompt, summary
from eval.grading import JUDGE_SYSTEM, grade_auto, judge_prompt, parse_judge

RUNS_DIR = Path(__file__).parent / "runs"
CALIBRATOR_PATH = Path(__file__).parent / "calibrator.json"

# Which local tier feeds which gate. Gate `small` decides whether the 0.5B's
# answer stands; gate `large` does the same for the 3B.
TIER_TO_GATE = {"local_small": "small", "local_large": "large"}


@dataclass
class TierResult:
    tier: str
    model: str
    content: str
    features: dict[str, float]
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str = ""


@dataclass
class PromptResult:
    id: str
    category: str
    grade_mode: str
    prompt: str
    tiers: dict[str, TierResult] = field(default_factory=dict)
    reference: str = ""
    labels: dict[str, bool] = field(default_factory=dict)
    label_source: dict[str, str] = field(default_factory=dict)
    label_reason: dict[str, str] = field(default_factory=dict)


def _run_path(tag: str) -> Path:
    return RUNS_DIR / f"{tag}.json"


def load_cached(tag: str) -> dict[str, PromptResult]:
    """Previous responses, keyed by prompt id. Never re-spend."""
    path = _run_path(tag)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out = {}
    for pid, data in raw.items():
        tiers = {k: TierResult(**v) for k, v in (data.pop("tiers", {}) or {}).items()}
        out[pid] = PromptResult(**data, tiers=tiers)
    return out


def save_cached(tag: str, results: dict[str, PromptResult]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _run_path(tag).write_text(
        json.dumps(
            {pid: {**asdict(r), "tiers": {k: asdict(v) for k, v in r.tiers.items()}}
             for pid, r in results.items()},
            indent=2,
        )
    )


async def run_tier(
    provider: OllamaProvider, tier: str, model: str, prompt: str, *, max_tokens: int
) -> TierResult:
    """One tier's answer plus its confidence features. Errors are recorded, not
    raised — one dead model should not abort a 125-prompt run."""
    try:
        completion = await provider.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # calibrators are valid at the temperature they were fitted at
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
    except ProviderError as exc:
        return TierResult(tier, model, "", {}, 0, 0, 0.0, error=str(exc))

    metrics = extract_metrics(completion.logprobs)
    return TierResult(
        tier=tier,
        model=model,
        content=completion.content,
        features=metrics.as_dict(),
        tokens_in=completion.tokens_in,
        tokens_out=completion.tokens_out,
        latency_ms=completion.latency_ms,
    )


async def collect(
    prompts: list[EvalPrompt],
    *,
    tag: str,
    small_model: str,
    large_model: str,
    max_tokens: int,
    use_real_frontier: bool,
    frontier: FrontierProvider | None,
    judge: FrontierProvider | None,
    judge_model: str,
) -> dict[str, PromptResult]:
    cached = load_cached(tag)
    results: dict[str, PromptResult] = dict(cached)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
        ollama = OllamaProvider(client)
        for i, p in enumerate(prompts, start=1):
            if p.id in results and results[p.id].tiers:
                print(f"  [{i:>3}/{len(prompts)}] {p.id:<10} cached")
                continue

            result = PromptResult(
                id=p.id, category=p.category, grade_mode=p.grade, prompt=p.prompt
            )
            for tier, model in (("local_small", small_model), ("local_large", large_model)):
                result.tiers[tier] = await run_tier(
                    ollama, tier, model, p.prompt, max_tokens=max_tokens
                )

            # ---- labels -------------------------------------------------
            if p.grade == "auto":
                for tier, tr in result.tiers.items():
                    g = grade_auto(p, tr.content)
                    result.labels[tier] = g.acceptable
                    result.label_source[tier] = "auto"
                    result.label_reason[tier] = g.reason
            elif use_real_frontier and frontier is not None and judge is not None:
                ref = await frontier.chat(
                    model=os.getenv("FRONTIER_MODEL", "gpt-4o"),
                    messages=[{"role": "user", "content": p.prompt}],
                    max_tokens=max_tokens,
                )
                result.reference = ref.content
                for tier, tr in result.tiers.items():
                    verdict = await judge.chat(
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user",
                             "content": judge_prompt(p.prompt, ref.content, tr.content)},
                        ],
                        max_tokens=10,
                    )
                    g = parse_judge(verdict.content)
                    if g is not None:
                        result.labels[tier] = g.acceptable
                        result.label_source[tier] = "judge"
                        result.label_reason[tier] = g.reason
            # else: judge-gradable prompt with no real frontier -> no label.
            # Deliberately left unlabelled rather than guessed at.

            results[p.id] = result
            labelled = "".join("Y" if result.labels.get(t) else "." for t in result.tiers)
            print(f"  [{i:>3}/{len(prompts)}] {p.id:<10} {p.category:<16} labels={labelled}")
            save_cached(tag, results)  # checkpoint after every prompt

    return results


def build_matrix(results: dict[str, PromptResult], gate: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Feature matrix and labels for one gate."""
    tier = next(t for t, g in TIER_TO_GATE.items() if g == gate)
    X, y, ids = [], [], []
    for r in results.values():
        tr = r.tiers.get(tier)
        if tr is None or tr.error or not tr.features:
            continue
        if tier not in r.labels:
            continue
        X.append([float(tr.features.get(name, 0.0)) for name in FEATURE_NAMES])
        y.append(1.0 if r.labels[tier] else 0.0)
        ids.append(r.id)
    return np.array(X, dtype=float), np.array(y, dtype=float), ids


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-frontier", action="store_true",
                        help="call a real API for references and judging (COSTS MONEY)")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N prompts")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--tag", default="", help="run cache name (default: auto or full)")
    args = parser.parse_args()

    use_real = args.real_frontier or os.getenv("EVAL_USE_REAL_FRONTIER", "").lower() == "true"
    tag = args.tag or ("full" if use_real else "auto")

    prompts = DATASET if use_real else [p for p in DATASET if p.grade == "auto"]
    if args.limit:
        prompts = prompts[: args.limit]

    print(f"\ndataset: {summary()}")
    print(f"running: {len(prompts)} prompt(s), mode={'FULL (real API)' if use_real else 'AUTO (free)'}")
    print(f"cache:   eval/runs/{tag}.json\n")

    frontier = judge = None
    client = None
    if use_real:
        key = os.getenv("FRONTIER_API_KEY", "")
        judge_key = os.getenv("EVAL_JUDGE_API_KEY", "") or key
        if not key or key == "mock-key":
            print("ERROR: --real-frontier needs FRONTIER_API_KEY set to a real key.")
            return 2
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        frontier = FrontierProvider(
            client, base_url=os.getenv("FRONTIER_BASE_URL", "https://api.openai.com/v1"),
            api_key=key,
        )
        judge = FrontierProvider(
            client, base_url=os.getenv("EVAL_JUDGE_BASE_URL", "https://api.anthropic.com/v1"),
            api_key=judge_key,
        )

    try:
        started = time.perf_counter()
        results = await collect(
            prompts,
            tag=tag,
            small_model=os.getenv("SMALL_MODEL", "qwen2.5:0.5b"),
            large_model=os.getenv("LARGE_MODEL", "qwen2.5:3b"),
            max_tokens=args.max_tokens,
            use_real_frontier=use_real,
            frontier=frontier,
            judge=judge,
            judge_model=os.getenv("EVAL_JUDGE_MODEL", "claude-sonnet-5"),
        )
    finally:
        if client is not None:
            await client.aclose()

    print(f"\ncollected in {time.perf_counter() - started:.0f}s\n")

    calibrators, reports = {}, {}
    for gate in ("small", "large"):
        X, y, _ids = build_matrix(results, gate)
        report = validate(gate, X, y)
        reports[gate] = report
        if len(y) >= 10 and len(np.unique(y)) >= 2:
            calibrators[gate] = fit_calibrator(X, y)
        mark = "PASS" if report.passed else "FAIL"
        print(f"  gate {gate:<6} n={report.n:<4} acceptable={report.positive_rate:.0%}  "
              f"ECE={report.ece:.3f} (CI up {report.ece_ci_high:.3f})  "
              f"Brier={report.brier:.3f} vs base {report.brier_baseline:.3f}  [{mark}]")
        if not report.passed:
            print(f"           -> {report.verdict}")

    save(CALIBRATOR_PATH, calibrators, reports)
    print(f"\nwrote {CALIBRATOR_PATH.relative_to(Path.cwd()) if CALIBRATOR_PATH.is_relative_to(Path.cwd()) else CALIBRATOR_PATH}")
    print("run `python -m eval.report` for the cost-vs-quality frontier.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
