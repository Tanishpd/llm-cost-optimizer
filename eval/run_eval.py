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
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np

from app.confidence import extract_metrics
from app.config import get_settings
from app.providers.base import ProviderError, ProviderTimeout, ProviderUnavailable
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
    frontier_model: str,
    judge_max_tokens: int = 64,
    reference_max_tokens: int = 512,
    call_interval: float = 0.0,
) -> dict[str, PromptResult]:
    cached = load_cached(tag)
    unparseable: list[tuple[str, str, str]] = []
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
                await paced(call_interval)
                ref = await with_retry(
                    # Bound as defaults, not captured: the lambda is called
                    # immediately so late binding is harmless here, but a
                    # future edit that defers it would silently use the last
                    # loop iteration's prompt.
                    lambda p=p, m=frontier_model: frontier.chat(
                        model=m,
                        messages=[{"role": "user", "content": p.prompt}],
                        # Not `max_tokens`: that is the *local* tiers' budget.
                        max_tokens=reference_max_tokens,
                    ),
                    what=f"{p.id} reference",
                )
                if ref.finish_reason == "length":
                    # A truncated reference is worse than no reference: the
                    # judge compares against a fragment and the label looks
                    # real. Reasoning models eat this budget invisibly.
                    print(f"      {p.id}: reference truncated at "
                          f"{reference_max_tokens} tokens — raise "
                          "EVAL_REFERENCE_MAX_TOKENS")
                result.reference = ref.content
                for tier, tr in result.tiers.items():
                    await paced(call_interval)
                    verdict = await with_retry(
                        lambda tr=tr, p=p, ref=ref: judge.chat(
                            model=judge_model,
                            messages=[
                                {"role": "system", "content": JUDGE_SYSTEM},
                                {"role": "user",
                                 "content": judge_prompt(p.prompt, ref.content, tr.content)},
                            ],
                            max_tokens=judge_max_tokens,
                        ),
                        what=f"{p.id} judge/{tier}",
                    )
                    g = parse_judge(verdict.content)
                    if g is None:
                        # A judge that returns nothing produces no labels, and
                        # a calibration with no labels fails silently at the
                        # very end after every call has been paid for. Count
                        # them, and stop early rather than spend the rest.
                        unparseable.append((p.id, tier, verdict.content[:40]))
                        if len(unparseable) >= 5 and not any(r.labels for r in results.values()):
                            raise RuntimeError(
                                f"the judge returned {len(unparseable)} unparseable verdicts "
                                f"and zero labels so far — e.g. {unparseable[:3]}. "
                                "A reasoning model needs headroom: raise "
                                "EVAL_JUDGE_MAX_TOKENS (currently "
                                f"{judge_max_tokens}). Stopping before spending more."
                            )
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


_last_call_at = 0.0


async def paced(interval_s: float) -> None:
    """Hold a minimum gap between API calls.

    Staying under a rate limit is faster than retrying through one: a 429 costs
    the request *and* the wait, and the provider has to serve the rejection
    either way.
    """
    global _last_call_at
    if interval_s <= 0:
        return
    gap = interval_s - (time.monotonic() - _last_call_at)
    if gap > 0:
        await asyncio.sleep(gap)
    _last_call_at = time.monotonic()


async def with_retry(call, *, what: str, attempts: int = 6, base_delay: float = 30.0):
    """Retry a rate-limited or briefly-unavailable API call.

    Free tiers rate-limit, and until this existed a single 429 aborted the
    whole run. Progress was never *lost* — results are checkpointed after every
    prompt — but finishing meant restarting the script by hand however many
    times the provider decided to push back.

    Exponential backoff from `base_delay`: 4, 8, 16, 32, 64 seconds. That is
    generous on purpose. A per-minute quota needs a wait measured in tens of
    seconds, and this loop is competing with nothing — the run is sequential
    and nobody is waiting on it.

    Timeouts are retried too. A judge call that hangs is far more likely to be
    a slow provider than a broken prompt.
    """
    for attempt in range(attempts):
        try:
            return await call()
        except (ProviderUnavailable, ProviderTimeout) as exc:
            if attempt == attempts - 1:
                raise
            # Providers usually say how long to wait; Gemini puts it in the
            # 429 body ("Please retry in 23.7s"). Believing them beats guessing
            # — blind exponential backoff either waits far too long or retries
            # into the same closed window.
            hinted = re.search(r"retry in ([\d.]+)s", str(exc))
            delay = (float(hinted.group(1)) + 1.0 if hinted
                     else base_delay * (2 ** attempt))
            print(f"      {what}: {type(exc).__name__} — waiting {delay:.0f}s "
                  f"(attempt {attempt + 1}/{attempts})")
            await asyncio.sleep(delay)
    raise RuntimeError("unreachable")


def build_matrix(
    results: dict[str, PromptResult], gate: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    parser.add_argument("--only-judged", action="store_true",
                        help="only the prompts that need the real API — the path "
                             "--limit cannot reach, since the dataset starts with "
                             "auto-gradable ones")
    args = parser.parse_args()

    # Settings, not os.getenv — it is the project convention (CLAUDE.md) and,
    # more practically, it is what reads `.env`. Reading the environment
    # directly meant a populated `.env` was silently ignored here while
    # working everywhere else in the codebase, which is the worst kind of
    # inconsistency: the file is right, the app is right, and only this script
    # disagrees.
    settings = get_settings()
    use_real = args.real_frontier or settings.eval_use_real_frontier
    tag = args.tag or ("full" if use_real else "auto")

    prompts = DATASET if use_real else [p for p in DATASET if p.grade == "auto"]
    if args.only_judged:
        prompts = [p for p in prompts if p.grade != "auto"]
    if args.limit:
        prompts = prompts[: args.limit]

    print(f"\ndataset: {summary()}")
    mode_label = "FULL (real API)" if use_real else "AUTO (free)"
    print(f"running: {len(prompts)} prompt(s), mode={mode_label}")
    print(f"cache:   eval/runs/{tag}.json\n")

    frontier = judge = None
    client = None
    if use_real:
        key = settings.frontier_api_key.get_secret_value()
        judge_key = settings.eval_judge_api_key.get_secret_value() or key
        if not key or key == "mock-key":
            print("ERROR: --real-frontier needs FRONTIER_API_KEY set to a real key.")
            return 2

        base = settings.frontier_base_url
        judge_base = settings.eval_judge_base_url
        # The single most likely misconfiguration, and it would otherwise
        # produce 129 calls to a mock that returns canned text — a calibration
        # fitted to nothing, with no sign anything was wrong.
        if "localhost" in base or "127.0.0.1" in base:
            print(f"ERROR: FRONTIER_BASE_URL is still the mock ({base}).")
            print("       A real calibration needs a real provider. See the README.")
            return 2

        print(f"  frontier: {settings.frontier_model} at {base}")
        print(f"  judge:    {settings.eval_judge_model} at {judge_base}")
        if judge_base != base:
            print("  note:     judge and frontier are different providers "
                  "(deliberate — no model grades its own output, D11)")

        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        frontier = FrontierProvider(client, base_url=base, api_key=key)
        judge = FrontierProvider(client, base_url=judge_base, api_key=judge_key)

    try:
        started = time.perf_counter()
        results = await collect(
            prompts,
            tag=tag,
            small_model=settings.small_model,
            large_model=settings.large_model,
            max_tokens=args.max_tokens,
            use_real_frontier=use_real,
            frontier=frontier,
            judge=judge,
            judge_model=settings.eval_judge_model,
            frontier_model=settings.frontier_model,
            judge_max_tokens=settings.eval_judge_max_tokens,
            reference_max_tokens=settings.eval_reference_max_tokens,
            call_interval=settings.eval_min_call_interval_s,
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

    # A partial run must not replace the artifact the gateway loads at startup.
    #
    # `--limit 3 --tag smoke` overwrote eval/calibrator.json with a 3-sample
    # fit, and with CONFIDENCE_MODE=calibrated the gateway would have started
    # serving on it — no error, no warning, just a confidence number derived
    # from three prompts. A smoke test that silently repoints production is
    # worse than no smoke test.
    partial = bool(args.limit) or (args.tag and args.tag not in ("auto", "full"))
    if partial:
        print(f"\nNOT writing {CALIBRATOR_PATH.name}: this was a partial run "
              f"({'--limit ' + str(args.limit) if args.limit else '--tag ' + args.tag}). "
              "Re-run without --limit/--tag to publish a fit.")
        print("run `python -m eval.report` for the cost-vs-quality frontier.\n")
        return 0

    save(CALIBRATOR_PATH, calibrators, reports)
    shown = (CALIBRATOR_PATH.relative_to(Path.cwd())
             if CALIBRATOR_PATH.is_relative_to(Path.cwd()) else CALIBRATOR_PATH)
    print(f"\nwrote {shown}")
    print("run `python -m eval.report` for the cost-vs-quality frontier.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
