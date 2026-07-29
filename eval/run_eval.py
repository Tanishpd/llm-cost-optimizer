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
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

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
    # The budget this answer was generated under. Recorded because it is not a
    # detail — answer length feeds `n_tokens`, a calibrator feature, so a cache
    # mixing two budgets fits a model to a distribution that never existed.
    # `_check_budget` refuses to resume across a change.
    max_tokens: int = 0


@dataclass
class ApiUsage:
    """What one *paid* call consumed.

    The harness tracked `tokens_in`/`tokens_out` for the free local tiers and
    nothing at all for the API calls that cost money, so the only way to price
    a run was to estimate from character counts afterwards.

    `detail` keeps the provider's usage block whole rather than mapping it onto
    named fields. Reasoning models bill hidden thinking as output tokens and
    report them under keys that differ per provider — Gemini nests
    `reasoning_tokens` inside `completion_tokens_details`. A fixed schema drops
    exactly the tokens most likely to be a surprise on the bill.
    """

    kind: str  # "reference" | "judge"
    model: str
    tokens_in: int
    tokens_out: int
    detail: dict[str, Any] = field(default_factory=dict)
    # `total_tokens`, kept because it is the only place some providers admit to
    # thinking. Measured against Gemini's OpenAI layer on 2026-07-29:
    #
    #   judge, reasoning_effort=none   prompt 9 + completion 2 = total 11
    #   judge, unset                   prompt 9 + completion 2 = total 34
    #   gemini-3-flash-preview         prompt 9 + completion 2 = total 57
    #
    # No `reasoning_tokens` key anywhere in the response — the thinking is
    # visible *only* as the gap in `total_tokens`. A cost report keyed on
    # `completion_tokens` alone understates by whatever that gap is, silently,
    # which for the third row is 23x the visible output.
    tokens_total: int = 0


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
    usage: list[ApiUsage] = field(default_factory=list)


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
        # Runs recorded before usage accounting existed have no `usage` key.
        # They stay loadable — the cost report just reports what it has and
        # says how many calls predate the accounting, rather than either
        # crashing or quietly reporting $0.00 for a run that cost money.
        usage = [ApiUsage(**u) for u in (data.pop("usage", []) or [])]
        out[pid] = PromptResult(**data, tiers=tiers, usage=usage)
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


class SpendTracker:
    """What this run has spent so far, and whether it may spend more.

    Checked *before* each call rather than after, because a ceiling that only
    notices once it has been crossed is a report rather than a limit. The same
    reasoning as the gateway's budget check, and it inherits the same
    imprecision: cost is not knowable until the provider replies, so the run
    can exceed the ceiling by at most one call.
    """

    def __init__(self, ceiling_usd: Decimal,
                 prices: dict[str, tuple[Decimal, Decimal]]) -> None:
        self._ceiling = ceiling_usd
        self._prices = prices
        self.spent = Decimal(0)

    @property
    def armed(self) -> bool:
        return self._ceiling > 0

    def add(self, usage: ApiUsage) -> None:
        price_in, price_out = self._prices.get(usage.kind, (Decimal(0), Decimal(0)))
        self.spent += (_money(usage.tokens_in, price_in)
                       + _money(billable_out(usage), price_out))

    def check(self) -> None:
        if self.armed and self.spent >= self._ceiling:
            raise SpendCeilingReached(
                f"EVAL_MAX_SPEND_USD reached: ${self.spent:.4f} of ${self._ceiling}")


def usage_of(kind: str, model: str, completion) -> ApiUsage:
    """Pull one paid call's accounting off a Completion, keeping the provider's
    own usage block for whatever it reports beyond the two standard counts."""
    block = (completion.raw or {}).get("usage") or {}
    return ApiUsage(
        kind=kind,
        model=model,
        tokens_in=completion.tokens_in,
        tokens_out=completion.tokens_out,
        tokens_total=int(block.get("total_tokens") or 0),
        detail={k: v for k, v in block.items()
                if k not in ("prompt_tokens", "completion_tokens", "total_tokens")},
    )


# Hidden thinking is billed as output and reported — when it is reported at all
# — under a key that differs per provider. Gemini's OpenAI-compatible layer
# nests it; the native shape spells it `thoughts_token_count`.
_THINKING_KEYS = ("reasoning_tokens", "thoughts_token_count", "thinking_tokens")


def thinking_tokens(usage: ApiUsage) -> int:
    """Billed-but-invisible output tokens, wherever the provider put them.

    This project has already been bitten by these once: the judge was called
    with `max_tokens=10`, thinking consumed the entire budget, and every
    verdict came back empty — a run that would have produced zero labels after
    paying for every call.

    Two sources, and the second is the one that matters in practice. A named
    key (`reasoning_tokens`) is the documented shape and is what OpenAI sends.
    **Gemini's OpenAI-compatible layer sends no such key** — its thinking shows
    up only as the arithmetic gap in `total_tokens`, so a reader looking for
    the documented field finds nothing and reports zero.
    """
    def dig(d: dict[str, Any]) -> int:
        for key in _THINKING_KEYS:
            value = d.get(key)
            if isinstance(value, int) and value > 0:
                return value
        for value in d.values():
            if isinstance(value, dict) and (found := dig(value)):
                return found
        return 0

    if named := dig(usage.detail):
        return named
    gap = usage.tokens_total - usage.tokens_in - usage.tokens_out
    return max(0, gap)


def billable_out(usage: ApiUsage) -> int:
    """Output tokens actually charged, which is not `completion_tokens`.

    When the provider reports `total_tokens`, it is authoritative and settles
    the question outright: everything that is not input is output, thinking
    included. When it does not, `completion_tokens` is the best available
    figure — a named `reasoning_tokens` is already counted inside it per
    OpenAI's spec, so adding it would double-count.
    """
    if usage.tokens_total:
        return max(usage.tokens_out, usage.tokens_total - usage.tokens_in)
    return usage.tokens_out


def _money(tokens: int, price_per_million: Decimal) -> Decimal:
    """Costs are Decimal, never float accumulation (CLAUDE.md conventions)."""
    return (Decimal(tokens) / Decimal(1_000_000)) * price_per_million


def cost_report(
    results: dict[str, PromptResult],
    *,
    prices: dict[str, tuple[Decimal, Decimal]],
) -> str:
    """What the run actually cost, from provider-reported tokens.

    Reports a *range* when thinking tokens are present. Whether
    `completion_tokens` already includes them is provider-specific and not
    something this code can determine from one response, so it shows the
    as-reported figure and the if-billed-on-top figure rather than picking one
    and being quietly wrong by several multiples.
    """
    agg: dict[str, dict[str, int]] = {}
    for r in results.values():
        for u in r.usage:
            a = agg.setdefault(u.kind, {"calls": 0, "in": 0, "out": 0,
                                        "billed": 0, "think": 0, "exact": 0})
            a["calls"] += 1
            a["in"] += u.tokens_in
            a["out"] += u.tokens_out
            a["billed"] += billable_out(u)
            a["think"] += thinking_tokens(u)
            a["exact"] += 1 if u.tokens_total else 0

    if not agg:
        return ("  no paid-call usage recorded (free run, or a cache written "
                "before usage accounting existed)")

    lines = ["  paid API usage"]
    low = high = Decimal(0)
    for kind, a in sorted(agg.items()):
        price_in, price_out = prices.get(kind, (Decimal(0), Decimal(0)))
        base = _money(a["in"], price_in) + _money(a["billed"], price_out)
        low += base
        # Only ambiguous where the provider never reported a total; there,
        # thinking might sit outside `completion_tokens` and go uncharged here.
        unknown = a["think"] if a["exact"] < a["calls"] else 0
        high += base + _money(unknown, price_out)
        think = f"  thinking {a['think']:>7,}" if a["think"] else ""
        lines.append(f"    {kind:<10} {a['calls']:>4} calls   "
                     f"in {a['in']:>8,}   out {a['out']:>7,}{think}")
        lines.append(f"    {'':<10} billed out {a['billed']:>7,}  @ "
                     f"${price_in}/M in, ${price_out}/M out -> ${base:.4f}")

    total_think = sum(a["think"] for a in agg.values())
    if high > low:
        lines.append(f"\n  cost: ${low:.4f} to ${high:.4f}")
        lines.append("        some calls reported no total_tokens, so whether their")
        lines.append("        thinking is already inside completion_tokens is unknown.")
    else:
        lines.append(f"\n  cost: ${low:.4f}")
        if total_think:
            lines.append(f"        includes {total_think:,} thinking tokens, which are billed as")
            lines.append("        output and are absent from completion_tokens.")
    lines.append("        at configured prices — verify these against the provider's")
    lines.append("        current rates; the defaults are placeholders, not quotes.")
    return "\n".join(lines)


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
        return TierResult(tier, model, "", {}, 0, 0, 0.0, error=str(exc),
                          max_tokens=max_tokens)

    metrics = extract_metrics(completion.logprobs)
    return TierResult(
        tier=tier,
        model=model,
        content=completion.content,
        features=metrics.as_dict(),
        tokens_in=completion.tokens_in,
        tokens_out=completion.tokens_out,
        latency_ms=completion.latency_ms,
        max_tokens=max_tokens,
    )


class BudgetMismatch(RuntimeError):
    """The run cache was generated under a different `--max-tokens`."""


def check_budget(
    cached: dict[str, PromptResult], max_tokens: int, tag: str
) -> None:
    """Refuse to extend a run cache across a change of token budget.

    Answer length is not cosmetic here: `n_tokens` is one of the seven
    calibrator features, and a truncated answer scores differently from a
    complete one. Resuming a 120-token cache under a 512-token budget fits a
    distribution that no single configuration ever produced — and nothing
    downstream can detect it, because both halves look like valid rows.

    Caches written before `max_tokens` was recorded carry 0 and are treated as
    unknown rather than mismatched; there is no way to tell, and refusing to
    load every historical run would be worse than saying so.
    """
    budgets = {tr.max_tokens for r in cached.values()
               for tr in r.tiers.values() if tr.max_tokens}
    stale = sorted(b for b in budgets if b != max_tokens)
    if not stale:
        return
    raise BudgetMismatch(
        f"eval/runs/{tag}.json holds answers generated at max_tokens="
        f"{', '.join(str(b) for b in stale)}, but this run asks for "
        f"{max_tokens}. Mixing them corrupts the `n_tokens` feature.\n"
        f"       Either re-run at --max-tokens {stale[0]}, or start a clean "
        f"cache:\n"
        f"         rm eval/runs/{tag}.json        # discards paid results\n"
        f"         python -m eval.run_eval --real-frontier --tag {tag}2"
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
    judge_reasoning_effort: str = "",
    reference_reasoning_effort: str = "",
    spend: SpendTracker | None = None,
) -> dict[str, PromptResult]:
    cached = load_cached(tag)
    check_budget(cached, max_tokens, tag)
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
                # Before the call, not after: a ceiling that notices only once
                # it has been crossed is a report, not a limit.
                if spend is not None:
                    spend.check()
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
                        reasoning_effort=reference_reasoning_effort,
                    ),
                    what=f"{p.id} reference",
                )
                usage = usage_of("reference", frontier_model, ref)
                result.usage.append(usage)
                if spend is not None:
                    spend.add(usage)
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
                            reasoning_effort=judge_reasoning_effort,
                        ),
                        what=f"{p.id} judge/{tier}",
                    )
                    usage = usage_of("judge", judge_model, verdict)
                    result.usage.append(usage)
                    if spend is not None:
                        spend.add(usage)
                    g = parse_judge(verdict.content)
                    if g is None:
                        # A judge that returns nothing produces no labels, and
                        # a calibration with no labels fails silently at the
                        # very end after every call has been paid for. Count
                        # them, and stop early rather than spend the rest.
                        # Record the raw verdict, not just the count. Without
                        # it a parse failure is undiagnosable after the run:
                        # you know a label is missing and have no way to learn
                        # why, which is how a 25% label loss becomes a shrug.
                        raw = (verdict.content or "").strip()
                        result.label_reason[tier] = f"unparseable judge verdict: {raw[:120]!r}"
                        unparseable.append((p.id, tier, raw[:40]))
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


class SpendCeilingReached(RuntimeError):
    """`EVAL_MAX_SPEND_USD` bound mid-run.

    Mirrors `DailyQuotaExhausted` deliberately: both mean "stop calling the
    API, keep what you have". Everything collected so far is already
    checkpointed, so the caller fits on it rather than discarding a partial
    run that cost real money to produce.
    """


class DailyQuotaExhausted(RuntimeError):
    """A per-day quota is gone. Distinct from a per-minute limit because the
    right response is opposite: a minute limit is worth waiting out, a day
    limit is not — the wait is hours, and retrying only burns the backoff.

    Raised so the caller can stop cleanly with progress saved rather than
    discover it after five doublings and a crash.
    """


# Google spells it in the quotaId; the shape is stable across their APIs.
_PER_DAY_RE = re.compile(r"PerDay|per_day|RequestsPerDay", re.I)


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
            if _PER_DAY_RE.search(str(exc)):
                raise DailyQuotaExhausted(str(exc)[:200]) from exc
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


def build_parser() -> argparse.ArgumentParser:
    """Separate from `main` so a test can assert on the real flags rather than
    on a reconstruction of them, which would pass while the parser was wrong."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-frontier", action="store_true",
                        help="call a real API for references and judging (COSTS MONEY)")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N prompts")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="local tier output budget (default: "
                             "DEFAULT_MAX_TOKENS, what the gateway serves)")
    parser.add_argument("--tag", default="", help="run cache name (default: auto or full)")
    parser.add_argument("--fit-only", action="store_true",
                        help="fit from the existing run cache without collecting "
                             "anything new — for when a provider's quota ran out "
                             "mid-run and the partial data is worth fitting")
    parser.add_argument("--only-judged", action="store_true",
                        help="only the prompts that need the real API — the path "
                             "--limit cannot reach, since the dataset starts with "
                             "auto-gradable ones")
    return parser


async def main() -> int:
    args = build_parser().parse_args()

    # Settings, not os.getenv — it is the project convention (CLAUDE.md) and,
    # more practically, it is what reads `.env`. Reading the environment
    # directly meant a populated `.env` was silently ignored here while
    # working everywhere else in the codebase, which is the worst kind of
    # inconsistency: the file is right, the app is right, and only this script
    # disagrees.
    settings = get_settings()
    use_real = args.real_frontier or settings.eval_use_real_frontier
    tag = args.tag or ("full" if use_real else "auto")

    # Default to what the gateway actually serves. It used to default to 120,
    # which meant the calibrator was fitted to answers a quarter the length of
    # the ones it would score in production — and `n_tokens` is one of its
    # seven features. It also made the judge compare a truncated local answer
    # against a full-length reference, which reads as "incomplete" and cost
    # the local tiers labels they had earned.
    max_tokens = args.max_tokens or settings.default_max_tokens

    prompts = DATASET if use_real else [p for p in DATASET if p.grade == "auto"]
    if args.fit_only:
        # Only what is already cached. No provider is contacted, so this is
        # free and cannot fail on a quota.
        have = set(load_cached(args.tag or ("full" if use_real else "auto")))
        prompts = [p for p in prompts if p.id in have]
    if args.only_judged:
        prompts = [p for p in prompts if p.grade != "auto"]
    if args.limit:
        prompts = prompts[: args.limit]

    print(f"\ndataset: {summary()}")
    mode_label = "FULL (real API)" if use_real else "AUTO (free)"
    print(f"running: {len(prompts)} prompt(s), mode={mode_label}, "
          f"local max_tokens={max_tokens}")
    print(f"cache:   eval/runs/{tag}.json\n")

    # Armed by default (D27: a safety limit may not default to off). The
    # gateway's MAX_DAILY_SPEND_USD lives in BudgetLedger on the request path,
    # which this script never touches — so without this the one component that
    # deliberately spends money had no cap at all.
    spend = SpendTracker(settings.eval_max_spend_usd, {
        "reference": (settings.frontier_price_in, settings.frontier_price_out),
        "judge": (settings.eval_judge_price_in, settings.eval_judge_price_out),
    })

    frontier = judge = None
    client = None
    if use_real and not args.fit_only:
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

        def _effort(value: str) -> str:
            return f", reasoning_effort={value}" if value else ""

        print(f"  frontier: {settings.frontier_model} at {base}"
              f"{_effort(settings.eval_reference_reasoning_effort)}")
        print(f"  judge:    {settings.eval_judge_model} at {judge_base}"
              f"{_effort(settings.eval_judge_reasoning_effort)}")
        if judge_base != base:
            print("  note:     judge and frontier are different providers "
                  "(deliberate — no model grades its own output, D11)")

        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        frontier = FrontierProvider(client, base_url=base, api_key=key)
        judge = FrontierProvider(client, base_url=judge_base, api_key=judge_key)

        if spend.armed:
            print(f"  ceiling:  ${settings.eval_max_spend_usd} for this run "
                  f"(EVAL_MAX_SPEND_USD)")
        else:
            print("  ceiling:  DISARMED (EVAL_MAX_SPEND_USD=0) — this run has no "
                  "spend limit")

    try:
        started = time.perf_counter()
        results = {}
        try:
            results = await collect(
            prompts,
            tag=tag,
            small_model=settings.small_model,
            large_model=settings.large_model,
            max_tokens=max_tokens,
            use_real_frontier=use_real,
            frontier=frontier,
            judge=judge,
            judge_model=settings.eval_judge_model,
            frontier_model=settings.frontier_model,
            judge_max_tokens=settings.eval_judge_max_tokens,
            reference_max_tokens=settings.eval_reference_max_tokens,
            call_interval=settings.eval_min_call_interval_s,
            judge_reasoning_effort=settings.eval_judge_reasoning_effort,
            reference_reasoning_effort=settings.eval_reference_reasoning_effort,
            spend=spend,
        )
        except SpendCeilingReached as exc:
            # Same shutdown as a daily quota: everything so far is checkpointed,
            # so fit on it rather than discard a partial run that cost money.
            results = load_cached(tag)
            print(f"\n  SPEND CEILING REACHED after {len(results)}/{len(DATASET)} prompts.")
            print(f"  {exc}")
            print("  Progress is saved. Raise EVAL_MAX_SPEND_USD to continue, or")
            print("  re-run to resume from the cache — nothing is paid for twice.")
            print("  Fitting on what has been collected so far:\n")
        except DailyQuotaExhausted as exc:
            # A per-day quota is not worth waiting out, and every prompt
            # completed so far is already on disk. Reload and fit what we have
            # — a run that stops with a usable partial result beats one that
            # crashes with the same data and no fit.
            results = load_cached(tag)
            done = len(results)
            print(f"\n  DAILY QUOTA EXHAUSTED after {done}/{len(DATASET)} prompts.")
            print(f"  {exc}")
            print("  Progress is saved. Re-run the same command after the quota")
            print("  resets (Google's is midnight Pacific) to continue.")
            print("  Fitting on what has been collected so far:\n")
        except BudgetMismatch as exc:
            # Not a crash — a refusal, with the two ways out spelled in the
            # message. A traceback here would bury them.
            print(f"\nERROR: {exc}\n")
            return 2
    finally:
        if client is not None:
            await client.aclose()

    print(f"\ncollected in {time.perf_counter() - started:.0f}s\n")

    if use_real:
        print(cost_report(results, prices={
            "reference": (settings.frontier_price_in, settings.frontier_price_out),
            "judge": (settings.eval_judge_price_in, settings.eval_judge_price_out),
        }))
        print()

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

    # What was calibrated, so a later reader — or the gateway at startup — can
    # tell whether these weights apply to the models actually configured.
    budgets = sorted({tr.max_tokens for r in results.values()
                      for tr in r.tiers.values() if tr.max_tokens})
    save(CALIBRATOR_PATH, calibrators, reports, provenance={
        "small_model": settings.small_model,
        "large_model": settings.large_model,
        "max_tokens": budgets[0] if len(budgets) == 1 else budgets,
        "n_prompts": len(results),
        "judge_model": settings.eval_judge_model if use_real else "",
        "frontier_model": settings.frontier_model if use_real else "",
        "graded_against": "real frontier" if use_real else "ground truth only",
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    shown = (CALIBRATOR_PATH.relative_to(Path.cwd())
             if CALIBRATOR_PATH.is_relative_to(Path.cwd()) else CALIBRATOR_PATH)
    print(f"\nwrote {shown}")
    print("run `python -m eval.report` for the cost-vs-quality frontier.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
