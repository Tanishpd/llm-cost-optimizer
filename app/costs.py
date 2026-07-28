"""Token and USD arithmetic.

Pure functions. Money is `Decimal` throughout — never float. Float accumulation
over thousands of requests visibly drifts, and this module's output is the
project's headline number.

Three distinct accounting concepts, per docs/API.md. Conflating them was a real
bug in an earlier draft of the spec, so they are named apart here:

- **billed**    — tokens the frontier actually charged for. Drives `cost_usd`.
- **baseline**  — prompt tokens counted *once* plus the *serving* tier's
                  completion tokens, priced at frontier rates, for *every*
                  request including cache hits. Drives `baseline_cost_usd`,
                  i.e. "what would this traffic have cost with no gateway".
- **throughput** — tokens across every tier attempted, including discarded
                  intermediate completions. Load analysis only; **never** cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# Money is stored and accumulated as integer micro-USD (1e-6 USD) to make
# summation exact. Formatting to USD happens once, at the edge.
MICRO = Decimal("1000000")
USD_PLACES = Decimal("0.000001")


@dataclass(frozen=True)
class TokenUsage:
    """Ollama returns `prompt_eval_count` / `eval_count`; OpenAI returns
    `prompt_tokens` / `completion_tokens`. Normalized here."""

    tokens_in: int
    tokens_out: int

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    @classmethod
    def from_ollama(cls, response: dict) -> TokenUsage:
        """Read Ollama's counts.

        Note `eval_count` includes the stopping token, which produces no
        logprob entry — so `eval_count` and `len(logprobs)` legitimately
        differ. Verified in M0, RESEARCH.md §2. Cost uses `eval_count`
        (that is what a provider would bill); confidence uses the logprob
        array length.
        """
        return cls(
            tokens_in=int(response.get("prompt_eval_count") or 0),
            tokens_out=int(response.get("eval_count") or 0),
        )

    @classmethod
    def from_openai(cls, response: dict) -> TokenUsage:
        usage = response.get("usage") or {}
        return cls(
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )


def price_micro_usd(
    usage: TokenUsage, price_in_per_m: Decimal, price_out_per_m: Decimal
) -> int:
    """Cost of `usage` at the given per-million rates, in integer micro-USD.

    Integer result so callers can accumulate without drift.
    """
    micro = (
        (Decimal(usage.tokens_in) * price_in_per_m + Decimal(usage.tokens_out) * price_out_per_m)
        / Decimal(1_000_000)
    ) * MICRO
    return int(micro.to_integral_value(rounding=ROUND_HALF_UP))


def micro_to_usd(micro: int) -> Decimal:
    """Integer micro-USD -> Decimal USD, quantized to 6 dp."""
    return (Decimal(micro) / MICRO).quantize(USD_PLACES, rounding=ROUND_HALF_UP)


def usd_to_micro(usd: Decimal | str | int | float) -> int:
    """Decimal USD -> integer micro-USD, the inverse of `micro_to_usd`.

    Accepts `float` for convenience at the config edge, where a budget may
    arrive as JSON, but converts via `str` so a value like `0.1` becomes
    exactly 100000 rather than 99999 — `Decimal(0.1)` is the binary expansion,
    and this function's output is compared against accumulated spend.
    """
    if isinstance(usd, float):
        usd = Decimal(str(usd))
    return int((Decimal(usd) * MICRO).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class CostBreakdown:
    """What lands in the `x_hlr` block (docs/API.md)."""

    tokens_in: int
    tokens_out: int
    cost_micro_usd: int
    baseline_micro_usd: int

    @property
    def saved_micro_usd(self) -> int:
        return self.baseline_micro_usd - self.cost_micro_usd

    def as_dict(self) -> dict[str, int | float]:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": float(micro_to_usd(self.cost_micro_usd)),
            "baseline_cost_usd": float(micro_to_usd(self.baseline_micro_usd)),
            "cost_saved_usd": float(micro_to_usd(self.saved_micro_usd)),
        }


def compute_cost(
    *,
    served_by: str,
    usage: TokenUsage,
    price_in_per_m: Decimal,
    price_out_per_m: Decimal,
) -> CostBreakdown:
    """Cost and baseline for one request.

    `usage` is the *serving* tier's usage: prompt tokens once, plus the
    completion actually returned. Discarded intermediate completions from
    escalated tiers do not belong here — they are throughput.

    Local and cache tiers bill $0. That is a deliberate simplification:
    electricity and amortized hardware are real but not meaningfully
    attributable per request, and inventing a number would make the headline
    figure look more rigorous while being less honest (DECISIONS.md D13).

    The baseline prices *every* request at frontier rates, cache hits included —
    the "what would this traffic cost with no gateway" question.
    """
    baseline = price_micro_usd(usage, price_in_per_m, price_out_per_m)
    billed = baseline if served_by == "frontier" else 0
    return CostBreakdown(
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        cost_micro_usd=billed,
        baseline_micro_usd=baseline,
    )


def saved_pct(baseline_micro: int, actual_micro: int) -> float:
    """Headline savings percentage. 0.0 when nothing would have been spent."""
    if baseline_micro <= 0:
        return 0.0
    return round((baseline_micro - actual_micro) / baseline_micro * 100, 1)


def ns_to_ms(nanoseconds: int | float | None) -> float:
    """Ollama reports durations in nanoseconds (`total_duration`,
    `load_duration`, `prompt_eval_duration`, `eval_duration`)."""
    if not nanoseconds:
        return 0.0
    return round(float(nanoseconds) / 1_000_000, 1)
