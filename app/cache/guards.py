"""False-hit guards for the semantic cache.

Pure functions, in their own module because this is the highest-risk logic in
the project. Every other failure mode is visible — a slow answer, a missing
answer, an escalation. **A false cache hit is a wrong answer served instantly,
for free, with no signal that anything happened** (D15).

High cosine is not sameness. `capital of Australia` / `capital of Austria` and
`convert 50 USD` / `convert 70 USD` both exceed 0.95 with any general-purpose
embedder, because the embedding captures *topic*, and these pairs share one.
What differs is a single token that changes the answer completely.

So a semantic hit must clear the cosine threshold **and** every guard here.
The asymmetry is severe enough that every judgment call goes the same way:
a miss costs ~120 ms, a false hit costs correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Numbers, including decimals and thousands separators. "50" vs "70" is the
# whole point — `convert 50 USD` must not match `convert 70 USD`.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")

# Capitalized words as a proper-noun proxy. Deliberately crude: a real NER
# model would be more accurate but would mean a second inference runtime (D6),
# and over-rejecting is the safe direction here.
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Words that are capitalized only because they start a sentence, so they carry
# no entity signal and would otherwise cause spurious mismatches.
_SENTENCE_STARTERS = frozenset({
    "The", "What", "Who", "Where", "When", "Why", "How", "Which", "Can", "Could",
    "Would", "Should", "Is", "Are", "Was", "Were", "Do", "Does", "Did", "Please",
    "Give", "Tell", "List", "Explain", "Describe", "Write", "Show", "Find",
    "Summarize", "Summarise", "Translate", "Convert", "Compare", "Name", "If",
    "In", "On", "At", "For", "And", "But", "Or", "It", "This", "That", "There",
    "A", "An", "As", "So", "To", "Of",
})

# Negation. "is safe" vs "is not safe" are opposite questions with near-identical
# embeddings. Contractions matter: "isn't" must count as negation.
_NEGATION_PATTERNS = (
    r"\bnot\b", r"\bnever\b", r"\bwithout\b", r"\bexcept\b", r"\bneither\b",
    r"\bnor\b", r"\bno\b", r"\bnone\b", r"\bnothing\b", r"\bcannot\b",
    r"\bunable\b", r"\bexclude\b", r"\bexcluding\b", r"\bdoesn't\b", r"\bdon't\b",
    r"\bisn't\b", r"\baren't\b", r"\bwasn't\b", r"\bweren't\b", r"\bwon't\b",
    r"\bcan't\b", r"\bshouldn't\b", r"\bwouldn't\b", r"\bcouldn't\b", r"\bhasn't\b",
    r"\bhaven't\b", r"\bdidn't\b",
)
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE)

# Guard names, as they appear in `x_hlr.cache.guard_rejected`.
NUMBER_MISMATCH = "number_mismatch"
ENTITY_MISMATCH = "entity_mismatch"
NEGATION_MISMATCH = "negation_mismatch"
MULTI_TURN = "multi_turn"
SYSTEM_PROMPT_DIFFERS = "system_prompt_differs"
TOO_LONG = "too_long"
PARAMS_DIFFER = "params_differ"


@dataclass(frozen=True)
class GuardResult:
    """Whether a candidate may be served, and if not, which guard stopped it.

    The rejecting guard is named so over-conservatism is *measurable* rather
    than invisible — `/metrics` counts rejections per guard, and a guard that
    fires constantly is a tuning signal, not a mystery.
    """

    passed: bool
    rejected_by: str | None = None
    detail: str = ""


PASSED = GuardResult(passed=True)


def extract_numbers(text: str) -> set[str]:
    """Numbers, normalized so `1,000` and `1000` compare equal."""
    return {m.replace(",", "") for m in _NUMBER_RE.findall(text)}


def extract_entities(text: str) -> set[str]:
    """Capitalized words that aren't merely sentence-initial.

    A word is kept if it appears capitalized somewhere other than the first
    position of a sentence, or if it isn't a common sentence-starter — so
    `Australia` survives while `What` does not.
    """
    found: set[str] = set()
    for match in _ENTITY_RE.finditer(text):
        word = match.group(0)
        if word in _SENTENCE_STARTERS:
            continue
        found.add(word.lower())
    return found


def has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def check_pair(candidate: str, stored: str) -> GuardResult:
    """Do these two prompts want the same answer?

    Called only after cosine similarity has already cleared the threshold —
    this is the second, lexical opinion on a pair the embedder thinks are alike.
    """
    cand_numbers, stored_numbers = extract_numbers(candidate), extract_numbers(stored)
    if cand_numbers != stored_numbers:
        return GuardResult(
            False,
            NUMBER_MISMATCH,
            f"numbers differ: {sorted(cand_numbers)} vs {sorted(stored_numbers)}",
        )

    cand_entities, stored_entities = extract_entities(candidate), extract_entities(stored)
    if cand_entities != stored_entities:
        only_cand = sorted(cand_entities - stored_entities)
        only_stored = sorted(stored_entities - cand_entities)
        return GuardResult(
            False,
            ENTITY_MISMATCH,
            f"entities differ: +{only_cand} -{only_stored}",
        )

    if has_negation(candidate) != has_negation(stored):
        return GuardResult(
            False,
            NEGATION_MISMATCH,
            "one prompt is negated and the other is not",
        )

    return PASSED


def check_eligible(
    *,
    prompt: str,
    is_multi_turn: bool,
    max_chars: int,
) -> GuardResult:
    """Is this request eligible for the semantic cache at all?

    Checked *before* embedding, so an ineligible request costs nothing.

    Multi-turn is excluded because the embedding of the last message flattens
    away the conversation that makes its answer correct — "what about it?" is
    identical in every conversation and means something different in each.

    Long prompts are excluded because they carry attached context (a document,
    a stack trace) that dominates length but not the embedding; two prompts can
    look alike while their attached material differs entirely.
    """
    if is_multi_turn:
        return GuardResult(False, MULTI_TURN, "multi-turn conversations are never semantic-cached")
    if len(prompt) > max_chars:
        return GuardResult(
            False, TOO_LONG, f"prompt is {len(prompt)} chars, over the {max_chars} limit"
        )
    return PASSED


# Known residual risk, documented rather than guarded:
#
#   "convert USD to EUR"  vs  "convert EUR to USD"
#
# Same numbers, same entities, same negation — the guards pass, and cosine is
# high. Catching argument-order swaps needs parsing, not lexical sets, which is
# more machinery than the failure justifies at this scale. Recorded here so it
# is a known limitation rather than an undiscovered bug.
ARGUMENT_ORDER_IS_UNGUARDED = True
