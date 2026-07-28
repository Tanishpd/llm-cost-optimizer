"""Prompt feature extraction — pure functions, no model call.

The pre-router's whole job is to pick an *entry point* into the cascade cheaply.
It is allowed to be wrong in one direction only:

- **too low** → costs latency; the confidence gates still catch a weak answer;
- **too high** → costs money, and the frontier tier has no gate above it.

So features that push *downward* can be soft signals, while anything that pushes
toward the frontier must be a strong, multi-signal case (DECISIONS.md D10).

Everything here is `def`, not `async def` — no I/O, microseconds to run.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# lexicons
# ---------------------------------------------------------------------------

# Verbs that transform text already supplied. These are the canonical cheap
# workload: the answer is mostly in the prompt, so a small model does fine.
SIMPLE_VERBS = frozenset({
    "summarize", "summarise", "tldr", "translate", "rewrite", "rephrase",
    "paraphrase", "shorten", "expand", "list", "extract", "format", "convert",
    "capitalize", "lowercase", "spell", "correct", "proofread", "define",
})

# Verbs that ask for genuine reasoning or construction.
HARD_VERBS = frozenset({
    "prove", "derive", "design", "architect", "optimize", "optimise", "debug",
    "refactor", "implement", "analyze", "analyse", "evaluate", "critique",
    "compare", "contrast", "justify", "reconcile", "diagnose",
})

# Phrases that signal multi-step reasoning is expected.
REASONING_MARKERS = (
    r"step[- ]by[- ]step", r"\bexplain why\b", r"\bwalk me through\b",
    r"\breasoning\b", r"\bderive\b", r"\bprove\b", r"\btrade[- ]?offs?\b",
    r"\bpros and cons\b", r"\bwhy does\b", r"\bwhy is\b", r"\bhow would you\b",
    r"\bwhat if\b", r"\bimplications?\b", r"\bfirst principles\b",
)
_REASONING_RE = re.compile("|".join(REASONING_MARKERS), re.IGNORECASE)

_CODE_FENCE_RE = re.compile(r"```|~~~")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_ARITHMETIC_RE = re.compile(r"\d+\s*[-+*/^%]\s*\d+|\b\d+\s*(?:percent|%)\b")
_NUMBER_RE = re.compile(r"\d")
_QUESTION_RE = re.compile(r"\?")
# A capitalized word not at the start of a sentence — a crude proper-noun proxy.
_ENTITY_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b", re.MULTILINE)
_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Approximate token count **without** a tokenizer.

    The router runs before any model call, so the exact count (Ollama's
    `prompt_eval_count`) is not yet available — and pulling in a tokenizer to
    get it would add a dependency for a decision that only needs a rough
    magnitude.

    Uses `max(words × 1.3, chars / 4)`: the word factor covers ordinary prose,
    the char factor catches dense code and CJK where words are few but tokens
    are many. Deliberately biased to over-estimate slightly; under-estimating
    would make long prompts look cheap.
    """
    if not text:
        return 0
    words = len(_WORD_RE.findall(text))
    return max(int(words * 1.3), len(text) // 4, 1)


@dataclass(frozen=True)
class PromptFeatures:
    """What the policy scores. Serialized into `x_hlr.router.features`, so a
    routing decision can be second-guessed from the response alone."""

    n_tokens: int
    n_chars: int
    has_code_fence: bool
    has_inline_code: bool
    imperative_class: str  # "simple" | "hard" | "question" | "other"
    reasoning_markers: int
    has_arithmetic: bool
    question_count: int
    entity_density: float
    context_chars: int
    is_multi_turn: bool
    system_prompt_chars: int

    def as_dict(self) -> dict:
        return asdict(self)


def classify_imperative(text: str) -> str:
    """Which verb class opens the prompt.

    Only the first few words are inspected: "summarize this article about
    proving theorems" is a summarization task, not a proof task, and scanning
    the whole prompt would misread it.
    """
    words = re.findall(r"[a-z]+", text[:120].lower())
    for word in words[:6]:
        if word in SIMPLE_VERBS:
            return "simple"
        if word in HARD_VERBS:
            return "hard"
    if "?" in text[:200]:
        return "question"
    return "other"


def estimate_context_chars(text: str) -> int:
    """Characters that look like *supplied material* rather than instruction.

    Fenced blocks and long quoted runs are context the model must read but did
    not have to reason up. A prompt that is mostly context is usually a
    transformation task — long, but cheap.
    """
    total = 0
    for block in re.findall(r"```.*?```|~~~.*?~~~", text, re.DOTALL):
        total += len(block)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith((">", "    ")) and len(stripped) > 20:
            total += len(line)
    return total


def extract_features(
    prompt: str, *, system_prompt: str = "", is_multi_turn: bool = False
) -> PromptFeatures:
    """Everything the policy needs, in one pass. Microseconds."""
    n_chars = len(prompt)
    entities = len(set(_ENTITY_RE.findall(prompt)))
    words = max(len(_WORD_RE.findall(prompt)), 1)

    return PromptFeatures(
        n_tokens=estimate_tokens(prompt),
        n_chars=n_chars,
        has_code_fence=bool(_CODE_FENCE_RE.search(prompt)),
        has_inline_code=bool(_INLINE_CODE_RE.search(prompt)),
        imperative_class=classify_imperative(prompt),
        reasoning_markers=len(_REASONING_RE.findall(prompt)),
        has_arithmetic=bool(_ARITHMETIC_RE.search(prompt)),
        question_count=len(_QUESTION_RE.findall(prompt)),
        entity_density=round(entities / words, 4),
        context_chars=estimate_context_chars(prompt),
        is_multi_turn=is_multi_turn,
        system_prompt_chars=len(system_prompt),
    )


def has_numbers(text: str) -> bool:
    """Used by the semantic cache's guards (M5), kept here with the other
    lexical helpers."""
    return bool(_NUMBER_RE.search(text))
