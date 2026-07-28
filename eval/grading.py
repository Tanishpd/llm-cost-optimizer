"""Grading a tier's answer.

Two modes, and they measure different things — the distinction matters when
reading a calibration result:

- **auto** — objective correctness against a known answer. Ground truth, free,
  deterministic. Nothing's opinion.
- **judge** — "as good as the reference", scored by a separate model. Needed for
  anything where quality is a judgement call, and it costs money.

Both produce the same binary label, `acceptable`, which is what the calibrators
are fitted against. `mode` travels with the label so a report can say which
kind of evidence a number rests on.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from eval.dataset import EvalPrompt


@dataclass(frozen=True)
class Grade:
    acceptable: bool
    mode: Literal["auto", "judge"]
    reason: str
    matched: str = ""


def _normalize(text: str) -> str:
    """Fold case, strip accents and punctuation, collapse whitespace.

    Accent folding matters: a model may write "Brasilia" or "Brasília", and
    both are the same answer. Punctuation goes because "Paris." and "Paris"
    are too.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = stripped.lower()
    cleaned = re.sub(r"[^\w\s%/^*+-]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains(haystack: str, needle: str) -> bool:
    """Whole-token containment, so "12" does not match inside "120".

    Substring matching is the obvious approach and it is wrong: `"4" in "42"`
    is True, which would mark a wrong arithmetic answer correct.
    """
    h, n = _normalize(haystack), _normalize(needle)
    if not n:
        return False
    return re.search(rf"(?<!\w){re.escape(n)}(?!\w)", h) is not None


def grade_auto(prompt: EvalPrompt, answer: str) -> Grade:
    """Objective check against the accepted and forbidden answers.

    `forbidden` is checked **first**: a model that names a plausible-but-wrong
    answer has failed, even if it also mentions a correct token somewhere. This
    catches the confident-hallucination case — the M0 probe answered "Treaty of
    Kadesh, 1956, Egypt and Israel", which mentions Egypt (correct!) alongside
    a fabricated date and the wrong counterparty.
    """
    if not answer.strip():
        return Grade(False, "auto", "empty answer")

    for bad in prompt.forbidden:
        if _contains(answer, bad):
            return Grade(
                False, "auto", f"contains the plausible-but-wrong answer {bad!r}", matched=bad
            )

    for good in prompt.expected:
        if _contains(answer, good):
            return Grade(True, "auto", f"contains expected answer {good!r}", matched=good)

    return Grade(
        False,
        "auto",
        f"none of {list(prompt.expected)} found in answer",
    )


JUDGE_SYSTEM = (
    "You grade whether a candidate answer is as good as a reference answer for the same "
    "question. Reply with exactly one word: ACCEPTABLE or UNACCEPTABLE.\n\n"
    "ACCEPTABLE means the candidate is factually equivalent to the reference and complete "
    "enough to be useful. Style, length and phrasing differences do not matter. Being less "
    "polished is fine.\n"
    "UNACCEPTABLE means the candidate is factually wrong, omits something essential, "
    "contradicts itself, or refuses to answer."
)


def judge_prompt(question: str, reference: str, candidate: str) -> str:
    """The rubric, kept minimal and explicit.

    A vague instruction ("is this good?") produces a judge that drifts between
    runs; M6 measures judge self-agreement on a subsample precisely because
    this is a model call and not an oracle.
    """
    return (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}\n\n"
        "Is the candidate ACCEPTABLE or UNACCEPTABLE?"
    )


def parse_judge(reply: str) -> Grade | None:
    """Read the judge's verdict. Returns None if it did not follow the rubric —
    an unparseable verdict must not silently become a label."""
    text = reply.strip().upper()
    if "UNACCEPTABLE" in text:
        return Grade(False, "judge", "judge said UNACCEPTABLE")
    if "ACCEPTABLE" in text:
        return Grade(True, "judge", "judge said ACCEPTABLE")
    return None


def grade(prompt: EvalPrompt, answer: str, *, judge_reply: str | None = None) -> Grade | None:
    """Dispatch on the prompt's mode. None means "not gradable here"."""
    if prompt.grade == "auto":
        return grade_auto(prompt, answer)
    if judge_reply is None:
        return None
    return parse_judge(judge_reply)
