"""The seam for a learned router.

Deliberately empty of implementations. The heuristic in `policy.py` ships first
because M6's eval harness is what generates honest training labels — writing a
learned router before that means training on hand-labelled guesses about what
the models can handle, which is circular (DECISIONS.md D10).

When M6 has run, implement :class:`Classifier` against `eval/runs/*.json` and
pass it to the policy. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.router.heuristics import PromptFeatures

EntryTier = str  # "local_small" | "local_large" | "frontier"


@dataclass(frozen=True)
class Prediction:
    """A classifier's output.

    `score` is a difficulty estimate in [0, 1] and `confidence` is how sure the
    classifier is *of that estimate* — separate numbers. A low-confidence
    prediction should fall back to the heuristic rather than be trusted, which
    is only expressible if the two are distinct.
    """

    entry_tier: EntryTier
    score: float
    confidence: float
    reason: str


@runtime_checkable
class Classifier(Protocol):
    """Predicts which tier a prompt should enter at.

    Implementations must be **pure and fast** — the router runs before any model
    call, and a classifier that needs its own inference pass would defeat the
    purpose. Feature-vector-plus-linear-model is the intended shape; anything
    requiring a second runtime should be weighed against D6.
    """

    name: str

    def predict(self, features: PromptFeatures, prompt: str) -> Prediction: ...
