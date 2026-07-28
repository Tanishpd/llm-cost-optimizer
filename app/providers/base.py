"""Provider protocol — the seam every LLM backend implements.

New backends go here. The gateway talks to providers only through
:class:`Provider`, which is what keeps "swap the mock frontier for a real
provider" a config change rather than a code edit (invariant 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(Exception):
    """Base class. Every provider failure is one of the two below, so callers
    can decide *degrade* vs *fail* without matching on strings."""


class ProviderUnavailable(ProviderError):
    """The backend could not be reached, or a model is missing.

    Recoverable: the cascade should skip this tier and continue. Invariant 4 —
    degrade, don't fail.
    """

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix  # actionable remedy, surfaced in `reason` and /healthz


class ProviderTimeout(ProviderError):
    """The backend accepted the request but did not answer in time."""


@dataclass
class Completion:
    """One model response, normalized across backends.

    `logprobs` is empty for providers that don't expose them (the frontier
    tier). :func:`app.confidence.extract_metrics` maps that to metrics which
    force escalation rather than scoring an unmeasurable answer as confident.
    """

    content: str
    model: str
    tokens_in: int
    tokens_out: int
    logprobs: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    # Ollama reports a separate `load_duration`; tracked apart so cold-start
    # time doesn't silently pollute the latency percentiles (docs/API.md).
    load_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def was_cold_load(self) -> bool:
        """A model eviction happened. Expected with OLLAMA_MAX_LOADED_MODELS=2."""
        return self.load_ms > 100.0


@runtime_checkable
class Provider(Protocol):
    """What the cascade needs from any backend."""

    name: str

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> Completion:
        """Generate a completion. Raises ProviderUnavailable / ProviderTimeout."""
        ...

    async def health(self) -> tuple[bool, str]:
        """(reachable, detail). Must not raise."""
        ...
