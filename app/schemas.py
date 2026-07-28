"""Pydantic v2 models for the HTTP surface.

The contract is docs/API.md — if this file and that document disagree, one of
them is a bug. Field names and nullability here are taken from it directly.

`x_hlr` ("hybrid LLM router") is the routing-metadata block that rides on every
response, including cache hits and errors. Invariant 1: a response without it is
a bug, because the visibility into the gateway's decisions *is* the product.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["cache_exact", "cache_semantic", "local_small", "local_large", "frontier"]
EntryTier = Literal["local_small", "local_large", "frontier"]
GateId = Literal["small", "large"]
Verdict = Literal["accept", "escalate"]

# Reserved `model` values that bypass routing — for benchmarking and debugging.
RESERVED_MODELS = {
    "gateway",
    "gateway:small",
    "gateway:large",
    "gateway:frontier",
    "gateway:nocache",
}


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    """OpenAI's `stream_options`. Only `include_usage` is meaningful here."""

    model_config = ConfigDict(extra="allow")

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible. An existing client works by changing `base_url`."""

    model_config = ConfigDict(extra="allow")  # tolerate unknown OpenAI fields

    model: str = "gateway"
    messages: list[Message] = Field(min_length=1)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    # Supported via buffer-then-stream: the cascade runs to completion, then
    # the finished answer is re-emitted as SSE. It does NOT lower
    # time-to-first-token — see app/streaming.py and D8.
    stream: bool = False
    stream_options: StreamOptions | None = None

    @property
    def wants_usage_chunk(self) -> bool:
        return bool(self.stream_options and self.stream_options.include_usage)

    @property
    def user_prompt(self) -> str:
        """Last user message — what the router and cache key operate on."""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""

    @property
    def system_prompt(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def is_multi_turn(self) -> bool:
        """Multi-turn conversations are never semantic-cached (D15) — the
        embedding of the last message flattens away the context that makes the
        answer correct."""
        return sum(1 for m in self.messages if m.role in ("user", "assistant")) > 1

    @property
    def forced_tier(self) -> EntryTier | None:
        """Reserved-model escape hatch, or None for normal routing."""
        return {
            "gateway:small": "local_small",
            "gateway:large": "local_large",
            "gateway:frontier": "frontier",
        }.get(self.model)  # type: ignore[return-value]

    @property
    def bypass_cache(self) -> bool:
        return self.model == "gateway:nocache"


# --------------------------------------------------------------------------- #
# x_hlr metadata
# --------------------------------------------------------------------------- #


class GateRecord(BaseModel):
    """One gate evaluation. Gate ids are `small` (local_small -> local_large)
    and `large` (local_large -> frontier); prose elsewhere says "gate 1/2"."""

    gate: GateId
    confidence: float
    threshold: float
    verdict: Verdict
    reason: str
    metrics: dict[str, float | int] = Field(default_factory=dict)


class RouterRecord(BaseModel):
    entry_tier: EntryTier
    score: float
    reason: str
    features: dict[str, Any] = Field(default_factory=dict)


class CacheRecord(BaseModel):
    exact_checked: bool = False
    exact_hit: bool = False
    semantic_checked: bool = False
    semantic_hit: bool = False
    similarity: float | None = None
    # Names the specific guard that rejected a near-hit: number_mismatch,
    # entity_mismatch, negation_mismatch, multi_turn, system_prompt_differs,
    # too_long. This is how you debug a cache that "should" have hit.
    guard_rejected: str | None = None


class HlrMetadata(BaseModel):
    """The block on every response. See docs/API.md for field semantics."""

    trace_id: str
    tier: Tier | None
    # Every tier attempted, in order. A request that walked 0.5B -> 3B ->
    # frontier paid for all three, and this says so rather than hiding it.
    path: list[str] = Field(default_factory=list)
    gates: list[GateRecord] = Field(default_factory=list)
    confidence: float | None = None
    confidence_mode: Literal["raw", "calibrated"] = "raw"
    reason: str = ""
    router: RouterRecord | None = None
    cache: CacheRecord = Field(default_factory=CacheRecord)
    # Keys present only for stages that ran; `total` is wall-clock and may
    # exceed the sum of stages by gateway overhead.
    latency_ms: dict[str, float] = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0
    # Non-fatal dependency failures, e.g. ["redis_unavailable"]. Empty when healthy.
    degraded: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str = "stop"
    logprobs: None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Standard OpenAI `chat.completion` plus the additive `x_hlr` block.
    Clients that don't know about `x_hlr` ignore it; nothing breaks."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    # The model that actually served it, not the requested "gateway" — makes
    # the response self-describing in logs.
    model: str
    choices: list[Choice]
    usage: Usage
    x_hlr: HlrMetadata

    @classmethod
    def build(
        cls,
        *,
        content: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        hlr: HlrMetadata,
        finish_reason: str = "stop",
    ) -> ChatCompletionResponse:
        return cls(
            model=model,
            choices=[Choice(message=ResponseMessage(content=content), finish_reason=finish_reason)],
            usage=Usage(
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
            ),
            x_hlr=hlr,
        )


# --------------------------------------------------------------------------- #
# Errors and health
# --------------------------------------------------------------------------- #


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    """Errors carry `x_hlr` too, so a failure is as traceable as a success."""

    error: ErrorDetail
    x_hlr: HlrMetadata


class RouteDecisionResponse(BaseModel):
    """`POST /route` — the decision without calling any chat model."""

    entry_tier: EntryTier
    reason: str
    score: float
    features: dict[str, Any]
    cache_preview: CacheRecord


class DependencyHealth(BaseModel):
    ok: bool
    latency_ms: float | None = None
    error: str | None = None
    impact: str | None = None
    models: dict[str, bool] | None = None
    base_url: str | None = None


class HealthResponse(BaseModel):
    """200 when the gateway can serve at all — not when everything is perfect.
    Redis down disables caching but the gateway still answers, so that is
    `degraded`, not unhealthy (invariant 4)."""

    status: Literal["ok", "degraded", "unhealthy"]
    dependencies: dict[str, DependencyHealth]
    config: dict[str, Any]
