"""FastAPI gateway — the single entry point.

M1 scope: the skeleton. `/v1/chat/completions` is hardwired to the `local_large`
tier so the request path, cost accounting, and `x_hlr` block can be exercised
end to end. Caches (M2), routing (M3), and the confidence gates (M4) slot in
later without changing this file's shape — the response assembly already
carries every field they will populate.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.costs import TokenUsage, compute_cost, ns_to_ms
from app.providers.base import ProviderTimeout, ProviderUnavailable
from app.providers.ollama import OllamaProvider
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    DependencyHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    HlrMetadata,
)

logger = logging.getLogger("hlr")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own the pooled HTTP client and warm the models.

    One `AsyncClient` for the process — creating one per request would discard
    connection reuse and leak sockets under load.
    """
    settings = get_settings()
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            settings.ollama_timeout_s, connect=settings.ollama_connect_timeout_s
        ),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )
    ollama = OllamaProvider(
        client,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )

    app.state.settings = settings
    app.state.http = client
    app.state.ollama = ollama

    reachable, detail = await ollama.health()
    logger.info("ollama: %s (%s)", "up" if reachable else "DOWN", detail)

    if settings.warm_models_on_startup and reachable:
        # A 1-token call per chat model, so the first real request skips the
        # ~1.9 s cold load measured in M0. Never fatal.
        for model in (settings.small_model, settings.large_model):
            load_ms = await ollama.warm(model)
            logger.info("warmed %s (load %.0f ms)", model, load_ms)

    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(
    title="LLM Cost Optimizer",
    description="Routes each prompt to the cheapest model tier that can answer it.",
    version="0.1.0",
    lifespan=lifespan,
)


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _ollama(request: Request) -> OllamaProvider:
    return request.app.state.ollama


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """Every request gets a trace id, propagated into telemetry and returned as
    a header so a caller can correlate without parsing the body."""
    trace_id = request.headers.get("x-hlr-trace-id") or uuid.uuid4().hex[:12]
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-HLR-Trace-Id"] = trace_id
    return response


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    """OpenAI-compatible entry point.

    M1: hardwired to `local_large`. The `x_hlr` block is assembled in full even
    though most of it is empty until M2-M4, because invariant 1 says every
    response carries it — including the fields that are not populated yet.
    """
    settings = _settings(request)
    ollama = _ollama(request)
    trace_id = getattr(request.state, "trace_id", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    latency: dict[str, float] = {}
    degraded: list[str] = []

    if body.stream:
        # Deliberate, documented limitation: the gate cannot decide whether to
        # escalate until the full response and its logprobs exist, so streaming
        # would mean streaming tokens that may be discarded (D8).
        return _error(
            status=400,
            code="unsupported_streaming",
            error_type="invalid_request_error",
            message=(
                "stream=true is not supported: confidence gating needs the complete "
                "response before it can decide whether to escalate. See docs/DECISIONS.md D8."
            ),
            trace_id=trace_id,
            degraded=degraded,
            started=started,
        )

    tier = body.forced_tier or "local_large"
    model = settings.model_for(tier)

    if tier == "frontier":
        # The frontier provider arrives in M4; until then say so plainly rather
        # than pretending or 500ing.
        return _error(
            status=501,
            code="tier_not_implemented",
            error_type="not_implemented",
            message="The frontier tier is not wired up until M4. Use gateway:small or gateway:large.",
            trace_id=trace_id,
            degraded=degraded,
            started=started,
        )

    messages = [m.model_dump() for m in body.messages]
    try:
        completion = await ollama.chat(
            model=model,
            messages=messages,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            logprobs=True,
            top_logprobs=settings.top_logprobs,
        )
    except ProviderUnavailable as exc:
        degraded.append("ollama_unavailable")
        return _error(
            status=503,
            code="all_tiers_down",
            error_type="upstream_unavailable",
            message=f"{exc}{f' — {exc.fix}' if exc.fix else ''}",
            trace_id=trace_id,
            degraded=degraded,
            started=started,
            path=[tier],
        )
    except ProviderTimeout as exc:
        degraded.append("ollama_timeout")
        return _error(
            status=504,
            code="upstream_timeout",
            error_type="upstream_timeout",
            message=str(exc),
            trace_id=trace_id,
            degraded=degraded,
            started=started,
            path=[tier],
        )

    latency[tier] = completion.latency_ms
    if completion.was_cold_load:
        # Recorded separately so it doesn't distort the tier's latency stats.
        latency["model_load"] = completion.load_ms
    latency["total"] = round((time.perf_counter() - started) * 1000, 1)

    usage = TokenUsage(completion.tokens_in, completion.tokens_out)
    cost = compute_cost(
        served_by=tier,
        usage=usage,
        price_in_per_m=settings.frontier_price_in,
        price_out_per_m=settings.frontier_price_out,
    )

    hlr = HlrMetadata(
        trace_id=trace_id,
        tier=tier,
        path=[tier],
        confidence_mode=settings.confidence_mode,
        reason=f"M1 skeleton: all requests served by {tier} ({model}); routing lands in M3",
        latency_ms=latency,
        degraded=degraded,
        **cost.as_dict(),
    )

    response = ChatCompletionResponse.build(
        content=completion.content,
        model=completion.model,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        hlr=hlr,
        finish_reason=completion.finish_reason,
    )
    return JSONResponse(
        content=response.model_dump(),
        headers={
            "X-HLR-Tier": tier,
            "X-HLR-Cost-Usd": f"{hlr.cost_usd:.6f}",
            "X-HLR-Saved-Usd": f"{hlr.cost_saved_usd:.6f}",
        },
    )


def _error(
    *,
    status: int,
    code: str,
    error_type: str,
    message: str,
    trace_id: str,
    degraded: list[str],
    started: float,
    path: list[str] | None = None,
) -> JSONResponse:
    """Errors carry `x_hlr` too — a failure should be as traceable as a
    success (docs/API.md)."""
    payload = ErrorResponse(
        error=ErrorDetail(message=message, type=error_type, code=code),
        x_hlr=HlrMetadata(
            trace_id=trace_id,
            tier=None,
            path=path or [],
            reason=message,
            degraded=degraded,
            latency_ms={"total": round((time.perf_counter() - started) * 1000, 1)},
        ),
    )
    return JSONResponse(status_code=status, content=payload.model_dump())


@app.get("/healthz")
async def healthz(request: Request) -> Any:
    """200 when the gateway can serve at all — not when everything is perfect.

    Redis down disables caching but the gateway still answers, so that is
    `degraded`, not unhealthy (invariant 4). Only "no model tier reachable at
    all" is a 503.
    """
    settings = _settings(request)
    ollama = _ollama(request)

    started = time.perf_counter()
    reachable, detail = await ollama.health()
    probe_ms = round((time.perf_counter() - started) * 1000, 1)

    models: dict[str, bool] = {}
    if reachable:
        try:
            installed = await ollama.list_models()
            models = {
                m: (m in installed)
                for m in (settings.small_model, settings.large_model, settings.embed_model)
            }
        except Exception:  # noqa: BLE001
            models = {}

    dependencies = {
        "ollama": DependencyHealth(
            ok=reachable,
            latency_ms=probe_ms,
            error=None if reachable else detail,
            models=models or None,
        ),
        # Wired in M2 and M4 respectively; reported as not-yet-configured
        # rather than silently omitted.
        "redis": DependencyHealth(ok=False, error="not wired until M2", impact="caching disabled"),
        "frontier": DependencyHealth(
            ok=False, error="not wired until M4", base_url=settings.frontier_base_url
        ),
    }

    missing = [m for m, present in models.items() if not present]
    if not reachable:
        status = "unhealthy"
    elif missing:
        status = "degraded"
    else:
        status = "degraded"  # redis/frontier still pending in M1

    body = HealthResponse(
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        config={
            "milestone": "M1",
            "confidence_mode": settings.confidence_mode,
            "thresholds": {
                "small": settings.threshold_for("small"),
                "large": settings.threshold_for("large"),
            },
            "semantic_enabled": settings.semantic_enabled,
            "models": {
                "small": settings.small_model,
                "large": settings.large_model,
                "embed": settings.embed_model,
            },
            "missing_models": missing,
        },
    )
    return JSONResponse(status_code=503 if status == "unhealthy" else 200, content=body.model_dump())


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "llm-cost-optimizer",
        "milestone": "M1 — skeleton gateway",
        "endpoints": ["/v1/chat/completions", "/healthz", "/docs"],
        "note": "Routing (M3) and confidence gating (M4) are not wired yet; "
        "all requests are served by the local_large tier.",
    }
