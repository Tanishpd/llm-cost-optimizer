"""FastAPI gateway — the single entry point.

Current scope (M3): exact cache → pre-router → the chosen local tier.

The confidence gates and the frontier provider arrive in M4. Until then a
request the router sends to `frontier` returns 501 rather than being quietly
downgraded — saying "not built yet" beats pretending.

The `x_hlr` block has carried every field since M1, including the ones that
were empty then, so each milestone fills fields in rather than changing the
response contract.
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
from redis.asyncio import Redis

from app.cache.exact import CachedAnswer, ExactCache
from app.cache.keys import build_key
from app.config import Settings, get_settings
from app.costs import TokenUsage, compute_cost, micro_to_usd, price_micro_usd
from app.providers.base import ProviderTimeout, ProviderUnavailable
from app.providers.frontier import FrontierProvider
from app.providers.ollama import OllamaProvider
from app.cascade import build_tiers, entry_index, execute
from app.router.policy import decide
from app.telemetry import RequestRecord, Telemetry
from app.schemas import (
    CacheRecord,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DependencyHealth,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    GateRecord,
    HlrMetadata,
    RouteDecisionResponse,
    RouterRecord,
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

    # Redis is optional by design: if it cannot be reached the gateway runs with
    # caching disabled rather than refusing to start (invariant 4).
    redis: Redis | None = None
    try:
        redis = Redis.from_url(settings.redis_url, socket_connect_timeout=2, decode_responses=True)
        await redis.ping()
        logger.info("redis: up at %s", settings.redis_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis: DOWN (%s) — caching disabled, still serving", exc)
        redis = None

    app.state.settings = settings
    app.state.http = client
    app.state.ollama = ollama
    app.state.redis = redis
    app.state.exact_cache = ExactCache(
        redis, ttl_local=settings.cache_ttl_local, ttl_frontier=settings.cache_ttl_frontier
    )
    app.state.telemetry = Telemetry(redis)
    app.state.frontier = FrontierProvider(
        client,
        base_url=settings.frontier_base_url,
        api_key=settings.frontier_api_key,
        timeout_s=settings.frontier_timeout_s,
    )

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
        if redis is not None:
            await redis.aclose()


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


def _exact_cache(request: Request) -> ExactCache:
    return request.app.state.exact_cache


def _telemetry(request: Request) -> Telemetry:
    return request.app.state.telemetry


def _frontier(request: Request) -> FrontierProvider:
    return request.app.state.frontier


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

    Order: streaming refusal → pre-router → exact cache → chosen tier.

    The cache lookup needs the tier's model in its key (an answer from the 0.5B
    must never be served for a 3B request), so routing happens first. That
    costs microseconds and calls no model.
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

    cache = _exact_cache(request)
    cache_record = CacheRecord()

    # ---- pre-router ------------------------------------------------------
    # Cheap heuristics pick an *entry point*; no model is called. The router is
    # allowed to guess low, because the gates (M4) catch a weak answer. It is
    # not allowed to guess high casually — frontier entry is ungated.
    route_started = time.perf_counter()
    decision = decide(
        body.user_prompt,
        system_prompt=body.system_prompt,
        is_multi_turn=body.is_multi_turn,
        small_max_tokens=settings.router_small_max_tokens,
        escalate_min_tokens=settings.router_escalate_min_tokens,
    )
    latency["route"] = round((time.perf_counter() - route_started) * 1000, 1)

    # A reserved model name overrides the router entirely — that is what the
    # escape hatch is for (benchmarking, debugging one tier).
    tier = body.forced_tier or decision.entry_tier
    model = settings.model_for(tier)
    router_record = RouterRecord(
        entry_tier=tier,
        score=decision.score,
        reason=(
            decision.reason
            if body.forced_tier is None
            else f"forced to {tier} by model={body.model!r}; router would have chosen "
                 f"{decision.entry_tier} ({decision.reason})"
        ),
        features=decision.features.as_dict(),
    )

    # ---- T0: exact cache ------------------------------------------------
    # The key needs the tier's model, so this runs after the router — but the
    # router costs microseconds and calls nothing, so a hit is still served
    # without touching a model.
    key = build_key(
        prompt=body.user_prompt,
        model=model,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=body.max_tokens,
        system_prompt=body.system_prompt,
    )
    if not body.bypass_cache:
        cache_started = time.perf_counter()
        hit = await cache.get(key)
        latency["cache_exact"] = round((time.perf_counter() - cache_started) * 1000, 1)
        cache_record.exact_checked = True
        if not cache.available:
            degraded.append("redis_unavailable")
        if hit is not None:
            cache_record.exact_hit = True
            latency["total"] = round((time.perf_counter() - started) * 1000, 1)
            hlr = HlrMetadata(
                trace_id=trace_id,
                tier="cache_exact",
                path=["cache_exact"],
                confidence_mode=settings.confidence_mode,
                reason=f"exact cache hit; answer originally from {hit.tier} ({hit.model})",
                router=router_record,
                cache=cache_record,
                latency_ms=latency,
                degraded=degraded,
                tokens_in=hit.tokens_in,
                tokens_out=hit.tokens_out,
                # A cache hit costs nothing, but the baseline still prices it as
                # if it had gone to the frontier — that is the "what would this
                # traffic cost with no gateway" question (D13).
                cost_usd=0.0,
                baseline_cost_usd=float(micro_to_usd(hit.baseline_micro_usd)),
                cost_saved_usd=float(micro_to_usd(hit.baseline_micro_usd)),
            )
            await _telemetry(request).record(
                RequestRecord(
                    trace_id=trace_id,
                    tier="cache_exact",
                    path=["cache_exact"],
                    reason=hlr.reason,
                    latency_ms=latency,
                    tokens_in=hit.tokens_in,
                    tokens_out=hit.tokens_out,
                    cost_micro_usd=0,
                    baseline_micro_usd=hit.baseline_micro_usd,
                    cache_hit="exact",
                    degraded=degraded,
                    created_at=time.time(),
                )
            )
            response = ChatCompletionResponse.build(
                content=hit.content,
                model=hit.model,
                tokens_in=hit.tokens_in,
                tokens_out=hit.tokens_out,
                hlr=hlr,
                finish_reason=hit.finish_reason,
            )
            return JSONResponse(
                content=response.model_dump(),
                headers={
                    "X-HLR-Tier": "cache_exact",
                    "X-HLR-Cost-Usd": "0.000000",
                    "X-HLR-Saved-Usd": f"{hlr.cost_saved_usd:.6f}",
                },
            )

    # ---- the cascade -----------------------------------------------------
    # Walk the tier chain from the router's entry point. Each tier's gate
    # decides whether its answer stands or the walk continues.
    messages = [m.model_dump() for m in body.messages]
    tiers = build_tiers(
        small_model=settings.small_model,
        large_model=settings.large_model,
        frontier_model=settings.frontier_model,
        threshold_small=settings.threshold_for("small"),
        threshold_large=settings.threshold_for("large"),
    )

    async def run_local(model_name: str, msgs: list[dict[str, str]]) -> Any:
        return await ollama.chat(
            model=model_name,
            messages=msgs,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            logprobs=True,
            top_logprobs=settings.top_logprobs,
        )

    async def run_frontier(model_name: str, msgs: list[dict[str, str]]) -> Any:
        return await _frontier(request).chat(
            model=model_name,
            messages=msgs,
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
        )

    # A forced tier pins the walk to exactly that rung: no gate, no escalation.
    # The escape hatch exists to benchmark one tier in isolation.
    start = entry_index(tiers, tier)
    walk = tiers[start : start + 1] if body.forced_tier else tiers

    outcome = await execute(
        walk,
        start=0 if body.forced_tier else start,
        messages=messages,
        runners={
            "local_small": run_local,
            "local_large": run_local,
            "frontier": run_frontier,
        },
        confidence_mode=settings.confidence_mode,
    )

    latency.update(outcome.latency_ms)
    degraded.extend(outcome.degraded)

    # Checked directly rather than via `outcome.served` so the type narrows.
    if outcome.completion is None or outcome.tier is None:
        exc = outcome.error
        is_timeout = isinstance(exc, ProviderTimeout)
        latency["total"] = round((time.perf_counter() - started) * 1000, 1)
        return _error(
            status=504 if is_timeout else 503,
            code="upstream_timeout" if is_timeout else "all_tiers_down",
            error_type="upstream_timeout" if is_timeout else "upstream_unavailable",
            message=outcome.reason
            + (f" — {exc.fix}" if isinstance(exc, ProviderUnavailable) and exc.fix else ""),
            trace_id=trace_id,
            degraded=degraded,
            started=started,
            path=list(outcome.path),
        )

    completion = outcome.completion
    tier: str = outcome.tier  # the tier that actually served, not the entry point
    latency["total"] = round((time.perf_counter() - started) * 1000, 1)

    usage = TokenUsage(completion.tokens_in, completion.tokens_out)
    cost = compute_cost(
        served_by=tier,
        usage=usage,
        price_in_per_m=settings.frontier_price_in,
        price_out_per_m=settings.frontier_price_out,
    )

    # ---- write-through --------------------------------------------------
    # Every served answer is cached, so the next identical request is a T0 hit.
    if not body.bypass_cache:
        await cache.set(
            key,
            CachedAnswer(
                content=completion.content,
                model=completion.model,
                tier=tier,
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                baseline_micro_usd=cost.baseline_micro_usd,
                finish_reason=completion.finish_reason,
                created_at=time.time(),
                params={
                    "temperature": body.temperature,
                    "top_p": body.top_p,
                    "max_tokens": body.max_tokens,
                    "system_hash": key.system_hash,
                },
            ),
        )

    # Every gate evaluated on the way, with its full metric vector — this is
    # how a caller sees *why* the cascade stopped where it did.
    gate_records = [
        GateRecord(
            gate=v.gate,  # type: ignore[arg-type]
            confidence=round(v.confidence, 6),
            threshold=v.threshold,
            verdict=v.verdict,
            reason=v.reason,
            metrics=v.metrics.as_dict(),
        )
        for v in outcome.gates
    ]

    hlr = HlrMetadata(
        trace_id=trace_id,
        tier=tier,  # type: ignore[arg-type]
        # Every tier attempted, in order. A request that walked 0.5B -> 3B ->
        # frontier paid for all three, and this says so rather than hiding it.
        path=list(outcome.path),
        gates=gate_records,
        confidence=outcome.confidence,
        confidence_mode=settings.confidence_mode,
        reason=outcome.reason,
        router=router_record,
        cache=cache_record,
        latency_ms=latency,
        degraded=degraded,
        **cost.as_dict(),
    )

    await _telemetry(request).record(
        RequestRecord(
            trace_id=trace_id,
            tier=tier,
            path=list(outcome.path),
            reason=hlr.reason,
            latency_ms=latency,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_micro_usd=cost.cost_micro_usd,
            baseline_micro_usd=cost.baseline_micro_usd,
            entry_tier=router_record.entry_tier,
            confidence=outcome.confidence,
            gates=[g.model_dump() for g in gate_records],
            cold_load_ms=completion.load_ms if completion.was_cold_load else 0.0,
            degraded=degraded,
            created_at=time.time(),
        )
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

    redis_ok, redis_detail = await _exact_cache(request).ping()
    frontier_ok, frontier_detail = await _frontier(request).health()
    dependencies = {
        "ollama": DependencyHealth(
            ok=reachable,
            latency_ms=probe_ms,
            error=None if reachable else detail,
            models=models or None,
        ),
        "redis": DependencyHealth(
            ok=redis_ok,
            error=None if redis_ok else redis_detail,
            impact=None if redis_ok else "caching disabled; gateway still serving",
        ),
        "frontier": DependencyHealth(
            ok=frontier_ok,
            error=None if frontier_ok else frontier_detail,
            impact=None if frontier_ok else "escalation unavailable; best local answer returned",
            base_url=settings.frontier_base_url,
        ),
    }

    missing = [m for m, present in models.items() if not present]
    if not reachable:
        # The only real failure: no model tier can be reached at all.
        status = "unhealthy"
    elif missing or not redis_ok or not frontier_ok:
        status = "degraded"
    else:
        status = "ok"

    body = HealthResponse(
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        config={
            "milestone": "M4",
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


@app.post("/route")
async def route_only(body: ChatCompletionRequest, request: Request) -> RouteDecisionResponse:
    """The routing decision **without calling any chat model** — free and fast.

    Useful for testing policy and for building a labelled dataset for M6.

    One caveat, stated rather than hidden: `cache_preview.exact_hit` costs a
    Redis lookup. No generation happens either way.
    """
    settings = _settings(request)
    decision = decide(
        body.user_prompt,
        system_prompt=body.system_prompt,
        is_multi_turn=body.is_multi_turn,
        small_max_tokens=settings.router_small_max_tokens,
        escalate_min_tokens=settings.router_escalate_min_tokens,
    )
    preview = CacheRecord()
    if not body.bypass_cache:
        key = build_key(
            prompt=body.user_prompt,
            model=settings.model_for(decision.entry_tier),
            temperature=body.temperature,
            top_p=body.top_p,
            max_tokens=body.max_tokens,
            system_prompt=body.system_prompt,
        )
        preview.exact_checked = True
        preview.exact_hit = await _exact_cache(request).get(key) is not None

    return RouteDecisionResponse(
        entry_tier=decision.entry_tier,  # type: ignore[arg-type]
        reason=decision.reason,
        score=round(decision.score, 4),
        features=decision.features.as_dict(),
        cache_preview=preview,
    )


@app.get("/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    """Aggregates for the dashboard. O(1) — counters, not a scan over the stream.

    Returns structurally complete zeros when Redis is down, so a dashboard
    renders "no data" rather than crashing.
    """
    return await _telemetry(request).metrics()


@app.get("/metrics/recent")
async def recent_requests(request: Request, count: int = 20) -> dict[str, Any]:
    """The live request log — newest first."""
    return {"requests": await _telemetry(request).recent(count=min(max(count, 1), 200))}


@app.delete("/metrics")
async def reset_metrics(request: Request) -> dict[str, Any]:
    """Clear telemetry. Dev convenience; scoped to the telemetry keys only."""
    return {"cleared_keys": await _telemetry(request).reset()}


@app.delete("/cache")
async def flush_cache(request: Request) -> dict[str, Any]:
    """Drop every exact-cache entry. Dev convenience — scoped to the
    `hlr:exact:` prefix, so it cannot touch anything else in the database."""
    removed = await _exact_cache(request).flush()
    return {"flushed": removed, "tier": "cache_exact"}


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "llm-cost-optimizer",
        "milestone": "M4 — confidence gates + frontier escalation",
        "endpoints": [
            "/v1/chat/completions", "/route", "/healthz", "/metrics", "/metrics/recent",
            "/cache", "/docs",
        ],
        "note": "Full cascade: exact cache -> pre-router -> qwen2.5:0.5b -> [gate] -> "
        "qwen2.5:3b -> [gate] -> frontier. Semantic cache lands in M5.",
    }
