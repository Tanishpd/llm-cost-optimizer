# API Contract

The HTTP surface the gateway exposes. This is the contract M1–M8 are built against — if the code and
this document disagree, one of them is a bug. Related: [PLAN.md](PLAN.md) for milestones,
[RESEARCH.md](RESEARCH.md) for the upstream API shapes this wraps.

Base URL: `http://localhost:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | Main entry point. OpenAI-compatible. |
| `/route` | POST | Routing decision only — no model is called. |
| `/metrics` | GET | JSON aggregates for the dashboard. |
| `/healthz` | GET | Liveness + per-dependency status. |
| `/dashboard` | GET | Self-contained HTML dashboard. |
| `/cache` | DELETE | Flush cache tiers (dev convenience). |

---

## POST /v1/chat/completions

Drop-in OpenAI-compatible. An existing client works unchanged by pointing `base_url` here.

### Request

```json
{
  "model": "gateway",
  "messages": [
    {"role": "system",  "content": "You are concise."},
    {"role": "user",    "content": "What is the capital of France?"}
  ],
  "temperature": 0.0,
  "top_p": 1.0,
  "max_tokens": 512,
  "stream": false
}
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `"gateway"` | Advisory. The gateway picks the real model. Reserved values below. |
| `messages` | array | required | Standard OpenAI roles. `system` supported; multi-turn supported. |
| `temperature` | float | `0.0` | Forwarded to whichever tier runs. **Part of the cache key.** |
| `top_p` | float | `1.0` | Forwarded. **Part of the cache key.** |
| `max_tokens` | int | `512` | Forwarded. **Part of the cache key.** |
| `stream` | bool | `false` | `true` returns 400 — see [Streaming](#streaming). |

**Reserved `model` values** — escape hatches that bypass routing, for benchmarking and debugging:

| Value | Effect |
|---|---|
| `gateway` | Normal cascade (default) |
| `gateway:small` | Force `local_small`, no gate, no escalation |
| `gateway:large` | Force `local_large`, no gate |
| `gateway:frontier` | Force the frontier tier |
| `gateway:nocache` | Run the cascade but skip T0 and T1 |

### Response

Standard OpenAI `chat.completion`, plus a non-standard `x_hlr` block. Clients that don't know about
`x_hlr` ignore it; the field is additive and breaks nothing.

```json
{
  "id": "chatcmpl-8f3a1c2e",
  "object": "chat.completion",
  "created": 1785155391,
  "model": "qwen2.5:3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Paris."},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
  "x_hlr": { ... }
}
```

Note `model` in the response is the **model that actually served it**, not the requested `"gateway"`.
That's deliberate: it makes the response self-describing in logs.

### The `x_hlr` block

Invariant 1 in [CLAUDE.md](../CLAUDE.md): **every** response carries this, including cache hits and
errors. The gateway's product is the visibility into its decisions.

```json
{
  "x_hlr": {
    "trace_id": "01J8XQ2M7K3P",
    "tier": "local_large",
    "path": ["local_small", "local_large"],
    "gates": [
      {"gate": "small", "confidence": 0.41, "threshold": 0.55, "verdict": "escalate",
       "metrics": {"geo_mean": 0.41, "p10_prob": 0.09, "mean_margin": 0.82,
                   "mean_norm_entropy": 0.61, "frac_low": 0.38, "n_tokens": 156}},
      {"gate": "large", "confidence": 0.68, "threshold": 0.62, "verdict": "accept",
       "metrics": {"geo_mean": 0.68, "p10_prob": 0.31, "mean_margin": 1.94,
                   "mean_norm_entropy": 0.34, "frac_low": 0.11, "n_tokens": 156}}
    ],
    "confidence": 0.68,
    "confidence_mode": "raw",
    "reason": "explanatory 'why' question, 41 tokens; 0.5B uncertain on causal chain",
    "router": {"entry_tier": "local_small", "score": 0.34,
               "features": {"n_tokens": 41, "has_code_fence": false, "reasoning_markers": 1,
                            "imperative_class": "explain", "arithmetic": false,
                            "entity_density": 0.07, "context_chars": 0}},
    "cache": {"exact_checked": true, "exact_hit": false,
              "semantic_checked": true, "semantic_hit": false,
              "similarity": 0.82, "guard_rejected": null},
    "latency_ms": {"route": 0.4, "cache_exact": 1.2, "cache_semantic": 18.9,
                   "local_small": 118.0, "local_large": 441.0, "total": 580.9},
    "tokens_in": 41,
    "tokens_out": 156,
    "cost_usd": 0.0,
    "cost_saved_usd": 0.00166,
    "baseline_cost_usd": 0.00166,
    "degraded": []
  }
}
```

| Field | Type | Meaning |
|---|---|---|
| `trace_id` | string | Propagated into telemetry and logs. Also returned as the `X-HLR-Trace-Id` header. |
| `tier` | enum | Who served it: `cache_exact` \| `cache_semantic` \| `local_small` \| `local_large` \| `frontier`. |
| `path` | array | **Every tier attempted, in order.** A request that walked all three models paid for all three. |
| `gates` | array | One entry per gate evaluated, with the full metric vector. Empty on cache hits. |
| `confidence` | float\|null | The serving tier's score. `null` for cache hits and the frontier tier. |
| `confidence_mode` | enum | `raw` \| `calibrated`. **Changes what `confidence` means** — see RESEARCH.md §3. |
| `reason` | string | Human-readable, never an enum. Invariant 2. |
| `router.*` | object | Entry-tier decision and the features behind it. |
| `cache.*` | object | Which tiers were consulted, similarity score, and which guard rejected a near-hit. |
| `latency_ms` | object | Per stage plus `total`. Keys present only for stages that ran. |
| `cost_usd` | float | Actual spend. `0.0` for cache and local tiers. |
| `baseline_cost_usd` | float | What an all-frontier policy would have spent on this request. |
| `cost_saved_usd` | float | `baseline_cost_usd − cost_usd`. The headline number, summed on the dashboard. |
| `degraded` | array | Non-fatal dependency failures, e.g. `["redis_unavailable"]`. Empty when healthy. |

`guard_rejected` names the specific semantic-cache guard that fired — `number_mismatch`,
`entity_mismatch`, `negation_mismatch`, `multi_turn`, `system_prompt_differs`, `too_long` — or `null`.
This is how you debug a cache that "should" have hit.

### Headers

Returned on every response so a proxy or log pipeline can route on them without parsing the body:

```
X-HLR-Trace-Id: 01J8XQ2M7K3P
X-HLR-Tier: local_large
X-HLR-Cost-Usd: 0.000000
X-HLR-Saved-Usd: 0.001660
```

### Streaming

`"stream": true` returns **400** with `unsupported_streaming`. This is a deliberate limitation, not an
oversight: the confidence gate cannot decide whether to escalate until the full response and its
logprobs exist, so streaming a local answer would mean streaming tokens that may then be discarded and
replaced. Deferred in PLAN.md with the resolution sketched (buffer locally, stream only post-decision).

---

## POST /route

Runs the pre-router and returns the decision **without calling any model**. Free and instant — useful
for testing routing policy and for building a labeled dataset.

Request: same shape as `/v1/chat/completions` (only `messages` is read).

```json
{
  "entry_tier": "local_small",
  "reason": "short factual question, 8 tokens, no reasoning markers",
  "score": 0.12,
  "features": {"n_tokens": 8, "has_code_fence": false, "reasoning_markers": 0,
               "imperative_class": "question", "arithmetic": false,
               "entity_density": 0.25, "context_chars": 0},
  "cache_preview": {"exact_hit": true, "semantic_hit": false, "similarity": null}
}
```

---

## GET /metrics

> The payload below is **illustrative** — plausible shape and internally consistent arithmetic, not
> measured output. Real numbers arrive with M6.

```json
{
  "window": "all_time",
  "requests": 1284,
  "tiers": {
    "served": {"cache_exact": 402, "cache_semantic": 117, "local_small": 511,
               "local_large": 198, "frontier": 56},
    "attempted": {"local_small": 690, "local_large": 239, "frontier": 56}
  },
  "router": {"entry": {"local_small": 690, "local_large": 60, "frontier": 15}},
  "cache": {"hit_rate": 0.404, "exact_hit_rate": 0.313, "semantic_hit_rate": 0.091,
            "guard_rejections": {"entity_mismatch": 14, "number_mismatch": 9,
                                 "negation_mismatch": 2, "too_long": 31},
            "entries": 867},
  "gates": {"small": {"evaluated": 690, "accepted": 511, "escalated": 179, "escalation_rate": 0.259},
            "large": {"evaluated": 239, "accepted": 198, "escalated": 41, "escalation_rate": 0.172}},
  "cost": {"actual_usd": 0.0891, "baseline_usd": 1.2437,
           "saved_usd": 1.1546, "saved_pct": 92.8},
  "latency_ms": {"cache_exact": {"p50": 1.1, "p95": 2.4},
                 "cache_semantic": {"p50": 17.8, "p95": 31.2},
                 "local_small": {"p50": 116.0, "p95": 189.0},
                 "local_large": {"p50": 438.0, "p95": 702.0},
                 "frontier": {"p50": 1980.0, "p95": 3110.0},
                 "end_to_end": {"p50": 121.0, "p95": 1840.0}},
  "tokens": {"in": 48211, "out": 129884},
  "model_loads": {"cold_loads": 7, "mean_load_ms": 1420.0},
  "degraded": []
}
```

`saved_pct` is the headline. `model_loads` is tracked separately so Ollama cold-start time
(`OLLAMA_MAX_LOADED_MODELS=2` means occasional evictions) doesn't silently pollute the latency
percentiles.

The counts must reconcile, and a test should assert it:

```
sum(tiers.served)                            == requests
router.entry.local_small                     == tiers.attempted.local_small
router.entry.local_large  + gates.small.escalated == tiers.attempted.local_large
router.entry.frontier     + gates.large.escalated == tiers.attempted.frontier
gates.X.accepted + gates.X.escalated         == gates.X.evaluated
cost.baseline_usd − cost.actual_usd          == cost.saved_usd
```

Verify against the example above: served sums to 1284 ✓; 60 + 179 = 239 ✓; 15 + 41 = 56 ✓;
1.2437 − 0.0891 = 1.1546 ✓, and 1.1546 / 1.2437 = 92.8% ✓.

---

## GET /healthz

`200` when the gateway can serve at all — **not** when everything is perfect. Invariant 4 is degrade,
don't fail: Redis being down disables caching but the gateway still answers, so that is `degraded`,
not unhealthy.

```json
{
  "status": "degraded",
  "dependencies": {
    "ollama":       {"ok": true,  "latency_ms": 3.1,
                     "models": {"qwen2.5:0.5b": true, "qwen2.5:3b": true, "nomic-embed-text": true}},
    "redis":        {"ok": false, "error": "Connection refused",
                     "impact": "caching disabled; gateway still serving"},
    "frontier":     {"ok": true,  "base_url": "http://localhost:8001/v1"}
  },
  "config": {"confidence_mode": "raw", "thresholds": {"small": 0.55, "large": 0.62},
             "semantic_enabled": true}
}
```

`status`: `ok` | `degraded` | `unhealthy`. Only `unhealthy` (no model tier reachable at all) returns
**503**.

---

## Errors

Errors also carry `x_hlr`, so a failure is as traceable as a success.

```json
{
  "error": {"message": "No model tier is reachable. Is `ollama serve` running?",
            "type": "upstream_unavailable", "code": "all_tiers_down"},
  "x_hlr": {"trace_id": "01J8XQ2M7K3P", "tier": null, "path": ["local_small", "local_large", "frontier"],
            "reason": "ollama connection refused; frontier connection refused",
            "degraded": ["ollama_unavailable", "frontier_unavailable"],
            "latency_ms": {"total": 84.2}}
}
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed body, empty `messages` |
| 400 | `unsupported_streaming` | `stream: true` |
| 404 | `model_not_found` | Reserved `model` value naming a tier that isn't configured |
| 503 | `all_tiers_down` | Neither Ollama nor the frontier is reachable |
| 504 | `upstream_timeout` | Every tier exceeded its timeout |

**Not errors — these degrade instead:** Redis unreachable (caching off), one local model missing
(that tier is skipped, `reason` names the `ollama pull` to run), embedder missing (semantic cache off).

---

## The frontier contract

What the gateway sends *upstream*, and therefore what `mock_frontier/` must faithfully implement.
Getting this right is what makes `FRONTIER_BASE_URL=https://api.openai.com/v1` a config change rather
than a rewrite (Invariant 8).

```http
POST {FRONTIER_BASE_URL}/chat/completions
Authorization: Bearer {FRONTIER_API_KEY}
Content-Type: application/json

{"model": "gpt-4o-mock", "messages": [...], "temperature": 0.0, "max_tokens": 512}
```

The mock must return a complete `chat.completion` object — **including a truthful `usage` block**,
since all cost math derives from it:

```json
{"id": "chatcmpl-mock-1a2b", "object": "chat.completion", "created": 1785155391,
 "model": "gpt-4o-mock",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."},
              "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 41, "completion_tokens": 156, "total_tokens": 197}}
```

Mock-only knobs (env): `MOCK_LATENCY_MS=1800`, `MOCK_JITTER_MS=400`, `MOCK_FAILURE_RATE=0.0`.
The failure rate exists to exercise the degradation paths in M8.

---

## Cost math

```
cost_usd          = (tokens_in/1e6 × FRONTIER_PRICE_IN) + (tokens_out/1e6 × FRONTIER_PRICE_OUT)
                    for the frontier tier; 0 for cache and local tiers
baseline_cost_usd = the same formula applied to every request, as if it had gone straight to frontier
cost_saved_usd    = baseline_cost_usd − cost_usd
```

Local tiers are counted as `$0` marginal. That is a deliberate simplification — electricity and
amortized hardware are real but not meaningfully attributable per request, and inventing a number
would make the headline savings figure less honest, not more. The README states this openly.

Accumulate in integer micro-USD or `Decimal`, never float (see CLAUDE.md conventions) — float
accumulation over thousands of requests visibly drifts.
