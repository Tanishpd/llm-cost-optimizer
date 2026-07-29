# API Contract

The HTTP surface the gateway exposes. If the code and this document disagree, one of them is a bug. Related: [PLAN.md](PLAN.md) for milestones,
[RESEARCH.md](RESEARCH.md) for the upstream API shapes this wraps.

Base URL: `http://localhost:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | Main entry point. OpenAI-compatible. |
| `/route` | POST | Routing decision only — no model is called. |
| `/metrics` | GET | JSON aggregates for the dashboard. |
| `/metrics/recent` | GET | Live request log, newest first. |
| `/dashboard` | GET | Self-contained HTML dashboard. |
| `/cache` | DELETE | Flush both cache tiers (dev convenience). |
| `/healthz` | GET | Full dependency detail. **Public.** |
| `/livez` | GET | Liveness — is the process up. **Public.** |
| `/readyz` | GET | Readiness — can this instance serve *now*. **Public.** |

---

## Authentication

Send an API key as either header:

```
Authorization: Bearer sk-...
X-API-Key: sk-...
```

Keys come from `GATEWAY_API_KEYS` (comma-separated) and are compared in constant
time. **An empty `GATEWAY_API_KEYS` means the gateway runs open** — acceptable
on a loopback bind, and startup warns loudly. `REQUIRE_AUTH=true` turns an open
start into a hard failure, because a config that silently disables auth while
claiming to enforce it is worse than a crash.

`/healthz`, `/livez`, `/readyz`, `/`, and the OpenAPI routes are **public** — a
load balancer has no credentials, and none of them expose traffic or cost data.
Everything else, `/metrics` and `/dashboard` included, requires a key.

### Limits

Applied before any model runs, so an abusive request costs a string comparison
rather than a generation:

| Limit | Setting | Exceeded → |
|---|---|---|
| Request body bytes | `MAX_BODY_BYTES` (256 000) | **413** `request_too_large` |
| Prompt characters | `MAX_PROMPT_CHARS` (32 000) | **413** `prompt_too_long` |
| Message count | `MAX_MESSAGES` (50) | **413** `too_many_messages` |
| Requests per minute | `RATE_LIMIT_PER_MINUTE` (60) | **429** `rate_limited` + `Retry-After` |
| Output tokens | `MAX_OUTPUT_TOKENS` (4096) | **clamped**, not rejected |

`max_tokens` is clamped rather than refused: a caller asking for more almost
always means "as much as you'll give me". The clamp appears in
`x_hlr.degraded` as `max_tokens_clamped`, so nobody is misled about what they
got.

The rate-limit window lives in Redis (a sorted set behind one Lua script), so
it is shared across workers. If Redis is unreachable it falls back to
per-process state and says so — the effective limit then multiplies by worker
count, which is worse than shared and far better than none.
`/healthz` reports `config.rate_limit_distributed` so you can tell which is live.

Successful responses carry `X-RateLimit-Remaining`.

### Tenants and budgets

A key authenticates; a **tenant** pays. Without `TENANTS` every key draws on the
same frontier spend — that is the shipped default and changes nothing.

```bash
GATEWAY_API_KEYS=sk-acme-1,sk-globex-1
TENANTS='{"acme":   {"keys": ["sk-acme-1"],   "daily_budget_usd": "5.00"},
          "globex": {"keys": ["sk-globex-1"], "daily_budget_usd": "1.00"}}'
```

JSON rather than a delimited string on purpose: API keys are opaque and may
contain `:` or `=`, so an inline separator eventually mis-splits key material —
and the failure is silent, because the key still authenticates while the spend
lands on the wrong account ([D22](DECISIONS.md)).

Keys not listed under a tenant belong to `default`; an open gateway bills
`anonymous`. Both take `DEFAULT_DAILY_BUDGET_USD`, `0` (unlimited) by default.

**Windows are UTC calendar days**, and only the frontier tier costs money — so
local answers and cache hits never consume budget.

**Over budget degrades; it does not reject.** The frontier is dropped from the
cascade and the request is served locally:

```json
{
  "tier": "local_large",
  "path": ["local_large"],
  "degraded": ["budget_exhausted:acme"],
  "reason": "daily budget exhausted: $0.000628 of $0.000500 spent today (UTC); frontier tier withheld, so: served by local_large: confidence 0.78 at or above 0.62",
  "budget": {"tenant": "acme", "spent_usd": 0.000628, "limit_usd": 0.0005,
             "remaining_usd": 0.0, "enforced": true}
}
```

The one exception is a caller who pinned `model: "gateway:frontier"` — they
removed the fallback themselves, so they get **402** ([D23](DECISIONS.md)).

**The cap is approximate by one request, and the approximation has a number.**
The budget is read before the call and recorded after, because cost is not
knowable until the provider replies. The overshoot is bounded by
`MAX_OUTPUT_TOKENS × FRONTIER_PRICE_OUT` — about **$0.04** at the defaults, and
startup logs the exact figure ([D24](DECISIONS.md)).

**If Redis is unreachable, budgets are not enforced.** Failing closed would turn
a cache outage into a total frontier outage. It is impossible to miss:
`degraded` carries `budget_unavailable`, `budget.enforced` is `false`,
`/healthz` reports `budgets_enforced: false`, and startup warns
([D25](DECISIONS.md)).

`GET /metrics` carries per-tenant attribution under its own window, because a
daily cap is a daily question and the rest of `/metrics` is all-time:

```json
"tenants": {"window": "utc_day",
            "usage": {"acme": {"spent_usd": 0.000628, "limit_usd": 0.0005,
                               "remaining_usd": 0.0, "enforced": true}}}
```

#### The gateway's own ceiling

Separate from tenant budgets, and **armed by default** at `MAX_DAILY_SPEND_USD=10.00`.

Invariant 4 says *"Ollama down → escalate everything"*, which keeps the gateway
answering and, at the shipped rate limit, spends roughly **$138/day per client**
while doing it. Tenant budgets can bound that but are opt-in, so before this the
protection existed and was not armed. A business policy may default to off; a
safety limit may not.

It outranks a tenant's own allowance — an operator limit is not something a
caller can be *within* — so a tenant with headroom is still refused, and
`x_hlr.budget` keeps reporting their true position while `degraded` carries
`spend_ceiling_reached`.

When it binds, requests degrade to the local tiers exactly as a tenant budget
does. If the local tiers are also down — the usual reason it binds at all —
there is nothing cheaper to fall back to and the request gets a **503**, not a
402: the caller's account is fine and they can do nothing about it.

`/metrics` reports it under the reserved `__all__` bucket, alongside the
tenants, so gateway-wide spend is never mistaken for an account's.

### Audit trail

Every request writes one JSON line to `AUDIT_PATH`, **including refusals** — a
402 is a spend decision, and omitting it would leave the most consequential ones
invisible in the one place built to record them.

```
audit/audit-8961.jsonl
{"ts":1785258422.7, "trace_id":"b6ee310fb7af", "tenant":"acme",
 "principal":"777a832b78f0", "tier":"local_large", "path":["local_large"],
 "cost_micro_usd":0, "degraded":["budget_exhausted:acme"],
 "reason":"daily budget exhausted: ...", "gates":[...]}
```

`principal` is a blake2b digest — the raw key is never written. One file per
worker PID (a line can exceed `PIPE_BUF`, above which concurrent `O_APPEND` is
not atomic), rotated at `AUDIT_MAX_BYTES` keeping `AUDIT_KEEP_FILES` siblings.
`cat audit-*.jsonl` reassembles them.

This is **not** `/metrics/recent`, which reads a Redis stream capped at 10k
entries. That answers "what is happening now"; this answers "why did request
`abc123` cost $0.04 last Tuesday", and survives restarts ([D26](DECISIONS.md)).

Writes never touch the request path — records are queued and drained by a
background task. A full queue **drops** rather than blocking, and `/healthz`
reports the count, because a trail with unreported holes would be worse than no
trail at all.

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
| `temperature` | float | `DEFAULT_TEMPERATURE` (`0.0`) | Forwarded to whichever tier runs. **Part of the cache key.** |
| `top_p` | float | `1.0` | Forwarded. **Part of the cache key.** |
| `max_tokens` | int | `DEFAULT_MAX_TOKENS` (`512`) | Forwarded, then clamped to `MAX_OUTPUT_TOKENS`. **Part of the cache key.** |
| `stream` | bool | `false` | `true` returns SSE — see [Streaming](#streaming). Does **not** lower time-to-first-token. |
| `stream_options` | object | `null` | `{"include_usage": true}` appends a final usage frame. Only meaningful with `stream: true`. |

The defaults for `temperature` and `max_tokens` are **operator policy**, not
constants: omit either and the gateway substitutes its configured default. An
explicit value always wins. Because both are part of the cache key, `/route`
applies the same defaults as `/v1/chat/completions` — otherwise its `exact_hit`
preview would be computed from a different key than the real lookup.

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

> **Illustrative.** Every example payload in this document uses invented values to show field layout.
> No run has produced them. This one and the `x_hlr` example below are the *same* request, so their
> numbers line up.

```json
{
  "id": "chatcmpl-8f3a1c2e",
  "object": "chat.completion",
  "created": 1785155391,
  "model": "qwen2.5:3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Sunlight refracts as it enters each raindrop, and ..."},
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": {"prompt_tokens": 41, "completion_tokens": 156, "total_tokens": 197},
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
       "metrics": {"geo_mean": 0.41, "min_prob": 0.04, "p10_prob": 0.09, "mean_margin": 0.82,
                   "mean_norm_entropy": 0.61, "frac_low": 0.38, "n_tokens": 94}},
      {"gate": "large", "confidence": 0.68, "threshold": 0.62, "verdict": "accept",
       "metrics": {"geo_mean": 0.68, "min_prob": 0.18, "p10_prob": 0.31, "mean_margin": 1.94,
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
| `gates` | array | One entry per gate evaluated, with the full metric vector. Empty on cache hits. Gate ids are **`small`** (gate 1, `local_small` → `local_large`) and **`large`** (gate 2, `local_large` → `frontier`). Prose elsewhere calls them "gate 1"/"gate 2"; the wire format uses these ids. |
| `confidence` | float\|null | The serving tier's score. `null` for cache hits and the frontier tier. |
| `confidence_mode` | enum | `raw` \| `calibrated`. **Changes what `confidence` means** — see RESEARCH.md §3. |
| `reason` | string | Human-readable, never an enum. Invariant 2. |
| `router.entry_tier` | enum | Where the cascade was entered: `local_small` \| `local_large` \| `frontier`. Same vocabulary as `tier`, minus the cache tiers — the router never chooses a cache. |
| `router.*` | object | Score and the features behind the decision. |
| `latency_ms.total` | float | Wall-clock. Exceeds the sum of the stage keys by a small amount of gateway overhead — do not assert exact equality. |
| `cache.*` | object | Which tiers were consulted, similarity score, and which guard rejected a near-hit. |
| `latency_ms` | object | Per stage plus `total`. Keys present only for stages that ran. |
| `tokens_in` | int | Prompt tokens, counted **once** per request — not once per attempted tier. |
| `tokens_out` | int | The **serving** tier's completion tokens. Discarded intermediate completions (e.g. the 94 tokens the small tier emitted before gate 1 escalated) appear in `gates[].metrics.n_tokens` and in `/metrics` throughput, not here. |
| `cost_usd` | float | Actual spend. `0.0` for cache and local tiers. |
| `baseline_cost_usd` | float | What an all-frontier policy would have spent on this request: `tokens_in` + `tokens_out` at frontier prices. |
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

`"stream": true` returns `text/event-stream` in the OpenAI `chat.completion.chunk` format. Any
OpenAI-compatible client works unchanged.

**It does not lower time-to-first-token, and is not meant to.** The confidence gate cannot decide
whether to escalate until the complete answer and its logprobs exist, so the cascade runs to
completion first and the finished answer is then re-emitted as frames. This is *buffer-then-stream*
([D8](DECISIONS.md)) — the feature is client compatibility, not latency.

Frame order:

| # | Frame | Carries |
|---|---|---|
| 1 | role | `delta: {"role": "assistant"}` |
| 2..n | content | `delta: {"content": "..."}` — concatenate to rebuild the answer exactly |
| n+1 | terminal | `finish_reason`, **and the full `x_hlr` block** |
| n+2 | usage *(optional)* | `choices: []` and `usage`, only with `stream_options.include_usage` |
| last | `data: [DONE]` | the sentinel every client watches for |

`x_hlr` rides on the **terminal** frame, not the first: `path` is not final until the cascade is, and
a request that walked 0.5B → 3B → frontier only knows its full path at the end. Invariant 1 still
holds — every response carries the block.

```
data: {"id":"chatcmpl-271c539f071d","object":"chat.completion.chunk","created":1785257451,
       "model":"qwen2.5:0.5b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {...,"choices":[{"index":0,"delta":{"content":"The capital of France "},"finish_reason":null}]}

data: {...,"choices":[{"index":0,"delta":{"content":"is Paris."},"finish_reason":null}]}

data: {...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"x_hlr":{...}}

data: [DONE]
```

Two consequences of buffering, both useful:

- **Errors are still ordinary HTTP errors.** Everything that can fail has already failed before the
  first byte, so a failure is a 4xx/5xx with a JSON body — never a truncated stream.
- **The `X-HLR-*` headers survive.** The routing decision is final when the response starts, so it
  rides in the headers too.

`STREAM_CHUNK_CHARS` (default `24`) sets how much text each content frame carries. It only trades
frame smoothness against ~200 bytes of JSON envelope per frame; `0` sends the answer in one frame.

---

## POST /route

Runs the pre-router and cache lookups **without calling any chat model** — no generation cost.
Useful for testing routing policy and for building a labeled dataset. One caveat: computing
`cache_preview.semantic_hit` costs an embedding call (~20 ms) when the exact cache misses; when the
semantic check doesn't run, `semantic_hit` and `similarity` are `null`, not `false`.

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
  "cost": {"actual_usd": 0.0859, "baseline_usd": 1.7057,
           "saved_usd": 1.6198, "saved_pct": 95.0},
  "latency_ms": {"cache_exact": {"p50": 1.1, "p95": 2.4},
                 "cache_semantic": {"p50": 17.8, "p95": 31.2},
                 "local_small": {"p50": 116.0, "p95": 189.0},
                 "local_large": {"p50": 438.0, "p95": 702.0},
                 "frontier": {"p50": 1980.0, "p95": 3110.0},
                 "end_to_end": {"p50": 121.0, "p95": 1840.0}},
  "tokens": {"in": 48211, "out": 129884,
             "baseline_in": 61430, "baseline_out": 155210,
             "frontier_in": 3104, "frontier_out": 7812},
  "model_loads": {"cold_loads": 7, "mean_load_ms": 1420.0},
  "degraded": []
}
```

`saved_pct` is the headline. `model_loads` is tracked separately so Ollama cold-start time
(`OLLAMA_MAX_LOADED_MODELS=2` means occasional evictions) doesn't silently pollute the latency
percentiles.

Three token accumulators, three distinct jobs — do not conflate them:

- `tokens.in`/`out` — **throughput** totals across every tier attempted (a cascaded request counts
  its prompt once per attempt, and every intermediate completion). Load/latency analysis only;
  **excluded from all cost formulas.**
- `tokens.baseline_in`/`baseline_out` — the **baseline** accumulators: per request, prompt tokens
  counted **once** plus the *serving* tier's completion tokens, **cache hits included** (a cache hit
  contributes the incoming prompt's token count and the cached entry's stored completion count).
- `tokens.frontier_in`/`frontier_out` — the subset actually **billed**; drives actual spend.

The counts must reconcile, and a test should assert it:

```
sum(tiers.served)                                 == requests
router.entry.local_small                          == tiers.attempted.local_small
router.entry.local_large + gates.small.escalated  == tiers.attempted.local_large
router.entry.frontier    + gates.large.escalated  == tiers.attempted.frontier
gates.X.accepted         + gates.X.escalated      == gates.X.evaluated
cost.actual_usd   == frontier_in/1e6   × PRICE_IN + frontier_out/1e6   × PRICE_OUT
cost.baseline_usd == baseline_in/1e6   × PRICE_IN + baseline_out/1e6   × PRICE_OUT
cost.baseline_usd == Σ per-request x_hlr.baseline_cost_usd
cost.baseline_usd − cost.actual_usd               == cost.saved_usd
```

Against the example, at `PRICE_IN=2.50` / `PRICE_OUT=10.00` per 1M:
served sums to 1284 ✓; 60 + 179 = 239 ✓; 15 + 41 = 56 ✓;
actual = 3104/1e6×2.50 + 7812/1e6×10.00 = 0.0859 ✓;
baseline = 61430/1e6×2.50 + 155210/1e6×10.00 = 1.7057 ✓;
1.7057 − 0.0859 = 1.6198 ✓, and 1.6198 / 1.7057 = 95.0% ✓.
(`tokens.in`/`out` appear in no identity — deliberately.)

Note what the baseline assumes: that **every** request, cache hits included, would otherwise have gone
to the frontier. That's the "what would this traffic cost with no gateway" question, which is the
useful one — but it does mean cache hits are credited at full frontier price (see README, "How
savings are counted", and D13).

---

## GET /metrics/prometheus

Prometheus exposition format, unconditionally. Authenticated like `/metrics` —
it carries cost data, and making it public so scraping is easier would be the
wrong trade. Scrapers send a bearer token.

```
# HELP hlr_cost_saved_usd_total baseline_cost_usd_total minus cost_usd_total.
# TYPE hlr_cost_saved_usd_total counter
hlr_cost_saved_usd_total 1.07
# HELP hlr_requests_by_tier_total Requests served, by the tier that answered.
# TYPE hlr_requests_by_tier_total counter
hlr_requests_by_tier_total{tier="local_small"} 30
hlr_requests_by_tier_total{tier="frontier"} 12
# HELP hlr_circuit_state Circuit breaker: 0 closed, 1 half-open, 2 open, -1 unknown.
# TYPE hlr_circuit_state gauge
hlr_circuit_state{tier="frontier"} 2
```

**`GET /metrics` also negotiates.** `/metrics` is the name the ecosystem
reserves for exposition format, and this endpoint has always served JSON for
the dashboard. Both stay true: send `Accept: text/plain` (which every
Prometheus scraper does) and you get exposition; send anything else — including
curl's bare `*/*` — and you get the JSON. This endpoint exists for the human
who would rather not think about Accept headers.

The series worth alerting on:

| Series | Alert when |
|---|---|
| `hlr_budgets_enforced` | `0` — caps are configured and **not being applied** |
| `hlr_tenant_budget_enforced{tenant=}` | `0` — same, per tenant |
| `hlr_audit_dropped_total` | increasing — the audit trail has holes |
| `hlr_circuit_state{tier=}` | `2` — that tier is out |
| `hlr_dependency_up{dependency=}` | `0` |
| `hlr_tier_attempted_total{tier="frontier"}` | rate spikes — the signature of a local-tier outage, and of spend |

`hlr_tier_attempted_total` exceeds `hlr_requests_by_tier_total` for any tier
that escalated: a request that walked 0.5B → 3B → frontier was *attempted* by
three tiers and *served* by one. The gap between them is the escalation cost.

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

## GET /livez

Liveness. Is the process up — nothing else.

```json
{"status": "alive"}
```

**Deliberately dependency-free.** A liveness probe that checks Redis will
restart a healthy gateway during a Redis blip, turning a degradation into an
outage — the opposite of what the probe is for. The container `HEALTHCHECK`
uses this endpoint for the same reason.

---

## GET /readyz

Readiness. Can *this instance* serve a request right now?

```json
{"status": "ready", "ollama": true, "frontier": true, "detail": ""}
```

**200** when at least one model tier is reachable, **503** otherwise. Stricter
than `/healthz`: an instance with no reachable tier should be pulled from the
load balancer even though the process is fine, and returned on recovery without
a restart.

The split matters. `/healthz` answers "what is the state of everything",
`/livez` answers "should you restart me", `/readyz` answers "should you send me
traffic". Collapsing them means a Redis blip either restarts a healthy process
or keeps routing to a broken one.

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

On errors, `x_hlr` keys are **always present**; fields that never got a value are `null`
(`tier`, `confidence`) or `0` (`cost_usd`, `tokens_out`), and `gates`/`path` may be empty arrays. The
example above elides the zero-valued keys for brevity — an implementation must still emit them, per
Invariant 1.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed body, empty `messages` |
| 401 | `unauthorized` | Missing or invalid API key. Carries `WWW-Authenticate: Bearer` |
| 404 | `model_not_found` | Reserved `model` value naming a tier that isn't configured |
| 413 | `request_too_large` | Body over `MAX_BODY_BYTES` |
| 413 | `prompt_too_long` | Prompt over `MAX_PROMPT_CHARS` |
| 413 | `too_many_messages` | More than `MAX_MESSAGES` |
| 402 | `budget_exhausted` | Tenant over their daily cap **and** the caller pinned `gateway:frontier`. Without the pin, the request is served locally instead |
| 429 | `rate_limited` | Window exhausted. Carries `Retry-After` |
| 503 | `spend_ceiling_reached` | `MAX_DAILY_SPEND_USD` exhausted **and** no local tier could serve. Not 402 — the caller's account is fine and they can do nothing about it |
| 503 | `all_tiers_down` | No tier is reachable |
| 504 | `upstream_timeout` | Every tier exceeded its timeout |

401, 413 and 429 are raised in middleware **before** the handler, so they do not
carry the full `x_hlr` block — there is no routing decision to report on a
request that never reached the router.

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
cost_usd          = (frontier tokens billed):  in/1e6 × FRONTIER_PRICE_IN + out/1e6 × FRONTIER_PRICE_OUT
                    0 for cache and local tiers
baseline_cost_usd = tokens_in/1e6 × FRONTIER_PRICE_IN + tokens_out/1e6 × FRONTIER_PRICE_OUT
                    (prompt once + serving tier's completion, at frontier prices — every request,
                     cache hits included, as if no gateway existed)
cost_saved_usd    = baseline_cost_usd − cost_usd
```

Worked check against the `x_hlr` example: 41/1e6×2.50 + 156/1e6×10.00 = 0.0016625 → `0.00166` ✓.

Local tiers are counted as `$0` marginal. That is a deliberate simplification — electricity and
amortized hardware are real but not meaningfully attributable per request, and inventing a number
would make the headline savings figure less honest, not more (see D13).

Accumulate in integer micro-USD or `Decimal`, never float (see CLAUDE.md conventions) — float
accumulation over thousands of requests visibly drifts.
