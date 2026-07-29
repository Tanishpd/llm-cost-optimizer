# Using the gateway

The API surface and what you can do with it. Install is [SETUP.md](SETUP.md);
operating it is [RUNNING.md](RUNNING.md). The precise HTTP contract — every
field, every error code — is [API.md](API.md).

---

## The basic call

It speaks OpenAI's `/v1/chat/completions`, so any OpenAI-compatible client works
unchanged. Point the base URL at the gateway and send `model: "gateway"` to let
it route.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gateway","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

Pin a tier instead of routing by sending `model: "gateway:local_small"`,
`gateway:local_large` or `gateway:frontier`. Pinning the frontier removes your
own fallback, which is why it is the one case where a budget cap returns 402
rather than degrading.

---

## Trying each tier

`scripts/demo.py` walks all five in one run. By hand:

```bash
H='content-type: application/json'
Q='{"model":"gateway","messages":[{"role":"user","content":"What is the capital of France?"}]}'

curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" | jq .x_hlr   # → local_small
curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" | jq .x_hlr   # → cache_exact

# reworded — same question, different words
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Tell me France'\''s capital city."}]}' | jq .x_hlr
# → cache_semantic

# the 0.5B should hesitate here; the 3B picks it up
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Explain why a rainbow has the colour order it does."}]}' | jq .x_hlr
# → local_large

# neither local model confident
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Prove that every finite integral domain is a field."}]}' | jq .x_hlr
# → frontier
```

---

## Authentication

Set `GATEWAY_API_KEYS` to a comma-separated list and every request needs one, as either
`Authorization: Bearer <key>` or `X-API-Key: <key>`. Comparison is constant-time; logs carry a short
digest of the principal, never the key.

```bash
GATEWAY_API_KEYS=sk-local-abc123,sk-local-def456
REQUIRE_AUTH=true          # refuse to start if the list above is empty
```

Leave the list empty and **the gateway runs open**, logging a loud warning at startup. That's the
right default for the single-user quickstart and the wrong one for anything reachable — which is what
`REQUIRE_AUTH` is for: it turns a silent misconfiguration into a startup failure. `/livez`, `/readyz`
and `/healthz` stay public so an orchestrator can probe without a credential
([D17](DECISIONS.md)).

---

## Limits

Each is enforced in middleware, before any model is touched.

| Setting | Default | Exceeded → |
|---|---|---|
| `MAX_BODY_BYTES` | 256000 | `413` |
| `MAX_PROMPT_CHARS` | 32000 | `413` |
| `MAX_MESSAGES` | 50 | `413` |
| `RATE_LIMIT_PER_MINUTE` | 60 | `429` + `Retry-After` |
| `MAX_OUTPUT_TOKENS` | 4096 | clamped, reported in `x_hlr.degraded` |

The rate-limit window lives in a Redis sorted set behind a Lua script, so the cap is shared across
workers rather than multiplied by them. If Redis is unreachable it degrades to per-process counting
and says so in `/healthz` ([D18](DECISIONS.md)).

---

## Tenants, budgets and rate limits

A key authenticates; a **tenant** pays. Set `TENANTS` and each account gets its own daily cap and its
own line in `/metrics`:

```bash
GATEWAY_API_KEYS=sk-acme-1,sk-globex-1
TENANTS='{"acme":   {"keys": ["sk-acme-1"],   "daily_budget_usd": "5.00"},
          "globex": {"keys": ["sk-globex-1"], "daily_budget_usd": "1.00"}}'
```

**Over budget degrades, it does not reject** — and that is the whole point. If most prompts don't
need the expensive model, then running out of money for the expensive model shouldn't stop you being
served. The frontier drops out of the cascade and the local tiers carry on:

```
tier      local_large
path      ['local_large']
degraded  ['budget_exhausted:acme']
reason    daily budget exhausted: $0.000628 of $0.000500 spent today (UTC);
          frontier tier withheld, so: served by local_large: confidence 0.78 …
```

It also makes the cap safe to set aggressively: if exhaustion meant 402s you'd set it high enough
never to bind, which defeats the point ([D23](DECISIONS.md)).

Two things stated rather than glossed. **The cap is approximate by exactly one request** — cost isn't
knowable until the provider replies, so the budget is read before the call and recorded after; the
overshoot is bounded by `MAX_OUTPUT_TOKENS × FRONTIER_PRICE_OUT`, ~$0.04 at the defaults, and startup
logs the number. **If Redis is down, budgets aren't enforced** — failing closed would turn a cache
blip into a total frontier outage, so it fails open and says so in four places
([D24](DECISIONS.md), [D25](DECISIONS.md)).

**The gateway also caps itself.** `MAX_DAILY_SPEND_USD` (default **$10**) bounds spend across every
tenant, and unlike the per-tenant budgets it ships *armed*. It exists because invariant 4 — *"Ollama
down → escalate everything"* — is correct for availability and costs about **$138/day per client** at
the shipped rate limit. A business policy may default to off; a safety limit may not
([D27](DECISIONS.md)).

Measured, with Ollama pointed at a dead port and the ceiling set to $0.0006:

```
req 1-3   200  tier=frontier          (invariant 4, working as designed)
req 4-5   503  spend_ceiling_reached  (bounded)
spent     $0.000624 against a $0.0006 ceiling — one request's overshoot
```

**Caches are shared across tenants by default** — that sharing is where the savings come from, since
the second tenant to ask a common question pays nothing. The cost is that a semantic hit can match
another tenant's entry, so the reason string withholds the matched text in that mode.
`CACHE_SCOPE=tenant` isolates both caches instead, trading the cross-tenant saving for isolation
([D28](DECISIONS.md)).


### Per-tenant rate limits

Budgets bound what an account may *spend*. They do not bound what it may
*consume* — a request served from cache costs $0.00 and still occupies a slot,
so a tenant can saturate the gateway without touching their cap.

`RATE_LIMIT_SCOPE=tenant` gives an account one rate-limit window across all its
keys. The default, `key`, gives it one window *per key*, so ten keys buy ten
times the limit. A tenant may also carry its own ceiling:

```bash
TENANTS='{"acme": {"keys": ["sk-a"], "daily_budget_usd": "5.00", "rate_limit_per_minute": 30}}'
RATE_LIMIT_SCOPE=tenant
```

Unauthenticated callers stay keyed by client IP under either scope: every
anonymous request resolves to the same `anonymous` tenant, so scoping those by
tenant would put the whole internet in one bucket.

---

## Audit trail

One JSON line per request in `AUDIT_PATH`, **including refusals**, durable across restarts:

```json
{"ts": 1785258422.7, "trace_id": "b6ee310fb7af", "tenant": "acme", "principal": "777a832b78f0",
 "tier": "local_large", "path": ["local_large"], "cost_micro_usd": 0,
 "degraded": ["budget_exhausted:acme"], "gates": [...]}
```

This is the thing `/metrics/recent` can't be: that reads a Redis stream capped at 10k entries and
answers *what is happening now*. This answers *why did request `abc123` cost $0.04 last Tuesday*.

`principal` is a digest — the raw key is never written. Writes are queued and drained by a background
task so the request path never waits on a disk; a full queue drops rather than blocking, and
`/healthz` reports the count, because a trail with unreported holes is worse than no trail
([D26](DECISIONS.md)).

---

## Streaming

`stream: true` returns SSE that any OpenAI-compatible client reads. It **does
not lower time-to-first-token**: a gate cannot decide whether to escalate until
the full answer and its logprobs exist, so the cascade finishes first and the
completed answer is then re-emitted as frames — *buffer-then-stream*
([D8](DECISIONS.md)). The feature is client compatibility, not latency.

What buffering buys instead: no token is ever retracted, so you never watch the
0.5B's answer get overwritten by the frontier's; errors stay ordinary HTTP
errors rather than truncated streams; and the `X-HLR-*` headers survive, because
the routing decision is final before the first byte.

`STREAM_CHUNK_CHARS` sets characters per content frame. `0` sends the whole
answer as one frame.
