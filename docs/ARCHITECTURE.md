# Architecture

How the gateway is put together and why the seams fall where they do.

This describes the system as **built** — every component below exists and is tested.
[PLAN.md](PLAN.md) sequences how it got here, [API.md](API.md) is the external contract,
[DECISIONS.md](DECISIONS.md) records the rationale, and [RESEARCH.md](RESEARCH.md) holds the
measured facts underneath it all.

---

## 1. The idea

Most prompts don't need a frontier model. The gateway's job is to spend the least compute that still
produces an acceptable answer, and to be able to *prove* it did.

That reduces to two questions, and the architecture is organized around keeping them separate:

| Question | Answered by | When |
|---|---|---|
| **Where should this prompt start?** | Pre-router — cheap heuristics over prompt shape | Before any model runs |
| **Was that answer good enough?** | Confidence gate — the model's own token logprobs | After each model runs |

The first is a guess and is allowed to be wrong — in one direction. A *too-low* entry costs latency,
not quality: the gates still catch a weak answer and escalate it. A *too-high* entry has no such
safety net — `frontier` carries `gate=None`, so a router false-positive there is silent, ungated
spend. That asymmetry drives two rules: frontier entry must be deliberately rare (no single feature,
length included, should trigger it alone), and the eval reports a **false-expensive rate** — of the
prompts the router would send straight to frontier, how many would a local gate have accepted?

## 2. Request lifecycle

```
                    ┌────────────────────────────────────────────────────────┐
  POST              │  enforce_access — auth, body/prompt bounds, rate limit │
  /v1/chat/         │  rejects here never reach a model:  401 / 413 / 429    │
  completions       └───────────────────────────┬────────────────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────┐
                    │  add_trace_id — ContextVar, echoed in every log line   │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────┐
                    │  FastAPI handler — starts the clock                    │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌──────────────────────────────────────────┐
                    │  cache/keys.py                           │
                    │  normalize → blake2b(prompt+model+params)│
                    └───────────────────┬──────────────────────┘
                                        │
                 ┌──────────────────────┴────────────────────┐
                 ▼                                           │
        ┌─────────────────┐  hit                             │ miss
        │ T0 exact cache  ├──────────────────────────────▶ RETURN
        └────────┬────────┘
                 │ miss
                 ▼
        ┌─────────────────┐  cosine ≥ 0.95 AND guards pass
        │ T1 semantic     ├──────────────────────────────▶ RETURN
        │ cache           │
        └────────┬────────┘  guard fail → record why, continue
                 │
                 ▼
        ┌─────────────────┐
        │ router/policy   │  features → entry tier. No model call.
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐  over cap → drop the billed tier from the list.
        │ budgets.check   │  Free unless a cap is configured.
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────────────────────────────────────────────┐
        │  cascade.py executor — walks the tier list from `entry` │
        └────────┬────────────────────────────────────────────────┘
                 │
                 ▼
        ┌─────────────────┐  confidence ≥ THRESHOLD_SMALL
        │ T2a local_small ├──────────────────────────────▶ RETURN
        └────────┬────────┘
                 │ gate 1 escalates
                 ▼
        ┌─────────────────┐  confidence ≥ THRESHOLD_LARGE
        │ T2b local_large ├──────────────────────────────▶ RETURN
        └────────┬────────┘
                 │ gate 2 escalates
                 ▼
        ┌─────────────────┐
        │ T3 frontier     ├──────────────────────────────▶ RETURN
        └─────────────────┘

  every RETURN → write-through to cache → telemetry + audit → x_hlr assembled
```

Two properties worth naming:

- **Cache lookups happen before routing.** A cache hit makes the routing question moot, and routing
  is cheap but not free.
- **The write-through is on every path.** Frontier answers are cached with a longer TTL
  (`CACHE_TTL_FRONTIER`) than local ones — an expensive answer is worth more shelf space.
- **Access control runs before anything expensive.** Auth, size bounds and the rate limit are
  middleware, so a rejected request costs a Redis round-trip and nothing else. It also means those
  three responses (401/413/429) are the only ones without a full `x_hlr` block — no tier ran, so
  there is no routing story to tell.
- **Every tier call is wrapped** by `resilience.call_with_resilience`, which owns the timeout, the
  retry policy and the circuit breaker. The cascade executor doesn't know any of that exists.

## 3. Component map

| Module | Owns | Deliberately does not know about |
|---|---|---|
| `main.py` | HTTP surface, lifespan, trace IDs | Which tier will run |
| `cascade.py` | Tier ordering and the walk | How any individual tier works |
| `router/policy.py` | Entry-tier choice + reason string | Model internals, cache state |
| `router/heuristics.py` | Feature extraction (pure functions) | Everything else |
| `confidence.py` | Logprobs → score; calibrators | HTTP, Redis, which model produced them |
| `providers/ollama.py` | Local chat + embed over HTTP | Routing, caching, cost |
| `providers/frontier.py` | OpenAI-shaped upstream | That it might be a mock |
| `cache/keys.py` | Normalization + hashing (pure) | Redis |
| `cache/exact.py` | T0 get/set/TTL | Embeddings |
| `cache/semantic.py` | T1 index, cosine, guards | Which model answers |
| `streaming.py` | SSE framing of a finished response (pure) | Tiers, gates, why it was buffered |
| `budgets.py` | Tenant resolution, spend caps, the ledger | Auth, HTTP, the cascade |
| `audit.py` | Durable JSONL record + rotation | What a request *is* |
| `costs.py` | Token → USD (pure) | Who called it |
| `telemetry.py` | Records and aggregates | Business logic |
| `security.py` | Key auth, request bounds, public paths | Redis, tiers |
| `ratelimit.py` | Shared sliding window (Redis Lua) | What is being limited |
| `resilience.py` | Timeouts, retries, circuit breakers | Which tier it is wrapping |
| `logging_setup.py` | JSON formatter, trace `ContextVar` | Everything |
| `dashboard.py` | Serves `static/dashboard.html` | How metrics are computed |

The "does not know about" column is the actual design. Each module is testable without standing up
the ones beside it, and `confidence.py` in particular is pure functions over a logprob array — the
piece most worth unit-testing is the piece with no I/O in it.

The bottom five are **cross-cutting and generic on purpose**. `resilience.py` wraps an arbitrary
awaitable and knows nothing about tiers; `ratelimit.py` limits a string key. Had either been written
against the cascade it would have been untestable without one, and reusing it for a sixth tier would
mean editing it.

## 4. The cascade is data

```python
# app/cascade.py
TIERS: list[Tier] = [
    Tier("local_small", runner=run_small, gate=gate_small),
    Tier("local_large", runner=run_large, gate=gate_large),
    Tier("frontier",    runner=run_frontier, gate=None),   # nowhere left to escalate
]
```

The executor walks this list from the router's entry index. Adding, removing, or reordering a tier is
a change to a list literal, not to control flow.

The alternative — nested `if/else` across five tiers, two gates, and two cache short-circuits — grows
combinatorially and becomes untestable. As data, the ordering is also *inspectable at runtime*, which
is exactly what `x_hlr.path` reports back to the caller. See [D14](DECISIONS.md#d14-the-cascade-is-ordered-data).

## 5. Where state lives

| State | Home | Lifetime | Notes |
|---|---|---|---|
| Cached answers | Redis (`hlr:exact:*`) | TTL | Source of truth |
| Embedding vectors | Redis (`hlr:sem:vec:*`) | TTL + LRU | Source of truth |
| **Warm vector matrix** | **Process RAM (numpy)** | Process | **Replica.** Never authoritative |
| Index version | Redis (`hlr:sem:version`) | Forever | INCR on write; replica re-syncs when it moves |
| Telemetry | Redis stream, `MAXLEN ~ 10000` | Capped | |
| Calibrator params | `CALIBRATOR_PATH` on disk | Until refit | Read once in the lifespan; absent is fatal in `calibrated` mode |
| Model weights | Ollama's process, not ours | `keep_alive` | We never hold weights |

The one subtle rule: **Redis is the source of truth and the numpy matrix is a warm replica.** The
replica exists because brute-force cosine over an in-process matrix is microseconds while round-tripping
vectors through Redis is not. It re-syncs when `hlr:sem:version` moves. If the two ever disagree,
Redis wins.

**Multi-worker is verified**, not merely designed for. Every worker keeps its own replica and
re-syncs when `hlr:sem:version` moves; with 4 processes, a vector indexed by one worker was found
by all of them (8/8 lookups). The rate limiter was the one piece that was genuinely broken across
workers — per-process windows meant the configured limit multiplied by worker count — and now lives
in a Redis sorted set behind a single Lua script, so the check-and-record is atomic.

## 6. Concurrency

- **Everything touching I/O is `async`**, over one pooled `httpx.AsyncClient` created in the lifespan.
  A synchronous client would block the event loop during a ~450 ms model call, serializing the whole
  gateway behind one request.
- **Pure functions are `def`** — heuristics, key hashing, confidence math, cost math. They're
  CPU-trivial and don't need to be coroutines.
- **The cosine matmul is CPU-bound but sub-millisecond**, so it runs inline rather than in a thread
  pool. If the index ever grew past ~100k vectors this would need revisiting — that's the same
  threshold at which a real vector index starts to pay ([D7](DECISIONS.md#d7--brute-force-cosine-and-redis-stays-at-72)).
- **Single-flight** on cache misses: `SET NX PX` on `hlr:lock:{hash}` so N identical concurrent
  prompts produce one model call, not N.

## 7. Failure model

Invariant 4 is *degrade, don't fail*. The gateway's job is to answer; a dependency outage should cost
capability, not availability.

| Failure | Behaviour | Reported as |
|---|---|---|
| Redis unreachable | Caching disabled, models still serve | `degraded: ["redis_unavailable"]` |
| Embedder missing | T1 skipped, T0 still works | `degraded: ["embedder_unavailable"]` |
| One local model missing | That tier skipped; `reason` names the `ollama pull` | `degraded: ["model_missing:…"]` |
| Ollama down entirely | Everything escalates to frontier | `degraded: ["ollama_unavailable"]` |
| Frontier down | Best local answer returned, gate verdict overridden | `degraded: ["frontier_unavailable"]` |
| Logprobs unavailable | Fall back to `CONFIDENCE_STRATEGY=self_consistency` | `degraded: ["logprobs_unavailable"]` |
| A tier keeps failing | Breaker opens; requests skip it instead of waiting out the timeout | `short_circuited` in `/healthz` |
| **Gateway over its daily ceiling** | Frontier dropped for *everyone*; local tiers serve | `degraded: ["spend_ceiling_reached"]` |
| Ceiling reached **and** Ollama down | 503 — the only bound on invariant 4's cost | `spend_ceiling_reached` |
| Tenant over their daily cap | Frontier dropped from the cascade; local tiers serve | `degraded: ["budget_exhausted:<tenant>"]` |
| Over cap **and** frontier pinned | 402 — the caller removed the fallback themselves | `budget_exhausted` |
| Redis down, budgets configured | Caps unenforced rather than blocking all spend | `degraded: ["budget_unavailable"]` |
| Audit writer cannot keep up | Records dropped, never blocking the request | `audit.dropped` in `/healthz` |
| **Everything down** | 503 `all_tiers_down` | The only real failure |

**The most expensive failure is a healthy one.** *Ollama down* keeps the gateway answering by
escalating everything — correct for availability, and about $138/day per client at the shipped rate
limit. `MAX_DAILY_SPEND_USD` bounds it, ships armed, and turns an unbounded invoice into a loud 503
once the local tiers cannot serve either ([D27](DECISIONS.md)). It is a blunt instrument: it bounds
damage rather than detecting the anomaly, and a breaker on the *rate* of escalations would trip in
seconds rather than dollars.

The interesting one is *frontier down*. The gate said escalate, and escalation is impossible — so the
gateway returns the local answer it already has, and says plainly in `reason` that it did. Returning a
below-threshold answer while admitting it beats returning nothing.

**Hanging is worse than failing**, which is what the breakers are actually for. A tier that refuses
connections is already fast to fail; a tier that accepts and never replies makes every request pay
the full timeout before escalating. Measured against a black-hole frontier at a 3 s timeout:
**3193 ms → 42 ms** once the breaker opened. Recovery is one probe per cooldown — without a probe an
open breaker never closes, and with unlimited probes a recovering tier gets flooded by everything
queued behind the outage ([D19](DECISIONS.md)).

Retries are narrow by design: timeouts and connection errors only, full jitter, and the frontier is
never retried on timeout because those tokens may already be billed. Paying twice for one answer is
precisely the cost this gateway exists to control ([D20](DECISIONS.md)).

## 8. Trust boundaries

Three of them, in order of how badly each fails.

### 8.1 The network edge

Everything before the handler. The gateway spends money per request, so an open port is a spending
authority. `enforce_access` runs first and rejects in this order — auth, then size, then rate —
because each check is cheaper than the one after it and a request that fails auth shouldn't get its
body measured.

| Control | Enforces |
|---|---|
| `security.ApiKeyAuth` | Bearer / `X-API-Key`, constant-time compare, digest in logs never the key |
| `security.RequestLimits` | `MAX_BODY_BYTES`, `MAX_PROMPT_CHARS`, `MAX_MESSAGES` → 413 |
| `ratelimit.RedisRateLimiter` | `RATE_LIMIT_PER_MINUTE` per principal → 429 |
| `security.PUBLIC_PATHS` | `/livez`, `/readyz`, `/healthz` only |

Two deliberate softenings, both loud rather than silent: an empty key list runs the gateway **open**
with a startup warning (`REQUIRE_AUTH=true` makes that a startup failure instead), and a rate limiter
that can't reach Redis falls back to per-process counting and reports the degradation. Both are
choices about *which* failure is worse, recorded in [D17](DECISIONS.md) and [D18](DECISIONS.md).

`MAX_OUTPUT_TOKENS` is the odd one out — it clamps rather than rejects, and reports the clamp in
`x_hlr.degraded`. An oversized `max_tokens` is almost always a client default, not an attack, and a
clamped answer serves the caller better than a 413.

### 8.2 The semantic cache

The semantic cache is the one place the system can be **confidently wrong**. Every other failure mode
is visible: a slow answer, a missing answer, an escalation. A false cache hit is a wrong answer served
instantly, for free, with no signal that anything happened.

So the guards are layered and all must pass:

```
cosine ≥ SEMANTIC_THRESHOLD        (0.95, deliberately high)
  AND number sets equal            → "convert 50 USD" ≠ "convert 70 USD"
  AND capitalized-entity sets equal→ "capital of Australia" ≠ "capital of Austria"
  AND negation markers match       → "is safe" ≠ "is not safe"
  AND single-turn
  AND identical system prompt
  AND len(prompt) ≤ SEMANTIC_MAX_CHARS
```

Every rejection is recorded in `cache.guard_rejected` so over-conservatism is measurable rather than
invisible. The asymmetry is deliberate: a miss costs ~120 ms, a false hit costs correctness. Bias
every judgment call toward missing ([D15](DECISIONS.md#d15--bias-every-cache-judgment-toward-missing)).

### 8.3 The frontier

The only tier that costs money and the only one outside our control. It is treated as untrusted in
both directions: its `usage` block is what gets billed, so it is recorded verbatim rather than
recomputed from our own token counts, and its timeouts are never retried. `FRONTIER_API_KEY` is
`SecretStr` and never appears in a log line or an `x_hlr` block.

## 9. How this is verified

Four layers, each catching something the others structurally cannot. The split
is not aspirational — it is the result of counting where real defects were
actually found, and noticing that the unit suite caught one of fourteen.

| Layer | Command | Catches | Blind to |
|---|---|---|---|
| **Unit** (845, hermetic) | `pytest -m "not integration"` | Logic inside a module | Whether modules are connected |
| **Wiring** | `tests/test_wiring.py` | Components not plugged in; settings wired to nothing | Anything the real lifespan cannot reach |
| **Rendering** | `tests/test_dashboard.py` | JS that does not parse, run, or match the DOM | Visual layout — a bar can be the wrong width and still render |
| **Doc consistency** (185) | `scripts/check_docs.py` | Docs contradicting the code | Whether the code is right |
| **Integration** (10) | `pytest -m integration` | Real models, real Redis, the real OpenAI SDK | Anything needing sustained load |
| **Load** | `scripts/loadtest.py` | Cost identities under concurrency, duplicate model calls | Correctness of a single request |
| **Soak** | `scripts/soak.py` | Memory/fd growth, latency drift, counter divergence over time | Anything faster than one sampling window |
| **Multi-worker** | `scripts/multiworker_check.py` | State that stopped being shared across processes | Single-process logic |
| **Mutation** | `scripts/mutation.sh` | Assertions that would not fail if the code were wrong | Anything outside the pure modules |
| **Property** | `tests/test_properties.py` | Inputs nobody thought of, over the pure functions | Anything stateful |

Two properties are worth naming because they are easy to lose:

**The unit suite is hermetic and stays that way.** CI re-runs it behind a dead
proxy, so a test that starts reaching the network fails there rather than
becoming flaky later. Tests that boot the real lifespan point `REDIS_URL` at a
closed port — attaching to a developer's live Redis once made a test pass in CI
and fail locally, which is the worst arrangement available.

**Coverage is measured but is not a gate.** It sat at ~95% while
`CONFIDENCE_MODE=calibrated` was wired to nothing and `DEFAULT_MAX_TOKENS` was
read by nothing. Every line ran; they just ran disconnected. `test_wiring.py`
exists because coverage cannot express "and then it is used".

**Assertion strength is graded, not assumed.** `scripts/mutation.sh` asks
whether a test would *fail* if the code were wrong — a different question from
coverage, which only says the line ran. Baseline: 1953 mutants, 19 survivors,
all changed prose or provably equivalent. Weekly, not a gate: its output needs
reading rather than thresholding, and a percentage target pushes people into
brittle tests to chase it.

**Multi-worker state is verified in CI** by `scripts/multiworker_check.py`
against four real processes. It is written to fail when state stops being
*shared*, not when the gateway stops working — a per-process fallback works
fine in each worker separately, and the only symptom is a limit multiplied by
the worker count.

**The mathematical claims are quantified, not illustrated.**
`tests/test_properties.py` asserts `geo_mean`'s length invariance — the property
D3 rests on — over generated inputs, and asserts that
`sequence_probability` really does decay, because a counter-example that
stopped behaving as described would quietly stop making its point.

## 10. Extension seams

Each is a place the design expects to change, with the interface already drawn:

| To change | Seam | Cost |
|---|---|---|
| Add/reorder a tier | `TIERS` list in `cascade.py` | A list entry — but check RAM first |
| Swap the local model | `SMALL_MODEL` / `LARGE_MODEL` env | Config |
| Go to a real provider | `FRONTIER_BASE_URL` / `_API_KEY` | Config, no code |
| Learned router | `classifier.py` Protocol | Train on `eval/` labels, not guesses |
| Different confidence metric | `ConfidenceStrategy` in `confidence.py` | Pure function swap |
| Real vector index | `cache/semantic.py` interface | Redis 8 `VADD`/`VSIM`, if ever needed |
| New backend | `providers/base.Provider` | One new module |

## 11. What the architecture deliberately doesn't do

- **No *incremental* streaming.** `stream: true` works, but by buffering the whole cascade and then
  re-emitting the finished answer as SSE. Time-to-first-token is unchanged — the gate can't decide
  until the whole answer and its logprobs exist, and streaming tokens that may be discarded is worse
  than a pause ([D8](DECISIONS.md)).
- **Worker count is capped by RAM, not by design.** Each worker holds its own numpy replica and
  HTTP pool; on 8 GB, 2–4 is the practical ceiling.
- **No per-tenant rate limits or model allow-lists.** Tenancy exists and covers *spend* — named
  accounts, daily caps, per-tenant attribution ([D22](DECISIONS.md)) — but rate limiting is still per
  *key*, and every tenant may reach every tier. Adding either means extending `Tenant`, which the
  config shape (JSON) already accommodates.
- **The audit trail is local, not shipped.** JSONL on disk, rotated, one file per worker
  ([D26](DECISIONS.md)). Durable across restarts, which the capped telemetry stream never was — but
  it does not survive the host, and there is no cross-worker query. Both are shipper problems, and
  JSONL is already the format a shipper wants.
- **No local-inference cost model.** Local tiers book $0. Electricity and amortized hardware are real
  but not meaningfully attributable per request, and a made-up number would look more rigorous while
  being less honest ([D13](DECISIONS.md#d13--local-inference-counted-as-0)).
