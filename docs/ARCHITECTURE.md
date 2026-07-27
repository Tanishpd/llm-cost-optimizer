# Architecture

How the gateway is put together and why the seams fall where they do.

This describes the system as **designed**; no code exists yet. [PLAN.md](PLAN.md) sequences the build,
[API.md](API.md) is the external contract, [DECISIONS.md](DECISIONS.md) records the rationale, and
[RESEARCH.md](RESEARCH.md) holds the verified facts underneath it all.

---

## 1. The idea

Most prompts don't need a frontier model. The gateway's job is to spend the least compute that still
produces an acceptable answer, and to be able to *prove* it did.

That reduces to two questions, and the architecture is organized around keeping them separate:

| Question | Answered by | When |
|---|---|---|
| **Where should this prompt start?** | Pre-router — cheap heuristics over prompt shape | Before any model runs |
| **Was that answer good enough?** | Confidence gate — the model's own token logprobs | After each model runs |

The first is a guess and is allowed to be wrong. The second is evidence, and it's what makes the
system correct despite the first being a guess: a bad entry decision costs latency, not quality,
because the gates still catch a weak answer and escalate it.

## 2. Request lifecycle

```
                    ┌────────────────────────────────────────────────────────┐
  POST              │  FastAPI handler — assigns trace_id, starts the clock  │
  /v1/chat/         └───────────────────────────┬────────────────────────────┘
  completions                                   │
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

  every RETURN → write-through to cache → telemetry record → x_hlr assembled
```

Two properties worth naming:

- **Cache lookups happen before routing.** A cache hit makes the routing question moot, and routing
  is cheap but not free.
- **The write-through is on every path.** Frontier answers are cached with a longer TTL
  (`CACHE_TTL_FRONTIER`) than local ones — an expensive answer is worth more shelf space.

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
| `costs.py` | Token → USD (pure) | Who called it |
| `telemetry.py` | Records and aggregates | Business logic |

The "does not know about" column is the actual design. Each module is testable without standing up
the ones beside it, and `confidence.py` in particular is pure functions over a logprob array — the
piece most worth unit-testing is the piece with no I/O in it.

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
| Calibrator params | `eval/calibrator.json` on disk | Until refit | Loaded at startup |
| Model weights | Ollama's process, not ours | `keep_alive` | We never hold weights |

The one subtle rule: **Redis is the source of truth and the numpy matrix is a warm replica.** The
replica exists because brute-force cosine over an in-process matrix is microseconds while round-tripping
vectors through Redis is not. It re-syncs when `hlr:sem:version` moves. If the two ever disagree,
Redis wins.

This is also the single-worker assumption. One uvicorn process, one replica. The version counter makes
multi-worker correct in principle; it is untested, and the docs say so rather than implying otherwise.

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
| **Everything down** | 503 `all_tiers_down` | The only real failure |

The interesting one is *frontier down*. The gate said escalate, and escalation is impossible — so the
gateway returns the local answer it already has, and says plainly in `reason` that it did. Returning a
below-threshold answer while admitting it beats returning nothing.

## 8. Trust boundaries

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

## 9. Extension seams

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

## 10. What the architecture deliberately doesn't do

- **No streaming.** The gate can't decide until the whole response and its logprobs exist, so
  streaming a local answer means streaming tokens that may be discarded. `stream: true` returns 400
  rather than pretending ([D8](DECISIONS.md#d8--non-streaming-first)).
- **No multi-worker.** One process, one warm index.
- **No auth, tenancy, or rate limiting.** Single-user gateway.
- **No local-inference cost model.** Local tiers book $0. Electricity and amortized hardware are real
  but not meaningfully attributable per request, and a made-up number would look more rigorous while
  being less honest ([D13](DECISIONS.md#d13--local-inference-counted-as-0)).
