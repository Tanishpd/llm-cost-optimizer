# Build Plan

Read [RESEARCH.md](RESEARCH.md) first — it holds the verified API shapes and hard numbers this plan
depends on. [API.md](API.md) is the HTTP contract these milestones implement. [DECISIONS.md](DECISIONS.md)
records why each choice was made. Read [../CLAUDE.md](../CLAUDE.md) for architecture invariants before
writing code.

## Context

The goal is a single API gateway that sits in front of several models of escalating cost, and decides
— per request — the cheapest one that will actually do the job. Most prompts hitting a production LLM
app are cheap ones (summarize this, reformat that, answer a factual question); paying frontier prices
for all of them is the default failure mode, and it stays invisible until the bill arrives.

The interesting engineering claim is not "route simple prompts to a small model" — that's a
heuristic anyone can write. It's **"we can tell when the cheap model's answer is good enough"**,
using the model's own token log-probabilities as the signal. That claim is measurable, so this build
measures it instead of asserting it.

Constraints that shaped the design (all verified — see RESEARCH.md):
- **8 GB RAM with swap already 90% full** → the binding constraint. Caps local models at 3B–4B.
  `mistral:7b` from the original spec is excluded on RAM, not disk.
- **Redis 7.2.7 with no modules** → no vector index. Brute-force numpy cosine, sub-millisecond at
  any cache size this will see.
- **Ollama 0.30.10 exposes logprobs** → the confidence gate is real, not simulated.
- **~20 GB disk being freed** → room for two local models, so the cascade gets a second gate.

---

## Architecture in one pass

A request descends five tiers and stops at the first one that can serve it. Cost rises at every
step; the gateway's whole job is to stop early.

```
POST /v1/chat/completions
   │
   ├─ T0  EXACT CACHE      blake2b(normalized prompt + model + params) → Redis GET
   │                       hit → return                              $0      ~1 ms
   │
   ├─ T1  SEMANTIC CACHE   embed (nomic-embed-text) → cosine vs warm numpy matrix
   │                       ≥ threshold AND clears entity/number/negation guards
   │                       hit → return                              ~$0     ~20 ms
   │
   ├─ ──  PRE-ROUTER       heuristic features → entry tier. No model call.
   │                       {small | large | escalate}
   │
   ├─ T2a SMALL SLM        qwen2.5:0.5b, logprobs:true, top_logprobs:5
   │                       confidence ≥ THRESHOLD_SMALL → return      $0     ~120 ms
   │                       else ↓                                    (gate 1)
   │
   ├─ T2b LARGE SLM        qwen2.5:3b, logprobs:true, top_logprobs:5
   │                       confidence ≥ THRESHOLD_LARGE → return      $0     ~450 ms
   │                       else ↓                                    (gate 2)
   │
   └─ T3  FRONTIER         OpenAI-shaped HTTP → mock (or a real provider)
                           return                                     metered ~2 s

  every tier writes a telemetry record → Redis stream → /metrics + dashboard
```

**Why two local tiers.** A single gate only ever answers "local or not". Two gates make the gateway
choose *how much* compute a prompt deserves, which is the actual thesis. It also means the eval in M6
produces a genuine cost/quality frontier over four options rather than a single threshold sweep — and
the 0.5B model is weak enough that gate 1 fires often, making the mechanism visible in a demo.

**Why the pre-router has three outcomes.** Running a long reasoning prompt through a 0.5B model just
to discard it wastes ~120 ms and some RAM; `escalate` skips both local hops. Conversely a trivial
prompt should not pay for the 3B. The router picks an *entry point*; the gates handle everything after.

**RAM note.** Set `OLLAMA_MAX_LOADED_MODELS=2`. Keeping all three (0.5B + 3B + embedder ≈ 3.0 GB
resident) loaded on an 8 GB machine with a browser open is asking for swap thrash. Two resident with
`keep_alive=10m` means an occasional ~1–2 s cold load, which telemetry should record separately so it
doesn't pollute the latency stats.

---

## Milestones

Each is independently runnable and demoable. Do them in order — M0 gates everything.

### M0 — Preflight: prove logprobs work on *this* machine

The whole confidence design rests on one assumption. Verify it before building on it. No application
code in this milestone.

```bash
# free disk first — see RESEARCH.md §7 for what's reclaimable
df -h /System/Volumes/Data          # want ~20 GB free

ollama serve &
ollama pull qwen2.5:0.5b            # 0.40 GB
ollama pull qwen2.5:3b              # 1.93 GB
ollama pull nomic-embed-text        # 0.27 GB
curl -s localhost:11434/api/version

curl -s localhost:11434/api/chat -d '{
  "model": "qwen2.5:3b",
  "messages": [{"role":"user","content":"What is the capital of France? Answer in one word."}],
  "stream": false,
  "logprobs": true,
  "top_logprobs": 5
}' | python3 -m json.tool | head -60
```

**Pass:** the response contains a `logprobs` array of `{token, logprob, bytes, top_logprobs[]}`.
**If it fails** (unexpected — source says otherwise): set `CONFIDENCE_STRATEGY=self_consistency` and
implement the fallback gate — sample N=3 at `temperature=0.8`, embed the answers, escalate when mean
pairwise cosine < 0.9. Same `ConfidenceStrategy` interface, ~3× local latency, no logprobs needed.
Everything downstream is unaffected.

Then write `scripts/preflight.py`: Ollama reachable, all three models present, Redis reachable,
≥5 GB disk free, swap not saturated, logprobs functional. Every later milestone starts by running it.

### M1 — Skeleton gateway + local tiers + cost accounting

`python3 -m venv .venv` (3.14.6 is fine), install runtime requirements, then:

- `app/config.py` — pydantic-settings, every knob from the table below.
- `app/schemas.py` — OpenAI-compatible request/response + the `x_hlr` metadata block.
- `app/providers/ollama.py` — async httpx client, pooled, explicit timeouts. `chat()` and `embed()`.
  One client, model name as a parameter — the two local tiers are configuration, not two classes.
- `app/main.py` — lifespan (warm both models, open clients), `POST /v1/chat/completions` hardwired to
  T2b, `GET /healthz`.
- `app/costs.py` — token and USD math from Ollama's `prompt_eval_count` / `eval_count`.

Demoable: a real OpenAI-shaped response with latency and token breakdown.

### M2 — Exact cache + telemetry

- `app/cache/keys.py` — normalize, then `blake2b`. **The key must include model, temperature, top_p,
  max_tokens, and the system-prompt hash** — the same text at temperature 0 and 1.0 are not
  interchangeable answers.
- `app/cache/exact.py` — get/set with TTL; longer TTL for frontier answers, since they cost more and
  are worth keeping.
- `app/telemetry.py` — one record per request into a capped Redis stream + counter hashes. Record
  which tier *served* and which tiers were *attempted* — a request that ran 0.5B then 3B then frontier
  cost all three latencies, and the dashboard should say so.
- `GET /metrics` — JSON aggregates.

Demoable: same prompt twice, second returns in ~1 ms with `tier: "cache_exact"`.

### M3 — Pre-router

- `app/router/heuristics.py` — features, no model call: token count, code fences, imperative-verb
  class (`summarize`/`translate`/`rewrite` → simple), multi-step markers (`step by step`, `explain
  why`, `compare`, `derive`, `prove`), question depth, arithmetic, named-entity density,
  attached-context length.
- `app/router/policy.py` — weighted score → `RouteDecision(entry_tier, reason, features)`. The
  **reason string is a product requirement**, not a debug aid — it surfaces in the response and the
  dashboard.
- `app/router/classifier.py` — `Protocol` so a learned classifier can replace the heuristic without
  touching the policy engine.

Demoable: `POST /route` returns the decision and reason for a prompt without executing it.

### M4 — Confidence gates + escalation

- `mock_frontier/main.py` — separate FastAPI app on **:8001**, faithful OpenAI
  `/v1/chat/completions` including the `usage` block, with configurable simulated latency and
  per-token pricing. Because it's real HTTP, pointing `FRONTIER_BASE_URL` at a live provider is a
  config change, not a code change.
- `app/confidence.py` — the metrics from RESEARCH.md §3: `geo_mean`, `p10_prob`, `mean_margin`,
  `mean_norm_entropy`, `frac_low`, plus the hedge-phrase veto.
- `app/providers/frontier.py` — OpenAI-shaped client.
- `app/cascade.py` — the tier chain as **ordered data**, not nested ifs. Each entry is
  `(name, runner, gate)`; the executor walks it from the router's entry point. Adding a tier means
  adding a row.

**Ship with `CONFIDENCE_MODE=raw`**, `THRESHOLD_SMALL=0.55`, `THRESHOLD_LARGE=0.62`. Do not ship 0.85
against the raw geometric mean — RESEARCH.md §3 explains why that escalates nearly everything. The
0.85 gets earned in M6. The small model's threshold is lower on purpose: it only has to be good
enough to beat *its own* escalation cost, which is small.

Demoable: an easy prompt answered by the 0.5B; a medium one falling through to the 3B; an obscure one
reaching the frontier — each with its confidence score and reason.

### M5 — Semantic cache

- `app/cache/semantic.py`:
  - embed via `nomic-embed-text` (768-dim), store `float32` bytes in a Redis hash;
  - warm numpy matrix in-process, Redis as source of truth, re-sync on a version counter;
  - cosine over the matrix; `SEMANTIC_THRESHOLD=0.95`, deliberately conservative;
  - **false-hit guards, all must pass:** number sets equal, capitalized-entity sets equal, negation
    markers match. These catch the classic failures — *capital of Australia* vs *capital of Austria*,
    *convert 50 USD* vs *convert 70 USD*;
  - **never** semantic-cache multi-turn conversations, differing system prompts, or prompts over
    `SEMANTIC_MAX_CHARS` — long prompts carry attached context the embedding flattens away.

Demoable: "What's the capital of France?" then "Tell me France's capital city" → `tier:
"cache_semantic"` with the similarity score. And the Australia/Austria pair correctly *missing*.

### M6 — Eval harness and calibrating the 85%

This is what elevates the project from "works" to "measured".

- `eval/dataset.py` — ~120 labeled prompts across easy-factual / summarization / reasoning / code /
  obscure-factual.
- `eval/run_eval.py` — for each prompt, run **all three** model tiers, capturing confidence features
  at each. Then call a **real frontier API** for the reference answer and as the quality judge
  (`EVAL_USE_REAL_FRONTIER=true`, needs a key; ~120 × 2 calls ≈ a few dollars, one time). The mock
  cannot judge quality — this is the step that makes the calibrated threshold trustworthy.
  Cache every response to `eval/runs/*.json` so re-fitting the calibrator never re-spends money.
- Fit **two** calibrators (logistic regression, `requirements-eval.txt`), one per gate, mapping the
  feature vector to `P(this tier's answer is as good as the frontier's)`. Persist to
  `eval/calibrator.json`.
- Sweep both thresholds jointly; emit a cost-vs-quality frontier: escalation rate per gate, cost
  saved, quality preserved, and **false-cheap rate** (bad answers wrongly kept — the metric that
  actually matters).
- Flip to `CONFIDENCE_MODE=calibrated` with `0.85`. **Now the 85% means what the spec intended:** an
  85% probability the cheap answer is good enough.

Serving continues to use the mock frontier by default; only eval spends real money.

### M7 — Dashboard

`GET /dashboard` — self-contained HTML (no CDN), auto-refreshing: per-tier breakdown, cache hit rate,
cumulative spend vs the all-frontier baseline, gate-1 and gate-2 escalation rates, latency
percentiles per tier, and a live request log showing each routing reason and the full tier path taken.

### M8 — Hardening and the demo

Graceful degradation at every dependency — **Redis down → cache off, still serve; Ollama down →
escalate everything and say so; a model not pulled → skip that tier with a reason, don't 500.** A
degraded gateway must still answer. Add trace IDs, timeouts, single-flight stampede protection, and
`scripts/demo.py` walking all five tiers with a savings summary.

---

## Configuration surface

| Env var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local tiers |
| `SMALL_MODEL` | `qwen2.5:0.5b` | Tier 2a |
| `LARGE_MODEL` | `qwen2.5:3b` | Tier 2b |
| `EMBED_MODEL` | `nomic-embed-text` | Embeddings (768-dim) |
| `OLLAMA_NUM_CTX` | `2048` | Context window; keeps RAM sane on 8 GB |
| `OLLAMA_KEEP_ALIVE` | `10m` | Avoid cold reloads between requests |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | **Do not raise on 8 GB** — see RAM note above |
| `FRONTIER_BASE_URL` | `http://localhost:8001/v1` | Point at a real provider to go live |
| `FRONTIER_API_KEY` | `mock-key` | Bearer token |
| `FRONTIER_MODEL` | `gpt-4o-mock` | Escalation model name |
| `FRONTIER_PRICE_IN` / `_OUT` | `2.50` / `10.00` | USD per 1M tokens — **verify before quoting** |
| `EVAL_USE_REAL_FRONTIER` | `false` | M6 only; flips eval to a real API |
| `EVAL_JUDGE_MODEL` | `claude-sonnet-5` | Quality judge for eval |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache + telemetry |
| `CACHE_TTL_LOCAL` | `3600` | TTL for locally-served answers |
| `CACHE_TTL_FRONTIER` | `86400` | Longer — expensive answers are worth keeping |
| `SEMANTIC_ENABLED` | `true` | Kill switch for T1 |
| `SEMANTIC_THRESHOLD` | `0.95` | Cosine floor; conservative on purpose |
| `SEMANTIC_MAX_CHARS` | `500` | Above this, skip semantic caching |
| `CONFIDENCE_MODE` | `raw` | `raw` \| `calibrated` |
| `THRESHOLD_SMALL` | `0.55` | Gate 1, against `geo_mean` in raw mode |
| `THRESHOLD_LARGE` | `0.62` | Gate 2, against `geo_mean` in raw mode |
| `CONFIDENCE_THRESHOLD` | `0.85` | Both gates, calibrated mode (M6+) |
| `CONFIDENCE_STRATEGY` | `logprobs` | `logprobs` \| `self_consistency` |
| `TOP_LOGPROBS` | `5` | 0–20, server-validated |
| `ROUTER_SMALL_MAX_TOKENS` | `60` | Enter at 2a below this |
| `ROUTER_ESCALATE_MIN_TOKENS` | `600` | Skip both local tiers above this |

## Redis key schema

| Key | Type | Contents |
|---|---|---|
| `hlr:exact:{hash}` | string | JSON answer envelope, TTL'd |
| `hlr:sem:vec:{id}` | hash | `vec` (float32 bytes), `prompt`, `exact_key` |
| `hlr:sem:ids` | zset | id → insertion ts |
| `hlr:sem:version` | string | INCR on write; workers re-sync when it moves |
| `hlr:lru` | zset | id → last-access ts |
| `hlr:telemetry` | stream | one record per request, `MAXLEN ~ 10000` |
| `hlr:stats:tiers` | hash | counters: served-by and attempted-by, per tier |
| `hlr:stats:gates` | hash | gate-1 / gate-2 pass and fail counts |
| `hlr:stats:cost` | hash | cumulative actual vs baseline USD |
| `hlr:lock:{hash}` | string | `SET NX PX` single-flight |

## File tree

```
app/
  main.py              FastAPI app, lifespan, routes
  config.py            pydantic-settings
  schemas.py           OpenAI-compatible models + x_hlr metadata
  cascade.py           the tier chain as ordered data + executor
  costs.py             token + USD math
  confidence.py        logprob metrics, calibrators, hedge veto
  telemetry.py         per-request records, aggregates
  dashboard.py         HTML + /metrics
  router/
    heuristics.py      feature extraction
    policy.py          features → RouteDecision(entry_tier, reason)
    classifier.py      Protocol for a swappable learned classifier
  providers/
    base.py            Provider protocol
    ollama.py          chat + embed, model as a parameter
    frontier.py        OpenAI-shaped client
  cache/
    keys.py            normalization + hashing
    exact.py           T0
    semantic.py        T1: embeddings, cosine, guards
mock_frontier/main.py  faithful OpenAI mock on :8001
eval/
  dataset.py           labeled prompts
  run_eval.py          run all tiers, real-API reference + judge
  report.py            cost-vs-quality frontier
  runs/                cached API responses (never re-spend)
  calibrator.json      fitted params (generated)
scripts/
  preflight.py         environment checks
  demo.py              five-tier walkthrough
static/dashboard.html
docs/{PLAN,API,DECISIONS,RESEARCH}.md
CLAUDE.md  README.md  requirements.txt  requirements-eval.txt  .env.example
```

---

## Verification

Each command proves one tier actually fired. Check `x_hlr.tier` and `x_hlr.path` in every response.

```bash
# Setup
ollama serve &
redis-cli ping                                   # PONG
python3 scripts/preflight.py                     # all green
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &

H='content-type: application/json'
Q='{"model":"gateway","messages":[{"role":"user","content":"What is the capital of France?"}]}'

# T2a small — first call, cache cold, easy prompt
curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" \
  | jq '.x_hlr | {tier, path, confidence, reason, latency_ms}'
# expect tier="local_small", path=["local_small"]

# T0 exact — identical prompt
curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" \
  | jq '.x_hlr | {tier, latency_ms, cost_saved_usd}'
# expect tier="cache_exact", latency_ms.total < 5

# T1 semantic — reworded
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Tell me France'\''s capital city."}]}' \
  | jq '.x_hlr | {tier, similarity}'
# expect tier="cache_semantic", similarity >= 0.95

# T1 guard — must MISS despite high cosine
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"What is the capital of Australia?"}]}' \
  | jq '.x_hlr | {tier, guard_rejected}'
# expect tier != "cache_semantic"  (entity guard fired)

# T2b — gate 1 fires, 0.5B not confident enough, 3B answers
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Explain why a rainbow has the colour order it does."}]}' \
  | jq '.x_hlr | {tier, path, gates}'
# expect tier="local_large", path=["local_small","local_large"]

# T3 — both gates fire
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Prove that every finite integral domain is a field, rigorously."}]}' \
  | jq '.x_hlr | {tier, path, gates, cost_usd}'
# expect tier="frontier"

# Router only, no execution
curl -s localhost:8000/route -H "$H" -d "$Q" | jq

# Degradation — gateway must survive Redis loss
redis-cli shutdown nosave
curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" | jq '.x_hlr.tier'
# expect a served answer with caching disabled, not a 500
brew services start redis

# Aggregates
curl -s localhost:8000/metrics | jq
open http://localhost:8000/dashboard
python3 scripts/demo.py
```

---

## Deliberately deferred

- **Streaming (SSE).** Confidence gating needs the whole response before deciding, so streaming a
  local answer means streaming tokens you may then discard. Resolvable (buffer, stream only after the
  decision) but it complicates M4 for no demo value. Non-streaming is the honest first build.
- **Redis 8 Vector Sets.** Available via `brew upgrade redis` (RESEARCH.md §4) but unnecessary below
  ~100k entries and risky to swap under a running server. `cache/semantic.py` is written behind an
  interface so it's a drop-in later.
- **A learned routing classifier.** The Protocol exists; the heuristic ships first because M6's eval
  is what would generate the training labels anyway. Measure, then learn.
- **A third local tier / 7B model.** RAM-blocked, not a design choice. Revisit on different hardware.
- **Multi-tenancy, auth, rate limiting.** Out of scope for a single-user gateway.
- **Multi-worker uvicorn.** The warm numpy index assumes one process. The version counter makes
  multi-worker correct in principle, but it's untested until there's a reason to run it.
