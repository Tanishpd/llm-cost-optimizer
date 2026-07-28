# CLAUDE.md — agent working notes

Context for an AI agent working in this repo. Human-facing docs live in [README.md](README.md).

**Read before writing any code:** [docs/RESEARCH.md](docs/RESEARCH.md) (verified API shapes and hard
numbers), [docs/PLAN.md](docs/PLAN.md) (milestones, config surface, file tree), and
[docs/API.md](docs/API.md) (the HTTP contract you are implementing — if code and that doc disagree,
one of them is a bug). [docs/DECISIONS.md](docs/DECISIONS.md) records why each choice was made and
what would justify reversing it — read it before proposing a different approach.

---

## What this is

An API gateway that fronts two local small language models and an expensive frontier API, and decides
per request how much compute a prompt deserves. Five tiers, first match wins:

```
exact cache → semantic cache → qwen2.5:0.5b →[gate 1]→ qwen2.5:3b →[gate 2]→ frontier
```

The load-bearing idea is the **confidence gate**: a model's own token log-probabilities are used to
estimate whether its answer is good enough, and sub-threshold answers escalate to the next tier. That
claim is treated as empirical — `eval/` measures it rather than assuming it.

## Environment constraints — non-negotiable

Verified 2026-07-27. These are the reason the architecture looks the way it does. Do not design
around imagined constraints; design around these.

- **8 GB RAM. This is the binding constraint, and it cannot be relieved.** Local models are capped at
  3B–4B. `mistral:7b` (~4.8 GB resident, estimated) will thrash — excluded on RAM, not disk. Keep
  `OLLAMA_MAX_LOADED_MODELS=2`; three resident models plus a browser will swap.
- **Disk: target ~20 GB free. Measured 7.9 GB (97% full) as of 2026-07-27**, up from 3.2 GB — being
  reclaimed, not yet at target. RESEARCH.md §7 lists what's reclaimable. `scripts/preflight.py` gates
  on this; **verify with `df -h /System/Volumes/Data` before pulling models** rather than assuming
  headroom exists.
- **Swap was 90% full at 3.2 GB disk; 71% at 7.9 GB.** Swap lives on disk, so the two constraints are
  coupled — this is why disk headroom matters beyond storing weights.
- **No torch, `transformers`, or `sentence-transformers`.** Not a disk limit any more — Ollama
  already serves embeddings over HTTP, so torch would add a second inference runtime competing for
  the same scarce RAM. Runtime Python is **numpy-only** for math; `scikit-learn` is eval-only.
- **Redis 7.2.7, no modules.** No `FT.*`, no RedisJSON, no vector index. Brute-force numpy cosine.
- **No Docker daemon.** Nothing may require a container.
- **Python 3.14.6** is the target and every dependency has arm64 wheels for it.

If a change would violate one of these, stop and raise it rather than working around it silently.

## Invariants

1. **Every response carries an `x_hlr` metadata block** — `tier` (who served it), `path` (every tier
   attempted, in order), `gates` (each gate's score and verdict), `reason`, `confidence`,
   `latency_ms` per stage, `tokens_in`/`tokens_out`, `cost_usd`, `cost_saved_usd`. The gateway's
   product *is* the visibility into its decisions; a response without it is a bug. A request that
   walked 0.5B → 3B → frontier paid for all three, and the metadata must show that.
2. **The routing `reason` is a human-readable string**, not an enum. It shows up in the dashboard.
3. **Never block the event loop.** All I/O is `async` via a shared pooled `httpx.AsyncClient` from
   the lifespan. No `requests`, no sync Ollama client, no blocking file reads in a handler.
4. **Degrade, don't fail.** Redis down → caching off, still serve. Ollama down → escalate everything
   and say so in `reason`. One model missing → skip that tier with a reason naming the `ollama pull`
   to run; don't take the whole cascade down. A dependency outage must never turn into a 500 on the
   happy path.
5. **Cache keys include model and sampling params.** Same prompt at `temperature=0` and
   `temperature=1.0` are different questions. Key = `blake2b(normalized_prompt + model + temperature
   + top_p + max_tokens + system_prompt_hash)`.
6. **Semantic cache hits must clear the guards**, not just the cosine threshold. See below.
7. **Redis is the source of truth**; the in-process numpy matrix is a warm replica re-synced on
   `hlr:sem:version`. Never let the replica be authoritative.
8. **The frontier tier talks real HTTP** to an OpenAI-shaped endpoint. Swapping the mock for a real
   provider must remain a `FRONTIER_BASE_URL` change with no code edit.

## Things that are easy to get wrong

**Confidence math.** Do not use sequence probability (`∏ p_i`). It decays exponentially with length —
a 100-token answer at a healthy p=0.9 per token scores 2.6e-5, so any fixed threshold rejects
everything. Use `geo_mean = exp(mean(logprob_i))`, which is length-invariant (= 1/perplexity). Full
derivation in RESEARCH.md §3.

**The 85% threshold.** The spec says "escalate below 85% confidence." Raw `geo_mean` on a small model
sits around an expected 0.5–0.75 *even when correct* (a prior, not yet measured here — RESEARCH.md
§3), because of normal lexical freedom. Shipping `0.85` against
the raw metric escalates nearly everything and looks broken. Two modes exist for this reason:
`CONFIDENCE_MODE=raw` (thresholds 0.55 / 0.62 on `geo_mean`) until `eval/run_eval.py` fits the
calibrators, then `CONFIDENCE_MODE=calibrated` where 0.85 genuinely means "85% chance this tier's
answer is as good as the frontier's". Don't collapse these two modes into one.

**Two gates, two calibrators.** Gate 1 (`small`, 0.5B → 3B) and gate 2 (`large`, 3B → frontier) are
*not* the same decision and must not share fitted parameters. `eval/calibrator.json` holds one
parameter set per gate.

**Raw-mode thresholds are not comparable across gates.** `THRESHOLD_SMALL=0.55` and
`THRESHOLD_LARGE=0.62` apply to different models, whose `geo_mean` distributions differ — the 0.5B
scores systematically lower than the 3B on the same prompt. So the ordering of those two numbers
carries *no* economic meaning; they're per-model starting points, to be replaced by measurement in
M6. Do not write prose claiming the ordering encodes a policy — an earlier draft did, and it was
backwards.

**The economics only apply in calibrated mode**, where both scores are comparable probabilities.
There the argument is: escalating from gate 1 costs ~450 ms and no money, so gate 1 should demand a
*higher* acceptance probability (`TARGET_SMALL=0.92`) — when the next tier is nearly free, use it
readily. Escalating from gate 2 costs real money, so it accepts at `TARGET_LARGE=0.85`, the spec's
number, on the gate that actually spends. **Gate 1's bar is higher than gate 2's, not lower.**

**The cascade is ordered data.** `app/cascade.py` holds a list of `(name, runner, gate)`; the
executor walks it from the router's entry point. Adding or reordering a tier is a data change. Never
reintroduce nested if/else routing — that's what the ordering exists to prevent.

**Ollama logprobs.** Confirmed in the Ollama source (v0.12.11+; installed client is 0.30.10) but
not yet observed on this machine — M0 verifies it before anything is built on it. Request
`"logprobs": true, "top_logprobs": 5` with **`"stream": false`** — the non-streaming path accumulates
all chunks' logprobs into the final response, which is what the gate needs. `top_logprobs` must be
0–20 or the server 400s. GitHub issue #16117 is closed "not planned" and looks like a denial — the
source disagrees; trust the source.

**Embeddings endpoint.** Use `POST /api/embed` with `{"model":..., "input":...}` returning
`{"embeddings": [[...]]}` — plural, array-of-arrays. The legacy `/api/embeddings` returns
`{"embedding": [...]}` — singular, flat. Mixing them up produces a silent shape bug. `input` accepts
an array; batch when you can.

**Semantic cache false hits.** High cosine is not sameness. `capital of Australia` / `capital of
Austria` and `convert 50 USD` / `convert 70 USD` both exceed 0.95. All guards must pass:
- number sets equal between the two prompts
- capitalized-entity sets equal
- negation markers (`not`, `never`, `without`, `except`) match
- skip semantic caching entirely for multi-turn, differing system prompts, or prompts over
  `SEMANTIC_MAX_CHARS`

A wrong cache hit is worse than a cache miss — it's a confidently wrong answer served for free. Bias
every judgment call toward missing.

## Conventions

- `async def` for anything touching I/O; `def` for pure functions.
- Pydantic v2 models in `app/schemas.py`; settings via pydantic-settings in `app/config.py`. No
  `os.getenv` scattered through the code.
- Providers implement the `Provider` protocol in `app/providers/base.py`. New backends go there.
- The router returns a `RouteDecision` dataclass — never a bare string.
- Redis keys are namespaced `hlr:` (schema table in PLAN.md). Don't invent keys outside it.
- Costs are `Decimal` or integer micro-USD, never float accumulation.
- Every handler gets a trace ID, propagated into telemetry.

## Where things live

| Need | File |
|---|---|
| Understand the shape of it | `docs/ARCHITECTURE.md` — read before changing structure |
| Add a config knob | `app/config.py` (+ table in PLAN.md, + `.env.example`) |
| Change routing logic | `app/router/policy.py`; features in `heuristics.py` |
| Add/reorder a tier | `app/cascade.py` — it's a list |
| Change confidence math | `app/confidence.py` |
| Add an LLM backend | `app/providers/`, implement `base.Provider` |
| Cache behaviour | `app/cache/{keys,exact,semantic}.py` |
| Response shape | `app/schemas.py` |
| Metrics / dashboard | `app/telemetry.py`, `app/dashboard.py`, `static/dashboard.html` |
| Recalibrate the thresholds | `eval/run_eval.py` → writes `eval/calibrator.json` |

## Extending

- **Better router:** implement the `classifier.py` Protocol. Train on labels from `eval/`, not
  hand-written ones. Measure before learning.
- **Real provider:** set `FRONTIER_BASE_URL`, `FRONTIER_API_KEY`, `FRONTIER_MODEL`,
  `FRONTIER_PRICE_IN`, `FRONTIER_PRICE_OUT`. All env vars — no code change. Verify current prices;
  the defaults are placeholders, not quotes.
- **Redis 8 Vector Sets:** `brew upgrade redis` gets 8.8.1 with `VADD`/`VSIM` compiled in (this
  machine has the required C11 atomics). Swap behind `cache/semantic.py`'s interface. Unnecessary
  below ~100k entries.
- **More tiers:** add a row to `app/cascade.py`. Check the RAM budget first — see the constraints
  above; a fourth resident model does not fit.

## Cost of running eval

`eval/run_eval.py` with `EVAL_USE_REAL_FRONTIER=true` spends real money (~120 prompts × 3 calls
≈ 360 — 1 reference completion + 1 judge call per local tier).
Every API response is cached to `eval/runs/*.json`; **always reuse that cache when re-fitting
calibrators.** Never re-spend to recompute something already on disk, and never enable the real
frontier for serving — only eval.

## Running

```bash
ollama serve &                     # required; not running by default
redis-cli ping                     # PONG expected; brew services start redis
python3 scripts/preflight.py       # verifies models, disk, swap, logprobs
uvicorn app.main:app --port 8000 --reload
uvicorn mock_frontier.main:app --port 8001    # separate terminal
```

Single uvicorn worker. The warm numpy index assumes one process — the version counter makes
multi-worker correct in principle but it's untested.

## Current state

**M0–M8 complete (2026-07-28).** The gateway runs: all five tiers, both gates, eval harness,
dashboard. 456 unit + 6 integration tests. `python3 scripts/demo.py` walks every tier.

**Shipping in `raw` confidence mode, not `calibrated`** — M6 fitted both calibrators and gate
`large` failed its validation (good ECE, but no better than the base rate on Brier, i.e. calibrated
without being useful). `should_flip()` requires both gates, so 0.85 has not been earned. RESEARCH.md
§3.4.

Measured, and now recorded in RESEARCH.md §3.1–3.2:

- `geo_mean` per model — the 3B scores systematically higher than the 0.5B on identical prompts,
  which is the evidence for `THRESHOLD_SMALL < THRESHOLD_LARGE` being a distribution artifact.
- **`geo_mean` does not detect hallucination.** The 0.5B scored 0.5535 on a wholly fabricated answer
  and the gate accepted it at the 0.55 default — a demonstrated false-cheap. Read §3.2 before
  touching the gating logic; it constrains what any single-statistic threshold can achieve.

Next: **M1** in [docs/PLAN.md](docs/PLAN.md) — skeleton gateway, `providers/ollama.py`, cost
accounting.
