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

Re-measured 2026-07-28. These are the reason the architecture looks the way it does. Do not design
around imagined constraints; design around these — and **re-measure rather than trusting these
numbers**, which is what `scripts/preflight.py` is for.

- **8 GB RAM. This is the binding constraint, and it cannot be relieved.** Local models are capped
  at 3B–4B. `mistral:7b` (~4.8 GB resident, estimated) will thrash — excluded on RAM, not disk. Keep
  `OLLAMA_MAX_LOADED_MODELS=2`; three resident models plus a browser will swap.
- **RAM also caps worker count.** Each uvicorn worker holds its own numpy replica and HTTP pool, so
  2–4 is the practical ceiling here. Measured RSS is ~10 MB per worker under load, so the ceiling is
  about the models, not the gateway.
- **Disk: 32 GB free (85% used) as of 2026-07-28**, up from 3.2 GB after a deliberate reclaim. At
  target. Do not treat the headroom as spare capacity to fill — ~3 GB is this project and the rest is
  swap room. **Verify with `df -h /System/Volumes/Data` before pulling models.**
- **Swap: 3.3 GB of 4.0 GB used (81%).** Swap lives on disk, so the two constraints are coupled —
  that is why disk headroom matters beyond storing weights. It was 90% at 3.2 GB free disk.
- **No torch, `transformers`, or `sentence-transformers`.** Not a disk limit any more — Ollama
  already serves embeddings over HTTP, so torch would add a second inference runtime competing for
  the same scarce RAM. Runtime Python is **numpy-only** for math; `scikit-learn` is eval-only.
- **Redis 7.2.7, no modules.** No `FT.*`, no RedisJSON, no vector index. Brute-force numpy cosine.
- **No Docker daemon on this machine.** Nothing may *require* a container, and you cannot verify the
  `Dockerfile`/`docker-compose.yml` locally — CI builds and smoke-tests the image. Don't claim a
  container change works because it looks right.
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
sits around 0.5–0.75 *even when correct* — **measured**, RESEARCH.md §3.1 — because of normal lexical
freedom. Shipping `0.85` against the raw metric escalates nearly everything and looks broken. Two
modes exist for this reason: `CONFIDENCE_MODE=raw` (thresholds 0.55 / 0.62 on `geo_mean`), and
`CONFIDENCE_MODE=calibrated` where 0.85 genuinely means "85% chance this tier's answer is as good as
the frontier's". Don't collapse these two modes into one.

**`calibrated` is the shipped default since 0.4.0.** M6 failed (gate `large`: good ECE, no better
than the base rate on Brier — calibrated without being useful, §3.4). The corrected real-frontier run
passes both gates at n=110 and survives every counterfactual refit, §3.7, so 0.85 is earned. It
stayed `raw` at first on portability grounds; the provenance guard changed that, by turning a model
mismatch into a startup refusal rather than a silent wrong number.

**That default only holds because `eval/calibrator.json` is committed and was fitted on the models
`config.py` also defaults to.** Change `SMALL_MODEL` or `LARGE_MODEL` without refitting and every
fresh start refuses, naming both model sets. `tests/test_config_schemas.py` asserts the two agree, so
changing one without the other fails locally rather than in someone's deployment.

**Test factories must pin `confidence_mode="raw"` explicitly.** They bypass the lifespan, so they
bypass the guard too — and `confidence.py` falls back to raw `geo_mean` when no calibrator is loaded
while `threshold_for` returns the calibrated 0.92. Inheriting the default pointed ~900 tests at a
configuration the gateway will not boot, and they *all still passed*, because `REAL_LOGPROBS` scores
0.9725 and clears both 0.55 and 0.92. Green by coincidence.

**A calibrator is only valid for what it was fitted on.** `calibrator.json` carries `provenance`
(models, token budget, dataset size, judge) and the lifespan **refuses to start** in `calibrated`
mode when those disagree with the configured models. Not a warning: these weights map one model's
logprob statistics onto P(acceptable), so pointed at another model they still return a number in
[0, 1] that looks exactly like a probability. Nothing downstream can tell.

**The calibrated path is wired but not exercised by default**, which is exactly how it silently
rotted once: `CONFIDENCE_MODE=calibrated` was settable while `main.py` never loaded
`eval/calibrator.json`, so the gates used raw thresholds under a `calibrated` label. If you touch
`confidence.py` or the lifespan, keep the tests in `tests/test_cascade.py` under "calibrated mode"
green — they are the only thing executing that path.

**Calibrator weights are in standardized space.** `mean`/`scale` travel with them and must be
applied before the dot product. Skipping that step yields a number in [0, 1] that looks like a
probability and is not one — nothing downstream can detect it.

**Two gates, two calibrators.** Gate 1 (`small`, 0.5B → 3B) and gate 2 (`large`, 3B → frontier) are
*not* the same decision and must not share fitted parameters. `eval/calibrator.json` holds one
parameter set per gate.

**Raw-mode thresholds are not comparable across gates.** `THRESHOLD_SMALL=0.55` and
`THRESHOLD_LARGE=0.62` apply to different models, whose `geo_mean` distributions differ — the 0.5B
scores systematically lower than the 3B on the same prompt. So the ordering of those two numbers
carries *no* economic meaning; they're per-model starting points. Do not write prose claiming the
ordering encodes a policy — an earlier draft did, and it was backwards.

**The economics only apply in calibrated mode**, where both scores are comparable probabilities.
There the argument is: escalating from gate 1 costs ~450 ms and no money, so gate 1 should demand a
*higher* acceptance probability (`TARGET_SMALL=0.92`) — when the next tier is nearly free, use it
readily. Escalating from gate 2 costs real money, so it accepts at `TARGET_LARGE=0.85`, the spec's
number, on the gate that actually spends. **Gate 1's bar is higher than gate 2's, not lower.**

**The safety behaviour is the expensive behaviour.** Invariant 4 sends everything to the frontier
when Ollama dies — right for availability, ~$138/day per client at the shipped rate limit.
`MAX_DAILY_SPEND_USD` bounds it and **ships armed** ($10), unlike per-tenant budgets which default to
unlimited. Do not "fix" that inconsistency by disarming it: a business policy may default to off, a
safety limit may not (D27). Note `verdict.tenant` is the pseudo-bucket `__all__` when the ceiling
binds — attribute spend with `tenant_name`, never `verdict.tenant`, or it lands on a fake account and
double-counts.

**Sampling defaults live in two endpoints and must not drift.** `DEFAULT_TEMPERATURE`
and `DEFAULT_MAX_TOKENS` are operator policy applied by `_apply_sampling_defaults`, which
**both** `/v1/chat/completions` and `/route` call. They must: temperature and max_tokens are
part of the cache key (invariant 5), so if only one applied them, `/route`'s `exact_hit`
preview would be computed from a different key than the real lookup. These two settings were
defined, documented in `.env.example` *and* PLAN.md, and read by nothing at all until
`tests/test_wiring.py` asked what every setting does.

**The eval must generate answers at the length production serves.** `--max-tokens` defaulted to
120 while the gateway serves `DEFAULT_MAX_TOKENS=512` and references got 1024, so the calibrator was
fitted to answers a quarter the length of the ones it scores, and the judge compared truncated
candidates against full-length references. 26% of local answers were cut off mid-sentence; every
`reasoning` prompt was, which is why that category graded 0/26 and made `should_flip()` return
`True` for the first time — on an artifact. RESEARCH.md §3.6. The default now follows
`DEFAULT_MAX_TOKENS`, and `check_budget` refuses to extend a run cache across a change of budget:
answer length feeds `n_tokens`, a calibrator feature, and both halves of a mixed cache look like
valid rows.

**A "pass" that moves when you re-slice the data has not been earned.** Before believing a flip,
refit on subsets — it costs nothing, the run cache is on disk. Gate `large` passed on all 110
prompts, failed with `reasoning` excluded, and passed by 0.006 Brier with truncated answers
excluded. Check whether the *baseline* moved too: a more balanced dataset makes the base-rate
predictor worse, and beating a weaker baseline is not the same as carrying more information.

**The cascade is ordered data.** `app/cascade.py` holds a list of `(name, runner, gate)`; the
executor walks it from the router's entry point. Adding or reordering a tier is a data change. Never
reintroduce nested if/else routing — that's what the ordering exists to prevent.

**Ollama logprobs.** **Verified working on this machine in M0** (v0.30.10) — RESEARCH.md §2 has the
captured response. Request `"logprobs": true, "top_logprobs": 5` with **`"stream": false`** — the
non-streaming path accumulates all chunks' logprobs into the final response, which is what the gate
needs. `top_logprobs` must be 0–20 or the server 400s. GitHub issue #16117 is closed "not planned"
and reads like a denial; it is wrong. This was the project's single highest-risk assumption, which
is why M0 existed to settle it before anything was built on top.

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
| SSE framing | `app/streaming.py` — pure; takes a finished response, yields frames |
| Tenants, spend caps | `app/budgets.py` — a tenant is an account, a key is a credential |
| Durable request record | `app/audit.py` — JSONL; never blocks the loop, counts its drops |
| Metrics / dashboard | `app/telemetry.py`, `app/dashboard.py`, `static/dashboard.html` |
| Recalibrate the thresholds | `eval/run_eval.py` → writes `eval/calibrator.json` |
| Auth, request size bounds | `app/security.py` (+ `PUBLIC_PATHS` if adding an unauthenticated route) |
| Rate limiting | `app/ratelimit.py` — Redis sorted set behind one Lua script |
| Timeouts, retries, breakers | `app/resilience.py` — wraps any awaitable; knows nothing about tiers |
| Log format / trace id | `app/logging_setup.py` |
| Prove the docs still match | `scripts/check_docs.py` — add a check when you add an invariant |
| Prove no credential is committed | `scripts/check_secrets.py` — runs in CI; GitHub's own scanning needs Advanced Security, which this repo has not got |
| Prove the dashboard still works | `tests/test_dashboard.py` — parses *and executes* its JS |
| Prove a new setting does something | `tests/test_wiring.py` — every knob is tested or declared inert |
| Prove it survives load | `scripts/loadtest.py` — asserts cost identities, not just latency |
| Prove a claim holds for all inputs | `tests/test_properties.py` — geo_mean's length invariance lives here |
| Prove the tests would catch a bug | `scripts/mutation.sh` — baseline 19 survivors, all prose or equivalent |
| Prove shared state is shared | `scripts/multiworker_check.py` — runs in CI on 4 workers |
| Prove it survives *time* | `scripts/soak.py` — memory, fds, latency drift; distinguishes a leak from a step |
| Export metrics to a scraper | `app/prometheus.py` — pure renderer; `/metrics` negotiates on `Accept` |
| Measure the gate against reality | `app/shadow.py` — off by default; spends real money |
| Diagnose a live problem | `docs/RUNBOOK.md` — add an entry when you add an alertable signal |
| Bump the version | `app/__init__.py` only; `check_docs.py` asserts the rest agree |

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

## Security posture

The gateway spends money per request, so three surfaces are closed and must
stay closed:

- **Auth** (`app/security.py`). `GATEWAY_API_KEYS` empty means the gateway runs
  OPEN — acceptable only on a loopback bind, and startup warns loudly.
  `REQUIRE_AUTH=true` makes an open start a hard failure. Keys are compared with
  `secrets.compare_digest`; never add a comparison that short-circuits.
- **Rate limiting**, per key (per client IP when open). In-process, so under
  multiple workers the effective limit multiplies by the worker count.
- **Request bounds**: body size, prompt length, message count, and `max_tokens`.
  `max_tokens` is *clamped*, not rejected — a caller asking for more almost
  always means "as much as you'll give me", and the response says it happened.

Only `/healthz` and `/` are public (a load balancer has no credentials).
`/metrics` exposes cost data and is protected; a test asserts it is not in
`PUBLIC_PATHS`.

## Cost of running eval

`eval/run_eval.py` with `EVAL_USE_REAL_FRONTIER=true` spends real money — but far less than the
prompt count suggests. Of 110 prompts, **67 are auto-gradable and make no API calls at
all**; only the 43 judged ones do, at 1 reference + 1 judge per local tier = **129 calls**.
(An earlier note here said ~360, by assuming every prompt hits the API.)
Every API response is cached to `eval/runs/*.json`; **always reuse that cache when re-fitting
calibrators.** Never re-spend to recompute something already on disk, and never enable the real
frontier for serving — only eval.

**The harness has its own spend ceiling, and it ships armed.** `MAX_DAILY_SPEND_USD` lives in
`BudgetLedger` on the gateway's request path; `run_eval.py` calls the provider directly and never
touches it, so the one component that deliberately spends real money had no cap at all.
`EVAL_MAX_SPEND_USD` (default $5; a full run measures $0.47 at the placeholder prices, $0.11 at
verified Gemini rates) is checked **before** each paid call — a ceiling that notices only once it has been crossed is a report, not a limit. When it
binds, the run stops the way a daily quota does: progress checkpointed, fit on what was collected.
`0` disarms it and the run says so at startup.

**The run reports what it spent** — `ApiUsage` per paid call, printed as a cost summary at the end.
It reports a *range* when the provider reports thinking tokens, because whether `completion_tokens`
already includes them is provider-specific and guessing wrong is a multiple, not a rounding error.
Prices (`FRONTIER_PRICE_*`, `EVAL_JUDGE_PRICE_*`) are placeholders — the judge is deliberately a
different model from the reference (D11), so it needs its own rates, and neither pair is a quote.

## Running

```bash
ollama serve &                     # required; not running by default
redis-cli ping                     # PONG expected; brew services start redis
python3 scripts/preflight.py       # verifies models, disk, swap, logprobs
uvicorn app.main:app --port 8000 --reload
uvicorn mock_frontier.main:app --port 8001    # separate terminal
```

Before opening a PR — all three run in CI, so run them here first:

```bash
.venv/bin/python -m pytest -m "not integration"   # hermetic; needs nothing running
.venv/bin/ruff check app eval scripts mock_frontier tests
.venv/bin/python scripts/check_docs.py            # imports app.config, needs the venv
```

`scripts/loadtest.py` needs a running gateway and is the only check that
exercises the cost identities under concurrency — run it after touching
`telemetry.py`, `costs.py`, or anything in `cache/`.

**Multi-worker is supported and verified** (M10). All shared state lives in Redis:

| State | Mechanism | Verified with 4 workers |
|---|---|---|
| Rate limit window | Redis sorted set, one Lua script | 20 requests, cap 10 → exactly 10 allowed |
| Semantic index | numpy replica + `hlr:sem:version` counter | seeded on one worker, 8/8 hits across all |
| Single flight | `SET NX PX` lease | 8 concurrent identical → 1 model call |
| Cache, telemetry | Redis | counters agree across workers |

The numpy matrix is still per-process, but it is a *replica* — the version
counter drives a resync, which is what makes it correct rather than merely
lucky. If Redis is unreachable the rate limiter falls back to per-process
state and says so in the log; the limit then multiplies by worker count,
which is worse than shared but far better than none.

## Current state

**M0–M25 complete (2026-07-29).** The gateway runs and is hardened: all five tiers, both gates, eval
harness, dashboard, auth, shared rate limiting, circuit breakers, structured logging, container
image, streaming, per-tenant budgets and rate limits, a durable audit trail. **913 unit + 10 integration
tests** at 96% coverage, 208 doc-consistency checks. `python3 scripts/demo.py`
walks every tier.

**The calibration passes, and `calibrated` ships by default (0.4.0).** The corrected real-frontier run
(M23) passes both gates at n=110 — `small` ECE 0.040, Brier 0.136 vs base 0.242; `large` ECE 0.027,
Brier 0.129 vs base 0.214 — and holds under every counterfactual refit except one that strips both
hard categories at once. `should_flip()` is `True` and **0.85 is earned**. RESEARCH.md §3.7.

The fit is specific to `qwen2.5:0.5b` / `qwen2.5:3b`, so the default is only safe because the
lifespan **refuses to start** when `calibrator.json`'s provenance disagrees with the configured
models. Running other models means `CONFIDENCE_MODE=raw`, or refitting.

Measured, and now recorded in RESEARCH.md §3.1–3.2:

- `geo_mean` per model — the 3B scores systematically higher than the 0.5B on identical prompts,
  which is the evidence for `THRESHOLD_SMALL < THRESHOLD_LARGE` being a distribution artifact.
- **`geo_mean` does not detect hallucination.** The 0.5B scored 0.5535 on a wholly fabricated answer
  and the gate accepted it at the 0.55 default — a demonstrated false-cheap. Read §3.2 before
  touching the gating logic; it constrains what any single-statistic threshold can achieve.

**One thing is queued:** a real-frontier calibration run, deferred by choice — it needs an API key
and a few dollars, and M6's negative result stands until it happens.

**Per-tenant rate limits landed in M24.** `RATE_LIMIT_SCOPE=tenant` gives an account one window
across all its keys; `TENANTS[name].rate_limit_per_minute` overrides the global ceiling for that
account. The default stays `key` — the prior behaviour — because changing what a window means is not
something an upgrade should do to a running deployment. Note `key` scope is *not* isolation: N keys
buy N times the limit, and per-tenant budgets do not close that, since a cache-served request spends
nothing and still consumes capacity.

Still deliberately not built: model allow-lists (every tenant may reach every tier), and shipping the
audit trail off-host — that one is not a code gap at all. `AuditLog` already writes rotated per-PID
JSONL, which is exactly what Vector or Fluent Bit want pointed at them; it is host config, not a
missing feature.

Before changing anything, run `.venv/bin/python scripts/check_docs.py` — it enforces the
cross-document invariants below, and it has caught drift four times.
