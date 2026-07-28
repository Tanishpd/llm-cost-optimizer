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
- **8 GB RAM** → the binding constraint, measured. Caps local models at 3B–4B. `mistral:7b` from the
  original spec is excluded on RAM, not disk.
- **Redis 7.2.7 with no modules** → no vector index. Brute-force numpy cosine — calculated
  sub-millisecond to ~10k entries, a few ms at 100k. The measured semantic tier p50 is 40 ms
  end-to-end, but that is dominated by the embedding call, not the matmul.
- **Ollama 0.30.10 exposes logprobs** → **verified on this machine in M0** (RESEARCH.md §2), which is
  what the whole cascade rests on. The fallback (`CONFIDENCE_STRATEGY=self_consistency`) survives for
  the case where a future model or backend doesn't provide them.
- **Disk ~32 GB free as of 2026-07-28**, reclaimed from 3.2 GB → room for two local models, so the
  cascade gets a second gate.

---

## Architecture

**See [ARCHITECTURE.md](ARCHITECTURE.md)** — request lifecycle, component boundaries, state ownership,
concurrency, failure model, and extension seams. It's the single source of truth for how the system is
shaped; this document only sequences the work of building it.

The short version: a request descends five tiers — exact cache, semantic cache, `qwen2.5:0.5b`,
`qwen2.5:3b`, frontier — and stops at the first that can serve it. A pre-router picks the *entry*
tier from cheap heuristics; confidence gates on token logprobs decide each escalation after that.

Two constraints that shape the milestones below:

**RAM.** Set `OLLAMA_MAX_LOADED_MODELS=2`. All three models resident (0.5B + 3B + embedder, an
estimated ~3.0 GB) on an 8 GB machine with a browser open invites swap thrash. Two resident with
`keep_alive=10m` means an occasional ~1–2 s cold load — telemetry must record those separately
(`model_loads` in `/metrics`) so they don't pollute the latency percentiles.

**Router errors are cheap in one direction only.** A too-low entry costs latency — the gates still
catch a weak answer. A too-high entry (straight to `frontier`) is ungated spend: nothing checks it.
So the heuristic router is fine for choosing *between the local tiers*, frontier entry must require
multiple hardness signals rather than any single feature, and M6 measures the **false-expensive
rate** to keep that path honest. That asymmetry is why M3 ships a heuristic and M6 invests in
calibrating the gates.

---

## Milestones

Each is independently runnable and demoable. Do them in order — M0 gates everything.

### M0 — Preflight: prove logprobs work on *this* machine ✅ DONE (2026-07-28)

The whole confidence design rests on one assumption. Verify it before building on it.

**Outcome: logprobs confirmed.** Verbatim response in RESEARCH.md §2; measured `geo_mean`
distributions in §3.1; measured resident RAM in §5. Shipped: `app/confidence.py` (37 unit tests),
`scripts/preflight.py`, project scaffolding. The `self_consistency` fallback was not needed.

**The probe also produced a finding that constrains everything downstream** — RESEARCH.md §3.2: raw
`geo_mean` accepted a wholly fabricated answer at the 0.55 default, scoring it *above* an honest
struggle on a proof. Read §3.2 before implementing M4's gates.

Original instructions retained below for reproducibility.

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
logprobs functional, swap not saturated, and **≥10 GB disk free — hard fail below that** (per
RESEARCH.md §7, the machine is unpleasant below 10 GB because swap grows on disk), warning below the
~20 GB target. Also record each model's actual resident RAM and its `geo_mean` on a fixed probe
prompt, so RESEARCH.md §5's estimates and M4's thresholds get replaced by measurements. Every later
milestone starts by running this.

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
- `DELETE /cache` — flush T0/T1 (dev convenience; see API.md).

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
- `app/confidence.py` — the metrics from RESEARCH.md §3: `geo_mean`, `min_prob`, `p10_prob`,
  `mean_margin`, `mean_norm_entropy`, `frac_low`, plus the hedge-phrase veto. Pure functions over a
  logprob array — no I/O, so this is the most testable module in the codebase. Unit-test it first.
- `app/providers/frontier.py` — OpenAI-shaped client.
- `app/cascade.py` — the tier chain as **ordered data**, not nested ifs. Each entry is
  `(name, runner, gate)`; the executor walks it from the router's entry point. Adding a tier means
  adding a row.

**Ship with `CONFIDENCE_MODE=raw`**, `THRESHOLD_SMALL=0.55`, `THRESHOLD_LARGE=0.62`. Do not ship 0.85
against the raw geometric mean — RESEARCH.md §3 explains why that escalates nearly everything. The
0.85 gets earned in M6.

These two raw numbers are **per-model starting points and are not comparable to each other** — the
0.5B's `geo_mean` distribution sits lower than the 3B's, so `0.55 < 0.62` reflects the models, not a
policy. The economic asymmetry (escalating from gate 1 is nearly free, so gate 1 should demand a
*higher* acceptance bar) only becomes expressible in calibrated mode, where the scores are comparable
probabilities. M0 already recorded the starting distributions (RESEARCH.md §3.1): the 0.5B measured
`geo_mean` 0.97 / 0.47 / 0.59 on easy / medium / hard, the 3B 1.00 / 0.60 / 0.67 — the 3B
systematically higher, as the non-comparability argument requires.

**Also mandatory reading before writing the gates: RESEARCH.md §3.2.** The M0 probe produced a
measured false-cheap — the 0.5B scored a fabricated answer at 0.5535 and the 0.55 gate accepted it,
while scoring an honest proof attempt *lower*. Consequences for this milestone: the hedge veto is
load-bearing rather than a nicety, and no amount of tuning the single raw threshold fixes the
underlying fluency-vs-factuality gap. Do not "fix" it by raising 0.55 — that fits one example.

Demoable: an easy prompt answered by the 0.5B; a medium one falling through to the 3B; an obscure one
reaching the frontier — each with its confidence score and reason.

### M5 — Semantic cache

- `app/cache/semantic.py`:
  - embed via `nomic-embed-text` (768-dim), store `float32` bytes in a Redis hash. **Use nomic's
    task-prefix convention consistently — the same `search_query:`-style prefix on both the stored
    and the incoming side.** Mismatched prefixes shift the whole similarity distribution and silently
    invalidate the threshold;
  - warm numpy matrix in-process, Redis as source of truth, re-sync on a version counter;
  - cosine over the matrix; `SEMANTIC_THRESHOLD=0.95`, deliberately conservative. The threshold is
    an assertion about this embedder — so preflight gains a **20-pair probe** (10 paraphrase pairs,
    10 near-miss pairs like Australia/Austria) that prints both similarity distributions, replacing
    "0.95 seems safe" with a measured margin;
  - a semantic hit must also **match the stored entry's sampling params** (`exact_key` check —
    same temperature/top_p/max_tokens/system prompt), or it's a miss. Cosine similarity says the
    *questions* match; the params say the *answers* are interchangeable;
  - a vector whose exact-cache entry has TTL-expired is a **dangling vector**: treat as a miss and
    lazily delete it;
  - **false-hit guards, all must pass:** number sets equal, capitalized-entity sets equal, negation
    markers match — including contractions (*isn't*, *won't*, *can't*) and bare *no*. These catch
    the classic failures — *capital of Australia* vs *capital of Austria*, *convert 50 USD* vs
    *convert 70 USD*. Argument-order swaps (*USD to EUR* vs *EUR to USD*) are an accepted residual
    risk, documented rather than guarded;
  - **never** semantic-cache multi-turn conversations, differing system prompts, or prompts over
    `SEMANTIC_MAX_CHARS` — long prompts carry attached context the embedding flattens away;
  - LRU eviction over `hlr:lru` (zset of id → last-access ts), capped at `SEMANTIC_MAX_ENTRIES`.

Demoable: "What's the capital of France?" then "Tell me France's capital city" → `tier:
"cache_semantic"` with the similarity score. And the Australia/Austria pair correctly *missing*.

### M6 — Eval harness and calibrating the 85% ✅ DONE (2026-07-28)

**Outcome: gate `small` passed, gate `large` failed, gateway stays in `raw` mode.**
Full numbers in RESEARCH.md §3.4. The free auto-gradable path (67 prompts, ground-truth
grading, zero spend) produced the result; the 65 judge-gradable prompts remain unlabelled
and are the natural next step via `--real-frontier`.


This is what elevates the project from "works" to "measured".

- `eval/dataset.py` — ~120 labeled prompts across easy-factual / summarization / reasoning / code /
  obscure-factual.
- `eval/run_eval.py` — for each prompt, run **all three** model tiers, capturing confidence features
  at each. Then call a **real frontier API** (`FRONTIER_BASE_URL`/`FRONTIER_API_KEY`) for the
  reference answer, and a **separate judge model** (`EVAL_JUDGE_BASE_URL`/`EVAL_JUDGE_MODEL`/
  `EVAL_JUDGE_API_KEY`) to score each local answer against it. Using a different provider as judge
  than as reference avoids a model grading its own output.
  Cache every response to `eval/runs/*.json` so re-fitting never re-spends.
- Fit **two** calibrators (logistic regression, `requirements-eval.txt`), one per gate, mapping the
  feature vector to `P(this tier's answer is as good as the frontier's)`. Persist to
  `eval/calibrator.json`.
- **Validate the calibration, don't assume it.** Fit under k-fold cross-validation and compute all
  reliability metrics on the **pooled out-of-fold predictions** (per-fold bins at N≈120 are noise):
  reliability curve with equal-count bins, Brier score, and ECE, per gate. Judge quality is part of
  the measurement: give the judge a minimal rubric (acceptable = factually equivalent and complete
  vs the reference), re-judge a ~30-prompt subsample 2–3× and report judge self-agreement. If power
  is short, growing the set to ~500 prompts (~1,500 calls, ~4.2× the spend) is the cheapest fix.
- Sweep both thresholds jointly; emit a cost-vs-quality frontier: escalation rate per gate, cost
  saved, quality preserved, and **false-cheap rate** (bad answers wrongly kept — the metric that
  actually matters).
- Report the **false-expensive rate** too — of the prompts the pre-router would enter at `frontier`,
  how many would a local gate have accepted? Free to compute: eval already runs all three tiers per
  prompt. This is the only check on the one path with no gate (see the entry-asymmetry note above).
- Flip to `CONFIDENCE_MODE=calibrated` (`TARGET_SMALL=0.92`, `TARGET_LARGE=0.85`) **only on a
  quantitative pass, per gate: ECE ≤ 0.10 on pooled out-of-fold predictions, with a bootstrap 95%
  CI upper bound < 0.20.** "Looks close to the diagonal" is not a criterion at this sample size —
  ~12 points per bin puts ±0.28 of noise on each bin. Fail → stay in `raw` and report that honestly;
  a negative result here is a real finding, not a failure.

**Cost.** Per prompt: 1 reference completion + 1 judge call per local tier = 3 frontier-class calls;
~120 prompts ≈ 360 calls. Compute the actual figure from your configured rates before running — the
price constants in `.env.example` are placeholders, not quotes. Serving continues to use the mock by
default; only eval spends real money.

### M7 — Dashboard ✅ DONE (2026-07-28)

`GET /dashboard` — self-contained HTML (no CDN), auto-refreshing: per-tier breakdown, cache hit rate,
cumulative spend vs the all-frontier baseline, per-gate (`small`, `large`) escalation rates, latency
percentiles per tier, and a live request log showing each routing reason and the full tier path taken.
This is also where the estimated latency figures quoted in README.md get replaced by real percentiles.

### M8 — Hardening and the demo ✅ DONE (2026-07-28)

Graceful degradation at every dependency — **Redis down → cache off, still serve; Ollama down →
escalate everything and say so; a model not pulled → skip that tier with a reason, don't 500.** A
degraded gateway must still answer. Add trace IDs, timeouts, single-flight stampede protection, and
`scripts/demo.py` walking all five tiers with a savings summary.


### M9 — Auth, limits, CI ✅ DONE (2026-07-28)

The gateway spends money per request, and until M9 anyone who could reach the
port could spend it. Probed before and after on a running instance:

| Probe | Before | After |
|---|---|---|
| No credentials | 200 | **401** |
| `/metrics` unauthenticated | 200 — cost data exposed | **401** |
| 400 KB prompt | 200, 9.2 s | **413** |
| `max_tokens: 999999` | accepted | clamped, reported in `x_hlr.degraded` |
| 6 rapid requests, cap 5 | all 200 | `200 200 200 200 429 429` |

`app/security.py` holds auth, rate limiting and request bounds. CI runs unit
tests on 3.12 and 3.14, ruff, and `scripts/check_docs.py`. The unit suite is
re-run behind a dead proxy so it stays hermetic.

CI failed on its first run and caught a real portability bug: the suite only
worked under `python -m pytest`, because only that form puts the CWD on
`sys.path`. Bare `pytest` — what CI and most editors run — could not import
`app` at all.

### M10 — Production hardening ✅ DONE (2026-07-28)

Three parts, each verified rather than asserted.

**Multi-worker.** The docs had claimed "correct in principle but untested"
since M1. One piece was not correct in principle either: the rate limiter kept
its window in process memory, so N workers each enforced their own and a
configured 60/min silently became 60N. Moved to a Redis sorted set behind one
Lua script. Verified with 4 real worker processes:

    rate limit      20 requests, shared cap 10  -> exactly 10 allowed
    semantic index  indexed by one worker       -> 8/8 lookups hit
    single flight   8 concurrent identical      -> 1 model call

**Resilience.** `app/resilience.py` adds circuit breakers and narrow retries.
The failure they address: when a tier *hangs*, every request waits out its full
timeout before escalating. Against a black-hole frontier at a 3 s timeout,
**3193 ms → 42 ms** once the circuit opened. Retries cover only timeouts and
connection errors, and the frontier is not retried on timeout because the
tokens may already be billed. Structured logging via `LOG_FORMAT=json`, trace
id attached from a `ContextVar`.

**Deployment and load.** Multi-stage Dockerfile (non-root), compose file,
`/livez` and `/readyz` split from `/healthz`, and `scripts/loadtest.py`.

The load test found a real bug on its first run: per-request costs were rounded
to whole micro-USD before being summed, so the identity `/metrics` is supposed
to satisfy exactly held only approximately — a 75 micro-USD gap over 200
requests. Aggregates now derive from the exactly-summed integer token counters,
making the identity true by construction. After the fix:

    200 req @ concurrency 16   200/200 ok   75 req/s    8 model calls
    500 req @ concurrency 48   500/500 ok   75 req/s   18 model calls
    p50 28ms  p95 2069ms  p99 4853ms   worker RSS 10 MB

**The calibrated path was a knob with nothing behind it.** Auditing the docs
against the code turned up the worst kind of bug — one that only fires when you
change a setting, and reports success while doing so. `CONFIDENCE_MODE=calibrated`
was documented and settable, but nothing ever loaded `eval/calibrator.json`
into serving: `cascade.execute()` accepted a `calibrators` dict that `main.py`
never passed. The gates fell back to raw thresholds while `x_hlr.confidence_mode`
reported `calibrated`.

Underneath it was a second, quieter defect. `eval/calibration.Calibrator`
standardizes features before applying weights; the serving-side
`app.confidence.LogisticCalibrator` had no `mean`/`scale` at all. Even once
loaded, it would have applied standardized-space weights to raw features and
returned a number in [0, 1] that looks exactly like a probability and is not
one. Nothing downstream could have detected that.

Both fixed: `CALIBRATOR_PATH` is read once in the lifespan, standardization
travels with the weights, a feature-order mismatch raises instead of scoring,
and `CONFIDENCE_MODE=calibrated` with no calibrators **refuses to start** — the
same rule as `REQUIRE_AUTH`, for the same reason. A test asserts the serving
calibrator and the eval calibrator produce bit-identical probabilities, so the
validated ECE describes the model that actually routes.

This is a fair criticism of M6: a mode that is fitted and validated but never
*executed* is not finished. It was invisible because the shipped default is
`raw` and gate `large` failed validation, so nothing ever exercised the path.

Verified against live models with `CONFIDENCE_MODE=calibrated`, and the
calibrator is visibly doing work rather than echoing `geo_mean`:

    "capital of France, one word"        gate small  P=0.9891  vs 0.92  -> accept
    "why is the sky blue then red"       gate small  P=0.1320  vs 0.92  -> escalate
                                           (geo_mean 0.4751 — a different number)
                                         gate large  P=0.1198  vs 0.85  -> escalate
                                           (geo_mean 0.5519)
                                         served by frontier

    CONFIDENCE_MODE=calibrated CALIBRATOR_PATH=/nonexistent.json
      -> RuntimeError: ... refusing to start and report a confidence the gates
         are not actually using.

This does **not** change the shipped default. Gate `large` still failed M6, so
`raw` remains correct; what changed is that flipping the switch now does what
the switch says.

### M11 — Streaming, by buffering first ✅ DONE (2026-07-28)

D8 said `stream: true` returns 400, and named its own reversal condition:
"buffer the local answer, stream only after the decision." That is what
shipped. The decision was reversed, not softened.

`app/streaming.py` takes a *finished* `ChatCompletionResponse` and yields
OpenAI `chat.completion.chunk` frames. Nothing above it changed — the cascade,
the gates and the caches run exactly as before, which is the point: a gate
cannot decide without the whole answer, so the answer is complete before the
first byte goes out.

**What it does not do is lower time-to-first-token**, and the docs say so in
those words (`check_docs.py` now enforces it). The feature is client
compatibility. Verified against the real OpenAI Python SDK rather than against
our own belief about the format:

    openai 2.49.0, stream=True, stream_options={"include_usage": True}
      -> reassembled text, exactly one role frame, usage frame present

Three properties fall out of buffering, and they are worth more than the TTFB
it doesn't win:

    escalating request, cache bypassed
      path        ['local_small', 'local_large', 'frontier']
      streamed    only the frontier's text — the two rejected local answers
                  appear nowhere in the SSE bytes
      x_hlr       still reports all three tiers were paid for

    both tiers down + stream=true
      -> 503 with a JSON body, not a truncated stream

    X-HLR-Tier / -Cost-Usd / -Saved-Usd survive streaming, because the routing
    decision is final before the response starts

The escalation case is the load-bearing test. A streaming implementation that
got the framing right and leaked a discarded 0.5B answer would pass a naive
format test and be wrong; `tests/test_streaming.py` asserts the rejected text
is absent from the raw bytes.

`STREAM_CHUNK_CHARS` (default 24) trades frame smoothness against ~200 bytes
of JSON envelope per frame. The chunker is lossless — a property test asserts
the deltas concatenate back to the source across unicode, code fences and
mixed whitespace, because a chunker that quietly eats a newline looks fine in
a terminal and is broken in rendered markdown.

### M12 — Tenancy, budgets, and a durable audit trail ✅ DONE (2026-07-28)

The last two entries under "deliberately deferred", and the two that most
changed what this *is*. Before them the gateway could tell you what a request
cost but not who owed for it, and the record of that decision was gone within
10k requests.

**Tenancy (`app/budgets.py`).** A key was a credential; now it can belong to an
account with a daily USD cap. `TENANTS` is JSON rather than a delimited string
because API keys may contain `:` or `=`, and mis-splitting one fails
*silently* — the key still authenticates while the spend lands on the wrong
tenant (D22).

Over budget **degrades rather than rejects** (D23): the frontier drops out of
the cascade and the local tiers carry on. This is one line —
`[t for t in tiers if not t.billed]` — because `app/cascade.py` is ordered data
(D14). Had the cascade been nested if/else it would have been a new branch
through every path. Measured against two configured tenants:

    acme, $0.0005/day cap
      request 1  frontier      spent $0.000313  remaining $0.000187
      request 3  frontier      spent $0.000628  OVER
      request 4  local_large   degraded=['budget_exhausted:acme']
                               reason="daily budget exhausted: $0.000628 of
                               $0.000500 spent today (UTC); frontier tier
                               withheld, so: served by local_large..."
      pinned frontier while over  -> HTTP 402
    globex, $5.00/day cap
      unaffected throughout      spent $0.000208  remaining $4.999792

Two caveats are stated in the docs rather than glossed, and `check_docs.py` now
enforces that they stay stated. The cap is **approximate by exactly one
request** — cost is not knowable until the provider replies, so it is checked
before and recorded after; the overshoot is bounded by `MAX_OUTPUT_TOKENS x
FRONTIER_PRICE_OUT` (~$0.04 at the defaults, logged at startup, asserted by a
test). And **Redis down means budgets are not enforced** — failing closed would
turn a cache blip into a total frontier outage, so it fails open and says so in
four places (D24, D25).

**Audit trail (`app/audit.py`).** Append-only JSONL, one line per request,
durable across restarts — which the capped telemetry stream never was. They
answer different questions and the docs are now checked for keeping them apart
(D26).

Writes never touch the request path: records go on a queue, a background task
batches them and writes in a worker thread. A full queue **drops** rather than
blocking, and counts every drop, because an audit trail with unreported holes
invites conclusions drawn from absence. One file per worker PID, because a line
carrying `reason` + `gates` + metrics can exceed `PIPE_BUF`, above which
concurrent `O_APPEND` is no longer atomic.

**Refusals are audited too.** Telemetry counts served traffic only, which is
right for a rate aggregate and wrong for a record of what the gateway decided —
a 402 *is* a spend decision. This gap surfaced while reconciling the audit file
against `/metrics` and finding 5 against 6; the discrepancy turned out to be an
all-time Redis counter versus a freshly-deleted file, but looking for it
exposed the real hole.

    5 requests audited, tenants ['acme','globex'], raw key present: False

**CI found a real bug in the shutdown path**, on 3.14 only, as a flaky rotation
test. The writer pulls a batch off the queue and then awaits a thread to write
it; `stop()` cancelled that task. But `asyncio.to_thread` cannot cancel a
*running* thread — so cancelling there either lost the batch (thread not yet
started; it was no longer in the queue, so the shutdown flush could not recover
it) or duplicated it (thread already running, completed anyway, then the
cancel handler wrote it again). Both were reproduced deterministically by
making the write slow.

Shutdown now **signals** with a queued sentinel and waits for the writer to
acknowledge; cancellation is the backstop for a wedged disk, not the mechanism.
A `_closing` flag covers the case where the queue was full and the sentinel
could not be enqueued — without it the writer drained everything and then
blocked on `get()` until the timeout fired, a 5-second stall on exit under
exactly the sustained load that filled the queue.

### M13 — Testing the two things tests kept missing ✅ DONE (2026-07-28)

Prompted by a direct question — *is it tested at a production grade?* — and
answered by counting rather than asserting. Fourteen real defects were found
across this project's life; **one** was caught by the unit suite. The rest came
from CI, the load test, the demo, screenshots, or reading code against docs.
Coverage sat at ~95% throughout.

That is not a bad suite. It is a suite aimed at one class of bug while this
project mostly produces two others.

**Rendering.** `static/dashboard.html` was 380 lines with exactly one assertion
about it ("contains no `http://`"). Its JavaScript was never parsed, let alone
run — which is how `${LABEL[r.tier] ?? r.tier || "—"}`, a SyntaxError, blanked
the entire page while 441 tests passed. Demonstrated still-open by injecting
the same defect and watching 686 tests, 123 doc checks and ruff all pass.

`tests/test_dashboard.py` now attacks it in four escalating steps: the JS
parses (`node --check`), every `el("x")` matches an `id="x"`, every fetched URL
is a real route, and `render()` executes against a real `/metrics` payload
under a DOM shim. Fault-injected against all four:

    original ?? / || SyntaxError   -> 5 failed
    renamed element id             -> 2 failed
    fetches a route that is gone   -> 1 failed
    runtime error inside render()  -> 4 failed

The fourth is why this beats a linter: it runs the renderer against JSON the
gateway actually produced, so a field renamed in `telemetry.py` breaks a test
instead of a panel.

**Wiring.** Unit tests verify components; nothing verified they were plugged
in. That gap produced the calibrated-mode bug — 99% coverage on
`confidence.py`, every calibrator test green, and `main.py` never passing them
to `execute()`.

`tests/test_wiring.py` boots the *real* lifespan and asserts every
`app.state.X` the request path reads is set. Then the more interesting half:
**every setting must either be proven to change behaviour or be explicitly
declared inert, with a reason.** Adding a knob now forces that decision instead
of allowing silence.

It found a dead one on its first run. `DEFAULT_MAX_TOKENS` and
`DEFAULT_TEMPERATURE` were defined in `config.py`, documented in
`.env.example` **and** in the table below ("When the request omits
`max_tokens`") — and read by nothing. The real defaults were hardcoded in
`schemas.py` and happened to match, so setting them did nothing and no test or
doc check could tell. `check_docs.py` passed them because they *were*
documented; documented and wired are different properties.

Both are now applied via `_apply_sampling_defaults`, shared by
`/v1/chat/completions` and `/route` — they must agree, because temperature and
max_tokens are part of the cache key (invariant 5), so a preview computed from
different defaults would lie about a hit.

Fault-injected:

    un-wire the sampling defaults          -> 2 failed
    un-wire the calibrators (the real bug) -> 1 failed
    lifespan forgets app.state.audit       -> 4 failed
    /route stops applying defaults         -> 1 failed

**One of my own tests was wrong, twice.** The `/route` version first asserted
something true either way, and the fixtures attached to the developer's live
Redis — so a cache hit from an earlier run served the request without calling
the model. CI has no Redis and would have stayed green: red locally, green in
CI, which is the worst arrangement. Both fixed; the fixtures now point at a
closed port.

707 unit tests (was 686), coverage 96%.

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
| `OLLAMA_TIMEOUT_S` | `120` | Read timeout for a local generation |
| `OLLAMA_CONNECT_TIMEOUT_S` | `5` | Connect timeout; short, so a dead Ollama degrades fast |
| `FRONTIER_TIMEOUT_S` | `60` | Read timeout for the frontier tier |
| `DEFAULT_MAX_TOKENS` | `512` | When the request omits `max_tokens` |
| `DEFAULT_TEMPERATURE` | `0.0` | When the request omits `temperature` |
| `BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failures before a tier is skipped |
| `BREAKER_COOLDOWN_S` | `30` | How long a tripped circuit stays open before one probe |
| `RETRY_ATTEMPTS` | `2` | Total attempts per tier call |
| `RETRY_BASE_DELAY_S` | `0.25` | Exponential backoff base, with full jitter |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | `text` | `text` \| `json` — JSON attaches the trace id per line |
| `WARM_MODELS_ON_STARTUP` | `true` | 1-token call per chat model in the lifespan, so the first real request skips the ~1.9 s cold load measured in M0 |
| `FRONTIER_BASE_URL` | `http://localhost:8001/v1` | Point at a real provider to go live |
| `FRONTIER_API_KEY` | `mock-key` | Bearer token |
| `FRONTIER_MODEL` | `gpt-4o-mock` | Escalation model name |
| `FRONTIER_PRICE_IN` | `2.50` | USD per 1M input tokens — **placeholder, verify before quoting** |
| `FRONTIER_PRICE_OUT` | `10.00` | USD per 1M output tokens — **placeholder, verify before quoting** |
| `EVAL_USE_REAL_FRONTIER` | `false` | M6 only; flips eval to a real API |
| `EVAL_JUDGE_MODEL` | `claude-sonnet-5` | Quality judge for eval |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache + telemetry |
| `CACHE_TTL_LOCAL` | `3600` | TTL for locally-served answers |
| `CACHE_TTL_FRONTIER` | `86400` | Longer — expensive answers are worth keeping |
| `SEMANTIC_ENABLED` | `true` | Kill switch for T1 |
| `SEMANTIC_THRESHOLD` | `0.95` | Cosine floor; conservative on purpose |
| `SEMANTIC_MAX_CHARS` | `500` | Above this, skip semantic caching |
| `STREAM_CHUNK_CHARS` | `24` | Characters per SSE content frame. Trades frame smoothness against ~200 bytes of envelope each; `0` sends one frame |
| `TENANTS` | `{}` | JSON: `{"acme": {"keys": [...], "daily_budget_usd": "5.00"}}`. JSON rather than a delimited string because API keys may contain `:` or `=` |
| `DEFAULT_DAILY_BUDGET_USD` | `0` | Cap for the implicit `default` and `anonymous` tenants. 0 = unlimited |
| `AUDIT_ENABLED` | `true` | Durable JSONL record of every routing decision, refusals included |
| `AUDIT_PATH` | `audit` | Directory. One file per worker PID — a line can exceed `PIPE_BUF`, above which concurrent `O_APPEND` is not atomic |
| `AUDIT_MAX_BYTES` | `52428800` | Rotate at 50 MB. Granularity is the batch, so a file may exceed this by up to one batch |
| `AUDIT_KEEP_FILES` | `5` | Rotated siblings retained; older are deleted |
| `AUDIT_QUEUE_SIZE` | `10000` | Full queue drops rather than blocking the request path. Drops are counted in `/healthz` |
| `AUDIT_FSYNC` | `false` | Per-record fsync. Off: a per-batch flush survives a process crash, which is the common failure |
| `CONFIDENCE_MODE` | `raw` | `raw` \| `calibrated` |
| `CALIBRATOR_PATH` | `eval/calibrator.json` | Read once at startup. Absent is fine in `raw`; absent in `calibrated` refuses to start rather than silently serving raw thresholds under a calibrated label |
| `THRESHOLD_SMALL` | `0.55` | Gate 1, `geo_mean`, raw mode. **Not comparable to `THRESHOLD_LARGE`** — different model, different distribution |
| `THRESHOLD_LARGE` | `0.62` | Gate 2, `geo_mean`, raw mode |
| `TARGET_SMALL` | `0.92` | Gate 1 acceptance probability, calibrated mode. Higher than gate 2 on purpose — escalating from here is nearly free |
| `TARGET_LARGE` | `0.85` | Gate 2 acceptance probability, calibrated mode. The spec's number, on the gate that spends money |
| `CONFIDENCE_STRATEGY` | `logprobs` | `logprobs` \| `self_consistency` |
| `TOP_LOGPROBS` | `5` | 0–20, server-validated |
| `ROUTER_SMALL_MAX_TOKENS` | `60` | Enter at 2a below this |
| `ROUTER_ESCALATE_MIN_TOKENS` | `600` | Above this, enter at `local_large` — **never frontier on length alone** (long ≠ hard: summarization is long and cheap). Frontier entry needs length *plus* other hardness features |
| `SEMANTIC_MAX_ENTRIES` | `10000` | LRU cap on the semantic index |
| `MOCK_LATENCY_MS` | `1800` | Mock frontier simulated latency (`mock_frontier` only) |
| `MOCK_JITTER_MS` | `400` | Mock frontier latency jitter |
| `MOCK_FAILURE_RATE` | `0.0` | Mock failure injection; raise to exercise M8 degradation |
| `EVAL_JUDGE_BASE_URL` | `https://api.anthropic.com/v1` | Judge provider; separate from the frontier tier |
| `EVAL_JUDGE_API_KEY` | *(unset)* | Key for the judge provider; M6 only |

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
| `hlr:stats:gates` | hash | gate `small` / `large` pass and fail counts |
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
tests/
  test_confidence.py   pure-function metrics — first tests written (M4)
  test_keys.py         normalization + hashing (M2)
  test_guards.py       semantic-cache guard cases incl. Australia/Austria (M5)
  test_metrics_reconciliation.py   the /metrics identities from API.md (M2+)
static/dashboard.html
docs/{ARCHITECTURE,PLAN,API,DECISIONS,RESEARCH}.md
CLAUDE.md  README.md  requirements.txt  requirements-eval.txt  .env.example  .gitignore
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
  | jq '{tier: .x_hlr.tier, similarity: .x_hlr.cache.similarity}'
# expect tier="cache_semantic", similarity >= 0.95

# T1 guard — must MISS despite high cosine
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"What is the capital of Australia?"}]}' \
  | jq '{tier: .x_hlr.tier, cache: .x_hlr.cache}'
# expect tier != "cache_semantic", cache.guard_rejected == "entity_mismatch"

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

- ~~**Streaming (SSE).**~~ **Done in M11**, by the route this entry sketched: buffer the whole
  cascade, then re-emit the finished answer as frames. It does not lower time-to-first-token and the
  docs say so; the win is client compatibility, plus two things a live stream cannot do — no token is
  ever retracted, and a failure stays an ordinary HTTP error instead of a truncated stream.
- **Redis 8 Vector Sets.** Available via `brew upgrade redis` (RESEARCH.md §4) but unnecessary below
  ~100k entries and risky to swap under a running server. `cache/semantic.py` is written behind an
  interface so it's a drop-in later.
- **A learned routing classifier.** The Protocol exists; the heuristic ships first because M6's eval
  is what would generate the training labels anyway. Measure, then learn.
- **A third local tier / 7B model.** RAM-blocked, not a design choice. Revisit on different hardware.
- ~~**Auth and rate limiting.**~~ **Done in M9/M10** — per-key auth, request bounds, and a
  Redis-backed sliding window shared across workers.
- ~~**Multi-tenancy.**~~ **Done in M12** — named accounts with daily USD caps, per-tenant
  attribution in `/metrics`, and degradation to local tiers rather than refusal when a cap binds.
  Still deferred within it: per-tenant *rate* limits and model allow-lists. Tenancy covers spend.
- ~~**Multi-worker uvicorn.**~~ **Done in M10** and verified with 4 real worker processes: the
  rate limiter moved to a Redis sorted set (it was the one genuinely broken piece — per-process
  windows meant a 60/min cap silently became 60N), and the semantic replica's version counter was
  confirmed to resync across processes rather than merely being correct on paper.
