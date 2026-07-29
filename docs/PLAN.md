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
only the 43 judged prompts call out (129 calls; the other 67 are auto-gradable and free).
Compute the actual figure from your configured rates before running — the
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

**A follow-up pass caught the drift this created**, and one more gap. Correcting
`DEFAULT_MAX_TOKENS` in code left `docs/API.md` still documenting
`max_tokens | int | 512` — a constant where a knob now exists, which is the same
lie as a knob wired to nothing, told from the other end. `check_docs.py` now ties
that table to `config.py` (133 checks).

Then a fault-injection sweep over the whole gate found one real hole: replacing
`secrets.compare_digest` with `==` passed all 707 tests. The security posture in
CLAUDE.md says "never add a comparison that short-circuits" and nothing enforced
it. Two structural tests now do — asserting the source, because a timing
measurement would be flaky on a shared runner:

    dashboard SyntaxError                 -> 5 failed
    calibrators un-wired                  -> 1 failed
    a setting wired to nothing            -> 3 failed
    geo_mean -> sequence probability      -> 15 failed
    cache key drops temperature           -> 3 failed
    cache key drops the system prompt     -> 1 failed
    budget check inverted                 -> 7 failed
    compare_digest -> ==                  -> 1 failed  (was: not caught)
    loop breaks on first key match        -> 1 failed  (was: not caught)

One of those injections was my own mistake worth recording: the first
"cache key drops temperature" attempt edited a docstring, so the pass it
produced meant nothing. An injection that does not change behaviour proves
nothing about the tests.

709 unit tests, 133 doc checks.

### M14 — A spend ceiling that ships armed ✅ DONE (2026-07-28)

The blocking item on the production list, found by measuring rather than
listing best practices:

    DEFAULT_DAILY_BUDGET_USD = 0   (unlimited)
    TENANTS                  = {}  (none)
    RATE_LIMIT_PER_MINUTE    = 60

    Ollama dies -> invariant 4 sends every request to the frontier
      ~$0.0016/req x 60/min x 24h = $138/day, PER CLIENT

The safety behaviour and the expensive behaviour are the same behaviour.
Per-tenant budgets could already bound it and default to unlimited, so the
protection was present and unarmed — the worst state for a control, because it
looks handled in the code and in the docs and stops nothing.

`MAX_DAILY_SPEND_USD` defaults to $10 and **ships armed**. The asymmetry with
per-tenant budgets is deliberate: a tenant budget is a business policy, this is
a safety limit, and a safety limit that defaults to off is how you learn about
it from an invoice (D27).

Verified with Ollama pointed at a dead port and the ceiling at $0.0006:

    req 1-3   200  tier=frontier          (invariant 4, as designed)
    req 4-5   503  spend_ceiling_reached  (bounded)
    spent     $0.000624 against $0.0006 — one request's overshoot

The new tests caught a bug in the change itself: when the ceiling binds,
`verdict` describes the `__all__` pseudo-bucket, and using `verdict.tenant` to
record spend would have attributed real money to a fake account *and*
double-counted it. Attribution now always uses `tenant_name`.

Stated rather than oversold: a daily ceiling is blunt. It bounds damage rather
than detecting the anomaly, and only after the money is spent. A breaker on the
*rate* of frontier escalations is the sharper instrument and is recorded as
D27's revisit condition.

### M15 — Observability and endurance ✅ DONE (2026-07-29)

Items 2 and 3 of the production list. They pair: the soak test is what you
point the scraper at.

**Prometheus.** `/metrics` returned custom JSON under a name the ecosystem
reserves for exposition format — the first thing anyone deploying does is point
a scraper at it and get a parse error. It now **negotiates**: `Accept:
text/plain` (what every scraper sends) gets exposition, anything else —
including curl's bare `*/*` — keeps the JSON the dashboard consumes.
`/metrics/prometheus` is unconditional, for the human who will not think about
Accept headers.

`app/prometheus.py` is a pure renderer, so the format's silent failure modes
are testable: a redeclared `# TYPE` makes Prometheus *drop* the series, an
unescaped quote truncates a line, a missing trailing newline can lose the last
sample. All three are asserted, and the registry raises on a duplicate rather
than emitting a metric that vanishes.

Health comes from the same `_health_snapshot` `/healthz` uses — two health
checks that could disagree would be worse than one. Breaker state is exported
numerically because Prometheus cannot alert on the string `"open"`.

**Soak.** `loadtest.py` runs 500 requests in seven seconds; every bug it can
find is a concurrency bug. `scripts/soak.py` asks the other question — is it
still the same process an hour later — and found three bugs on its first run,
all in itself:

    3 rps against RATE_LIMIT_PER_MINUTE=60 -> 684 "failures"
      ...which were 429s. The limiter working, counted as breakage.
    RSS "grew 40.9%"
      ...on a 20 MB heap, which is ordinary allocator movement.
    KeyError: 'tiers' at the end
      ...because the final /metrics probe was rate-limited too.

Fixed: 429 is its own category, the probe retries, and memory is judged on
medians of thirds rather than single windows.

Then the interesting one. An 8-minute run showed RSS flat at 32 MB for seven
minutes, one jump to 66 MB, then flat again — and the fixed check reported that
as "stable", because the absolute growth was under its floor. Trading a false
positive for a false negative is not an improvement. Watched for a further
minute under no load it stayed at exactly 72.1 MB, and a later run came back
down to 40.7 MB: transient allocation, not a leak.

So the check now reports the **shape** rather than a ratio, because the two
failure modes are different questions:

    still climbing at the end  -> FAIL, that is a leak
    jumped once and plateaued  -> WARN, not a leak, but size the host for
                                  the peak rather than the median

Measured, 8 minutes at 0.8 rps:

    381 requests, 12 windows, 0 failed
    RSS   32.0 -> 49.2 MB (peak 65.7)      fds 69 -> 68
    p95   15ms -> 15ms                     counters reconcile

The script states its own limit rather than implying otherwise: 30 minutes is
not 30 days, and a leak smaller than measurement noise is invisible.

751 unit tests, 135 doc checks.

### M16 — Release engineering, supply chain, runbook ✅ DONE (2026-07-29)

Items 4, 5 and 7. All three are "how do you operate this", which is why they
went together.

**One version.** `0.1.0` was hardcoded in `pyproject.toml` *and* in the FastAPI
constructor — the arrangement that produces a release reporting a version it is
not. `app/__init__.py::__version__` is now the only source; `check_docs.py`
asserts the rest agree, and the release workflow refuses a tag that disagrees.
Verified by setting `pyproject.toml` to `9.9.9` and watching the check fail.

**Pinned dependencies.** `fastapi>=0.115` meant an upstream release could ship
itself into a container build with no review, and the first anyone knew was a
behaviour change in production. Now exact pins including transitives, so the
image is reproducible — which only works if something bumps them, hence
Dependabot, whose PR *is* the review the ranges skipped. `pip-audit --strict`
runs in CI, because a pin freezes a CVE in place just as well as it freezes a
working version. Currently clean.

**A release path.** A Dockerfile that only ever builds in CI is not one: there
is nothing to deploy and nothing to roll back to. `.github/workflows/release.yml`
re-runs the full gate on the tag, publishes a digest-addressed image to GHCR,
and cuts the GitHub release from the matching `CHANGELOG.md` section.

Two details worth keeping. The notes are extracted **by tag**, not "the first
section" — the first section is `[Unreleased]`, and an earlier version of this
would have published an empty page. And the job fails when no section matches,
because a release nobody wrote notes for is a release nobody can review.

**A runbook.** `/healthz` reports problems and says nothing about what to do
about them; `docs/RUNBOOK.md` is that half. Organised symptom → cause → action,
because the middle step is the one that gets skipped.

Its opening observation is the one that matters for this system: it is built to
degrade rather than fail, so **the usual symptom of a serious problem is a
higher bill, not an outage**. Cost is the alarm, not the aftermath. It also
names the trap directly — when `spend_ceiling_reached` starts returning 503s, do
not raise the ceiling to make them stop; the 503s are the symptom and the
missing local tiers are the fault.

`check_docs.py` now asserts every signal the docs say to alert on has a runbook
entry. An alert with no next step is a page at 3am and nothing else. Verified by
deleting an entry and watching it fail.

751 unit tests, 161 doc checks.

### M17 — Security hardening ✅ DONE (2026-07-29)

Item 6. Started by looking for real problems rather than adding headers and
calling it done, which turned up one worth the whole milestone.

**A cross-tenant prompt leak, reproduced.** The caches are global — no tenant in
the key, and the semantic index searched by vector across everyone. That
sharing is deliberate and is where the savings come from. What was not
deliberate: a semantic hit's `reason` string echoed up to 60 characters of the
*matched* prompt, which may have been written by a different tenant.

Two tenants configured, acme asked "how much is the quarterly retention bonus
we pay", globex asked a near-identical question:

    globex tier   : cache_semantic
    globex reason : semantic cache hit at cosine 1.0000 (threshold 0.95);
                    matched 'how much is the quarterly retention bonus we pay'

acme's exact wording, in globex's response and in globex's logs.

**Honest severity.** The guards bound how different the two prompts can be —
equal number sets, equal capitalized entities, matching negation, cosine ≥ 0.95
— so a cross-tenant hit means near-identical questions. The first attempt at
this reproduction *failed* for exactly that reason ("Jane Doe" vs "Jane" is an
entity mismatch). Narrow, and not nothing.

Fixed two ways. Under `CACHE_SCOPE=global` the matched text is withheld and the
reason says so. Under `CACHE_SCOPE=tenant` both caches are isolated and the text
is shown, because a hit can only be your own (D28).

**Scoping the exact key was not sufficient**, and the tests caught that: the
semantic index is searched by *vector*, so a scoped key isolated the stored
answer while the search still matched another tenant's entry. Both needed
changing. Fixing one and believing it fixed both is the obvious mistake here,
so there is a test per cache.

Old entries stay usable: an empty scope hashes to exactly the digests produced
before the setting existed.

**The audit trail was world-readable.** 0644 files in a 0755 directory —
on a shared host, every local account could read who asked for what and what it
cost. Now 0700/0600, created with `os.open(..., 0o600)` rather than chmod'd
afterwards so there is no window where the file is readable, and the mode
survives rotation.

**Browser-side headers**, because the gateway serves HTML from the same origin
as an API that spends money: `X-Content-Type-Options`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, and `Cache-Control: no-store` on `/v1/*` so
prompts do not sit in an intermediary cache.

Two deliberate omissions, both tested so they stay deliberate. **No HSTS** —
this process should not terminate TLS, and asserting HSTS from a plaintext
origin either does nothing or is stripped by the proxy that should own it.
**No CORS middleware** — with none configured a browser blocks cross-origin
reads by default, so the only thing a CORS policy could do here is make it less
safe.

**A remote unauthenticated Redis now warns at startup.** It is a read/write
handle on every cached answer, every budget counter and every rate-limit window
— including the ability to zero a counter and uncap spend.

765 unit tests, 165 doc checks.

### M18 — Shadow sampling ✅ DONE (2026-07-29)

The last production item, and the one that changes what this project *is*.

Every claim about the confidence gate rested on M6: one offline run, 67
auto-gradable prompts, and gate `large` failing its validation. Nothing since
would have noticed the thresholds drifting — a prompt distribution shifts, a
model gets updated, and the gate degrades silently while the answers keep
coming and nothing looks broken. That is exactly the failure this design exists
to make visible, and it was the one thing the design could not see.

`SHADOW_SAMPLE_RATE` re-runs a fraction of gate-**accepted** local answers
against the frontier, out of band, and records the agreement. The caller never
waits and never sees it; the result lands in `/metrics`, Prometheus and the
audit trail as a continuously updated false-cheap rate.

Verified live at rate 1.0 against the mock:

    3 requests: 2 served by local_small (sampled), 1 by frontier (correctly not)
    sampled 2, completed 2, failed 0, cost $0.00177

**And that run produced a trap worth closing.** It reported
`divergence_rate: 1.0` at `mean_similarity: 0.03` — which is a fact about the
*mock*, whose content is canned (D9), not about the gate. Someone enabling
shadow mode against the mock would conclude their gate was catastrophically
broken. Startup now warns loudly when shadow sampling is on and the frontier
looks like the mock; verified by starting one and reading the log.

Three properties are asserted rather than assumed, because each would turn the
instrument into a liability:

- **It cannot break what it measures.** A failing frontier costs a measurement,
  never a request.
- **It cannot outspend what it measures.** Shadow calls are billed to the
  tenant, count against the ceiling, and are skipped when either is exhausted.
- **It samples the right thing.** Cache hits were checked when produced and a
  frontier answer *is* the reference; shadowing either pays to compare
  something with itself.

Sampling is deterministic in the trace id so an anomaly is reproducible — and
hashed rather than `int(trace_id, 16) % N`, because the first version sampled
2000/2000 at rate 0.5 on sequential ids. The modulo of a raw id is uniform only
if the ids are.

Similarity is Jaccard over word sets, deliberately not a judge model: a judge
would double the cost and put a second model's opinion inside a number meant to
be a tripwire. **The absolute value is not a quality score** and the docs say so
in three places.

794 unit tests, 169 doc checks.

### M19 — Mutation testing and multi-worker in CI ✅ DONE (2026-07-29)

The two testing gaps the README had been naming since M13.

**Mutation testing.** Coverage says a line ran; this asks whether a test would
*fail* if that line were wrong. Scoped to the pure modules in `setup.cfg` —
against I/O code most mutants are unkillable noise (a reordered await, a changed
log line) and they bury the findings. The tool is only useful while its output
is short enough to read.

1953 mutants in ~3 minutes, 29 survivors on the first run. Six were real:

    the global budget counter ignored the `now` it was given, so at a UTC day
      boundary the tenant and gateway totals landed in different windows and
      silently disagreed about the same request
    a zero-cost request wrote to Redis twice — on the fastest path in the system
    a 1 micro-USD charge was dropped rather than recorded
    ns_to_ms could be rescaled or re-rounded and nothing noticed
    max_overshoot_micro's per-million divisor was unasserted at any scale that
      could separate 1_000_000 from 1_000_001
    sequence_probability([]) — a deliberately-unused counter-example whose
      behaviour was unpinned, so it could have taught the wrong thing

The first fix for `ns_to_ms` did not work, which is itself the lesson: a
one-in-a-million divisor error needs ~1e11 ns before it moves the first decimal
place, so asserting at small values cannot separate them. The test now asserts
at scale.

Baseline after: **1953 mutants, 19 survivors**, all either a changed detail
string (the prose explaining a guard rejection, a log message) or provably
equivalent — `similarity()`'s `or` becoming `and` computes 0/n = 0.0 on the
same input, so nothing observable changes. Neither is worth a test; asserting
exact prose is brittle and equivalent mutants are unkillable by definition.

**Not a CI gate**, deliberately. Minutes rather than seconds, and a percentage
target would push someone into writing brittle tests to chase equivalent
mutants. Weekly, reporting.

**Multi-worker in CI.** M10 verified shared state by hand, once, and never
again — which for this class of bug is close to not verifying it. A
per-process fallback does not *fail*; it works, in each worker separately, and
the only symptom is a limit quietly multiplied by the worker count.

`scripts/multiworker_check.py` runs against four real processes, no Ollama —
requests are pinned to the mock frontier, since the properties under test are
about Redis and process boundaries. Verified in both directions, which is the
part that matters:

    with Redis         distributed ok | cache 6/6 | single-flight 8/8 by 1 call
                       | rate limit 3 of 60 admitted            -> PASS
    Redis unreachable  distributed FAIL | cache 0/6 | rate limit 45 of 60
                       admitted (the per-process multiplication) -> FAILED

That second run exposed a flaw in the check itself: single-flight *passed*
vacuously, because with no Redis the telemetry delta is 0 and 0 slipped under a
`<= 2` bound. A vacuous pass is worse than a failure — it is a failure you have
been told not to worry about. The bound is now `1 <= calls <= 2`.

799 unit tests, 175 doc checks.

### M20 — Property-based tests ✅ DONE (2026-07-29)

The last entry on the "still not covered" list.

Example-based tests assert what a function does for inputs someone thought of.
These assert what must be true for inputs nobody thought of — the right
instrument for the parts of this system that are *mathematical claims* rather
than behaviour.

The central one is `geo_mean`'s **length invariance**. The entire design rests
on it (D3): sequence probability decays exponentially with length, so a fixed
threshold against it measures length rather than certainty. That claim had been
carried by two hand-picked examples and a docstring. It is now quantified over
generated inputs — including the stronger form, that repeating a token sequence
does not change the score — alongside the counter-claim that
`sequence_probability` genuinely *does* decay, because a counter-example that
stopped behaving as described would quietly stop making its point.

**It found a real bug on the first run.** A carriage return in a Prometheus
label value split one sample across two lines:

    hlr_thing{label="
    "} 1

Two fragments, neither of which parses. `escape_label_value` handled the three
escapes the format defines — backslash, quote, line feed — and nothing else.
Tenant names come from operator JSON and tier names from config; neither is
guaranteed tame.

Fixed, and then **`\x85` (NEL, a C1 control) did the same thing** — which is
the argument for enumerating the whole class of line terminators rather than
the character that happened to fail. Python's `splitlines()` breaks on
`\v \f \x1c \x1d \x1e \x85 \u2028 \u2029` as well as `\r\n`, and the escaper
now replaces all of them.

**Then it found two bugs in the test itself**, which is the same instrument
doing the same job one level up:

- the assertion counted quotes not preceded by a backslash, and reported
  `label="\\"` — a valid value containing one backslash — as unbalanced;
- the round-trip un-escaped with a chain of `.replace()` calls, which on the
  two-character string `\n` (backslash, n) undoes `\n` before `\\` and
  produces a real newline. Sequential replace on escape sequences is
  order-dependent and wrong.

The test now scans properly and compares against the contract directly.

Property testing and mutation testing are complementary and neither replaces
the other: mutation asks whether the existing assertions would catch a change,
property testing asks whether the assertions cover the right inputs. This run
is the evidence — mutation testing had already been over `prometheus.py` and
found nothing here, because no assertion existed to mutate.

823 unit tests, 179 doc checks.

### M21 — The real-frontier calibration ✅ RUN (2026-07-29), partial

The deferred item, finally run against a real provider. **The answer did not
change: `should_flip()` is still False and the gateway stays in `raw`.** What
changed is how well-founded that answer is — RESEARCH.md §3.5 has the numbers.

Gate `large` reproduced M6 exactly (ECE 0.047, same Brier-vs-baseline failure).
Gate `small` flipped PASS → FAIL, but the ECE intervals overlap heavily, so the
two runs are not statistically distinguishable — what the flip shows is that
M6's pass was never robust, not that anything degraded.

**It stopped at 71/110.** The free tier caps at 20 requests per day *per model*
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), so 32 judged prompts
were never collected. `--fit-only` exists now to fit whatever is cached without
contacting a provider.

**Eight bugs surfaced, every one silent**, in code that had never been run
against a real API:

    run_eval never read .env            os.getenv, against this project's own
                                        convention; the file was ignored here
                                        and honoured everywhere else
    judge max_tokens hardcoded to 10    a REASONING model spends hidden thinking
                                        tokens against that budget -> empty
                                        string -> every verdict None -> 129 paid
                                        calls, zero labels, no error
    reference shared the local budget   references truncated to "The
                                        mitochondrion is an"; the judge compared
                                        full answers against four words
    a partial run overwrote the         --limit 3 replaced eval/calibrator.json
      production calibrator             with a 3-sample fit
    --limit could not reach the         the dataset starts with auto-gradable
      paid path                         prompts, so smoke tests only ran free ones
    the judge parser was too literal    gemini-2.5-flash says ACCEPT/UNACCEPT,
                                        the parser wanted ACCEPTABLE; it
                                        discriminated perfectly and scored nothing
    parse failures were undiagnosable   the raw verdict was discarded, so a
                                        missing label could not be explained
    429 bodies were truncated at 300    Gemini puts "retry in 23.7s" after ~430
      chars, discarding the retry hint  chars of boilerplate, so backoff went
                                        blind: 30+60+120+240+480s on ONE call,
                                        then killed the run at 71/110 twice

The last one was mine, introduced while fixing the one above it — I wrote
hint-honouring backoff without checking the hint survived to the exception.

**The evidence is now committed.** `eval/runs/full.json` was gitignored, so 119
KB of paid API responses backing a documented result lived on one laptop with
no way to recreate it that day. The docs said "never re-spend to recompute
something already on disk"; that only holds if the disk is version-controlled.

889 unit tests, 185 doc checks.

---

## M22 — the complete run, and why its result was not believed (2026-07-29)

Billing removed the daily quota, so all 110 prompts collected in 483s and
`should_flip()` returned **`True`** for the first time. It was not flipped.

The eval generated local answers at `--max-tokens 120` while references got
1024 and the gateway serves 512. So the judge was asked whether a candidate
matched a reference that was 2.9x longer, **26% of local answers were truncated
mid-sentence**, and every `reasoning` prompt was — grading that category 0/26,
dragging the acceptable rate from 85% to 72%, and worsening the base-rate Brier
from 0.127 to 0.202. Gate `large` "passed" partly by clearing a lower bar.

Refitting on subsets of the same run — free, the cache is on disk — showed the
verdict was not robust: `False` with `reasoning` excluded, `True` by 0.006 Brier
with truncated answers excluded. RESEARCH.md §3.6 has the full table.

Fixed:

| | |
|---|---|
| `--max-tokens` defaulted to 120 | now `DEFAULT_MAX_TOKENS` — what the gateway serves |
| a run cache could mix two budgets | `check_budget` refuses; `n_tokens` is a calibrator feature |
| paid calls recorded no usage | `ApiUsage` per call; the run prints what it cost |
| the judge was priced at the reference's rate | `EVAL_JUDGE_PRICE_IN`/`_OUT` — a different model bills differently |

The harness tracked tokens precisely for the free local tiers and not at all
for the calls that cost money, so pricing a run meant estimating from character
counts afterwards. `ApiUsage` keeps the provider's usage block whole: reasoning
models bill hidden thinking as output under keys that vary by provider, and a
fixed schema drops exactly the tokens likeliest to surprise you.

---

## M23 — the calibration passes, and it survives being taken apart (2026-07-29)

Re-ran M22's collection with the two defects fixed: local tiers at
`--max-tokens 512` (what the gateway serves) and the judge at
`reasoning_effort=none`.

| gate | n | acceptable | ECE | Brier | base | verdict |
|---|---|---|---|---|---|---|
| `small` | 110 | 59% | 0.040 | 0.136 | 0.242 | **PASS** |
| `large` | 110 | 69% | 0.027 | 0.129 | 0.214 | **PASS** |

`should_flip()` is `True` and **stays** `True` with `reasoning` removed, with
`code` removed, with truncated references removed, and with capped local
answers removed. It fails only when both hard categories go at once, leaving 75
easy prompts — the over-easy distribution that made M6 uninformative. That is
the expected failure, not a counter-example.

Label coverage went from 96 to 110 per gate with zero unlabelled slots; local
truncation from 26% to 9%.

**Cost, measured rather than estimated** — 129 calls, in 61,656 / billed out
31,783. `reasoning_effort=none` gave 2.7 output tokens per verdict with no
thinking. On the reference, thinking exceeded visible output by 36% and was
reported *only* as the gap in `total_tokens` — Gemini's OpenAI layer sends no
`reasoning_tokens` key, so `billable_out` derives it from the arithmetic.

**Provenance shipped with the fit.** `calibrator.json` now records the models,
token budget, dataset size and judge, and the lifespan refuses to start in
`calibrated` mode when they disagree with what is configured. A calibrator
fitted on other models is worse than none: it returns a number in [0, 1] that
looks exactly like a probability.

**The shipped default stays `raw`** — a portability decision, not a verdict on
the calibration. See RESEARCH.md §3.7.

Also hardened: `.gitignore` went from `.env` + `.env.local` (exact matches) to
`.env*`, after a routine `cp .env .env.backup` produced an untracked file
holding a live API key that `git status` offered to stage.

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
| `SHADOW_SAMPLE_RATE` | `0.0` | Fraction of gate-accepted local answers re-run against the frontier to measure false-cheap. Off by default: it spends real money (D29) |
| `SHADOW_DIVERGENCE_THRESHOLD` | `0.35` | Token overlap below which a shadow comparison counts as divergent |
| `EVAL_JUDGE_MAX_TOKENS` | `64` | Eval only. A reasoning judge spends hidden thinking tokens against this; at 10 it returns nothing and every verdict parses as None |
| `EVAL_REFERENCE_MAX_TOKENS` | `1024` | Eval only. Separate from the local tiers' budget — a truncated reference makes the judge compare against a fragment |
| `EVAL_MIN_CALL_INTERVAL_S` | `0.0` | Eval only. Pace under a rate limit rather than retry through it; Gemini's free tier is 20/min |
| `EVAL_JUDGE_PRICE_IN` | `2.50` | Eval only. The judge is a different model from the reference (D11), so it bills at different rates. Placeholder — verify |
| `EVAL_JUDGE_PRICE_OUT` | `10.00` | Eval only. As above. `run_eval` prints what a run cost from provider-reported tokens, at these prices |
| `EVAL_JUDGE_REASONING_EFFORT` | *(empty)* | Eval only. Passed through when set. `none` disables thinking on gemini-2.5-flash; Gemini 3 has no off switch; Anthropic ignores the field. Empty by default or it would 400 against the default judge |
| `EVAL_REFERENCE_REASONING_EFFORT` | *(empty)* | Eval only. Leave alone — the reference is the gold standard every label is graded against |
| `CACHE_SCOPE` | `global` | `global` \| `tenant`. Global shares cached answers across tenants (the savings); tenant isolates them (D28) |
| `MAX_DAILY_SPEND_USD` | `10.00` | **Armed by default.** The gateway's own daily ceiling across all tenants, bounding the cost of invariant 4. 0 disables |
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
