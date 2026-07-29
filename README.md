# LLM Cost Optimizer

An API gateway that sits in front of several models of escalating cost and decides — per request —
the cheapest one that will actually do the job.

> **Status: M0–M25 complete. The gateway runs, and it's hardened.**
> All five tiers, both confidence gates, the eval harness and the dashboard are built and working —
> **906 unit tests** (hermetic, no network) plus 10 integration tests against live models. Auth,
> rate
> limiting, per-tenant budgets and rate limits, a durable audit trail, circuit breakers, structured
> logging and a container image are in. Run
> `python3 scripts/demo.py` to walk every tier in one go.
>
> **The calibration passes**, measured against a real frontier API: both gates clear their
> validation at n=110, and the result survives every counterfactual refit but one that strips the
> two hardest categories at once. 0.85 is earned — [RESEARCH.md §3.7](docs/RESEARCH.md).
>
> **`calibrated` is the shipped default** as of 0.4.0, so `0.85` means what it says out of the box.
> That works because `eval/calibrator.json` is committed and was fitted on the models this repo also
> defaults to. **Swap `SMALL_MODEL` or `LARGE_MODEL` and the gateway refuses to start**, naming both
> model sets — a calibrator fitted elsewhere still returns a number in [0, 1] that looks exactly like
> a probability. Run other models with `CONFIDENCE_MODE=raw`, or refit.
> The earlier negative result is still in [§3.4](docs/RESEARCH.md), not quietly deleted.

Point any OpenAI-compatible client at it. It answers from cache when it can, tries a 0.5B model
first, steps up to a 3B when that isn't confident enough, and reaches for the expensive API only
when it can justify the spend. Every response reports which path it took, why, and what it cost.

```
   your app  ──▶  gateway :8000  ──┬──▶  exact cache (Redis)   $0      ~1 ms
                                   ├──▶  semantic cache       ~$0     ~20 ms
                                   ├──▶  qwen2.5:0.5b          $0    ~120 ms
                                   ├──▶  qwen2.5:3b            $0    ~450 ms
                                   └──▶  frontier API          $$$     ~2 s
```

---

## Why

Most prompts hitting a production LLM app are cheap ones — summarize this, reformat that, answer a
factual question. Paying frontier prices for all of them is the default failure mode, and it's
invisible until the bill arrives.

Routing on prompt *shape* alone ("is it short? is it a summary request?") is easy but shallow. The
harder and more interesting question is:

> **Can we tell when the cheap model's answer is good enough?**

This gateway answers that using the model's own **token log-probabilities**. When a model is
uncertain — hesitating between tokens, spreading probability mass thin — that shows up numerically
before you ever read the answer. Low confidence escalates. High confidence saves the money.

That's a testable claim, so the project is built to test it: `eval/` measures cost against quality
across a labeled prompt set and calibrates the thresholds, rather than asserting a number and hoping.

## How it works

A request descends five tiers and stops at the first one that can serve it.

| Tier | What it does | Cost | p50 | p95 |
|---|---|---|---|---|
| **Exact cache** | Hash of the normalized prompt + model + sampling params | $0 | **0.2 ms** | 0.3 ms |
| **Semantic cache** | Embedding similarity — catches rewordings of a question you've already asked | ~$0 | **47 ms** | 92 ms |
| **Small SLM** | `qwen2.5:0.5b`, with a confidence score from its logprobs | $0 | **312 ms** | 2.5 s |
| **Large SLM** | `qwen2.5:3b`, same treatment — reached only when the 0.5B wasn't sure | $0 | **4.5 s** | 4.6 s |
| **Frontier** | Escalation over real HTTP to an OpenAI-shaped endpoint | metered | **1.7 s** | 2.1 s |

**Measured 2026-07-29** on an Apple M2 with 8 GB, models warm, `max_tokens=120`. These rows are *not*
comparable as "cost per tier": the tiers answer different questions, and generation time is close to
linear in output length. The 3B's 4.5 s is `36.4 tok/s × ~120 tokens` — **not CPU load**, since
`ollama ps` reports 100% GPU and load time was 0.15 s. Full method and the per-model token rates:
[docs/RESULTS.md](docs/RESULTS.md).

Between the caches and the models, a **pre-router** looks at the prompt — length, code fences,
reasoning markers, arithmetic, attached context — and picks where to *enter* the cascade:

- `local_small` — clearly simple, start at the 0.5B
- `local_large` — start at the 3B, skip a hop that would obviously fail
- `frontier` — clearly hard, skip both local models rather than waste them

From there the **confidence gates** take over. The router's guess is allowed to be wrong in one
direction only: a *too-low* entry costs latency, because the gates still catch a weak answer and
escalate it. A *too-high* entry — straight to `frontier` — is ungated spend with no gate to catch it,
which is why frontier entry is deliberately rare and why the eval measures its precision
(the false-expensive rate).

**Streaming works, and the honest version of how:** `stream: true` returns SSE that any
OpenAI-compatible client reads, but it **does not lower time-to-first-token**. A gate can't decide
whether to escalate until the full answer and its logprobs exist, so the cascade finishes first and
the completed answer is then re-emitted as frames — *buffer-then-stream* ([D8](docs/DECISIONS.md)).
The feature is client compatibility, not latency.

What buffering buys instead: no token is ever retracted (you never watch the 0.5B's answer get
overwritten by the frontier's), errors stay ordinary HTTP errors rather than truncated streams, and
the `X-HLR-*` headers survive because the routing decision is final before the first byte.

## Reading the numbers

Every response carries an `x_hlr` block (**h**ybrid **L**LM **r**outer — the gateway's routing
metadata). *(Illustrative — invented values showing the shape.)*

```json
{
  "x_hlr": {
    "tier": "local_large",
    "path": ["local_small", "local_large"],
    "gates": [
      {"gate": "small", "confidence": 0.41, "threshold": 0.55, "verdict": "escalate"},
      {"gate": "large", "confidence": 0.68, "threshold": 0.62, "verdict": "accept"}
    ],
    "confidence": 0.68,
    "confidence_mode": "raw",
    "reason": "explanatory 'why' question, 41 tokens; 0.5B uncertain on causal chain",
    "latency_ms": {"route": 0.4, "cache_exact": 1.2, "cache_semantic": 18.9,
                   "local_small": 118.0, "local_large": 441.0, "total": 580.9},
    "tokens_in": 41, "tokens_out": 156,
    "cost_usd": 0.0,
    "baseline_cost_usd": 0.00166,
    "cost_saved_usd": 0.00166
  }
}
```

- **`tier`** — who actually served the answer.
- **`path`** — every tier attempted, in order. This request paid for two model calls; the metadata
  says so rather than hiding it.
- **`gates`** — each gate's score and verdict. The routing decision, exposed.
- **`reason`** — plain English, why the router decided what it decided.
- **`cost_saved_usd`** — `baseline_cost_usd − cost_usd`, summed on the dashboard.

**How savings are counted — two assumptions you should know about.** Local model calls are booked at
**$0**: electricity and amortized hardware are real, but not meaningfully attributable per request,
and inventing a number would make the figure look more rigorous while being less honest. And the
baseline charges frontier prices for *every* request, including ones a cache served — it answers
"what would this traffic have cost with no gateway at all", which is the comparison that matters, but
it does flatter the cache tiers. Rationale for both choices: [D13](docs/DECISIONS.md).

`GET /metrics` gives aggregates; `GET /dashboard` renders them live — tier breakdown, cache hit
rate, per-gate escalation rates, latency percentiles, and cumulative spend against the baseline.

### About that 85% threshold

The original spec said *escalate when confidence falls below 85%*. Getting that to mean something
real took some care, and it's worth understanding.

The obvious approach — multiply every token's probability together — **doesn't work**. That number
shrinks exponentially with length. A 100-token answer where the model was a healthy 90% sure of every
single token scores 0.000027. Threshold it at 85% and you escalate literally everything. It measures
length, not certainty.

The fix is the **geometric mean** of token probabilities (equivalently, inverse perplexity), which is
length-invariant. But even that isn't a probability of being *right* — a small model producing a
perfectly correct answer scores around 0.5–0.75 (M0 measured 0.55–0.75 across correct answers on the
0.5B), simply because language offers many valid ways to phrase the same thing.

So the gateway ships in two modes:

- **`raw`** (default) — confidence is the geometric mean, thresholds 0.55 and 0.62. Works
  immediately, and the number is *not* comparable to 85%. Nor are the two thresholds comparable to
  *each other*: they apply to different models with different score distributions.
- **`calibrated`** (after `eval/run_eval.py`) — a fitted model maps the raw signals to
  `P(this tier's answer is as good as the frontier's)`. **Now the number means what you'd expect.**
  Gate 2 accepts at **0.85** — the spec's number, on the gate that actually spends money. Gate 1's
  bar is deliberately *higher* (0.92): escalating from it costs ~450 ms and no dollars, so when the
  next tier is nearly free you should use it readily.

**Both gates now pass, and it took three runs to get an answer worth trusting.** M6 fitted gate 1
cleanly and gate 2 failed — not on calibration but on *usefulness*: its ECE was the better of the
two (0.047), because a model predicting the base rate everywhere is trivially well-calibrated and
carries no information. Brier-against-baseline is what separates "calibrated" from "calibrated and
useful" ([§3.4](docs/RESEARCH.md)).

A later run against a real frontier API appeared to pass both gates — and didn't survive scrutiny.
The eval was generating local answers at 120 tokens while references got 1024, so the judge was
comparing candidates against references three times longer, every `reasoning` prompt was truncated,
and that category graded 0/26. Refitting on subsets showed the verdict flipping depending on which
defensible slice you took ([§3.6](docs/RESEARCH.md)).

Corrected — local tiers at the length production actually serves — **both gates pass at n=110**
(`small`: ECE 0.040, Brier 0.136 vs base 0.242; `large`: ECE 0.027, Brier 0.129 vs base 0.214), and
the result now holds under every counterfactual refit except one that removes both hard categories
at once. **0.85 is earned** ([§3.7](docs/RESEARCH.md)). Full derivation in
[docs/RESEARCH.md](docs/RESEARCH.md) §3.

`raw` remains the default for portability, not doubt: the fitted weights describe `qwen2.5:0.5b` /
`qwen2.5:3b` on this dataset. Flipping `CONFIDENCE_MODE` loads `CALIBRATOR_PATH` at startup and
**refuses to start** if the fit is missing — or if its recorded provenance names different models
than the ones configured, since a mismatched calibrator returns numbers that look exactly like
probabilities and aren't.

**And M0 already found the limit of the raw metric.** Asked when the Treaty of Kadesh was signed
(really c. 1259 BC, Egypt and the Hittites), `qwen2.5:0.5b` answered *"1956 … between Egypt and
Israel … ended the Six-Day War ongoing since 1967"* — wrong in every particular, internally
contradictory, and it scored `geo_mean` **0.5535**, so the 0.55 gate **accepted it**. On the same
model a genuine struggle through an algebra proof scored *lower* (0.5468) and was correctly escalated.

> Raw `geo_mean` measures fluency, not factuality — and a small model is most fluent exactly when
> it's confabulating from a familiar-sounding template.

That's a measured false-cheap at the shipped default, written up in
[RESEARCH.md §3.2](docs/RESEARCH.md). It's the strongest argument for why calibrated mode and the
7-feature vector aren't optional polish. It's recorded rather than papered over by nudging the
threshold: tuning 0.55 upward to exclude one example would fit to that example and leave the actual
problem intact.

## Getting started

```bash
ollama serve & brew services start redis
ollama pull qwen2.5:0.5b && ollama pull qwen2.5:3b && ollama pull nomic-embed-text
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && cp .env.example .env
python3 scripts/preflight.py                 # verifies models, RAM, disk, logprobs
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &
python3 scripts/demo.py                      # walks all five tiers
```

Then open <http://localhost:8000/dashboard>.

Full instructions, including the 8 GB constraints this was designed around:
**[docs/SETUP.md](docs/SETUP.md)**.

## Guides

| | |
|---|---|
| **[RESULTS.md](docs/RESULTS.md)** | What was measured, and what it does not show |
| **[SETUP.md](docs/SETUP.md)** | Install it, and whether your machine can host it |
| **[RUNNING.md](docs/RUNNING.md)** | Start it, point it at a real provider, containers, health, load |
| **[USAGE.md](docs/USAGE.md)** | Call it: auth, tenants, budgets, rate limits, streaming, the audit trail |
| **[API.md](docs/API.md)** | The exact HTTP contract — every field and error code |
| **[RUNBOOK.md](docs/RUNBOOK.md)** | When it misbehaves, organised by symptom |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | The development loop and what CI checks |

## How it's tested

Contributing? [CONTRIBUTING.md](CONTRIBUTING.md) has the loop these three
commands belong to, and what CI will check that they do not.

```bash
.venv/bin/python -m pytest -m "not integration"   # 906, hermetic — needs nothing running
.venv/bin/ruff check app eval scripts mock_frontier tests
.venv/bin/python scripts/check_docs.py            # 208 cross-document checks
```

Six layers, split by what each can and cannot see. The split isn't aspirational —
it came from counting where defects were actually found, and noticing the unit
suite had caught **one of fourteen**.

| Layer | Catches | Blind to |
|---|---|---|
| **Unit** (906) | logic inside a module | whether modules are connected |
| **Wiring** | components not plugged in; settings wired to nothing | anything the real lifespan can't reach |
| **Rendering** | dashboard JS that doesn't parse, run, or match the DOM | visual layout |
| **Doc checks** (208) | docs contradicting the code | whether the code is right |
| **Integration** (10) | real models, real Redis, the real OpenAI SDK | sustained load |
| **Load** | cost identities under concurrency | correctness of one request |
| **Soak** | memory/fd growth, latency drift over time | anything faster than a window |
| **Multi-worker** | state that stopped being shared across processes | single-process logic |
| **Mutation** | assertions that would not fail if the code were wrong | anything outside the pure modules |
| **Property** | inputs nobody thought of, over the pure functions | anything stateful |

Two things worth knowing:

**Coverage is measured and is deliberately not a gate.** It sat at ~95% while
`CONFIDENCE_MODE=calibrated` was wired to nothing and `DEFAULT_MAX_TOKENS` was
read by nothing. Every line ran — they just ran disconnected. `test_wiring.py`
exists because coverage can't express *"and then it is used"*, and it found the
dead setting on its first run.

**The unit suite is hermetic and CI keeps it that way**, re-running it behind a
dead proxy so a test that starts reaching the network fails there rather than
becoming flaky later.

**Mutation testing** grades the assertions themselves — would a test *fail* if the code were wrong,
rather than merely execute the line? `./scripts/mutation.sh` mutates the pure modules; the current
baseline is 1953 mutants, 19 survivors, all of them changed prose or provably equivalent. The run
that established it found six real gaps, including a budget counter that ignored the window it was
given and would have made the tenant and gateway totals disagree across a UTC day boundary.

It is **not** a CI gate — it takes minutes and its output needs reading, not thresholding. It runs
weekly and reports.

**Multi-worker state is verified in CI** with four real worker processes against Redis: the rate limit
is one shared window, the cache spans workers, single-flight collapses duplicates. Verified in both
directions — with Redis unreachable the check *fails*, which is the point, since a per-process
fallback doesn't break anything, it just quietly multiplies your limit by the worker count.

**Property-based tests** assert what must be true for inputs nobody thought of. The central one is
`geo_mean`'s **length invariance** — the claim the whole design rests on ([D3](docs/DECISIONS.md)) —
now quantified over generated inputs rather than two hand-picked examples, alongside the counter-claim
that `sequence_probability` genuinely *does* decay with length.

They found a real bug on the first run: a carriage return in a Prometheus label split one sample
across two lines. Then `\x85` (NEL) did the same after the first fix — which is the argument for
handling the whole class of line terminators rather than the character that happened to fail.

### Measuring the claim, continuously

The gate's premise — *logprobs tell you when the cheap answer is good enough* — was validated **once**,
offline, on 67 prompts, and one of the two gates failed. Nothing since would notice it drifting.

```bash
SHADOW_SAMPLE_RATE=0.01
```

That re-runs 1% of gate-**accepted** local answers against the frontier, out of band, and records how
much the two agree. The caller never waits for it and never sees it; the result lands in `/metrics`
and the audit trail as a false-cheap rate that updates continuously.

**Off by default, because it spends real money** — a shadow call is a real frontier call for an answer
nobody reads. That is the honest cost, and it is the mirror of `MAX_DAILY_SPEND_USD`: one must be
switched on deliberately, the other switched off deliberately. Shadow spend is billed to the tenant,
counts against the ceiling, and is skipped when either is exhausted.

Three things stated rather than glossed:

- **It cannot break what it measures.** It runs after the response; a failing frontier costs a
  measurement, never a request. There is a test for exactly that.
- **It measures false-cheap only** — answers served locally that should have escalated. It says
  nothing about unnecessary escalations, which cost money rather than quality.
- **It needs a real frontier.** The mock's content is canned, so every comparison diverges and the
  rate reads 100%. Startup warns loudly if you point it at the mock — verified, because that
  misconfiguration looks exactly like a catastrophic gate ([D29](docs/DECISIONS.md)).

### Metrics and endurance

`/metrics` negotiates on `Accept` and speaks Prometheus. `scripts/soak.py`
watches memory, file descriptors and latency drift over a long run, and
distinguishes a leak from a step. Both are covered in
[RUNNING.md](docs/RUNNING.md).

### Running the calibration

The one step that spends money, and what earned `calibrated` as the shipped default. You only need
to re-run it if you change models — the fitted calibrator is committed, and the gateway refuses to
start if it was fitted on models other than the ones configured.

Any OpenAI-compatible provider works — the eval harness sends a plain
`POST {base_url}/chat/completions` with a bearer token, so pointing it somewhere else is config only.
Google's OpenAI-compatibility endpoint, for instance:

```bash
EVAL_USE_REAL_FRONTIER=true
FRONTIER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
FRONTIER_API_KEY=...
FRONTIER_MODEL=gemini-2.0-flash
EVAL_JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
EVAL_JUDGE_API_KEY=...
EVAL_JUDGE_MODEL=gemini-2.0-flash

.venv/bin/python -m eval.run_eval --real-frontier
```

**129 calls**, sequential, with backoff on rate limits — a free tier finishes it, slowly. Every
response is cached to `eval/runs/` after *each* prompt, so an interrupted run resumes rather than
restarts.

**One thing to be deliberate about.** Calibrating against a given provider fits the thresholds to
*that* model's answers — the labels mean "as good as **this** model", not "good". If the frontier you
calibrate against is not the one you serve with, the thresholds are measuring a proxy, and
[D11](docs/DECISIONS.md) is already candid that judge-vs-human agreement is itself unverified here.

## Milestones

Twenty-two milestones, each independently runnable. M0 came first deliberately: it resolves the one
assumption everything else rests on, before anything is built on top of it.

| | Milestone | State |
|---|---|---|
| **M0** | Verify logprobs; `confidence.py`; `preflight.py` | **done** |
| **M1** | Skeleton gateway, Ollama provider, cost accounting | **done** |
| **M2** | Exact cache + telemetry | **done** |
| **M3** | Pre-router | **done** |
| **M4** | Confidence gates + escalation + mock frontier | **done** |
| **M5** | Semantic cache + false-hit guards | **done** |
| **M6** | Eval harness; calibrate the 85% | **done** — one gate passed, one failed; ships `raw` |
| **M7** | Dashboard | **done** |
| **M8** | Hardening, degradation, demo | **done** |
| **M9** | Auth, request limits, CI | **done** |
| **M10** | Multi-worker, breakers, container, load test | **done** |
| **M11** | Streaming, by buffering first | **done** — reverses D8 |
| **M12** | Tenancy, budgets, durable audit trail | **done** |
| **M13** | Test the dashboard and the wiring | **done** — found a dead setting |
| **M14** | A spend ceiling that ships armed | **done** — bounds invariant 4 |
| **M15** | Prometheus metrics, soak testing | **done** |
| **M16** | Release engineering, supply chain, runbook | **done** |
| **M17** | Security hardening | **done** — found a cross-tenant leak |
| **M18** | Shadow sampling | **done** — the claim, measured continuously |
| **M19** | Mutation testing, multi-worker in CI | **done** — found 6 real gaps |
| **M20** | Property-based tests | **done** — found a Prometheus corruption bug |
| **M21** | The real-frontier calibration | **run** — partial; answer unchanged, better founded |

```
$ .venv/bin/python -m pytest -q -m "not integration"
827 passed, 10 deselected in 9.5s

$ .venv/bin/python scripts/preflight.py
  ✓ python 3.14.6   ✓ venv   ✓ disk 34.1 GB free   ! swap 87%
  ✓ ollama v0.30.10  ✓ model:small  ✓ model:large  ✓ model:embed
  ✓ logprobs 1 token(s), top_logprobs=5, first='Paris' @ -0.0282
  ✓ embeddings nomic-embed-text, 768 dims
  ✓ redis reachable, no modules (brute-force cosine, as designed)

$ .venv/bin/python scripts/check_docs.py
  all 208 checks passed
```

The 10 integration tests need a live Ollama — one also needs the `openai` SDK, because a
compatibility claim is worth checking against the client that actually makes it. They are deselected
in CI; the unit suite is hermetic and is re-run behind a dead proxy to keep it that way.

## Documentation

| Document | What's in it |
|---|---|
| **[docs/RESULTS.md](docs/RESULTS.md)** | What was measured — the calibration, the limits, the cost |
| **[docs/SETUP.md](docs/SETUP.md)** | Install it, and whether your machine can host it |
| **[docs/RUNNING.md](docs/RUNNING.md)** | Start it, real providers, containers, multi-worker, health, load |
| **[docs/USAGE.md](docs/USAGE.md)** | Call it — auth, tenants, budgets, rate limits, streaming, audit |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | The development loop, and what CI checks |
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | How it's put together — lifecycle, components, state, failure model, seams |
| **[docs/PLAN.md](docs/PLAN.md)** | Milestones M0–M25, config surface, Redis schema, verification commands |
| **[docs/API.md](docs/API.md)** | The HTTP contract — endpoints, the `x_hlr` block, error shapes, cost math |
| **[docs/DECISIONS.md](docs/DECISIONS.md)** | Why it's built this way, what was rejected, what would change each call |
| **[docs/RUNBOOK.md](docs/RUNBOOK.md)** | What to do when something is wrong — symptom, cause, action |
| **[CHANGELOG.md](CHANGELOG.md)** | What changed in each release, and why |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | Verified API shapes, confidence math, measured constraints, sources |
| **[CLAUDE.md](CLAUDE.md)** | Working notes for AI agents in this repo |

## License

[MIT](LICENSE).
