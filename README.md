# LLM Cost Optimizer

An API gateway that sits in front of several models of escalating cost and decides — per request —
the cheapest one that will actually do the job.

> **Status: M0–M16 complete. The gateway runs, and it's hardened.**
> All five tiers, both confidence gates, the eval harness and the dashboard are built and working —
> **751 unit tests** (hermetic, no network) plus 10 integration tests against live models. Auth,
> rate
> limiting, per-tenant budgets, a durable audit trail, circuit breakers, structured
> logging and a container image are in. Run
> `python3 scripts/demo.py` to walk every tier in one go.
>
> The one honest caveat: **the gateway ships in `raw` confidence mode, not `calibrated`.** M6 fitted
> the calibrators and one of the two gates failed its validation, so 0.85 has not been earned yet.
> That result is in [RESEARCH.md §3.4](docs/RESEARCH.md), not hidden.

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
| **Exact cache** | Hash of the normalized prompt + model + sampling params | $0 | **0.2 ms** | 0.2 ms |
| **Semantic cache** | Embedding similarity — catches rewordings of a question you've already asked | ~$0 | **40 ms** | 60 ms |
| **Small SLM** | `qwen2.5:0.5b`, with a confidence score from its logprobs | $0 | **506 ms** | 1.2 s |
| **Large SLM** | `qwen2.5:3b`, same treatment — reached only when the 0.5B wasn't sure | $0 | **862 ms** | 2.2 s |
| **Frontier** | Escalation over real HTTP to an OpenAI-shaped endpoint | metered | **1.9 s** | 1.9 s |

**Measured** on this machine via `/metrics`, not estimated — small sample (n = 1–7 per tier), so read
them as orders of magnitude rather than benchmarks. The frontier figure is the mock's configured
`MOCK_LATENCY_MS`; a real provider will differ.

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

**M6 has now run, and the honest outcome is `raw` mode.** Gate 1 passed validation (ECE 0.062,
Brier better than base rate). **Gate 2 failed** — not on calibration, but on *usefulness*: its ECE
was actually the better of the two (0.047), because a model that predicts the base rate everywhere
is trivially well-calibrated and carries no information. Brier-against-baseline is what separates
"calibrated" from "calibrated and useful", and the 3B's confidence didn't clear it on this sample.
Shipping one measured gate beside one guessed gate would put both behind the same 0.85 label, so
the gateway stays in `raw`. Full numbers in [RESEARCH.md §3.4](docs/RESEARCH.md). Full derivation in [docs/RESEARCH.md](docs/RESEARCH.md) §3.

The calibrated path itself is wired and tested even though it isn't the default — flipping
`CONFIDENCE_MODE` loads `CALIBRATOR_PATH` at startup and **refuses to start** if the fit is missing,
rather than falling back to raw thresholds while still reporting `calibrated`.

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

## Constraints this was built under

This machine has **8 GB of RAM**, which is a more interesting constraint than it sounds:

- **RAM caps model size, and you can't buy your way out of it with disk.** A 7B model needs an
  estimated ~4.8 GB resident and would thrash. `mistral:7b` from the original spec is out for this
  reason. The practical ceiling is 3B–4B — which is why the design spends its RAM on a *two-model
  cascade* rather than one bigger model: two gates let the gateway decide how much compute a prompt
  deserves, not just "local or not". The 0.5B earns its place: at the shipped threshold gate 1
  escalates only 15% of the eval set, so the cheapest tier serves most traffic. D1's reversal
  condition was gate 1 accepting <10% — it accepts 85% ([D1](docs/DECISIONS.md)).
- **No PyTorch, no `transformers`, no `sentence-transformers`.** Ollama already serves embeddings
  over HTTP; a second inference runtime would compete for the same scarce memory. The Python side is
  numpy-only at runtime.
- **Redis 7.2.7 with no modules** — no vector index available. Semantic search is brute-force cosine
  in numpy: an estimated sub-millisecond up to ~10k cached entries, a few milliseconds at 100k, which
  is where an index would start to pay. A demo cache won't approach either.
- **No Docker daemon on the dev machine.** So everything runs as local processes here, and nothing in
  the codebase may *require* a container. A `Dockerfile` and `docker-compose.yml` ship anyway for
  deployment, built and smoke-tested in CI rather than locally — see [Containers](#containers).

Total footprint: ~2.6 GB of models plus ~300 MB of Python dependencies.

## Getting started

**Prerequisites:** Ollama, Redis, Python 3.14+.

```bash
# 1. Services
ollama serve &
brew services start redis

# 2. Models (~2.6 GB on disk; the two chat models are ~2.7 GB resident when
#    loaded, the embedder ~0.3 GB more — OLLAMA_MAX_LOADED_MODELS=2 keeps at
#    most two in RAM)
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 3. Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 4. Verify the environment before building on it — checks models, disk, RAM, and logprobs
python3 scripts/preflight.py

# 5. Run
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &
python3 scripts/demo.py
```

Then open <http://localhost:8000/dashboard>.

### Try each tier

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

## The expensive tier

By default it's a local mock on `:8001` speaking genuine OpenAI `/v1/chat/completions` JSON —
including the `usage` block — with configurable latency and per-token pricing, so the cost math is
realistic without spending anything.

The envelope is faithful; the *content* is canned. The mock proves the escalation path and the
accounting, not answer quality — that's what the eval harness needs a real API for.

Because the gateway talks to it over real HTTP, switching to a live provider is configuration only:

```bash
FRONTIER_BASE_URL=https://api.openai.com/v1
FRONTIER_API_KEY=sk-...
FRONTIER_MODEL=gpt-4o
FRONTIER_PRICE_IN=2.50      # verify current rates — defaults are placeholders
FRONTIER_PRICE_OUT=10.00
```

No code changes.

**One step does cost money:** the eval harness (M6) calls a real API for reference answers and a
judge model for quality scoring — roughly 3 calls per prompt across ~120 prompts. Compute the cost
from your actual rates before running; the price constants are placeholders. Every response is cached
under `eval/runs/` so re-fitting calibrators never re-spends.

## Running it for real

### Authentication

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
([D17](docs/DECISIONS.md)).

### Limits

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
and says so in `/healthz` ([D18](docs/DECISIONS.md)).

### Tenants and budgets

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
never to bind, which defeats the point ([D23](docs/DECISIONS.md)).

Two things stated rather than glossed. **The cap is approximate by exactly one request** — cost isn't
knowable until the provider replies, so the budget is read before the call and recorded after; the
overshoot is bounded by `MAX_OUTPUT_TOKENS × FRONTIER_PRICE_OUT`, ~$0.04 at the defaults, and startup
logs the number. **If Redis is down, budgets aren't enforced** — failing closed would turn a cache
blip into a total frontier outage, so it fails open and says so in four places
([D24](docs/DECISIONS.md), [D25](docs/DECISIONS.md)).

**The gateway also caps itself.** `MAX_DAILY_SPEND_USD` (default **$10**) bounds spend across every
tenant, and unlike the per-tenant budgets it ships *armed*. It exists because invariant 4 — *"Ollama
down → escalate everything"* — is correct for availability and costs about **$138/day per client** at
the shipped rate limit. A business policy may default to off; a safety limit may not
([D27](docs/DECISIONS.md)).

Measured, with Ollama pointed at a dead port and the ceiling set to $0.0006:

```
req 1-3   200  tier=frontier          (invariant 4, working as designed)
req 4-5   503  spend_ceiling_reached  (bounded)
spent     $0.000624 against a $0.0006 ceiling — one request's overshoot
```

### Audit trail

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
([D26](docs/DECISIONS.md)).

### Health

| | Answers | Use for |
|---|---|---|
| `GET /livez` | is the process alive | container healthcheck, restart policy |
| `GET /readyz` | can it serve — Ollama and at least one model reachable | load-balancer admission |
| `GET /healthz` | full dependency detail, degraded modes, breaker states | humans and debugging |

`/livez` deliberately does **not** test Redis or Ollama: a liveness probe that fails during a Redis
blip converts a degradation into a restart loop.

### Resilience

Each tier sits behind a circuit breaker. After `BREAKER_FAILURE_THRESHOLD` consecutive failures it opens and
requests skip straight past — measured against a frontier that accepts connections and never replies,
at a 3 s timeout: **3193 ms → 42 ms**. One probe is admitted per cooldown to detect recovery.

Retries cover timeouts and connection errors only, with full jitter, and the frontier is **not**
retried on timeout — those tokens may already be billed ([D20](docs/DECISIONS.md)).

Set `LOG_FORMAT=json` for structured logs; every line carries the request's trace id.

### Containers

```bash
ollama serve &                 # on the HOST — not a compose service
docker compose up --build
```

Ollama stays outside the compose network on purpose: it needs the host's GPU/Metal, and bundling it
would silently give you CPU-only inference ([D21](docs/DECISIONS.md)). The gateway image is
multi-stage and runs as a non-root uid.

### Load

```bash
python3 scripts/loadtest.py --concurrency 48 --requests 500
```

Not a benchmark — it asserts things a smoke test can't: that nothing *fails* under concurrency, that
the cost identities still reconcile after hundreds of requests, and that single-flight actually
collapses duplicates.

```
500 req @ concurrency 48   500/500 ok   75 req/s   18 model calls for 8 distinct prompts
p50 28ms   p95 2069ms   p99 4853ms      worker RSS 10 MB
```

Its first run found a real bug: per-request costs were rounded to whole micro-USD *before* summing,
so a documented `/metrics` identity was off by 75 micro-USD over 200 requests. Aggregates now derive
from exactly-summed integer token counters.

## How it's tested

```bash
.venv/bin/python -m pytest -m "not integration"   # 751, hermetic — needs nothing running
.venv/bin/ruff check app eval scripts mock_frontier tests
.venv/bin/python scripts/check_docs.py            # 133 cross-document checks
```

Six layers, split by what each can and cannot see. The split isn't aspirational —
it came from counting where defects were actually found, and noticing the unit
suite had caught **one of fourteen**.

| Layer | Catches | Blind to |
|---|---|---|
| **Unit** (751) | logic inside a module | whether modules are connected |
| **Wiring** | components not plugged in; settings wired to nothing | anything the real lifespan can't reach |
| **Rendering** | dashboard JS that doesn't parse, run, or match the DOM | visual layout |
| **Doc checks** (157) | docs contradicting the code | whether the code is right |
| **Integration** (10) | real models, real Redis, the real OpenAI SDK | sustained load |
| **Load** | cost identities under concurrency | correctness of one request |
| **Soak** | memory/fd growth, latency drift over time | anything faster than a window |

Two things worth knowing:

**Coverage is measured and is deliberately not a gate.** It sat at ~95% while
`CONFIDENCE_MODE=calibrated` was wired to nothing and `DEFAULT_MAX_TOKENS` was
read by nothing. Every line ran — they just ran disconnected. `test_wiring.py`
exists because coverage can't express *"and then it is used"*, and it found the
dead setting on its first run.

**The unit suite is hermetic and CI keeps it that way**, re-running it behind a
dead proxy so a test that starts reaching the network fails there rather than
becoming flaky later.

**Still not covered**, stated rather than implied: no property-based or mutation
testing — nothing currently grades whether an assertion would fail if the code
were wrong. Concurrency is tested in a few places, but M10's multi-worker
verification was done by hand and isn't in CI.

### Metrics for a scraper

`/metrics` negotiates: `Accept: text/plain` (what every Prometheus scraper sends) gets exposition
format, anything else — including curl's bare `*/*` — keeps the JSON the dashboard consumes.
`/metrics/prometheus` always serves exposition, for the human who would rather not think about
Accept headers.

```
hlr_cost_saved_usd_total 1.07
hlr_requests_by_tier_total{tier="local_small"} 30
hlr_circuit_state{tier="frontier"} 2
hlr_budgets_enforced 0
```

The last one is the alertable one: caps are configured and **not being applied**, so spend is
uncapped right now. Full series list and what to alert on: [API.md](docs/API.md).

### Endurance

```bash
python3 scripts/soak.py --minutes 30 --rps 2
```

`loadtest.py` answers *"does it survive 500 requests at concurrency 48"* in seven seconds — every bug
it can find is a concurrency bug. This answers *"is it still the same process an hour later"*, which
is where leaks, fd exhaustion and latency drift live. An 8-minute run:

```
381 requests, 12 windows
  ok    no request failed
  ok    RSS stable — 32.0 -> 49.2 MB (peak 65.7 MB)
  ok    file descriptors stable (69 -> 68)
  ok    p95 steady (15ms -> 15ms)
  ok    served counts still reconcile
```

It distinguishes **a leak** (still climbing at the end) from **a step** (jumps once, then plateaus —
an arena CPython never returns to the OS). The first is a failure; the second is a warning that your
working set is the larger number. And it says plainly what it cannot tell you: 30 minutes is not 30
days.

## Milestones

Seventeen milestones, each independently runnable. M0 came first deliberately: it resolves the one
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

```
$ .venv/bin/python -m pytest -q -m "not integration"
751 passed, 10 deselected in 10.4s

$ .venv/bin/python scripts/preflight.py
  ✓ python 3.14.6   ✓ venv   ✓ disk 34.1 GB free   ! swap 87%
  ✓ ollama v0.30.10  ✓ model:small  ✓ model:large  ✓ model:embed
  ✓ logprobs 1 token(s), top_logprobs=5, first='Paris' @ -0.0282
  ✓ embeddings nomic-embed-text, 768 dims
  ✓ redis reachable, no modules (brute-force cosine, as designed)

$ .venv/bin/python scripts/check_docs.py
  all 157 checks passed
```

The 10 integration tests need a live Ollama — one also needs the `openai` SDK, because a
compatibility claim is worth checking against the client that actually makes it. They are deselected
in CI; the unit suite is hermetic and is re-run behind a dead proxy to keep it that way.

## Documentation

| Document | What's in it |
|---|---|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | How it's put together — lifecycle, components, state, failure model, seams |
| **[docs/PLAN.md](docs/PLAN.md)** | Milestones M0–M13, config surface, Redis schema, verification commands |
| **[docs/API.md](docs/API.md)** | The HTTP contract — endpoints, the `x_hlr` block, error shapes, cost math |
| **[docs/DECISIONS.md](docs/DECISIONS.md)** | Why it's built this way, what was rejected, what would change each call |
| **[docs/RUNBOOK.md](docs/RUNBOOK.md)** | What to do when something is wrong — symptom, cause, action |
| **[CHANGELOG.md](CHANGELOG.md)** | What changed in each release, and why |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | Verified API shapes, confidence math, measured constraints, sources |
| **[CLAUDE.md](CLAUDE.md)** | Working notes for AI agents in this repo |

## License

[MIT](LICENSE).
