# Running it

Starting it, pointing it at a real provider, and the operational surface. Install
is [SETUP.md](SETUP.md); the API is [USAGE.md](USAGE.md).

---

## Starting

```bash
ollama serve &                                  # required; not running by default
brew services start redis                       # PONG from `redis-cli ping`
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &    # the stand-in frontier
```

Both extra services are optional in the sense that the gateway starts without
them and says so — Redis down means caching off, Ollama down means everything
escalates. It will not crash-loop because a dependency is missing (invariant 4).

Configuration is environment variables, all of them listed with defaults in
[PLAN.md's config table](PLAN.md#configuration-surface) and
[`.env.example`](../.env.example).

---

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
judge model for quality scoring. Of the 110 prompts, 67 are auto-gradable and cost nothing;
only the 43 judged ones call out, at 3 calls each — **129 calls total**. Compute the cost
from your actual rates before running; the price constants are placeholders. Every response is cached
under `eval/runs/` so re-fitting calibrators never re-spends.

---

### Containers

```bash
ollama serve &                 # on the HOST — not a compose service
docker compose up --build
```

Ollama stays outside the compose network on purpose: it needs the host's GPU/Metal, and bundling it
would silently give you CPU-only inference ([D21](DECISIONS.md)). The gateway image is
multi-stage and runs as a non-root uid.

---

## Multi-worker

Supported and verified with 4 real worker processes. All shared state lives in
Redis: the rate-limit window is a sorted set behind one Lua script, the semantic
index is a numpy replica resynced on a version counter, and single-flight is a
`SET NX PX` lease.

```bash
uvicorn app.main:app --workers 4
python3 scripts/multiworker_check.py --workers 4 --rate-limit 20
```

The numpy matrix is per-process, but it is a *replica* — the version counter
drives the resync, which is what makes it correct rather than lucky. If Redis is
unreachable the rate limiter falls back to per-process state and logs that it
has: the limit then multiplies by worker count, which is worse than shared and
far better than none.

---

### Health

| | Answers | Use for |
|---|---|---|
| `GET /livez` | is the process alive | container healthcheck, restart policy |
| `GET /readyz` | can it serve — Ollama and at least one model reachable | load-balancer admission |
| `GET /healthz` | full dependency detail, degraded modes, breaker states | humans and debugging |

`/livez` deliberately does **not** test Redis or Ollama: a liveness probe that fails during a Redis
blip converts a degradation into a restart loop.

---

### Resilience

Each tier sits behind a circuit breaker. After `BREAKER_FAILURE_THRESHOLD` consecutive failures it opens and
requests skip straight past — measured against a frontier that accepts connections and never replies,
at a 3 s timeout: **3193 ms → 42 ms**. One probe is admitted per cooldown to detect recovery.

Retries cover timeouts and connection errors only, with full jitter, and the frontier is **not**
retried on timeout — those tokens may already be billed ([D20](DECISIONS.md)).

Set `LOG_FORMAT=json` for structured logs; every line carries the request's trace id.

---

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

---

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

---

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
uncapped right now. Full series list and what to alert on: [API.md](API.md).

