# Runbook

What to do when something is wrong, written for whoever is on call rather than
whoever wrote it.

`/healthz` reports problems and says nothing about what to do about them. This
is that half. Every entry names the signal, what it actually means, and what to
do — in that order, because the middle step is the one that gets skipped.

**The general shape of this system's failures:** it is built to degrade rather
than fail (invariant 4), so *the usual symptom of a serious problem is a higher
bill, not an outage*. Cost is the alarm, not the aftermath.

---

## First 60 seconds

```bash
curl -s localhost:8000/healthz | jq '{status, degraded: .config, deps: .dependencies}'
curl -s -H 'Accept: text/plain' localhost:8000/metrics | grep -E '^hlr_(up|dependency_up|circuit|budgets_enforced|audit_dropped)'
tail -50 audit/audit-*.jsonl | jq -r '[.ts, .tier, .tenant, .cost_micro_usd, (.degraded|join(","))] | @tsv'
```

Those three answer: is it serving, what is broken, and what has it been doing.

---

## Symptom → cause → action

### The bill is climbing

**Signal.** `hlr_tier_attempted_total{tier="frontier"}` rate jumps; or
`hlr_tenant_spend_usd` rising far faster than usual.

**Almost always the same cause: the local tiers are gone.** Invariant 4 sends
everything to the frontier when Ollama is unreachable, which keeps the gateway
answering *and* is the single most expensive state it has — roughly $138/day per
client at `RATE_LIMIT_PER_MINUTE=60`.

```bash
curl -s localhost:8000/healthz | jq '.dependencies.ollama'
ollama ps          # is anything actually loaded?
ollama serve &     # if not
```

`MAX_DAILY_SPEND_USD` bounds the damage while you fix it (default `$10`). If it
has already bound, you will see `spend_ceiling_reached` and 503s — that is the
protection working, not a second fault.

**Do not raise the ceiling to make the 503s stop.** The 503s are the symptom;
the missing local tiers are the fault.

### `hlr_budgets_enforced 0`

**Meaning.** Caps are configured and are **not being applied**. Frontier spend
is uncapped right now.

**Cause.** Redis is unreachable. Budgets fail *open* deliberately — failing
closed would turn a cache blip into a total frontier outage
([D25](DECISIONS.md)) — but that means this line is the only thing standing
between you and an unbounded bill.

```bash
redis-cli ping
curl -s localhost:8000/healthz | jq '.dependencies.redis'
```

**While Redis is down**, if the traffic is not trusted, stop the spend rather
than hope: point `FRONTIER_BASE_URL` at the mock, or set
`RATE_LIMIT_PER_MINUTE` low. Both are restarts.

### `hlr_audit_dropped_total` is increasing

**Meaning.** The audit writer cannot keep up and records are being dropped. The
trail has holes, and anything you conclude from its *absence* is now unsound.

**Cause.** Disk slower than request rate, or `AUDIT_QUEUE_SIZE` too small for a
burst.

```bash
curl -s localhost:8000/healthz | jq '.config.audit'
df -h .            # is the disk full or slow?
```

Raise `AUDIT_QUEUE_SIZE`, or move `AUDIT_PATH` to a faster disk. Dropping is
deliberate — the alternative is blocking the request path
([D26](DECISIONS.md)) — so this is a capacity problem, not a bug.

### `hlr_circuit_state{tier=...} 2`

**Meaning.** That tier failed `BREAKER_FAILURE_THRESHOLD` times in a row and is
being skipped. Requests are *not* failing; they are escalating past it.

**If `local_small` or `local_large`:** cost is now rising. See *the bill is
climbing*.
**If `frontier`:** escalation is unavailable and below-threshold local answers
are being returned — quality is degraded, cost is not.

The breaker admits one probe per `BREAKER_COOLDOWN_S`, so it recovers by itself
once the tier does. Nothing to do but fix the tier.

### 429s, and the client says they are within their limit

Rate limiting is **per principal** — per API key, or per client IP when the
gateway runs open. Several clients behind one NAT share a bucket.

```bash
curl -s localhost:8000/healthz | jq '.config.rate_limit_distributed'
```

`false` means Redis is unreachable and each worker is enforcing its own window,
so the *effective* limit is `RATE_LIMIT_PER_MINUTE × workers`. That is more
permissive, not less — if you are seeing unexpected 429s with this `false`, the
cause is elsewhere.

### 402 `budget_exhausted`

Working as designed. A tenant hit their daily cap **and** pinned
`gateway:frontier`, removing their own fallback. Without the pin they would have
been served locally.

Raise that tenant's `daily_budget_usd`, or tell them to drop the pin.

### 503 `spend_ceiling_reached`

The gateway's own ceiling bound **and** no local tier could serve. Two faults at
once, and the second one is the real one — see *the bill is climbing*.

### 503 `all_tiers_down`

Everything is unreachable. The only genuine outage this system has.

```bash
ollama ps && redis-cli ping && curl -s $FRONTIER_BASE_URL/models
```

### `REDIS IS REMOTE AND UNAUTHENTICATED` at startup

**Meaning.** `REDIS_URL` points somewhere other than loopback and carries no
password. Anyone who can reach that port has a read/write handle on every
cached answer, every budget counter and every rate-limit window — including the
ability to zero a counter and uncap your spend.

```bash
REDIS_URL=redis://:<password>@host:6379/0   # password
REDIS_URL=rediss://host:6379/0              # TLS
```

Both are restarts. Loopback and the compose service name are exempt from the
warning, because neither is reachable from outside the host.

### TLS

**This process should not terminate TLS.** Put it behind a proxy that does, and
let the proxy own `Strict-Transport-Security` — the gateway deliberately does
not set it, because asserting HSTS from a plaintext origin either does nothing
or is stripped, and either way it misleads a reader into thinking transport
security is handled somewhere.

The gateway sets `X-Content-Type-Options`, `X-Frame-Options` and
`Referrer-Policy` itself, and `Cache-Control: no-store` on `/v1/*` so prompts
do not sit in an intermediary cache.

### Answers are wrong, but nothing is unhealthy

The most likely cause is the semantic cache serving a near-miss. A wrong cache
hit is a confidently wrong answer served for free, with no signal — which is
why the guards are deliberately over-conservative ([D15](DECISIONS.md)).

```bash
# Was it a cache hit, and at what similarity?
grep '"trace_id":"<id>"' audit/audit-*.jsonl | jq '{tier, cache_hit, reason}'
```

To rule the cache out entirely while you investigate:

```bash
SEMANTIC_ENABLED=false        # restart; exact cache still works
```

Then send the request with `"model": "gateway:nocache"` to bypass both caches
for one call.

### Latency is up, cost is not

Usually a cold model load — `hlr_model_cold_loads_total` increasing. Ollama
unloads after `OLLAMA_KEEP_ALIVE`, and the next request pays ~1.5 s for the
reload. Raise `OLLAMA_KEEP_ALIVE`, or accept it on low-traffic instances.

If `hlr_latency_ms{stage="cache_semantic"}` specifically is up, the index has
grown; brute-force cosine is linear in entries and starts to matter past ~100k
([D7](DECISIONS.md)).

---

## Rolling back

Releases publish a digest-addressed image. **Pin the digest, not the tag** —
`latest` moves and a tag can be re-pushed; a digest cannot.

```bash
docker run ghcr.io/<owner>/llm-cost-optimizer@sha256:<digest>
```

The digest for each release is in that release's build summary.

**What a rollback does not undo.** Redis holds cached answers keyed partly by
`KEY_SCHEMA_VERSION`; if a release changed it, the old build will simply miss
and re-populate rather than read the new format. Budget counters and the
telemetry stream are version-independent. Audit files are append-only and are
never rewritten.

## Cutting a release

```bash
# 1. Bump the single source of truth, and write down what changed.
$EDITOR app/__init__.py CHANGELOG.md   # __version__ and a new section

# 2. The gate you will be judged by anyway.
.venv/bin/python -m pytest -m "not integration" && \
  .venv/bin/ruff check app eval scripts mock_frontier tests && \
  .venv/bin/python scripts/check_docs.py

# 3. Tag. The workflow refuses a tag that disagrees with __version__.
git tag -a v0.2.0 -m "…" && git push origin v0.2.0
```

The release workflow re-runs the suite, builds and pushes the image, and cuts
the GitHub release from the matching `CHANGELOG.md` section.

## Things that look like incidents and are not

| Looks like | Actually |
|---|---|
| `degraded: ["redis_unavailable"]` and it is still serving | Invariant 4. Caching is off; nothing is broken |
| RSS jumped and stopped | An arena CPython does not return to the OS. `scripts/soak.py` distinguishes this from a leak |
| `tier_attempted` > `requests_by_tier` | Escalation. A request that walked three tiers was attempted by three and served by one |
| Confidence ~0.5–0.75 on correct answers | Expected for `geo_mean` on a small model. It measures fluency, not correctness ([RESEARCH.md §3.1](RESEARCH.md)) |
| `x_hlr.confidence_mode: "raw"` | The shipped default. Gate `large` failed calibration; 0.85 has not been earned |
| A tenant getting another's cached answer | `CACHE_SCOPE=global`, the default — that sharing *is* the saving. Set `CACHE_SCOPE=tenant` if isolation matters more (D28) |
| `text withheld: CACHE_SCOPE=global` in a reason | Working as designed: the match may be another tenant's prompt, so its text is not echoed |
