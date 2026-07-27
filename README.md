# Autonomous Hybrid LLM Routing Engine & Cost Optimizer

An API gateway that sits in front of several models of escalating cost and decides — per request —
the cheapest one that will actually do the job.

> **Status: design complete, implementation not started.**
> This document describes the system as *specified*, not as measured. The commands below are the
> contract the build ([docs/PLAN.md](docs/PLAN.md), M0–M8) is written against, and every number in
> every example payload is invented to show the field layout. Nothing has been benchmarked. Real
> figures arrive with M6/M7.

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

| Tier | What it does | Cost | Expected latency (est.) |
|---|---|---|---|
| **Exact cache** | Hash of the normalized prompt + model + sampling params | $0 | ~1 ms |
| **Semantic cache** | Embedding similarity — catches rewordings of a question you've already asked | ~$0 | ~20 ms |
| **Small SLM** | `qwen2.5:0.5b`, with a confidence score from its logprobs | $0 | ~120 ms |
| **Large SLM** | `qwen2.5:3b`, same treatment — reached only when the 0.5B wasn't sure | $0 | ~450 ms |
| **Frontier** | Escalation over real HTTP to an OpenAI-shaped endpoint | metered | ~2 s |

Latency figures are order-of-magnitude estimates for this hardware, **not measurements** — nothing has
been benchmarked. The frontier figure is the mock's configured `MOCK_LATENCY_MS`, not a real
provider's. M7 replaces all of these with recorded percentiles.

Between the caches and the models, a **pre-router** looks at the prompt — length, code fences,
reasoning markers, arithmetic, attached context — and picks where to *enter* the cascade:

- `local_small` — clearly simple, start at the 0.5B
- `local_large` — start at the 3B, skip a hop that would obviously fail
- `frontier` — clearly hard, skip both local models rather than waste them

From there the **confidence gates** take over. The router's guess is allowed to be wrong: a bad entry
decision costs latency, not quality, because the gates still catch a weak answer and escalate it.

**One compatibility gap, on purpose:** `stream: true` returns 400. A gate can't decide whether to
escalate until the full response and its logprobs exist, so streaming a local answer would mean
streaming tokens that may then be discarded and replaced. Buffer-then-stream resolves it; it's
deferred, not overlooked ([D8](docs/DECISIONS.md)).

## Reading the numbers

Every response carries an `x_hlr` block. *(Illustrative — invented values showing the shape.)*

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
it does flatter the cache tiers. Both choices are stated here rather than buried
([D13](docs/DECISIONS.md)).

`GET /metrics` gives aggregates; `GET /dashboard` will render them live — tier breakdown, cache hit
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
perfectly correct answer typically scores 0.5–0.75, simply because language offers many valid ways to
phrase the same thing.

So the gateway ships in two modes:

- **`raw`** (default) — confidence is the geometric mean, thresholds 0.55 and 0.62. Works
  immediately, and the number is *not* comparable to 85%. Nor are the two thresholds comparable to
  *each other*: they apply to different models with different score distributions.
- **`calibrated`** (after `eval/run_eval.py`) — a fitted model maps the raw signals to
  `P(this tier's answer is as good as the frontier's)`. **Now the number means what you'd expect.**
  Gate 2 accepts at **0.85** — the spec's number, on the gate that actually spends money. Gate 1's
  bar is deliberately *higher* (0.92): escalating from it costs ~450 ms and no dollars, so when the
  next tier is nearly free you should use it readily.

Whether the calibration is trustworthy is itself something M6 has to demonstrate — held-out
reliability curves, not just a fit. If it doesn't hold up, the honest outcome is staying in `raw`
mode and saying so. Full derivation in [docs/RESEARCH.md](docs/RESEARCH.md) §3.

## Constraints this was built under

This machine has **8 GB of RAM**, which is a more interesting constraint than it sounds:

- **RAM caps model size, and you can't buy your way out of it with disk.** A 7B model needs an
  estimated ~4.8 GB resident and would thrash. `mistral:7b` from the original spec is out for this
  reason. The practical ceiling is 3B–4B — which is why the design spends its RAM on a *two-model
  cascade* rather than one bigger model: two gates let the gateway decide how much compute a prompt
  deserves, not just "local or not". Whether the 0.5B tier earns its place is an open question M6
  answers; if gate 1 rarely accepts, the honest move is collapsing to a single local tier
  ([D1](docs/DECISIONS.md)).
- **No PyTorch, no `transformers`, no `sentence-transformers`.** Ollama already serves embeddings
  over HTTP; a second inference runtime would compete for the same scarce memory. The Python side is
  numpy-only at runtime.
- **Redis 7.2.7 with no modules** — no vector index available. Semantic search is brute-force cosine
  in numpy: an estimated sub-millisecond up to ~10k cached entries, a few milliseconds at 100k, which
  is where an index would start to pay. A demo cache won't approach either.
- **No Docker.** Everything runs as local processes.

Total footprint: ~2.6 GB of models plus ~300 MB of Python dependencies.

## Getting started

**Prerequisites:** Ollama, Redis, Python 3.14+.

```bash
# 1. Services
ollama serve &
brew services start redis

# 2. Models (~2.6 GB on disk, ~2.7 GB resident)
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

*Intended behaviour once M0–M5 are built — these are the acceptance criteria, not recorded output.*

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

## Documentation

| Document | What's in it |
|---|---|
| **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** | How it's put together — lifecycle, components, state, failure model, seams |
| **[docs/PLAN.md](docs/PLAN.md)** | Milestones M0–M8, config surface, Redis schema, verification commands |
| **[docs/API.md](docs/API.md)** | The HTTP contract — endpoints, the `x_hlr` block, error shapes, cost math |
| **[docs/DECISIONS.md](docs/DECISIONS.md)** | Why it's built this way, what was rejected, what would change each call |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | Verified API shapes, confidence math, measured constraints, sources |
| **[CLAUDE.md](CLAUDE.md)** | Working notes for AI agents in this repo |
