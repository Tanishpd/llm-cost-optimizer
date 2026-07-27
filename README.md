# Autonomous Hybrid LLM Routing Engine & Cost Optimizer

An API gateway that sits in front of several models of escalating cost and decides — per request —
the cheapest one that will actually do the job.

Point any OpenAI-compatible client at it. It answers from cache when it can, tries a 0.5B model
first, steps up to a 3B when that isn't confident enough, and reaches for the expensive API only
when it can justify the spend. Every response tells you which path it took, why, and what it cost.

```
   your app  ──▶  gateway :8000  ──┬──▶  Redis cache        $0      ~1 ms
                                   ├──▶  qwen2.5:0.5b       $0    ~120 ms
                                   ├──▶  qwen2.5:3b         $0    ~450 ms
                                   └──▶  frontier API       $$$     ~2 s
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

That's a testable claim, so the project tests it: `eval/` measures cost against quality across a
labeled prompt set and calibrates the thresholds, rather than asserting a number and hoping.

## How it works

A request descends five tiers and stops at the first one that can serve it.

| Tier | What it does | Cost | Typical latency |
|---|---|---|---|
| **Exact cache** | Hash of the normalized prompt + model + sampling params | $0 | ~1 ms |
| **Semantic cache** | Embedding similarity — catches rewordings of a question you've already asked | ~$0 | ~20 ms |
| **Small SLM** | `qwen2.5:0.5b`, with a confidence score from its logprobs | $0 | ~120 ms |
| **Large SLM** | `qwen2.5:3b`, same treatment — reached only when the 0.5B wasn't sure | $0 | ~450 ms |
| **Frontier** | Escalation over real HTTP to an OpenAI-shaped endpoint | metered | ~2 s |

Between the caches and the models, a **pre-router** looks at the prompt — length, code fences,
reasoning markers, arithmetic, attached context — and picks where to *enter* the cascade:

- `small` — clearly simple, start at the 0.5B
- `large` — start at the 3B, skip a hop that would obviously fail
- `escalate` — clearly hard, skip both local models rather than waste them

From there the **confidence gates** take over. There are two of them, and they're deliberately not
tuned the same way: gate 1 escalates to something nearly free, so it can afford to be trigger-happy;
gate 2 escalates to real money, so it's reluctant.

## Reading the numbers

Every response carries an `x_hlr` block:

```json
{
  "x_hlr": {
    "tier": "local_large",
    "path": ["local_small", "local_large"],
    "gates": [
      {"gate": "small", "confidence": 0.41, "verdict": "escalate"},
      {"gate": "large", "confidence": 0.68, "verdict": "accept"}
    ],
    "reason": "explanatory 'why' question, 41 tokens; 0.5B uncertain on causal chain",
    "confidence_mode": "raw",
    "latency_ms": {"route": 0.4, "cache": 1.2, "local_small": 118.0, "local_large": 441.0, "total": 560.6},
    "tokens_in": 41, "tokens_out": 156,
    "cost_usd": 0.0,
    "cost_saved_usd": 0.00166
  }
}
```

- **`tier`** — who actually served the answer.
- **`path`** — every tier attempted, in order. This request paid for two model calls; the metadata
  says so rather than hiding it.
- **`gates`** — each gate's score and verdict. This is the routing decision, exposed.
- **`reason`** — plain English, why the router decided what it decided.
- **`cost_saved_usd`** — what this request *would* have cost going straight to the frontier, minus
  what it actually cost. Summed over time on the dashboard, this is the headline number.

`GET /metrics` gives aggregates; `GET /dashboard` renders them live — tier breakdown, cache hit rate,
per-gate escalation rates, latency percentiles, and cumulative spend against an all-frontier baseline.

### About that 85% threshold

The original spec said *escalate when confidence falls below 85%*. Getting that to mean something
real took some care, and it's worth understanding.

The obvious approach — multiply every token's probability together — **doesn't work**. That number
shrinks exponentially with length. A 100-token answer where the model was a healthy 90% sure of every
single token scores 0.000026. Threshold it at 85% and you escalate literally everything. It measures
length, not certainty.

The fix is the **geometric mean** of token probabilities (equivalently, inverse perplexity), which is
length-invariant. But even that isn't a probability of being *right* — a small model producing a
perfectly correct answer typically scores 0.5–0.75, simply because language offers many valid ways to
phrase the same thing.

So the gateway ships in two modes:

- **`raw`** (default) — confidence is the geometric mean, thresholds 0.55 and 0.62. Honest, works
  immediately, and the number is *not* directly comparable to 85%.
- **`calibrated`** (after you run `eval/run_eval.py`) — a fitted model maps the raw signals to
  `P(this tier's answer is as good as the frontier's)`. **Now 0.85 means what you'd expect:** an 85%
  chance the cheap answer is good enough. Each gate gets its own calibrator, because "as good as
  frontier" is a different bar for a 0.5B than for a 3B.

Full derivation in [docs/RESEARCH.md](docs/RESEARCH.md) §3.

## Constraints this was built under

This machine has **8 GB of RAM with swap already 90% full**, which is a more interesting constraint
than it sounds:

- **RAM caps model size, and you can't buy your way out of it with disk.** A 7B model needs ~4.8 GB
  resident and would thrash against a saturated swap file. `mistral:7b` from the original spec is out
  for this reason. The practical ceiling is 3B–4B — which is exactly why a *two-model cascade* beats
  one bigger model here.
- **No PyTorch, no `transformers`, no `sentence-transformers`.** Ollama already serves embeddings
  over HTTP; a second inference runtime would just compete for the same scarce memory. The Python
  side is numpy-only at runtime.
- **Redis 7.2.7 with no modules** — no vector index available. Semantic search is brute-force cosine
  in numpy, sub-millisecond well past any cache size this will see.
- **No Docker.** Everything runs as local processes.

Total footprint: ~3 GB of models and dependencies.

## Getting started

**Prerequisites:** Ollama, Redis, Python 3.14 (all already installed here).

```bash
# 0. Disk — models need ~2.6 GB, and macOS needs swap headroom on top
df -h /System/Volumes/Data          # want ~20 GB free; see docs/RESEARCH.md §7

# 1. Services
ollama serve &
brew services start redis

# 2. Models (~2.6 GB)
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 3. Python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. Verify the environment before building on it
python3 scripts/preflight.py

# 5. Run
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &
python3 scripts/demo.py
```

Then open <http://localhost:8000/dashboard>.

### Try each tier

```bash
H='content-type: application/json'
Q='{"model":"gateway","messages":[{"role":"user","content":"What is the capital of France?"}]}'

curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" | jq .x_hlr   # → local_small
curl -s localhost:8000/v1/chat/completions -H "$H" -d "$Q" | jq .x_hlr   # → cache_exact

# reworded — same question, different words
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Tell me France'\''s capital city."}]}' | jq .x_hlr
# → cache_semantic

# the 0.5B will hesitate here; the 3B picks it up
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Explain why a rainbow has the colour order it does."}]}' | jq .x_hlr
# → local_large

# neither local model is confident
curl -s localhost:8000/v1/chat/completions -H "$H" \
  -d '{"model":"gateway","messages":[{"role":"user","content":"Prove that every finite integral domain is a field."}]}' | jq .x_hlr
# → frontier
```

## The expensive tier

By default it's a local mock on `:8001` that speaks genuine OpenAI `/v1/chat/completions` JSON —
including the `usage` block — with configurable latency and per-token pricing, so the cost math is
realistic without spending anything.

Because the gateway talks to it over real HTTP, switching to a live provider is configuration only:

```bash
FRONTIER_BASE_URL=https://api.openai.com/v1
FRONTIER_API_KEY=sk-...
FRONTIER_MODEL=gpt-4o
```

No code changes. Update the price constants in `app/config.py` and verify current rates — the
defaults are placeholders.

**One exception, and it costs money:** the eval harness (M6) calls a *real* frontier API to get
reference answers and to judge quality — roughly 120 prompts × 2 calls, a few dollars, once. A mock
can't tell you whether the 3B's answer was actually good, and without that the calibrated 85% would
be meaningless. Every response is cached under `eval/runs/` so re-fitting never re-spends.

## Project status

Design and documentation complete; **implementation not started**. The build runs in nine milestones,
each independently runnable — starting with M0, which verifies the logprobs assumption on this
machine before anything depends on it.

| Document | What's in it |
|---|---|
| **[docs/PLAN.md](docs/PLAN.md)** | Milestones, config surface, Redis schema, file tree, verification commands |
| **[docs/API.md](docs/API.md)** | The HTTP contract — endpoints, the `x_hlr` block, error shapes, cost math |
| **[docs/DECISIONS.md](docs/DECISIONS.md)** | Why it's built this way, what was rejected, and what would change each call |
| **[docs/RESEARCH.md](docs/RESEARCH.md)** | Verified API shapes, confidence math, measured constraints, sources |
| **[CLAUDE.md](CLAUDE.md)** | Working notes for AI agents in this repo |
