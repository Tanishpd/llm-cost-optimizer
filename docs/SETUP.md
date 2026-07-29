# Setup

Getting the gateway running on your machine, once. Day-to-day operation is
[RUNNING.md](RUNNING.md); using the API is [USAGE.md](USAGE.md).

---

## Will it run here?

This machine has **8 GB of RAM**, which is a more interesting constraint than it sounds:

- **RAM caps model size, and you can't buy your way out of it with disk.** A 7B model needs an
  estimated ~4.8 GB resident and would thrash. `mistral:7b` from the original spec is out for this
  reason. The practical ceiling is 3B–4B — which is why the design spends its RAM on a *two-model
  cascade* rather than one bigger model: two gates let the gateway decide how much compute a prompt
  deserves, not just "local or not". The 0.5B earns its place, though by less than an earlier version
  of this file claimed — measured on the 110-prompt eval set:

  | mode | gate 1 accepts | escalates |
  |---|---|---|
  | `raw` (0.55 on `geo_mean`) | 67% | 33% |
  | `calibrated` (0.92, the shipped default) | **31%** | 69% |

  So the cheapest tier serves about a third of this set under the shipped default, not "most of it".
  D1's reversal condition — gate 1 accepting under 10% — is still comfortably clear, so the two-model
  cascade stands ([D1](DECISIONS.md)). The old "accepts 85%" was true of the 67 auto-gradable prompts
  in `raw` mode and became wrong twice over: the set grew to include `reasoning` and `code`, and the
  default became `calibrated`, which demands 0.92 rather than 0.55. Both figures above are an upper
  bound on acceptance — the gate's hard vetoes (hedging, suspiciously short answers, missing
  logprobs) can only push more traffic upward.
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


---

## Install

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
pip install -r requirements.txt          # to run the gateway
# pip install -r requirements-dev.txt    # to run the tests as well
cp .env.example .env

# 4. Verify the environment before building on it — checks models, disk, RAM, and logprobs
python3 scripts/preflight.py

# 5. Run
uvicorn app.main:app --port 8000 &
uvicorn mock_frontier.main:app --port 8001 &
python3 scripts/demo.py
```

Then open <http://localhost:8000/dashboard>.


`scripts/preflight.py` is the step worth not skipping. It checks the models are
pulled, that disk and RAM have room, and — the one that matters — that your
Ollama build really returns token logprobs. The entire confidence gate rests on
that one API detail, and finding out in ten seconds beats finding out after a
day of building on it.

## Verifying the install

```bash
python3 scripts/demo.py          # walks all five tiers in one run
```

Then open <http://localhost:8000/dashboard>.

If something is wrong, [RUNBOOK.md](RUNBOOK.md) is organised by symptom.

## Developing

The runtime requirements deliberately carry no test dependencies. To run the
suite you also need:

```bash
pip install -r requirements-dev.txt
```

[CONTRIBUTING.md](../CONTRIBUTING.md) has the rest of the loop.
