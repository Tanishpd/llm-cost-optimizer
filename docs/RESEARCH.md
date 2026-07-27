# Verified Research Findings

Everything here was verified against primary sources (Ollama Go source, Redis source/docs, PyPI
metadata) or measured directly on this machine on **2026-07-27**. Do not re-research these; if you
contradict something here, re-verify first and update this file with new evidence.

---

## 1. Machine constraints (measured)

| Fact | Value | How measured |
|---|---|---|
| Platform | macOS Darwin 25.5.0, Apple Silicon `arm64` | `uname -m` |
| **RAM** | **8 GB unified**, 37% free | `sysctl hw.memsize`, `memory_pressure` |
| **Swap** | **5,532 MB used of 6,144 MB (90%)** | `sysctl vm.swapusage` |
| Free disk (initial) | 3.2 GB (99% full) — being expanded to ~20 GB | `df -h /System/Volumes/Data` |
| Ollama | v0.30.10 installed, **server not running**, **zero models pulled** | `ollama --version`, empty manifests dir |
| Redis | **7.2.7** running, `MODULE LIST` **empty**, `FT._LIST` unknown command | `redis-cli` |
| Redis atomics | `atomicvar_api:c11-builtin` | `redis-cli INFO server` |
| Homebrew Redis stable | **8.8.1** available (upgrade path exists) | `brew info redis` |
| Python | 3.14.6 default; 3.13, 3.12, 3.10 also present | `ls /opt/homebrew/bin/python3.*` |
| Docker | CLI present, **daemon not running** | `docker ps` fails |

### RAM is the binding constraint, not disk

Disk was the initial constraint and is being relieved (see §7). **RAM cannot be relieved**, and it is
what actually caps model size. Two compounding facts:

1. **8 GB unified memory**, shared with the GPU. macOS idle takes ~3 GB; a browser takes 1–2 GB.
   Realistically **~3.5–4.5 GB is available for models.**
2. **Swap is 90% full** (5.5 GB of 6.1 GB) *before* any model loads. macOS grows swap on disk, so
   the near-full disk was preventing that growth — which is the other reason freeing disk matters.

Practical ceiling: **a 3B–4B Q4 model**. A 7B needs ~4.8 GB resident and will thrash against an
already-saturated swap file. `mistral:7b` from the original spec is excluded for this reason — not
disk. See the RAM column in §5.

---

## 2. Ollama exposes logprobs — CONFIRMED IN SOURCE, pending empirical check (M0)

This was the highest-risk assumption in the project. The Ollama source is unambiguous that logprobs
are supported. **It has not yet been observed on this machine** — no model has been pulled and the
server has never been started. M0 exists to close that gap with one curl, and carries a fallback
(`CONFIDENCE_STRATEGY=self_consistency`) if the check fails.

**Shipped in Ollama v0.12.11.** Release notes: *"Ollama's API and OpenAI-compatible API now support
log probabilities."* The installed client is **0.30.10**, far past that.

### Request fields (`api/types.go`, applies to both `GenerateRequest` and `ChatRequest`)

```go
// Logprobs specifies whether to return log probabilities of the output tokens.
Logprobs bool `json:"logprobs,omitempty"`

// TopLogprobs is the number of most likely tokens to return at each token position,
// each with an associated log probability. Only applies when Logprobs is true.
// Valid values are 0-20. Default is 0 (only return the selected token's logprob).
TopLogprobs int `json:"top_logprobs,omitempty"`
```

Validation is enforced server-side in `server/routes.go`:

```go
if req.TopLogprobs < 0 || req.TopLogprobs > 20 {
    c.AbortWithStatusJSON(http.StatusBadRequest,
        gin.H{"error": "top_logprobs must be between 0 and 20"})
}
```

### Response shape (`api/types.go`)

```go
type TokenLogprob struct {
    Token   string `json:"token"`
    Logprob float64 `json:"logprob"`   // natural log
    Bytes   []int   `json:"bytes,omitempty"`
}

type Logprob struct {
    TokenLogprob                              // embedded: token, logprob, bytes
    TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}
```

`GenerateResponse` and `ChatResponse` both carry `Logprobs []Logprob \`json:"logprobs,omitempty"\``.

Concretely, the JSON is:

```json
"logprobs": [
  { "token": "Paris",
    "logprob": -0.0234,
    "bytes": [80,97,114,105,115],
    "top_logprobs": [
      {"token":"Paris","logprob":-0.0234,"bytes":[...]},
      {"token":"The",  "logprob":-4.11,  "bytes":[...]}
    ]}
]
```

### Streaming behaviour

`server/routes.go` accumulates per-chunk logprobs into the final non-streaming response:

```go
// Accumulate logprobs from all chunks for non-streaming response
if len(t.Logprobs) > 0 { allLogprobs = append(allLogprobs, t.Logprobs...) }
...
r.Logprobs = allLogprobs
```

It also explicitly emits chunks carrying logprobs even when content is buffered:

```go
// Emit chunks that carry logprobs even if the parser is still buffering
// visible content, otherwise generate logprobs disappear for models with ...
if res.Response != "" || res.Thinking != "" || res.Done || len(res.ToolCalls) > 0 || len(res.Logprobs) > 0 {
```

**Conclusion: use `"stream": false`.** Non-streaming returns the complete accumulated `logprobs`
array, which is exactly what confidence gating needs (the gate cannot decide until the whole
response exists). Streaming works too, but is pointless here — see [CLAUDE.md](../CLAUDE.md).

### OpenAI-compatibility layer

`openai/openai.go` defines `ChoiceLogprobs { content []api.Logprob }`, and **both** `Choice`
(non-streaming) and `ChunkChoice` (streaming) carry an optional `logprobs` field. So
`/v1/chat/completions` supports it too.

> Caveat: GitHub issue #16117 ("Add logprobs support to OpenAI-compatible endpoint") is closed as
> *not planned*, which reads as "unsupported" if you only skim issues. The source says otherwise —
> it was implemented via the main API path. **Trust the source, not the issue tracker.** Milestone 0
> verifies empirically on this machine anyway.

---

## 3. Confidence scoring math

Ollama gives natural-log probabilities. Let `logprob_i` be the sampled token's logprob, `N` tokens.

### Why the obvious metric is wrong

Sequence probability `∏ p_i = exp(Σ logprob_i)` **decays exponentially with length**. At a healthy
per-token `p = 0.9`, a 100-token answer scores `0.9^100 ≈ 2.7e-5`. Any fixed threshold — certainly
0.85 — rejects every answer longer than a couple of tokens. It measures length, not certainty.

### Metrics actually worth computing

| Metric | Formula | Reads as |
|---|---|---|
| `geo_mean` (primary) | `exp( (1/N) Σ logprob_i )` | Geometric-mean token probability = **1/perplexity**. Length-invariant, range (0,1]. |
| `min_prob` | `min_i exp(logprob_i)` | Weakest link — one wildly uncertain token. |
| `p10_prob` | 10th percentile of `exp(logprob_i)` | Robust weakest-link; less noisy than `min`. |
| `mean_margin` | `mean_i (logprob_top1,i − logprob_top2,i)` | Decisiveness in nats. Needs `top_logprobs >= 2`. |
| `mean_norm_entropy` | `mean_i H_i / log k`, `H_i` over renormalized top-k | Spread of the distribution. Lower = more confident. |
| `frac_low` | `\|{i : p_i < 0.5}\| / N` | How much of the answer was a coin flip. |

### The calibration problem (important)

Even `geo_mean` is **not** a probability of correctness. A small model producing a *correct* answer
typically lands around **0.5–0.75**, because natural language has legitimate lexical freedom
("Paris is the capital" vs "The capital is Paris" — both fine, both lower the geometric mean).

So a raw `geo_mean >= 0.85` threshold would escalate almost everything, and the gateway would look
broken while being arithmetically correct.

**Resolution — two modes:**

- `raw` (default before calibration): `confidence = geo_mean`, thresholds **0.55** (gate 1) and
  **0.62** (gate 2), clearly documented as *not* the same number as the spec's 85%.
- `calibrated` (after `eval/run_eval.py` runs): fit a logistic/isotonic map from the feature vector
  `[geo_mean, min_prob, p10_prob, mean_margin, mean_norm_entropy, frac_low, n_tokens]` to
  `P(this tier's answer is as good as the frontier answer)`. Then **0.85 means what the spec
  intended**: an 85% chance the cheap answer is good enough.

**Fit one calibrator per gate**, and do not let the two thresholds be compared naively.

In **raw** mode the two numbers apply to *different models*, whose `geo_mean` distributions differ —
a 0.5B scores systematically lower than a 3B on the same prompt. So `0.55 < 0.62` reflects those
distributions, **not** a policy. The ordering carries no economic meaning, and prose that claims
otherwise is wrong (an earlier draft of these docs made exactly that error, in the opposite
direction).

In **calibrated** mode both scores are comparable probabilities, so economics *do* apply — and they
run the other way from intuition:

| Gate | Escalating costs | Therefore | Target |
|---|---|---|---|
| 1 (`small`) | ~450 ms, no money | Escalate readily → **high** acceptance bar | `0.92` |
| 2 (`large`) | real money | Escalate reluctantly → lower acceptance bar | `0.85` |

Gate 1's bar is *higher* than gate 2's. When the next tier is nearly free, be quick to use it; when
it costs money, be slow. 0.85 sits on gate 2 — the gate that actually spends — which is where the
spec's number belongs.

**Calibration is a claim that needs its own evidence.** Fitting a model does not make it calibrated.
M6 must report held-out reliability (curve, Brier score, ECE per gate under k-fold CV), because ~120
prompts is a small sample for a 7-feature fit. Flip to `calibrated` **only if** the held-out
reliability curve tracks the diagonal in the 0.7–0.95 region; otherwise stay in `raw` and say so.
Claiming "measured, not asserted" without that check would be asserting a different thing.

### Hard veto (independent of score)

Escalate regardless of confidence when the local output matches hedge patterns: `I don't know`,
`I'm not sure`, `As an AI`, `cannot provide`, `insufficient information`, or is suspiciously short
(< 3 tokens) for a non-trivial prompt. Cheap, high-precision, catches failures logprobs miss —
a model can be confidently fluent while refusing.

---

## 4. Redis: no vector index today, but an upgrade path exists

**Installed Redis 7.2.7 has no modules.** No `FT.CREATE`, no `FT.SEARCH`, no RedisJSON, no vector
index. `redis-stack` needs Docker (daemon down) or a tap, and costs scarce disk.

**Redis 8 changes this, and it is closer than expected:**
- Homebrew stable is **8.8.1** — an upgrade, not a new install.
- Vector Sets are **statically compiled into `redis-server`**, not a loadable module. The Makefile
  gates on `SKIP_VEC_SETS`, which is only set when C11 atomics are unavailable. This machine reports
  `atomicvar_api:c11-builtin`, so a homebrew Redis 8.8.1 **would** ship `VADD`/`VSIM` natively.
- Syntax: `VADD key [REDUCE dim] (FP32 | VALUES num) vector element [CAS] [NOQUANT | Q8 | BIN]`
  and `VSIM key (ELE | FP32 | VALUES num) (vector|element) [WITHSCORES] [WITHATTRIBS] [COUNT num]`.
  Scores run 1 → 0, 1 being identical. Vectors are normalized at insertion.

**Decision: do NOT upgrade Redis.** Reasons: upgrading a running server mid-project risks
RDB/AOF-compat pain with an awkward downgrade path, and it buys nothing at this scale (see below).
Vector Sets are documented in [PLAN.md](PLAN.md) as an *optional* swap behind the same interface.

### Brute-force cosine is genuinely sufficient here

768-dim `float32` (`nomic-embed-text`) = 3072 bytes/vector.

| Entries | Index RAM | Cosine matmul (calculated, not benchmarked) |
|---|---|---|
| 1,000 | 3 MB | 768K MACs — microseconds |
| 10,000 | 31 MB | 7.7M MACs — expected well under 1 ms via numpy/BLAS |
| 100,000 | 307 MB | ~77M MACs — a few ms; the point where an index starts paying |

A demo cache will not approach 10k entries. Redis stays the source of truth; the process keeps a
warm numpy matrix and re-syncs on a version counter.

---

## 5. Ollama model sizes (exact, from registry manifests)

Disk sizes are **exact** — summed layer bytes from
`registry.ollama.ai/v2/library/{model}/manifests/{tag}`.

**The RAM column is an estimate**, roughly 1.15× on-disk to cover the KV cache at `num_ctx=2048`.
Nothing has been loaded and measured. `scripts/preflight.py` (M0) records actual residency, and this
table should be updated with real figures once it has. The verdicts follow from the estimate, so
treat the borderline rows (`phi4-mini`, `qwen3:4b`) as genuinely uncertain rather than settled.

| Model | Disk | Resident RAM (est.) | Verdict on this machine |
|---|---|---|---|
| `qwen2.5:0.5b` | 0.40 GB | ~0.5 GB | ✅ **Tier-2a**, the fast first hop |
| `qwen2.5:1.5b` | 0.99 GB | ~1.1 GB | ✅ middle option |
| `llama3.2:1b` | 1.32 GB | ~1.5 GB | ✅ previous choice, superseded |
| **`qwen2.5:3b`** | **1.93 GB** | **~2.2 GB** | ✅ **Tier-2b**, the capable local hop |
| `llama3.2:3b` | 2.02 GB | ~2.3 GB | ✅ equivalent alternative |
| `phi4-mini` | 2.49 GB | ~2.8 GB | ⚠️ workable, tight |
| `qwen3:4b` | 2.50 GB | ~2.8 GB | ⚠️ workable, tight |
| `mistral:7b` | 4.37 GB | ~4.8 GB | ❌ **thrashes** — spec named it; RAM excludes it |
| `qwen2.5:7b` | 4.68 GB | ~5.2 GB | ❌ thrashes |
| `llama3.1:8b` | 4.92 GB | ~5.5 GB | ❌ thrashes |
| `gpt-oss:20b` | 13.79 GB | ~14 GB | ❌ impossible |

Embedding models:

| Model | Disk | Dims | Notes |
|---|---|---|---|
| `snowflake-arctic-embed:22m` | 45.8 MB | 384 | leanest |
| `all-minilm:latest` | 46.0 MB | 384 | |
| `all-minilm:33m` | 67.3 MB | 384 | prior choice |
| **`nomic-embed-text`** | **274.3 MB** | **768** | ✅ **chosen** — better separation, worth it now |
| `embeddinggemma` | 621.9 MB | 768 | diminishing returns |

Higher-dimensional embeddings matter here specifically because the semantic cache leans on cosine
precision to avoid false hits. 768-dim `nomic-embed-text` separates near-miss pairs (Australia /
Austria) more cleanly than 384-dim MiniLM, and 274 MB is affordable once disk is freed.

### Embeddings API

`POST /api/embed` is current; `POST /api/embeddings` is explicitly **superseded** by it.

```json
// request                              // response
{"model":"all-minilm","input":"text"}   {"model":"all-minilm","embeddings":[[0.0100, -0.0017, ...]]}
```

Note `embeddings` (plural, array-of-arrays) on the new endpoint vs `embedding` (singular, flat) on
the legacy one. `input` accepts a string **or an array of strings** — batch your embeds.

---

## 6. Python packaging — 3.14 is safe, torch is unnecessary

All required packages publish **cp314 arm64/universal2 macOS wheels** (verified against the PyPI
JSON API):

| Package | cp312 | cp313 | cp314 |
|---|---|---|---|
| numpy 2.5.1 | ✅ | ✅ | ✅ `macosx_11_0_arm64` |
| scikit-learn 1.9.0 | ✅ | ✅ | ✅ `macosx_12_0_arm64` |
| pydantic-core 2.47.0 | ✅ | ✅ | ✅ |
| uvloop 0.22.1 | ✅ | ✅ | ✅ `universal2` |
| httptools 0.8.0 | ✅ | ✅ | ✅ |
| watchfiles 1.2.0 | ✅ | ✅ | ✅ |

**Use the default `python3` (3.14.6).** No need to hunt for 3.12.

Wheel sizes: numpy 11.8 MB, pydantic-core 2.0 MB, redis 0.5 MB, fastapi 0.1 MB, uvicorn 0.1 MB,
httpx 0.1 MB.

> Correction worth recording: **torch on macOS arm64 unpacks to ~471 MB**, not the ~2 GB commonly
> cited (that figure is Linux + bundled CUDA). With disk freed it would fit comfortably. It is still
> excluded, now on merit rather than budget: Ollama already serves embeddings over HTTP, so torch
> would add a second inference runtime, a second copy of model weights, and RAM contention with the
> two local models — to compute vectors we can already get. If a learned router later needs it, the
> `classifier.py` Protocol is the seam, and the decision can be revisited with data from `eval/`.

`scikit-learn` pulls `scipy` (~90 MB installed), so it lives in `requirements-eval.txt`, not the
runtime requirements. The gateway runs on numpy alone.

---

## 7. Disk budget

| Item | Size |
|---|---|
| `qwen2.5:0.5b` (tier 2a) | 400 MB |
| `qwen2.5:3b` (tier 2b) | 1930 MB |
| `nomic-embed-text` (embedder) | 274 MB |
| Runtime venv (fastapi, uvicorn, httpx, redis-py, numpy, pydantic) | ~120 MB |
| Eval extras (scikit-learn + scipy) | ~180 MB |
| Eval artifacts, logprob traces, logs | ~100 MB |
| **Project total** | **~3.0 GB** |

**Target ~20 GB free**, of which the project is only ~3 GB. The rest is machine health: with swap at
90% and models about to claim 3 GB of RAM, macOS needs room to grow its swap file or it starts
killing processes. Below ~10 GB free this machine will be unpleasant to work on regardless of what
this project does.

### Reclaimable space found (measured, nothing deleted)

| Location | Size | Notes |
|---|---|---|
| `~/Library/Containers/com.utmapp.UTM` | **24 GB** | UTM virtual-machine disk images |
| `~/Library/Containers/com.docker.docker` | **9.2 GB** | Docker VM disk — **daemon isn't even running** |
| `~/Library/Application Support/Code` | 7.0 GB | VS Code workspace storage + extensions |
| `~/Library/Android` | 4.1 GB | Android SDK/emulator images |
| `~/.cache` | 2.5 GB | `uv` 788M, `codex-runtimes` 740M, `puppeteer` 496M, `torch` 248M, `whisper` 211M |
| `~/.npm` | 1.7 GB | npm cache — `npm cache clean --force` |
| `~/Library/Caches/pip` | 627 MB | `pip cache purge` |
| `~/Library/Caches/Google` | 337 MB | |
| `~/Library/Developer/CoreSimulator` | 264 MB | iOS simulator runtimes |

Docker (9.2 GB, unused) plus the caches (~5.5 GB) clears ~15 GB without touching anything you'd
miss. UTM's 24 GB is the single biggest win if those VMs are stale.

---

## 8. Sources

- Ollama source: `api/types.go`, `server/routes.go`, `server/logprob.go`, `openai/openai.go`,
  `llm/server.go` (raw.githubusercontent.com, `main`)
- Ollama v0.12.11 release notes (logprobs announcement)
- Ollama API docs: `docs.ollama.com/api/generate`, `/api/embed`
- Redis: `VADD`/`VSIM` command docs, Redis 8.0 release notes. Vector-set syntax was read from
  `modules/vector-sets/README.md` **@ tag 8.2.3**, while the homebrew stable this would install is
  8.8.1 — re-verify against 8.8.1 before relying on the syntax, though no breaking change is expected.
- PyPI JSON API for every wheel-availability claim
- Local measurement for every machine-state claim
