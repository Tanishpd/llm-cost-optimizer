# Verified Research Findings

Everything here was verified against primary sources (Ollama Go source, Redis source/docs, PyPI
metadata) or measured directly on this machine — **2026-07-27** for the environment survey,
**2026-07-28** for the M0 runtime measurements (§2, §3.1, §3.2, §5). Do not re-research these; if you
contradict something here, re-verify first and update this file with new evidence.

---

## 1. Machine constraints (measured)

| Fact | Value | How measured |
|---|---|---|
| Platform | macOS Darwin 25.5.0, Apple Silicon `arm64` | `uname -m` |
| **RAM** | **8 GB unified**, 37% free | `sysctl hw.memsize`, `memory_pressure` |
| **Swap** | 90% initially → 71% after disk was freed → **87–91% observed under M0 load** (2026-07-28) | `sysctl vm.swapusage` |
| Free disk | 3.2 GB initial (99% full) → 7.9 GB → **~34 GB as of 2026-07-28**, target met | `df -h /System/Volumes/Data` |
| Ollama | **v0.30.10 server running**; `qwen2.5:0.5b`, `qwen2.5:3b`, `nomic-embed-text` pulled (2026-07-28) | `curl /api/version`, `ollama list` |
| Redis | **7.2.7** running, `MODULE LIST` **empty**, `FT._LIST` unknown command | `redis-cli` |
| Redis atomics | `atomicvar_api:c11-builtin` | `redis-cli INFO server` |
| Homebrew Redis stable | **8.8.1** available (upgrade path exists) | `brew info redis` |
| Python | 3.14.6 default; 3.13, 3.12, 3.10 also present | `ls /opt/homebrew/bin/python3.*` |
| Docker | CLI present, **daemon not running** | `docker ps` fails |

### RAM is the binding constraint, not disk

Disk was the initial constraint and **has now been relieved** (~34 GB free). **RAM cannot be
relieved**, and it is what actually caps model size. Two compounding facts:

1. **8 GB unified memory**, shared with the GPU. macOS idle takes ~3 GB; a browser takes 1–2 GB.
   Realistically **~3.5–4.5 GB is available for models.**
2. **Swap pressure is real and observed.** 90% full before any model loaded, 71% after disk was
   freed, and back to **87–91% during the M0 probes with only two models resident**. Freeing disk
   let swap grow; it did not remove the pressure, because the pressure is a RAM shortage.

Practical ceiling: **a 3B–4B Q4 model**. `qwen2.5:3b` measures 2.2 GB resident (§5); a 7B would need
roughly twice that and thrash. `mistral:7b` from the original spec is excluded for this reason — not
disk. The 34 GB of free disk changes nothing about it.

---

## 2. Ollama exposes logprobs — VERIFIED ON THIS MACHINE (M0, 2026-07-28)

This was the highest-risk assumption in the project. **It is now confirmed empirically**, not just
from the source. Ollama server v0.30.10, `qwen2.5:0.5b`, `POST /api/chat` with `"logprobs": true,
"top_logprobs": 5, "stream": false` — verbatim response (truncated):

```json
{
  "model": "qwen2.5:0.5b",
  "message": {"role": "assistant", "content": "Paris"},
  "done": true,
  "done_reason": "stop",
  "logprobs": [
    {
      "token": "Paris",
      "logprob": -0.027870038524270058,
      "bytes": [80, 97, 114, 105, 115],
      "top_logprobs": [
        {"token": "Paris",  "logprob": -0.027870038524270058, "bytes": [80,97,114,105,115]},
        {"token": " Paris", "logprob": -4.606834411621094,    "bytes": [32,80,97,114,105,115]},
        {"token": "L",      "logprob": -5.052446365356445,    "bytes": [76]},
        {"token": "巴黎",   "logprob": -5.918153762817383,    "bytes": [229,183,180,233,187,142]}
      ]
    }
  ],
  "prompt_eval_count": 41, "eval_count": 2,
  "total_duration": 2175459583, "load_duration": 1957589125,
  "prompt_eval_duration": 180478000, "eval_duration": 21569000
}
```

Everything predicted from the source holds: `logprobs` is a **top-level** field on the chat response
(not nested under `message`), entries carry `{token, logprob, bytes, top_logprobs[]}`, values are
natural logs, and `top_logprobs` honours the requested count. The `prompt_eval_count` / `eval_count`
fields the cost model depends on are present too.

**One wrinkle worth knowing:** `eval_count` was 2 but `logprobs` had 1 entry — the stopping token
gets counted in `eval_count` without producing a logprob entry. So **never assume
`len(logprobs) == eval_count`**; derive `n_tokens` from the array you actually got.
`app/confidence.py` does.

The fallback (`CONFIDENCE_STRATEGY=self_consistency`) is retained in the design but is not needed.

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
> it was implemented via the main API path. **Trust the source, not the issue tracker**, which M0
> then confirmed on the native endpoint (see the top of this section). The OpenAI-compat path itself
> remains source-verified but untested here; the gateway uses `/api/chat`, so nothing depends on it.

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
lands around **0.5–0.75** — now measured on this machine, see §3.1 below. The reason it sits well
below 1.0 even when right: natural language has legitimate lexical freedom
("Paris is the capital" vs "The capital is Paris" — both fine, both lower the geometric mean).

So a raw `geo_mean >= 0.85` threshold would escalate almost everything, and the gateway would look
broken while being arithmetically correct.

**Resolution — two modes:**

- `raw` (default before calibration): `confidence = geo_mean`, thresholds **0.55** (gate 1) and
  **0.62** (gate 2), clearly documented as *not* the same number as the spec's 85%.
- `calibrated` (after `eval/run_eval.py` runs): fit a logistic map from the feature vector
  `[geo_mean, min_prob, p10_prob, mean_margin, mean_norm_entropy, frac_low, n_tokens]` to
  `P(this tier's answer is as good as the frontier answer)`. (Isotonic regression is the usual
  alternative but overfits badly at N≈120; revisit only with ~1k+ labels.) Then **0.85 means what
  the spec intended**: an 85% chance the cheap answer is good enough.

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
M6 must report held-out reliability per gate — reliability curve (equal-count bins), Brier score,
ECE — computed on **pooled out-of-fold predictions** under k-fold CV, because ~120 prompts is a
small sample for a 7-feature fit (~12 points per reliability bin ⇒ ±0.28 binomial noise per bin).
The flip rule is quantitative: `calibrated` mode only if per-gate **ECE ≤ 0.10 with a bootstrap 95%
CI upper bound < 0.20**; otherwise stay in `raw` and say so. Claiming "measured, not asserted"
without that check would be asserting a different thing.

One scoping note: logprob distributions shift with sampling temperature, so a calibrator is valid at
the temperature it was fitted at (eval runs at `temperature=0`). Requests at other temperatures
should be treated as out-of-calibration — served, but flagged in `reason`.

### Hard veto (independent of score)

Escalate regardless of confidence when the local output matches hedge patterns: `I don't know`,
`I'm not sure`, `As an AI`, `cannot provide`, `insufficient information`, or is suspiciously short
(< 3 tokens) for a non-trivial prompt. Cheap, high-precision, catches failures logprobs miss —
a model can be confidently fluent while refusing.

### 3.1 Measured `geo_mean` distributions (M0, 2026-07-28)

`scripts/preflight.py --probe`, `temperature=0`, `num_predict=120`:

| Prompt class | `qwen2.5:0.5b` | `qwen2.5:3b` |
|---|---|---|
| easy factual ("capital of France") | 0.9722 | 1.0000 |
| medium reasoning ("why rainbow colour order") | 0.4748 | 0.5958 |
| hard proof ("finite integral domain is a field") | 0.5930 | 0.6699 |

Two things this settles:

1. **The 0.5–0.75 range for multi-token answers is real**, not a borrowed prior.
2. **The 3B scores systematically higher than the 0.5B on identical prompts.** This is the measured
   basis for the claim that `THRESHOLD_SMALL=0.55 < THRESHOLD_LARGE=0.62` reflects the two models'
   distributions rather than any policy about escalation economics. Previously asserted; now
   evidenced.

### 3.2 The finding that matters: `geo_mean` does not detect hallucination

Same probe, an obscure-factual prompt — *"In what year was the Treaty of Kadesh signed, and between
whom?"* (ground truth: c. 1259 BC, Ramesses II of Egypt and Ḫattušili III of the Hittites):

| Model | `geo_mean` | Verdict at its raw threshold | Answer |
|---|---|---|---|
| `qwen2.5:0.5b` | **0.5535** | **accept** (0.55) | *"signed in 1956 … between Egypt and Israel … ended the Six-Day War ongoing since 1967"* |
| `qwen2.5:3b` | 0.5600 | escalate (0.62) | *"not an actual treaty … a historical legend associated with the Bronze Age"* |

The 0.5B's answer is wrong in every particular and internally contradictory — a 1956 treaty ending a
war that began in 1967 — and the gate **accepted it**, by 0.0035. That is a **demonstrated
false-cheap** at the shipped default: a confidently wrong answer served instantly, for free, with no
signal that anything went wrong.

Worse, on the same model the hallucination (0.5535) scored *higher* than an honest struggle through
the algebra proof (0.5468, correctly escalated). So:

> **Raw `geo_mean` measures fluency, not factuality.** A small model is most fluent precisely when it
> is confabulating from a familiar-sounding template.

This is the strongest available argument for the design's own commitments, and it is now empirical
rather than theoretical:

- **calibrated mode is not optional polish.** A single fluency statistic cannot carry the 85% claim.
- **the multi-feature vector earns its place.** `frac_low` and `p10_prob` differed between those two
  0.5B answers even where `geo_mean` misordered them — exactly the signal a fitted calibrator can use
  and a single threshold cannot.
- **the hedge veto matters.** The 3B's honest "not an actual treaty" hedge is the behaviour worth
  rewarding; pattern-matching it is cheap and independent of any score.
- **M6's false-cheap rate is the headline metric**, not escalation rate or cost saved. This probe is
  one measured instance of it.

Recording this rather than quietly retuning 0.55 upward: moving the threshold to exclude this one
case would be fitting to a single example, and the underlying problem — fluency ≠ correctness —
would survive the change untouched.

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

The RAM column was estimated at roughly 1.15× on-disk. **The two chosen models have now been
measured** (`ollama ps`, 2026-07-28, `num_ctx=4096`, both 100% GPU):

| Model | Estimated | **Measured resident** | Ratio to disk |
|---|---|---|---|
| `qwen2.5:0.5b` | ~0.5 GB | **479 MB** | 1.21× |
| `qwen2.5:3b` | ~2.2 GB | **2.2 GB** | 1.16× |

The 1.15× heuristic held, so the remaining rows are credible — but they are still **estimates**, and
the borderline ones (`phi4-mini`, `qwen3:4b`) stay genuinely uncertain until loaded. Both chosen
models resident together is ~2.7 GB, which is why `OLLAMA_MAX_LOADED_MODELS=2` is a real constraint
and not a superstition: adding the 274 MB embedder makes three, and swap was observed climbing to
87–91% during the M0 probes with only two loaded.

| Model | Disk | Resident RAM (est. unless noted) | Verdict on this machine |
|---|---|---|---|
| `qwen2.5:0.5b` | 0.40 GB | **479 MB (measured)** | ✅ **Tier-2a**, the fast first hop |
| `qwen2.5:1.5b` | 0.99 GB | ~1.1 GB | ✅ middle option |
| `llama3.2:1b` | 1.32 GB | ~1.5 GB | ✅ previous choice, superseded |
| **`qwen2.5:3b`** | **1.93 GB** | **2.2 GB (measured)** | ✅ **Tier-2b**, the capable local hop |
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

**Target ~20 GB free**, of which the project is only ~3 GB. The rest is machine health: with swap
recently at 90% (~71% at last measurement) and models about to claim 3 GB of RAM, macOS needs room
to grow its swap file or it starts
killing processes. Below ~10 GB free this machine will be unpleasant to work on regardless of what
this project does.

### Reclaimable space (surveyed 2026-07-27; sizes are as-of that scan)

The largest reclaimable items on this machine were, in order: UTM VM images (~24 GB), the Docker VM
disk (~9.2 GB, daemon not running), VS Code workspace storage (~7 GB), the Android SDK (~4.1 GB),
and ~5 GB of assorted package-manager caches (`~/.cache`, `~/.npm`, pip — the caches alone are safe
to clear blind: `pip cache purge`, `npm cache clean --force`). Disk went 3.2 → 7.9 GB free after a
first pass; re-run the survey rather than trusting these numbers if more is needed.

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
