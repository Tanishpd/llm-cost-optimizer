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

### 3.3 Measured: which cache guard actually does the work (M5, 2026-07-28)

`nomic-embed-text`, symmetric `search_query:` prefix on both sides, cosine after
unit-normalization:

| Pair | Cosine | Outcome at threshold 0.95 |
|---|---|---|
| *capital of France?* / *capital city of France?* | 0.9889 | ✅ hit |
| *reverse a list in Python?* / *how can I reverse a list…* | 0.9979 | ✅ hit |
| *Tell me France's capital.* / *Name the capital of France.* | 0.8460 | ❌ miss — a real paraphrase, lost |
| *Summarize this article.* / *Give me a summary…* | 0.9014 | ❌ miss — a real paraphrase, lost |
| **capital of Australia / Austria** | **0.7938** | below threshold; cosine alone catches it |
| **convert 50 USD / 70 USD** | **0.7984** | below threshold; cosine alone catches it |
| **Who founded Google / Microsoft** | **0.8028** | below threshold; cosine alone catches it |
| **Is this API safe / not safe** | **0.9831** | **above threshold — only the negation guard stops it** |

Two findings, both revising expectations set during design:

1. **Negation is the guard that earns its place.** The entity and number
   near-misses the docs treat as the canonical dangers score 0.79–0.80 with this
   embedder — the threshold alone rejects them. But *"is this API safe"* vs
   *"is this API **not** safe"* scores **0.9831**. The embedder cannot represent
   the negation, and without `negation_mismatch` the gateway would serve the
   opposite answer, instantly and for free. The entity and number guards stay:
   they cost microseconds, and a weaker embedder would not be as kind.

2. **0.95 is conservative enough to lose real paraphrases** (0.846, 0.901). That
   is the intended direction — a miss costs ~120 ms, a false hit costs
   correctness — but the semantic tier's hit rate will be lower than the design
   assumed. M6 should sweep this threshold against a measured false-hit rate
   rather than leaving 0.95 asserted.

Known residual risk, unguarded and documented: *convert USD to EUR* vs *convert
EUR to USD* shares numbers, entities and negation. Catching argument-order swaps
needs parsing, not lexical sets.

---

### 3.4 The calibration result (M6, 2026-07-28) — one gate passes, one fails

67 auto-gradable prompts, graded against ground truth (no API spend), both local
tiers run at `temperature=0`. Calibrators fitted under 5-fold CV, reliability on
pooled out-of-fold predictions.

| Gate | n | acceptable | ECE | ECE 95% CI upper | Brier | base-rate Brier | Verdict |
|---|---|---|---|---|---|---|---|
| `small` (0.5B) | 67 | 82% | 0.062 | 0.163 | **0.131** | 0.147 | **PASS** |
| `large` (3B) | 67 | 85% | 0.047 | 0.179 | **0.141** | 0.127 | **FAIL** |

**Gate `large` fails on the Brier check, not on ECE** — and that distinction is
the whole reason the check exists. Its ECE (0.047) is *better* than gate
`small`'s, because a model that predicts the base rate everywhere is trivially
well-calibrated. It just carries no information. Brier-against-baseline is what
separates "calibrated" from "calibrated and useful", and gate `large` is the
former only.

The raw sweep shows the same thing more starkly. The 3B's `geo_mean` barely
moves with correctness:

```
gate large, raw mode          gate small, raw mode
thresh  escalate%  false-cheap   thresh  escalate%  false-cheap
  0.40        2%    9 (13%)        0.40        0%   12 (18%)
  0.62        3%    9 (13%)  <--   0.55       15%    8 (12%)  <-- shipped
  0.70        3%    9 (13%)        0.70       42%    5 ( 8%)
  0.90       22%    4 ( 6%)        0.90       81%    0 ( 0%)
```

Between 0.40 and 0.70, gate `large` escalates 2–3% of requests no matter where
the threshold sits. **The shipped 0.62 is doing almost nothing.** Gate `small`
by contrast responds smoothly — its confidence genuinely tracks correctness.

Where calibration helps, it helps a lot. Gate `small` at `TARGET_SMALL=0.92`
cuts false-cheap from 12% to **2%**:

```
gate small, calibrated
thresh  escalate%  false-cheap  quality
  0.85       48%     2 ( 3%)      97%
  0.92       57%     1 ( 2%)      98%
```

**Decision: stay in `raw` mode.** `should_flip()` requires every gate to pass,
and shipping one measured gate beside one guessed gate would put both behind the
same 0.85 label. This is a real negative result, not a bug — the validator did
exactly the job it was built for.

Three caveats on the sample, stated so the numbers are not over-read:

1. **The auto subset is easier than production traffic.** Short factual and
   arithmetic questions; the 3B scores 85% on them. There is little failure
   signal to learn from, which is a plausible cause of gate `large`'s flat
   confidence here rather than a property of the model.
2. **65 judge-gradable prompts are unlabelled** — the reasoning, summarization
   and open-ended code prompts, which is exactly where the 3B would be expected
   to fail more often. Running `--real-frontier` is the natural next step.
3. `quality_preserved` assumes escalation *fixes* the answer. Reasonable for the
   frontier tier, optimistic for gate `small` escalating to the 3B.

---

### 3.5 The real-frontier run (2026-07-29) — the answer holds, and one gate was softer than it looked

M6 graded against ground truth only, because no real API was configured. This
run used one: `gemini-3-flash-preview` for reference answers,
`gemini-2.5-flash` as a separate judge (D11 — no model grades its own output).

| gate | run | n | acceptable | ECE | ECE 95% CI | Brier | base-rate Brier | verdict |
|---|---|---|---|---|---|---|---|---|
| `small` | M6 | 67 | 82% | 0.062 | [0.030, 0.163] | **0.131** | 0.147 | PASS |
| `small` | now | 69 | 84% | 0.101 | [0.041, 0.175] | **0.134** | 0.134 | **FAIL** — ECE > 0.10 |
| `large` | M6 | 67 | 85% | 0.047 | [0.032, 0.179] | **0.141** | 0.127 | FAIL |
| `large` | now | 68 | 85% | 0.047 | [0.028, 0.157] | **0.135** | 0.125 | **FAIL** — Brier ≈ base rate |

`should_flip()` is still `False`. The gateway stays in `raw`.

**Gate `large` reproduced.** Identical ECE (0.047) and the same
Brier-vs-baseline failure, now with judge-graded labels M6 could not produce. A
partially independent run confirming the original finding is worth more than
the original finding alone.

**Gate `small` flipped PASS → FAIL, and that is not the same as getting
worse.** The ECE confidence intervals overlap heavily — [0.030, 0.163] against
[0.041, 0.175] — so the two runs are **not statistically distinguishable**. The
0.10 threshold is a bright line drawn through a fuzzy measurement; M6 landed
just under it and this run just over.

What the flip does show is that **M6's pass was never robust**. A verdict that
moves across the threshold on a modest change of sample composition was
marginal from the start, and reporting it as a clean pass overstated it.

**Where the difference came from.** M6's labels were 67 auto-graded, all of
them. This run is 60 auto + 9 judge-graded per gate, and the judge-graded ones
are the *harder* prompts — summarization rather than fact lookup, the ones with
no ground truth to check against. Gate `small`'s Brier collapsed from beating
the base rate (0.131 vs 0.147) to exactly matching it (0.134 vs 0.134) once
those were included. On the prompts where the gate's judgement matters most,
its confidence carries no information.

**This is not the full real-frontier calibration**, and should not be cited as
one. Only 17 of the ~137 labels came from a real judge. The run stopped at
71/110 prompts because the free tier caps at **20 requests per day per model**
(`GenerateRequestsPerDayPerProjectPerModel-FreeTier`), and 32 judged prompts
were never collected — which would have roughly quintupled the judge-graded
evidence. The wide ECE intervals are the visible consequence: at n≈69 with a
7-feature fit, neither gate's point estimate is firm.

The raw evidence is committed at `eval/runs/full-120tok.json` so this is
reproducible without spending again; `python -m eval.run_eval --real-frontier
--fit-only` refits it offline. (That file was `full.json` until §3.6 renamed it
— the name records the token budget, which turned out to be the thing that
mattered.)

**Eight harness bugs surfaced on the way**, every one of them silent, in code
that had never been run against a real provider — `EVAL_USE_REAL_FRONTIER` was
a documented, settable flag nothing had exercised end to end. The same shape as
the calibrated-mode bug of M13, and the same lesson: a path that ships disabled
is a path that rots. See PLAN.md M21.

### 3.6 The complete run (2026-07-29) — both gates "passed", and the result was an artifact

With billing enabled the quota ceiling lifted and all 110 prompts collected in
483s. For the first time `should_flip()` returned **`True`**.

| gate | run | n | acceptable | ECE | Brier | base-rate Brier | verdict |
|---|---|---|---|---|---|---|---|
| `small` | M6 | 67 | 82% | 0.062 | 0.131 | 0.147 | PASS |
| `small` | complete | 96 | 67% | 0.041 | **0.145** | 0.222 | PASS |
| `large` | M6 | 67 | 85% | 0.047 | 0.141 | 0.127 | FAIL |
| `large` | complete | 96 | 72% | 0.049 | **0.100** | 0.202 | PASS |

**It was not flipped, because the pass does not survive scrutiny.**

Note what moved. Gate `large`'s Brier improved (0.141 → 0.100) *and* the
baseline it must beat worsened (0.127 → 0.202). The second movement is the
larger one, and it has a specific cause: the acceptable rate fell from 85% to
72%, and a more balanced dataset makes the base-rate predictor worse. Beating a
weaker baseline is not the same as being more informative.

The rate fell because of one category:

```
reasoning         0/26 =  0% acceptable
code             15/25 = 60%
summarization    15/21 = 71%
obscure_factual  22/30 = 73%
easy_factual     54/60 = 90%
arithmetic       27/30 = 90%
```

**`--max-tokens` defaulted to 120 while `EVAL_REFERENCE_MAX_TOKENS` was 1024.**
Reference answers ran to a median 1573 characters against the local tiers' 544
— a 2.9x gap — and **26% of local answers (58/220) were cut off mid-sentence at
the cap.** The judge was asked whether a candidate was as good as the reference
while the candidate had a third of the budget. Every reasoning prompt was
truncated, which is why that category reads 0/26.

The local answers *are* genuinely bad on reasoning — qwen2.5:0.5b attributes
rainbow colour order to "the Doppler effect", and the 3B defines density as "a
material's ability to expand upon freezing". Those labels are correct. But
truncation is an aggravating factor of unknown size, and it is not independent
of category.

**The counterfactual settles it.** Refitting on subsets of the same run
(free — no new API calls):

| subset | n (`large`) | `large` Brier vs base | `should_flip()` |
|---|---|---|---|
| all data | 96 | 0.100 vs 0.202 | **True** |
| minus `reasoning` | 82 | 0.118 vs 0.133 | **False** |
| minus truncated answers | 73 | 0.112 vs 0.118 | True |

Removing the confound one way kills the flip; removing it another way leaves it
alive by 0.006 Brier at n=73. A verdict that reverses depending on which
defensible slice you take has not been earned. (The last two rows are the same
73 prompts either way — *every* reasoning prompt was truncated, so the two
confounds are one confound.)

**Consequences, all now fixed:**

- `--max-tokens` defaults to `DEFAULT_MAX_TOKENS` (512), what the gateway
  actually serves. The old default fitted the calibrator to answers a quarter
  the length of the ones it scores in production — and `n_tokens` is one of its
  seven features.
- `check_budget` refuses to extend a run cache across a change of token budget.
  Both halves would look like valid rows, so nothing downstream could detect
  it.
- The harness records what it spends. It tracked tokens for the free local
  tiers and nothing for the paid calls, so pricing a run meant estimating from
  character counts afterwards. `ApiUsage` keeps the provider's usage block
  whole — reasoning models bill hidden thinking as output under keys that vary
  by provider, and a fixed schema drops exactly the tokens likeliest to
  surprise you on a bill.

The run is archived at `eval/runs/full-120tok.json`. This run's `True` was not
acted on; §3.7 re-ran it with the budgets corrected.

### 3.7 The corrected run (2026-07-29) — both gates pass, and it survives being taken apart

Same 110 prompts, two parameters fixed: local tiers at `--max-tokens 512`
(what the gateway serves) and the judge at `reasoning_effort=none`.

| gate | n | acceptable | ECE | ECE CI | Brier | base-rate Brier | verdict |
|---|---|---|---|---|---|---|---|
| `small` | 110 | 59% | 0.040 | [—, 0.120] | **0.136** | 0.242 | **PASS** |
| `large` | 110 | 69% | 0.027 | [—, 0.128] | **0.129** | 0.214 | **PASS** |

`should_flip()` is `True`, and unlike §3.6 it stays `True` under scrutiny:

| subset | flip | note |
|---|---|---|
| all data (n=110) | **True** | |
| minus `reasoning` | **True** | the category that broke §3.6 |
| minus `code` | **True** | where references truncated |
| minus truncated references (19) | **True** | |
| minus local answers at the cap (19) | **True** | |
| minus `code` **and** `reasoning` (n=75) | False | only easy prompts remain, at 79%/84% acceptable |

The single failure removes both hard categories at once, reconstructing the
over-easy distribution that made M6 uninformative — 75 prompts where the
base-rate predictor is already good and there is little left to be informative
about. That is the expected failure, not evidence against the gate.

**Label coverage is complete**: 0 unlabelled tier-slots, n=110 for both gates,
against 96 in §3.6. Local truncation fell from 26% to 9%.

**The acceptable rate fell as answers got longer** — 67%/72% at 120 tokens,
59%/69% at 512 — which is worth stating because the naive expectation is the
opposite. A longer answer has more room to be wrong, and the judge sees a
complete piece of flawed reasoning rather than a fragment that might have been
going somewhere. `reasoning` sits at 3/40 = 8% acceptable *with complete
answers*, so the local models genuinely cannot do these; §3.6's 0/26 was
mostly, but not only, truncation.

**Measured cost, from provider-reported tokens** — the first run in this
project to account for its own spend:

```
judge       86 calls   in 60,768   out    232   thinking 0
reference   43 calls   in    888   out 13,348   thinking 18,203  -> billed 31,551
```

**Total: $0.1139.** At rates verified 2026-07-29 against Google's own pricing
page — `gemini-3-flash-preview` $0.50 in / $3.00 out, `gemini-2.5-flash` $0.30
in / $2.50 out, paid tier, per 1M tokens.

Two things fall out. `reasoning_effort=none` works: 2.7 output tokens per
verdict, no thinking. And on the reference, **thinking exceeded visible output
by 36%** — Gemini's OpenAI layer reports no `reasoning_tokens` key, so those
18,203 billed tokens are visible *only* as the gap in `total_tokens`. A cost
report keyed on `completion_tokens` understates by that much and says nothing.
Google's pricing page labels the output column *"including thinking tokens"*,
which confirms the reading `billable_out` takes: everything that is not input
is charged at the output rate. Of the $0.0951 the references cost, **$0.0546 —
57% — was thinking.**

**Provenance is now recorded with the fit.** A calibrator maps one specific
model's logprob statistics onto P(acceptable); applied to a different model it
still returns a number in [0, 1] that looks exactly like a probability.
`calibrator.json` carries the models, token budget, dataset size and judge, and
the gateway **refuses to start** in `calibrated` mode when they disagree with
what it is configured to run.

**The shipped default stays `raw`** — not because 0.85 is unearned, but
because these weights are specific to `qwen2.5:0.5b` / `qwen2.5:3b` on this
dataset. A default that hard-fails on any model change serves nobody; `raw`
works everywhere and `CONFIDENCE_MODE=calibrated` is one line for a deployment
running the models this was fitted on.

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
