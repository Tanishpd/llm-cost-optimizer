# Results

What was measured, and what it means. Every number here comes from committed
evidence — [`eval/calibrator.json`](../eval/calibrator.json) and
[`eval/runs/`](../eval/runs/) — and can be recomputed offline for $0 with
`python -m eval.run_eval --real-frontier --fit-only`.

Method, caveats and the runs that were thrown away: [RESEARCH.md §3](RESEARCH.md).

---

## The claim

> **Can a model's own token log-probabilities tell you when its answer is good
> enough, so you only pay for a bigger model when you need one?**

Yes for routing, with a specific and important exception. Both halves below.

---

## 1. The confidence gates are calibrated and informative

Graded against a real frontier API, `qwen2.5:0.5b` / `qwen2.5:3b`, n = 110.

| gate | n | acceptable | ECE | ECE 95% CI | Brier | base-rate Brier | verdict |
|---|---|---|---|---|---|---|---|
| `small` (0.5B → 3B) | 110 | 59% | **0.040** | [0.025, 0.120] | **0.136** | 0.242 | **PASS** |
| `large` (3B → frontier) | 110 | 69% | **0.027** | [0.026, 0.128] | **0.129** | 0.214 | **PASS** |

Two things have to hold, and they are different:

- **ECE ≤ 0.10** — when the gate says 70%, it is right about 70% of the time.
- **Brier below the base rate** — the gate carries *information*, not just
  calibration. A model predicting the base rate everywhere is perfectly
  calibrated and useless; this is the check that separates the two, and it is
  the one an earlier run failed.

`should_flip()` is `True`, so `CONFIDENCE_MODE=calibrated` ships as the default
and **0.85 means "85% chance this tier's answer is as good as the frontier's"**
rather than a raw score that happens to sit near it.

### It survives being taken apart

A single passing number is weak evidence. Refitting on subsets — free, the run
cache is on disk — tests whether the verdict depends on any one slice:

| subset | n | `large` Brier vs base | flips? |
|---|---|---|---|
| everything | 110 | 0.129 vs 0.214 | **yes** |
| minus `reasoning` | 90 | 0.129 vs 0.153 | **yes** |
| minus `code` | 95 | 0.136 vs 0.212 | **yes** |
| minus answers that hit the token cap | 95 | 0.119 vs 0.183 | **yes** |
| minus `code` **and** `reasoning` | 75 | 0.131 vs 0.134 | no |

The one failure removes both hard categories at once, leaving 75 easy prompts
where the base rate is already 84% and there is little left to be informative
about. That is the expected failure, not a counter-example.

---

## 2. The result that was thrown away

An earlier complete run produced `flip = True` as well. It was an artifact, and
the difference between the two is the most useful thing in this document.

The eval generated local answers at 120 tokens while reference answers got
1024, so the judge compared candidates against references **2.9× longer**. 26%
of local answers were cut off mid-sentence; *every* `reasoning` prompt was.

| run | subset | flips? |
|---|---|---|
| 120-token (artifact) | everything | **yes** |
| 120-token (artifact) | minus `reasoning` | **no** |
| corrected | everything | **yes** |
| corrected | minus `reasoning` | **yes** |

Same headline, opposite robustness. The artifact's verdict reversed on the
first slice removed; the corrected one holds through all four.

It was caught by refitting on subsets, which cost nothing and took a minute.
The superseded run is kept at [`eval/runs/full-120tok.json`](../eval/runs/full-120tok.json)
— deleting the data behind a discarded conclusion would make the conclusion
unfalsifiable.

---

## 3. What the gate cannot do

**`geo_mean` does not detect hallucination.** Asked when the Treaty of Kadesh
was signed (c. 1259 BC, Egypt and the Hittites), `qwen2.5:0.5b` answered *"1956
… between Egypt and Israel"* — a fabricated date and the wrong counterparty —
and scored **0.5535**, comfortably above the 0.55 raw threshold. The gate
accepted it.

This is a property of the metric, not a tuning problem. Token log-probabilities
measure *fluency*: how confidently the model produced those tokens. A model
that has confidently learned something false is fluent about it. No threshold on
a single fluency statistic separates the two, and raising the bar only trades
false-cheap for false-expensive.

It is why the router's guess is allowed to be wrong in one direction only, and
why [RESEARCH.md §3.2](RESEARCH.md) is the section to read before proposing
changes to the gating logic.

---

## 4. Where the local models actually fail

Acceptable rate by category, n = 220 tier-answers, no unlabelled slots:

| category | acceptable | |
|---|---|---|
| arithmetic | 28/30 | 93% |
| easy factual | 54/60 | 90% |
| obscure factual | 22/30 | 73% |
| summarization | 18/30 | 60% |
| code | 16/30 | 53% |
| **reasoning** | **3/40** | **8%** |

`reasoning` at 8% is a real finding, measured with complete 512-token answers
rather than truncated ones. The small models genuinely cannot do multi-step
explanation — `qwen2.5:0.5b` attributes rainbow colour order to "the Doppler
effect", and the 3B defines density as "a material's ability to expand upon
freezing".

**The calibrator learned this.** Live, in calibrated mode:

```
"What is the capital of France?"      gate small 0.995 / 0.92  accept  → served by the 0.5B
"Why does ice float on water?"        gate small 0.503 / 0.92  escalate
                                      gate large 0.201 / 0.85  escalate → walked all three tiers
```

A trivially factual prompt is served free; a reasoning prompt is escalated on
sight. The measurement transferred to live traffic.

---

## 5. Labels

| source | count | how |
|---|---|---|
| auto | 134 | exact match against ground truth, token-bounded |
| judge | 86 | `gemini-2.5-flash` comparing against a `gemini-3-flash-preview` reference |
| unlabelled | **0** | |

Auto-grading is checked forbidden-answers-first: a model that names a
plausible-but-wrong answer has failed even if it also mentions something
correct. Matching is token-bounded, because `"4" in "42"` is `True` and would
silently inflate every arithmetic score.

---

## 6. What the measurement cost

129 API calls. Free-tier-gradable prompts make none.

| | calls | tokens in | tokens out | thinking |
|---|---|---|---|---|
| reference | 43 | 888 | 13,348 | 18,203 |
| judge | 86 | 60,768 | 232 | 0 |

**$0.1139** at rates verified against Google's pricing page on 2026-07-29
(`gemini-3-flash-preview` $0.50/$3.00, `gemini-2.5-flash` $0.30/$2.50 per 1M).

Two details worth carrying away:

- **Thinking tokens exceeded visible output by 36%** on the reference model, and
  Gemini reports them *only* as the gap in `total_tokens` — there is no
  `reasoning_tokens` key. A cost report keyed on `completion_tokens` understates
  by that much and says nothing. Of the $0.0951 the references cost, **$0.0546
  (57%) was thinking.**
- **`reasoning_effort=none` works**: 2.7 output tokens per judge verdict, zero
  thinking, across 86 calls.

---

## 7. Latency, measured

Via `/metrics` on the development machine. Small sample (n = 1–7 per tier) — read
as orders of magnitude, not benchmarks.

| tier | cost | p50 | p95 |
|---|---|---|---|
| exact cache | $0 | **0.2 ms** | 0.2 ms |
| semantic cache | ~$0 | **40 ms** | 60 ms |
| `qwen2.5:0.5b` | $0 | **506 ms** | 1.2 s |
| `qwen2.5:3b` | $0 | **862 ms** | 2.2 s |
| frontier | metered | **1.9 s** | 1.9 s |

The frontier figure is the mock's configured `MOCK_LATENCY_MS`; a real provider
will differ.

---

## Limits on all of the above

Stated here rather than left for a reader to discover:

1. **One measurement, not a track record.** n = 110 on one authored dataset. The
   ECE point estimates are strong (0.040 / 0.027) but their 95% intervals reach
   0.120 and 0.128 — at n = 110 with a 7-feature fit, that is the sample size
   talking.
2. **The judge is same-family with the reference model** — `gemini-2.5-flash`
   grading `gemini-3-flash-preview` output. Weaker independence than
   [D11](DECISIONS.md) intends.
3. **The calibrator describes two specific models.** These weights map
   `qwen2.5:0.5b`'s logprob statistics onto P(acceptable). Applied elsewhere they
   still return a number in [0, 1] that looks exactly like a probability, which
   is why the gateway refuses to start when its recorded provenance disagrees
   with the configured models.
4. **Never run under production load.** `scripts/loadtest.py` is synthetic, and
   the frontier tier in the default deployment is a mock. Real-provider *eval*
   has happened; real-provider *serving* has not.
