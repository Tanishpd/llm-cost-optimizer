# Design Decisions

Why this is built the way it is, including what was rejected. Each entry states the decision, the
reasoning, and what would make it worth revisiting — a decision without a reversal condition is a
prejudice.

Evidence for the factual claims lives in [RESEARCH.md](RESEARCH.md).

---

## D1 — Two local model tiers, not one

**Decision.** `qwen2.5:0.5b` → `qwen2.5:3b` → frontier, with a confidence gate at each hop.

**Why.** A single gate only ever answers "local or not". Two gates make the gateway decide *how much
compute a prompt deserves*, which is the actual thesis. It also produces a genuine cost/quality
frontier across four options in eval rather than a single threshold sweep, and the 0.5B is weak enough
that gate 1 fires often — the mechanism is visible in a demo instead of theoretical.

The RAM ceiling (D2) helps rather than hurts here: 0.5B + 3B together cost less resident memory than
one 7B, and give a graduated ladder instead of a binary.

**Rejected.** One larger local model (simpler, but only one gate to calibrate and no intermediate
step). Three local tiers (a fourth resident model doesn't fit in 8 GB).

**Revisit if.** Eval shows gate 1 accepting <10% of prompts — then the 0.5B is dead weight and the
cascade should collapse to one local tier.

## D2 — RAM, not disk, caps model size

**Decision.** Local models are capped at 3B–4B. `mistral:7b` — named in the original brief — is
excluded.

**Why.** 8 GB unified memory, with swap already 90% full before any model loads. macOS takes ~3 GB
idle and a browser 1–2 GB, leaving ~3.5–4.5 GB. A 7B Q4 needs ~4.8 GB resident and would thrash
against a saturated swap file.

This was initially misread as a disk constraint. Disk was freed; the ceiling didn't move. Worth
recording because it's the kind of thing that gets "fixed" by buying storage and stays broken.

**Revisit if.** Different hardware. On 16 GB+, a 7B–8B tier slots in as a new row in `app/cascade.py`.

## D3 — Geometric mean of token probabilities, not sequence probability

**Decision.** `confidence = exp(mean(logprob_i))`, equivalently 1/perplexity.

**Why.** Sequence probability `∏ p_i` decays exponentially with length — a 100-token answer at a
healthy p=0.9 per token scores 2.6e-5. Any fixed threshold rejects everything longer than a few
tokens; it measures length, not certainty. The geometric mean is length-invariant.

**Rejected.** Sequence probability (the obvious reading of the brief, and wrong). Min-token
probability *as the headline* (too noisy — one rare token tanks a good answer); `min_prob` is
retained in the feature vector as a secondary signal, alongside `p10_prob`, which is the robust form
of the same "weakest link" idea.

**Revisit if.** Eval shows another metric in the feature vector dominating `geo_mean` in the fitted
calibrator, which would be a reason to change the raw-mode headline.

## D4 — Ship `raw` mode, earn the 85% later

**Decision.** Two confidence modes. `raw` uses `geo_mean` with per-model thresholds 0.55/0.62.
`calibrated` (post-eval) maps the feature vector to `P(this tier's answer is as good as the
frontier's)`, with `TARGET_SMALL=0.92` and `TARGET_LARGE=0.85` — the spec's 85% landing on gate 2,
the gate that actually spends money.

**Why.** The brief says "escalate below 85% confidence." Raw `geo_mean` on a small model sits around
an expected 0.5–0.75 *even when correct* (RESEARCH.md §3 — a prior, measured from M0 on), because
language offers many valid phrasings. Shipping 0.85 against the
raw metric would escalate nearly everything and look broken while being arithmetically correct.

Rather than quietly retune the number and claim it's 85%, both modes exist explicitly. In calibrated
mode the 85% means what the brief intended.

**Rejected.** Hardcoding 0.85 against the raw metric (broken). Silently substituting a different
number and calling it 85% (dishonest). Dropping the threshold concept (loses the brief's intent).

**Revisit.** Never collapse the two modes — the distinction is the point. And treat calibration as a
claim requiring evidence: the flip is gated on the quantitative rule in PLAN M6 (per gate, pooled
out-of-fold ECE ≤ 0.10 with bootstrap 95% CI upper bound < 0.20). A fitted model is not a calibrated
one, and ~120 prompts is a small sample for a 7-feature fit. If reliability doesn't hold, staying in
`raw` and reporting that is the correct outcome.

## D5 — One calibrator per gate, and the thresholds are not comparable

**Decision.** Gate 1 (`small`) and gate 2 (`large`) get separately fitted parameters in
`eval/calibrator.json`. In calibrated mode gate 1's acceptance bar is **higher** than gate 2's
(0.92 vs 0.85).

**Why.** They're different decisions with different consequences and different feature distributions.
Sharing parameters would be a subtle correctness bug that still "works" — the worst kind.

The direction matters and is easy to get backwards. Escalating from gate 1 costs ~450 ms and no
money, so the gateway should escalate *readily* there — which means a **high** bar to accept.
Escalating from gate 2 costs real money, so it should accept more willingly — a **lower** bar. Gate 1
is the stricter gate.

**Note on the raw-mode numbers.** `THRESHOLD_SMALL=0.55 < THRESHOLD_LARGE=0.62` does **not** encode
that policy — it can't, because the two thresholds apply to different models whose `geo_mean`
distributions differ (the 0.5B scores systematically lower). The ordering is an artifact of the
models, not a decision. An earlier draft of these docs claimed it expressed the economics, and stated
the economics backwards on top of that; both errors are corrected here.

**Revisit if.** Eval shows the two fits converging to near-identical parameters, which would suggest
the tier difference doesn't matter and one local model suffices (see D1).

## D6 — Ollama serves embeddings; no torch

**Decision.** `nomic-embed-text` over Ollama's `/api/embed`. No `torch`, `transformers`, or
`sentence-transformers`. Runtime Python is numpy-only; `scikit-learn` is eval-only.

**Why.** Originally a disk decision, now a merit one — torch on macOS arm64 is ~471 MB unpacked (not
the ~2 GB usually cited; that figure is Linux + CUDA), so it *would* fit. It's still excluded because
it means a second inference runtime and a second copy of model weights competing for the RAM that D2
already says is scarce — to compute vectors Ollama already serves over HTTP.

**Rejected.** `sentence-transformers` (pulls torch). ONNX Runtime + a quantized MiniLM (leaner, but
still a second runtime and a second artifact to manage for no quality gain).

**Revisit if.** A learned router (D10) needs a model Ollama can't serve. `classifier.py`'s Protocol is
the seam.

## D7 — Brute-force cosine, and Redis stays at 7.2

**Decision.** Semantic search is a numpy matmul over a warm in-process matrix, Redis as source of
truth. No vector index. Do not `brew upgrade redis`.

**Why.** Installed Redis 7.2.7 has no modules — no `FT.*`, no vector index. Redis 8.8.1 *is* available
via homebrew and would ship `VADD`/`VSIM` compiled in (vector-sets is statically linked, gated only on
C11 atomics, which this machine has). It was still declined: upgrading a running server risks
RDB/AOF-compat pain with an awkward downgrade path, and it buys nothing at this scale.

768-dim float32 = 3 KB/vector. At 10k entries that's 31 MB and a 7.7M-MAC matmul — well under 1 ms via
BLAS. An index starts paying somewhere past 100k entries, which a demo cache will never reach.

**Revisit if.** Cache exceeds ~100k entries, or the process needs to run multi-worker. `semantic.py`
is written behind an interface so it's a drop-in.

## D8 — Non-streaming first

**Decision.** `stream: true` returns 400.

**Why.** The gate cannot decide whether to escalate until the complete response and its logprobs
exist. Streaming a local answer means streaming tokens that may then be discarded and replaced by the
frontier's — worse than not streaming. Ollama's non-streaming path accumulates all chunks' logprobs
into the final response, which is exactly what the gate needs.

**Rejected.** Streaming with post-hoc correction (visibly rewrites text the user already read).

**Revisit.** Buffer the local answer, stream only after the decision. Real work, no demo value,
deferred rather than dismissed.

## D9 — The frontier tier speaks real HTTP, even when mocked

**Decision.** `mock_frontier/` is a separate FastAPI service on `:8001` implementing genuine OpenAI
`/v1/chat/completions`, including a truthful `usage` block. The gateway makes real network calls to it.

**Why.** Keeps `FRONTIER_BASE_URL=https://api.openai.com/v1` a config change rather than a rewrite,
and means the escalation path exercises real serialization, timeouts, and failure modes rather than an
in-process stub that can't fail realistically. `MOCK_FAILURE_RATE` exists to test degradation.

**Rejected.** An in-process function returning a static string (the brief's literal suggestion). It
would make the cost numbers fake and the swap-to-real a refactor.

**Revisit if.** A provider requires a non-OpenAI request shape that can't be expressed through
`FRONTIER_BASE_URL` alone — then the seam moves down to `providers/base.Provider` and the mock
becomes one implementation of it rather than a URL you point at.

## D10 — Heuristic router first, learned classifier later

**Decision.** Ship hand-tuned features and weights. Define `classifier.py` as a Protocol from day one,
implement nothing behind it yet.

**Why.** The eval harness in M6 is what generates honest training labels. Writing a learned router
first means training on hand-labeled guesses about what the models can handle — circular. Measure,
then learn.

**One rule the heuristic must respect: router errors are only cheap downward.** A too-low entry
costs latency (the gates catch it); a too-high entry to `frontier` is ungated spend — no gate exists
past that decision. So a heuristic is acceptable for choosing between local tiers, but frontier
entry must require multiple concurring hardness signals (never length alone — summarization is long
and cheap), and M6's **false-expensive rate** audits it.

**Revisit.** After M6, when `eval/runs/` holds real per-tier outcomes.

## D11 — Real API for eval judging, mock for serving

**Decision.** `EVAL_USE_REAL_FRONTIER=true` calls a live provider for reference answers and quality
judgment during eval only. Responses cached to `eval/runs/`. Serving never uses it by default.

**Why.** The mock returns canned text — it cannot tell you whether the 3B's answer was actually good.
Without a real reference the calibrated 85% is decoration. Caching every response means re-fitting
calibrators never re-spends.

**The judge is a separate provider from the reference** (`EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_MODEL` /
`EVAL_JUDGE_API_KEY`, distinct from `FRONTIER_*`), so no model grades its own output. Volume is 1
reference completion + 1 judge call per local tier = 3 calls per prompt, ~360 across the set. Compute
the cost from your configured rates before running — the price constants are placeholders, not quotes.

**Rejected.** Hand-labeling (slow, and calibration is only as good as the labels). Using the 3B to
judge the 0.5B (works for gate 1, but nothing credible judges the top local tier — which is the gate
that controls actual spend).

**Revisit if.** Judge-vs-human agreement is measured and comes back poor. The whole scheme rests on
an unverified assumption — that a frontier model's quality judgment approximates a person's — and it
has not been checked here. If it doesn't hold, the calibrators are fitted to the judge's biases, and
a hand-labeled subset becomes worth the time after all.

Note that M6 ultimately ran **without** the real frontier: the 67 auto-gradable prompts have ground
truth, so no judge was needed and nothing was spent. That path is the cheaper one and should be
preferred whenever the question can be posed as an auto-gradable prompt.

## D12 — `qwen2.5` over `llama3.2`

**Decision.** `qwen2.5:0.5b` and `qwen2.5:3b`.

**Why.** Same family at both tiers keeps tokenization and prompt-format behaviour consistent, so
confidence metrics are comparable across gates — a meaningful property when you're fitting separate
calibrators on the same feature definitions. `qwen2.5:3b` is also 1.93 GB against `llama3.2:3b`'s 2.02
GB, and the 0.5B has no llama equivalent (`llama3.2:1b` is 1.32 GB for a weaker role).

**Revisit if.** Eval shows poor instruction-following at either tier. Swapping is a config change.

## D13 — Local inference counted as $0

**Decision.** `cost_usd = 0` for cache and local tiers. Savings are measured purely against an
all-frontier baseline.

**Why.** Electricity and amortized hardware are real but not meaningfully attributable per request.
Inventing a number would make the headline figure look more rigorous while being less honest.

**A second assumption rides along and must be disclosed with it:** the baseline charges frontier
prices for *every* request, cache hits included. It answers "what would this traffic have cost with no
gateway at all" — the right question — but it does credit cache hits at full frontier price. Both
assumptions are disclosed in README.md's "How savings are counted".

**Revisit if.** Deployed somewhere with metered compute, where per-request local cost is a real number
rather than an estimate.

## D14 — The cascade is ordered data

**Decision.** `app/cascade.py` holds a list of `(name, runner, gate)`. The executor walks it from the
router's entry point. Adding or reordering a tier is a data change.

**Why.** The obvious implementation — nested if/else across five tiers with two gates and cache
short-circuits — becomes unreadable and untestable fast, and every new tier multiplies the branches.
As data, each tier is independently testable and the ordering is inspectable at runtime (which is what
`x_hlr.path` reports).

**Revisit.** Don't. This is an invariant in CLAUDE.md.

## D15 — Bias every cache judgment toward missing

**Decision.** Cosine ≥ 0.95, plus number-set, entity-set, and negation guards that must *all* pass.
Never semantic-cache multi-turn, differing system prompts, or prompts over `SEMANTIC_MAX_CHARS`.

**Why.** High cosine is not sameness. *capital of Australia* / *capital of Austria* and *convert 50
USD* / *convert 70 USD* both exceed 0.95. A wrong cache hit is worse than a miss — it's a confidently
wrong answer served instantly and for free, with no signal that anything went wrong. A miss costs
~120 ms.

The asymmetry is severe enough that every judgment call goes the same way. `guard_rejected` is
reported so over-conservatism is visible and tunable rather than silent.

**Revisit if.** Telemetry shows guard rejections dominating hits, which would mean the guards are
miscalibrated rather than merely strict.

## D16 — Python 3.14.6, the system default

**Decision.** Build the venv on `python3` (3.14.6). No version pinning gymnastics.

**Why.** Every dependency publishes cp314 arm64 macOS wheels — numpy 2.5.1, scikit-learn 1.9.0,
pydantic-core 2.47.0, uvloop, httptools, watchfiles (all verified against the PyPI JSON API). The
usual reason to fall back to 3.12 doesn't apply, and using the default removes a setup step.

**Revisit if.** A future dependency lacks 3.14 wheels — then `python3.13 -m venv .venv`, nothing else
changes.

## D17 — The gateway runs open when no keys are configured

**Decision.** An empty `GATEWAY_API_KEYS` means no authentication. Startup logs
a loud warning; `REQUIRE_AUTH=true` turns an open start into a hard failure.

**Why.** Forcing a key would break the documented single-user quickstart, which
is the primary way anyone first runs this. Defaulting to closed sounds safer and
in practice means people paste a key into a README command without thinking
about it, which is not security either.

The mitigation is that open mode is *loud* and that demanding auth without
supplying keys **refuses to start**. A configuration that silently disables auth
while claiming to enforce it is worse than a crash, because nothing tells you.

**Rejected.** Auth on by default (breaks the quickstart). Auth optional and
silent (the actual dangerous option — it was the state before M9, and a probe
showed an unauthenticated request returning 200 with full access to the frontier
budget).

**Revisit if.** The gateway ever ships as a binary or image people run without
reading anything. Then closed-by-default with a generated key printed at startup
is the better trade.

## D18 — Rate-limit state lives in Redis, not in the process

**Decision.** A sliding window in a Redis sorted set, behind one Lua script.

**Why.** The in-process version was correct for one worker and **silently wrong**
for more: with N workers each enforcing its own window, a configured 60/min
becomes 60N and nothing reports it. On a gateway that spends money per request
that is not a rounding error.

Atomicity is the reason for the Lua script rather than a pipeline: without it, N
workers can each read "59 used" and each admit a request — precisely the race
the limiter exists to prevent.

A *sliding* window rather than a fixed one, because a fixed window lets a client
send 2× the limit across a boundary.

**Fallback.** If Redis is unreachable it uses per-process state and logs that it
has. A limit that multiplies by worker count is worse than a shared one and far
better than none. `/healthz` reports which mode is live rather than leaving it
to be guessed.

**Revisit if.** Multiple gateway hosts appear — the window is per-Redis, so
sharding Redis would reintroduce the same multiplication at a larger scale.

## D19 — Circuit breakers count consecutive failures and admit one probe

**Decision.** Per-tier breakers. Open after N *consecutive* failures, then admit
exactly one probe per cooldown.

**Why consecutive rather than a rate.** A tier failing once an hour is not
broken, and a rate window needs traffic to decay before it recovers. Two
failures in a row on a local model almost always means the model is gone.

**Why one probe.** Without a probe a breaker that opens never closes. Admitting
only one stops a recovering tier being immediately flooded by everything that
queued up behind the outage.

**What it buys.** Measured against a hanging frontier at a 3 s timeout:
**3193 ms → 42 ms** once open. Against a *connection-refused* tier the clock
barely moves, because refused is already fast — there the value shows up as
`short_circuited` in `/healthz` rather than as latency. Worth knowing before
quoting the speedup.

**Revisit if.** Tiers start failing intermittently rather than cleanly; a
consecutive-failure counter tolerates flapping better than a rate, but neither
is right for a tier that fails 40% of the time.

## D20 — Retries are narrow, and never re-bill the frontier

**Decision.** Retry only timeouts and connection errors, twice by default, with
full jitter. The frontier is **not** retried on timeout unless
`retry_billable=true`.

**Why.** A frontier call that timed out may still have been processed and
billed. Retrying it pays twice for one answer, which is exactly the cost this
project exists to control. A connection error is safe because the request never
landed.

Full jitter because without it N clients that failed together retry together and
reproduce the herd that caused the failure.

**Revisit if.** A provider offers idempotency keys — then billable retries
become safe and the flag can default on.

## D21 — Ollama is not containerised

**Decision.** The Dockerfile and compose file ship the gateway, Redis and the
mock frontier. Ollama stays a host service, reached via `OLLAMA_BASE_URL`.

**Why.** Ollama needs the host's GPU/Metal and its own model storage. Putting it
in the compose file would work and would silently give you **CPU-only
inference** — a container that runs correctly while being ten times slower is a
worse outcome than one that refuses to start.

**Consequence.** `docker compose up` is not sufficient on its own; `ollama serve`
must already be running on the host. That is stated in the compose file rather
than discovered.

**Revisit if.** Deploying to a Linux host with GPU passthrough configured, where
the container can actually reach the accelerator.
