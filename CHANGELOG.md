# Changelog

Notable changes, newest first. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semver](https://semver.org/).

`app/__init__.py::__version__` is the single source of the version number, and
the release workflow refuses to build a tag that disagrees with it.

## [Unreleased]

### Changed

- **The calibration passes.** Both gates clear validation against a real
  frontier API at n=110 — `small` ECE 0.040 / Brier 0.136 vs base 0.242,
  `large` ECE 0.027 / Brier 0.129 vs base 0.214 — and the result survives every
  counterfactual refit but one that strips the two hardest categories at once.
  `should_flip()` is `True`; **0.85 is earned**. This supersedes the 0.2.0 note
  below, which stands as the record of what was true then.
  [RESEARCH.md §3.7](docs/RESEARCH.md)
- **`raw` is still the default**, now as a portability choice rather than a
  verdict: the weights describe `qwen2.5:0.5b` / `qwen2.5:3b` on this dataset,
  and a default that hard-fails on any model swap serves nobody.

### Added

- **Calibrator provenance.** `calibrator.json` records the models, token
  budget, dataset size and judge it was fitted with, and the gateway refuses to
  start in `calibrated` mode when those disagree with the configured models. A
  calibrator fitted on other models is worse than none — it returns a number in
  [0, 1] that looks exactly like a probability.
- **The eval reports what it spent.** `ApiUsage` per paid call, priced with
  separate judge rates. Thinking tokens are billed as output and Gemini reports
  them *only* as the gap in `total_tokens`; on the last run they exceeded
  visible output by 36%.
- **`EVAL_JUDGE_REASONING_EFFORT`** — `none` disables judge thinking on Gemini
  2.5, which removes both the cost and the failure mode where hidden thinking
  consumed the whole token budget and every verdict came back empty.

### Fixed

- **The eval calibrated on answers it would never serve.** `--max-tokens`
  defaulted to 120 against a gateway serving 512, so 26% of local answers were
  truncated mid-sentence and judged against references 2.9x longer. It now
  follows `DEFAULT_MAX_TOKENS`, and `check_budget` refuses to extend a run
  cache across a change of budget — answer length feeds `n_tokens`, a
  calibrator feature.
- **`.gitignore` only matched `.env` and `.env.local` exactly**, so a routine
  `cp .env .env.backup` left a live API key untracked and fully committable.
  Now `.env*` with an explicit exception for `.env.example`.

## [0.2.0] — 2026-07-29

The version that made it operable. 0.1.0 worked; this one can be run by
somebody who did not write it.

### Added

- **A spend ceiling that ships armed.** `MAX_DAILY_SPEND_USD`, default `$10`,
  across all tenants. Invariant 4 ("Ollama down → escalate everything") is
  correct for availability and costs ~$138/day per client at the shipped rate
  limit; per-tenant budgets could bound it but default to unlimited, so the
  protection was present and unarmed. A business policy may default to off; a
  safety limit may not. ([D27](docs/DECISIONS.md))
- **Per-tenant budgets and attribution.** `TENANTS` maps named accounts to key
  sets and daily caps. Over budget *degrades to local tiers* rather than
  refusing — running out of money for the expensive model should not stop you
  being served. ([D22–D25](docs/DECISIONS.md))
- **A durable audit trail.** Append-only JSONL, one line per request including
  refusals, surviving restarts — which the capped telemetry stream never did.
  Never blocks the request path; drops are counted, never silent.
  ([D26](docs/DECISIONS.md))
- **Streaming**, by buffering the whole cascade and then re-emitting as SSE.
  Does **not** lower time-to-first-token and the docs say so; the gate cannot
  decide without the complete answer. Reverses the original D8.
- **Prometheus exposition.** `/metrics` negotiates on `Accept`;
  `/metrics/prometheus` is unconditional.
- **`scripts/soak.py`** — endurance rather than load: memory, file descriptors,
  latency drift. Distinguishes a leak from a one-off step.
- **Release engineering** — pinned dependencies, Dependabot, `pip-audit` in CI,
  a tagged release workflow publishing a digest-addressed image, and
  [docs/RUNBOOK.md](docs/RUNBOOK.md).

### Fixed

- **`CONFIDENCE_MODE=calibrated` was wired to nothing.** `main.py` never loaded
  `eval/calibrator.json`, so the gates used raw thresholds while `x_hlr`
  reported `calibrated`. The serving calibrator also lacked the
  standardization the fitted weights require, which would have produced a
  number in [0,1] that looked like a probability and was not.
- **`DEFAULT_MAX_TOKENS` and `DEFAULT_TEMPERATURE` were read by nothing** —
  defined, documented in two places, and inert. The real defaults were
  hardcoded in `schemas.py` and happened to match.
- **A forced `gateway:frontier` under a spend cap silently served
  `local_small`**, ignoring the pin.
- **The audit writer could lose or duplicate a batch at shutdown** —
  `asyncio.to_thread` cannot cancel a running thread, so cancellation was the
  wrong shutdown primitive. Now signals with a sentinel.
- **API keys were plain `str`**, not `SecretStr`.
- **Cost aggregates drifted** under load: per-request micro-USD rounding summed
  to a 75 micro-USD gap over 200 requests. Aggregates now derive from
  exactly-summed token counters.

### Changed

- Dependencies pinned exactly. `fastapi>=0.115` meant an upstream release could
  ship itself into a container build with no review.
- `/metrics` gained per-tenant attribution under a separate `utc_day` window.

### Security

- `secrets.compare_digest` is now *enforced* by test, not just used — swapping
  it for `==` previously passed all 707 tests while reintroducing a timing side
  channel.
- The audit trail records the blake2b principal digest, never the raw key.

## [0.1.0] — 2026-07-28

First working gateway: five tiers, both confidence gates, eval harness,
dashboard, auth, rate limiting, circuit breakers, structured logging, container
image.

Ships in `raw` confidence mode, not `calibrated` — M6 fitted both calibrators
and gate `large` failed its validation (good ECE, but no better than the base
rate on Brier: calibrated without being useful). The 85% has not been earned.
[RESEARCH.md §3.4](docs/RESEARCH.md)
