# Contributing

Thanks for looking. This file covers the loop: get it running, make a change,
convince yourself it works, open a PR.

[README.md](README.md) explains what the gateway does.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how it fits together and
is worth reading before changing structure.
[docs/DECISIONS.md](docs/DECISIONS.md) records why things are the way they are —
if you are about to propose a different approach, the argument against it may
already be there, along with what would justify reversing it.

---

## Setting up

[README's Getting started](README.md#getting-started) covers the runtime. For
development you also need the test dependencies, which `requirements.txt`
deliberately does not carry:

```bash
pip install -r requirements-dev.txt
```

Then confirm the machine can actually host this before you build on it:

```bash
python3 scripts/preflight.py
```

It checks the models are pulled, the disk and RAM have room, and — the one that
matters — that your Ollama build really returns token logprobs. The whole
project rests on that. `preflight` exists because it was the single highest-risk
assumption, and it is better to find out in ten seconds than after a day.

## The three commands CI runs

Run these before opening a PR. They are the same ones the workflow runs, so a
green local run is a green CI run.

```bash
.venv/bin/python -m pytest -m "not integration"    # hermetic; needs nothing running
.venv/bin/ruff check app eval scripts mock_frontier tests
.venv/bin/python scripts/check_docs.py             # cross-document consistency
```

`pytest -m integration` needs Ollama, Redis, and (for one test) a running
gateway. It skips rather than fails when a dependency is absent, so running it
without them tells you nothing rather than telling you something wrong.

## What CI checks that those do not

- **`scripts/check_secrets.py`** — every tracked file, scanned for credential
  shapes. GitHub's own secret scanning is enabled too, but this blocks *before*
  a push rather than alerting after one.
- **`scripts/multiworker_check.py`** — runs 4 worker processes and asserts the
  rate limiter, semantic index and single-flight lease are genuinely shared. A
  surprising amount of state is correct on one process and wrong on four.
- **The container** — built and booted with **no Redis and no Ollama**, because
  invariant 4 says the gateway degrades rather than crash-loops.

## Things worth knowing before you change something

**The docs are load-bearing and checked.** `check_docs.py` asserts that
`app/config.py`, `.env.example` and the config table in `docs/PLAN.md` agree,
among ~189 other things. Adding a setting means touching all three — the check
will tell you which one you forgot.

**Every setting must do something.** `tests/test_wiring.py` fails on any config
field that is neither exercised by a test nor explicitly declared inert with a
reason. This exists because `CONFIDENCE_MODE=calibrated` was once settable,
documented, and connected to nothing, while coverage sat at 99%.

**Milestone entries in PLAN.md are history.** They record what was true at that
milestone. Do not update their test counts when the current total moves — a
check now enforces this, after a global find-and-replace quietly rewrote one
entry eight times in an afternoon.

**Ask whether a passing test would fail if the code were broken.** Three times
in this repo's history a green test was green for the wrong reason: a tenant
isolation test whose two tenants each had one key (so they never shared a
window under any setting), ~900 tests pointed at a configuration the gateway
refuses to boot (passing because a stub scored 0.9725), and an integration test
that reported a missing dependency as a contract failure. None were found by
adding tests. All were found by deleting the feature and re-running.

`scripts/mutation.sh` automates the question for the pure modules.

## Changing the routing or the gates

Read [docs/RESEARCH.md](docs/RESEARCH.md) §3 first. It contains measurements
that constrain what is possible, most importantly that **`geo_mean` does not
detect hallucination** — a fabricated answer scored 0.5535 and the gate accepted
it. Any proposal along the lines of "just raise the threshold" runs into §3.2.

The confidence thresholds are not opinions. `eval/run_eval.py` fits and
validates them, and `should_flip()` decides whether the calibrated mode has
earned the default. If you change the gating logic, re-run it — and read §3.6
first, which is the run whose passing result turned out to be an artifact.

## Adding a tier, provider, or backend

- **A tier** is a row in `app/cascade.py`. It is an ordered list on purpose;
  nested if/else routing is what that list exists to prevent. Check the RAM
  budget in CLAUDE.md before adding a fourth resident model — it does not fit
  on 8 GB.
- **A provider** implements the `Provider` protocol in
  `app/providers/base.py`.
- **A backend for the semantic cache** goes behind `app/cache/semantic.py`'s
  interface.

## Pull requests

Branch, push, open a PR. CI runs nine checks; all must pass. `main` requires a
PR and does not accept direct pushes or force pushes.

Commit messages here tend to explain *why*, including what was tried and
rejected. That is a preference, not a rule — but the git log is the densest
documentation in the project, and it is worth keeping that way.

## Running the eval costs money

`eval/run_eval.py --real-frontier` calls a real API: 43 reference completions
and 86 judge calls, about $0.11 at Gemini Flash rates. Every response is cached
in `eval/runs/` and committed, so **refitting is free** — use
`--fit-only` rather than re-spending to recompute something already on disk.

The free path (`python -m eval.run_eval`, no flag) grades against ground truth
only, costs nothing, and still produces a real calibration on a smaller sample.
