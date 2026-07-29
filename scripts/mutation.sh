#!/usr/bin/env bash
# Mutation testing: does the suite actually *fail* when the code is wrong?
#
#   ./scripts/mutation.sh          # full run, ~3 minutes
#   ./scripts/mutation.sh results  # survivors from the last run
#
# Coverage says a line executed. This says an assertion would have caught it
# being wrong — which is a different question, and the one that matters. This
# project had 95% coverage while `CONFIDENCE_MODE=calibrated` was wired to
# nothing and `DEFAULT_MAX_TOKENS` was read by nothing.
#
# NOT a CI gate, deliberately:
#
#   - it takes minutes, not seconds, so gating every PR on it trades a lot of
#     feedback latency for a signal that moves slowly;
#   - a chunk of the output is *equivalent mutants* — changes that alter the
#     source without altering behaviour, which no test can kill and which a
#     percentage target would push someone into writing brittle tests to chase.
#
# The workflow is: run it when you change the pure modules, read the survivors,
# and either write the test or record why the mutant is equivalent.
#
# Scope is `only_mutate` in setup.cfg — the pure-logic modules. Against I/O
# code most mutants are unkillable noise (a reordered await, a changed log
# line) and they bury the real findings. The tool is only useful while its
# output is short enough to read.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin}

if [ "${1:-run}" = "results" ]; then
    "$PY/mutmut" results 2>/dev/null | grep ": survived" | sed 's/^ *//'
    exit 0
fi

rm -rf mutants
"$PY/mutmut" run --max-children "${MAX_CHILDREN:-6}"

echo
echo "Survivors (a change no test noticed):"
echo
"$PY/mutmut" results 2>/dev/null | grep ": survived" | sed 's/^ *//;s/: survived//' |
    while read -r m; do
        printf "  %-52s " "$m"
        "$PY/mutmut" show "$m" 2>/dev/null | grep -E "^\+" | grep -v "^+++" | head -1 | cut -c1-70
    done

cat <<'NOTE'

Baseline at 2026-07-29: 1953 mutants, 19 survivors.

All 19 are one of two kinds, and neither is worth a test:

  - a changed *detail string* — the prose explaining why a guard rejected, or
    a log message. The verdict and the machine-readable code are killed; only
    the wording survives. Asserting exact prose makes tests brittle without
    protecting behaviour.

  - an *equivalent mutant* — e.g. `similarity()`'s `if not ta or not tb`
    becoming `and`: when exactly one side is empty the fall-through computes
    0/n = 0.0, which is the same answer. No test can kill it because nothing
    observable changed.

The run that established this baseline found six real gaps, since fixed:

  the global budget counter ignored the window it was given, so at a UTC day
    boundary the tenant and gateway totals silently disagreed
  a zero-cost request wrote to Redis twice on the fastest path in the system
  a 1 micro-USD charge was dropped rather than recorded
  ns_to_ms could be rescaled or re-rounded and nothing noticed
  max_overshoot_micro's per-million divisor was unasserted at any scale that
    could separate it
  sequence_probability([]) — a deliberately-unused counter-example whose
    behaviour was unpinned

If a run reports more than ~20 survivors, something regressed. Read them.
NOTE
