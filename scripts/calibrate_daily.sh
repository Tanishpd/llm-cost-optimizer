#!/usr/bin/env bash
# One day's worth of calibration, against a free tier that caps per day.
#
#   ./scripts/calibrate_daily.sh
#
# Run it once a day until it says COMPLETE. Everything is checkpointed after
# every prompt, so each run picks up exactly where the last stopped and nothing
# is ever paid for twice.
#
# Why a script rather than "just run eval": the interesting part is what
# happens when the quota runs out mid-run, and that is easy to get wrong.
# `run_eval` now distinguishes the two limits — a per-*minute* limit is waited
# out, a per-*day* limit stops the run cleanly and fits whatever is collected.
# Before that distinction existed, a daily cap burned the whole backoff ladder
# (30+60+120+240+480s) to rediscover something that would not change for hours,
# then crashed and lost the fit for data already on disk.
#
# The free tier that prompted this: 20 requests per day PER MODEL
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier). With two judge calls per
# prompt, that is ~10 prompts a day.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo
"$PY" - <<'PROGRESS'
import json, pathlib
import eval.dataset as d
cache = pathlib.Path("eval/runs/full.json")
have = json.loads(cache.read_text()) if cache.exists() else {}
judged = {p.id for p in d.DATASET if p.grade != "auto"}
left = [p.id for p in d.DATASET if p.id not in have]
left_judged = [i for i in left if i in judged]
print(f"  before: {len(have)}/{len(d.DATASET)} prompts collected")
print(f"          {len(left_judged)} judged prompts left "
      f"({len(left_judged) * 3} API calls at 1 reference + 2 judge each)")
PROGRESS

echo
"$PY" -m eval.run_eval --real-frontier || true

echo
"$PY" - <<'AFTER'
import json, pathlib
import eval.dataset as d
cache = pathlib.Path("eval/runs/full.json")
# The run can stop before writing anything — a bad key, the budget guard, a
# provider outage on the first call. Reading unconditionally turned each of
# those into a FileNotFoundError traceback that buried the real error printed
# directly above it.
have = json.loads(cache.read_text()) if cache.exists() else {}
left = [p.id for p in d.DATASET if p.id not in have]
print(f"  after:  {len(have)}/{len(d.DATASET)} prompts collected")
if left:
    days = -(-len([i for i in left
                   if i in {p.id for p in d.DATASET if p.grade != 'auto'}]) // 10)
    print(f"          {len(left)} to go — roughly {days} more day(s). Re-run tomorrow.")
else:
    print("          COMPLETE. eval/calibrator.json holds the full fit;")
    print("          record the result in docs/RESEARCH.md alongside 3.5.")
AFTER
echo
