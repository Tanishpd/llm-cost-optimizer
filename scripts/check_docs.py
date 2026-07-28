#!/usr/bin/env python3
"""Cross-document consistency checks, run in CI.

This doc set has drifted twice already — once with a threshold rationale that
contradicted its own numbers, once with `.env.example` and the PLAN config
table falling out of sync. Both were caught by hand. These are the same checks,
automated, so the third time is caught by a machine.

Deliberately stdlib-only and dependency-free: it should run on a bare checkout.

    python3 scripts/check_docs.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted(ROOT.glob("*.md")) + sorted((ROOT / "docs").glob("*.md"))

# Vars owned by the separate mock_frontier service, not app/config.py.
MOCK_VARS = {"MOCK_LATENCY_MS", "MOCK_JITTER_MS", "MOCK_FAILURE_RATE"}

failures: list[str] = []
checks_run = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks_run
    checks_run += 1
    if ok:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{f': {detail}' if detail else ''}")
        failures.append(name)


def check_links() -> None:
    broken = []
    for doc in DOCS:
        base = doc.parent
        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", doc.read_text()):
            target = match.group(2)
            if target.startswith(("http", "#", "mailto")):
                continue
            path = (base / target.split("#")[0]).resolve()
            if not path.exists():
                broken.append(f"{doc.name}: {match.group(0)}")
    check("all internal doc links resolve", not broken, "; ".join(broken[:3]))


def check_env_parity() -> None:
    """app/config.py, .env.example and the PLAN config table must agree."""
    sys.path.insert(0, str(ROOT))
    from app.config import Settings

    fields = {f.upper() for f in Settings.model_fields}
    env = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M))
    plan = set(re.findall(r"\|\s*`([A-Z][A-Z0-9_]+)`\s*\|", (ROOT / "docs/PLAN.md").read_text()))

    check("every .env.example var exists in config.py", not (env - fields - MOCK_VARS),
          str(sorted(env - fields - MOCK_VARS)))
    check("every PLAN table var exists in config.py", not (plan - fields - MOCK_VARS),
          str(sorted(plan - fields - MOCK_VARS)))
    check("every config.py setting is documented", not (fields - env - plan),
          str(sorted(fields - env - plan)))


def check_constants_agree() -> None:
    """Numbers that appear in both code and prose."""
    sys.path.insert(0, str(ROOT))
    from app.config import Settings
    from eval.calibration import MAX_ECE

    s = Settings(_env_file=None)
    corpus = "\n".join(d.read_text() for d in DOCS)

    for label, value in [
        ("THRESHOLD_SMALL", s.threshold_small),
        ("THRESHOLD_LARGE", s.threshold_large),
        ("TARGET_SMALL", s.target_small),
        ("TARGET_LARGE", s.target_large),
        ("SEMANTIC_THRESHOLD", s.semantic_threshold),
    ]:
        check(f"{label}={value} appears in the docs", str(value) in corpus)

    check("calibration ECE gate documented", str(MAX_ECE) in corpus)
    check("gate 1's calibrated bar is higher than gate 2's",
          s.target_small > s.target_large,
          "escalating from gate 1 is nearly free, so it should demand more")
    check("raw thresholds are NOT claimed to encode economics",
          "trigger-happy" not in corpus,
          "an earlier draft made this claim, and it was backwards")


def check_model_names() -> None:
    sys.path.insert(0, str(ROOT))
    from app.config import Settings

    s = Settings(_env_file=None)
    corpus = "\n".join(d.read_text() for d in DOCS)
    for model in (s.small_model, s.large_model, s.embed_model):
        check(f"{model} documented", model in corpus)
    check("mistral:7b still recorded as excluded", "mistral:7b" in corpus)


def check_no_stale_status() -> None:
    corpus = "\n".join(d.read_text() for d in DOCS)
    for phrase in ("No code written yet", "implementation not started", "not wired until M4",
                   "no code exists yet", "assumes one process", "it's untested",
                   "single-worker assumption"):
        check(f"no stale status claim: {phrase!r}", phrase not in corpus)


def check_promised_files_exist() -> None:
    """Anything the file tree promises should be real, or the quickstart lies."""
    for rel in ("requirements.txt", "requirements-dev.txt", ".env.example", "LICENSE",
                "app/main.py", "app/cascade.py", "app/confidence.py", "app/security.py",
                "eval/dataset.py", "eval/run_eval.py", "scripts/preflight.py",
                "scripts/demo.py", "scripts/loadtest.py", "static/dashboard.html",
                "mock_frontier/main.py", "Dockerfile", "docker-compose.yml",
                "app/resilience.py", "app/ratelimit.py", "app/logging_setup.py"):
        check(f"{rel} exists", (ROOT / rel).exists())


def check_dashboard_is_self_contained() -> None:
    body = (ROOT / "static/dashboard.html").read_text()
    for pattern in ("http://", "https://", "@import"):
        check(f"dashboard has no {pattern}", pattern not in body)


def main() -> int:
    print("\ncross-document consistency\n")
    check_links()
    check_env_parity()
    check_constants_agree()
    check_model_names()
    check_no_stale_status()
    check_promised_files_exist()
    check_dashboard_is_self_contained()

    print()
    if failures:
        print(f"  {len(failures)} of {checks_run} checks FAILED: {', '.join(failures)}\n")
        return 1
    print(f"  all {checks_run} checks passed\n")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
