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
                   "single-worker assumption",
                   # M9 shipped auth, limits and rate limiting; ARCHITECTURE §10
                   # claimed the opposite for a while after they existed.
                   "No auth, tenancy, or rate limiting", "Single-user gateway",
                   "Intended behaviour once",
                   # M0 verified logprobs and measured the geo_mean prior; M6
                   # ran the calibration. All three were still written as
                   # pending long after they were answered.
                   "not yet observed on this machine", "not yet measured here",
                   "not yet a recorded figure", "M6 answers", "M0 verifies it",
                   # These were superseded by a later measurement.
                   "not yet at target", "being reclaimed"):
        check(f"no stale status claim: {phrase!r}", phrase not in corpus)


def check_milestones_are_not_still_pending() -> None:
    """A milestone marked DONE must not also be described as upcoming.

    Every one of these docs has, at some point, claimed a finished milestone
    was next. The claim is mechanical to check, so check it.
    """
    plan = (ROOT / "docs/PLAN.md").read_text()
    done = set(re.findall(r"^### (M\d+).*(?:DONE|✅)", plan, re.M))
    corpus = "\n".join(d.read_text() for d in DOCS)

    for m in sorted(done):
        for claim in (f"Next: **{m}**", f"Start at {m} in", f"| {m} | ", f"**{m}** | next"):
            check(f"{m} is done and not also advertised as next ({claim!r})",
                  claim not in corpus)


def check_endpoints_documented() -> None:
    """Every route the app serves must appear in API.md.

    /livez, /readyz, the Docker setup and three error codes all shipped
    undocumented. Routes are enumerable, so this is checkable rather than
    a thing to remember.
    """
    sys.path.insert(0, str(ROOT))
    from app.main import app

    api = (ROOT / "docs/API.md").read_text()
    paths = {
        r.path for r in app.routes
        if getattr(r, "path", "").startswith(("/v1", "/health", "/live", "/ready",
                                              "/metrics", "/dashboard"))
    }
    missing = sorted(p for p in paths if p not in api)
    check("every served route appears in API.md", not missing, str(missing))

    for code in ("401", "413", "429", "503"):
        check(f"error code {code} documented in API.md", code in api)


def check_middleware_limits_documented() -> None:
    """Limits that reject a request must be findable by the person hitting them."""
    corpus = "\n".join(d.read_text() for d in DOCS)
    for var in ("MAX_BODY_BYTES", "MAX_PROMPT_CHARS", "MAX_MESSAGES",
                "RATE_LIMIT_PER_MINUTE", "MAX_OUTPUT_TOKENS",
                "GATEWAY_API_KEYS", "REQUIRE_AUTH"):
        check(f"{var} explained in the docs", var in corpus)


def check_secrets_are_secret() -> None:
    """Anything key-shaped in config must be SecretStr, not str."""
    sys.path.insert(0, str(ROOT))
    from pydantic import SecretStr

    from app.config import Settings

    leaky = [
        name for name, f in Settings.model_fields.items()
        if ("api_key" in name or name.endswith("_secret") or name.endswith("_token"))
        and f.annotation is not SecretStr
        # gateway_api_keys is the *server's* accepted list, compared with
        # compare_digest and never sent anywhere; SecretStr would only make it
        # awkward to iterate.
        and name != "gateway_api_keys"
    ]
    check("credential settings are SecretStr", not leaky, str(leaky))


def check_decisions_are_reversible() -> None:
    """Every decision needs a reversal condition — that's the format's whole point."""
    text = (ROOT / "docs/DECISIONS.md").read_text()
    entries = re.split(r"^## (D\d+) ", text, flags=re.M)[1:]
    pairs = list(zip(entries[::2], entries[1::2], strict=True))
    check("DECISIONS.md has entries", len(pairs) > 0, f"found {len(pairs)}")
    missing = [d for d, body in pairs if "Revisit" not in body]
    check("every decision states what would reverse it", not missing, str(missing))


def check_milestones_are_current() -> None:
    """The README milestone table and PLAN.md must agree on how far this got."""
    plan_ms = set(re.findall(r"^### (M\d+)", (ROOT / "docs/PLAN.md").read_text(), re.M))
    readme = (ROOT / "README.md").read_text()
    readme_ms = set(re.findall(r"\|\s*\*?\*?(M\d+)\*?\*?\s*\|", readme))
    check("README milestone table covers every PLAN milestone",
          not (plan_ms - readme_ms), str(sorted(plan_ms - readme_ms)))
    check("README does not advertise a milestone PLAN has not defined",
          not (readme_ms - plan_ms), str(sorted(readme_ms - plan_ms)))


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
    check_endpoints_documented()
    check_middleware_limits_documented()
    check_secrets_are_secret()
    check_decisions_are_reversible()
    check_milestones_are_current()
    check_milestones_are_not_still_pending()

    print()
    if failures:
        print(f"  {len(failures)} of {checks_run} checks FAILED: {', '.join(failures)}\n")
        return 1
    print(f"  all {checks_run} checks passed\n")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
