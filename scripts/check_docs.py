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
    # Case-insensitive: PLAN.md carried "single-user gateway" in lowercase for
    # two milestones after auth shipped, and a case-sensitive check walked
    # straight past it.
    corpus = "\n".join(d.read_text() for d in DOCS).lower()
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
                   "not yet at target", "being reclaimed",
                   # M11 shipped streaming. `unsupported_streaming` is an error
                   # code that no longer exists, so any mention is stale — that
                   # is why this list checks the identifier rather than prose
                   # like "returns 400", which a history note may legitimately
                   # quote when describing what D8 used to say.
                   "unsupported_streaming",
                   "No streaming.",
                   # M12 shipped tenancy and the audit trail.
                   "No tenancy.", "No persistent audit log.",
                   "no per-tenant budgets"):
        check(f"no stale status claim: {phrase!r}", phrase.lower() not in corpus)


def check_streaming_is_not_oversold() -> None:
    """Streaming here does NOT improve time-to-first-token, and the docs must
    keep saying so.

    This is the easiest overclaim in the project: "we support streaming" reads
    as "responses start sooner", and it is the opposite of what buffer-then-
    stream does. Every doc that describes the feature has to carry the caveat.
    """
    # Markdown wraps, and the caveat is a full sentence, so it lands across a
    # line break as often as not. Normalize whitespace and drop emphasis rather
    # than forcing the prose to fit a grep.
    for name in ("README.md", "docs/API.md", "docs/DECISIONS.md"):
        text = (ROOT / name).read_text()
        flat = re.sub(r"\s+", " ", text.replace("*", "")).lower()
        caveat = any(p in flat for p in (
            "not lower time-to-first-token",
            "time-to-first-token does not improve",
            "time-to-first-token is unchanged",
        ))
        check(f"{name} does not oversell streaming", "stream" not in flat or caveat,
              "say plainly that TTFB is unchanged")


def check_budget_caveats_are_stated() -> None:
    """Two things about budgets must never be quietly dropped from the docs.

    A spend cap that is *approximate* and *fails open* is fine — both are
    defensible trades (D24, D25). What would not be fine is documenting the
    feature without them, because a reader would then reasonably believe the
    cap is exact and always applied. Neither is true.
    """
    for name in ("README.md", "docs/API.md"):
        flat = re.sub(r"\s+", " ", (ROOT / name).read_text().replace("*", "")).lower()
        if "daily_budget" not in flat and "budget" not in flat:
            continue
        check(f"{name} states the budget overshoot bound",
              "overshoot" in flat or "approximate by" in flat,
              "the cap is exceeded by up to one request; say so")
        check(f"{name} states that budgets fail open without Redis",
              "not enforced" in flat or "fails open" in flat,
              "a reader would otherwise assume the cap always applies")


def check_audit_is_not_confused_with_telemetry() -> None:
    """They answer different questions and the docs must keep them apart.

    "We have an audit log" plus "we have /metrics/recent" reads as duplication
    unless the distinction is written down, and the first person to
    consolidate them would delete the durable one.
    """
    for name in ("README.md", "docs/API.md", "docs/DECISIONS.md"):
        flat = re.sub(r"\s+", " ", (ROOT / name).read_text()).lower()
        if "audit" not in flat:
            continue
        check(f"{name} distinguishes the audit trail from telemetry",
              "capped" in flat or "10k" in flat or "survives restart" in flat,
              "say what telemetry cannot do that this does")


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


def check_request_defaults_match_config() -> None:
    """API.md's request table must not hardcode a default the operator can change.

    It said `max_tokens | int | 512` while the real default had become
    `DEFAULT_MAX_TOKENS` — a doc stating a constant where a knob exists is the
    same lie as a knob wired to nothing, told from the other end.
    """
    sys.path.insert(0, str(ROOT))
    from app.config import Settings

    s = Settings(_env_file=None)
    api = (ROOT / "docs/API.md").read_text()

    for field, setting in (("temperature", "DEFAULT_TEMPERATURE"),
                           ("max_tokens", "DEFAULT_MAX_TOKENS")):
        row = re.search(rf"^\|\s*`{field}`\s*\|.*$", api, re.M)
        check(f"API.md documents a `{field}` row", row is not None)
        if row:
            check(f"API.md's `{field}` default names {setting}",
                  setting in row.group(0),
                  "it is operator-configurable, not a constant")

    # And the parenthesised value in that row must be the real one.
    for field, value in (("temperature", s.default_temperature),
                         ("max_tokens", s.default_max_tokens)):
        row = re.search(rf"^\|\s*`{field}`\s*\|.*$", api, re.M)
        if row:
            check(f"API.md's `{field}` default value is {value}",
                  str(value) in row.group(0))


def check_version_is_single_sourced() -> None:
    """`app/__init__.py` is the version; everything else must agree.

    It was hardcoded in pyproject.toml *and* in the FastAPI constructor, which
    is how a release ends up reporting a version it is not. The release
    workflow refuses a tag that disagrees; this catches the disagreement
    earlier, on the PR that introduced it.
    """
    sys.path.insert(0, str(ROOT))
    from app import __version__

    pyproject = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version = "([^"]+)"', pyproject, re.M)
    check("pyproject.toml declares a version", m is not None)
    if m:
        check(f"pyproject.toml version matches app/__init__.py ({__version__})",
              m.group(1) == __version__, f"pyproject says {m.group(1)}")

    check("app/main.py does not hardcode a version",
          'version="0.' not in (ROOT / "app/main.py").read_text(),
          "use `version=__version__`")

    changelog = (ROOT / "CHANGELOG.md").read_text()
    check(f"CHANGELOG.md has a section for {__version__}",
          f"[{__version__}]" in changelog,
          "a release with no notes is a release nobody can review")


def check_runbook_covers_the_alertable_signals() -> None:
    """Every signal the docs tell you to alert on must have an entry saying
    what to do about it. An alert with no runbook entry is a page at 3am with
    no next step."""
    runbook = (ROOT / "docs/RUNBOOK.md").read_text()
    for signal in ("hlr_budgets_enforced", "hlr_audit_dropped_total",
                   "hlr_circuit_state", "spend_ceiling_reached",
                   "budget_exhausted", "all_tiers_down"):
        check(f"RUNBOOK covers {signal}", signal in runbook)


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
                "scripts/demo.py", "scripts/loadtest.py", "scripts/soak.py",
                "static/dashboard.html",
                "mock_frontier/main.py", "Dockerfile", "docker-compose.yml",
                "app/resilience.py", "app/ratelimit.py", "app/logging_setup.py",
                "app/streaming.py", "app/budgets.py", "app/audit.py",
                "app/prometheus.py", "CHANGELOG.md", "docs/RUNBOOK.md",
                ".github/dependabot.yml", ".github/workflows/release.yml"):
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
    check_version_is_single_sourced()
    check_runbook_covers_the_alertable_signals()
    check_request_defaults_match_config()
    check_endpoints_documented()
    check_middleware_limits_documented()
    check_secrets_are_secret()
    check_decisions_are_reversible()
    check_milestones_are_current()
    check_milestones_are_not_still_pending()
    check_streaming_is_not_oversold()
    check_budget_caveats_are_stated()
    check_audit_is_not_confused_with_telemetry()

    print()
    if failures:
        print(f"  {len(failures)} of {checks_run} checks FAILED: {', '.join(failures)}\n")
        return 1
    print(f"  all {checks_run} checks passed\n")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
