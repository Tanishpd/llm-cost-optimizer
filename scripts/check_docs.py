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


def docs_documenting(*markers: str) -> list[str]:
    """Docs that *document* a feature, identified by its configuration knob.

    A doc that explains budgets names `DEFAULT_DAILY_BUDGET_USD`; one that
    merely says the word "budget" may be talking about disk space, which
    RESEARCH.md 7 does. Heading text was tried first and matched both that and
    RUNBOOK's `### 402 budget_exhausted`, which is a symptom entry rather than
    an explanation.

    Returns paths rather than a hardcoded list, so the caveat follows the
    content. Splitting the README into SETUP/RUNNING/USAGE would otherwise
    leave the moved sections unchecked while the checks stayed green.
    """
    out = []
    for doc in DOCS:
        text = doc.read_text()
        if any(m in text for m in markers):
            out.append(doc.relative_to(ROOT).as_posix())
    return out


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
    # Not a hardcoded list: when the README was split into SETUP/RUNNING/
    # USAGE, a fixed trio would have kept passing while the streaming section
    # moved somewhere unchecked.
    for name in docs_documenting("STREAM_CHUNK_CHARS", "stream: true", '"stream": true'):
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
    for name in docs_documenting("DEFAULT_DAILY_BUDGET_USD", "daily_budget_usd"):
        flat = re.sub(r"\s+", " ", (ROOT / name).read_text().replace("*", "")).lower()
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
    for name in docs_documenting("AUDIT_ENABLED", "AUDIT_PATH"):
        flat = re.sub(r"\s+", " ", (ROOT / name).read_text()).lower()
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
        # `| M6 |` only means "a milestone table row" at the START of a line.
        # Mid-row it is an ordinary data cell — RESEARCH.md compares runs in a
        # table whose second column is the run name, and this check flagged it.
        # A rule that fires on legitimate prose pushes people to distort the
        # prose, which is worse than the drift it was guarding against.
        for claim in (f"Next: **{m}**", f"Start at {m} in", f"**{m}** | next"):
            check(f"{m} is done and not also advertised as next ({claim!r})",
                  claim not in corpus)
        check(f"{m} has no pending milestone-table row",
              not re.search(rf"^\|\s*\*?\*?{m}\*?\*?\s*\|.*\b(next|todo|pending|planned)\b",
                            corpus, re.M | re.I))


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


def check_results_match_the_fitted_calibrator() -> None:
    """RESULTS.md showcases numbers that live in eval/calibrator.json.

    A showcase document that drifts from its artifact is worse than none: it
    reads as evidence while being a memory of evidence. Refitting on new data
    changes the JSON and leaves the prose behind, and nothing else would
    notice — the prose is not executed by anything.

    So the headline figures are checked against the file the gateway actually
    loads. If they disagree, one of them is stale and this says which.

    It is a substring test, and the limit is worth naming: a figure quoted in
    two places with only one updated still passes, because the other still
    matches. What it does catch is the case that actually happens — a refit
    moves every number and the write-up keeps the old ones.
    """
    results = ROOT / "docs/RESULTS.md"
    calibrator = ROOT / "eval/calibrator.json"
    if not results.exists() or not calibrator.exists():
        return

    import json

    text = results.read_text()
    data = json.loads(calibrator.read_text())
    for gate, report in (data.get("validation") or {}).items():
        for label, value in (("ECE", report["ece"]), ("Brier", report["brier"]),
                             ("base-rate Brier", report["brier_baseline"])):
            check(f"RESULTS.md quotes gate {gate}'s {label} ({value:.3f})",
                  f"{value:.3f}" in text,
                  "refit without updating the write-up?")
        check(f"RESULTS.md quotes gate {gate}'s n ({report['n']})",
              str(report["n"]) in text)


def check_readme_agrees_with_itself_on_counts() -> None:
    """The README states the test and doc-check totals in several places.

    They must agree with each other. Milestone tables and header blurbs get
    updated; a bare `# 845, hermetic` in a shell comment does not, because it
    does not match whatever pattern the last sweep used. That one sat four
    hundred lines from a table saying 903, contradicting it, for as long as
    nobody read both in the same sitting.

    Cross-file totals are not checked here: PLAN.md's milestone entries are
    history and are *supposed* to disagree with the present. This is strictly
    "does one document contradict itself".
    """
    readme = (ROOT / "README.md").read_text()
    tests = set(re.findall(r"\*\*(\d+) unit tests?\*\*", readme))
    tests |= set(re.findall(r"\|\s*\*\*Unit\*\*\s*\((\d+)\)", readme))
    tests |= set(re.findall(r'pytest -m "not integration"\s*#\s*(\d+)', readme))
    checks = set(re.findall(r"\|\s*\*\*Doc checks\*\*\s*\((\d+)\)", readme))
    checks |= set(re.findall(r"check_docs\.py\s*#\s*(\d+)", readme))

    check("README states a unit-test count", bool(tests))
    check("README's unit-test counts agree with each other", len(tests) <= 1,
          f"found {sorted(tests)} — a sweep updated some and missed others")
    check("README's doc-check counts agree with each other", len(checks) <= 1,
          f"found {sorted(checks)}")


def check_milestone_counts_are_history_not_status() -> None:
    """Milestone entries in PLAN.md record what was true *at that milestone*.

    They must not be rewritten when the current totals move. Bumping the test
    count across the docs with a global find-and-replace does exactly that, and
    it happened repeatedly: M21's line was dragged 845 -> 863 -> 867 -> 872 ->
    888 -> 889 -> 902 -> 903 across one afternoon, quietly restating history to
    match the present each time.

    The tell is cheap to check. Milestones only ever add tests and checks, so
    the counts must not decrease down the file. A global replace makes an early
    entry equal to the newest one, which shows up here as a *later* entry no
    longer being the largest — or as an early one overtaking its successor.
    """
    plan = (ROOT / "docs/PLAN.md").read_text()
    counts = [(int(t), int(d)) for t, d in
              re.findall(r"^(\d+) unit tests, (\d+) doc checks\.$", plan, re.M)]
    check("docs/PLAN.md records milestone test counts", len(counts) >= 2)
    if len(counts) < 2:
        return
    tests_ok = all(a[0] <= b[0] for a, b in zip(counts, counts[1:], strict=False))
    docs_ok = all(a[1] <= b[1] for a, b in zip(counts, counts[1:], strict=False))
    check("milestone test counts never decrease down PLAN.md", tests_ok,
          f"got {[c[0] for c in counts]} — an earlier milestone claims more "
          "tests than a later one, which usually means a global replace "
          "rewrote history")
    check("milestone doc-check counts never decrease down PLAN.md", docs_ok,
          f"got {[c[1] for c in counts]}")


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

    # ...and it must be the *newest* section, not merely present somewhere.
    #
    # Having a section is satisfied forever once a version ships, so bumping
    # `__version__` and forgetting the notes passed silently — which is how
    # `[0.2.0]` ended up describing a release that was never tagged while the
    # package had moved on. The first version heading is the one a reader takes
    # as current, so that is the one that has to agree.
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", changelog, re.M)
    check("CHANGELOG.md's newest section is the current version",
          bool(versions) and versions[0] == __version__,
          f"newest section is {versions[0] if versions else 'none'}, "
          f"package is {__version__}")


def check_runbook_covers_the_alertable_signals() -> None:
    """Every signal the docs tell you to alert on must have an entry saying
    what to do about it. An alert with no runbook entry is a page at 3am with
    no next step."""
    runbook = (ROOT / "docs/RUNBOOK.md").read_text()
    for signal in ("hlr_budgets_enforced", "hlr_audit_dropped_total",
                   "hlr_circuit_state", "spend_ceiling_reached",
                   "budget_exhausted", "all_tiers_down"):
        check(f"RUNBOOK covers {signal}", signal in runbook)


def check_eval_cost_matches_the_dataset() -> None:
    """The documented API-call count must come from the dataset, not memory.

    It said ~360 for a long time — computed as "every prompt × 3 calls" — when
    the true figure is a third of that, because most prompts are auto-gradable
    and never call out. A cost figure that overstates by 3x is the kind of
    thing that stops someone running the eval at all.
    """
    sys.path.insert(0, str(ROOT))
    import eval.dataset as dataset

    auto = sum(1 for p in dataset.DATASET if p.grade == "auto")
    judged = len(dataset.DATASET) - auto
    calls = judged * 3
    corpus = "\n".join(d.read_text() for d in DOCS)

    check(f"docs state the real eval call count ({calls})", str(calls) in corpus,
          f"{judged} judged prompts x 3 calls; {auto} auto-gradable ones are free")
    check("no doc still claims the old ~360 figure",
          "360 calls" not in corpus and "≈ 360 —" not in corpus)


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
                "scripts/mutation.sh", "scripts/multiworker_check.py",
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
    check_eval_cost_matches_the_dataset()
    check_results_match_the_fitted_calibrator()
    check_readme_agrees_with_itself_on_counts()
    check_milestone_counts_are_history_not_status()
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
