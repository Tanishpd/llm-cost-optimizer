#!/usr/bin/env python3
"""Refuse to let a credential reach the repository.

GitHub's own secret scanning is not available here — it needs Advanced
Security, which private repos on this plan do not have — so the check lives in
CI instead. That is not purely a downgrade: this runs before a push rather than
alerting after one, which is the half that actually prevents the leak.

The gap it closes was real and nearly cost something. `.gitignore` matched
`.env` and `.env.local` as *exact* strings, so a routine

    cp .env .env.backup

produced an untracked file holding a live Google API key that `git status`
offered to stage. The pattern is now `.env*`, but a gitignore fix only protects
the filenames somebody thought of. This checks *content*, so a key pasted into
a README, a test fixture, or a docstring is caught too.

Scans tracked files only. Anything untracked is either ignored (fine) or about
to be added, in which case it shows up here the moment it is staged.

    python scripts/check_secrets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each entry is (name, compiled pattern). Keep these specific: a pattern that
# fires on "password" would be ignored within a week, and a check nobody trusts
# is worse than no check because it launders inattention into a green tick.
PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    # `(?!ant-)`: without it this also matches `sk-ant-...`, since "ant-..." is
    # itself a run of 32+ permitted characters. Detection still fired, but
    # under the wrong provider's name — and the label is what tells an operator
    # which console to go and revoke the key in.
    ("OpenAI key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{32,}")),
    ("Anthropic key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]

# Placeholders that must stay greppable. `.env.example` documents the shape of
# every credential the gateway takes, so it is the one file guaranteed to look
# like it holds secrets. Listing the literal allowed values, rather than
# skipping the file wholesale, means a real key pasted there is still caught.
ALLOWED = {
    "mock-key",
    "sk-mock",
    "test-key",
    "changeme",
}

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2"}


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / p for p in out.stdout.split("\0") if p]


def scan() -> list[tuple[Path, int, str, str]]:
    findings = []
    for path in tracked_files():
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable; nothing to match against
        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in PATTERNS:
                match = pattern.search(line)
                if match and match.group(0) not in ALLOWED:
                    findings.append((path, lineno, name, match.group(0)))
    return findings


def main() -> int:
    findings = scan()
    if not findings:
        n = len(tracked_files())
        print(f"  no credentials found in {n} tracked files "
              f"({len(PATTERNS)} patterns)")
        return 0

    print("  CREDENTIALS FOUND IN TRACKED FILES\n")
    for path, lineno, name, value in findings:
        rel = path.relative_to(ROOT)
        # Show enough to identify it, never the whole thing — this output goes
        # to CI logs, which are themselves a place secrets leak from.
        shown = f"{value[:6]}...{value[-4:]}" if len(value) > 12 else "<short>"
        print(f"    {rel}:{lineno}  {name}  {shown}")
    print("\n  A committed credential must be treated as compromised even after")
    print("  removal: it is in the reflog, in every clone, and in any fork.")
    print("  Revoke it at the provider first, then rewrite history.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
