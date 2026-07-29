"""The credential scanner must actually detect credentials.

`scripts/check_secrets.py` exists because GitHub's secret scanning needs
Advanced Security, which this private repo does not have. A detector nobody
tests is the worst kind of security control: it reports green either way, and
the green is what people act on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_secrets import ALLOWED, PATTERNS  # noqa: E402


def _match(text: str) -> str | None:
    for name, pattern in PATTERNS:
        m = pattern.search(text)
        if m and m.group(0) not in ALLOWED:
            return name
    return None


# Every sample is assembled from fragments, never written whole.
#
# Test fixtures for a credential detector are, by construction, the one thing
# that must look exactly like a credential — and a file full of them is
# indistinguishable from a real leak to any scanner that reads it, including
# this one. The first version of this file wrote them literally, and CI failed
# on four counts: `check_secrets.py`, its own `test_the_repo_is_clean`, and
# GitGuardian twice.
#
# Splitting each prefix from its body is what keeps the file *scannable*. An
# allowlist entry for `tests/` would have silenced the noise too, and would
# also have meant a real key pasted here was never caught again.
def _sample(prefix: str, body: str) -> str:
    return prefix + body


CASES = [
    ("Google API key", _sample("AIza", "SyEXAMPLE1234567890abcdefghijklmnopq")),
    ("AWS access key id", _sample("AKIA", "IOSFODNN7EXAMPLE")),
    ("GitHub token", _sample("ghp", "_" + "a" * 36)),
    ("OpenAI key", _sample("sk", "-" + "b" * 40)),
    ("Anthropic key", _sample("sk", "-ant-" + "c" * 40)),
    ("Slack token", _sample("xox", "b-123456789012-abcdefghijklmno")),
    ("private key block", _sample("-----BEGIN RSA ", "PRIVATE KEY-----")),
]


@pytest.mark.parametrize("name,sample", CASES)
def test_each_pattern_detects_its_credential(name, sample):
    assert _match(sample) == name


def test_this_file_does_not_itself_trip_the_scanner():
    """The fixtures above must match the patterns *at runtime* while the source
    they are written in does not. Without this, someone tidying the
    concatenation away would re-break CI and only find out after pushing."""
    from check_secrets import PATTERNS

    source = Path(__file__).read_text()
    for line in source.splitlines():
        for name, pattern in PATTERNS:
            assert not pattern.search(line), (
                f"a literal {name} appears in this file's source: "
                "split the prefix from the body, as the others are"
            )


@pytest.mark.parametrize("line", [
    "FRONTIER_API_KEY=mock-key",           # .env.example's placeholder
    "FRONTIER_API_KEY=",                   # the key removed
    "GATEWAY_API_KEYS=",
    "api_key: changeme",
    "password = hunter2",                  # deliberately not a pattern
    "AIza",                                # too short to be a key
    "# set FRONTIER_API_KEY to a real key",
])
def test_placeholders_and_prose_do_not_fire(line):
    """A check that cries wolf gets ignored within a week, and an ignored check
    launders inattention into a green tick."""
    assert _match(line) is None


def test_the_env_template_stays_scannable():
    """`.env.example` documents every credential the gateway takes, so it is
    the file most likely to look like it holds secrets — and the one a real key
    is most likely to be pasted into. It must pass, but not by being skipped."""
    root = Path(__file__).resolve().parent.parent
    for lineno, line in enumerate((root / ".env.example").read_text().splitlines(), 1):
        assert _match(line) is None, f".env.example:{lineno} looks like a real credential"


def test_the_repo_is_clean():
    """The end-to-end assertion: no tracked file holds a credential."""
    from check_secrets import scan
    findings = scan()
    assert not findings, f"credentials in tracked files: {[(str(p), n) for p, _, n, _ in findings]}"
