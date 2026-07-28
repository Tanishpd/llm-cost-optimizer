#!/usr/bin/env python3
"""Verify this machine can actually run the gateway, before anything is built on it.

Every milestone from M1 on starts by running this. It fails loudly rather than
letting a later stage fail mysteriously.

    python3 scripts/preflight.py            # all checks
    python3 scripts/preflight.py --json     # machine-readable
    python3 scripts/preflight.py --probe    # also record geo_mean per tier (slow)

Exit codes: 0 all pass (warnings allowed), 1 at least one hard failure.

Deliberately stdlib-only — preflight must work before `pip install` has, and
its job includes telling you when the venv is the thing that's missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "qwen2.5:0.5b")
LARGE_MODEL = os.environ.get("LARGE_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Hard floor from docs/RESEARCH.md §7: below this, macOS cannot grow its swap
# file and starts killing processes. Swap lives on the same disk as the models.
DISK_HARD_FAIL_GB = 10.0
DISK_TARGET_GB = 20.0
SWAP_WARN_PCT = 80.0

Status = Literal["pass", "warn", "fail"]

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = YELLOW = RED = DIM = RESET = ""
MARK = {"pass": f"{GREEN}✓{RESET}", "warn": f"{YELLOW}!{RESET}", "fail": f"{RED}✗{RESET}"}


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _get(path: str, timeout: float = 5.0) -> Any:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, payload: dict, timeout: float = 600.0) -> Any:
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 12)
    return Check(
        "python",
        "pass" if ok else "fail",
        f"{v.major}.{v.minor}.{v.micro}",
        "" if ok else "Python 3.12+ required; see docs/RESEARCH.md §6",
    )


def check_venv() -> Check:
    in_venv = sys.prefix != sys.base_prefix
    try:
        import numpy  # noqa: F401

        deps = "numpy present"
        status: Status = "pass" if in_venv else "warn"
    except ImportError:
        deps = "numpy missing"
        status = "warn"
    return Check(
        "venv",
        status,
        f"{'active' if in_venv else 'NOT active'}, {deps}",
        "" if status == "pass" else "python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt",
    )


def check_disk() -> Check:
    usage = shutil.disk_usage("/System/Volumes/Data" if sys.platform == "darwin" else "/")
    free_gb = usage.free / 1e9
    if free_gb < DISK_HARD_FAIL_GB:
        status: Status = "fail"
        fix = (
            f"Free at least {DISK_HARD_FAIL_GB:.0f} GB. Swap grows on this disk; "
            "below the floor macOS starts killing processes. See docs/RESEARCH.md §7."
        )
    elif free_gb < DISK_TARGET_GB:
        status, fix = "warn", f"Below the {DISK_TARGET_GB:.0f} GB target, but workable."
    else:
        status, fix = "pass", ""
    return Check("disk", status, f"{free_gb:.1f} GB free", fix, {"free_gb": round(free_gb, 1)})


def check_swap() -> Check:
    if sys.platform != "darwin":
        return Check("swap", "pass", "skipped (not macOS)")
    try:
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        ).stdout
        total = float(out.split("total =")[1].split("M")[0])
        used = float(out.split("used =")[1].split("M")[0])
        pct = (used / total * 100) if total else 0.0
    except Exception as exc:  # noqa: BLE001
        return Check("swap", "warn", f"could not read: {exc}")
    status: Status = "warn" if pct > SWAP_WARN_PCT else "pass"
    return Check(
        "swap",
        status,
        f"{used:.0f} MB / {total:.0f} MB ({pct:.0f}%)",
        "Swap is nearly full; free disk or close applications before loading models."
        if status == "warn"
        else "",
        {"used_mb": used, "total_mb": total, "pct": round(pct, 1)},
    )


def check_ollama() -> Check:
    try:
        version = _get("/api/version").get("version", "?")
    except Exception:  # noqa: BLE001
        return Check("ollama", "fail", "not reachable", "Run: ollama serve &")
    return Check("ollama", "pass", f"v{version} at {OLLAMA_BASE_URL}", data={"version": version})


def check_models() -> list[Check]:
    try:
        installed = {m["name"] for m in _get("/api/tags").get("models", [])}
    except Exception:  # noqa: BLE001
        return [Check("models", "fail", "cannot list (is ollama serve running?)", "ollama serve &")]

    checks: list[Check] = []
    for role, model in (("small", SMALL_MODEL), ("large", LARGE_MODEL), ("embed", EMBED_MODEL)):
        present = model in installed or f"{model}:latest" in installed
        checks.append(
            Check(
                f"model:{role}",
                "pass" if present else "fail",
                f"{model} {'present' if present else 'MISSING'}",
                "" if present else f"ollama pull {model}",
            )
        )
    return checks


def check_logprobs() -> Check:
    """The load-bearing assumption. If this fails, the confidence gate cannot
    work as designed and CONFIDENCE_STRATEGY=self_consistency is the fallback."""
    try:
        resp = _post(
            "/api/chat",
            {
                "model": SMALL_MODEL,
                "messages": [
                    {"role": "user", "content": "What is the capital of France? Answer in one word."}
                ],
                "stream": False,
                "logprobs": True,
                "top_logprobs": 5,
                "options": {"temperature": 0, "num_predict": 5},
            },
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        return Check("logprobs", "fail", f"request failed: {exc}", "Check that SMALL_MODEL is pulled.")

    logprobs = resp.get("logprobs")
    if not logprobs:
        return Check(
            "logprobs",
            "fail",
            "no `logprobs` in response",
            "Set CONFIDENCE_STRATEGY=self_consistency — see docs/PLAN.md M0 fallback.",
        )

    first = logprobs[0]
    missing = [k for k in ("token", "logprob") if k not in first]
    if missing:
        return Check("logprobs", "fail", f"entry missing {missing}", "Response shape changed.")

    n_top = len(first.get("top_logprobs") or [])
    return Check(
        "logprobs",
        "pass",
        f"{len(logprobs)} token(s), top_logprobs={n_top}, first={first['token']!r} "
        f"@ {first['logprob']:.4f}",
        data={"n_tokens": len(logprobs), "top_logprobs": n_top},
    )


def check_embeddings() -> Check:
    """Guards the /api/embed vs /api/embeddings shape trap (RESEARCH.md §5)."""
    try:
        resp = _post("/api/embed", {"model": EMBED_MODEL, "input": "preflight"}, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return Check("embeddings", "fail", f"failed: {exc}", f"ollama pull {EMBED_MODEL}")
    embeddings = resp.get("embeddings")
    if not embeddings or not isinstance(embeddings, list) or not isinstance(embeddings[0], list):
        return Check(
            "embeddings",
            "fail",
            f"unexpected shape: {list(resp)}",
            "Expected {'embeddings': [[...]]} from /api/embed (plural, array-of-arrays).",
        )
    dims = len(embeddings[0])
    return Check("embeddings", "pass", f"{EMBED_MODEL}, {dims} dims", data={"dims": dims})


def check_redis() -> Check:
    """Redis being down is a degradation, not a failure (invariant 4) — so this
    warns rather than fails."""
    try:
        import redis  # noqa: PLC0415
    except ImportError:
        return Check("redis", "warn", "redis-py not installed", "pip install -r requirements.txt")
    try:
        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=2)
        client.ping()
        modules = [m.get("name") for m in client.module_list()]
    except Exception as exc:  # noqa: BLE001
        return Check(
            "redis",
            "warn",
            f"unreachable: {type(exc).__name__}",
            "brew services start redis — the gateway still serves with caching disabled.",
        )
    note = "no modules (brute-force cosine, as designed)" if not modules else f"modules: {modules}"
    return Check("redis", "pass", f"reachable, {note}", data={"modules": modules})


def probe_geo_mean() -> list[Check]:
    """Record each chat model's geo_mean on fixed prompts.

    This is what turns THRESHOLD_SMALL/THRESHOLD_LARGE from guesses into
    measurements — the raw-mode thresholds are per-model artifacts of these
    distributions (docs/DECISIONS.md D5).
    """
    from app.confidence import extract_metrics  # noqa: PLC0415

    prompts = [
        ("easy", "What is the capital of France? Answer in one word."),
        ("medium", "Explain why a rainbow has the colour order it does."),
        ("hard", "Prove that every finite integral domain is a field."),
    ]
    checks: list[Check] = []
    for role, model in (("small", SMALL_MODEL), ("large", LARGE_MODEL)):
        scores: dict[str, float] = {}
        for label, prompt in prompts:
            try:
                resp = _post(
                    "/api/chat",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "logprobs": True,
                        "top_logprobs": 5,
                        "options": {"temperature": 0, "num_predict": 120},
                    },
                )
            except Exception as exc:  # noqa: BLE001
                checks.append(Check(f"probe:{role}", "warn", f"{label} failed: {exc}"))
                break
            scores[label] = round(extract_metrics(resp.get("logprobs") or []).geo_mean, 4)
        else:
            summary = ", ".join(f"{k}={v}" for k, v in scores.items())
            checks.append(Check(f"probe:{role}", "pass", f"{model} geo_mean: {summary}", data=scores))
    return checks


# --------------------------------------------------------------------------- #


def run(probe: bool = False) -> list[Check]:
    checks = [check_python(), check_venv(), check_disk(), check_swap(), check_ollama()]
    if checks[-1].status == "pass":
        checks += check_models()
        if any(c.name == "model:small" and c.status == "pass" for c in checks):
            checks.append(check_logprobs())
        if any(c.name == "model:embed" and c.status == "pass" for c in checks):
            checks.append(check_embeddings())
    checks.append(check_redis())
    if probe and all(c.status != "fail" for c in checks):
        checks += probe_geo_mean()
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--probe", action="store_true", help="also measure geo_mean per model (slow)")
    args = parser.parse_args()

    checks = run(probe=args.probe)

    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        width = max(len(c.name) for c in checks)
        print()
        for c in checks:
            print(f"  {MARK[c.status]} {c.name:<{width}}  {c.detail}")
            if c.fix:
                print(f"    {DIM}→ {c.fix}{RESET}")
        print()

    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    if failed:
        if not args.json:
            print(f"  {RED}{len(failed)} check(s) failed{RESET} — fix the above before continuing.\n")
        return 1
    if not args.json:
        suffix = f" ({len(warned)} warning{'s' if len(warned) != 1 else ''})" if warned else ""
        print(f"  {GREEN}All checks passed{suffix}.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
