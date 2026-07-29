#!/usr/bin/env python3
"""Endurance check: run for a while, and watch for the things that only appear
after a while.

    python3 scripts/soak.py --minutes 30 --rps 2

`loadtest.py` answers "does it survive 500 requests at concurrency 48" and
takes seven seconds to do it. Everything it can find is a *concurrency* bug.
This answers a different question — "is it still the same process an hour
later" — and the bugs it can find are the ones that are invisible in seven
seconds:

- **Memory growth.** A leak of a few KB per request is nothing in a load test
  and is a restart every day in production.
- **File-descriptor growth.** A socket or file handle not returned to the pool
  shows up as `Too many open files` after hours, at which point everything
  fails at once.
- **Latency drift.** Steady p95 growth means something is accumulating —
  a dict that only grows, an index that is never pruned.
- **Counter divergence.** The cost identities hold at request 500; the
  question is whether they still hold at request 50,000.

Deliberately gentle by default (2 rps): this is not a load test, and running it
hot would mean the machine's thermal behaviour becomes the variable.

**What it cannot tell you.** Thirty minutes is not thirty days, and a leak
smaller than measurement noise stays invisible. It bounds the problem rather
than proving its absence — run it for as long as you can stand before believing
anything about a week.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import shutil
import statistics
import subprocess
import time

import httpx

PROMPTS = [
    "What is the capital of France? One word.",
    "What is 2 + 2? Just the number.",
    "Name a primary colour.",
    "What is the chemical symbol for gold?",
    "How many continents are there?",
]


def rss_kb(pid: int) -> int | None:
    """Resident set size via `ps` — stdlib-only and works on macOS and Linux.

    psutil would be nicer and is a dependency this project does not have; the
    runtime is numpy-only on purpose (D6) and a soak script is a poor reason to
    change that.
    """
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) if out.stdout.strip() else None
    except (subprocess.SubprocessError, ValueError):
        return None


def fd_count(pid: int) -> int | None:
    """Open file descriptors. `lsof` is slow, which is why samples are minutes
    apart rather than seconds."""
    if not shutil.which("lsof"):
        return None
    try:
        out = subprocess.run(["lsof", "-p", str(pid)],
                             capture_output=True, text=True, timeout=30)
        return max(0, len(out.stdout.splitlines()) - 1)
    except subprocess.SubprocessError:
        return None


def find_pid(port: int) -> int | None:
    if not shutil.which("lsof"):
        return None
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                             capture_output=True, text=True, timeout=10)
        pids = [int(x) for x in out.stdout.split()]
        return pids[0] if pids else None
    except (subprocess.SubprocessError, ValueError):
        return None


class Window:
    """One sampling interval's worth of observations."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.latencies: list[float] = []
        self.ok = 0
        self.failed = 0
        self.limited = 0
        self.rss: int | None = None
        self.fds: int | None = None

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        return ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


async def one(client: httpx.AsyncClient, url: str, prompt: str, key: str) -> tuple[str, float]:
    """-> ("ok" | "limited" | "failed", milliseconds).

    A 429 is counted apart from a failure. The rate limiter refusing traffic is
    the gateway working; folding it into an error count would make a correctly
    behaving gateway look broken, which is what the first run of this script
    did.
    """
    headers = {"X-API-Key": key} if key else {}
    started = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 12},
            headers=headers,
        )
        ms = (time.perf_counter() - started) * 1000
        if r.status_code == 429:
            return "limited", ms
        return ("ok" if r.status_code == 200 else "failed"), ms
    except Exception:  # noqa: BLE001
        return "failed", (time.perf_counter() - started) * 1000


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--minutes", type=float, default=30.0)
    p.add_argument("--rps", type=float, default=2.0, help="requests per second")
    p.add_argument("--window-s", type=float, default=60.0, help="sampling interval")
    p.add_argument("--key", default="")
    p.add_argument("--pid", type=int, default=0, help="gateway pid (auto-detected if omitted)")
    # A leak is a *trend*, not a number: RSS legitimately rises early as pools
    # fill and numpy warms. The threshold applies to growth from the second
    # window onward, so startup is not mistaken for a leak.
    p.add_argument("--max-rss-growth-pct", type=float, default=25.0)
    p.add_argument("--max-rss-growth-mb", type=float, default=20.0,
                   help="both this and --max-rss-growth-pct must trip to fail")
    p.add_argument("--max-fd-growth", type=int, default=20)
    args = p.parse_args()
    url = args.url.rstrip("/")

    pid = args.pid or find_pid(int(url.rsplit(":", 1)[-1]))
    if pid:
        print(f"watching pid {pid}")
    else:
        print("could not find the gateway pid — memory and fd checks will be skipped")

    headers = {"X-API-Key": args.key} if args.key else {}

    async def read_metrics() -> dict | None:
        """The probe is subject to the same rate limit as the traffic, so it can
        be refused — the first run of this script crashed on exactly that."""
        async with httpx.AsyncClient(timeout=120) as probe:
            for _ in range(6):
                r = await probe.get(f"{url}/metrics", headers=headers)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    await asyncio.sleep(5)
                    continue
                return None
        return None

    start_metrics = await read_metrics()
    if start_metrics is None:
        print(f"cannot read {url}/metrics — wrong key, or rate limited for too long")
        return 2

    if args.rps * 60 > 0:
        print(f"note: {args.rps:g} rps is {args.rps * 60:.0f}/min. If that exceeds "
              "RATE_LIMIT_PER_MINUTE the excess is counted as 429, not failure.")

    deadline = time.monotonic() + args.minutes * 60
    interval = 1.0 / args.rps if args.rps > 0 else 0.5
    windows: list[Window] = []
    stop = asyncio.Event()

    async def sampler() -> None:
        idx = 0
        while not stop.is_set():
            w = Window(idx)
            windows.append(w)
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=args.window_s)
            if pid:
                w.rss = rss_kb(pid)
                w.fds = fd_count(pid)
            print(f"  [{idx:>3}] {w.ok:>5} ok  {w.limited:>4} 429  {w.failed:>3} failed   "
                  f"p50 {w.p50:>6.0f}ms  p95 {w.p95:>7.0f}ms   "
                  f"rss {(w.rss or 0)/1024:>7.1f}MB  fds {w.fds if w.fds is not None else '-'}")
            idx += 1

    print(f"\nsoaking {url} for {args.minutes:g} min at ~{args.rps:g} rps, "
          f"sampling every {args.window_s:g}s\n")
    sampler_task = asyncio.create_task(sampler())
    await asyncio.sleep(0)

    n = 0
    async with httpx.AsyncClient(timeout=120) as client:
        while time.monotonic() < deadline:
            outcome, ms = await one(client, url, PROMPTS[n % len(PROMPTS)], args.key)
            if windows:
                w = windows[-1]
                if outcome == "ok":
                    w.ok += 1
                    # Only successful requests time the gateway; a 429 is
                    # rejected in middleware and would drag the percentile down
                    # to something that says nothing about serving.
                    w.latencies.append(ms)
                elif outcome == "limited":
                    w.limited += 1
                else:
                    w.failed += 1
            n += 1
            await asyncio.sleep(interval)

    stop.set()
    await sampler_task

    end_metrics = await read_metrics()

    # ---- verdict --------------------------------------------------------
    print(f"\n  {n} requests over {args.minutes:g} min, {len(windows)} windows\n")
    print("  checks:")
    problems = 0
    measured = [w for w in windows if w.latencies]

    limited = sum(w.limited for w in measured)
    if limited:
        print(f"    note  {limited} request(s) were rate-limited (429) — the limiter "
              "working, not a failure")
    failed = sum(w.failed for w in measured)
    if failed:
        print(f"    FAIL  {failed} request(s) failed")
        problems += 1
    else:
        print(f"    ok    no request failed ({sum(w.ok for w in measured)} served)")

    # Memory: compare the last window against the *second*, so the startup
    # ramp (pools filling, numpy warming) is not counted as a leak.
    # Memory is reported as a *shape*, not a ratio, because the two failure
    # modes look completely different and only one of them is a leak:
    #
    #   a leak      RSS rises across every window and is still rising at the end
    #   a step      RSS jumps once and then plateaus — a buffer, an arena
    #               CPython never returns to the OS, an index materialising
    #
    # An 8-minute run of this gateway showed exactly the second: flat at 32 MB
    # for seven minutes, one jump to 66 MB, then flat again for as long as it
    # was watched. Reporting that as "+54%, FAIL" would be wrong, and reporting
    # it as "stable" — which an earlier version of this check did — hides a
    # doubling of the working set. So say which one it is.
    rss = [w.rss for w in measured if w.rss]
    if len(rss) >= 6:
        third = max(1, len(rss) // 3)
        early = statistics.median(rss[:third])
        late = statistics.median(rss[-third:])
        growth_pct = (late - early) / early * 100
        growth_mb = (late - early) / 1024
        # Still rising at the end? Compare the last third's first and second
        # halves — a plateau shows no movement there, a leak does.
        tail = rss[-third:]
        half = max(1, len(tail) // 2)
        still_rising = statistics.median(tail[half:]) > statistics.median(tail[:half]) * 1.02

        summary = (f"{early/1024:.1f} -> {late/1024:.1f} MB "
                   f"({growth_pct:+.1f}%, peak {max(rss)/1024:.1f} MB)")
        if still_rising and growth_pct > args.max_rss_growth_pct \
                and growth_mb > args.max_rss_growth_mb:
            print(f"    FAIL  RSS still climbing at the end — {summary}")
            problems += 1
        elif growth_pct > args.max_rss_growth_pct and growth_mb > args.max_rss_growth_mb:
            print(f"    WARN  RSS stepped up and plateaued — {summary}")
            print("          not a leak, but the working set is now the larger "
                  "number; size the host for the peak")
        else:
            print(f"    ok    RSS stable — {summary}")
    else:
        print(f"    skip  RSS needs >=6 windows to judge a trend (have {len(rss)})")

    fds = [w.fds for w in measured if w.fds is not None]
    if len(fds) >= 3:
        base, last = fds[1], fds[-1]
        if last - base > args.max_fd_growth:
            print(f"    FAIL  file descriptors grew {base} -> {last}")
            problems += 1
        else:
            print(f"    ok    file descriptors stable ({base} -> {last})")
    else:
        print("    skip  fds not sampled (lsof unavailable?)")

    # Latency drift: first third vs last third, so a single slow window does
    # not decide it.
    if len(measured) >= 6:
        third = len(measured) // 3
        early = statistics.mean(w.p95 for w in measured[:third])
        late = statistics.mean(w.p95 for w in measured[-third:])
        if early > 0 and late > early * 1.5:
            print(f"    FAIL  p95 drifted {early:.0f}ms -> {late:.0f}ms (>50%)")
            problems += 1
        else:
            print(f"    ok    p95 steady ({early:.0f}ms -> {late:.0f}ms)")
    else:
        print("    skip  too few windows to judge latency drift")

    # The identity loadtest.py checks, re-checked after a long run: does the
    # arithmetic still reconcile once the numbers are large?
    if end_metrics is None:
        print("    WARN  could not read /metrics at the end; skipping counter checks")
        end_metrics = {}
    served = sum((end_metrics.get("tiers") or {}).get("served", {}).values())
    total = end_metrics.get("requests")
    if total is None:
        print("    skip  no /metrics at the end")
        total = 0
    elif served == total:
        print(f"    ok    served counts still reconcile ({total} requests)")
    else:
        print(f"    FAIL  served {served} != requests {total}")
        problems += 1

    delta = total - start_metrics.get("requests", 0)
    # 429s never reach telemetry — they are rejected in middleware — so the
    # expected count is the served ones.
    expected = sum(w.ok for w in measured)
    if delta >= expected:
        print(f"    ok    telemetry counted every served request ({delta} >= {expected})")
    elif total:
        print(f"    FAIL  telemetry counted {delta} of {expected} served requests")
        problems += 1

    print(f"\n  {'PASS' if problems == 0 else f'{problems} problem(s)'}")
    print("\n  Note: this bounds the problem, it does not prove absence. A leak "
          f"smaller than\n  measurement noise is invisible over {args.minutes:g} minutes.\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
