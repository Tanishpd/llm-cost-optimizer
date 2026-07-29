#!/usr/bin/env python3
"""Assert the shared state is actually shared, against real worker processes.

    python3 scripts/multiworker_check.py --url http://localhost:8000 --workers 4

M10 moved the rate limiter, the semantic index version and the single-flight
lease into Redis so multiple uvicorn workers would agree. It was verified once,
by hand, and never again — which for the class of bug involved is close to not
verifying it. A per-process fallback does not fail; it *works*, in each worker
separately, and the only symptom is a limit that quietly multiplies by the
worker count.

So the checks here are written to fail when state stops being shared, not when
the gateway stops working:

- **the rate limit is one window, not N.** Send more than the cap and count how
  many are admitted. Per-process state admits roughly `cap × workers`.
- **single flight collapses duplicates.** Fire identical requests
  concurrently; exactly one should reach a model.
- **the exact cache spans workers.** Populate through one connection, then read
  repeatedly — every worker must see it.

**No Ollama.** The frontier tier is the mock, which is pure Python and starts
anywhere, and requests are pinned to it with `model: "gateway:frontier"`. The
properties under test are about Redis and process boundaries; involving 2.6 GB
of local models would make this untestable in CI for no gain.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

import httpx


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"    {'ok  ' if ok else 'FAIL'}  {name}{f' — {detail}' if detail else ''}")
    return ok


async def _metrics(client: httpx.AsyncClient, url: str, key: str) -> dict | None:
    """Read /metrics, tolerating the rate limit it is here to test.

    The probe is subject to the same window as the traffic. `soak.py` learned
    this the hard way and crashed on a 429 body; there is no reason to learn it
    twice.
    """
    headers = {"X-API-Key": key} if key else {}
    for _ in range(8):
        r = await client.get(f"{url}/metrics", headers=headers)
        if r.status_code == 200:
            return r.json()
        if r.status_code != 429:
            return None
        await asyncio.sleep(4)
    return None


async def _post(client: httpx.AsyncClient, url: str, prompt: str, key: str,
                model: str = "gateway:frontier") -> httpx.Response:
    return await client.post(
        f"{url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 8},
        headers={"X-API-Key": key} if key else {},
    )


async def check_rate_limit(url: str, cap: int, workers: int, key: str) -> bool:
    """Send 3× the cap and count admissions.

    Shared: about `cap`. Per-process: about `cap × workers`, and the gap is
    the entire point of this check.
    """
    n = cap * 3
    async with httpx.AsyncClient(timeout=60) as c:
        rs = await asyncio.gather(*(
            _post(c, url, f"rate probe {uuid.uuid4().hex[:6]}", key) for _ in range(n)
        ), return_exceptions=True)

    allowed = sum(1 for r in rs if not isinstance(r, Exception) and r.status_code == 200)
    limited = sum(1 for r in rs if not isinstance(r, Exception) and r.status_code == 429)

    # Generous upper bound: the sliding window and request interleaving make an
    # exact count flaky. `cap * workers` is what a per-process limiter would
    # admit, so anything near it is the failure.
    ceiling = cap + max(2, cap // 2)
    ok = allowed <= ceiling and limited > 0
    return report(
        "rate limit is one shared window",
        ok,
        f"{allowed} admitted of {n} (cap {cap}, {workers} workers; "
        f"per-process would admit ~{cap * workers})",
    )


async def check_single_flight(url: str, key: str) -> bool:
    """Identical concurrent prompts must produce one model call."""
    prompt = f"single flight probe {uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(timeout=120) as c:
        before = await _metrics(c, url, key)
        rs = await asyncio.gather(*(_post(c, url, prompt, key) for _ in range(8)),
                                  return_exceptions=True)
        after = await _metrics(c, url, key)

    if before is None or after is None:
        return report("single flight collapses duplicates", False,
                      "could not read /metrics (rate limited for too long?)")

    served = sum(1 for r in rs if not isinstance(r, Exception) and r.status_code == 200)
    calls = (after["tiers"]["attempted"].get("frontier", 0)
             - before["tiers"]["attempted"].get("frontier", 0))
    # `1 <=`, not just `<= 2`: with Redis down the counters are empty and the
    # delta is 0, which sailed under a `<= 2` bound and reported success for a
    # gateway whose state was not shared at all. A vacuous pass is worse than a
    # failure — it is a failure you have been told not to worry about.
    ok = served >= 1 and 1 <= calls <= 2
    return report(
        "single flight collapses duplicates",
        ok,
        f"{served}/8 served by {calls} model call(s)"
        + ("" if calls else " — no telemetry, so nothing was actually verified"),
    )


async def check_shared_cache(url: str, key: str) -> bool:
    """A cached answer must be visible from every worker.

    Without shared caching, a request landing on a worker that did not produce
    the answer misses — and with N workers, N-1 of them do.
    """
    prompt = f"shared cache probe {uuid.uuid4().hex[:8]}"
    async with httpx.AsyncClient(timeout=120) as c:
        first = await _post(c, url, prompt, key)
        if first.status_code != 200:
            return report("exact cache spans workers", False,
                          f"seed request returned {first.status_code}")
        # New connections, so the load balancer spreads them across workers.
        tiers = []
        for _ in range(6):
            async with httpx.AsyncClient(timeout=60) as fresh:
                r = await _post(fresh, url, prompt, key)
                if r.status_code == 200:
                    tiers.append(r.json()["x_hlr"]["tier"])

    hits = sum(1 for t in tiers if t.startswith("cache"))
    ok = bool(tiers) and hits == len(tiers)
    return report("exact cache spans workers", ok,
                  f"{hits}/{len(tiers)} follow-ups were cache hits")


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rate-limit", type=int, default=10,
                   help="must match the gateway's RATE_LIMIT_PER_MINUTE")
    p.add_argument("--key", default="")
    args = p.parse_args()
    url = args.url.rstrip("/")

    async with httpx.AsyncClient(timeout=30) as c:
        try:
            health = (await c.get(f"{url}/healthz")).json()
        except Exception as exc:  # noqa: BLE001
            print(f"cannot reach {url}: {exc}")
            return 2

    print(f"\n{args.workers} workers at {url}\n")
    distributed = health.get("config", {}).get("rate_limit_distributed")
    print("  checks:")
    ok = report(
        "the limiter reports itself as distributed", bool(distributed),
        "false means Redis is unreachable and each worker has its own window",
    )

    # Order matters: the cache check seeds an entry the others would otherwise
    # hit, and the rate-limit check exhausts the window for a minute.
    ok &= await check_shared_cache(url, args.key)
    ok &= await check_single_flight(url, args.key)
    ok &= await check_rate_limit(url, args.rate_limit, args.workers, args.key)

    print(f"\n  {'PASS' if ok else 'FAILED'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
