#!/usr/bin/env python3
"""Sustained-load check against a running gateway.

    python3 scripts/loadtest.py --concurrency 16 --requests 200

Not a benchmark — the numbers depend entirely on the host and the models. It
exists to answer questions a smoke test cannot:

- does anything **fail** under sustained concurrency, or only get slower?
- do the **cost identities still reconcile** after hundreds of requests?
- does the **cache actually absorb** repeated traffic, or does the single-flight
  window let duplicates through?
- does **latency degrade gracefully** or fall off a cliff?

The default mix is deliberately cache-heavy, because that is what real traffic
looks like and it is the case where a concurrency bug would show up as
duplicate model calls rather than as an error.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import time

import httpx

# A small prompt pool: repeats are the point. With ~8 distinct prompts and
# hundreds of requests, almost everything after the first pass should be a
# cache hit — and if it is not, either the cache or single-flight is wrong.
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 + 2? Just the number.",
    "What is the chemical symbol for gold?",
    "Name a primary colour.",
    "What is the largest ocean on Earth?",
    "How many continents are there?",
    "What planet is known as the Red Planet?",
    "What is the freezing point of water in Celsius?",
]


async def one(client: httpx.AsyncClient, url: str, prompt: str, key: str) -> dict:
    headers = {"X-API-Key": key} if key else {}
    started = time.perf_counter()
    try:
        r = await client.post(
            f"{url}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 12},
            headers=headers,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "ms": elapsed, "tier": None}
        hlr = r.json()["x_hlr"]
        return {"ok": True, "status": 200, "ms": elapsed, "tier": hlr["tier"]}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": type(exc).__name__,
            "ms": (time.perf_counter() - started) * 1000,
            "tier": None,
        }


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--key", default="")
    args = p.parse_args()
    url = args.url.rstrip("/")

    print(f"\n{args.requests} requests, concurrency {args.concurrency}, "
          f"{len(PROMPTS)} distinct prompts\n")

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=300) as client:
        async def worker(i: int) -> None:
            async with sem:
                results.append(await one(client, url, PROMPTS[i % len(PROMPTS)], args.key))

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(args.requests)))
        wall = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = sorted(r["ms"] for r in ok)
    tiers = collections.Counter(r["tier"] for r in ok)

    def pct(q: float) -> float:
        return latencies[min(int(q * len(latencies)), len(latencies) - 1)] if latencies else 0.0

    print(f"  wall clock     {wall:.1f}s   ({len(results) / wall:.1f} req/s)")
    print(f"  succeeded      {len(ok)}/{len(results)}")
    if failed:
        by_status = dict(collections.Counter(r["status"] for r in failed))
        print(f"  FAILED         {len(failed)}  {by_status}")
    print(f"  latency        p50 {pct(0.50):.0f}ms   p95 {pct(0.95):.0f}ms   "
          f"p99 {pct(0.99):.0f}ms   max {latencies[-1]:.0f}ms" if latencies else "")
    print(f"  served by      {dict(tiers)}")

    cached = sum(v for k, v in tiers.items() if k and k.startswith("cache"))
    model_calls = len(ok) - cached
    print(f"  model calls    {model_calls} for {len(ok)} requests across "
          f"{len(PROMPTS)} distinct prompts")

    # ---- the checks that make this more than a benchmark ----------------
    print("\n  checks:")
    problems = 0

    if failed:
        print(f"    FAIL  {len(failed)} request(s) errored under load")
        problems += 1
    else:
        print("    ok    no request failed")

    # Every distinct prompt needs at least one real call; much more than that
    # means duplicates slipped past the cache or the single-flight window.
    if model_calls > len(PROMPTS) * 3:
        print(f"    WARN  {model_calls} model calls for {len(PROMPTS)} prompts — "
              "more duplication than expected")
        problems += 1
    else:
        print(f"    ok    model calls bounded ({model_calls} for {len(PROMPTS)} prompts)")

    try:
        headers = {"X-API-Key": args.key} if args.key else {}
        async with httpx.AsyncClient(timeout=30) as c:
            m = (await c.get(f"{url}/metrics", headers=headers)).json()
        served, cost, tok = m["tiers"]["served"], m["cost"], m["tokens"]

        if sum(served.values()) == m["requests"]:
            print(f"    ok    served counts sum to requests ({m['requests']})")
        else:
            print(f"    FAIL  served {sum(served.values())} != requests {m['requests']}")
            problems += 1

        expected = round(tok["baseline_in"] / 1e6 * 2.50 + tok["baseline_out"] / 1e6 * 10.00, 6)
        if abs(expected - cost["baseline_usd"]) < 1e-5:
            print(f"    ok    cost identities reconcile (saved {cost['saved_pct']}%)")
        else:
            print(f"    FAIL  baseline {cost['baseline_usd']} != recomputed {expected}")
            problems += 1
    except Exception as exc:  # noqa: BLE001
        print(f"    WARN  could not read /metrics: {exc}")

    print(f"\n  {'PASS' if problems == 0 else f'{problems} problem(s)'}\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
