#!/usr/bin/env python3
"""Walk every tier of the cascade and show what each one cost.

    python3 scripts/demo.py                 # against a running gateway
    python3 scripts/demo.py --reset         # clear caches/metrics first
    python3 scripts/demo.py --url http://…  # somewhere else

Needs the gateway on :8000 and the mock frontier on :8001.

Each step states what it is *trying* to demonstrate and then reports what
actually happened. Where the two differ the script says so rather than
narrating a success — a demo that cannot fail is not evidence of anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

GATEWAY = "http://localhost:8000"

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"
)
if not sys.stdout.isatty():
    GREEN = YELLOW = RED = DIM = BOLD = RESET = ""

TIER_LABEL = {
    "cache_exact": "exact cache",
    "cache_semantic": "semantic cache",
    "local_small": "qwen2.5:0.5b",
    "local_large": "qwen2.5:3b",
    "frontier": "frontier API",
}


def request(url: str, payload: dict | None = None, method: str | None = None) -> dict:
    """A body implies POST unless the caller says otherwise — urllib will
    happily send a GET with a body, and the server answers 405."""
    data = json.dumps(payload).encode() if payload is not None else None
    if method is None:
        method = "POST" if data is not None else "GET"
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def ask(url: str, prompt: str, *, max_tokens: int = 40, model: str = "gateway") -> dict:
    return request(
        f"{url}/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": prompt}],
         "max_tokens": max_tokens},
    )


def money(usd: float) -> str:
    return f"${usd:.6f}"


def step(n: int, title: str, expect: str) -> None:
    print(f"\n{BOLD}{n}. {title}{RESET}")
    print(f"   {DIM}expecting: {expect}{RESET}")


def outcome(body: dict, want_tier: str | None) -> bool:
    """Print what happened and whether it matched. Returns True on a match."""
    h = body["x_hlr"]
    tier = h["tier"]
    ok = want_tier is None or tier == want_tier
    mark = f"{GREEN}✓{RESET}" if ok else f"{YELLOW}!{RESET}"

    path = " → ".join(h["path"]) if len(h.get("path") or []) > 1 else ""
    print(f"   {mark} served by {BOLD}{TIER_LABEL.get(tier, tier)}{RESET}"
          f"{f'   path: {path}' if path else ''}"
          f"   {h['latency_ms']['total']:.0f}ms"
          f"   cost {money(h['cost_usd'])}   saved {money(h['cost_saved_usd'])}")

    for g in h.get("gates") or []:
        verdict = "escalate" if g["verdict"] == "escalate" else "accept"
        colour = YELLOW if verdict == "escalate" else GREEN
        print(f"     {DIM}gate {g['gate']:<5}{RESET} confidence {g['confidence']:.4f} "
              f"vs {g['threshold']} → {colour}{verdict}{RESET}")

    cache = h.get("cache") or {}
    if cache.get("similarity") is not None:
        note = f"   guard: {cache['guard_rejected']}" if cache.get("guard_rejected") else ""
        print(f"     {DIM}cosine {cache['similarity']:.4f}{note}{RESET}")

    print(f"     {DIM}{h['reason'][:150]}{RESET}")
    if not ok:
        print(f"   {YELLOW}note: expected {want_tier}, got {tier}. Model behaviour varies; "
              f"the gate decides at run time.{RESET}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=GATEWAY)
    parser.add_argument("--reset", action="store_true", help="clear caches and metrics first")
    args = parser.parse_args()
    url = args.url.rstrip("/")

    try:
        health = request(f"{url}/healthz")
    except urllib.error.URLError as exc:
        print(f"{RED}Cannot reach the gateway at {url}: {exc}{RESET}")
        print("  uvicorn app.main:app --port 8000 &")
        print("  uvicorn mock_frontier.main:app --port 8001 &")
        return 1

    deps = health["dependencies"]
    print(f"{BOLD}LLM Cost Optimizer — cascade demo{RESET}")
    print(f"{DIM}gateway {url}   status: {health['status']}   "
          f"ollama={deps['ollama']['ok']} redis={deps['redis']['ok']} "
          f"frontier={deps['frontier']['ok']}   "
          f"confidence mode: {health['config']['confidence_mode']}{RESET}")

    if not deps["frontier"]["ok"]:
        print(f"{YELLOW}   the mock frontier is down — step 5 will escalate and fail over "
              f"to the best local answer instead.{RESET}")

    if args.reset:
        request(f"{url}/cache", method="DELETE")
        request(f"{url}/metrics", method="DELETE")
        print(f"{DIM}   caches and metrics cleared{RESET}")

    matched = 0
    total = 0

    step(1, "A short factual question",
         "the router enters at the 0.5B, and its own logprobs clear the gate")
    total += 1
    matched += outcome(ask(url, "What is the capital of France? Answer in one word."),
                       "local_small")

    step(2, "The identical question again",
         "exact cache — a hash lookup, no model at all")
    total += 1
    matched += outcome(ask(url, "What is the capital of France? Answer in one word."),
                       "cache_exact")

    step(3, "The same question, reworded",
         "semantic cache — the embedding recognises it despite different words")
    total += 1
    matched += outcome(ask(url, "What's the capital city of France? One word."),
                       "cache_semantic")

    step(4, "A question the 0.5B should hesitate on",
         "gate 1 escalates on low confidence and the 3B answers")
    total += 1
    matched += outcome(ask(url, "Explain why a rainbow has the colour order it does.",
                           max_tokens=60), "local_large")

    step(5, "A prompt with several independent hardness signals",
         "the router skips both local tiers — this is the only path that costs money")
    total += 1
    matched += outcome(ask(url,
        "Prove that every finite integral domain is a field. Walk me through it step by "
        "step, explain why each lemma holds, and discuss the trade-offs of the alternative "
        "proofs. What if the ring were infinite? How would you generalize?\n```\nR = Z/pZ\n```",
        max_tokens=40), "frontier")

    step(6, "A near-miss the embedder cannot tell apart",
         "cosine clears 0.95, and only the negation guard prevents serving the opposite answer")
    ask(url, "Is this API safe to use?", max_tokens=20)
    total += 1
    body = ask(url, "Is this API not safe to use?", max_tokens=20)
    guard = (body["x_hlr"].get("cache") or {}).get("guard_rejected")
    served_from_cache = body["x_hlr"]["tier"].startswith("cache")
    ok = guard == "negation_mismatch" or not served_from_cache
    matched += ok
    outcome(body, None)
    if guard:
        print(f"   {GREEN}✓{RESET} guard {BOLD}{guard}{RESET} rejected the near-match")
    elif not served_from_cache:
        print(f"   {GREEN}✓{RESET} not served from cache")
    else:
        print(f"   {RED}✗ served a cached answer for a negated question{RESET}")

    # ---- the summary -----------------------------------------------------
    m = request(f"{url}/metrics")
    cost, served, gates = m["cost"], m["tiers"]["served"], m["gates"]

    print(f"\n{BOLD}{'─' * 62}{RESET}")
    print(f"{BOLD}Totals across {m['requests']} request(s){RESET}\n")

    width = max(len(v) for v in TIER_LABEL.values())
    for tier, label in TIER_LABEL.items():
        n = served.get(tier, 0)
        if n:
            bar = "█" * max(1, round(n / max(m["requests"], 1) * 24))
            print(f"  {label:<{width}}  {n:>3}  {DIM}{bar}{RESET}")

    print(f"\n  spent    {money(cost['actual_usd'])}")
    print(f"  baseline {money(cost['baseline_usd'])}   {DIM}(every request at frontier "
          f"prices, cache hits included){RESET}")
    print(f"  {GREEN}{BOLD}saved    {money(cost['saved_usd'])}  "
          f"({cost['saved_pct']}%){RESET}")

    for gate in ("small", "large"):
        g = gates[gate]
        if g["evaluated"]:
            print(f"\n  gate {gate}: {g['escalated']}/{g['evaluated']} escalated "
                  f"({g['escalation_rate']:.0%})")

    rejections = (m.get("cache") or {}).get("guard_rejections") or {}
    if rejections:
        print(f"\n  guard rejections: {rejections}   "
              f"{DIM}(false hits that did not happen){RESET}")

    print(f"\n{DIM}  Local tiers bill $0 and the baseline prices every request at frontier")
    print("  rates — both assumptions are documented in docs/DECISIONS.md D13.")
    print(f"  Live view: {url}/dashboard{RESET}")

    if matched < total:
        print(f"\n{YELLOW}  {total - matched} of {total} steps did not land on the expected "
              f"tier. Confidence is measured at run time, so this varies.{RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
