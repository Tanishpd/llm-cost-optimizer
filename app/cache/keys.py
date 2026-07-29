"""Cache-key construction. Pure functions, no Redis.

The key decides what counts as "the same request". Getting it wrong in either
direction is bad, but not symmetrically:

- Too *loose* → a wrong answer served instantly and for free, with no signal.
- Too *tight*  → a cache miss, costing one model call.

So this module is deliberately conservative (D15).

**Sampling parameters are part of the key.** The same text at `temperature=0`
and `temperature=1.0` are different questions — one asks for the model's best
guess, the other for a sample. Serving one for the other is a correctness bug,
not a cache optimization (invariant 5).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

# Bumping this invalidates every existing entry. Change it whenever
# normalization or the key composition changes — otherwise old entries would be
# served under a new scheme, which is exactly the silent-wrong-answer case.
KEY_SCHEMA_VERSION = "v1"

_WHITESPACE = re.compile(r"\s+")
_DIGEST_BYTES = 16  # 128-bit; collision risk is negligible at any cache size here


def normalize_prompt(text: str) -> str:
    """Canonical form for hashing.

    Deliberately minimal: Unicode NFC, collapse internal whitespace runs, strip
    the ends. That is all.

    **Not** lowercased and **not** punctuation-stripped, though both would raise
    the hit rate. Case is semantic in code (`myVar` vs `myvar`) and in
    identifiers, and punctuation carries meaning (`Let's eat, Grandma`). The
    semantic tier exists to catch benign rewordings; the exact tier should only
    catch things that are genuinely identical.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _hash(*parts: str) -> str:
    """blake2b over NUL-joined parts.

    NUL is the separator because it cannot appear in the inputs, so
    `("ab", "c")` and `("a", "bc")` cannot collide — a real hazard with a
    printable separator.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update("\x00".join(parts).encode("utf-8"))
    return h.hexdigest()


def system_prompt_hash(system_prompt: str) -> str:
    """Hashed rather than inlined: system prompts can be long, and the key
    should stay a fixed size."""
    return _hash(normalize_prompt(system_prompt)) if system_prompt else ""


@dataclass(frozen=True)
class CacheKey:
    """A cache key plus the components that produced it.

    Carrying the components is not decoration — the semantic tier must verify
    that a candidate near-match shares the *same* sampling params before serving
    it, and telemetry needs them for debugging a cache that "should" have hit.
    """

    digest: str
    prompt: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    system_hash: str

    def params_match(self, other: CacheKey) -> bool:
        """Do these two requests want interchangeable answers?

        Used by the semantic tier: high cosine says the *questions* match, this
        says the *answers* would be interchangeable. Both are required.
        """
        return (
            self.model == other.model
            and self.temperature == other.temperature
            and self.top_p == other.top_p
            and self.max_tokens == other.max_tokens
            and self.system_hash == other.system_hash
        )


def build_key(
    *,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_prompt: str = "",
    scope: str = "",
) -> CacheKey:
    """Compose the exact-cache key.

    `model` is the *tier's* model, so an answer from the 0.5B is never served
    for a request that would have gone to the 3B.

    `scope` is the tenant under `CACHE_SCOPE=tenant` and empty under `global`.
    Empty is the default and hashes to the same digests as before this
    parameter existed, so turning isolation *off* again does not orphan a
    cache.

    Floats are formatted with `repr` so `0.0` and `-0.0` and `0` all render
    identically — `str(float)` is stable in Python 3, but being explicit here
    prevents a future refactor introducing drift.
    """
    normalized = normalize_prompt(prompt)
    sys_hash = system_prompt_hash(system_prompt)
    digest = _hash(
        KEY_SCHEMA_VERSION,
        # Empty under CACHE_SCOPE=global, which is the default and keeps every
        # existing key valid. Under CACHE_SCOPE=tenant this is the tenant name,
        # so a hit can only ever be that tenant's own earlier request.
        scope,
        normalized,
        model,
        repr(float(temperature)),
        repr(float(top_p)),
        str(int(max_tokens)),
        sys_hash,
    )
    return CacheKey(
        digest=digest,
        prompt=normalized,
        model=model,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
        system_hash=sys_hash,
    )


def redis_key(digest: str) -> str:
    """Namespaced key. All keys live under `hlr:` (schema table in PLAN.md)."""
    return f"hlr:exact:{digest}"
