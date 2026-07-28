"""Tier 0 — exact cache over Redis.

Every method is failure-tolerant by construction: Redis being unreachable
disables caching and the gateway keeps serving (invariant 4). A cache is an
optimization; it must never be able to take the request path down.

That means **no method here raises on a Redis error**. `get` returns None, `set`
returns False, and the caller records `redis_unavailable` in `degraded`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.cache.keys import CacheKey, redis_key

logger = logging.getLogger("hlr.cache.exact")

# Tiers whose answers cost real money, and are therefore worth keeping longer.
BILLED_TIERS = {"frontier"}


@dataclass
class CachedAnswer:
    """What gets stored. Deliberately self-describing: the entry records which
    tier produced it and what it cost, so a cache hit can report honest
    savings rather than guessing."""

    content: str
    model: str
    tier: str
    tokens_in: int
    tokens_out: int
    baseline_micro_usd: int
    finish_reason: str = "stop"
    created_at: float = 0.0
    # The params this answer was generated under. The semantic tier compares
    # these before serving a near-match; the exact tier gets it for free via
    # the key, but storing them makes an entry independently auditable.
    params: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> CachedAnswer | None:
        """Tolerates corrupt or stale-schema entries by returning None — a bad
        entry must be a miss, never an exception on the request path.

        "Corrupt" includes valid JSON of the wrong *type*: `"[]"` parses fine
        but is not a mapping, and an unguarded `.items()` on it raises
        AttributeError, which would escape a `(ValueError, TypeError)` handler.
        """
        try:
            data = json.loads(raw)
        except ValueError as exc:
            logger.warning("discarding unparseable cache entry: %s", exc)
            return None
        if not isinstance(data, dict):
            logger.warning("discarding cache entry of type %s, expected object",
                type(data).__name__)
            return None
        try:
            known = set(cls.__dataclass_fields__)
            return cls(**{k: v for k, v in data.items() if k in known})
        except TypeError as exc:  # missing required field
            logger.warning("discarding cache entry with bad schema: %s", exc)
            return None


class ExactCache:
    """Hash-keyed answer cache. All operations degrade to no-ops on failure."""

    def __init__(
        self,
        redis: Redis | None,
        *,
        ttl_local: int = 3600,
        ttl_frontier: int = 86400,
    ) -> None:
        self._redis = redis
        self._ttl_local = ttl_local
        self._ttl_frontier = ttl_frontier
        self.available = redis is not None

    def ttl_for(self, tier: str) -> int:
        """Frontier answers cost real money, so they earn more shelf space."""
        return self._ttl_frontier if tier in BILLED_TIERS else self._ttl_local

    async def get(self, key: CacheKey) -> CachedAnswer | None:
        """Look up an answer. Returns None on miss *or* on any Redis failure."""
        return await self.get_by_digest(key.digest)

    async def get_by_digest(self, digest: str) -> CachedAnswer | None:
        """Look up by raw digest.

        The semantic tier needs this: it finds a *vector* whose stored
        `exact_key` points at an answer, and must fetch that answer without
        reconstructing the original request. A None here means the vector is
        dangling — its answer has expired — which the caller treats as a miss
        and cleans up.
        """
        if self._redis is None:
            return None
        try:
            raw = await self._redis.get(redis_key(digest))
        except RedisError as exc:
            logger.warning("exact cache read failed, treating as miss: %s", exc)
            self.available = False
            return None
        if raw is None:
            return None
        return CachedAnswer.from_json(raw)

    async def set(self, key: CacheKey, answer: CachedAnswer) -> bool:
        """Store an answer. Returns False on any failure — a write that didn't
        happen is a future miss, not an error worth surfacing to the caller."""
        if self._redis is None:
            return False
        try:
            await self._redis.set(
                redis_key(key.digest), answer.to_json(), ex=self.ttl_for(answer.tier)
            )
            return True
        except RedisError as exc:
            logger.warning("exact cache write failed: %s", exc)
            self.available = False
            return False

    async def delete(self, key: CacheKey) -> bool:
        if self._redis is None:
            return False
        try:
            return bool(await self._redis.delete(redis_key(key.digest)))
        except RedisError:
            return False

    async def flush(self) -> int:
        """Drop every exact-cache entry. Dev convenience behind DELETE /cache.

        Uses SCAN rather than KEYS: KEYS blocks the Redis event loop for the
        whole keyspace, which is fine at demo scale and a production incident
        anywhere else. Scoped to the `hlr:exact:` prefix so it cannot touch
        anything else in the database.
        """
        if self._redis is None:
            return 0
        removed = 0
        try:
            async for k in self._redis.scan_iter(match="hlr:exact:*", count=500):
                removed += await self._redis.delete(k)
        except RedisError as exc:
            logger.warning("exact cache flush failed after %d keys: %s", removed, exc)
        return removed

    async def size(self) -> int:
        if self._redis is None:
            return 0
        try:
            return sum([1 async for _ in self._redis.scan_iter(match="hlr:exact:*", count=500)])
        except RedisError:
            return 0

    async def ping(self) -> tuple[bool, str]:
        """Never raises — /healthz reports, it does not explode."""
        if self._redis is None:
            return False, "not configured"
        try:
            await self._redis.ping()
            self.available = True
            return True, "reachable"
        except RedisError as exc:
            self.available = False
            return False, f"{type(exc).__name__}: {exc}"
