"""Tier 1 — semantic cache.

Redis is the source of truth; the in-process numpy matrix is a **warm replica**
(invariant 7). The replica exists because brute-force cosine over an in-process
matrix is microseconds, while round-tripping vectors through Redis is not. It
re-syncs when `hlr:sem:version` moves. If the two ever disagree, Redis wins.

No vector index: Redis 7.2 here has no modules, and at this scale it does not
matter — 768-dim float32 is 3 KB/vector, so 10k entries is 31 MB and a 7.7M-MAC
matmul (RESEARCH.md §4). An index starts paying somewhere past 100k entries.

Serving a hit requires **all** of:
  1. cosine >= SEMANTIC_THRESHOLD
  2. every lexical guard in `guards.py` passes
  3. the stored entry's sampling params match the request
  4. the underlying exact-cache entry still exists

Any one failing is a miss. A miss costs ~120 ms; a false hit costs correctness.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.cache import guards
from app.cache.keys import CacheKey

logger = logging.getLogger("hlr.cache.semantic")

VEC_PREFIX = "hlr:sem:vec:"
IDS_KEY = "hlr:sem:ids"
VERSION_KEY = "hlr:sem:version"
LRU_KEY = "hlr:lru"

# nomic-embed-text expects a task prefix. For question-to-question matching the
# task is *symmetric*, so the same prefix goes on both the stored and the
# incoming side. Mixing `search_query:` with `search_document:` shifts the whole
# similarity distribution and silently invalidates the threshold.
QUERY_PREFIX = "search_query: "

DTYPE = np.float32


@dataclass
class SemanticHit:
    """A candidate that cleared every check."""

    exact_key: str
    prompt: str
    similarity: float
    entry_id: str


@dataclass
class SemanticMiss:
    """A miss, and why — so over-conservatism is measurable."""

    similarity: float | None = None
    guard_rejected: str | None = None
    detail: str = ""


class SemanticCache:
    """Embedding-similarity lookup with a warm numpy replica."""

    def __init__(
        self,
        redis: Redis | None,
        *,
        threshold: float = 0.95,
        max_chars: int = 500,
        max_entries: int = 10_000,
        enabled: bool = True,
    ) -> None:
        self._redis = redis
        self._threshold = threshold
        self._max_chars = max_chars
        self._max_entries = max_entries
        self._enabled = enabled and redis is not None

        # The warm replica. Never authoritative.
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        self._prompts: list[str] = []
        self._exact_keys: list[str] = []
        # Which tenant wrote each entry. Empty string under
        # CACHE_SCOPE=global, which is every entry written before
        # the setting existed — so an old index stays usable.
        self._scopes: list[str] = []
        self._version: int = -1

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        return len(self._ids)

    # ---- embedding helpers ----------------------------------------------

    @staticmethod
    def prefix(text: str) -> str:
        """Apply the task prefix. Both sides, always — see QUERY_PREFIX."""
        return f"{QUERY_PREFIX}{text}"

    @staticmethod
    def normalize(vector: list[float] | np.ndarray) -> np.ndarray:
        """Unit-normalize, so cosine similarity is a plain dot product.

        A zero vector would divide by zero; it is returned unchanged, and will
        score 0.0 against everything — which is the correct outcome for a
        degenerate embedding.
        """
        arr = np.asarray(vector, dtype=DTYPE)
        norm = float(np.linalg.norm(arr))
        return arr if norm == 0.0 else (arr / norm).astype(DTYPE)

    # ---- replica management ---------------------------------------------

    async def _remote_version(self) -> int:
        if self._redis is None:
            return -1
        try:
            raw = await self._redis.get(VERSION_KEY)
            return int(raw) if raw else 0
        except (RedisError, ValueError):
            return -1

    async def sync(self, *, force: bool = False) -> bool:
        """Refresh the replica from Redis if the version moved.

        Returns True if a reload happened. In single-worker operation the
        process is the only writer, so this is a no-op after startup; the
        version check is what makes multi-worker correct in principle.
        """
        if self._redis is None:
            return False
        remote = await self._remote_version()
        if not force and remote == self._version and self._matrix is not None:
            return False

        try:
            ids = await self._redis.zrange(IDS_KEY, 0, -1)
            if not ids:
                self._matrix, self._ids, self._prompts = None, [], []
                self._exact_keys, self._scopes = [], []
                self._version = remote
                return True

            pipe = self._redis.pipeline(transaction=False)
            for entry_id in ids:
                pipe.hgetall(f"{VEC_PREFIX}{entry_id}")
            rows = await pipe.execute()
        except RedisError as exc:
            logger.warning("semantic replica sync failed, keeping stale matrix: %s", exc)
            return False

        vectors, kept_ids, prompts, exact_keys, scopes = [], [], [], [], []
        for entry_id, row in zip(ids, rows, strict=False):
            if not row or not row.get("vec"):
                continue
            raw = row["vec"]
            blob = raw.encode("latin-1") if isinstance(raw, str) else raw
            vectors.append(np.frombuffer(blob, dtype=DTYPE))
            kept_ids.append(entry_id)
            prompts.append(row.get("prompt", ""))
            exact_keys.append(row.get("exact_key", ""))
            scopes.append(row.get("scope", ""))

        self._matrix = np.vstack(vectors) if vectors else None
        self._ids, self._prompts, self._exact_keys = kept_ids, prompts, exact_keys
        self._scopes = scopes
        self._version = remote
        logger.info("semantic replica synced: %d vector(s) at version %d", len(kept_ids), remote)
        return True

    # ---- lookup ----------------------------------------------------------

    async def lookup(
        self,
        *,
        embedding: list[float] | np.ndarray,
        prompt: str,
        key: CacheKey,
        is_multi_turn: bool,
        scope: str = "",
    ) -> SemanticHit | SemanticMiss:
        """Find an interchangeable earlier answer, or explain why not.

        `scope` isolates the search under `CACHE_SCOPE=tenant`. Scoping the
        *exact* key is not enough on its own: this index is searched by vector
        similarity, so without this a tenant's query would still match another
        tenant's entry and be served their answer.
        """
        if not self._enabled:
            return SemanticMiss(detail="semantic cache disabled")

        eligible = guards.check_eligible(
            prompt=prompt, is_multi_turn=is_multi_turn, max_chars=self._max_chars
        )
        if not eligible.passed:
            return SemanticMiss(guard_rejected=eligible.rejected_by, detail=eligible.detail)

        await self.sync()
        if self._matrix is None or not len(self._ids):
            return SemanticMiss(detail="semantic index empty")

        query = self.normalize(embedding)
        if query.shape[0] != self._matrix.shape[1]:
            # Dimension drift means EMBED_MODEL changed under a populated index.
            logger.warning(
                "embedding dim %d != index dim %d; treating as miss",
                query.shape[0], self._matrix.shape[1],
            )
            return SemanticMiss(detail="embedding dimension mismatch")

        # Brute force. Vectors are unit-normalized, so this dot product IS cosine.
        scores = self._matrix @ query

        if scope:
            # Mask out every entry belonging to someone else. -inf rather than
            # deletion so `best` still indexes the real arrays.
            mine = np.array([s == scope for s in self._scopes], dtype=bool)
            if not mine.any():
                return SemanticMiss(detail="no entries in this cache scope")
            scores = np.where(mine, scores, -np.inf)

        best = int(np.argmax(scores))
        similarity = float(scores[best])

        if similarity < self._threshold:
            return SemanticMiss(similarity=similarity, detail="below similarity threshold")

        stored_prompt = self._prompts[best]
        verdict = guards.check_pair(prompt, stored_prompt)
        if not verdict.passed:
            # High cosine, different question. This is the case the guards exist
            # for, and it is worth logging: a near-miss caught is a false hit
            # that did not happen.
            logger.info(
                "semantic near-miss rejected by %s at cos=%.4f: %r vs %r",
                verdict.rejected_by, similarity, prompt[:60], stored_prompt[:60],
            )
            return SemanticMiss(
                similarity=similarity, guard_rejected=verdict.rejected_by, detail=verdict.detail
            )

        return SemanticHit(
            exact_key=self._exact_keys[best],
            prompt=stored_prompt,
            similarity=similarity,
            entry_id=self._ids[best],
        )

    # ---- writes ----------------------------------------------------------

    async def store(
        self,
        *,
        embedding: list[float] | np.ndarray,
        prompt: str,
        key: CacheKey,
        is_multi_turn: bool = False,
        ttl: int = 3600,
        scope: str = "",
    ) -> bool:
        """Index a prompt so future rewordings can find it.

        Ineligible prompts are not stored — indexing a multi-turn or
        context-heavy prompt would create a future false-hit opportunity for no
        benefit, since it can never be looked up either.
        """
        if not self._enabled or self._redis is None:
            return False
        if not guards.check_eligible(
            prompt=prompt, is_multi_turn=is_multi_turn, max_chars=self._max_chars
        ).passed:
            return False

        vector = self.normalize(embedding)
        entry_id = uuid.uuid4().hex[:16]
        now = time.time()
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.hset(
                f"{VEC_PREFIX}{entry_id}",
                mapping={
                    "vec": vector.tobytes().decode("latin-1"),
                    "prompt": prompt,
                    "exact_key": key.digest,
                    "scope": scope,
                },
            )
            pipe.expire(f"{VEC_PREFIX}{entry_id}", ttl)
            pipe.zadd(IDS_KEY, {entry_id: now})
            pipe.zadd(LRU_KEY, {entry_id: now})
            pipe.incr(VERSION_KEY)
            await pipe.execute()
        except RedisError as exc:
            logger.warning("semantic store failed: %s", exc)
            return False

        await self._evict_if_needed()
        # Force a resync so this process sees its own write immediately.
        await self.sync(force=True)
        return True

    async def touch(self, entry_id: str) -> None:
        """Record a use, for LRU. Best-effort."""
        if self._redis is None:
            return
        try:
            await self._redis.zadd(LRU_KEY, {entry_id: time.time()})
        except RedisError:
            pass

    async def drop(self, entry_id: str) -> None:
        """Remove one entry — used when its exact-cache answer has expired,
        leaving a dangling vector that can only ever produce misses."""
        if self._redis is None:
            return
        try:
            pipe = self._redis.pipeline(transaction=False)
            pipe.delete(f"{VEC_PREFIX}{entry_id}")
            pipe.zrem(IDS_KEY, entry_id)
            pipe.zrem(LRU_KEY, entry_id)
            pipe.incr(VERSION_KEY)
            await pipe.execute()
        except RedisError:
            pass

    async def _evict_if_needed(self) -> int:
        """Trim to `max_entries`, least-recently-used first."""
        if self._redis is None:
            return 0
        try:
            count = await self._redis.zcard(IDS_KEY)
            if count <= self._max_entries:
                return 0
            victims = await self._redis.zrange(LRU_KEY, 0, count - self._max_entries - 1)
            for entry_id in victims:
                await self.drop(entry_id)
            return len(victims)
        except RedisError:
            return 0

    async def flush(self) -> int:
        """Drop the whole index. Scoped to semantic keys only."""
        if self._redis is None:
            return 0
        removed = 0
        try:
            async for k in self._redis.scan_iter(match=f"{VEC_PREFIX}*", count=500):
                removed += await self._redis.delete(k)
            await self._redis.delete(IDS_KEY, LRU_KEY)
            await self._redis.incr(VERSION_KEY)
        except RedisError as exc:
            logger.warning("semantic flush failed after %d keys: %s", removed, exc)
        self._matrix, self._ids, self._prompts, self._exact_keys = None, [], [], []
        return removed
