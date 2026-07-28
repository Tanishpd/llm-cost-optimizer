"""Tests for app/cache/semantic.py.

Uses hand-built vectors rather than a real embedder, so the similarity values
are exact and the tests assert *policy* rather than an embedding model's
behaviour. Integration tests below use the real model.
"""

from __future__ import annotations

import numpy as np
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.cache.guards import ENTITY_MISMATCH, MULTI_TURN, NUMBER_MISMATCH, TOO_LONG
from app.cache.keys import build_key
from app.cache.semantic import QUERY_PREFIX, SemanticCache, SemanticHit, SemanticMiss

BASE = dict(model="qwen2.5:3b", temperature=0.0, top_p=1.0, max_tokens=512)
DIM = 8


def vec(*values: float) -> list[float]:
    """A DIM-length vector from the given leading components."""
    out = [0.0] * DIM
    for i, v in enumerate(values):
        out[i] = v
    return out


class FakeRedis:
    """Hash + zset + counter subset, enough for the replica protocol."""

    def __init__(self, *, fail_on: set[str] | None = None):
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.strings: dict[str, str] = {}
        self.fail_on = fail_on or set()

    def _check(self, op):
        if op in self.fail_on:
            raise RedisConnectionError(f"simulated {op} failure")

    def pipeline(self, transaction=False):
        return FakePipeline(self)

    async def get(self, key):
        self._check("get")
        return self.strings.get(key)

    async def incr(self, key):
        self._check("incr")
        self.strings[key] = str(int(self.strings.get(key, 0)) + 1)
        return int(self.strings[key])

    async def zrange(self, key, start, end):
        self._check("zrange")
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        ids = [k for k, _ in items]
        return ids[start:] if end == -1 else ids[start : end + 1]

    async def zcard(self, key):
        self._check("zcard")
        return len(self.zsets.get(key, {}))

    async def zadd(self, key, mapping):
        self._check("zadd")
        self.zsets.setdefault(key, {}).update(mapping)
        return len(mapping)

    async def zrem(self, key, *members):
        for m in members:
            self.zsets.get(key, {}).pop(m, None)
        return len(members)

    async def hgetall(self, key):
        self._check("hgetall")
        return dict(self.hashes.get(key, {}))

    async def delete(self, *keys):
        n = 0
        for k in keys:
            if self.hashes.pop(k, None) is not None:
                n += 1
            if self.zsets.pop(k, None) is not None:
                n += 1
        return n

    async def scan_iter(self, match=None, count=None):
        prefix = (match or "").rstrip("*")
        for k in list(self.hashes):
            if k.startswith(prefix):
                yield k


class FakePipeline:
    def __init__(self, r: FakeRedis):
        self.r, self.ops = r, []

    def hset(self, key, mapping=None):
        self.ops.append(("hset", key, mapping)); return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl)); return self

    def zadd(self, key, mapping):
        self.ops.append(("zadd", key, mapping)); return self

    def incr(self, key):
        self.ops.append(("incr", key)); return self

    def delete(self, key):
        self.ops.append(("delete", key)); return self

    def zrem(self, key, member):
        self.ops.append(("zrem", key, member)); return self

    def hgetall(self, key):
        self.ops.append(("hgetall", key)); return self

    async def execute(self):
        self.r._check("execute")
        out = []
        for op in self.ops:
            if op[0] == "hset":
                self.r.hashes.setdefault(op[1], {}).update(op[2])
                out.append(True)
            elif op[0] == "zadd":
                self.r.zsets.setdefault(op[1], {}).update(op[2])
                out.append(1)
            elif op[0] == "incr":
                await self.r.incr(op[1])
                out.append(1)
            elif op[0] == "delete":
                self.r.hashes.pop(op[1], None)
                out.append(1)
            elif op[0] == "zrem":
                self.r.zsets.get(op[1], {}).pop(op[2], None)
                out.append(1)
            elif op[0] == "hgetall":
                out.append(dict(self.r.hashes.get(op[1], {})))
            else:
                out.append(True)
        return out


@pytest.fixture
def cache():
    return SemanticCache(FakeRedis(), threshold=0.95, max_chars=500, max_entries=100)


async def store(cache, prompt, embedding, **key_overrides):
    key = build_key(prompt=prompt, **{**BASE, **key_overrides})
    return await cache.store(embedding=embedding, prompt=prompt, key=key)


async def lookup(cache, prompt, embedding, *, is_multi_turn=False, **key_overrides):
    key = build_key(prompt=prompt, **{**BASE, **key_overrides})
    return await cache.lookup(
        embedding=embedding, prompt=prompt, key=key, is_multi_turn=is_multi_turn
    )


# --------------------------------------------------------------------------
# normalization and the task prefix
# --------------------------------------------------------------------------


def test_vectors_are_unit_normalized_so_dot_is_cosine():
    v = SemanticCache.normalize([3.0, 4.0])
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)


def test_zero_vector_does_not_divide_by_zero():
    """A degenerate embedding should score 0 against everything, not crash."""
    v = SemanticCache.normalize([0.0, 0.0])
    assert not np.isnan(v).any()


def test_task_prefix_is_applied():
    """nomic needs a task prefix, and the SAME one on both sides — mixing
    search_query with search_document shifts the whole distribution."""
    assert SemanticCache.prefix("hello") == f"{QUERY_PREFIX}hello"


# --------------------------------------------------------------------------
# hits and misses
# --------------------------------------------------------------------------


async def test_identical_vector_hits(cache):
    await store(cache, "What is the capital of France?", vec(1.0))
    result = await lookup(cache, "What's the capital city of France?", vec(1.0))
    assert isinstance(result, SemanticHit)
    assert result.similarity == pytest.approx(1.0)


async def test_dissimilar_vector_misses(cache):
    await store(cache, "capital of France", vec(1.0))
    result = await lookup(cache, "how to bake bread", vec(0.0, 1.0))
    assert isinstance(result, SemanticMiss)
    assert result.similarity == pytest.approx(0.0, abs=1e-6)


async def test_just_below_threshold_misses(cache):
    """0.94 vs a 0.95 threshold — the conservative default doing its job."""
    await store(cache, "capital of France", vec(1.0))
    near = SemanticCache.normalize(vec(0.94, 0.341))  # cos ~= 0.94
    result = await lookup(cache, "capital of France please", near)
    assert isinstance(result, SemanticMiss)
    assert result.similarity < 0.95


async def test_empty_index_misses(cache):
    result = await lookup(cache, "anything", vec(1.0))
    assert isinstance(result, SemanticMiss)
    assert "empty" in result.detail


async def test_best_match_wins(cache):
    await store(cache, "capital of France", vec(1.0))
    await store(cache, "capital of France indeed", vec(0.99, 0.141))
    result = await lookup(cache, "capital of France?", vec(1.0))
    assert isinstance(result, SemanticHit)
    assert result.similarity == pytest.approx(1.0)


# --------------------------------------------------------------------------
# guards fire even at cosine 1.0 — this is the whole point of the tier
# --------------------------------------------------------------------------


async def test_entity_guard_rejects_an_identical_vector(cache):
    """Even at perfect similarity, Australia is not Austria."""
    await store(cache, "What is the capital of Austria?", vec(1.0))
    result = await lookup(cache, "What is the capital of Australia?", vec(1.0))
    assert isinstance(result, SemanticMiss)
    assert result.guard_rejected == ENTITY_MISMATCH
    assert result.similarity == pytest.approx(1.0)  # the embedder was fooled; the guard was not


async def test_number_guard_rejects_an_identical_vector(cache):
    await store(cache, "convert 50 USD to EUR", vec(1.0))
    result = await lookup(cache, "convert 70 USD to EUR", vec(1.0))
    assert result.guard_rejected == NUMBER_MISMATCH


async def test_multi_turn_is_rejected_before_embedding_is_used(cache):
    await store(cache, "What is the capital of France?", vec(1.0))
    result = await lookup(cache, "What about it?", vec(1.0), is_multi_turn=True)
    assert result.guard_rejected == MULTI_TURN


async def test_long_prompt_is_rejected(cache):
    await store(cache, "short", vec(1.0))
    result = await lookup(cache, "x" * 600, vec(1.0))
    assert result.guard_rejected == TOO_LONG


async def test_ineligible_prompts_are_not_indexed(cache):
    """Indexing something that can never be looked up only creates future
    false-hit opportunities."""
    assert await cache.store(
        embedding=vec(1.0), prompt="x" * 600,
        key=build_key(prompt="x" * 600, **BASE), is_multi_turn=False,
    ) is False
    assert cache.size == 0


# --------------------------------------------------------------------------
# the replica — Redis is the source of truth
# --------------------------------------------------------------------------


async def test_store_makes_the_entry_immediately_visible(cache):
    await store(cache, "a prompt", vec(1.0))
    assert cache.size == 1


async def test_replica_reloads_when_the_version_moves(cache):
    await store(cache, "first", vec(1.0))
    assert cache.size == 1
    # Another worker writes: bump the version behind our back.
    await cache._redis.incr("hlr:sem:version")
    assert await cache.sync() is True


async def test_sync_is_a_noop_when_the_version_is_unchanged(cache):
    await store(cache, "first", vec(1.0))
    assert await cache.sync() is False


async def test_stale_matrix_is_kept_when_redis_read_fails(cache):
    """A sync failure must degrade to a stale replica, not an empty one —
    serving from slightly-stale data beats losing the cache entirely."""
    await store(cache, "first", vec(1.0))
    cache._redis.fail_on.add("zrange")
    cache._version = -99  # force a resync attempt
    assert await cache.sync() is False
    assert cache.size == 1


async def test_dimension_mismatch_is_a_miss_not_a_crash(cache):
    """EMBED_MODEL changed under a populated index."""
    await store(cache, "prompt", vec(1.0))
    result = await lookup(cache, "prompt", [1.0, 0.0])  # 2-dim vs 8-dim index
    assert isinstance(result, SemanticMiss)
    assert "dimension" in result.detail


# --------------------------------------------------------------------------
# eviction and removal
# --------------------------------------------------------------------------


async def test_drop_removes_an_entry(cache):
    await store(cache, "prompt", vec(1.0))
    entry_id = cache._ids[0]
    await cache.drop(entry_id)
    await cache.sync(force=True)
    assert cache.size == 0


async def test_lru_eviction_trims_to_the_cap():
    c = SemanticCache(FakeRedis(), threshold=0.95, max_entries=3)
    for i in range(5):
        await store(c, f"prompt number {i}", vec(1.0, float(i)))
    await c.sync(force=True)
    assert c.size <= 3


async def test_flush_clears_the_index(cache):
    await store(cache, "prompt", vec(1.0))
    await cache.flush()
    assert cache.size == 0


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------


async def test_disabled_cache_always_misses():
    c = SemanticCache(FakeRedis(), enabled=False)
    assert c.enabled is False
    result = await lookup(c, "q", vec(1.0))
    assert isinstance(result, SemanticMiss)
    assert "disabled" in result.detail


async def test_no_redis_disables_the_tier():
    c = SemanticCache(None)
    assert c.enabled is False
    assert await c.store(embedding=vec(1.0), prompt="q", key=build_key(prompt="q", **BASE)) is False
    assert await c.flush() == 0


async def test_store_failure_returns_false_not_an_exception():
    c = SemanticCache(FakeRedis(fail_on={"execute"}))
    assert await store(c, "q", vec(1.0)) is False


# --------------------------------------------------------------------------
# integration — the real embedder
# --------------------------------------------------------------------------


@pytest.mark.integration
async def test_real_embeddings_separate_paraphrase_from_near_miss():
    """The threshold is an assertion about *this* embedder. Measure it rather
    than trusting 0.95 — RESEARCH.md wants this probe, and preflight runs a
    larger version of it."""
    import httpx

    from app.providers.ollama import OllamaProvider

    pairs_same = [
        ("What is the capital of France?", "What's the capital city of France?"),
        ("How do I reverse a list in Python?", "How can I reverse a list in Python?"),
    ]
    pairs_different = [
        ("What is the capital of Australia?", "What is the capital of Austria?"),
        ("convert 50 USD to EUR", "convert 70 USD to EUR"),
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        provider = OllamaProvider(client)
        texts = [SemanticCache.prefix(t) for pair in pairs_same + pairs_different for t in pair]
        vectors = await provider.embed(model="nomic-embed-text", texts=texts)

    normed = [SemanticCache.normalize(v) for v in vectors]
    sims = [float(normed[i] @ normed[i + 1]) for i in range(0, len(normed), 2)]
    same, different = sims[: len(pairs_same)], sims[len(pairs_same) :]

    # Measured with nomic-embed-text on 2026-07-28: entity and number
    # near-misses land around 0.79-0.80, comfortably under the 0.95 threshold,
    # while the *negation* pair scores 0.9831 — above it. So cosine alone would
    # catch the first two but NOT negation, which is the guard doing the real
    # work here. Asserted loosely because these are model-dependent.
    assert all(s > 0.9 for s in same), f"paraphrases scored too low: {same}"
    assert all(s > 0.7 for s in different), f"near-misses implausibly low: {different}"

    # The point of the guards: at least one near-miss clears the threshold and
    # is stopped by lexical checks rather than by similarity.
    from app.cache.guards import check_pair

    above = [p for p, s in zip(pairs_different, different, strict=True) if s >= 0.95]
    for a, b in above:
        assert check_pair(a, b).passed is False, f"guards would have served {a!r} for {b!r}"
