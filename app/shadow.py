"""Shadow sampling: check the gate's decisions against the answer it declined.

This project's entire claim is that a model's own logprobs can tell you when its
answer is good enough. That was validated once, offline, on 67 auto-gradable
prompts — and gate `large` failed (RESEARCH.md §3.4). Everything since has run
on the assumption that the raw thresholds are approximately right for real
traffic, with no way to notice if they stop being.

Shadow mode closes that. On a sampled fraction of requests the gate *accepted*,
the frontier is called anyway, out of band, and the two answers are compared.
The caller never waits for it and never sees it; the comparison lands in the
audit trail and in `/metrics`.

**It costs money on purpose.** A shadow call is a real frontier call for an
answer nobody reads. `SHADOW_SAMPLE_RATE=0.01` on traffic the gate accepts is
roughly 1% of what running everything through the frontier would cost, and buys
a continuously updated estimate of how often the cheap tier was wrong. Off by
default (`0.0`) — spending money to measure must be something you switch on
deliberately, exactly as `MAX_DAILY_SPEND_USD` is something you switch off
deliberately.

**It respects the budget.** A shadow call is billed like any other, counts
against tenant budgets and the gateway ceiling, and is skipped entirely when
either is exhausted. Measurement that could exhaust the budget it is measuring
would be a poor trade.

**It needs a real frontier.** The mock's envelope is faithful and its content is
canned (D9), so every comparison against it diverges and the rate reads 100% —
a misconfiguration that looks exactly like a catastrophic gate. Startup warns
loudly when shadow sampling is on and the frontier looks like the mock.

**What it can and cannot tell you.** It samples only requests the gate
*accepted*, so it measures the false-cheap rate — answers that were served
locally and should not have been. It says nothing about false-expensive
(escalations that were unnecessary), because those already have the frontier's
answer and no cheaper one to compare against. Both matter; this measures the one
that costs quality rather than money.

Similarity here is lexical, not semantic — a token-overlap ratio, not a judge
model. That is deliberate: a judge would double the cost and introduce a second
model's opinion into a number meant to be a tripwire. A sustained *drop* in this
ratio is the signal; its absolute value is not a quality score and must not be
read as one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hlr.shadow")

_WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class ShadowStats:
    """Running totals, reported by `/metrics` and `/healthz`."""

    sampled: int = 0
    completed: int = 0
    failed: int = 0
    skipped_budget: int = 0
    divergent: int = 0
    similarity_sum: float = 0.0
    cost_micro_usd: int = 0

    @property
    def mean_similarity(self) -> float:
        return self.similarity_sum / self.completed if self.completed else 0.0

    @property
    def divergence_rate(self) -> float:
        return self.divergent / self.completed if self.completed else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sampled": self.sampled,
            "completed": self.completed,
            "failed": self.failed,
            "skipped_budget": self.skipped_budget,
            "divergent": self.divergent,
            "divergence_rate": round(self.divergence_rate, 4),
            "mean_similarity": round(self.mean_similarity, 4),
            "cost_usd": round(self.cost_micro_usd / 1e6, 6),
        }


def similarity(a: str, b: str) -> float:
    """Token-overlap ratio in [0, 1]. Order-insensitive, length-normalized.

    Jaccard over word sets rather than a sequence ratio: two correct answers to
    the same question routinely differ in phrasing and ordering while agreeing
    on content, and a sequence measure punishes that as though it were
    disagreement.

    This is a **tripwire, not a grade**. Correct answers to the same question
    score anywhere from ~0.3 to ~0.9 depending on verbosity; what carries
    information is the *trend*, and a sustained drop against this deployment's
    own baseline.
    """
    ta = set(_WORD_RE.findall(a.lower()))
    tb = set(_WORD_RE.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class ShadowSampler:
    """Decides what to shadow, runs it out of band, and records the outcome.

    Sampling is **deterministic in the trace id**, not random: the same request
    replayed produces the same decision, which makes an anomaly reproducible
    instead of a thing that happened once. It also avoids `random` in a request
    path, where a global RNG is shared mutable state across workers.
    """

    rate: float = 0.0
    divergence_threshold: float = 0.35
    stats: ShadowStats = field(default_factory=ShadowStats)
    _tasks: set[asyncio.Task] = field(default_factory=set)

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def should_sample(self, trace_id: str, *, tier: str, gate_accepted: bool) -> bool:
        """Only accepted local answers are worth shadowing.

        A cache hit was already checked when it was first produced, and a
        frontier answer *is* the reference — shadowing either would spend money
        to compare something with itself.
        """
        if not self.enabled or not gate_accepted:
            return False
        if tier not in ("local_small", "local_large"):
            return False
        # blake2b rather than `int(trace_id, 16) % N`: the modulo of a raw id
        # is only uniform if the ids are, and sequential or prefixed ids are
        # common enough that "it works with uuid4" is not a property to rely
        # on. Hashing makes the distribution independent of the id scheme.
        bucket = int.from_bytes(
            hashlib.blake2b(trace_id.encode(), digest_size=4).digest(), "big"
        ) % 10_000
        return bucket < self.rate * 10_000

    def spawn(self, coro) -> None:
        """Run a shadow comparison detached from the request.

        The reference is kept so the task is not garbage-collected mid-flight,
        and discarded on completion so the set does not grow without bound —
        the same leak the rate limiter's pruner exists for.
        """
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for in-flight comparisons at shutdown.

        They are already paid for; losing the measurement means having spent
        the money and learned nothing.
        """
        if not self._tasks:
            return
        done, pending = await asyncio.wait(set(self._tasks), timeout=timeout)
        for task in pending:
            task.cancel()

    def record(self, *, served: str, reference: str, cost_micro_usd: int) -> float:
        score = similarity(served, reference)
        self.stats.completed += 1
        self.stats.similarity_sum += score
        self.stats.cost_micro_usd += cost_micro_usd
        if score < self.divergence_threshold:
            self.stats.divergent += 1
        return score
