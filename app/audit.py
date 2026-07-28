"""A durable, append-only record of every routing decision.

Telemetry answers "what is happening now" from a Redis stream capped at 10k
entries. That is right for a dashboard and wrong for the question this module
exists for: *why did request `abc123` cost $0.04 last Tuesday?* By Tuesday the
stream has rolled over, and there is nothing to look at.

So: one JSONL line per request, on disk, never deleted until rotation says so.

**Never blocks the event loop** (invariant 3). Records go onto an
`asyncio.Queue` and a single background task drains them, batching writes and
performing the actual `write()` in a worker thread. A request never waits for a
disk.

**Drops are counted, never silent.** If the writer cannot keep up, the queue
fills and new records are dropped — the alternative is blocking the request
path, which is worse. But an audit trail that silently loses records is worse
than no audit trail at all, because it invites conclusions drawn from absence.
So drops increment a counter that `/healthz` reports and shutdown logs.

**One file per process.** A line here can exceed `PIPE_BUF` (4096 bytes) once
`reason`, `gates` and the per-gate metrics are included, and above that size
`O_APPEND` writes from concurrent processes are no longer atomic — two workers
would interleave and corrupt each other's JSON. Per-PID files avoid the problem
outright rather than betting on line length. Read them with `cat audit-*.jsonl`.

**Fsync is not per-record.** A hard kill can lose whatever is in the queue and
the OS buffer — bounded by `flush_interval_s` plus the queue depth, not
unbounded, and stated here rather than discovered. Per-record fsync would cost
a disk round-trip per request and is available by setting `AUDIT_FSYNC=true`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("hlr.audit")

# Queued to tell the writer to finish and exit. A sentinel rather than
# cancellation because `asyncio.to_thread` cannot cancel a *running* thread:
# cancelling at that await either loses a batch the writer had already taken
# off the queue (if the thread had not started) or duplicates it (if it had and
# then completed anyway). Neither is acceptable in an audit trail, and CI caught
# the first one on 3.14 as a flaky rotation test.
_STOP = object()


@dataclass
class AuditStats:
    """What `/healthz` reports. `dropped` is the number that matters."""

    written: int = 0
    dropped: int = 0
    rotations: int = 0
    last_error: str | None = None
    path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "dropped": self.dropped,
            "rotations": self.rotations,
            "last_error": self.last_error,
            "path": self.path,
        }


@dataclass
class _Rotation:
    max_bytes: int
    keep: int
    written_bytes: int = field(default=0)


class AuditLog:
    """Append-only JSONL sink for `RequestRecord`-shaped dicts.

    Disabled by construction when `enabled=False`, in which case every method
    is a no-op and no file is ever created — so a dev checkout does not sprout
    an `audit/` directory just by running the tests.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        enabled: bool = True,
        max_bytes: int = 50 * 1024 * 1024,
        keep_files: int = 5,
        queue_size: int = 10_000,
        flush_interval_s: float = 1.0,
        stop_timeout_s: float = 5.0,
        fsync: bool = False,
    ) -> None:
        self.enabled = enabled
        self._dir = Path(directory)
        self._rotation = _Rotation(max_bytes=max_bytes, keep=max(1, keep_files))
        self._flush_interval = flush_interval_s
        self._stop_timeout_s = stop_timeout_s
        self._fsync = fsync
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max(1, queue_size))
        self._task: asyncio.Task | None = None
        self._closing = False
        self._handle = None
        self.stats = AuditStats()
        # PID in the name so concurrent workers never share a file handle.
        self._path = self._dir / f"audit-{os.getpid()}.jsonl"
        self.stats.path = str(self._path)

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        try:
            await asyncio.to_thread(self._dir.mkdir, parents=True, exist_ok=True)
            self._handle = await asyncio.to_thread(self._open)
        except OSError as exc:
            # A gateway that cannot write its audit log should still answer
            # requests (invariant 4) — but loudly, because someone asked for
            # an audit trail and is not getting one.
            logger.error("audit log disabled: cannot open %s: %s", self._path, exc)
            self.enabled = False
            self.stats.last_error = str(exc)
            return
        self._task = asyncio.create_task(self._drain(), name="audit-writer")
        logger.info("audit log: %s (rotate at %d bytes, keep %d)",
                    self._path, self._rotation.max_bytes, self._rotation.keep)

    def _open(self):
        handle = self._path.open("a", encoding="utf-8")
        self._rotation.written_bytes = self._path.stat().st_size
        return handle

    async def stop(self) -> None:
        """Drain what is queued, then close.

        Signals rather than cancels, and waits for the writer to acknowledge.
        Cancellation is the backstop for a wedged disk, not the mechanism —
        see `_STOP`.
        """
        if self._task is None:
            return
        self._closing = True
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(_STOP)
        try:
            await asyncio.wait_for(self._task, timeout=self._stop_timeout_s)
        except TimeoutError:
            # Only here — a write that has not returned in `stop_timeout_s`.
            # Whatever is still queued is recovered by `_flush_pending` below.
            logger.warning("audit writer did not finish in %.1fs; cancelling",
                           self._stop_timeout_s)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        await asyncio.to_thread(self._flush_pending)
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None
        if self.stats.dropped:
            logger.warning("audit log: %d record(s) dropped this run (queue full)",
                           self.stats.dropped)

    # -- writing ----------------------------------------------------------

    def write(self, record: dict[str, Any]) -> bool:
        """Queue one record. Returns False if it was dropped.

        Deliberately **not** `async`: the caller must not be able to await a
        disk, and `put_nowait` cannot block. Callers do not have to check the
        result — the drop is counted either way.
        """
        if not self.enabled or self._handle is None:
            return False
        try:
            self._queue.put_nowait(json.dumps(record, separators=(",", ":"), default=str))
            return True
        except asyncio.QueueFull:
            self.stats.dropped += 1
            if self.stats.dropped == 1 or self.stats.dropped % 1000 == 0:
                logger.warning(
                    "audit queue full, dropping records (%d so far) — the writer is not "
                    "keeping up with request volume", self.stats.dropped,
                )
            return False

    async def _drain(self) -> None:
        """Batch queued lines and write them in a worker thread.

        Exits when it sees `_STOP`, having first written everything queued
        ahead of it — so a clean shutdown never loses an accepted record.
        """
        while True:
            try:
                item = await self._queue.get()
                stopping = item is _STOP
                batch = [] if stopping else [item]
                # Opportunistically take everything else already queued: one
                # write() for N records rather than N.
                while not self._queue.empty() and len(batch) < 500:
                    nxt = self._queue.get_nowait()
                    if nxt is _STOP:
                        stopping = True
                        break
                    batch.append(nxt)
                if batch:
                    await asyncio.to_thread(self._write_batch, batch)
                # `_closing` is the belt to `_STOP`'s braces: if the queue was
                # full at shutdown the sentinel could not be enqueued, and
                # without this the writer would drain everything and then block
                # on `get()` until `stop()` timed out and cancelled it — a
                # five-second stall on exit under exactly the sustained load
                # that filled the queue.
                if stopping or (self._closing and self._queue.empty()):
                    return
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # A failing disk must not kill the writer task and silently
                # stop auditing for the rest of the process's life.
                self.stats.last_error = str(exc)
                logger.error("audit write failed: %s", exc)
                await asyncio.sleep(self._flush_interval)

    def _flush_pending(self) -> None:
        """Synchronous drain, for shutdown."""
        batch = [x for x in self._drain_nowait() if x is not _STOP]
        if batch:
            self._write_batch(batch)

    def _drain_nowait(self) -> list:
        out = []
        while not self._queue.empty():
            out.append(self._queue.get_nowait())
        return out

    def _write_batch(self, lines: list[str]) -> None:
        """Write one batch, then rotate if the file has outgrown its limit.

        Rotation granularity is the **batch**, not the record: a batch is one
        `write()` and is never split across files. So a file can exceed
        `max_bytes` by up to one batch — at most 500 records, a few hundred KB
        against a 50 MB default. Splitting mid-batch would buy exactness at the
        cost of an extra syscall per write and a partial-batch failure mode,
        which is a bad trade for a bound this loose.
        """
        if self._handle is None:
            return
        payload = "".join(f"{line}\n" for line in lines)
        self._handle.write(payload)
        self._handle.flush()
        if self._fsync:
            os.fsync(self._handle.fileno())
        self.stats.written += len(lines)
        self._rotation.written_bytes += len(payload.encode("utf-8"))
        if self._rotation.written_bytes >= self._rotation.max_bytes:
            self._rotate()

    def _rotate(self) -> None:
        """`audit-PID.jsonl` -> `.1`, `.1` -> `.2`, drop the oldest."""
        if self._handle is not None:
            self._handle.close()
        for i in range(self._rotation.keep - 1, 0, -1):
            src, dst = self._sibling(i), self._sibling(i + 1)
            if src.exists():
                src.replace(dst)
        if self._path.exists():
            self._path.replace(self._sibling(1))
        # Anything past `keep` is deleted here rather than left to accumulate:
        # "bounded disk" is only true if something does the bounding.
        stale = self._sibling(self._rotation.keep + 1)
        if stale.exists():
            stale.unlink()
        self._handle = self._path.open("a", encoding="utf-8")
        self._rotation.written_bytes = 0
        self.stats.rotations += 1

    def _sibling(self, n: int) -> Path:
        return self._path.with_name(f"{self._path.name}.{n}")


def audit_entry(
    record: Any,
    *,
    tenant: str,
    model: str | None = None,
    principal: str | None = None,
) -> dict[str, Any]:
    """Shape one `RequestRecord` into an audit line.

    Reuses the telemetry record rather than defining a second schema — two
    descriptions of the same event drift, and then you have to work out which
    one lied. The additions are the fields telemetry has no reason to carry:
    who paid, which key, and a wall-clock timestamp that survives the process.

    `principal` is the blake2b digest, never the key.
    """
    return {
        "ts": record.created_at or time.time(),
        "trace_id": record.trace_id,
        "tenant": tenant,
        "principal": principal,
        "tier": record.tier,
        "path": list(record.path),
        "entry_tier": record.entry_tier,
        "model": model,
        "reason": record.reason,
        "confidence": record.confidence,
        "gates": record.gates,
        "tokens_in": record.tokens_in,
        "tokens_out": record.tokens_out,
        "cost_micro_usd": record.cost_micro_usd,
        "baseline_micro_usd": record.baseline_micro_usd,
        "cache_hit": record.cache_hit,
        "guard_rejected": record.guard_rejected,
        "latency_ms": record.latency_ms,
        "degraded": list(record.degraded),
    }
