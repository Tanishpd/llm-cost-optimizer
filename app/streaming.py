"""Server-sent events for `stream: true` — buffer first, then stream.

The gate cannot decide whether to escalate until the complete answer and its
logprobs exist (D8). So the cascade runs to completion exactly as it does for a
non-streaming request, and only *then* is the finished answer re-emitted as
OpenAI-shaped `chat.completion.chunk` events.

**This does not reduce time-to-first-token, and it is not meant to.** The whole
answer is already in memory before the first byte goes out. What it buys is
client compatibility: `stream=True` in the OpenAI SDK, LangChain, and every
other client that assumes SSE now works instead of returning 400.

Buffering first also has two properties a true streaming proxy cannot offer,
and they are the reason this design was chosen over streaming-with-correction:

- **No token is ever retracted.** A tier's answer is either accepted by its
  gate or discarded before anything is sent. Streaming the 0.5B's answer and
  then overwriting it with the frontier's would be worse than not streaming.
- **Errors stay HTTP errors.** Everything that can fail has already failed by
  the time the response starts, so a failure is a 4xx/5xx with a JSON body —
  never a half-written stream a client has to detect and unwind.

Pure functions over a finished response. No I/O, no HTTP objects.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from app.schemas import ChatCompletionResponse

# Split into words while preserving every whitespace byte, so concatenating the
# deltas reproduces the original content exactly. `\S+\s*` takes a word plus
# its trailing space; the `\s+` alternative catches a leading run. A test
# asserts the round trip, because a chunker that quietly drops a newline is the
# kind of bug that only shows up in someone's rendered markdown.
_WORD_RE = re.compile(r"\S+\s*|\s+")

# The terminator every OpenAI-compatible client watches for.
DONE = "data: [DONE]\n\n"


def split_content(text: str, chunk_chars: int) -> list[str]:
    """Group words into chunks of at most `chunk_chars`, never splitting a word.

    Per-word events would be smoother and cost ~200 bytes of JSON envelope per
    word — 80 KB of overhead on a 2 KB answer. Grouping trades a little
    smoothness for that. A word longer than `chunk_chars` becomes its own chunk
    rather than being cut in half.
    """
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]

    chunks: list[str] = []
    current = ""
    for word in _WORD_RE.findall(text):
        if current and len(current) + len(word) > chunk_chars:
            chunks.append(current)
            current = word
        else:
            current += word
    if current:
        chunks.append(current)
    return chunks


def _event(payload: dict) -> str:
    """One SSE frame. `separators` keeps the envelope small — it is sent once
    per chunk, so the bytes add up."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def sse_stream(
    response: ChatCompletionResponse,
    *,
    chunk_chars: int = 24,
    include_usage: bool = False,
) -> Iterator[str]:
    """Re-emit a finished response as `chat.completion.chunk` events.

    Frame order, matching the OpenAI streaming contract:

    1. a role-only chunk (`delta: {"role": "assistant"}`)
    2. N content chunks (`delta: {"content": "..."}`)
    3. a terminal chunk carrying `finish_reason` **and the `x_hlr` block**
    4. optionally a usage chunk with empty `choices`, if the caller asked
    5. ``data: [DONE]``

    `x_hlr` rides on the terminal chunk rather than the first because the
    routing story is not complete until the answer is: a request that walked
    0.5B -> 3B -> frontier only knows its full `path` at the end. Invariant 1
    still holds — every response carries the block, streaming included.
    """
    head = {
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
    }

    yield _event({**head, "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})

    choice = response.choices[0]
    for piece in split_content(choice.message.content, chunk_chars):
        yield _event({**head, "choices": [
            {"index": 0, "delta": {"content": piece}, "finish_reason": None}]})

    yield _event({
        **head,
        "choices": [{"index": 0, "delta": {}, "finish_reason": choice.finish_reason}],
        "x_hlr": response.x_hlr.model_dump(),
    })

    if include_usage:
        # OpenAI's shape for this: a final chunk with no choices at all.
        yield _event({**head, "choices": [], "usage": response.usage.model_dump()})

    yield DONE
