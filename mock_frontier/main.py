"""A faithful mock of an OpenAI-compatible frontier API.

Runs as a **separate service** on its own port, and the gateway reaches it over
real HTTP. That is the point (DECISIONS.md D9): pointing `FRONTIER_BASE_URL` at
a live provider must be a config change, not a rewrite, and the escalation path
should exercise real serialization, timeouts, and failure modes rather than an
in-process stub that cannot fail realistically.

    uvicorn mock_frontier.main:app --port 8001

The envelope is faithful — including a truthful `usage` block, which is what
makes the gateway's cost math real. The *content* is canned. This mock proves
the escalation path and the accounting, not answer quality; that is what the
M6 eval harness needs a real API for.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Frontier API", version="1.0.0")

# Simulated latency, so the gateway's tier timings look like reality rather
# than localhost. Named apart from the real thing in every doc: this is
# MOCK_LATENCY_MS, not a provider's measured p50.
MOCK_LATENCY_MS = float(os.getenv("MOCK_LATENCY_MS", "1800"))
MOCK_JITTER_MS = float(os.getenv("MOCK_JITTER_MS", "400"))
# Raise to exercise the gateway's degradation paths (M8).
MOCK_FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0.0"))
# Seeded so tests are deterministic; unset for realistic jitter.
_SEED = os.getenv("MOCK_SEED")
_rng = random.Random(int(_SEED)) if _SEED else random.Random()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-4o-mock"
    messages: list[Message] = Field(min_length=1)
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    stream: bool = False


def estimate_tokens(text: str) -> int:
    """Rough token count for the `usage` block.

    A real provider counts exactly; this approximates, and the number is
    *honest about being an approximation* rather than a fabricated round
    figure. The gateway's cost math is only as good as this, which is the
    other reason M6 uses a real API.
    """
    return max(len(text) // 4, 1)


ANSWER_TEMPLATE = (
    "[mock frontier response] This is canned text from the mock frontier service, "
    "not a real model answer. It exists so the gateway can exercise escalation, "
    "latency, and cost accounting end to end. The prompt began: {preview!r}. "
    "For genuine answer quality, point FRONTIER_BASE_URL at a real provider — "
    "no code change is required."
)


@app.post("/v1/chat/completions")
async def chat_completions(
    body: ChatRequest, request: Request, authorization: str | None = Header(default=None)
) -> Any:
    """OpenAI `chat.completion`. Same shape a real provider returns."""
    if body.stream:
        return _error(400, "Streaming is not implemented in the mock.", "invalid_request_error")

    # A real provider rejects a missing key; the mock does too, so the gateway's
    # auth wiring is exercised rather than assumed.
    if not authorization or not authorization.lower().startswith("bearer "):
        return _error(401, "Missing or malformed Authorization header.", "invalid_request_error")

    if MOCK_FAILURE_RATE > 0 and _rng.random() < MOCK_FAILURE_RATE:
        return _error(503, "Simulated upstream failure (MOCK_FAILURE_RATE).", "server_error")

    delay = max(MOCK_LATENCY_MS + _rng.uniform(-MOCK_JITTER_MS, MOCK_JITTER_MS), 0.0)
    await asyncio.sleep(delay / 1000.0)

    prompt = "\n".join(m.content for m in body.messages)
    preview = (body.messages[-1].content or "")[:60]
    content = ANSWER_TEMPLATE.format(preview=preview)

    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = min(estimate_tokens(content), body.max_tokens)

    return {
        "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _error(status: int, message: str, err_type: str) -> JSONResponse:
    """OpenAI's error envelope, so the gateway's parsing is exercised too."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "code": None, "param": None}},
    )


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mock-frontier",
        "config": {
            "latency_ms": MOCK_LATENCY_MS,
            "jitter_ms": MOCK_JITTER_MS,
            "failure_rate": MOCK_FAILURE_RATE,
            "seeded": _SEED is not None,
        },
    }


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "mock-frontier",
        "note": "Faithful OpenAI envelope with canned content. Swap for a real "
        "provider by pointing FRONTIER_BASE_URL elsewhere.",
        "endpoints": ["/v1/chat/completions", "/healthz"],
    }
