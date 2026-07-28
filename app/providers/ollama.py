"""Ollama provider — local chat and embeddings over HTTP.

One client, model name as a parameter: the two local tiers are configuration,
not two classes.

Everything is `async` over a shared pooled `httpx.AsyncClient` created in the
lifespan. A synchronous client would block the event loop for the duration of a
~450 ms generation, serializing the whole gateway behind one request
(invariant 3).

API shapes verified in M0 — see docs/RESEARCH.md §2 and §5.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import (
    Completion,
    ProviderTimeout,
    ProviderUnavailable,
)


class OllamaProvider:
    """Local models. Serves both `local_small` and `local_large`."""

    name = "ollama"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 2048,
        keep_alive: str = "10m",
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive

    # ---- internals -------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Ollama timed out on {path}: {exc}") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailable(
                f"Ollama unreachable at {self._base_url}: {exc}",
                fix="Run: ollama serve &",
            ) from exc

        if response.status_code == 404:
            # Ollama's shape for "model not pulled". Name the exact remedy —
            # a missing model should skip its tier with an actionable reason,
            # not take the cascade down (invariant 4).
            model = payload.get("model", "?")
            raise ProviderUnavailable(
                f"model {model!r} not found on the Ollama server",
                fix=f"ollama pull {model}",
            )
        if response.status_code >= 400:
            detail = response.text[:300]
            raise ProviderUnavailable(f"Ollama {response.status_code} on {path}: {detail}")

        return response.json()

    # ---- Provider protocol ----------------------------------------------

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> Completion:
        """POST /api/chat.

        **`stream` is always False.** The non-streaming path accumulates every
        chunk's logprobs into the final response, which is exactly what the
        confidence gate needs — it cannot decide until the whole answer exists
        (docs/RESEARCH.md §2).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": self._num_ctx,
            },
        }
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs:
                # Server validates 0-20 and 400s outside it; config.py bounds
                # the setting, this is belt-and-braces for direct callers.
                payload["top_logprobs"] = max(0, min(20, top_logprobs))

        started = time.perf_counter()
        data = await self._post("/api/chat", payload)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return Completion(
            content=(data.get("message") or {}).get("content", ""),
            model=data.get("model", model),
            tokens_in=int(data.get("prompt_eval_count") or 0),
            # Includes the stopping token, which produces no logprob entry —
            # so this legitimately exceeds len(logprobs). Verified in M0.
            tokens_out=int(data.get("eval_count") or 0),
            logprobs=data.get("logprobs") or [],
            finish_reason=data.get("done_reason") or "stop",
            latency_ms=round(elapsed_ms, 1),
            load_ms=round(float(data.get("load_duration") or 0) / 1e6, 1),
            raw=data,
        )

    async def embed(self, *, model: str, texts: str | list[str]) -> list[list[float]]:
        """POST /api/embed — the current endpoint.

        Returns `{"embeddings": [[...]]}` (plural, array-of-arrays). The legacy
        `/api/embeddings` returns `{"embedding": [...]}` (singular, flat);
        mixing them up produces a silent shape bug, so this normalizes to
        list-of-vectors always, even for a single input.

        `input` accepts an array — batch when you can.
        """
        payload = {"model": model, "input": texts, "keep_alive": self._keep_alive}
        data = await self._post("/api/embed", payload)
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise ProviderUnavailable(
                f"unexpected /api/embed response shape: keys={list(data)}",
                fix="Expected {'embeddings': [[...]]}. Is EMBED_MODEL an embedding model?",
            )
        return embeddings

    async def version(self) -> str:
        data = await self._get("/api/version")
        return str(data.get("version", "unknown"))

    async def list_models(self) -> set[str]:
        """Names present on the server, both bare and `:latest` forms, so a
        caller can check membership without guessing which form was pulled."""
        data = await self._get("/api/tags")
        names: set[str] = set()
        for model in data.get("models") or []:
            name = model.get("name", "")
            if name:
                names.add(name)
                if name.endswith(":latest"):
                    names.add(name.removesuffix(":latest"))
        return names

    async def _get(self, path: str) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self._base_url}{path}")
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"Ollama timed out on {path}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"Ollama unreachable at {self._base_url}: {exc}", fix="Run: ollama serve &"
            ) from exc
        return response.json()

    async def health(self) -> tuple[bool, str]:
        """Never raises — /healthz must report, not explode."""
        try:
            return True, f"v{await self.version()}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

    async def warm(self, model: str) -> float:
        """One-token generation so the first real request skips the cold load
        (~1.9 s measured in M0). Returns load time in ms; never raises — a
        failed warm is not a reason to refuse to start."""
        try:
            completion = await self.chat(
                model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=1
            )
            return completion.load_ms
        except Exception:  # noqa: BLE001
            return 0.0
