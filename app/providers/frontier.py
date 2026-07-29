"""Frontier provider — an OpenAI-shaped client.

Talks to whatever `FRONTIER_BASE_URL` points at. By default that is the local
mock; setting it to a real provider's base URL is the entire migration
(invariant 8). Nothing in this module knows or cares which it is.

No logprobs: providers rarely expose them, and the frontier is the last tier
anyway — there is nothing above it to escalate to, so there is nothing for a
gate to decide. `Completion.logprobs` stays empty, which
`app.confidence.extract_metrics` maps to metrics that would force escalation if
anything ever asked.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import Completion, ProviderTimeout, ProviderUnavailable


class FrontierProvider:
    """The expensive tier."""

    name = "frontier"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "mock-key",
        timeout_s: float = 60.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    @property
    def base_url(self) -> str:
        return self._base_url

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = False,  # accepted for Provider conformance; not sent
        top_logprobs: int = 0,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"frontier timed out after {self._timeout_s}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailable(
                f"frontier unreachable at {self._base_url}: {exc}",
                fix="Start the mock: uvicorn mock_frontier.main:app --port 8001",
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 401:
            raise ProviderUnavailable(
                "frontier rejected the API key (401)", fix="Check FRONTIER_API_KEY"
            )
        if response.status_code == 429:
            # Carry the body through. Providers put the wait in it — Gemini
            # says "Please retry in 23.7s" — and a caller that backs off blind
            # either waits far too long or retries into the same closed window.
            # Discarding it cost one eval run ~15 minutes on a single call
            # (30+60+120+240+480s) before giving up on a limit that clears in
            # under a minute.
            detail = response.text[:300].replace("\n", " ").strip()
            raise ProviderUnavailable(
                f"frontier rate-limited the request (429): {detail}",
                fix="Back off and retry",
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"frontier {response.status_code}: {_error_message(response)}"
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderUnavailable(f"frontier returned no choices: {list(data)}")

        choice = choices[0]
        usage = data.get("usage") or {}
        return Completion(
            content=(choice.get("message") or {}).get("content", ""),
            model=data.get("model", model),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            logprobs=[],  # not requested; the last tier has nothing to escalate to
            finish_reason=choice.get("finish_reason") or "stop",
            latency_ms=round(elapsed_ms, 1),
            raw=data,
        )

    async def health(self) -> tuple[bool, str]:
        """Never raises — /healthz reports, it does not explode."""
        try:
            response = await self._client.get(
                f"{self._base_url.removesuffix('/v1')}/healthz", timeout=5.0
            )
            if response.status_code == 200:
                return True, f"reachable at {self._base_url}"
            return False, f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


def _error_message(response: httpx.Response) -> str:
    """Pull the message out of OpenAI's error envelope, falling back to raw
    text — a provider having an outage may not return JSON at all."""
    try:
        return str((response.json().get("error") or {}).get("message"))[:300]
    except Exception:  # noqa: BLE001
        return response.text[:300]
