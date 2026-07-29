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

import re
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
        reasoning_effort: str = "",
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        # Only sent when asked for. `reasoning_effort` is a standard OpenAI
        # field, but the values a provider accepts are not standard at all:
        # Gemini 2.5 Flash takes "none" to disable thinking outright, Gemini 3
        # dropped that ability entirely (its floor is "low"), and Anthropic
        # does not use this field. Sending it unconditionally would 400 against
        # the default judge model.
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        # An empty key is a configuration error, and it must not be reported
        # as an unreachable upstream. `Authorization: Bearer ` is an illegal
        # header, so httpx raises before any connection is attempted — which
        # lands in the `RequestError` branch below and advises starting a mock
        # that is very often already running. Diagnosed the slow way once:
        # `curl` to the same URL returned 401, so the mock looked like it was
        # rejecting the key, when in fact the gateway had never sent a request.
        if not self._api_key.strip():
            raise ProviderUnavailable(
                f"frontier has no API key configured ({self._base_url})",
                fix="Set FRONTIER_API_KEY. The bundled mock accepts `mock-key`, "
                    "which is what .env.example ships.",
            )

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
            # Search the WHOLE body for the delay, then lead with it.
            #
            # Truncating first does not work: Gemini puts ~430 characters of
            # quota boilerplate and documentation links before "Please retry in
            # 23.7s", so a 300-character clip cuts off the one field a caller
            # needs. Found the hard way — the truncated version still produced
            # blind exponential backoff and killed a run at the same prompt
            # twice.
            body = response.text
            hint = re.search(r"retry in ([\d.]+)s", body)
            lead = f"retry in {hint.group(1)}s — " if hint else ""
            raise ProviderUnavailable(
                f"frontier rate-limited the request (429): {lead}"
                f"{body[:200].replace(chr(10), ' ').strip()}",
                fix="Back off and retry",
            )
        if response.status_code >= 400:
            # Name the likely culprit. Which `reasoning_effort` values a model
            # accepts varies per model *within* one provider — gemini-2.5-flash
            # takes "none", gemini-3-* rejects it — so this 400 lands mid-run
            # with an opaque body and no obvious cause.
            fix = ("Unset it, or use a value this model accepts "
                   "(\"none\" works on gemini-2.5-flash; Gemini 3 has no off "
                   "switch and its floor is \"low\")") if reasoning_effort else ""
            raise ProviderUnavailable(
                f"frontier {response.status_code}: {_error_message(response)}",
                fix=fix,
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
