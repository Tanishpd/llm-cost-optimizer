"""Authentication, rate limiting, and request bounds.

Three separate abuse surfaces, each cheap to close and expensive to leave open
on a gateway that spends money per request:

1. **Auth** — without it, anyone who can reach the port can spend the frontier
   budget. This is the one that turns "unfinished" into "dangerous".
2. **Rate limiting** — a single client should not be able to exhaust the budget
   or the 8 GB of RAM by looping.
3. **Request bounds** — an unbounded `max_tokens` or a 400 KB prompt is a
   denial-of-service and a cost event at the same time.

Keys are compared with `secrets.compare_digest`, so a timing side channel
cannot be used to recover one byte at a time.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger("hlr.security")

# Reachable without a key: liveness probes must work for a load balancer that
# has no credentials, and neither endpoint exposes traffic or cost data.
PUBLIC_PATHS = frozenset({
    "/healthz", "/livez", "/readyz", "/", "/docs", "/openapi.json", "/redoc",
})


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    principal: str = ""      # a stable id for rate limiting; never the key itself
    reason: str = ""
    # Which account pays for this request. Resolved inside `check()` so the raw
    # key is never handed back to the caller — request state is a place things
    # get logged from, and a credential should not live there.
    tenant: str = "anonymous"


class ApiKeyAuth:
    """Bearer / `X-API-Key` authentication against a configured key set.

    With no keys configured the gateway runs **open**, because forcing a key on
    a single-user local install would break the documented quickstart. That is
    only safe on a loopback bind, so `require_auth` exists to make it a hard
    failure anywhere else, and startup warns loudly when open.
    """

    def __init__(
        self,
        keys: set[str],
        *,
        require_auth: bool = False,
        tenant_for: Callable[[str | None], str] | None = None,
    ) -> None:
        self._keys = {k.strip() for k in keys if k.strip()}
        self._require = require_auth
        # A function, not a registry: auth should not grow a dependency on
        # billing to answer "who is this". Defaults to the single-account
        # behaviour this class had before tenants existed.
        self._tenant_for = tenant_for or (lambda key: "default" if key else "anonymous")

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    @property
    def misconfigured(self) -> bool:
        """Auth demanded but no keys supplied — refuse to start rather than
        silently serving open traffic while claiming to be locked down."""
        return self._require and not self._keys

    @staticmethod
    def principal_for(key: str) -> str:
        """A stable, non-reversible id for a key, safe to put in logs and to
        use as a rate-limit bucket."""
        return hashlib.blake2b(key.encode(), digest_size=6).hexdigest()

    def check(self, *, authorization: str | None, api_key_header: str | None) -> AuthResult:
        if not self.enabled:
            return AuthResult(True, principal="anonymous", tenant=self._tenant_for(None))

        presented = api_key_header or ""
        if not presented and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                presented = value.strip()

        if not presented:
            return AuthResult(False, reason="missing API key")

        # compare_digest against every key: constant time, and the loop does not
        # short-circuit on the first mismatch.
        matched = False
        for known in self._keys:
            if secrets.compare_digest(presented, known):
                matched = True
        if not matched:
            return AuthResult(False, reason="invalid API key")

        return AuthResult(
            True,
            principal=self.principal_for(presented),
            tenant=self._tenant_for(presented),
        )


class RateLimiter:
    """Sliding-window limiter, per principal.

    In-process, which is correct for the documented single-worker deployment
    and is stated rather than implied — under multiple workers each process
    would enforce its own window, so the effective limit multiplies by the
    worker count. A Redis-backed counter is the fix if that day comes.

    A sliding window rather than a fixed one: fixed windows let a client send
    2× the limit across a boundary, which on a gateway that spends money per
    request is a real amount of money.
    """

    def __init__(self, *, limit: int = 60, window_s: float = 60.0) -> None:
        self._limit = limit
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def check(
        self, principal: str, *, now: float | None = None, limit: int | None = None
    ) -> tuple[bool, int, float]:
        """(allowed, remaining, retry_after_seconds).

        `limit` overrides the configured one for this call, so a tenant can
        carry its own ceiling without needing a limiter instance each. Passing
        it per call rather than per instance keeps the window shared: a tenant
        whose limit is edited at runtime keeps its existing history instead of
        starting from an empty bucket.
        """
        effective = self._limit if limit is None else limit
        if effective <= 0:
            return True, -1, 0.0
        now = time.monotonic() if now is None else now
        window = self._hits[principal]
        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= effective:
            retry_after = max(0.0, window[0] + self._window - now)
            return False, 0, round(retry_after, 1)

        window.append(now)
        return True, effective - len(window), 0.0

    def reset(self) -> None:
        self._hits.clear()

    def prune(self, *, now: float | None = None) -> int:
        """Drop buckets with no recent hits.

        Without this the dict grows one entry per distinct principal forever —
        a slow memory leak that only shows up in a long-running process, which
        is precisely where it would hurt.
        """
        now = time.monotonic() if now is None else now
        cutoff = now - self._window
        stale = [p for p, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for p in stale:
            del self._hits[p]
        return len(stale)


@dataclass(frozen=True)
class LimitViolation:
    code: str
    message: str


class RequestLimits:
    """Bounds on what a single request may ask for.

    Every one of these is both a cost control and a stability control: an
    unbounded `max_tokens` is an unbounded bill *and* a request that occupies
    the only local model for minutes.
    """

    def __init__(
        self,
        *,
        max_prompt_chars: int = 32_000,
        max_messages: int = 50,
        max_output_tokens: int = 4_096,
        max_body_bytes: int = 256_000,
    ) -> None:
        self.max_prompt_chars = max_prompt_chars
        self.max_messages = max_messages
        self.max_output_tokens = max_output_tokens
        self.max_body_bytes = max_body_bytes

    def check_body_size(self, content_length: int | None) -> LimitViolation | None:
        if content_length is not None and content_length > self.max_body_bytes:
            return LimitViolation(
                "request_too_large",
                f"Request body is {content_length} bytes, over the "
                f"{self.max_body_bytes} limit.",
            )
        return None

    def check(self, *, messages: list, total_chars: int) -> LimitViolation | None:
        if len(messages) > self.max_messages:
            return LimitViolation(
                "too_many_messages",
                f"{len(messages)} messages, over the {self.max_messages} limit.",
            )
        if total_chars > self.max_prompt_chars:
            return LimitViolation(
                "prompt_too_long",
                f"Prompt is {total_chars} characters, over the "
                f"{self.max_prompt_chars} limit.",
            )
        return None

    def clamp_output_tokens(self, requested: int) -> tuple[int, bool]:
        """Clamp rather than reject.

        A caller asking for more than the cap almost always wants "as much as
        you'll give me", and failing the request would be unhelpful. The
        response says it was clamped, so nobody is misled about what they got.
        """
        if requested <= 0:
            return self.max_output_tokens, True
        if requested > self.max_output_tokens:
            return self.max_output_tokens, True
        return requested, False
