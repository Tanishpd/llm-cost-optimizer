"""Tests for app/security.py.

Auth is the one gap that turns "unfinished" into "dangerous" on a gateway that
spends money per request, so these lean on the negative cases: what must be
rejected, and what must not silently pass.
"""

from __future__ import annotations

import pytest

from app.security import PUBLIC_PATHS, ApiKeyAuth, RateLimiter, RequestLimits

# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def test_no_keys_configured_runs_open():
    """Documented behaviour: a single-user local install must not need a key,
    or the quickstart breaks."""
    auth = ApiKeyAuth(set())
    assert auth.enabled is False
    assert auth.check(authorization=None, api_key_header=None).ok is True


def test_require_auth_without_keys_is_a_misconfiguration():
    """A config that silently disables auth while claiming to enforce it is
    worse than a crash, because nothing tells you."""
    assert ApiKeyAuth(set(), require_auth=True).misconfigured is True
    assert ApiKeyAuth({"k"}, require_auth=True).misconfigured is False
    assert ApiKeyAuth(set(), require_auth=False).misconfigured is False


def test_valid_key_via_bearer():
    auth = ApiKeyAuth({"secret-key"})
    assert auth.check(authorization="Bearer secret-key", api_key_header=None).ok is True


def test_valid_key_via_x_api_key_header():
    auth = ApiKeyAuth({"secret-key"})
    assert auth.check(authorization=None, api_key_header="secret-key").ok is True


def test_missing_key_is_rejected():
    result = ApiKeyAuth({"secret-key"}).check(authorization=None, api_key_header=None)
    assert result.ok is False
    assert "missing" in result.reason


def test_wrong_key_is_rejected():
    result = ApiKeyAuth({"secret-key"}).check(authorization="Bearer nope", api_key_header=None)
    assert result.ok is False
    assert "invalid" in result.reason


@pytest.mark.parametrize("header", [
    "secret-key",          # no scheme
    "Basic secret-key",    # wrong scheme
    "bearersecret-key",    # no space
])
def test_malformed_authorization_headers_are_rejected(header):
    assert ApiKeyAuth({"secret-key"}).check(authorization=header, api_key_header=None).ok is False


def test_bearer_scheme_is_case_insensitive():
    auth = ApiKeyAuth({"k"})
    assert auth.check(authorization="bearer k", api_key_header=None).ok is True
    assert auth.check(authorization="BEARER k", api_key_header=None).ok is True


def test_multiple_keys_are_all_accepted():
    auth = ApiKeyAuth({"alice", "bob"})
    assert auth.check(authorization="Bearer alice", api_key_header=None).ok is True
    assert auth.check(authorization="Bearer bob", api_key_header=None).ok is True
    assert auth.check(authorization="Bearer carol", api_key_header=None).ok is False


def test_principal_is_stable_and_not_the_key():
    """It ends up in logs and rate-limit buckets, so it must not be reversible."""
    p1 = ApiKeyAuth.principal_for("super-secret")
    p2 = ApiKeyAuth.principal_for("super-secret")
    assert p1 == p2
    assert "super-secret" not in p1
    assert ApiKeyAuth.principal_for("other") != p1


def test_different_keys_get_different_principals():
    """Otherwise one client's traffic would consume another's rate budget."""
    auth = ApiKeyAuth({"alice", "bob"})
    a = auth.check(authorization="Bearer alice", api_key_header=None).principal
    b = auth.check(authorization="Bearer bob", api_key_header=None).principal
    assert a != b


def test_health_endpoints_are_public():
    """A load balancer has no credentials, and neither endpoint exposes
    traffic or cost data."""
    assert "/healthz" in PUBLIC_PATHS
    assert "/" in PUBLIC_PATHS
    assert "/metrics" not in PUBLIC_PATHS  # cost data — must be protected


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------


def test_requests_under_the_limit_pass():
    rl = RateLimiter(limit=3, window_s=60)
    for _ in range(3):
        allowed, _, _ = rl.check("alice")
        assert allowed is True


def test_the_request_over_the_limit_is_blocked():
    rl = RateLimiter(limit=3, window_s=60)
    for _ in range(3):
        rl.check("alice")
    allowed, remaining, retry_after = rl.check("alice")
    assert allowed is False
    assert remaining == 0
    assert retry_after > 0


def test_principals_have_independent_budgets():
    rl = RateLimiter(limit=2, window_s=60)
    rl.check("alice"), rl.check("alice")
    assert rl.check("alice")[0] is False
    assert rl.check("bob")[0] is True


def test_window_slides_rather_than_resetting():
    """A fixed window lets a client send 2x the limit across a boundary —
    real money on a gateway that spends per request."""
    rl = RateLimiter(limit=2, window_s=10)
    rl.check("alice", now=0.0)
    rl.check("alice", now=1.0)
    assert rl.check("alice", now=2.0)[0] is False
    # The first hit ages out at t=10, so one slot frees — not the whole budget.
    assert rl.check("alice", now=10.5)[0] is True
    assert rl.check("alice", now=10.6)[0] is False


def test_remaining_counts_down():
    rl = RateLimiter(limit=3, window_s=60)
    assert rl.check("alice")[1] == 2
    assert rl.check("alice")[1] == 1
    assert rl.check("alice")[1] == 0


def test_zero_limit_disables_the_limiter():
    rl = RateLimiter(limit=0)
    assert rl.enabled is False
    for _ in range(100):
        assert rl.check("alice")[0] is True


def test_prune_drops_stale_buckets():
    """Without this the dict grows one entry per distinct principal forever —
    a leak that only appears in a long-running process, which is exactly where
    it would hurt."""
    rl = RateLimiter(limit=5, window_s=10)
    for i in range(50):
        rl.check(f"client-{i}", now=0.0)
    assert len(rl._hits) == 50
    assert rl.prune(now=100.0) == 50
    assert len(rl._hits) == 0


def test_prune_keeps_active_buckets():
    rl = RateLimiter(limit=5, window_s=10)
    rl.check("old", now=0.0)
    rl.check("recent", now=95.0)
    rl.prune(now=100.0)
    assert "recent" in rl._hits
    assert "old" not in rl._hits


# --------------------------------------------------------------------------
# request bounds
# --------------------------------------------------------------------------


def test_oversized_body_is_rejected():
    limits = RequestLimits(max_body_bytes=1000)
    assert limits.check_body_size(1001) is not None
    assert limits.check_body_size(1000) is None
    assert limits.check_body_size(None) is None  # no header — checked later by content


def test_long_prompt_is_rejected():
    limits = RequestLimits(max_prompt_chars=100)
    violation = limits.check(messages=[1], total_chars=101)
    assert violation is not None
    assert violation.code == "prompt_too_long"
    assert "101" in violation.message


def test_too_many_messages_is_rejected():
    limits = RequestLimits(max_messages=3)
    violation = limits.check(messages=[1, 2, 3, 4], total_chars=10)
    assert violation is not None
    assert violation.code == "too_many_messages"


def test_within_bounds_passes():
    limits = RequestLimits(max_prompt_chars=100, max_messages=5)
    assert limits.check(messages=[1, 2], total_chars=50) is None


def test_max_tokens_is_clamped_not_rejected():
    """A caller asking for more than the cap almost always means "as much as
    you'll give me" — failing them would be unhelpful."""
    limits = RequestLimits(max_output_tokens=100)
    assert limits.clamp_output_tokens(999_999) == (100, True)
    assert limits.clamp_output_tokens(50) == (50, False)
    assert limits.clamp_output_tokens(100) == (100, False)


def test_nonpositive_max_tokens_becomes_the_cap():
    limits = RequestLimits(max_output_tokens=100)
    assert limits.clamp_output_tokens(0) == (100, True)
    assert limits.clamp_output_tokens(-5) == (100, True)
