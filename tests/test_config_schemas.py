"""Unit tests for app/config.py and app/schemas.py.

The config tests exist mostly to pin invariants that a well-meaning edit could
break — the raw/calibrated threshold split, the RAM ceiling, the top_logprobs
server bound.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HlrMetadata,
    Message,
)

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_defaults_match_the_documented_config_table():
    s = Settings(_env_file=None)
    assert s.small_model == "qwen2.5:0.5b"
    assert s.large_model == "qwen2.5:3b"
    assert s.embed_model == "nomic-embed-text"
    assert s.confidence_mode == "calibrated"
    assert (s.threshold_small, s.threshold_large) == (0.55, 0.62)
    assert (s.target_small, s.target_large) == (0.92, 0.85)
    assert s.semantic_threshold == 0.95


def test_the_shipped_default_matches_the_committed_calibrator():
    """`calibrated` is only a safe default because `eval/calibrator.json` is
    committed *and* was fitted on the models this file also defaults to. Change
    either without the other and every fresh clone refuses to start."""
    import json
    from pathlib import Path

    s = Settings(_env_file=None)
    prov = json.loads(
        (Path(__file__).resolve().parent.parent / s.calibrator_path).read_text()
    ).get("provenance") or {}
    assert prov.get("small_model") == s.small_model
    assert prov.get("large_model") == s.large_model


def test_threshold_for_respects_confidence_mode():
    raw = Settings(_env_file=None, confidence_mode="raw")
    assert raw.threshold_for("small") == 0.55
    assert raw.threshold_for("large") == 0.62

    cal = Settings(_env_file=None, confidence_mode="calibrated")
    assert cal.threshold_for("small") == 0.92
    assert cal.threshold_for("large") == 0.85


def test_gate_one_demands_a_higher_bar_than_gate_two_when_calibrated():
    """The corrected economics: escalating from gate 1 is nearly free, so it
    should escalate readily -> higher acceptance bar. Gate 2 spends real money.
    An earlier draft had this backwards; this test stops it regressing."""
    s = Settings(_env_file=None, confidence_mode="calibrated")
    assert s.threshold_for("small") > s.threshold_for("large")


def test_raw_thresholds_are_not_claimed_to_be_comparable():
    """In raw mode the ordering reverses, because the two numbers apply to
    different models with different geo_mean distributions (measured: 0.5B
    0.47 vs 3B 0.60 on the same prompt). Documenting via test that this
    inversion is expected, not a bug."""
    s = Settings(_env_file=None, confidence_mode="raw")
    assert s.threshold_for("small") < s.threshold_for("large")


def test_top_logprobs_respects_the_server_bound():
    """Ollama 400s outside 0-20 (server/routes.go)."""
    assert Settings(_env_file=None, top_logprobs=20).top_logprobs == 20
    assert Settings(_env_file=None, top_logprobs=0).top_logprobs == 0
    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_logprobs=21)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, top_logprobs=-1)


def test_max_loaded_models_defaults_to_the_ram_ceiling():
    """0.5B (479 MB) + 3B (2.2 GB) measured is already ~2.7 GB on an 8 GB box."""
    assert Settings(_env_file=None).ollama_max_loaded_models == 2


def test_base_urls_are_normalized():
    s = Settings(_env_file=None, ollama_base_url="http://x:1/", frontier_base_url="http://y:2/")
    assert s.ollama_base_url == "http://x:1"
    assert s.frontier_base_url == "http://y:2"


def test_model_for_maps_tiers():
    s = Settings(_env_file=None)
    assert s.model_for("local_small") == s.small_model
    assert s.model_for("local_large") == s.large_model
    assert s.model_for("frontier") == s.frontier_model


def test_prices_are_decimal_not_float():
    """Money must never be float — accumulation drifts."""
    from decimal import Decimal

    s = Settings(_env_file=None)
    assert isinstance(s.frontier_price_in, Decimal)
    assert isinstance(s.frontier_price_out, Decimal)


def test_frontier_api_key_never_renders_in_a_repr():
    """The one setting that is a credential must not leak through a repr.

    Nothing currently logs a Settings object, but exception handlers and
    debuggers print whatever they are handed. SecretStr makes that safe by
    construction instead of by everyone remembering.
    """
    s = Settings(_env_file=None, frontier_api_key="sk-super-secret-123")

    assert "sk-super-secret-123" not in repr(s)
    assert "sk-super-secret-123" not in str(s)
    assert "sk-super-secret-123" not in str(s.model_dump())
    # ...and is still readable when actually needed.
    assert s.frontier_api_key.get_secret_value() == "sk-super-secret-123"


# --------------------------------------------------------------------------
# request schema
# --------------------------------------------------------------------------


def test_request_extracts_last_user_prompt():
    req = ChatCompletionRequest(
        messages=[
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="second"),
        ]
    )
    assert req.user_prompt == "second"


def test_request_detects_multi_turn():
    """Multi-turn is never semantic-cached — the embedding of the last message
    flattens away the context that makes the answer correct (D15)."""
    single = ChatCompletionRequest(messages=[Message(role="user", content="hi")])
    assert single.is_multi_turn is False

    with_system = ChatCompletionRequest(
        messages=[Message(role="system", content="be terse"), Message(role="user", content="hi")]
    )
    assert with_system.is_multi_turn is False  # a system prompt is not a turn

    multi = ChatCompletionRequest(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="again"),
        ]
    )
    assert multi.is_multi_turn is True


def test_request_concatenates_system_prompts():
    req = ChatCompletionRequest(
        messages=[
            Message(role="system", content="a"),
            Message(role="system", content="b"),
            Message(role="user", content="q"),
        ]
    )
    assert req.system_prompt == "a\nb"


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gateway", None),
        ("gateway:small", "local_small"),
        ("gateway:large", "local_large"),
        ("gateway:frontier", "frontier"),
        ("gpt-4o", None),
    ],
)
def test_reserved_models_force_a_tier(model, expected):
    req = ChatCompletionRequest(model=model, messages=[Message(role="user", content="q")])
    assert req.forced_tier == expected


def test_nocache_bypass_flag():
    req = ChatCompletionRequest(model="gateway:nocache", messages=[Message(role="user",
        content="q")])
    assert req.bypass_cache is True


def test_empty_messages_rejected():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])


def test_unknown_openai_fields_are_tolerated():
    """A real client may send seed, n, presence_penalty... don't 422 on those."""
    req = ChatCompletionRequest(
        messages=[Message(role="user", content="q")], seed=42, presence_penalty=0.1
    )
    assert req.user_prompt == "q"


# --------------------------------------------------------------------------
# response schema
# --------------------------------------------------------------------------


def test_response_reports_the_serving_model_not_the_requested_one():
    """Requested "gateway", served by qwen2.5:3b — the response says the latter,
    which makes it self-describing in logs."""
    resp = ChatCompletionResponse.build(
        content="Paris",
        model="qwen2.5:3b",
        tokens_in=41,
        tokens_out=2,
        hlr=HlrMetadata(trace_id="t", tier="local_large"),
    )
    assert resp.model == "qwen2.5:3b"
    assert resp.object == "chat.completion"
    assert resp.usage.total_tokens == 43
    assert resp.choices[0].message.role == "assistant"


def test_response_always_carries_x_hlr():
    """Invariant 1."""
    resp = ChatCompletionResponse.build(
        content="x", model="m", tokens_in=1, tokens_out=1,
        hlr=HlrMetadata(trace_id="t", tier="cache_exact"),
    )
    assert "x_hlr" in resp.model_dump()
    assert resp.x_hlr.trace_id == "t"


def test_hlr_defaults_are_serving_safe():
    """A minimally-populated block must still serialize with every documented
    key present — clients and the dashboard read them unconditionally."""
    d = HlrMetadata(trace_id="t", tier=None).model_dump()
    for key in (
        "trace_id", "tier", "path", "gates", "confidence", "confidence_mode",
        "reason", "router", "cache", "latency_ms", "tokens_in", "tokens_out",
        "cost_usd", "baseline_cost_usd", "cost_saved_usd", "degraded",
    ):
        assert key in d, f"missing {key}"


def test_hlr_path_records_every_tier_attempted():
    """A request that walked 0.5B -> 3B -> frontier paid for all three."""
    h = HlrMetadata(
        trace_id="t", tier="frontier", path=["local_small", "local_large", "frontier"]
    )
    assert len(h.path) == 3
    assert h.path[-1] == h.tier


def test_response_ids_are_unique():
    a = ChatCompletionResponse.build(
        content="x", model="m", tokens_in=0, tokens_out=0,
        hlr=HlrMetadata(trace_id="t", tier=None),
    )
    b = ChatCompletionResponse.build(
        content="x", model="m", tokens_in=0, tokens_out=0,
        hlr=HlrMetadata(trace_id="t", tier=None),
    )
    assert a.id != b.id and a.id.startswith("chatcmpl-")


def test_project_root_is_importable_without_module_invocation():
    """The suite must work under bare `pytest`, not only `python -m pytest`.

    Only the `-m` form puts the CWD on sys.path, so without `pythonpath` in
    pyproject the whole suite fails to collect under CI and most editors —
    which is exactly what happened on this project's first CI run.
    """
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "." in cfg["tool"]["pytest"]["ini_options"]["pythonpath"]
