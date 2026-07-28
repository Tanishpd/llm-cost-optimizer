"""Settings. Every knob lives here — no `os.getenv` scattered through the code.

Mirrors the configuration table in docs/PLAN.md and `.env.example`; those three
must stay in agreement. A `Settings` instance is built once in the lifespan and
passed down, so nothing reads the environment at request time.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ---- Local models (Ollama) -------------------------------------------
    ollama_base_url: str = "http://localhost:11434"
    small_model: str = "qwen2.5:0.5b"
    large_model: str = "qwen2.5:3b"
    embed_model: str = "nomic-embed-text"
    ollama_num_ctx: int = 2048
    ollama_keep_alive: str = "10m"
    # Do not raise on 8 GB: 0.5B (479 MB) + 3B (2.2 GB) measured resident is
    # already ~2.7 GB, and swap hit 87-91% during the M0 probes. RESEARCH.md §5.
    ollama_max_loaded_models: int = 2
    ollama_timeout_s: float = 120.0
    ollama_connect_timeout_s: float = 5.0

    # ---- Frontier tier ---------------------------------------------------
    frontier_base_url: str = "http://localhost:8001/v1"
    # SecretStr so it renders as `**********` in any repr, log line, or exception
    # that happens to carry a Settings object. Read it with .get_secret_value().
    frontier_api_key: SecretStr = SecretStr("mock-key")
    frontier_model: str = "gpt-4o-mock"
    frontier_timeout_s: float = 60.0
    # USD per 1M tokens. PLACEHOLDERS — verify current rates before quoting any
    # savings figure. Decimal, not float: these feed money arithmetic.
    frontier_price_in: Decimal = Decimal("2.50")
    frontier_price_out: Decimal = Decimal("10.00")

    # ---- Redis -----------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_local: int = 3600
    # Longer: frontier answers cost real money and are worth keeping.
    cache_ttl_frontier: int = 86400

    # ---- Semantic cache --------------------------------------------------
    semantic_enabled: bool = True
    # Conservative on purpose. A wrong hit is worse than a miss (D15).
    semantic_threshold: float = 0.95
    semantic_max_chars: int = 500
    semantic_max_entries: int = 10_000

    # ---- Confidence gating -----------------------------------------------
    confidence_mode: Literal["raw", "calibrated"] = "raw"
    # Where the fitted per-gate calibrators live. Read once at startup; missing
    # is fine in raw mode, fatal in calibrated mode (app/main.py).
    calibrator_path: str = "eval/calibrator.json"
    # Raw-mode thresholds are PER-MODEL and not comparable to each other: the
    # 0.5B's geo_mean distribution sits lower than the 3B's (measured 0.47 vs
    # 0.60 on the same medium prompt). The ordering is an artifact of the
    # models, not a policy. RESEARCH.md §3.1.
    threshold_small: float = 0.55
    threshold_large: float = 0.62
    # Calibrated mode. Gate 1's bar is HIGHER on purpose: escalating from it
    # costs ~450 ms and no money, so escalate readily. Gate 2 spends real money
    # and carries the spec's 0.85. RESEARCH.md §3.
    target_small: float = 0.92
    target_large: float = 0.85
    confidence_strategy: Literal["logprobs", "self_consistency"] = "logprobs"
    top_logprobs: int = Field(default=5, ge=0, le=20)  # server-validated 0-20

    # ---- Pre-router ------------------------------------------------------
    router_small_max_tokens: int = 60
    # Above this, enter at local_large — never frontier on length alone.
    # Long != hard: summarization is long and cheap. DECISIONS.md D10.
    router_escalate_min_tokens: int = 600

    # ---- Eval (M6 only) --------------------------------------------------
    eval_use_real_frontier: bool = False
    eval_judge_base_url: str = "https://api.anthropic.com/v1"
    eval_judge_model: str = "claude-sonnet-5"
    eval_judge_api_key: SecretStr = SecretStr("")

    # ---- Security --------------------------------------------------------
    # Comma-separated. Empty means the gateway runs OPEN, which is only
    # acceptable on a loopback bind — startup warns loudly, and REQUIRE_AUTH
    # turns it into a hard failure.
    gateway_api_keys: str = ""
    require_auth: bool = False
    # Sliding window, per API key (per client IP when running open).
    # 0 disables. In-process, so under multiple workers the effective limit
    # multiplies by the worker count.
    rate_limit_per_minute: int = 60
    # Request bounds. Each is a cost control and a stability control at once.
    max_prompt_chars: int = 32_000
    max_messages: int = 50
    max_output_tokens: int = 4_096
    max_body_bytes: int = 256_000

    # ---- Resilience ------------------------------------------------------
    # A dead tier otherwise costs every request a full timeout before the
    # cascade escalates past it. The breaker turns that into a fast skip.
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 30.0
    # Narrow by design: only timeouts and connection errors, and the frontier
    # is not retried on timeouts because the tokens may already be billed.
    retry_attempts: int = 2
    retry_base_delay_s: float = 0.25

    # ---- Logging ---------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # ---- Gateway ---------------------------------------------------------
    default_max_tokens: int = 512
    default_temperature: float = 0.0
    warm_models_on_startup: bool = True

    @field_validator("ollama_base_url", "frontier_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}

    def threshold_for(self, gate: Literal["small", "large"]) -> float:
        """The active threshold for a gate, respecting `confidence_mode`.

        Keeps the raw/calibrated distinction in one place — callers should never
        have to remember which pair of settings applies.
        """
        if self.confidence_mode == "calibrated":
            return self.target_small if gate == "small" else self.target_large
        return self.threshold_small if gate == "small" else self.threshold_large

    def model_for(self, tier: str) -> str:
        """Map a tier name to the model that serves it."""
        return {
            "local_small": self.small_model,
            "local_large": self.large_model,
            "frontier": self.frontier_model,
        }[tier]


@lru_cache
def get_settings() -> Settings:
    """Cached accessor. The lifespan builds one instance; tests override by
    calling `get_settings.cache_clear()` or constructing `Settings(...)` directly."""
    return Settings()
