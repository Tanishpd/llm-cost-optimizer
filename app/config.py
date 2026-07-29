"""Settings. Every knob lives here — no `os.getenv` scattered through the code.

Mirrors the configuration table in docs/PLAN.md and `.env.example`; those three
must stay in agreement. A `Settings` instance is built once in the lifespan and
passed down, so nothing reads the environment at request time.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# One-way: budgets.py knows nothing about settings, so this cannot cycle.
from app.budgets import Tenant


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
    # Characters per SSE chunk when `stream: true`. The answer is already
    # complete, so this only trades event smoothness against envelope overhead
    # (~200 bytes of JSON per event). 0 sends the whole answer as one chunk.
    stream_chunk_chars: int = 24

    # `calibrated` since M25, because the number finally means what it says:
    # both gates passed validation against a real frontier at n=110 and held
    # under every counterfactual refit (RESEARCH.md 3.7), so 0.85 is a genuine
    # "85% chance this tier's answer is as good as the frontier's" rather than
    # a raw geo_mean that happens to sit near it.
    #
    # This default only works because `eval/calibrator.json` is committed and
    # was fitted on the models this file also defaults to. Change SMALL_MODEL
    # or LARGE_MODEL and the gateway REFUSES TO START, naming both model sets
    # — deliberately, since a calibrator fitted elsewhere still returns a
    # number in [0, 1] that looks exactly like a probability. Set
    # CONFIDENCE_MODE=raw to run other models without refitting.
    confidence_mode: Literal["raw", "calibrated"] = "calibrated"
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
    # The judge answers in one word, but a *reasoning* model spends hidden
    # thinking tokens against this budget before emitting anything — and those
    # do not show up in `completion_tokens`. At 10 (the old hardcoded value)
    # gemini-3-flash-preview returned an empty string and every verdict parsed
    # as None: 129 paid calls, zero labels, no error. 64 is ample for a
    # one-word verdict and cheap enough not to matter.
    eval_judge_max_tokens: int = 64
    # The reference answer's budget, separate from the local tiers'.
    #
    # One knob used to drive both, which quietly broke the calibration against
    # a reasoning model: at 120 the reference came back as
    # "The mitochondrion is an" — a fragment — because ~490 tokens went to
    # hidden reasoning. The judge then compared a full local answer against
    # four words and produced garbage labels, or none.
    #
    # Local tiers must stay at the length production would generate; that is
    # what is being calibrated. Only the reference needs the headroom.
    eval_reference_max_tokens: int = 1024
    # Seconds to wait between eval API calls. 0 = as fast as the provider
    # allows, which is right for a paid tier.
    #
    # Free tiers are tight — Gemini's is 20 requests/minute — and retrying
    # through a limit is slower and ruder than staying under it. 3.5s paces
    # ~17/min, comfortably inside a 20/min window.
    eval_min_call_interval_s: float = 0.0
    # The judge's prices, per million tokens. Separate from FRONTIER_PRICE_*
    # because the judge is deliberately a different model from the one being
    # judged (D11), and a different model has different rates — pricing 86
    # judge calls at the reference model's rate is simply a wrong number.
    #
    # Like FRONTIER_PRICE_*, these are placeholders. Verify them against the
    # provider's current rates before believing any total.
    eval_judge_price_in: Decimal = Decimal("2.50")
    eval_judge_price_out: Decimal = Decimal("10.00")
    # A ceiling on ONE eval run, and it ships ARMED for the same reason
    # `MAX_DAILY_SPEND_USD` does (D27): a business policy may default to off, a
    # safety limit may not.
    #
    # That ceiling lives in `BudgetLedger`, on the gateway's request path.
    # `eval/run_eval.py` calls the provider directly and never touches it — so
    # the one component that deliberately spends real money was the one with no
    # cap on it. A fixed 110-prompt dataset and a response cache make a runaway
    # unlikely, not impossible: a retry loop, a larger dataset or a `--repeat`
    # flag would each have had nothing standing in the way.
    #
    # Headroom depends on the configured prices: the same measured run is
    # $0.47 at the placeholder defaults (11x) and $0.11 at verified Gemini
    # rates (45x). Either way it should never bind in normal use; when it
    # does, something is wrong.
    eval_max_spend_usd: Decimal = Decimal("5.00")
    # OpenAI's `reasoning_effort`, passed through when non-empty.
    #
    # Empty by default and it must stay that way: the field is standard but its
    # accepted *values* are not. "none" disables thinking on gemini-2.5-flash;
    # Gemini 3 removed the off switch entirely (floor is "low"); Anthropic —
    # the default judge here — does not use the field at all. A non-empty
    # default would 400 against this project's own documented judge.
    #
    # Worth setting to "none" for a Gemini 2.5 judge. The judge emits one word
    # against a rubric, so thinking buys little, and it is what broke the run
    # at EVAL_JUDGE_MAX_TOKENS=10: hidden thinking ate the whole budget and
    # every verdict came back empty. Off, that failure mode cannot recur.
    eval_judge_reasoning_effort: str = ""
    # The reference is the gold standard every label is graded against, so
    # think twice before turning its thinking down to save pennies. Gemini 3
    # cannot disable it anyway.
    eval_reference_reasoning_effort: str = ""

    # ---- Security --------------------------------------------------------
    # Comma-separated. Empty means the gateway runs OPEN, which is only
    # acceptable on a loopback bind — startup warns loudly, and REQUIRE_AUTH
    # turns it into a hard failure.
    gateway_api_keys: str = ""
    require_auth: bool = False
    # Sliding window, per API key (per client IP when running open).
    # 0 disables. Shared through Redis; falls back to per-process state if
    # Redis is unreachable, where the effective limit multiplies by worker
    # count — worse than shared, far better than none.
    rate_limit_per_minute: int = 60
    # What one window belongs to.
    #
    #   key    — one window per API key (the default, and prior behaviour)
    #   tenant — one window per account, shared by all of that tenant's keys
    #
    # `key` is not tenant isolation: a tenant with ten keys gets ten windows
    # and can consume ten times the limit, starving everyone else. Per-tenant
    # *budgets* do not close that, because they cap spend and a cache-served
    # request spends nothing while still consuming capacity.
    #
    # Unauthenticated requests stay keyed by IP under either setting. Every
    # anonymous caller resolves to the same `anonymous` tenant, so scoping them
    # by tenant would put the entire internet in one bucket — strictly worse
    # than the per-IP behaviour it replaced.
    rate_limit_scope: Literal["key", "tenant"] = "key"
    # Request bounds. Each is a cost control and a stability control at once.
    max_prompt_chars: int = 32_000
    max_messages: int = 50
    max_output_tokens: int = 4_096
    max_body_bytes: int = 256_000

    # ---- Tenancy and budgets ---------------------------------------------
    # JSON, deliberately, rather than a delimited string: API keys are opaque
    # and may contain ':' or '=', so any inline separator risks silently
    # mis-splitting key material. Keys listed in GATEWAY_API_KEYS but not here
    # belong to the `default` tenant, so adding this to a running deployment
    # changes nothing until a named tenant appears.
    #
    #   TENANTS='{"acme": {"keys": ["sk-a","sk-b"], "daily_budget_usd": "5.00"}}'
    tenants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Applies to `default` and `anonymous`. 0 = unlimited, which is the
    # shipped default — turning on per-tenant budgets must be a deliberate act,
    # not something an upgrade does to you.
    default_daily_budget_usd: Decimal = Decimal("0")

    # The gateway's own daily ceiling, across every tenant. UNLIKE the per-
    # tenant budgets above, this ships ARMED, and the asymmetry is deliberate:
    # tenant budgets are a business policy, this is a safety limit, and a safety
    # limit that defaults to off is how you find out about it from an invoice.
    #
    # It exists because invariant 4 ("Ollama down -> escalate everything") keeps
    # the gateway answering at roughly $138/day per client at the shipped rate
    # limit. $10 is far above any demo and far below a surprise. Raise it
    # deliberately for real traffic; 0 disables it entirely.
    max_daily_spend_usd: Decimal = Decimal("10.00")

    # Whether cached answers may cross tenant boundaries.
    #
    #   global  (default) one cache for everyone. This is where the savings
    #           come from: the second tenant to ask a common question pays
    #           nothing. Right for public-knowledge traffic.
    #   tenant  the tenant is part of the cache key, so a hit can only ever be
    #           your own earlier request. Costs the cross-tenant savings and
    #           buys isolation.
    #
    # `global` also means a semantic hit can match *another tenant's* prompt,
    # so the reason string withholds the matched text in that mode — see
    # app/main.py. In `tenant` mode the match is always your own and the text
    # is shown, because it is the most useful thing in the string.
    cache_scope: Literal["global", "tenant"] = "global"

    # ---- Shadow sampling -------------------------------------------------
    # Fraction of gate-ACCEPTED local answers to re-run against the frontier,
    # out of band, to measure how often the cheap tier was wrong.
    #
    # Off by default. It costs real money — a shadow call is a real frontier
    # call for an answer nobody reads — and spending money to measure has to be
    # a deliberate act, exactly as MAX_DAILY_SPEND_USD is deliberate to raise.
    #
    # 0.01 samples 1% of accepted answers: roughly 1% of what routing everything
    # to the frontier would cost, for a continuously updated false-cheap rate.
    # Shadow spend counts against tenant budgets and the gateway ceiling, and is
    # skipped when either is exhausted.
    shadow_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Below this token-overlap ratio the two answers are counted as divergent.
    # A tripwire, not a grade — see app/shadow.py.
    shadow_divergence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # ---- Audit trail -----------------------------------------------------
    # On by default. The gateway spends money per request, so a durable record
    # of what it decided is the kind of thing you want on unless you turned it
    # off -- and a path that ships disabled is a path that rots (the calibrated
    # mode did exactly that). Telemetry is a capped stream and answers a
    # different question; see app/audit.py.
    audit_enabled: bool = True
    audit_path: str = "audit"
    audit_max_bytes: int = 50 * 1024 * 1024
    audit_keep_files: int = 5
    # Records queued but unwritten when the process is killed are lost. The
    # queue bounds that loss; a full queue drops rather than blocking the
    # request path, and the drops are counted and reported.
    audit_queue_size: int = 10_000
    # Per-record fsync: durable across a machine crash, at a disk round-trip
    # per request. Off by default; a flush per batch survives a process crash,
    # which is the failure people actually have.
    audit_fsync: bool = False

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
        """Every key the gateway accepts — the flat list plus every tenant's.

        A key configured only under `TENANTS` still authenticates, so tenants
        do not have to be listed twice. Listing a key in both places is fine
        and idempotent.
        """
        flat = {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}
        for spec in self.tenants.values():
            flat |= {str(k).strip() for k in (spec.get("keys") or []) if str(k).strip()}
        return flat

    def tenant_list(self) -> list[Tenant]:
        """Parse `TENANTS` into the registry's shape.

        Raises on a malformed entry rather than skipping it: a tenant that
        silently fails to load is a tenant with no budget, which is the exact
        outcome the setting exists to prevent.
        """
        out = []
        for name, spec in self.tenants.items():
            if not isinstance(spec, dict):
                raise ValueError(f"TENANTS[{name!r}] must be an object, got {type(spec).__name__}")
            keys = spec.get("keys") or []
            if isinstance(keys, str):
                raise ValueError(f"TENANTS[{name!r}].keys must be a list, not a string")
            try:
                budget = Decimal(str(spec.get("daily_budget_usd", "0")))
            except (ArithmeticError, ValueError) as exc:
                raise ValueError(
                    f"TENANTS[{name!r}].daily_budget_usd is not a number: "
                    f"{spec.get('daily_budget_usd')!r}"
                ) from exc
            if budget < 0:
                raise ValueError(f"TENANTS[{name!r}].daily_budget_usd is negative")
            try:
                rpm = int(spec.get("rate_limit_per_minute", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"TENANTS[{name!r}].rate_limit_per_minute is not an integer: "
                    f"{spec.get('rate_limit_per_minute')!r}"
                ) from exc
            if rpm < 0:
                raise ValueError(f"TENANTS[{name!r}].rate_limit_per_minute is negative")
            out.append(Tenant(
                name=name,
                keys=frozenset(str(k).strip() for k in keys if str(k).strip()),
                daily_budget_usd=budget,
                rate_limit_per_minute=rpm,
            ))
        return out

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
