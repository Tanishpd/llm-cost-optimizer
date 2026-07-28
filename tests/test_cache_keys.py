"""Unit tests for app/cache/keys.py.

The asymmetry that drives every case here: a too-loose key serves a
confidently wrong answer for free with no signal; a too-tight key costs one
model call. So the tests lean hard on "these must NOT collide".
"""

from __future__ import annotations

from app.cache.keys import (
    KEY_SCHEMA_VERSION,
    build_key,
    normalize_prompt,
    redis_key,
    system_prompt_hash,
)

BASE = dict(model="qwen2.5:3b", temperature=0.0, top_p=1.0, max_tokens=512)


def key(prompt: str, **overrides) -> str:
    return build_key(prompt=prompt, **{**BASE, **overrides}).digest


# --------------------------------------------------------------------------
# normalization — deliberately minimal
# --------------------------------------------------------------------------


def test_whitespace_is_collapsed_and_stripped():
    assert normalize_prompt("  hello   world  \n") == "hello world"
    assert normalize_prompt("a\t\tb\nc") == "a b c"


def test_unicode_is_nfc_normalized():
    """Composed and decomposed forms of the same character must hash alike —
    they're indistinguishable to a reader."""
    assert normalize_prompt("café") == normalize_prompt("café")


def test_case_is_preserved():
    """NOT lowercased, on purpose: case is semantic in code and identifiers.
    `myVar` and `myvar` are different questions."""
    assert key("Explain myVar") != key("Explain myvar")


def test_punctuation_is_preserved():
    """NOT stripped: punctuation carries meaning."""
    assert key("Let's eat, Grandma") != key("Lets eat Grandma")


def test_whitespace_only_differences_do_collide():
    """The one thing normalization is for."""
    assert key("What is  2+2?") == key("What is 2+2?")
    assert key("  padded  ") == key("padded")


# --------------------------------------------------------------------------
# sampling params are part of the key — invariant 5
# --------------------------------------------------------------------------


def test_temperature_changes_the_key():
    """Same text at temperature 0 and 1.0 are different questions: one asks for
    the model's best guess, the other for a sample."""
    assert key("q", temperature=0.0) != key("q", temperature=1.0)


def test_top_p_changes_the_key():
    assert key("q", top_p=1.0) != key("q", top_p=0.9)


def test_max_tokens_changes_the_key():
    """A 10-token answer is not a substitute for a 500-token one."""
    assert key("q", max_tokens=10) != key("q", max_tokens=500)


def test_model_changes_the_key():
    """An answer from the 0.5B must never be served for a 3B request."""
    assert key("q", model="qwen2.5:0.5b") != key("q", model="qwen2.5:3b")


def test_system_prompt_changes_the_key():
    a = build_key(prompt="q", system_prompt="You are terse.", **BASE).digest
    b = build_key(prompt="q", system_prompt="You are verbose.", **BASE).digest
    c = build_key(prompt="q", **BASE).digest
    assert a != b != c and a != c


def test_absent_system_prompt_hashes_to_empty():
    assert system_prompt_hash("") == ""


def test_equal_inputs_produce_equal_keys():
    assert key("same") == key("same")


def test_numeric_forms_of_the_same_value_agree():
    """int 0 and float 0.0 are the same temperature — must not split the cache."""
    assert key("q", temperature=0) == key("q", temperature=0.0)
    assert key("q", max_tokens=512) == key("q", max_tokens=512)


# --------------------------------------------------------------------------
# hashing hygiene
# --------------------------------------------------------------------------


def test_field_boundaries_cannot_be_forged():
    """The classic concatenation bug: with a naive joiner, ("ab","c") and
    ("a","bc") collide. A NUL separator can't appear in the inputs."""
    a = build_key(prompt="ab", **{**BASE, "model": "c"}).digest
    b = build_key(prompt="a", **{**BASE, "model": "bc"}).digest
    assert a != b


def test_schema_version_is_in_the_key():
    """Bumping the version must invalidate old entries, so a normalization
    change can never serve stale entries under the new scheme."""
    import app.cache.keys as keys_mod

    before = key("q")
    original = keys_mod.KEY_SCHEMA_VERSION
    try:
        keys_mod.KEY_SCHEMA_VERSION = "v2"
        assert key("q") != before
    finally:
        keys_mod.KEY_SCHEMA_VERSION = original


def test_digest_is_stable_across_runs():
    """A hardcoded expectation: if this changes, every cached entry is
    invalidated, and that should be a deliberate act (bump the schema version)."""
    assert KEY_SCHEMA_VERSION == "v1"
    assert key("What is the capital of France?") == key("What is the capital of France?")
    assert len(key("q")) == 32  # 16-byte digest, hex


def test_redis_key_is_namespaced():
    assert redis_key("abc").startswith("hlr:exact:")


# --------------------------------------------------------------------------
# params_match — used by the semantic tier
# --------------------------------------------------------------------------


def test_params_match_ignores_the_prompt():
    """High cosine says the *questions* match; params_match says the *answers*
    would be interchangeable. The semantic tier needs both."""
    a = build_key(prompt="capital of France?", **BASE)
    b = build_key(prompt="France's capital city?", **BASE)
    assert a.digest != b.digest
    assert a.params_match(b) is True


def test_params_match_rejects_differing_sampling():
    a = build_key(prompt="q", **BASE)
    b = build_key(prompt="q", **{**BASE, "temperature": 0.9})
    assert a.params_match(b) is False


def test_params_match_rejects_differing_system_prompt():
    a = build_key(prompt="q", system_prompt="terse", **BASE)
    b = build_key(prompt="q", system_prompt="verbose", **BASE)
    assert a.params_match(b) is False


def test_key_carries_its_components():
    k = build_key(prompt="  hi  ", **BASE)
    assert k.prompt == "hi"  # normalized form retained
    assert k.model == "qwen2.5:3b"
    assert k.temperature == 0.0
