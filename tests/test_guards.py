"""Tests for app/cache/guards.py.

The named pairs from the design docs are here as executable cases. If any of
them ever passes the guards, the gateway will serve a confidently wrong answer
for free — which is the single worst thing this project can do.
"""

from __future__ import annotations

import pytest

from app.cache.guards import (
    ENTITY_MISMATCH,
    MULTI_TURN,
    NEGATION_MISMATCH,
    NUMBER_MISMATCH,
    TOO_LONG,
    check_eligible,
    check_pair,
    extract_entities,
    extract_numbers,
    has_negation,
)

# --------------------------------------------------------------------------
# the named failures from the docs — these must never pass
# --------------------------------------------------------------------------


def test_australia_vs_austria_is_rejected():
    """The canonical example. Cosine exceeds 0.95; the answers are different
    countries."""
    r = check_pair("What is the capital of Australia?", "What is the capital of Austria?")
    assert r.passed is False
    assert r.rejected_by == ENTITY_MISMATCH


def test_convert_50_vs_70_usd_is_rejected():
    r = check_pair("convert 50 USD to EUR", "convert 70 USD to EUR")
    assert r.passed is False
    assert r.rejected_by == NUMBER_MISMATCH


def test_is_safe_vs_is_not_safe_is_rejected():
    r = check_pair("Is this API safe to use?", "Is this API not safe to use?")
    assert r.passed is False
    assert r.rejected_by == NEGATION_MISMATCH


# --------------------------------------------------------------------------
# genuine paraphrases must still pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [
    ("What is the capital of France?", "What's the capital city of France?"),
    ("Tell me France's capital.", "Name the capital of France."),
    ("How do I reverse a list in Python?", "How can I reverse a list in Python?"),
])
def test_paraphrases_pass(a, b):
    """Over-rejecting is safe but useless — the cache has to actually hit."""
    assert check_pair(a, b).passed is True


def test_whitespace_and_case_of_entities_do_not_matter():
    assert check_pair("capital of FRANCE", "capital of France").passed is True


def test_identical_prompts_pass():
    assert check_pair("same text", "same text").passed is True


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def test_numbers_are_extracted():
    assert extract_numbers("convert 50 USD to 45.5 EUR") == {"50", "45.5"}


def test_thousands_separators_normalize():
    """1,000 and 1000 are the same number and must not split the cache."""
    assert extract_numbers("1,000 items") == extract_numbers("1000 items")
    assert check_pair("Convert 1,000 USD", "Convert 1000 USD").passed is True


def test_same_numbers_different_order_still_pass():
    assert check_pair("add 3 and 5", "add 5 and 3").passed is True


def test_extra_number_is_rejected():
    assert check_pair("top 5 results", "top results").rejected_by == NUMBER_MISMATCH


# --------------------------------------------------------------------------
# entities
# --------------------------------------------------------------------------


def test_sentence_starters_are_not_entities():
    """Otherwise "What is X" vs "Tell me X" would mismatch on What/Tell."""
    assert "what" not in extract_entities("What is the capital of France?")
    assert "france" in extract_entities("What is the capital of France?")


def test_entity_comparison_is_case_insensitive():
    assert extract_entities("Paris") == extract_entities("PARIS")


def test_different_entities_are_rejected():
    assert check_pair("Who founded Google?",
        "Who founded Microsoft?").rejected_by == ENTITY_MISMATCH


def test_rejection_detail_names_the_difference():
    """A guard that fires should be debuggable from the metadata alone."""
    detail = check_pair("capital of Australia?", "capital of Austria?").detail
    assert "australia" in detail.lower() or "austria" in detail.lower()


# --------------------------------------------------------------------------
# negation, including contractions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "this is not safe", "never do this", "without a key", "except on Tuesday",
    "it isn't working", "don't do that", "cannot be done", "no results",
    "I haven't tried", "that won't work",
])
def test_negation_forms_are_detected(text):
    assert has_negation(text) is True


@pytest.mark.parametrize("text", [
    "this is safe", "always do this", "with a key", "it is working", "some results",
])
def test_non_negated_text_is_clean(text):
    assert has_negation(text) is False


def test_contraction_vs_expanded_negation_both_count():
    """"isn't" and "is not" are the same question — both negated, so they match
    each other while differing from the positive form."""
    assert check_pair("it isn't safe", "it is not safe").passed is True
    assert check_pair("it isn't safe", "it is safe").rejected_by == NEGATION_MISMATCH


# --------------------------------------------------------------------------
# eligibility — checked before embedding, so an ineligible request costs nothing
# --------------------------------------------------------------------------


def test_multi_turn_is_never_semantic_cached():
    """"What about it?" is identical in every conversation and means something
    different in each."""
    r = check_eligible(prompt="What about it?", is_multi_turn=True, max_chars=500)
    assert r.passed is False
    assert r.rejected_by == MULTI_TURN


def test_long_prompts_are_excluded():
    r = check_eligible(prompt="x" * 600, is_multi_turn=False, max_chars=500)
    assert r.passed is False
    assert r.rejected_by == TOO_LONG
    assert "600" in r.detail


def test_short_single_turn_is_eligible():
    assert check_eligible(prompt="short", is_multi_turn=False, max_chars=500).passed is True


def test_boundary_length_is_eligible():
    assert check_eligible(prompt="x" * 500, is_multi_turn=False, max_chars=500).passed is True


# --------------------------------------------------------------------------
# the documented residual risk
# --------------------------------------------------------------------------


def test_argument_order_swap_is_a_known_gap():
    """Same numbers, same entities, same negation — the guards pass, and cosine
    is high. Catching this needs parsing, not lexical sets. Pinned as a known
    limitation so it is a documented gap rather than an undiscovered bug."""
    assert check_pair("convert USD to EUR", "convert EUR to USD").passed is True


# --------------------------------------------------------------------------
# over-rejection — found by the demo, not by the original tests
# --------------------------------------------------------------------------


def test_sentence_initial_common_words_are_not_entities():
    """The demo caught this: "Answer in one word." vs "One word." rejected a
    genuine paraphrase, because both "Answer" and "One" are capitalized only by
    sentence position and were read as proper nouns."""
    stored = "What is the capital of France? Answer in one word."
    query = "What's the capital city of France? One word."
    assert extract_entities(stored) == {"france"}
    assert extract_entities(query) == {"france"}
    assert check_pair(query, stored).passed is True


def test_a_word_seen_lowercase_is_not_a_proper_noun():
    """The general rule behind the fix — the stopword list will always be
    incomplete, so this catches what it misses."""
    assert check_pair("Widget costs 5", "the widget costs 5").passed is True


def test_real_proper_nouns_survive_the_lowercase_rule():
    """The fix must not weaken the guard: a name never written lowercase is
    still an entity, so genuinely different subjects still mismatch."""
    assert check_pair("Who founded Google?", "Who founded Microsoft?").passed is False
    assert check_pair("capital of Australia?", "capital of Austria?").passed is False


@pytest.mark.parametrize("opener", [
    "Answer", "One", "Just", "Only", "Return", "Provide", "Compute", "Each", "Now",
])
def test_common_instruction_openers_are_stopwords(opener):
    assert opener.lower() not in extract_entities(f"{opener} the value for France.")
