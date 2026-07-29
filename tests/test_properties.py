"""Property-based tests over the pure functions.

The example-based tests elsewhere assert what these functions do for inputs
someone thought of. These assert what must be true for inputs nobody thought
of — which is the right instrument for the parts of this system that are
mathematical claims rather than behaviour.

The central one is `geo_mean`'s **length invariance**. The whole design rests
on it (D3): sequence probability decays exponentially with length, so any fixed
threshold against it measures length rather than certainty. That claim has been
supported by two hand-picked examples and a docstring. Here it is quantified
over generated inputs, alongside the counter-claim that `sequence_probability`
does *not* have the property — because a counter-example is only useful if it
actually behaves the way it is said to.

Mutation testing (M19) and this are complementary and neither replaces the
other: mutation asks whether the existing assertions would catch a change,
property testing asks whether the assertions cover the right inputs.
"""

from __future__ import annotations

import math
import re

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.cache.guards import check_eligible, check_pair
from app.cache.keys import build_key, normalize_prompt
from app.confidence import geo_mean, sequence_probability
from app.costs import TokenUsage, micro_to_usd, price_micro_usd, usd_to_micro
from app.prometheus import Registry, escape_label_value
from app.shadow import similarity
from app.streaming import split_content

# Token logprobs, i.e. log(p) for p in (0, 1]. Bounded below at log(1e-9)
# because Ollama does not emit anything smaller and denormals would test
# floating point rather than this code.
LOGPROBS = st.floats(min_value=math.log(1e-9), max_value=0.0,
                     allow_nan=False, allow_infinity=False)
PROBS = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
TEXT = st.text(max_size=200)


# --------------------------------------------------------------------------
# geo_mean — the claim the design rests on
# --------------------------------------------------------------------------


@given(p=PROBS, n=st.integers(min_value=1, max_value=500))
def test_geo_mean_is_length_invariant(p, n):
    """N tokens all at probability p score p, for every N.

    This is D3 in one line, and it is why a fixed threshold against `geo_mean`
    means the same thing for a 3-token answer and a 300-token one.
    """
    assert geo_mean([math.log(p)] * n) == pytest.approx(p, rel=1e-9)


@given(logprobs=st.lists(LOGPROBS, min_size=1, max_size=200),
       repeat=st.integers(min_value=2, max_value=5))
def test_geo_mean_ignores_repetition(logprobs, repeat):
    """Repeating an answer's token sequence does not change its score.

    A stronger statement than the constant-probability case: the *distribution*
    is what matters, not how many tokens carry it.
    """
    assert geo_mean(logprobs * repeat) == pytest.approx(geo_mean(logprobs), rel=1e-9)


@given(logprobs=st.lists(LOGPROBS, min_size=1, max_size=300))
def test_geo_mean_is_always_a_probability(logprobs):
    """Range (0, 1]. A confidence outside that would be compared against a
    threshold that means nothing."""
    result = geo_mean(logprobs)
    assert 0.0 < result <= 1.0 + 1e-12


@given(a=st.lists(LOGPROBS, min_size=1, max_size=100),
       b=st.lists(LOGPROBS, min_size=1, max_size=100))
def test_geo_mean_is_monotone_in_confidence(a, b):
    """A uniformly more-confident answer never scores lower.

    The gate's entire meaning depends on this: if a better answer could score
    worse, the threshold would be sorting on something other than confidence.
    """
    assume(len(a) == len(b))
    assume(all(x <= y for x, y in zip(a, b, strict=True)))
    assert geo_mean(a) <= geo_mean(b) + 1e-12


@given(p=PROBS, n=st.integers(min_value=30, max_value=200))
def test_sequence_probability_is_NOT_length_invariant(p, n):
    """The counter-example, asserted rather than assumed.

    `sequence_probability` exists in the codebase only to demonstrate why it is
    the wrong metric (D3). A counter-example whose behaviour drifted would
    quietly stop making the point it exists to make.
    """
    assume(p < 0.99)                       # p=1.0 is length-invariant trivially
    short = sequence_probability([math.log(p)] * 3)
    long = sequence_probability([math.log(p)] * n)
    assert long < short, "sequence probability failed to decay with length"
    assert geo_mean([math.log(p)] * 3) == pytest.approx(
        geo_mean([math.log(p)] * n), rel=1e-9), "…while geo_mean did not"


# --------------------------------------------------------------------------
# money — integer micro-USD must not drift
# --------------------------------------------------------------------------


@given(micro=st.integers(min_value=0, max_value=10**12))
def test_micro_usd_round_trips_exactly(micro):
    """`costs.py` accumulates in integer micro-USD precisely so summation is
    exact (D13). A lossy conversion at the edge would undo that."""
    assert usd_to_micro(micro_to_usd(micro)) == micro


@given(amounts=st.lists(st.integers(min_value=0, max_value=10**7),
                        min_size=1, max_size=300))
def test_summing_costs_never_drifts(amounts):
    """The bug the load test found: per-request rounding summed to a 75
    micro-USD gap over 200 requests. Integer accumulation cannot drift, and
    this asserts the property rather than the one instance."""
    total = sum(amounts)
    assert usd_to_micro(micro_to_usd(total)) == total


@given(tin=st.integers(min_value=0, max_value=10**6),
       tout=st.integers(min_value=0, max_value=10**6))
def test_price_is_monotone_in_tokens(tin, tout):
    from decimal import Decimal
    a = price_micro_usd(TokenUsage(tin, tout), Decimal("2.50"), Decimal("10.00"))
    b = price_micro_usd(TokenUsage(tin + 1, tout), Decimal("2.50"), Decimal("10.00"))
    assert b >= a, "one more input token cost less"


# --------------------------------------------------------------------------
# cache keys — same question in, same key out
# --------------------------------------------------------------------------


@given(prompt=TEXT)
def test_normalization_is_idempotent(prompt):
    """Normalizing twice must equal normalizing once, or the key for a prompt
    depends on how many times it has been through the pipeline."""
    once = normalize_prompt(prompt)
    assert normalize_prompt(once) == once


@given(prompt=TEXT, model=st.text(min_size=1, max_size=40),
       temp=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
def test_the_same_request_always_hashes_the_same(prompt, model, temp):
    a = build_key(prompt=prompt, model=model, temperature=temp, top_p=1.0, max_tokens=10)
    b = build_key(prompt=prompt, model=model, temperature=temp, top_p=1.0, max_tokens=10)
    assert a.digest == b.digest


@given(prompt=TEXT, model=st.text(min_size=1, max_size=40),
       t1=st.floats(min_value=0.0, max_value=2.0, allow_nan=False),
       t2=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
def test_different_sampling_params_never_share_a_key(prompt, model, t1, t2):
    """Invariant 5. The same text at temperature 0 and 1 are different
    questions, and serving one for the other is a wrong answer."""
    assume(t1 != t2)
    a = build_key(prompt=prompt, model=model, temperature=t1, top_p=1.0, max_tokens=10)
    b = build_key(prompt=prompt, model=model, temperature=t2, top_p=1.0, max_tokens=10)
    assert a.digest != b.digest


@given(prompt=TEXT, m1=st.text(min_size=1, max_size=40), m2=st.text(min_size=1, max_size=40))
def test_different_models_never_share_a_key(prompt, m1, m2):
    """An answer from the 0.5B must never be served for a 3B request."""
    assume(m1 != m2)
    a = build_key(prompt=prompt, model=m1, temperature=0.0, top_p=1.0, max_tokens=10)
    b = build_key(prompt=prompt, model=m2, temperature=0.0, top_p=1.0, max_tokens=10)
    assert a.digest != b.digest


@given(prompt=TEXT, scope=st.text(min_size=1, max_size=20))
def test_tenant_scope_always_separates_keys(prompt, scope):
    """D28: under CACHE_SCOPE=tenant a hit can only ever be your own."""
    shared = build_key(prompt=prompt, model="m", temperature=0.0, top_p=1.0, max_tokens=10)
    scoped = build_key(prompt=prompt, model="m", temperature=0.0, top_p=1.0,
                       max_tokens=10, scope=scope)
    assert shared.digest != scoped.digest


# --------------------------------------------------------------------------
# semantic guards — a wrong hit is worse than a miss
# --------------------------------------------------------------------------


@given(prompt=st.text(min_size=1, max_size=150))
def test_a_prompt_always_matches_itself(prompt):
    """The guards may be over-conservative (D15) but must never reject an
    identical pair — that would make the semantic tier reject its own exact
    hits, which is not conservatism, it is breakage."""
    assert check_pair(prompt, prompt).passed


@given(a=st.text(min_size=1, max_size=120), b=st.text(min_size=1, max_size=120))
def test_pair_checking_is_symmetric(a, b):
    """Whether two prompts are interchangeable cannot depend on which arrived
    first — the same two prompts would otherwise hit or miss depending on
    cache insertion order."""
    assert check_pair(a, b).passed == check_pair(b, a).passed


@given(prompt=st.text(min_size=1), limit=st.integers(min_value=1, max_value=500))
def test_over_long_prompts_are_always_ineligible(prompt, limit):
    assume(len(prompt) > limit)
    assert not check_eligible(prompt=prompt, is_multi_turn=False, max_chars=limit).passed


@given(prompt=st.text(min_size=1, max_size=100))
def test_multi_turn_is_never_eligible(prompt):
    """Regardless of anything else about the prompt."""
    assert not check_eligible(prompt=prompt, is_multi_turn=True, max_chars=10_000).passed


# --------------------------------------------------------------------------
# streaming — the deltas must reassemble
# --------------------------------------------------------------------------


@given(text=st.text(max_size=600), chunk=st.integers(min_value=0, max_value=80))
@settings(max_examples=300)
def test_chunking_is_always_lossless(text, chunk):
    """The parametrized version of this covers eleven strings someone chose.
    This covers whatever hypothesis can think of, including the whitespace and
    unicode edge cases that are exactly where a chunker goes wrong."""
    assert "".join(split_content(text, chunk)) == text


@given(text=st.text(min_size=1, max_size=400), chunk=st.integers(min_value=1, max_value=50))
def test_chunking_never_produces_an_empty_piece(text, chunk):
    """An empty content delta is a wasted SSE frame and confuses clients that
    treat one as end-of-stream."""
    assert all(piece for piece in split_content(text, chunk))


# --------------------------------------------------------------------------
# similarity — a tripwire that must at least be well-formed
# --------------------------------------------------------------------------


@given(a=TEXT, b=TEXT)
def test_similarity_is_bounded_and_symmetric(a, b):
    s = similarity(a, b)
    assert 0.0 <= s <= 1.0
    assert s == similarity(b, a)


@given(text=st.text(min_size=1, max_size=200))
def test_similarity_with_itself_is_one(text):
    assume(re.search(r"[a-z0-9']", text.lower()))     # must contain a token
    assert similarity(text, text) == 1.0


# --------------------------------------------------------------------------
# prometheus — output must stay parseable whatever the labels contain
# --------------------------------------------------------------------------


def _parse_label_line(line: str) -> tuple[str, str]:
    """Parse `name{label="value"} number`, honouring the escapes.

    A real scan rather than a regex: an earlier version of this test counted
    quotes not preceded by a backslash, which reports `label="\\\\"` — a
    perfectly valid value containing one backslash — as unbalanced. Property
    testing found that flaw in the test itself, which is the same instrument
    doing the same job one level up.
    """
    prefix, rest = line.split('{', 1)
    assert rest.startswith('label="'), rest
    body = rest[len('label="'):]

    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            assert i + 1 < len(body), "line ends inside an escape"
            nxt = body[i + 1]
            out.append({"n": "\n", '"': '"', "\\": "\\"}[nxt])
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    else:
        raise AssertionError("no closing quote")

    tail = body[i:]
    assert tail.startswith('"} '), tail
    return "".join(out), tail[3:]


@given(value=st.text(max_size=100))
def test_escaped_labels_never_break_the_line_format(value):
    """Tier names come from config and tenant names from operator JSON —
    neither is guaranteed tame, and an unescaped quote or a stray control
    character silently corrupts a series.

    Found here on the first run: a lone `\\r` split the sample across two
    lines. Then `\\x85` (NEL) did the same after the first fix — which is the
    argument for handling the whole class of line terminators rather than the
    character that happened to fail.
    """
    r = Registry()
    r.add("thing", 1, help="x", labels={"label": value})
    rendered = [ln for ln in r.render().splitlines() if not ln.startswith("#")]

    assert len(rendered) == 1, "a label value split the sample across lines"
    parsed, number = _parse_label_line(rendered[0])
    assert number == "1"
    # The contract, stated directly: what comes back out is the input with
    # control characters replaced, and nothing else changed.
    #
    # Not a chain of `.replace()` calls to un-escape — that is what the first
    # version did, and hypothesis broke it with the two-character string
    # `\\n` (backslash, n): un-escaping `\\n` before `\\\\` turns it into a
    # backslash and a real newline. Sequential replace on escape sequences is
    # order-dependent and wrong; `_parse_label_line` scans instead.
    expected = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f\u2028\u2029]", "\ufffd", value)
    assert parsed == expected


@given(value=st.text(max_size=60,
                     alphabet=st.characters(blacklist_categories=("Cc",))))
def test_escaping_is_reversible_for_printable_text(value):
    """Escaping must not lose information — a label that renders identically
    for two different inputs is a label that lies.

    Control characters are excluded because they are deliberately *replaced*
    rather than escaped: the format defines no sequence for them, so emitting
    one raw is how a sample becomes two lines. That trade is asserted below.
    """
    escaped = escape_label_value(value)
    # Scan, do not chain `.replace()`. Hypothesis found this on a later run:
    # the two-character value `\` + `n` escapes to `\\n`, and a naive
    # `.replace("\\n", "\n")` matches the *second* backslash with the `n` and
    # returns backslash + a real newline. Escaping was correct; the test's
    # inverse was not — the same flaw this file already documents for
    # `_parse_label_line` above, reintroduced twenty lines below it.
    restored, i = [], 0
    while i < len(escaped):
        if escaped[i] == "\\":
            assert i + 1 < len(escaped), "value ends inside an escape"
            restored.append({"n": "\n", '"': '"', "\\": "\\"}[escaped[i + 1]])
            i += 2
            continue
        restored.append(escaped[i])
        i += 1
    assert "".join(restored) == value


@given(value=st.text(min_size=1, max_size=40,
                     alphabet=st.characters(whitelist_categories=("Cc",))))
def test_control_characters_are_replaced_not_emitted(value):
    """Found by this file on its first run: a lone `\r` split a sample across
    two lines. Replacement is lossy on purpose — a control character in a
    metric label is an accident or an injection, never data worth keeping."""
    escaped = escape_label_value(value)
    assert "\r" not in escaped
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f]", escaped)
    # Line feed keeps its defined escape rather than being replaced.
    assert escape_label_value("a\nb") == "a\\nb"
