"""The #758 pure core: keys, protection, alignment, outcomes and rendering."""

import itertools
import random
import re
import uuid

import pytest

from tests.unit.test_filler_trace import _turn
from voxint.adjudication.turns import (
    PieceRule,
    TextMapping,
    TurnPiece,
    WordAnchor,
    join_pieces,
)
from voxint.enrichment.cleanup import (
    Accepted,
    Rejected,
    SourceLine,
    Unchanged,
    cleanup_key,
    is_protected,
    render_cleaned,
    source_line,
    validate_proposal,
)
from voxint.export.filler_lists import TIER_1, effective_filler_list
from voxint.export.fillers import UnplaceableWordMarkError, delete_spans
from voxint.export.turn_filters import apply_turn_filters

_SEG = uuid.UUID(int=7)
_NO_FILLERS = effective_filler_list(keep=TIER_1)


def _anchors(text: str) -> tuple[WordAnchor, ...]:
    return tuple(
        WordAnchor(_SEG, i, i + 1, m.start(), m.end())
        for i, m in enumerate(re.finditer(r"\S+", text))
    )


def _line(text: str) -> SourceLine:
    return source_line(text, _anchors(text))


def _deleted_texts(line: SourceLine, outcome: Accepted) -> list[str]:
    return [line.words[i].text for i in sorted(outcome.deleted)]


# ------------------------------------------------------------------- keys


@pytest.mark.parametrize(
    "left, right",
    [
        ("don't", "don\u2019t"),
        ("don't", "dont"),
        ("Don't,", "dont"),
        ("twenty-one", "twentyone"),
        ("twenty\u2013one", "twenty-one"),
        ("\uff28ello", "hello"),
        ("\u201cWell\u201d", "well"),
        ("STRASSE", "strasse"),
    ],
)
def test_cleanup_key_equivalences(left: str, right: str) -> None:
    assert cleanup_key(left) == cleanup_key(right)


@pytest.mark.parametrize("token", ["...", "--", "\u2014", ",", "\"'"])
def test_punctuation_only_tokens_have_empty_keys(token: str) -> None:
    assert cleanup_key(token) == ""


# ------------------------------------------------------------- protection


@pytest.mark.parametrize(
    "lexical, previous, expected",
    [
        ("15", "about", True),
        ("3pm,", "at", True),
        ("\u0663", "at", True),
        ("\u00b2", "need", True),
        ("\u2460", "need", True),
        ("twenty-one", "about", True),
        ("Twenty-one", None, True),
        ("hundred", "a", True),
        ("first", "the", True),
        ("not", "did", True),
        ("didn't", "I", True),
        ("didn\u2019t", "I", True),
        ("Never.", "said", True),
        ("cannot", "I", True),
        ("Sarah", "told", True),
        ("Sarah,", "and", True),
        ("\u201cSarah", "told", True),
        ("Sarah", None, False),
        ("Sarah", "done.", False),
        ("Sarah", "done?\u201d", False),
        ("Sarah", "done!", False),
        ("Smith", "Mr.", True),
        ("Jones", "Dr.", True),
        ("Main", "on", True),
        ("I", "and", False),
        ("I'm", "and", False),
        ("I\u2019ve", "so", False),
        ("well", "and", False),
        ("mean,", "I", False),
        ("iPhone", "my", False),
    ],
)
def test_protection(lexical: str, previous: str | None, expected: bool) -> None:
    assert is_protected(lexical, previous) is expected


def test_source_line_judges_sentence_starts_from_preceding_word() -> None:
    line = _line("Okay. Sarah said Bob, and Sarah.")
    assert [(w.text, w.protected) for w in line.words] == [
        ("Okay.", False),
        ("Sarah", False),
        ("said", False),
        ("Bob,", True),
        ("and", False),
        ("Sarah.", True),
    ]


def test_punctuation_only_source_word_is_never_protected_or_deleted() -> None:
    line = _line("um \u2014 we go")
    assert [w.keys for w in line.words] == [("um",), (), ("we",), ("go",)]
    outcome = validate_proposal(line, "we go")
    assert outcome == Accepted(frozenset({0}))


# --------------------------------------------------------------- outcomes


@pytest.mark.parametrize(
    "source, proposed, expected",
    [
        # Substitution and insertion.
        ("we went to the store", "we went to the stores", Rejected("not_deletion")),
        ("we went to the store", "we went to the big store", Rejected("not_deletion")),
        ("we went to the store", "the store we went to", Rejected("not_deletion")),
        ("it's fine", "it is fine", Rejected("not_deletion")),
        # Protected words.
        ("I did not go", "I did go", Rejected("protected")),
        ("I didn't go there", "I go there", Rejected("protected")),
        ("we need 15 units", "we need units", Rejected("protected")),
        ("about twenty-one people", "about people", Rejected("protected")),
        ("so I told Sarah today", "so I told today", Rejected("protected")),
        ("No, no, I said no", "no I said no", Rejected("protected")),
        # Whole line, and its precedence below protection.
        ("um, uh, you know", "", Rejected("whole_line")),
        ("um, uh, you know", "...", Rejected("whole_line")),
        ("um, not", "", Rejected("protected")),
        # Unchanged.
        ("I mean, it's fine.", "I mean, it's fine.", Unchanged()),
        ("I mean, it's fine.", "i mean its fine", Unchanged()),
        ("I mean, it\u2019s fine.", "I mean, it's fine!", Unchanged()),
        ("", "", Unchanged()),
    ],
)
def test_outcome_table(source: str, proposed: str, expected: object) -> None:
    assert validate_proposal(_line(source), proposed) == expected


def test_not_deletion_takes_precedence_over_protection() -> None:
    assert validate_proposal(_line("I did not go"), "I did go home") == Rejected("not_deletion")


def test_llm_punctuation_and_casing_are_ignored() -> None:
    line = _line("I mean, I think it's, uh, fine.")
    outcome = validate_proposal(line, "i think its fine")
    assert isinstance(outcome, Accepted)
    # Earliest occurrences are kept on ties, so the second "I" is the one deleted.
    assert _deleted_texts(line, outcome) == ["mean,", "I", "uh,"]
    assert render_cleaned(line, outcome.deleted) == "I think it's, fine."


def test_protection_preferring_alignment_keeps_the_protected_occurrence() -> None:
    line = _line("Mark said, Mark it down")
    outcome = validate_proposal(line, "Mark it down")
    assert isinstance(outcome, Accepted)
    assert _deleted_texts(line, outcome) == ["Mark", "said,"]
    assert render_cleaned(line, outcome.deleted) == "Mark it down"


def test_ties_keep_the_earliest_occurrences() -> None:
    line = _line("you know I know you know")
    outcome = validate_proposal(line, "I know you know")
    assert isinstance(outcome, Accepted)
    assert sorted(outcome.deleted) == [0, 1]
    repeat = _line("the the cat")
    tie = validate_proposal(repeat, "the cat")
    assert tie == Accepted(frozenset({1}))


def test_unanchored_line() -> None:
    line = source_line("I mean, it was fine", ())
    assert validate_proposal(line, "it was fine") == Rejected("unanchored")
    assert validate_proposal(line, "I mean it was fine") == Unchanged()


# ---------------------------------------------------------------- rendering


@pytest.mark.parametrize(
    "source, proposed, expected",
    [
        ("Um, we start now.", "we start now", "We start now."),
        ("We start, um.", "we start", "We start."),
        ("I mean, you know, it works.", "it works", "It works."),
        ("So (basically) it works", "so it works", "So it works"),
        ("It works, I guess.", "it works", "It works."),
        ("Okay. Like, we go.", "okay we go", "Okay. We go."),
    ],
)
def test_render_follows_omit_rules(source: str, proposed: str, expected: str) -> None:
    line = _line(source)
    outcome = validate_proposal(line, proposed)
    assert isinstance(outcome, Accepted)
    assert render_cleaned(line, outcome.deleted) == expected


@pytest.mark.parametrize(
    "source, drop",
    [
        ("Um, we start now.", [0]),
        ("I mean, you know, it works.", [0, 1, 2, 3]),
        ("So, like, the the cat sat.", [1, 2]),
        ("\u201cUm\u201d, she said.", [0]),
        ("We were, uh, done. Uh, next.", [2, 4]),
    ],
)
def test_render_matches_turns_omit_path(source: str, drop: list[int]) -> None:
    """The stored text equals the single-turn export with the same omits."""
    line = _line(source)
    piece = TurnPiece(
        source, 0.0, 1.0, True, True, PieceRule.WORD_LEVEL, TextMapping.VERBATIM,
        _anchors(source),
    )
    marks = {line.words[i].identity: "omit" for i in drop}
    filtered = apply_turn_filters(
        [_turn(piece)], fillers=_NO_FILLERS, drop_repeats=False, marks=marks,  # type: ignore[arg-type]
    )
    assert filtered.fillers_removed == 0
    assert filtered.omitted == len(drop)
    rendered = render_cleaned(line, frozenset(drop))
    assert [join_pieces(t.pieces) for t in filtered.turns] == [rendered]


def test_marks_apply_with_an_empty_filler_list() -> None:
    piece = TurnPiece(
        "um, I mean, the cat.", 0.0, 1.0, True, True, PieceRule.WORD_LEVEL,
        TextMapping.VERBATIM, _anchors("um, I mean, the cat."),
    )
    result = apply_turn_filters(
        [_turn(piece)], fillers=_NO_FILLERS, drop_repeats=False,
        marks={(_SEG, 1, 2): "omit", (_SEG, 2, 3): "omit"},
    )
    assert [join_pieces(t.pieces) for t in result.turns] == ["um, the cat."]
    assert (result.fillers_removed, result.omitted, result.kept) == (0, 2, 0)


def test_delete_spans_refuses_an_unplaceable_identity() -> None:
    with pytest.raises(UnplaceableWordMarkError):
        delete_spans("we go", _anchors("we go"), [(_SEG, 5, 6)])


def test_delete_spans_with_nothing_deleted_is_verbatim() -> None:
    assert delete_spans("  we, go  ", _anchors("  we, go  "), []) == "  we, go  "


# --------------------------------------------------------------- properties

_VOCAB = (
    "um", "uh", "you", "know", "I", "mean", "like", "so", "well", "the", "the", "cat",
    "sat", "on", "mat", "it's", "don't", "not", "no", "Sarah", "Bob", "15", "twenty-one",
    "first", "okay,", "right.", "and,", "then?", "yes!", "\u2014", "...",
)


def _random_line(rng: random.Random) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(1, 12)))


def _random_proposal(rng: random.Random, line: SourceLine) -> str:
    kept = [w.text for w in line.words if rng.random() < 0.7]
    if rng.random() < 0.15:
        kept.insert(rng.randint(0, len(kept)), rng.choice(_VOCAB))
    if rng.random() < 0.1 and kept:
        rng.shuffle(kept)
    return " ".join(kept)


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(key == item for item in it) for key in needle)


@pytest.mark.parametrize("seed", range(40))
def test_accepted_outcomes_hold_their_invariants(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(50):
        line = _line(_random_line(rng))
        proposed = _random_proposal(rng, line)
        outcome = validate_proposal(line, proposed)
        if not isinstance(outcome, Accepted):
            continue
        assert outcome.deleted
        assert all(line.words[i].keys and not line.words[i].protected for i in outcome.deleted)
        kept = [
            key for i, w in enumerate(line.words) if i not in outcome.deleted for key in w.keys
        ]
        assert kept, "an accepted line keeps at least one word"
        proposal = [k for k in map(cleanup_key, proposed.split()) if k]
        assert kept == proposal
        source = [key for w in line.words for key in w.keys]
        assert _is_subsequence(kept, source)
        rendered = render_cleaned(line, outcome.deleted)
        assert [k for k in map(cleanup_key, rendered.split()) if k] == kept


def _oracle(line: SourceLine, proposed: str) -> object:
    """Brute force over every deletion set, independent of the DP."""
    proposal = [k for k in map(cleanup_key, proposed.split()) if k]
    keyed = [i for i, w in enumerate(line.words) if w.keys]

    def kept_keys(deleted: frozenset[int]) -> list[str]:
        return [key for i in keyed if i not in deleted for key in line.words[i].keys]

    subsets = [
        frozenset(combo)
        for size in range(len(keyed) + 1)
        for combo in itertools.combinations(keyed, size)
    ]
    embeddings = [d for d in subsets if kept_keys(d) == proposal]
    if not embeddings:
        return Rejected("not_deletion")
    allowed = [d for d in embeddings if not any(line.words[i].protected for i in d)]
    if not allowed:
        return Rejected("protected")
    # Earliest occurrences kept: the lexicographically smallest kept index list.
    best = min(allowed, key=lambda d: [i for i in keyed if i not in d])
    if not best:
        return Unchanged()
    if best == frozenset(keyed):
        return Rejected("whole_line")
    return Accepted(best)


@pytest.mark.parametrize("seed", range(30))
def test_outcomes_match_a_brute_force_oracle(seed: int) -> None:
    rng = random.Random(seed)
    seen: set[str] = set()
    for _ in range(60):
        line = _line(" ".join(rng.choice(_VOCAB) for _ in range(rng.randint(1, 9))))
        proposed = _random_proposal(rng, line)
        expected = _oracle(line, proposed)
        assert validate_proposal(line, proposed) == expected, (line.text, proposed)
        seen.add(type(expected).__name__ + getattr(expected, "reason", ""))
    assert "Accepted" in seen


def _multiword(text: str, units: list[str]) -> tuple[WordAnchor, ...]:
    """Anchors over whole units of ``text``, as #757 builds for joined tokens."""
    anchors, position = [], 0
    for index, unit in enumerate(units):
        start = text.index(unit, position)
        anchors.append(WordAnchor(_SEG, index, index + 1, start, start + len(unit)))
        position = start + len(unit)
    return tuple(anchors)


def test_multiword_unit_is_protected_by_any_of_its_words() -> None:
    text = "we will not go"
    line = source_line(text, _multiword(text, ["we", "will not", "go"]))
    assert [w.keys for w in line.words] == [("we",), ("will", "not"), ("go",)]
    assert line.words[1].protected
    assert validate_proposal(line, "we go") == Rejected("protected")
    assert validate_proposal(line, "we will not go") == Unchanged()


def test_multiword_unit_is_kept_or_deleted_whole() -> None:
    text = "so you know we go"
    line = source_line(text, _multiword(text, ["so", "you know", "we", "go"]))
    assert validate_proposal(line, "so we go") == Accepted(frozenset({1}))
    assert render_cleaned(line, frozenset({1})) == "so we go"
    assert validate_proposal(line, "so you we go") == Rejected("not_deletion")
    assert validate_proposal(line, "so know we go") == Rejected("not_deletion")


def test_superscript_digit_cannot_be_deleted() -> None:
    assert validate_proposal(_line("we need \u00b2 units"), "we need units") == (
        Rejected("protected")
    )


def test_delete_spans_with_nothing_deleted_keeps_whitespace_only_text() -> None:
    assert delete_spans("   ", (), []) == "   "
