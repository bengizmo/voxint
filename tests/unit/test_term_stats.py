"""Unit tests for the TF-IDF term statistics module (issue #334)."""

from __future__ import annotations

import uuid

from voxint.api.term_stats import TermStat, compute_tfidf, source_hash, tokenize


def test_tokenize_strips_stopwords_and_short_tokens() -> None:
    tokens = tokenize("The quick brown fox jumps over a lazy dog")
    assert "the" not in tokens
    assert "a" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens


def test_tokenize_strips_spoken_fillers() -> None:
    tokens = tokenize("um yeah so basically I uh literally went downtown")
    assert "um" not in tokens
    assert "yeah" not in tokens
    assert "basically" not in tokens
    assert "uh" not in tokens
    assert "literally" not in tokens
    assert "downtown" in tokens


def test_tokenize_lowercases() -> None:
    tokens = tokenize("HELLO World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_ignores_single_char_tokens() -> None:
    tokens = tokenize("I a x am building")
    assert "building" in tokens
    assert len([t for t in tokens if len(t) == 1]) == 0


def test_compute_tfidf_empty_documents() -> None:
    result = compute_tfidf([])
    assert result == []


def test_compute_tfidf_single_document() -> None:
    doc_id = uuid.uuid4()
    result = compute_tfidf([(doc_id, "energy efficiency heating cooling energy")])
    terms = {s.term: s for s in result}
    assert "energy" in terms
    assert terms["energy"].count == 2
    assert terms["energy"].doc_count == 1
    assert terms["energy"].tfidf > 0


def test_compute_tfidf_multiple_documents() -> None:
    docs = [
        (uuid.uuid4(), "energy efficiency heating cooling systems"),
        (uuid.uuid4(), "solar panels energy production renewable"),
        (uuid.uuid4(), "building insulation thermal performance heating"),
    ]
    result = compute_tfidf(docs)
    terms = {s.term: s for s in result}
    assert "energy" in terms
    assert terms["energy"].doc_count == 2
    assert "insulation" in terms
    assert terms["insulation"].doc_count == 1
    # TF-IDF: a term in fewer docs gets higher IDF
    assert terms["insulation"].tfidf > terms["energy"].tfidf


def test_compute_tfidf_respects_top_n() -> None:
    doc_id = uuid.uuid4()
    words = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima",
        "mike", "november", "oscar", "papa", "quebec", "romeo",
        "sierra", "tango",
    ]
    text = " ".join(words)
    result = compute_tfidf([(doc_id, text)], top_n=10)
    assert len(result) == 10


def test_compute_tfidf_returns_term_stat_dataclass() -> None:
    result = compute_tfidf([(uuid.uuid4(), "hello world hello")])
    assert len(result) > 0
    assert isinstance(result[0], TermStat)
    assert isinstance(result[0].term, str)
    assert isinstance(result[0].count, int)
    assert isinstance(result[0].doc_count, int)
    assert isinstance(result[0].tfidf, float)


def test_source_hash_deterministic() -> None:
    pairs = [("abc", "2026-08-30"), ("def", "2026-08-29")]
    h1 = source_hash(pairs)
    h2 = source_hash(pairs)
    assert h1 == h2
    assert len(h1) == 64


def test_source_hash_order_independent() -> None:
    h1 = source_hash([("abc", "ts1"), ("def", "ts2")])
    h2 = source_hash([("def", "ts2"), ("abc", "ts1")])
    assert h1 == h2


def test_source_hash_changes_on_different_input() -> None:
    h1 = source_hash([("abc", "ts1")])
    h2 = source_hash([("abc", "ts2")])
    assert h1 != h2


def test_source_hash_empty() -> None:
    h = source_hash([])
    assert len(h) == 64


def test_artifact_lock_key_distinct_per_scope() -> None:
    from voxint.api.explore_query import _artifact_lock_key

    project_a = uuid.uuid4()
    project_b = uuid.uuid4()
    keys = [
        _artifact_lock_key("all", None),
        _artifact_lock_key("project", project_a),
        _artifact_lock_key("project", project_b),
    ]
    assert len(set(keys)) == len(keys), "each scope must get its own advisory lock"
    for key in keys:
        assert 0 <= key <= 0x7FFFFFFF
    # Deterministic: the same scope always maps to the same key.
    assert _artifact_lock_key("project", project_a) == keys[1]
