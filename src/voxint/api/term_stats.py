"""TF-IDF term statistics for corpus visualization (issue #334).

Pure computation: tokenization, stopword filtering, and TF-IDF scoring over
a list of documents. No DB or framework imports.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z]{2,}")

STOPWORDS: frozenset[str] = frozenset(
    {
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "even",
        "few",
        "for",
        "from",
        "further",
        "get",
        "go",
        "going",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "ll",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "really",
        "right",
        "same",
        "she",
        "should",
        "so",
        "some",
        "still",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "us",
        "ve",
        "very",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        # Contractions without the apostrophe (Whisper often strips them)
        "ain",
        "aren",
        "couldn",
        "didn",
        "doesn",
        "don",
        "hadn",
        "hasn",
        "haven",
        "isn",
        "shouldn",
        "wasn",
        "weren",
        "won",
        "wouldn",
        # Spoken-content fillers (ASR transcripts carry these heavily)
        "uh",
        "um",
        "hm",
        "hmm",
        "mm",
        "mhm",
        "oh",
        "ah",
        "yeah",
        "yep",
        "yup",
        "nah",
        "nope",
        "ok",
        "okay",
        "like",
        "know",
        "mean",
        "actually",
        "basically",
        "literally",
        "gonna",
        "gotta",
        "wanna",
        "kinda",
        "sorta",
        "thing",
        "things",
        "stuff",
        "lot",
        "way",
        "kind",
        "sort",
        "bit",
        "something",
        "anything",
        "everything",
        "nothing",
        "someone",
        "anyone",
        "everyone",
        "one",
        "two",
        "see",
        "say",
        "said",
        "says",
        "think",
        "thought",
        "make",
        "made",
        "take",
        "took",
        "come",
        "came",
        "give",
        "gave",
        "tell",
        "told",
        "let",
        "put",
        "look",
        "looking",
        "need",
        "want",
        "use",
        "used",
        "try",
        "keep",
        "went",
    }
)


@dataclass
class TermStat:
    term: str
    count: int
    doc_count: int
    tfidf: float


def tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens (length >= 2), stopwords removed."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def compute_tfidf(
    documents: list[tuple[uuid.UUID, str]],
    *,
    top_n: int = 200,
) -> list[TermStat]:
    """Compute TF-IDF term statistics over (doc_id, text) pairs.

    Each document is one pipeline run's concatenated effective text.
    Returns the top_n terms ranked by average TF-IDF score.
    """
    if not documents:
        return []

    n_docs = len(documents)
    doc_tfs: list[Counter[str]] = []
    for _, text in documents:
        doc_tfs.append(Counter(tokenize(text)))

    df: Counter[str] = Counter()
    corpus_counts: Counter[str] = Counter()
    for tf in doc_tfs:
        for term in tf:
            df[term] += 1
        corpus_counts.update(tf)

    doc_lengths = [sum(tf.values()) or 1 for tf in doc_tfs]

    tfidf_avg: dict[str, float] = {}
    for term, doc_freq in df.items():
        idf = math.log(1.0 + n_docs / doc_freq)
        total = 0.0
        for i, tf in enumerate(doc_tfs):
            count = tf.get(term, 0)
            if count:
                total += (count / doc_lengths[i]) * idf
        tfidf_avg[term] = total / doc_freq

    ranked = sorted(tfidf_avg.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    return [
        TermStat(
            term=term,
            count=corpus_counts[term],
            doc_count=df[term],
            tfidf=round(score, 6),
        )
        for term, score in ranked
    ]


def source_hash(run_fingerprints: list[tuple[str, str]]) -> str:
    """Deterministic sha256 over sorted (run_id_hex, timestamp) pairs."""
    h = hashlib.sha256()
    for rid, ts in sorted(run_fingerprints):
        h.update(f"{rid}:{ts}\n".encode())
    return h.hexdigest()
