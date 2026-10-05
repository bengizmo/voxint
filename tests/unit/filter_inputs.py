"""Shared deterministic word-piece inputs for filler regression tests."""

import random
from collections.abc import Iterator

from voxint.adjudication.turns import PieceRule, TextMapping, TurnPiece


def seeded_filler_soups(count: int = 500) -> Iterator[tuple[TurnPiece, ...]]:
    rng = random.Random(753)
    tokens = ["um", "uh", "umm", "uhh", "uhm", "erm", "UM", "word", "hmm", "you know"]
    for _ in range(count):
        yield tuple(
            TurnPiece(
                rng.choice(["", " ", "\t", "\n"])
                + rng.choice(['"', "(", "", "["])
                + rng.choice(tokens)
                + rng.choice(["", ",", ".", "?", "!", "...", '"', "?!"]),
                i, i + 1, True, rng.choice([True, False]),
                PieceRule.WORD_LEVEL, TextMapping.VERBATIM,
            )
            for i in range(rng.randrange(1, 30))
        )
