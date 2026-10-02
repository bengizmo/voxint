"""Reading layout: speaker turns broken into paragraphs with minute markers.

Pure, and shared by every surface that shows the turn layout, so they cannot
disagree on where a paragraph begins.

* **P1** a pause of :data:`PARAGRAPH_PAUSE_SECONDS` or more between two pieces
  starts a paragraph. So does time going backwards (segments can overlap).
* **P2** after a sentence end, a new paragraph starts once the current one
  holds :data:`PARAGRAPH_MIN_WORDS` words. A coarse piece is atomic, so breaks
  fall only on piece boundaries and no maximum length is promised.
* **Minute markers** track one run-wide high-water mark. A paragraph start
  consumes every minute up to its own timestamp; inside a paragraph, a piece
  that starts in a later minute is preceded by that minute's marker. Markers
  therefore never stack, never go backwards and never sit inside a piece.

The thresholds are module constants, not settings. They were chosen by
measurement on two-speaker conversations: a 3 second pause with a 20 word
minimum gave paragraph counts and a median paragraph length within a few
percent of a commercial transcription tool's layout of the same recordings,
where a 60 word minimum gave paragraphs twice as long.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from voxint.adjudication.turns import SpeakerTurn, TurnPiece, join_pieces

PARAGRAPH_PAUSE_SECONDS = 3.0
PARAGRAPH_MIN_WORDS = 20
_SENTENCE_END = re.compile(r'''[.?!]["')\]\u201d\u2019]*\s*$''')


@dataclass(frozen=True)
class ReadingRun:
    """Non-empty stripped text, optionally preceded by a minute marker."""

    marker_seconds: int | None
    text: str


@dataclass(frozen=True)
class ReadingParagraph:
    """A named turn start or an unnamed continuation with ordered text runs."""

    speaker: str
    continuation: bool
    start_seconds: float
    runs: tuple[ReadingRun, ...]


def layout_turns(turns: Sequence[SpeakerTurn]) -> list[ReadingParagraph]:
    """Apply pause/sentence breaks and forward-only markers in piece order."""
    paragraphs: list[ReadingParagraph] = []
    last_minute = 0
    for turn in turns:
        groups: list[list[TurnPiece]] = []
        count = 0
        for piece in turn.pieces:
            if not piece.text.strip():
                continue
            previous = groups[-1][-1] if groups else None
            if previous is None or (
                piece.start_seconds - previous.end_seconds >= PARAGRAPH_PAUSE_SECONDS
                or piece.start_seconds < previous.start_seconds
                or (count >= PARAGRAPH_MIN_WORDS and _SENTENCE_END.search(previous.text))
            ):
                groups.append([])
                count = 0
            groups[-1].append(piece)
            count += len(piece.text.split())
        for index, pieces in enumerate(groups):
            last_minute = max(last_minute, int(max(pieces[0].start_seconds, 0) // 60))
            runs: list[ReadingRun] = []
            run_pieces: list[TurnPiece] = []
            marker: int | None = None
            for piece in pieces:
                minute = int(max(piece.start_seconds, 0) // 60)
                if run_pieces and minute > last_minute:
                    runs.append(ReadingRun(marker, join_pieces(run_pieces).strip()))
                    run_pieces = []
                    marker = minute * 60
                    last_minute = minute
                run_pieces.append(piece)
            runs.append(ReadingRun(marker, join_pieces(run_pieces).strip()))
            paragraphs.append(ReadingParagraph(
                turn.speaker, index > 0, pieces[0].start_seconds, tuple(runs),
            ))
    return paragraphs
