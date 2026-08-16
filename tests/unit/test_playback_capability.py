"""Fail-closed playback capability predicate (issue #55).

Pure-logic unit tests: the two DB-touching seams (``resolve_servable_media`` and
``_transcript_intervals``) are exercised directly with fakes, and
``playback_capability`` is driven by monkeypatching those seams so the
accumulation/fail-closed logic is tested without a live Postgres.
"""

from __future__ import annotations

import io
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest

from voxint.api import playback
from voxint.api.playback import (
    TAIL_TOLERANCE,
    MediaMissing,
    MediaReclaimed,
    MediaResolutionError,
    MediaUnservable,
    PlaybackCapability,
    playback_capability,
    representative_turns,
    resolve_servable_media,
)
from voxint.media.serving import MediaNotServableError
from voxint.pipeline.stages.context import StageDataError

_SETTINGS = SimpleNamespace(media_root=Path("/media"))


def _run(duration: float | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(), media_item=SimpleNamespace(duration_seconds=duration)
    )


def _servable(*_a: object, **_k: object) -> tuple[BinaryIO, int]:
    """Stand-in for a resolvable media handle; caller closes it."""
    return io.BytesIO(b"RIFF"), 4


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolve: object = _servable,
    intervals: list[tuple[float, float]] | None = None,
) -> None:
    monkeypatch.setattr(playback, "resolve_servable_media", resolve)
    monkeypatch.setattr(
        playback,
        "_transcript_intervals",
        lambda _session, _run_id: list(intervals if intervals is not None else []),
    )


# --------------------------------------------------------------------------- #
# Happy path.
# --------------------------------------------------------------------------- #
def test_happy_path_seek_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, intervals=[(0.0, 5.0), (5.0, 9.9)])
    cap = playback_capability(None, _run(10.0), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is True
    assert cap.media_duration == 10.0
    assert cap.reasons == []


# --------------------------------------------------------------------------- #
# Media servability reasons — one per resolution failure.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("error", "code"),
    [
        (MediaReclaimed("gone"), "media_reclaimed"),
        (MediaMissing("no artifact"), "media_missing"),
        (MediaUnservable("bad file"), "media_unservable"),
    ],
)
def test_media_resolution_failure_disables_seek(
    monkeypatch: pytest.MonkeyPatch, error: MediaResolutionError, code: str
) -> None:
    def _raise(*_a: object, **_k: object) -> tuple[BinaryIO, int]:
        raise error

    _patch(monkeypatch, resolve=_raise, intervals=[(0.0, 5.0)])
    cap = playback_capability(None, _run(10.0), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert code in {r.code for r in cap.reasons}
    # Duration is measured independently of media servability.
    assert cap.media_duration == 10.0


def test_capability_never_true_when_media_resolution_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(*_a: object, **_k: object) -> tuple[BinaryIO, int]:
        raise MediaMissing("no artifact")

    # Everything else is pristine; media alone is unresolved.
    _patch(monkeypatch, resolve=_raise, intervals=[(0.0, 9.0)])
    cap = playback_capability(None, _run(10.0), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False


# --------------------------------------------------------------------------- #
# Duration reasons.
# --------------------------------------------------------------------------- #
def test_duration_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, intervals=[(0.0, 5.0)])
    cap = playback_capability(None, _run(None), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert cap.media_duration is None
    assert {r.code for r in cap.reasons} == {"duration_unknown"}


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_duration_invalid(monkeypatch: pytest.MonkeyPatch, bad: float) -> None:
    _patch(monkeypatch, intervals=[(0.0, 5.0)])
    cap = playback_capability(None, _run(bad), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert cap.media_duration is None
    assert "duration_invalid" in {r.code for r in cap.reasons}


# --------------------------------------------------------------------------- #
# Timeline reasons.
# --------------------------------------------------------------------------- #
def test_no_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, intervals=[])
    cap = playback_capability(None, _run(10.0), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert {r.code for r in cap.reasons} == {"no_segments"}


@pytest.mark.parametrize(
    "intervals",
    [
        [(5.0, 3.0)],  # inverted (end < start)
        [(0.0, 0.0)],  # zero-length (end not > start)
        [(-1.0, 2.0)],  # negative start
        [(0.0, math.nan)],  # NaN end
        [(math.inf, math.inf)],  # non-finite
    ],
)
def test_timeline_malformed(
    monkeypatch: pytest.MonkeyPatch, intervals: list[tuple[float, float]]
) -> None:
    _patch(monkeypatch, intervals=intervals)
    cap = playback_capability(None, _run(100.0), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert "timeline_malformed" in {r.code for r in cap.reasons}


def test_tolerance_boundary_at_exactly_duration_plus_tol_is_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 100.0
    # Last end sits EXACTLY on the tolerance line — not out of bounds.
    _patch(monkeypatch, intervals=[(0.0, duration + TAIL_TOLERANCE)])
    cap = playback_capability(None, _run(duration), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is True
    assert cap.reasons == []


def test_tolerance_boundary_just_past_is_out_of_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 100.0
    _patch(monkeypatch, intervals=[(0.0, duration + 0.051)])
    cap = playback_capability(None, _run(duration), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert {r.code for r in cap.reasons} == {"timeline_out_of_bounds"}


# --------------------------------------------------------------------------- #
# Accumulation — every applicable reason, not just the first.
# --------------------------------------------------------------------------- #
def test_accumulates_all_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: object, **_k: object) -> tuple[BinaryIO, int]:
        raise MediaReclaimed("gone")

    _patch(monkeypatch, resolve=_raise, intervals=[])
    cap = playback_capability(None, _run(None), _SETTINGS, None)  # type: ignore[arg-type]
    assert cap.seek_enabled is False
    assert {r.code for r in cap.reasons} == {
        "media_reclaimed",
        "duration_unknown",
        "no_segments",
    }


def test_reasons_carry_plain_language_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, intervals=[])
    cap = playback_capability(None, _run(10.0), _SETTINGS, None)  # type: ignore[arg-type]
    (reason,) = cap.reasons
    assert reason.code == "no_segments"
    assert reason.message and not reason.message.startswith("no_segments")


# --------------------------------------------------------------------------- #
# to_props — island contract.
# --------------------------------------------------------------------------- #
def test_to_props_shape() -> None:
    cap = PlaybackCapability(
        seek_enabled=False,
        media_duration=12.5,
        reasons=[playback.CapabilityReason("no_segments", "nope")],
    )
    assert cap.to_props() == {
        "seekEnabled": False,
        "mediaDuration": 12.5,
        "reasons": [{"code": "no_segments", "message": "nope"}],
    }


def test_to_props_null_duration() -> None:
    cap = PlaybackCapability(seek_enabled=True, media_duration=None, reasons=[])
    props = cap.to_props()
    assert props["mediaDuration"] is None
    assert props["reasons"] == []


# --------------------------------------------------------------------------- #
# resolve_servable_media — the shared /media seam, mapping to typed errors.
# --------------------------------------------------------------------------- #
class _Gate:
    def __init__(self, *, raise_unservable: bool = False) -> None:
        self._raise = raise_unservable

    def open_for_serving(self, path: Path) -> tuple[BinaryIO, int]:
        if self._raise:
            raise MediaNotServableError("cannot serve")
        return io.BytesIO(b"data"), 4


def test_resolve_reclaimed_raises_410(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        playback, "run_intermediate_reclaimed_at", lambda _s, _r: datetime.now(UTC)
    )
    with pytest.raises(MediaReclaimed) as exc:
        resolve_servable_media(None, uuid.uuid4(), _SETTINGS, _Gate())  # type: ignore[arg-type]
    assert exc.value.http_status == 410
    assert exc.value.code == "media_reclaimed"


def test_resolve_missing_artifact_raises_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playback, "run_intermediate_reclaimed_at", lambda _s, _r: None)

    def _raise(*_a: object, **_k: object) -> Path:
        raise StageDataError("no artifact")

    monkeypatch.setattr(playback, "normalized_audio_path", _raise)
    with pytest.raises(MediaMissing) as exc:
        resolve_servable_media(None, uuid.uuid4(), _SETTINGS, _Gate())  # type: ignore[arg-type]
    assert exc.value.http_status == 404
    assert exc.value.code == "media_missing"


def test_resolve_gate_refusal_raises_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playback, "run_intermediate_reclaimed_at", lambda _s, _r: None)
    monkeypatch.setattr(
        playback, "normalized_audio_path", lambda _s, _r, _root: Path("/media/a.wav")
    )
    with pytest.raises(MediaUnservable) as exc:
        resolve_servable_media(
            None,  # type: ignore[arg-type]
            uuid.uuid4(),
            _SETTINGS,
            _Gate(raise_unservable=True),
        )
    assert exc.value.http_status == 404
    assert exc.value.code == "media_unservable"


def test_resolve_success_returns_open_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playback, "run_intermediate_reclaimed_at", lambda _s, _r: None)
    monkeypatch.setattr(
        playback, "normalized_audio_path", lambda _s, _r, _root: Path("/media/a.wav")
    )
    fh, size = resolve_servable_media(None, uuid.uuid4(), _SETTINGS, _Gate())  # type: ignore[arg-type]
    try:
        assert size == 4
        assert fh.read() == b"data"
    finally:
        fh.close()


# --------------------------------------------------------------------------- #
# _transcript_intervals — thin query wrapper.
# --------------------------------------------------------------------------- #
class _Result:
    def __init__(self, rows: list[tuple[float, float]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[float, float]]:
        return self._rows


class _Session:
    def __init__(self, rows: list[tuple[float, float]]) -> None:
        self._rows = rows

    def execute(self, _stmt: object) -> _Result:
        return _Result(self._rows)


def test_transcript_intervals_maps_rows() -> None:
    session = _Session([(0.0, 1.0), (1.0, 2.5)])
    assert playback._transcript_intervals(session, uuid.uuid4()) == [  # type: ignore[arg-type]
        (0.0, 1.0),
        (1.0, 2.5),
    ]


# --------------------------------------------------------------------------- #
# representative_turns — clean per-label sample for "preview this speaker".
# --------------------------------------------------------------------------- #
def _turn(
    label: str, start: float, end: float, *, overlap: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        label=label, start_seconds=start, end_seconds=end, overlap=overlap
    )


class _Scalars:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def all(self) -> list[SimpleNamespace]:
        return self._rows


class _TurnResult:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def scalars(self) -> _Scalars:
        return _Scalars(self._rows)


class _TurnSession:
    def __init__(self, rows: list[SimpleNamespace]) -> None:
        self._rows = rows

    def execute(self, _stmt: object) -> _TurnResult:
        return _TurnResult(self._rows)


def test_representative_prefers_longest_non_overlap() -> None:
    session = _TurnSession(
        [
            _turn("SPEAKER_00", 0.0, 10.0, overlap=True),  # longest but overlaps
            _turn("SPEAKER_00", 2.0, 8.0, overlap=False),  # 6s, clean → wins
            _turn("SPEAKER_00", 3.0, 5.0, overlap=False),  # 2s, clean
        ]
    )
    assert representative_turns(session, uuid.uuid4()) == {  # type: ignore[arg-type]
        "SPEAKER_00": (2.0, 8.0),
    }


def test_representative_falls_back_to_longest_when_all_overlap() -> None:
    session = _TurnSession(
        [
            _turn("SPEAKER_01", 0.0, 3.0, overlap=True),
            _turn("SPEAKER_01", 5.0, 12.0, overlap=True),  # 7s, longest → wins
        ]
    )
    assert representative_turns(session, uuid.uuid4()) == {  # type: ignore[arg-type]
        "SPEAKER_01": (5.0, 12.0),
    }


def test_representative_groups_multiple_labels() -> None:
    session = _TurnSession(
        [
            _turn("SPEAKER_00", 0.0, 4.0),
            _turn("SPEAKER_01", 4.0, 9.0),
        ]
    )
    assert representative_turns(session, uuid.uuid4()) == {  # type: ignore[arg-type]
        "SPEAKER_00": (0.0, 4.0),
        "SPEAKER_01": (4.0, 9.0),
    }


def test_representative_empty_when_no_turns() -> None:
    assert representative_turns(_TurnSession([]), uuid.uuid4()) == {}  # type: ignore[arg-type]


def test_representative_tie_breaks_on_earlier_start() -> None:
    # Two clean 4s turns for one label: the earlier-starting one represents.
    session = _TurnSession(
        [
            _turn("SPEAKER_00", 10.0, 14.0),
            _turn("SPEAKER_00", 2.0, 6.0),
        ]
    )
    assert representative_turns(session, uuid.uuid4()) == {  # type: ignore[arg-type]
        "SPEAKER_00": (2.0, 6.0),
    }
