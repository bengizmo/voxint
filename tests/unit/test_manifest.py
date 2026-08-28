"""Unit tests for the quote provenance manifest builder (issue #122)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from voxint.export.manifest import (
    SCHEMA_VERSION,
    ClipRef,
    QuoteLine,
    StageProvenance,
    StageRole,
    build_quote_bundle,
    build_quote_manifest,
)

_NOW = datetime(2026, 8, 28, 15, 0, 0, tzinfo=UTC)
_ANN_UPDATED = datetime(2026, 8, 27, 10, 0, 0, tzinfo=UTC)
_STAGE_FINISHED = datetime(2026, 8, 25, 9, 55, 0, tzinfo=UTC)
_ANN_ID = uuid.UUID("00000000000000000000000000000001")
_MEDIA_ID = uuid.UUID("00000000000000000000000000000002")
_RUN_ID = uuid.UUID("00000000000000000000000000000003")
_CLIP_ID = uuid.UUID("00000000000000000000000000000004")


def _stages() -> dict[str, StageProvenance]:
    return {
        "transcribe": StageProvenance(
            attempt=1,
            finished_at=_STAGE_FINISHED,
            roles={
                "asr": StageRole(
                    reachable=True, model="large-v2", revision=None, engine="ct2-legacy"
                ),
            },
        ),
        "diarize_embed": StageProvenance(
            attempt=1,
            finished_at=_STAGE_FINISHED,
            roles={
                "diarizer": StageRole(
                    reachable=True, model="speaker-diarization-3.1"
                ),
                "embedder": StageRole(
                    reachable=True, model="titanet-large-v1"
                ),
            },
        ),
    }


def _clip() -> ClipRef:
    return ClipRef(
        id=_CLIP_ID,
        download_url=f"/runs/{_RUN_ID}/clips/{_CLIP_ID}",
        filename="voxint-00000000-clip-00000000.wav",
        sha256="abc123",
        sample_rate=16000,
        channels=1,
        start_sample=196800,
        end_sample=729600,
    )


def _lines() -> list[QuoteLine]:
    return [
        QuoteLine(text="Hello world", speaker="Alice", start_seconds=12.3, end_seconds=15.0),
        QuoteLine(text="Goodbye", speaker="Bob", start_seconds=15.0, end_seconds=18.5),
    ]


def _build(**overrides: object) -> dict:
    defaults: dict = dict(
        exported_at=_NOW,
        annotation_id=_ANN_ID,
        source_text_hash="a" * 64,
        annotation_updated_at=_ANN_UPDATED,
        lines=_lines(),
        timing_precision="word",
        tags=["key-quote"],
        note="Reporter's note",
        clip=_clip(),
        media_id=_MEDIA_ID,
        run_id=_RUN_ID,
        source_title="Interview.mp3",
        media_sha256="b" * 64,
        app_version="0.29.0",
        stages=_stages(),
    )
    defaults.update(overrides)
    return build_quote_manifest(**defaults)


class TestSingleManifest:
    def test_schema_version(self) -> None:
        m = _build()
        assert m["schema_version"] == SCHEMA_VERSION

    def test_kind(self) -> None:
        assert _build()["kind"] == "quote_provenance"

    def test_exported_at_iso(self) -> None:
        assert _build()["exported_at"] == _NOW.isoformat()

    def test_multi_speaker_lines(self) -> None:
        lines = _build()["quote"]["lines"]
        assert len(lines) == 2
        assert lines[0]["speaker"] == "Alice"
        assert lines[1]["speaker"] == "Bob"
        assert lines[0]["text"] == "Hello world"

    def test_timing_precision(self) -> None:
        assert _build()["quote"]["timing_precision"] == "word"

    def test_tags_and_note(self) -> None:
        q = _build()["quote"]
        assert q["tags"] == ["key-quote"]
        assert q["note"] == "Reporter's note"

    def test_annotation_id_hex(self) -> None:
        assert _build()["quote"]["annotation_id"] == _ANN_ID.hex

    def test_source_text_hash(self) -> None:
        assert _build()["quote"]["source_text_hash"] == "a" * 64

    def test_annotation_updated_at(self) -> None:
        assert _build()["quote"]["annotation_updated_at"] == _ANN_UPDATED.isoformat()

    def test_clip_present(self) -> None:
        c = _build()["clip"]
        assert c is not None
        assert c["id"] == _CLIP_ID.hex
        assert c["sha256"] == "abc123"
        assert c["sample_rate"] == 16000
        assert c["start_sample"] == 196800

    def test_clip_nullable(self) -> None:
        assert _build(clip=None)["clip"] is None

    def test_source_fields(self) -> None:
        s = _build()["source"]
        assert s["media_id"] == _MEDIA_ID.hex
        assert s["run_id"] == _RUN_ID.hex
        assert s["title"] == "Interview.mp3"
        assert s["media_sha256"] == "b" * 64

    def test_media_sha256_nullable(self) -> None:
        assert _build(media_sha256=None)["source"]["media_sha256"] is None

    def test_pipeline_provenance_stages(self) -> None:
        p = _build()["pipeline_provenance"]
        assert p["observed_before_attempt"] is True
        assert "transcribe" in p["stages"]
        assert "diarize_embed" in p["stages"]

    def test_stage_roles(self) -> None:
        t = _build()["pipeline_provenance"]["stages"]["transcribe"]
        assert t["attempt"] == 1
        assert t["finished_at"] == _STAGE_FINISHED.isoformat()
        asr = t["roles"]["asr"]
        assert asr["reachable"] is True
        assert asr["model"] == "large-v2"
        assert asr["engine"] == "ct2-legacy"
        assert asr["revision"] is None

    def test_unreachable_role(self) -> None:
        stages = {
            "transcribe": StageProvenance(
                attempt=1,
                finished_at=_STAGE_FINISHED,
                roles={"asr": StageRole(reachable=False)},
            ),
        }
        p = _build(stages=stages)["pipeline_provenance"]
        asr = p["stages"]["transcribe"]["roles"]["asr"]
        assert asr["reachable"] is False
        assert asr["model"] is None

    def test_no_stages_recorded(self) -> None:
        m = _build(stages={})
        assert m["pipeline_provenance"]["stages"] == {}

    def test_note_null(self) -> None:
        assert _build(note=None)["quote"]["note"] is None

    def test_empty_tags(self) -> None:
        assert _build(tags=[])["quote"]["tags"] == []


class TestBundle:
    def test_bundle_schema(self) -> None:
        bundle = build_quote_bundle(
            exported_at=_NOW,
            media_id=_MEDIA_ID,
            run_id=_RUN_ID,
            source_title="Interview.mp3",
            media_sha256="b" * 64,
            app_version="0.29.0",
            stages=_stages(),
            quotes=[{"quote": {}, "clip": None}],
        )
        assert bundle["schema_version"] == SCHEMA_VERSION
        assert bundle["kind"] == "quote_provenance_bundle"
        assert len(bundle["quotes"]) == 1

    def test_bundle_run_level_facts(self) -> None:
        bundle = build_quote_bundle(
            exported_at=_NOW,
            media_id=_MEDIA_ID,
            run_id=_RUN_ID,
            source_title="Interview.mp3",
            media_sha256=None,
            app_version="0.29.0",
            stages=_stages(),
            quotes=[],
        )
        assert bundle["source"]["media_id"] == _MEDIA_ID.hex
        assert bundle["source"]["media_sha256"] is None
        assert "transcribe" in bundle["pipeline_provenance"]["stages"]
