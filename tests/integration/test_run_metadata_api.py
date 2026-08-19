"""Run-level metadata surfacing + operator notes routes (issue #36).

Capture itself is covered in ``test_acquire_metadata.py``; here the API layer:
the run detail's source-metadata section and notes form, the notes POST
(CSRF, bounds, persistence), the runs browser title fallback, and the
run-level JSON export envelope (a NEW endpoint — the pinned
``/review/{id}/export.json`` bare-array contract stays untouched).
"""

import json
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import MAX_OPERATOR_NOTES_CHARS, create_app
from voxint.api.csrf import CSRF_NOTES, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "notes-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def _nd(**kwargs: str) -> dict[str, str]:
    """Form fields with a valid notes CSRF token merged in."""
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_NOTES), **kwargs}


@pytest.fixture()
def client(session_factory: sessionmaker[Session]) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _make_run(
    session_factory: sessionmaker[Session],
    *,
    with_metadata: bool = False,
    with_segment: bool = False,
    notes: str | None = None,
) -> uuid.UUID:
    with session_factory() as session:
        media = MediaItem(
            source_path=f"incoming/{uuid.uuid4().hex}/source",
            source_url="https://example.com/watch?v=abc" if with_metadata else None,
        )
        session.add(media)
        session.flush()
        if with_metadata:
            session.add(
                MediaSourceMetadata(
                    media_item_id=media.id,
                    source_kind="ytdlp",
                    title="Episode 42 <em>unsafe</em>",
                    uploader="Example Uploader",
                    uploader_url="https://example.com/@uploader",
                    channel="Example Channel",
                    channel_url="https://example.com/channel/UC123",
                    description="About microphones.",
                    duration_seconds=125.0,
                    tags=["interviews", "acoustics"],
                    canonical_url="https://example.com/watch?v=abc123",
                    extractor="example",
                    extractor_version="2026.07.04",
                    raw={"id": "abc123", "webpage_url": "https://example.com/watch?v=abc123"},
                    raw_schema_version=1,
                    acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
                )
            )
        run = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            operator_notes=notes,
        )
        session.add(run)
        session.flush()
        if with_segment:
            session.add(
                TranscriptSegment(
                    pipeline_run_id=run.id,
                    segment_index=0,
                    start_seconds=0.0,
                    end_seconds=2.0,
                    raw_text="hello world",
                )
            )
        run_id = run.id
        session.commit()
    return run_id


def _notes_for(
    session_factory: sessionmaker[Session], run_id: uuid.UUID
) -> "str | None":
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        return run.operator_notes


class TestRunDetail:
    def test_metadata_section_renders_escaped(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory, with_metadata=True)
        page = client.get(f"/runs/{run_id}")
        assert page.status_code == 200
        assert "Source metadata" in page.text
        # Scraped text is untrusted: autoescape must render, not execute, it.
        assert "Episode 42 &lt;em&gt;unsafe&lt;/em&gt;" in page.text
        assert "<em>unsafe</em>" not in page.text
        assert "Example Uploader" in page.text
        assert "source-reported, not measured" in page.text
        assert "interviews, acoustics" in page.text
        assert "https://example.com/watch?v=abc123" in page.text

    def test_no_metadata_renders_notes_form_only(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory)
        page = client.get(f"/runs/{run_id}")
        assert page.status_code == 200
        assert "Source metadata" not in page.text
        assert "Operator notes" in page.text  # notes apply to upload runs too
        assert f"/runs/{run_id}/notes" in page.text

    def test_existing_notes_prefill_the_form(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory, notes="call the <b>uploader</b>")
        page = client.get(f"/runs/{run_id}")
        assert "call the &lt;b&gt;uploader&lt;/b&gt;" in page.text


class TestSaveNotes:
    def test_saves_and_redirects(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory)
        response = client.post(
            f"/runs/{run_id}/notes",
            data=_nd(notes="Speaker 2 sounds like Jim."),
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/runs/{run_id}"
        assert _notes_for(session_factory, run_id) == "Speaker 2 sounds like Jim."

    def test_blank_notes_clear_to_null(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory, notes="old notes")
        response = client.post(
            f"/runs/{run_id}/notes", data=_nd(notes="   "), follow_redirects=False
        )
        assert response.status_code == 303
        assert _notes_for(session_factory, run_id) is None

    def test_rejected_without_csrf_token(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory, notes="untouched")
        response = client.post(f"/runs/{run_id}/notes", data={"notes": "attacker"})
        assert response.status_code == 403
        assert _notes_for(session_factory, run_id) == "untouched"

    def test_over_cap_is_422(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory)
        response = client.post(
            f"/runs/{run_id}/notes",
            data=_nd(notes="x" * (MAX_OPERATOR_NOTES_CHARS + 1)),
        )
        assert response.status_code == 422
        assert _notes_for(session_factory, run_id) is None

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        response = client.post(f"/runs/{uuid.uuid4()}/notes", data=_nd(notes="x"))
        assert response.status_code == 404


class TestRunsBrowserTitle:
    def test_title_shown_with_path_fallback(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        with_meta = _make_run(session_factory, with_metadata=True)
        without_meta = _make_run(session_factory)
        page = client.get("/runs")
        assert page.status_code == 200
        assert "Episode 42 &lt;em&gt;unsafe&lt;/em&gt;" in page.text
        # Both rows still render their media path.
        with session_factory() as session:
            for run_id in (with_meta, without_meta):
                run = session.get(PipelineRun, run_id)
                assert run is not None
                assert run.media_item.source_path in page.text


class TestRunExportJson:
    def test_envelope_with_metadata_and_notes(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(
            session_factory, with_metadata=True, with_segment=True, notes="my notes"
        )
        response = client.get(f"/runs/{run_id}/export.json")
        assert response.status_code == 200
        body = json.loads(response.text)
        assert body["schema_version"] == 2  # v2: URL fields host-only (finding D4)
        assert body["run"]["id"] == str(run_id)
        assert body["run"]["operator_notes"] == "my notes"
        meta = body["source_metadata"]
        assert meta["source_kind"] == "ytdlp"
        # Descriptive metadata (the operator's own data) is retained verbatim.
        assert meta["title"] == "Episode 42 <em>unsafe</em>"
        assert meta["uploader"] == "Example Uploader"
        assert meta["channel"] == "Example Channel"
        assert meta["description"] == "About microphones."
        assert meta["tags"] == ["interviews", "acoustics"]
        # Finding D4: every URL field is reduced to bare host — no path/query that
        # could carry the full acquisition URL — including raw's webpage_url.
        assert meta["uploader_url"] == "example.com"
        assert meta["channel_url"] == "example.com"
        assert meta["canonical_url"] == "example.com"
        assert meta["raw"] == {"id": "abc123", "webpage_url": "example.com"}
        assert meta["acquired_at"] == "2026-08-01T12:00:00+00:00"
        assert body["segments"] == [
            {
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "speaker": "(no speaker)",  # attributed_transcript's placeholder
                "text": "hello world",
            }
        ]

    def test_envelope_without_metadata(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        run_id = _make_run(session_factory, with_segment=True)
        body = json.loads(client.get(f"/runs/{run_id}/export.json").text)
        assert body["source_metadata"] is None
        assert body["run"]["operator_notes"] is None

    def test_segments_match_the_pinned_review_export(
        self, client: TestClient, session_factory: sessionmaker[Session]
    ) -> None:
        """One shape source: the envelope's segments must be exactly the pinned
        bare-array export's objects (shared transcript_payload)."""
        run_id = _make_run(session_factory, with_segment=True)
        envelope = json.loads(client.get(f"/runs/{run_id}/export.json").text)
        legacy = json.loads(client.get(f"/review/{run_id}/export.json").text)
        assert envelope["segments"] == legacy

    def test_unknown_run_is_404(self, client: TestClient) -> None:
        assert client.get(f"/runs/{uuid.uuid4()}/export.json").status_code == 404
