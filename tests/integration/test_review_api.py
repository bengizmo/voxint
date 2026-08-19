"""The review console end to end: queue → claim → rulings → enrollment → export → media.

Real Postgres (migrated), real templates, real ffprobe on a real WAV.
"""

import io
import json
import re
import uuid
import wave
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_CLAIM, mint_csrf_token
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    MediaSourceMetadata,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v1"
_CSRF_KEY = "review-api-test-csrf-key"  # low-entropy; a known secret lets tests mint


def write_wav(path: Path, seconds: float = 0.5) -> None:
    frames = int(16000 * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)
    path.write_bytes(buf.getvalue())


def unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim] = 1.0
    return vector


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(
    session_factory: sessionmaker[Session], media_root: Path
) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        review_claim_ttl_seconds=600,
        csrf_secret=_CSRF_KEY,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def seed_run(session: Session, media_root: Path) -> uuid.UUID:
    """A completed run: two labels, segments, one grounded proposal + one hint."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()

    audio_rel = f"artifacts/{run.id}/normalized.wav"
    audio_abs = media_root / audio_rel
    audio_abs.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_abs)
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )

    for index, label in enumerate(["S0", "S0", "S1", "S1"]):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                label=label,
                embedding=unit(0 if label == "S0" else 1),
                embedding_space=SPACE,
            )
        )
    for index, (label, text) in enumerate(
        [("S0", "hello there"), ("S1", "hi back"), ("S0", "how are you")]
    ):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                raw_text=text,
                diarization_label=label,
            )
        )
    known = Speaker(display_name="Known Voice")
    session.add(known)
    session.flush()
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run.id,
            diarization_label="S0",
            speaker_id=known.id,
            method="cosine",
            confidence=0.92,
            grounded=True,
        )
    )
    session.add(
        SpeakerAssignment(
            pipeline_run_id=run.id,
            diarization_label="S1",
            method="llm_hint",
            proposed_name="Norma Newvoice",
            grounded=False,
        )
    )
    session.commit()
    return run.id


def claim_token(client: TestClient, run_id: uuid.UUID) -> str:
    resp = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    location = resp.headers["location"]
    return location.split("token=")[1]


def test_claim_rejected_without_csrf_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # /review/{id}/claim mints the run's claim token, so it has no unguessable
    # token of its own — a CSRF token gates a forged cross-site claim. No token ⇒
    # 403 before claim_run touches the DB (the run stays unclaimed).
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    resp = client.post(f"/review/{run_id}/claim", follow_redirects=False)
    assert resp.status_code == 403


def test_referrer_policy_on_every_response(client: TestClient) -> None:
    # Finding D1: the claim token rides in the URL, so no response may leak it via
    # a Referer header. Referrer-Policy: no-referrer is stamped on every response,
    # including unauthenticated ones like /healthz.
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers["referrer-policy"] == "no-referrer"


def test_referrer_policy_survives_unhandled_500(media_root: Path) -> None:
    # Finding D1 (review fix): an unhandled 500 is generated by Starlette's
    # ServerErrorMiddleware, which sits OUTSIDE the header middleware; the
    # registered exception handler re-applies the header so the guarantee holds
    # even on error pages. A raising route on a non-/review path proves the wiring.
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        csrf_secret=_CSRF_KEY,
    )
    app = create_app(settings=settings, session_factory=None)

    @app.get("/__boom_test")
    def _boom() -> None:
        raise RuntimeError("boom")

    boom_client = TestClient(app, raise_server_exceptions=False)
    boom_client.auth = CREDS
    resp = boom_client.get("/__boom_test")
    assert resp.status_code == 500
    assert resp.headers["referrer-policy"] == "no-referrer"


def test_review_pages_and_redirects_are_no_store(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Finding D1: token-bearing /review responses (pages that embed the token in
    # hidden fields/props, and the claim/mutation redirects that carry it in
    # Location) must never be cached. The security middleware stamps no-store on
    # every /review response; a non-/review page (/healthz) is not forced to.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    claim = client.post(
        f"/review/{run_id}/claim",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_CLAIM)},
        follow_redirects=False,
    )
    assert claim.status_code == 303
    assert claim.headers["cache-control"] == "no-store"
    assert claim.headers["referrer-policy"] == "no-referrer"
    token = claim.headers["location"].split("token=")[1]

    for path in (f"/review/{run_id}", f"/review/{run_id}/transcript"):
        page = client.get(path, params={"token": token})
        assert page.status_code == 200
        assert page.headers["cache-control"] == "no-store"
        assert page.headers["referrer-policy"] == "no-referrer"

    # A non-/review response carries the referrer policy but is not force-no-store.
    health = client.get("/healthz")
    assert health.headers.get("cache-control") != "no-store"


def test_queue_renders_operator_ergonomics(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """The queue row shows friendly label, duration, progress, and a sort control
    (issue #56). seed_run has 2 labels with 1 unresolved (S1) → 1 of 2 resolved."""
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    body = client.get("/review").text
    assert str(run_id) in body
    # Friendly label leads; the raw path stays as muted ground truth beneath.
    assert 'class="media-title"' in body
    # No probed duration on this upload → the honest em-dash, not "0:00".
    assert "—" in body
    # Progress bar fills toward done with always-visible text + full ARIA state.
    assert 'role="progressbar"' in body
    assert 'aria-valuenow="1"' in body
    assert 'aria-valuemax="2"' in body
    assert "1 of 2 resolved" in body
    # #93: the count text is ADJACENT (not overlaid on the fill); the slim bar
    # reuses the review-journey .progress-track and is aria-hidden. An overlaid
    # label on the accent gradient measured below AA across the filled/unfilled
    # split, so the old absolute-positioned .progress-fill is retired.
    assert 'class="progress-track" aria-hidden="true"' in body
    assert 'class="progress-fill"' not in body
    # role="progressbar" makes children presentational, so the accessible name
    # comes from aria-label alone — pin it so it can't silently drift from the
    # visible count on a future edit.
    assert 'aria-label="1 of 2 voices resolved"' in body
    # Responsive + a11y (issue #64): the wide queue table scrolls inside a
    # keyboard-reachable, labelled region, with scoped column headers.
    assert 'class="table-wrap" role="region" aria-label="Review queue" tabindex="0"' in body
    assert '<th scope="col">Media</th>' in body
    # The otherwise-empty action column header carries a visually-hidden label so
    # it isn't an unnamed column for assistive tech (locks the one novel a11y bit).
    assert '<th scope="col"><span class="visually-hidden">Action</span></th>' in body
    # #93: the per-row Review action is the one teal primary of the row.
    assert 'class="primary">Review</button>' in body
    # Sort control offers the actionability option; default stays oldest.
    assert "Most voices to resolve" in body
    sorted_body = client.get("/review", params={"sort": "unresolved"}).text
    assert 'href="/review?sort=unresolved"' in sorted_body
    assert 'aria-current="true"' in sorted_body


def test_queue_and_runs_escape_hostile_media_metadata(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """A fetched URL's title/path are attacker-influenced; the console must escape
    them in both element text and the title="..." attribute (issue #56 review)."""
    with session_factory() as session:
        media = MediaItem(
            source_path='incoming/x" onmouseover="alert(1) .wav',
            duration_seconds=None,
        )
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="<script>alert(1)</script>",
                raw={"id": "x"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, tzinfo=UTC),
            )
        )
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        vec = [0.0] * EMBEDDING_DIM
        vec[0] = 1.0
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=0,
                start_seconds=0.0,
                end_seconds=8.0,
                label="S0",
                embedding=vec,
                embedding_space=SPACE,
            )
        )
        session.commit()

    for path in ("/review", "/runs"):
        body = client.get(path).text
        # The live script tag never reaches the DOM; only its escaped form does.
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
        # The quote in source_path cannot break out of the title="..." attribute.
        assert 'onmouseover="alert(1)' not in body
        assert "&#34;" in body or "&quot;" in body


def test_full_review_flow(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    # The run is queued (S1 unresolved), S0 already grounded.
    queue = client.get("/review")
    assert queue.status_code == 200
    assert str(run_id) in queue.text

    token = claim_token(client, run_id)
    page = client.get(f"/review/{run_id}", params={"token": token})
    assert page.status_code == 200
    assert "needs ruling" in page.text  # S1
    assert "Known Voice" in page.text  # S0 machine identity
    assert "Norma Newvoice" in page.text  # hint shown as evidence

    # Enroll the unmatched voice under the hinted name (htmx fragment refresh).
    enroll = client.post(
        f"/review/{run_id}/labels/S1/enroll",
        data={"token": token, "nonce": uuid.uuid4().hex, "display_name": "Norma Newvoice"},
        headers={"HX-Request": "true"},
    )
    assert enroll.status_code == 200
    assert "assigned: Norma Newvoice" in enroll.text
    assert "needs ruling" not in enroll.text

    with session_factory() as session:
        embedding = session.execute(select(SpeakerEmbedding)).scalars().one()
        assert embedding.source_diarization_label == "S1"
        assert embedding.embedding_space == SPACE

    # The queue is now empty and the export resolves names through the resolver.
    assert str(run_id) not in client.get("/review").text
    export = client.get(f"/review/{run_id}/export.txt")
    assert export.status_code == 200
    assert "Known Voice: hello there" in export.text
    assert "Norma Newvoice: hi back" in export.text

    # Release and confirm the slot is free.
    release = client.post(
        f"/review/{run_id}/release", data={"token": token}, follow_redirects=False
    )
    assert release.status_code == 303
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.review_claim_token is None


def test_export_formats_content_types_and_payloads(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Structured/subtitle exports share the CLI formatters; here we prove the
    # routes wire the right media type and the attributed data through. S0 is
    # grounded ("Known Voice"); S1 stays a raw label until ruled on.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    srt = client.get(f"/review/{run_id}/export.srt")
    assert srt.status_code == 200
    assert srt.headers["content-type"].startswith("application/x-subrip")
    assert "1\n00:00:00,000 --> 00:00:08,000\nKnown Voice:\nhello there\n" in srt.text

    vtt = client.get(f"/review/{run_id}/export.vtt")
    assert vtt.status_code == 200
    assert vtt.headers["content-type"].startswith("text/vtt")
    assert vtt.text.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:08.000" in vtt.text

    resp = client.get(f"/review/{run_id}/export.json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    rows = resp.json()
    assert [r["speaker"] for r in rows] == ["Known Voice", "S1", "Known Voice"]
    assert rows[0] == {
        "start_seconds": 0.0,
        "end_seconds": 8.0,
        "speaker": "Known Voice",
        "text": "hello there",
    }

    rttm = client.get(f"/review/{run_id}/export.rttm")
    assert rttm.status_code == 200
    assert rttm.headers["content-type"].startswith("text/plain")
    # RTTM carries the four raw diarization turns (labels, never resolved names).
    lines = rttm.text.splitlines()
    assert len(lines) == 4
    assert lines[0] == f"SPEAKER {run_id} 1 0.000 8.000 <NA> <NA> S0 <NA> <NA>"
    assert "Known Voice" not in rttm.text


def test_export_text_variant_and_errors(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    # ?text=raw yields the immutable ASR text (here identical to enhanced, which
    # is NULL in the seed) and is accepted; an unknown variant is a 422.
    assert client.get(f"/review/{run_id}/export.srt", params={"text": "raw"}).status_code == 200
    assert (
        client.get(f"/review/{run_id}/export.srt", params={"text": "bogus"}).status_code == 422
    )

    # An unknown run 404s before any formatting.
    assert client.get(f"/review/{uuid.uuid4()}/export.json").status_code == 404
    assert client.get(f"/review/{uuid.uuid4()}/export.rttm").status_code == 404


def test_export_txt_timestamps_toggle(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # issue #52: ?timestamps=false drops the [start end] bracket column; the
    # default keeps it. Only the txt route exposes the flag.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    default = client.get(f"/review/{run_id}/export.txt")
    assert default.status_code == 200
    assert "[" in default.text and "Known Voice: hello there" in default.text

    plain = client.get(f"/review/{run_id}/export.txt", params={"timestamps": "false"})
    assert plain.status_code == 200
    assert "[" not in plain.text
    assert plain.text.startswith("Known Voice: hello there\n")


def test_export_txt_route_matches_cli_bytes(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
    tmp_path: Path,
) -> None:
    # AC #3 honesty: the downloaded file is byte-identical to the CLI export for
    # the equivalent options — proven directly, not just "by construction". The
    # CLI (`main`) and the TestClient share the migrated test DB (conftest sets
    # DATABASE_URL), so both read the same seeded run.
    from voxint.cli import main

    with session_factory() as session:
        run_id = seed_run(session, media_root)

    for label, query, cli_args in (
        ("default", {}, []),
        ("no-timestamps", {"timestamps": "false"}, ["--no-timestamps"]),
    ):
        route_bytes = client.get(f"/review/{run_id}/export.txt", params=query).content
        out = tmp_path / f"cli-{label}.txt"
        assert main(["export", str(run_id), "--format", "txt", *cli_args, "-o", str(out)]) == 0
        cli_bytes = out.read_bytes()
        assert cli_bytes == route_bytes, label
        # Both transports emit LF, never CRLF — the CLI writes UTF-8 bytes
        # directly rather than through a platform text stream, so the contract
        # holds on Windows too (where text mode would translate \n to \r\n).
        assert b"\r\n" not in cli_bytes and b"\r\n" not in route_bytes, label


def test_export_menu_surfaces_every_format(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # issue #52: the picker makes every built format reachable from the UI. The
    # transcript page and the workbench both render the shared menu.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    for page in (f"/runs/{run_id}/transcript", f"/review/{run_id}"):
        html = client.get(page).text
        for ext in ("txt", "md", "srt", "vtt", "json", "rttm"):
            assert f"/review/{run_id}/export.{ext}" in html, f"{ext} missing on {page}"
        # Every non-RTTM format offers all THREE variants the help text promises:
        # reviewed (corrected, the operator-effective default the picker used to
        # hide), enhanced, and raw (issue #65). Asserted per-format so dropping one
        # variant from a single format can never hide behind another's link.
        for ext in ("txt", "md", "srt", "vtt", "json"):
            for variant in ("corrected", "enhanced", "raw"):
                assert (
                    f"/review/{run_id}/export.{ext}?text={variant}" in html
                ), f"{ext} missing variant {variant} on {page}"
        # A timestamp-free reading copy and an on-screen read-mode entry.
        assert "timestamps=false" in html
        assert f"/runs/{run_id}/transcript?read=1" in html


def test_decision_correction_and_stale_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    token = claim_token(client, run_id)
    exclude = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
        headers={"HX-Request": "true"},
    )
    assert exclude.status_code == 200
    assert "excluded" in exclude.text

    # Correction: a later ruling supersedes at read time.
    unknown = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "unknown"},
        headers={"HX-Request": "true"},
    )
    assert unknown.status_code == 200
    assert "Unknown (S1)" in client.get(f"/review/{run_id}/export.txt").text

    # A rotated claim kills the old token: 409, and no ledger row is written.
    fresh = claim_token(client, run_id)
    assert fresh != token
    stale = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
    )
    assert stale.status_code == 409

    # Unknown action and mismatched assign shape are rejected.
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "promote"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "assign"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/review/{run_id}/labels/NOPE/decision",
            data={"token": fresh, "nonce": uuid.uuid4().hex, "action": "exclude"},
        ).status_code
        == 404
    )
    # A well-formed but nonexistent speaker_id is a clean 422, not a raw
    # IntegrityError masquerading as a replay.
    assert (
        client.post(
            f"/review/{run_id}/labels/S1/decision",
            data={
                "token": fresh,
                "nonce": uuid.uuid4().hex,
                "action": "assign",
                "speaker_id": str(uuid.uuid4()),
            },
        ).status_code
        == 422
    )


def test_assign_to_existing_speaker(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        known = session.execute(select(Speaker)).scalars().one()
        known_id = known.id

    token = claim_token(client, run_id)
    assign = client.post(
        f"/review/{run_id}/labels/S1/decision",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(known_id),
        },
        headers={"HX-Request": "true"},
    )
    assert assign.status_code == 200
    assert "assigned: Known Voice" in assign.text
    # Assign-to-existing must NOT create an enrollment centroid.
    with session_factory() as session:
        assert session.execute(select(SpeakerEmbedding)).scalars().all() == []


def test_media_range_serving(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    size = (media_root / "artifacts" / str(run_id) / "normalized.wav").stat().st_size

    head = client.head(f"/media/{run_id}")
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert int(head.headers["content-length"]) == size

    full = client.get(f"/media/{run_id}")
    assert full.status_code == 200
    assert len(full.content) == size
    assert full.content[:4] == b"RIFF"

    partial = client.get(f"/media/{run_id}", headers={"Range": "bytes=4-7"})
    assert partial.status_code == 206
    assert partial.headers["content-range"] == f"bytes 4-7/{size}"
    assert len(partial.content) == 4

    suffix = client.get(f"/media/{run_id}", headers={"Range": "bytes=-16"})
    assert suffix.status_code == 206
    assert len(suffix.content) == 16
    assert suffix.headers["content-range"] == f"bytes {size - 16}-{size - 1}/{size}"

    beyond = client.get(f"/media/{run_id}", headers={"Range": f"bytes={size}-"})
    assert beyond.status_code == 416
    assert beyond.headers["content-range"] == f"bytes */{size}"

    missing = client.get(f"/media/{uuid.uuid4()}")
    assert missing.status_code == 404


def test_media_rejects_non_audio_artifact(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
        artifact = session.execute(select(AudioArtifact)).scalars().one()
        (media_root / artifact.path).write_bytes(b"definitely not audio")
        session.commit()
    assert client.get(f"/media/{run_id}").status_code == 404


def test_reclaimed_media_returns_410_and_shows_notice(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # After GC reclaims the intermediate (issue #15): the row survives with a
    # reclaimed_at stamp but the file is gone. /media answers 410 Gone, and run
    # detail shows a "Media reclaimed" notice instead of a dead audio link.
    import datetime as _dt

    with session_factory() as session:
        run_id = seed_run(session, media_root)
        artifact = session.execute(select(AudioArtifact)).scalars().one()
        (media_root / artifact.path).unlink()
        artifact.reclaimed_at = _dt.datetime.now(tz=_dt.UTC)
        artifact.reclaimed_bytes = 4096
        session.commit()

    assert client.get(f"/media/{run_id}").status_code == 410
    assert client.head(f"/media/{run_id}").status_code == 410

    body = client.get(f"/runs/{run_id}").text
    assert "Media reclaimed" in body
    assert f"/media/{run_id}" not in body  # no dead audio link
    # The transcript link still stands — decisions and transcript are kept.
    assert f"/runs/{run_id}/transcript" in body


def test_metrics_endpoint_renders_prometheus(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        seed_run(session, media_root)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    body = resp.text
    # A seeded completed run + one enrolled speaker are reflected in the series.
    assert '# TYPE voxint_runs gauge' in body
    assert 'voxint_runs{status="completed"} 1' in body
    assert 'voxint_runs{status="failed"} 0' in body  # zero-filled, never absent
    assert 'voxint_roster_speakers 1' in body
    assert body.endswith("\n")


def _label_card_colors(body: str) -> dict[str, int]:
    """Map each label to its `.label-card` palette index in the workbench HTML."""
    pairs = re.findall(
        r'class="label-card(?: spk-(\d+))?">\s*<h2>\s*(\w+)', body
    )
    return {label: int(idx) for idx, label in pairs if idx}


def _transcript_line_colors(body: str) -> dict[str, int]:
    """Map each label to its transcript fallback-line palette index."""
    pairs = re.findall(
        r'class="preview tp-line spk-(\d+)">\s*<span class="t">[^<]*</span>\s*'
        r'<span class="spk-badge">(\w+)</span>',
        body,
    )
    return {label: int(idx) for idx, label in pairs}


def test_workbench_and_transcript_agree_on_speaker_color(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #50: color is derived from ONE canonical per-run label universe, so a
    # label's `.label-card` accent on the workbench matches its transcript line's
    # accent for the same run. This is the cross-surface agreement invariant.
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    token = claim_token(client, run_id)

    workbench = client.get(f"/review/{run_id}", params={"token": token}).text
    transcript = client.get(
        f"/runs/{run_id}/transcript", params={"text": "raw"}
    ).text

    card_colors = _label_card_colors(workbench)
    line_colors = _transcript_line_colors(transcript)
    # Both surfaces resolved S0 and S1 to a color...
    assert set(card_colors) == {"S0", "S1"} == set(line_colors)
    # ...and they agree label-for-label (no drift between independent renders).
    assert card_colors == line_colors


def _seed_run_with_confidences(
    session: Session, media_root: Path, confidences: list[float | None]
) -> uuid.UUID:
    """A completed run whose segments carry the given per-segment ASR confidences."""
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
    session.add(run)
    session.flush()
    audio_rel = f"artifacts/{run.id}/normalized.wav"
    audio_abs = media_root / audio_rel
    audio_abs.parent.mkdir(parents=True, exist_ok=True)
    write_wav(audio_abs)
    session.add(
        AudioArtifact(
            pipeline_run_id=run.id,
            kind=ArtifactKind.PREPROCESSED_AUDIO.value,
            path=audio_rel,
        )
    )
    for index, conf in enumerate(confidences):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                raw_text=f"segment {index}",
                diarization_label="S0",
                confidence=conf,
            )
        )
    session.commit()
    return run.id


def test_transcript_flags_low_confidence_segments(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #53: a segment below the (default 0.6) triage threshold is flagged
    # "uncertain" in the JS-off fallback; a confident one and a NULL one are not.
    # The island props carry the raw confidence + the shared threshold, so the
    # hydrated island flags identically.
    with session_factory() as session:
        run_id = _seed_run_with_confidences(session, media_root, [0.30, 0.95, None])

    body = client.get(f"/runs/{run_id}/transcript").text
    # Exactly one line is flagged uncertain (the 0.30 segment); its chip is honest.
    # (Assert on rendered HTML markers, not the class name — that also appears in
    # the stylesheet.) The chip tooltip text is HTML-only.
    assert body.count(' tp-uncertain"') == 1  # the flagged line's class list
    assert body.count("Low ASR confidence — uncertain, not necessarily wrong") == 1

    # Island props expose the threshold and every segment's confidence (incl. null).
    match = re.search(r"data-props='([^']*)'", body)
    assert match is not None
    props = json.loads(match.group(1))
    assert props["lowConfidenceThreshold"] == 0.6
    assert [s["confidence"] for s in props["segments"]] == [0.30, 0.95, None]


def test_transcript_island_props_offer_peaks_url_with_servable_media(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #57: with a real servable WAV, peaksUrl points at the lazy peaks
    # route (the first island fetch computes the envelope) and the turns carry
    # the diarization timeline for the strip.
    with session_factory() as session:
        run_id = seed_run(session, media_root)

    body = client.get(f"/runs/{run_id}/transcript").text
    match = re.search(r"data-props='([^']*)'", body)
    assert match is not None
    props = json.loads(match.group(1))
    assert props["peaksUrl"] == f"/media/{run_id}/peaks"
    # seed_run's four turns, in (start, turn_index) order, palette-aligned with
    # the segment list (S0 -> 0, S1 -> 1 over the sorted universe).
    assert [t["start"] for t in props["turns"]] == [0.0, 10.0, 20.0, 30.0]
    assert [t["paletteIndex"] for t in props["turns"]] == [0, 0, 1, 1]
    assert all(t["overlap"] is False for t in props["turns"])
    # And the offered URL actually serves an envelope.
    assert client.get(props["peaksUrl"]).status_code == 200


def test_peaks_url_null_when_media_missing_even_with_cached_row(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Review fix: a cached peaks row rescues peaksUrl ONLY for a reclaimed WAV
    # (served unverified by design). If the WAV goes missing with no reclaim
    # stamp, the route can only 404 — so peaksUrl must be null and the island
    # never fires a doomed fetch, even though a cached row/file exists.
    from datetime import UTC, datetime

    with session_factory() as session:
        run_id = seed_run(session, media_root)
    assert client.get(f"/media/{run_id}/peaks").status_code == 200  # cache a row

    with session_factory() as session:
        artifact = session.execute(
            select(AudioArtifact).where(
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
            )
        ).scalars().one()
        (media_root / artifact.path).unlink()
        session.commit()
    props = json.loads(
        re.search(r"data-props='([^']*)'", client.get(f"/runs/{run_id}/transcript").text).group(1)  # type: ignore[union-attr]
    )
    assert props["peaksUrl"] is None

    # But a formally RECLAIMED WAV with the same cached row DOES keep the URL
    # (the static waveform is honest derived evidence), and it serves.
    with session_factory() as session:
        artifact = session.execute(
            select(AudioArtifact).where(
                AudioArtifact.kind == ArtifactKind.PREPROCESSED_AUDIO.value
            )
        ).scalars().one()
        artifact.reclaimed_at = datetime.now(tz=UTC)
        artifact.reclaimed_bytes = 4096
        session.commit()
    props = json.loads(
        re.search(r"data-props='([^']*)'", client.get(f"/runs/{run_id}/transcript").text).group(1)  # type: ignore[union-attr]
    )
    assert props["peaksUrl"] == f"/media/{run_id}/peaks"
    assert client.get(props["peaksUrl"]).status_code == 200


def _segment_ids(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> list[uuid.UUID]:
    with session_factory() as session:
        return list(
            session.execute(
                select(TranscriptSegment.id)
                .where(TranscriptSegment.pipeline_run_id == run_id)
                .order_by(TranscriptSegment.segment_index)
            ).scalars()
        )


def test_verify_segment_updates_progress(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #53: verify marks a segment and reports N-of-M; unverify reverses it.
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # 3 segments
    segs = _segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    r = client.post(f"/review/{run_id}/segments/{segs[0]}/verify", data={"token": token})
    assert r.status_code == 200
    body = r.json()
    assert body["verified"] is True
    assert body["progress"] == {"verified": 1, "total": 3}

    # Idempotent: verifying again keeps the count at 1.
    again = client.post(f"/review/{run_id}/segments/{segs[0]}/verify", data={"token": token})
    assert again.json()["progress"]["verified"] == 1

    # Unverify.
    off = client.post(
        f"/review/{run_id}/segments/{segs[0]}/verify",
        data={"token": token, "verified": "false"},
    )
    assert off.json()["verified"] is False
    assert off.json()["progress"]["verified"] == 0


def test_correct_segment_precedence_and_clears_verification(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #58: a correction takes display precedence in the default view but
    # never touches ?text=raw (immutable evidence), and editing clears a prior
    # verified mark (edited text must be re-verified).
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # segment 0 raw = "hello there"
    segs = _segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    client.post(f"/review/{run_id}/segments/{segs[0]}/verify", data={"token": token})
    r = client.post(
        f"/review/{run_id}/segments/{segs[0]}/text",
        data={"token": token, "text": "hello THERE (fixed)"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["corrected"] is True
    assert body["text"] == "hello THERE (fixed)"
    assert body["verified"] is False  # editing cleared the verification

    default_view = client.get(f"/runs/{run_id}/transcript").text
    raw_view = client.get(f"/runs/{run_id}/transcript", params={"text": "raw"}).text
    assert "hello THERE (fixed)" in default_view  # correction applied by default
    assert "hello THERE (fixed)" not in raw_view  # raw is immutable ASR evidence
    assert "hello there" in raw_view

    # Re-verify, then REPLAY the identical correction: an unchanged save is a true
    # no-op and must NOT silently unverify the segment (idempotent state-setting).
    client.post(f"/review/{run_id}/segments/{segs[0]}/verify", data={"token": token})
    replay = client.post(
        f"/review/{run_id}/segments/{segs[0]}/text",
        data={"token": token, "text": "hello THERE (fixed)"},
    )
    assert replay.json()["verified"] is True  # unchanged text kept verification
    assert replay.json()["corrected"] is True

    # Reverting: text equal to the pipeline rendering (or empty) clears it.
    revert = client.post(
        f"/review/{run_id}/segments/{segs[0]}/text",
        data={"token": token, "text": "  "},
    )
    assert revert.json()["corrected"] is False
    assert "hello THERE (fixed)" not in client.get(f"/runs/{run_id}/transcript").text


def test_segment_review_writes_are_claim_gated(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    segs = _segment_ids(session_factory, run_id)
    claim_token(client, run_id)  # someone else holds the claim

    wrong = client.post(
        f"/review/{run_id}/segments/{segs[0]}/verify",
        data={"token": str(uuid.uuid4())},
    )
    assert wrong.status_code == 409


def test_segment_review_rejects_cross_run_segment(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_a = _seed_run_with_confidences(session, media_root, [None])
        run_b = _seed_run_with_confidences(session, media_root, [None])
    seg_b = _segment_ids(session_factory, run_b)[0]
    token_a = claim_token(client, run_a)

    resp = client.post(
        f"/review/{run_a}/segments/{seg_b}/verify", data={"token": token_a}
    )
    assert resp.status_code == 404


def test_review_transcript_mounts_stepper_with_token_and_props(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Issue #53: the claim-gated review surface reuses the token from ?token= (no
    # re-claim) and mounts the review-stepper island with the write token + the
    # N-of-M counter + per-segment review state in its props.
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # 3 segments
    token = claim_token(client, run_id)

    page = client.get(f"/review/{run_id}/transcript", params={"token": token})
    assert page.status_code == 200
    assert 'data-island="review-stepper"' in page.text
    assert "Claimed by you" in page.text
    match = re.search(r"data-props='([^']*)'", page.text)
    assert match is not None
    props = json.loads(match.group(1))
    assert props["reviewToken"] == token  # the SAME token, reused not re-minted
    assert props["initialProgress"] == {"verified": 0, "total": 3}
    # Every segment carries its write id + review flags for the loop.
    assert all(s["segmentId"] is not None for s in props["segments"])
    assert all(s["verified"] is False for s in props["segments"])
    # JS-off fallback: a real verify form per unverified segment.
    assert page.text.count(f"/review/{run_id}/segments/") >= 3


def test_review_transcript_degrades_read_only_without_token(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A stale/absent token renders read-only (no re-claim, honest copy), mirroring
    # the workbench GET. reviewToken is null so the island disables writes.
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    claim_token(client, run_id)  # a DIFFERENT tab holds the live claim

    stale = client.get(
        f"/review/{run_id}/transcript", params={"token": str(uuid.uuid4())}
    )
    assert stale.status_code == 200
    assert "Not claimed by this tab" in stale.text
    props = json.loads(re.search(r"data-props='([^']*)'", stale.text).group(1))  # type: ignore[union-attr]
    assert props["reviewToken"] is None
    # No verify forms are offered when this tab cannot write.
    assert f"/review/{run_id}/segments/" not in stale.text


def test_verify_form_navigation_redirects_back_to_review(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # The JS-off fallback POSTs a plain form (Accept: text/html); the write route
    # content-negotiates and 303s back to the review page instead of dumping JSON,
    # so the server-rendered list verifies for real. The island (Accept: json)
    # still gets JSON — asserted by every other verify test in this file.
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    segs = _segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    resp = client.post(
        f"/review/{run_id}/segments/{segs[0]}/verify",
        data={"token": token, "verified": "true"},
        headers={"accept": "text/html"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/review/{run_id}/transcript?token={token}"
    # The redirect is not cosmetic — the segment is verified.
    page = client.get(f"/review/{run_id}/transcript", params={"token": token})
    props = json.loads(re.search(r"data-props='([^']*)'", page.text).group(1))  # type: ignore[union-attr]
    assert props["initialProgress"]["verified"] == 1
