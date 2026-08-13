"""The /runs execution-history browser end to end against real Postgres.

Covers resolver parity of the SQL classification, the orthogonal status/review
filters, keyset pagination (including identical-timestamp ties), and the route
rendering + error mapping.
"""

import html
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.resolver import (
    Resolution,
    label_count,
    label_states,
    unresolved_label_count,
    unresolved_label_exists,
)
from voxint.adjudication.transcript import TranscriptText, parse_transcript_text
from voxint.api.app import create_app
from voxint.api.runs_query import Cursor, list_runs
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    ArtifactKind,
    AudioArtifact,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    StageRun,
    TranscriptSegment,
)

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v1"


def unit(dim: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    vector[dim % EMBEDDING_DIM] = 1.0
    return vector


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        runs_page_size=2,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def make_run(
    session: Session,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    created_at: datetime | None = None,
    labels: Iterable[str] = (),
    decided: Iterable[str] = (),
    grounded: Iterable[str] = (),
    ungrounded: Iterable[str] = (),
    orphan_decisions: Iterable[str] = (),
    orphan_grounded: Iterable[str] = (),
    claim: str | None = None,
    stages: Iterable[dict[str, Any]] = (),
    segments: Iterable[tuple[str | None, str, str | None]] = (),
    audio: bool = False,
) -> uuid.UUID:
    """Seed one media item + run with controllable review state.

    ``labels`` become diarization turns. ``decided`` get a human decision,
    ``grounded``/``ungrounded`` a cosine proposal. ``orphan_*`` attach evidence
    for a label with NO turn (must be ignored). ``claim`` ∈ {None, live, expired}.
    ``stages`` are ``StageRun`` kwargs (the attempt ledger); ``segments`` are
    ``(diarization_label, raw_text, enhanced_text)`` transcript rows; ``audio``
    attaches a PREPROCESSED_AUDIO artifact row (path only — no file written).
    """
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id, status=status.value)
    if created_at is not None:
        run.created_at = created_at
    if claim is not None:
        now = datetime.now(tz=UTC)
        run.review_claim_token = uuid.uuid4()
        run.review_claimed_by = "reviewer"
        run.review_claimed_at = now
        run.review_claim_expires_at = (
            now + timedelta(hours=1) if claim == "live" else now - timedelta(hours=1)
        )
    session.add(run)
    session.flush()

    for index, label in enumerate(labels):
        session.add(
            DiarizationTurn(
                pipeline_run_id=run.id,
                turn_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                label=label,
                embedding=unit(index),
                embedding_space=SPACE,
            )
        )
    for label in (*decided, *orphan_decisions):
        session.add(
            AdjudicationDecision(
                pipeline_run_id=run.id,
                diarization_label=label,
                decision="exclude",
                operator="reviewer",
                idempotency_key=uuid.uuid4().hex,
            )
        )
    for label in (*grounded, *ungrounded, *orphan_grounded):
        speaker = Speaker(display_name=f"spk-{uuid.uuid4().hex[:10]}")
        session.add(speaker)
        session.flush()
        session.add(
            SpeakerAssignment(
                pipeline_run_id=run.id,
                diarization_label=label,
                speaker_id=speaker.id,
                method="cosine",
                confidence=0.9,
                grounded=label not in set(ungrounded),
            )
        )
    for spec in stages:
        session.add(StageRun(pipeline_run_id=run.id, **spec))
    for index, (seg_label, raw_text, enhanced_text) in enumerate(segments):
        session.add(
            TranscriptSegment(
                pipeline_run_id=run.id,
                segment_index=index,
                start_seconds=float(index * 10),
                end_seconds=float(index * 10 + 8),
                raw_text=raw_text,
                enhanced_text=enhanced_text,
                diarization_label=seg_label,
            )
        )
    if audio:
        session.add(
            AudioArtifact(
                pipeline_run_id=run.id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path=f"artifacts/{run.id}/normalized.wav",
            )
        )
    session.commit()
    return run.id


# --- resolver parity: the SQL classification must match label_states() --------


def test_sql_classification_matches_resolver(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        cases = {
            "one_unresolved": make_run(
                session, labels=["S0", "S1"], grounded=["S0"]
            ),  # S1 needs a ruling
            "fully_resolved": make_run(
                session, labels=["S0", "S1"], decided=["S0"], grounded=["S1"]
            ),
            "ungrounded_is_unresolved": make_run(
                session, labels=["S0"], ungrounded=["S0"]
            ),
            "zero_labels": make_run(session, labels=[]),
            "orphan_decision_ignored": make_run(
                session, labels=["S0"], orphan_decisions=["GHOST"]
            ),  # GHOST has no turn → still 1 unresolved
            "orphan_grounded_ignored": make_run(
                session, labels=["S0"], grounded=["S0"], orphan_grounded=["GHOST"]
            ),
        }

    with session_factory() as session:
        for name, run_id in cases.items():
            states = label_states(session, run_id)
            py_unresolved = sum(
                1 for s in states if s.resolution is Resolution.UNRESOLVED
            )
            py_labels = len(states)

            sql_exists = session.scalar(select(unresolved_label_exists(run_id)))
            sql_unresolved = session.scalar(select(unresolved_label_count(run_id)))
            sql_labels = session.scalar(select(label_count(run_id)))

            assert sql_exists is (py_unresolved > 0), name
            assert sql_unresolved == py_unresolved, name
            assert sql_labels == py_labels, name


# --- filters ------------------------------------------------------------------


def test_review_needed_and_resolved_partition_completed(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        needs = make_run(session, labels=["S0", "S1"], grounded=["S0"])
        resolved = make_run(session, labels=["S0"], decided=["S0"])
        zero = make_run(session, labels=[])

    needed_page = client.get("/runs", params={"review": "needed"})
    assert needed_page.status_code == 200
    assert str(needs)[:8] in needed_page.text
    assert str(resolved)[:8] not in needed_page.text
    assert str(zero)[:8] not in needed_page.text

    resolved_page = client.get("/runs", params={"review": "resolved"}).text
    assert str(resolved)[:8] in resolved_page
    assert str(zero)[:8] in resolved_page  # zero-label completed is the complement
    assert str(needs)[:8] not in resolved_page


def test_shared_label_evidence_does_not_bleed_across_runs(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Regression: two completed runs share label "SHARED"; only one is grounded.
    # A mis-correlated EXISTS would let the grounded run's evidence resolve the
    # other run's identical label, hiding a run that genuinely needs a ruling.
    with session_factory() as session:
        needs = make_run(session, labels=["SHARED"])  # no evidence at all
        grounded = make_run(session, labels=["SHARED"], grounded=["SHARED"])

    needed = client.get("/runs", params={"review": "needed"}).text
    assert str(needs)[:8] in needed
    assert str(grounded)[:8] not in needed

    resolved = client.get("/runs", params={"review": "resolved"}).text
    assert str(grounded)[:8] in resolved
    assert str(needs)[:8] not in resolved


def test_zero_label_run_shows_no_speakers_badge(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        make_run(session, labels=[])
    body = client.get("/runs", params={"review": "resolved"}).text
    assert "no speakers" in body


def test_source_path_is_html_escaped(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Guards the autoescape invariant against a future "linkify the path" change.
    payload = "incoming/<script>alert(1)</script>.wav"
    with session_factory() as session:
        media = MediaItem(source_path=payload)
        session.add(media)
        session.flush()
        session.add(PipelineRun(media_item_id=media.id, status=RunStatus.FAILED.value))
        session.commit()
    body = client.get("/runs").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_status_filter_and_orthogonal_empty_combo(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        failed = make_run(session, status=RunStatus.FAILED, labels=["S0"])
        completed = make_run(session, labels=["S0"], grounded=["S0"])

    failed_page = client.get("/runs", params={"status": "failed"}).text
    assert str(failed)[:8] in failed_page
    assert str(completed)[:8] not in failed_page

    # status=failed & review=needed is intentionally empty (needed ⟹ completed).
    contradiction = client.get("/runs", params={"status": "failed", "review": "needed"})
    assert contradiction.status_code == 200
    assert str(failed)[:8] not in contradiction.text
    assert "No runs match" in contradiction.text


def test_claimed_filter_only_live_claims(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        live = make_run(session, labels=["S0"], grounded=["S0"], claim="live")
        expired = make_run(session, labels=["S0"], grounded=["S0"], claim="expired")

    body = client.get("/runs", params={"review": "claimed"}).text
    assert str(live)[:8] in body
    assert str(expired)[:8] not in body
    assert "claimed" in body


# --- keyset pagination --------------------------------------------------------


def _walk(session: Session) -> list[uuid.UUID]:
    seen: list[uuid.UUID] = []
    cursor: Cursor | None = None
    for _ in range(100):  # guard against a cursor that never terminates
        page = list_runs(
            session, status=None, review=None, cursor=cursor, page_size=2
        )
        seen.extend(item.run_id for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    return seen


def test_keyset_walks_all_runs_newest_first(
    session_factory: sessionmaker[Session],
) -> None:
    base = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    with session_factory() as session:
        ids = [
            make_run(session, labels=["S0"], created_at=base + timedelta(minutes=i))
            for i in range(5)
        ]
    with session_factory() as session:
        walked = _walk(session)
    assert walked == list(reversed(ids))  # newest first, every run exactly once


def test_keyset_breaks_identical_timestamp_ties(
    session_factory: sessionmaker[Session],
) -> None:
    same = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    with session_factory() as session:
        ids = {make_run(session, labels=["S0"], created_at=same) for _ in range(5)}
    with session_factory() as session:
        walked = _walk(session)
    assert len(walked) == 5
    assert set(walked) == ids  # no drops, no duplicates across the tie boundary


def test_route_older_link_advances(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    base = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    with session_factory() as session:
        for i in range(3):
            make_run(session, labels=["S0"], created_at=base + timedelta(minutes=i))

    first = client.get("/runs")
    assert first.status_code == 200
    assert "Older →" in first.text
    # The "Older" anchor, not the nav links; unescape &amp; back to & for the request.
    match = re.search(r'href="([^"]+)">Older', first.text)
    assert match is not None
    href = html.unescape(match.group(1))
    assert href.startswith("/runs?cursor=")

    second = client.get(href)
    assert second.status_code == 200
    # Page 1 held 2 rows; page 2 holds the last and has no further page.
    assert "Older →" not in second.text


# --- error mapping ------------------------------------------------------------


def test_invalid_cursor_is_400(client: TestClient) -> None:
    assert client.get("/runs", params={"cursor": "!!garbage!!"}).status_code == 400


@pytest.mark.parametrize(
    "params",
    [{"status": "nonsense"}, {"review": "someday"}],
)
def test_invalid_filter_is_422(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/runs", params=params).status_code == 422


def test_empty_filters_render_all(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        make_run(session, labels=["S0"], grounded=["S0"])
    # A blank select submission sends status=&review= — must mean "all", not 422.
    resp = client.get("/runs", params={"status": "", "review": ""})
    assert resp.status_code == 200


# --- run detail + stage ledger ------------------------------------------------


def test_run_detail_shows_stage_ledger(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    base = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0"],
            grounded=["S0"],
            stages=[
                {
                    "stage": "prepare",
                    "status": "completed",
                    "attempt": 1,
                    "worker_id": "worker-a",
                    "started_at": base,
                    "finished_at": base + timedelta(seconds=5),
                },
                {
                    "stage": "transcribe",
                    "status": "failed",
                    "attempt": 1,
                    "worker_id": "worker-a",
                    "started_at": base + timedelta(seconds=10),
                    "error": "asr exploded",
                },
                {
                    "stage": "transcribe",
                    "status": "completed",
                    "attempt": 2,
                    "worker_id": "worker-b",
                    "started_at": base + timedelta(seconds=20),
                },
            ],
        )
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    body = resp.text
    assert "prepare" in body and "transcribe" in body
    assert "asr exploded" in body  # the failed attempt's error surfaces
    assert "worker-a" in body and "worker-b" in body  # both attempts' workers
    # Chronological by started_at — prepare precedes the transcribe attempts.
    assert body.index("prepare") < body.index("transcribe")


def test_run_detail_unknown_run_404(client: TestClient) -> None:
    assert client.get(f"/runs/{uuid.uuid4()}").status_code == 404


def test_run_detail_present_only_links(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        full = make_run(
            session,
            labels=["S0"],
            grounded=["S0"],
            audio=True,
            segments=[("S0", "hello", None)],
        )
        bare = make_run(session, status=RunStatus.FAILED, labels=[])

    full_body = client.get(f"/runs/{full}").text
    assert f"/media/{full}" in full_body
    assert f"/runs/{full}/transcript" in full_body
    assert f"/review/{full}" in full_body  # completed → adjudication link

    bare_body = client.get(f"/runs/{bare}").text
    assert f"/media/{bare}" not in bare_body
    assert f"/runs/{bare}/transcript" not in bare_body
    assert f"/review/{bare}" not in bare_body  # failed → no adjudication link
    assert "No audio or transcript yet." in bare_body


def test_runs_list_links_to_detail(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(session, labels=["S0"], grounded=["S0"])
    assert f'href="/runs/{run_id}"' in client.get("/runs").text


# --- transcript (shared resolver-attributed presenter) ------------------------


def test_parse_transcript_text_defaults_and_rejects() -> None:
    assert parse_transcript_text(None) is TranscriptText.ENHANCED
    assert parse_transcript_text("") is TranscriptText.ENHANCED
    assert parse_transcript_text("raw") is TranscriptText.RAW
    assert parse_transcript_text("enhanced") is TranscriptText.ENHANCED
    with pytest.raises(ValueError):
        parse_transcript_text("sideways")


def test_transcript_variants_select_text(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0", "S1"],
            grounded=["S0"],
            decided=["S1"],  # exclude
            segments=[
                ("S0", "raw hello", "enhanced hello"),
                ("S1", "raw goodbye", None),  # no enhancement → raw fallback
            ],
        )
    enhanced = client.get(f"/runs/{run_id}/transcript", params={"text": "enhanced"})
    assert enhanced.status_code == 200
    assert "enhanced hello" in enhanced.text
    assert "raw goodbye" in enhanced.text  # enhanced NULL → falls back to raw
    assert "(excluded) S1" in enhanced.text
    # The requested variant's tab is marked current; the other is not.
    assert 'aria-current="page">enhanced' in enhanced.text
    assert 'aria-current="page">raw' not in enhanced.text

    raw = client.get(f"/runs/{run_id}/transcript", params={"text": "raw"}).text
    assert "raw hello" in raw
    assert "enhanced hello" not in raw  # raw view ignores enhancement
    assert "raw goodbye" in raw


def test_transcript_defaults_to_enhanced(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0"],
            grounded=["S0"],
            segments=[("S0", "raw only", "enhanced only")],
        )
    body = client.get(f"/runs/{run_id}/transcript").text  # no text= → enhanced
    assert "enhanced only" in body
    assert "raw only" not in body


def test_transcript_attribution_and_export_agree(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0", "S1", "S2", "S3"],
            grounded=["S0"],  # grounded cosine → speaker name
            decided=["S1"],  # exclude → "(excluded) S1"
            segments=[
                ("S0", "s0 raw", "s0 enh"),
                ("S1", "s1 raw", None),
                ("S2", "s2 raw", None),
                ("S3", "s3 raw", None),
                ("GHOST", "ghost raw", None),  # label with no turn → state None
                (None, "nameless raw", None),  # NULL label → "(no speaker)"
            ],
        )
        session.add(  # S2 gets an 'unknown' ruling; S3 stays unresolved
            AdjudicationDecision(
                pipeline_run_id=run_id,
                diarization_label="S2",
                decision="unknown",
                operator="reviewer",
                idempotency_key=uuid.uuid4().hex,
            )
        )
        session.commit()
        s0_name = session.execute(
            select(Speaker.display_name)
            .join(SpeakerAssignment, SpeakerAssignment.speaker_id == Speaker.id)
            .where(
                SpeakerAssignment.pipeline_run_id == run_id,
                SpeakerAssignment.diarization_label == "S0",
            )
        ).scalar_one()

    body = client.get(f"/runs/{run_id}/transcript", params={"text": "enhanced"}).text
    assert s0_name in body  # GROUNDED_COSINE
    assert "(excluded) S1" in body  # HUMAN_EXCLUDE
    assert "Unknown (S2)" in body  # HUMAN_UNKNOWN
    assert "S3:" in body  # UNRESOLVED → bare label
    assert "GHOST:" in body  # segment label with no turn → state None
    assert "(no speaker)" in body  # NULL diarization label

    # Shared presenter: export.txt attributes every label identically.
    export = client.get(f"/review/{run_id}/export.txt").text
    for speaker in (s0_name, "(excluded) S1", "Unknown (S2)", "(no speaker)"):
        assert speaker in export


def test_transcript_unknown_text_is_422(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(session, labels=["S0"], grounded=["S0"])
    resp = client.get(f"/runs/{run_id}/transcript", params={"text": "sideways"})
    assert resp.status_code == 422


def test_transcript_unknown_run_404(client: TestClient) -> None:
    assert client.get(f"/runs/{uuid.uuid4()}/transcript").status_code == 404


def test_transcript_unknown_run_beats_bad_text(client: TestClient) -> None:
    # Run lookup precedes text parsing: an unknown run with invalid text is a
    # 404 (run-not-found), never a 422 (bad text) — the spec's ordering claim.
    resp = client.get(f"/runs/{uuid.uuid4()}/transcript", params={"text": "sideways"})
    assert resp.status_code == 404


def test_run_detail_multi_artifact_hides_audio_link(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Two PREPROCESSED_AUDIO rows: /media requires exactly one and 404s, so the
    # detail page must not offer an Audio link it cannot serve. audio_available
    # reuses the same exactly-one predicate, so the link is withheld.
    with session_factory() as session:
        run_id = make_run(session, labels=["S0"], grounded=["S0"], audio=True)
        session.add(
            AudioArtifact(
                pipeline_run_id=run_id,
                kind=ArtifactKind.PREPROCESSED_AUDIO.value,
                path=f"artifacts/{run_id}/normalized-2.wav",
            )
        )
        session.commit()
    assert f"/media/{run_id}" not in client.get(f"/runs/{run_id}").text


def test_export_bytes_exact_and_html_shares_lines(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Byte-exact export (the presenter refactor must not shift a single space or
    # the trailing newline), and the same attributed lines land in the HTML view.
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0", "S1"],
            grounded=["S0"],
            decided=["S1"],  # exclude
            segments=[("S0", "s0 raw", "s0 enh"), ("S1", "s1 raw", None)],
        )
        s0 = session.execute(
            select(Speaker.display_name)
            .join(SpeakerAssignment, SpeakerAssignment.speaker_id == Speaker.id)
            .where(
                SpeakerAssignment.pipeline_run_id == run_id,
                SpeakerAssignment.diarization_label == "S0",
            )
        ).scalar_one()
    # Seg 0 @ [0,8] enhanced; seg 1 @ [10,18] excluded, enhanced NULL → raw.
    expected = (
        f"[{0.0:9.2f} {8.0:9.2f}] {s0}: s0 enh\n"
        f"[{10.0:9.2f} {18.0:9.2f}] (excluded) S1: s1 raw\n"
    )
    export = client.get(f"/review/{run_id}/export.txt")
    assert export.content == expected.encode()  # byte-exact, incl. trailing NL

    body = client.get(f"/runs/{run_id}/transcript", params={"text": "enhanced"}).text
    assert s0 in body and "s0 enh" in body
    assert "(excluded) S1" in body and "s1 raw" in body


def test_transcript_html_escaped(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    payload = "<script>alert(1)</script>"
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0"],
            grounded=["S0"],
            segments=[("S0", payload, None)],
        )
    body = client.get(f"/runs/{run_id}/transcript", params={"text": "raw"}).text
    assert payload not in body
    assert "&lt;script&gt;" in body
