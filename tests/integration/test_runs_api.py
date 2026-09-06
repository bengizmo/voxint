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
from voxint.export import format_timespan

CREDS = ("reviewer", "s3cret")
SPACE = "titanet-large-v2"


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
    language: str | None = None,
    language_probability: float | None = None,
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
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
        detected_language=language,
        detected_language_probability=language_probability,
    )
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
    match = re.search(r'href="([^"]+)"[^>]*>Older', first.text)
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
    # Scoped to the ledger table: cards above it (pipeline models, glossary,
    # detected language) legitimately contain these stage words in prose.
    ledger = body[body.index("Stage ledger") :]
    assert ledger.index("prepare") < ledger.index("transcribe")
    # V3 grid-table: the stage ledger renders as a CSS-grid data view with an
    # aria-labelled region.
    assert 'class="grid-table"' in body
    assert 'aria-label="Stage ledger"' in body
    assert "STAGE" in body
    # Status is never colour-only on the detail page: the run-status pill and each
    # stage-status pill carry their state word as text inside the span.
    assert 'class="pill completed">completed</span>' in body
    assert 'class="pill failed">failed</span>' in body


def test_run_detail_unknown_run_404(client: TestClient) -> None:
    assert client.get(f"/runs/{uuid.uuid4()}").status_code == 404


def _set_current_stage(
    session_factory: sessionmaker[Session], run_id: uuid.UUID, stage: str
) -> None:
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        run.current_stage = stage
        session.commit()


def test_failed_model_stage_shows_service_hint(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # A run failed at a model-service stage gets static guidance (start a
    # compute tier, then requeue) — no live probe on this page.
    with session_factory() as session:
        run_id = make_run(session, status=RunStatus.FAILED, labels=["S0"])
    _set_current_stage(session_factory, run_id, "transcribe")
    body = client.get(f"/runs/{run_id}").text
    assert "needs the model services" in body
    assert "compose.cpu.yaml" in body and "compose.gpu.yaml" in body


def test_failed_non_model_stage_has_no_service_hint(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(session, status=RunStatus.FAILED, labels=["S0"])
    _set_current_stage(session_factory, run_id, "acquire")
    body = client.get(f"/runs/{run_id}").text
    assert "needs the model services" not in body


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
    body = client.get("/runs").text
    assert f'href="/runs/{run_id}"' in body
    # V3 grid-table: the runs list renders as a CSS-grid data view with an
    # aria-labelled region.
    assert 'class="grid-table"' in body
    assert 'aria-label="Runs"' in body
    assert "RUN" in body
    # Status is never colour-only here either: the humanized label sits inside the
    # status pill span (the run is COMPLETED by default).
    assert 'class="pill completed">Completed</span>' in body


# --- transcript (shared resolver-attributed presenter) ------------------------


def test_parse_transcript_text_defaults_and_rejects() -> None:
    # Default is now 'corrected' (operator-effective) — issue #58. It renders
    # identically to 'enhanced' until a correction exists, so this is a contract
    # change, not a behavior change for uncorrected runs.
    assert parse_transcript_text(None) is TranscriptText.CORRECTED
    assert parse_transcript_text("") is TranscriptText.CORRECTED
    assert parse_transcript_text("corrected") is TranscriptText.CORRECTED
    assert parse_transcript_text("raw") is TranscriptText.RAW
    assert parse_transcript_text("enhanced") is TranscriptText.ENHANCED
    with pytest.raises(ValueError):
        parse_transcript_text("sideways")



# Interactive transcript page tests (variants, island props, fallback lines,
# color classes, text validation) were removed in issue #158: the interactive
# transcript stepper is retired and redirects to the media editor.


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



# test_export_bytes_exact_and_html_shares_lines and test_transcript_html_escaped
# were removed in issue #158 (interactive transcript retired; export byte-exact
# tests are covered by test_export_formats_content_types_and_payloads).


# --- read mode + Markdown export (issue #65) ----------------------------------


def test_transcript_read_mode_groups_and_drops_island(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Read mode renders paragraphs from the SAME resolver + the SAME grouping the
    # Markdown export uses: adjacent same-speaker lines merge into one paragraph,
    # a speaker change starts a new one, and no player island is mounted.
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0", "S1"],  # unresolved → speaker == bare label (predictable)
            segments=[
                ("S0", "hello", None),  # [0, 8]
                ("S0", "there", None),  # [10, 18]  merges with the line above
                ("S1", "bye", None),  # [20, 28]  new speaker → new paragraph
            ],
        )
    resp = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "timestamps": "false"}
    )
    assert resp.status_code == 200
    body = resp.text
    # Two paragraphs: S0's two lines merged, S1 alone.
    assert body.count("<h2>S0</h2>") == 1
    assert body.count("<h2>S1</h2>") == 1
    assert "hello there" in body  # joined with a single ASCII space
    # Read mode is pure server-rendered HTML — no player island, no props JSON.
    assert 'data-island="transcript-player"' not in body
    # timestamps=false → no time-range bracket anywhere in the reading copy.
    assert "[00:00:" not in body


def test_transcript_read_mode_timestamps_toggle(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0"],
            segments=[("S0", "hello", None), ("S0", "there", None)],  # [0,8]+[10,18]
        )
    # timestamps=true → the merged paragraph opens with the full run span.
    on = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "timestamps": "true"}
    ).text
    assert format_timespan(0.0, 18.0) in on
    # timestamps=false → no bracketed range.
    off = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "timestamps": "false"}
    ).text
    assert format_timespan(0.0, 18.0) not in off
    assert "[00:00:" not in off


def test_transcript_read_mode_preserves_query_in_toggles(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(
            session, labels=["S0"], segments=[("S0", "raw hi", "enh hi")]
        )
    body = client.get(
        f"/runs/{run_id}/transcript",
        params={"read": "1", "timestamps": "false", "text": "raw"},
    ).text
    base = f"/runs/{run_id}/transcript"
    # Exit drops read but keeps the text variant; the timestamps toggle flips the
    # flag while keeping read + variant; variant tabs keep read + timestamps.
    assert f'href="{base}?text=raw"' in body
    assert f'href="{base}?text=raw&read=1&timestamps=true"' in body
    assert f'href="{base}?text=raw&read=1&timestamps=false"' in body
    assert 'aria-current="page">raw' in body



# test_transcript_read_mode_entry_link_on_normal_view and
# test_export_menu_read_link_keeps_active_variant were removed in issue #158
# (interactive transcript retired; read mode entry is now via the editor).


def test_transcript_read_mode_attribution_matches_export(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # Read mode shares attributed_transcript with the exports: the SAME
    # corrected→enhanced→raw precedence and the SAME speaker attribution, so a
    # grounded name and an excluded label read identically across both surfaces.
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
    read = client.get(f"/runs/{run_id}/transcript", params={"read": "1"}).text
    assert f"<h2>{s0}</h2>" in read  # grounded → display name
    assert "s0 enh" in read  # default corrected → enhanced fallback
    assert "<h2>(excluded) S1</h2>" in read
    assert "s1 raw" in read  # enhanced NULL → raw fallback
    export = client.get(f"/review/{run_id}/export.txt").text
    assert s0 in export and "(excluded) S1" in export
    # A raw read view ignores the enhancement, exactly like the raw export.
    raw = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "text": "raw"}
    ).text
    assert "s0 raw" in raw and "s0 enh" not in raw


def test_transcript_read_mode_escapes_hostile_text(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    payload = "<script>alert(1)</script>"
    with session_factory() as session:
        run_id = make_run(
            session, labels=["S0"], segments=[("S0", payload, None)]
        )
    body = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "text": "raw"}
    ).text
    assert payload not in body  # Jinja autoescape neutralizes raw HTML
    assert "&lt;script&gt;" in body


def test_transcript_read_mode_empty_run(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(session, labels=[])  # no segments
    body = client.get(f"/runs/{run_id}/transcript", params={"read": "1"}).text
    assert "No transcript segments for this run." in body
    assert "<h2>" not in body


def test_transcript_read_mode_rejects_bad_text(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        run_id = make_run(session, labels=["S0"], grounded=["S0"])
    resp = client.get(
        f"/runs/{run_id}/transcript", params={"read": "1", "text": "sideways"}
    )
    assert resp.status_code == 422


def test_export_md_route_bytes_and_media_type(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    # The Markdown route funnels through the same render_transcript as every other
    # export; assert byte-exact output (headings, merged blockquote, time range,
    # trailing newline) and the Markdown media type.
    with session_factory() as session:
        run_id = make_run(
            session,
            labels=["S0", "S1"],
            segments=[("S0", "hello", None), ("S1", "bye", None)],
        )
    ts0 = format_timespan(0.0, 8.0)
    ts1 = format_timespan(10.0, 18.0)
    expected = f"## S0\n\n> {ts0} hello\n\n## S1\n\n> {ts1} bye\n"
    resp = client.get(f"/review/{run_id}/export.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    assert resp.content == expected.encode()
    # ?timestamps=false drops the per-paragraph range for a clean reading copy.
    clean = client.get(
        f"/review/{run_id}/export.md", params={"timestamps": "false"}
    ).text
    assert ts0 not in clean and "hello" in clean


def test_list_runs_sidecar_title_wins_over_scraped(
    session_factory: sessionmaker[Session],
) -> None:
    """/runs display title precedence (issue #104): the frozen sidecar title
    beats the acquisition-metadata title; without one the scraped title holds."""
    from voxint.db.models import MediaSourceMetadata

    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        session.add(
            MediaSourceMetadata(
                media_item_id=media.id,
                source_kind="ytdlp",
                title="Scraped title",
                raw={"id": "abc"},
                raw_schema_version=1,
                acquired_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            )
        )
        with_sidecar = PipelineRun(
            media_item_id=media.id,
            status=RunStatus.COMPLETED.value,
            sidecar={"title": "Operator title"},
        )
        without_sidecar = PipelineRun(
            media_item_id=media.id, status=RunStatus.COMPLETED.value
        )
        session.add_all([with_sidecar, without_sidecar])
        session.commit()
        page = list_runs(
            session, status=None, review=None, cursor=None, page_size=10
        )
        titles = {item.run_id: item.title for item in page.items}
        assert titles[with_sidecar.id] == "Operator title"
        assert titles[without_sidecar.id] == "Scraped title"


def _add_media_section(body: str) -> str:
    """The rendered "Add media" section (issue #117 Phase C), start to close."""
    start = body.index('id="add-media"')
    end = body.index("</section>", start)
    return body[start:end]


def test_runs_add_media_section_wraps_upload_and_url_fetch(client: TestClient) -> None:
    """Phase C (issue #117): the Runs header exposes a named "Add media" section —
    the dashboard "Add audio" card's link target — holding both ways in, so
    elevating the upload never demotes the URL/video workflow."""
    body = client.get("/runs").text
    # The anchor the dashboard task card points at (/runs#add-media).
    assert 'id="add-media"' in body
    section = _add_media_section(body)
    # V3: section title is uppercase with oc-section-title class.
    assert "ADD MEDIA" in section
    # Both affordances live inside the section: the upload form...
    assert 'action="/submit"' in section
    # ...and, with URL ingestion enabled (the default), the fetch form.
    assert 'action="/fetch"' in section


def test_runs_add_media_section_holds_upload_when_url_fetch_disabled(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """With URL ingestion off the section still holds the upload; the fetch form is
    gone but its honest disabled notice stays inside the section (issue #117)."""
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        ytdlp_enabled=False,
    )
    test_client = TestClient(create_app(settings=settings, session_factory=session_factory))
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    section = _add_media_section(test_client.get("/runs").text)
    assert 'action="/submit"' in section
    assert 'action="/fetch"' not in section
    assert "URL ingestion is disabled." in section
