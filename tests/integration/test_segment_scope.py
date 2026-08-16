"""Two-scope relabel: the this-segment override (issue #54 Phase B).

The correctness gate. Segment-scope rulings must override exactly their own
segment, never leak into label-scope resolution (unresolved counts, queue,
label_states, speaker search), reset cleanly via INHERIT (live, not frozen),
survive later label rulings, canonicalize through merge tombstones, and keep the
HTML transcript and text export byte-identical.
"""

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_review_api import CREDS, claim_token, seed_run
from voxint.adjudication.resolver import (
    Resolution,
    adjudication_queue,
    effective_decisions,
    label_states,
)
from voxint.api.app import create_app
from voxint.config import Settings
from voxint.db.models import MediaItem, PipelineRun, RunStatus, Speaker, TranscriptSegment
from voxint.speakers.roster import merge_speakers

_CSRF_KEY = "review-api-test-csrf-key"
HX = {"HX-Request": "true"}


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def client(session_factory: sessionmaker[Session], media_root: Path) -> TestClient:
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


def segment_ids(session_factory: sessionmaker[Session], run_id: uuid.UUID) -> dict[int, uuid.UUID]:
    """segment_index -> id for a run."""
    with session_factory() as session:
        rows = session.execute(
            select(TranscriptSegment.segment_index, TranscriptSegment.id).where(
                TranscriptSegment.pipeline_run_id == run_id
            )
        ).all()
    return {int(idx): sid for idx, sid in rows}


def bare_run_segment(session_factory: sessionmaker[Session]) -> uuid.UUID:
    """A minimal second run with one labelled segment; returns that segment id."""
    with session_factory() as session:
        media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
        session.add(media)
        session.flush()
        run = PipelineRun(media_item_id=media.id, status=RunStatus.COMPLETED.value)
        session.add(run)
        session.flush()
        seg = TranscriptSegment(
            pipeline_run_id=run.id,
            segment_index=0,
            start_seconds=0.0,
            end_seconds=1.0,
            raw_text="elsewhere",
            diarization_label="Z0",
        )
        session.add(seg)
        session.commit()
        return seg.id


def add_speaker(session_factory: sessionmaker[Session], name: str) -> uuid.UUID:
    with session_factory() as session:
        speaker = Speaker(display_name=name)
        session.add(speaker)
        session.commit()
        return speaker.id


def relabel(client: TestClient, run_id, seg_id, token, *, action, speaker_id=None):
    data = {"token": token, "nonce": uuid.uuid4().hex, "action": action}
    if speaker_id is not None:
        data["speaker_id"] = str(speaker_id)
    return client.post(
        f"/review/{run_id}/segments/{seg_id}/relabel", data=data, headers=HX
    )


def test_segment_override_hits_only_its_own_segment(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # S0 grounded -> "Known Voice", 2 segments
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    # Override S0's FIRST segment ("hello there", index 0); the other S0 segment
    # ("how are you", index 2) must be untouched.
    resp = relabel(client, run_id, segs[0], token, action="assign", speaker_id=other)
    assert resp.status_code == 200
    export = client.get(f"/review/{run_id}/export.txt").text
    assert "Other Person: hello there" in export
    assert "Known Voice: how are you" in export


def test_segment_override_does_not_resolve_the_label(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)  # S1 is UNRESOLVED (llm hint only)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)
    # S1's only segment is index 1 ("hi back").
    relabel(client, run_id, segs[1], token, action="assign", speaker_id=other)

    with session_factory() as session:
        states = {s.label: s for s in label_states(session, run_id)}
        # The label is STILL unresolved — a segment override is not a label ruling.
        assert states["S1"].resolution is Resolution.UNRESOLVED
        # And the segment row never enters label-effective.
        assert "S1" not in effective_decisions(session, run_id)
        # The run stays in the adjudication queue (S1 still needs a label ruling).
        assert any(e.run_id == run_id for e in adjudication_queue(session))


def test_inherit_resets_live_not_frozen(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    later = add_speaker(session_factory, "Later Label Speaker")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    # Override S0 seg0 to Other Person.
    relabel(client, run_id, segs[0], token, action="assign", speaker_id=other)
    assert "Other Person: hello there" in client.get(f"/review/{run_id}/export.txt").text

    # INHERIT resets it to follow the label (currently grounded -> Known Voice).
    relabel(client, run_id, segs[0], token, action="inherit")
    export = client.get(f"/review/{run_id}/export.txt").text
    assert "Known Voice: hello there" in export
    assert "Other Person" not in export

    # A LATER whole-label ruling must now be reflected in the inherited segment —
    # inherit is a live fall-through, not a frozen copy of the old resolution.
    client.post(
        f"/review/{run_id}/labels/S0/decision",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(later),
        },
        headers=HX,
    )
    export = client.get(f"/review/{run_id}/export.txt").text
    assert "Later Label Speaker: hello there" in export


def test_later_label_ruling_leaves_segment_override_intact(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    # Override S0 seg0, THEN exclude the whole S0 label.
    relabel(client, run_id, segs[0], token, action="assign", speaker_id=other)
    client.post(
        f"/review/{run_id}/labels/S0/decision",
        data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
        headers=HX,
    )
    export = client.get(f"/review/{run_id}/export.txt").text
    # The overridden segment keeps its segment speaker; the other S0 segment
    # follows the new label exclude.
    assert "Other Person: hello there" in export
    assert "(excluded) S0: how are you" in export


def test_segment_override_canonicalizes_through_merge(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    source = add_speaker(session_factory, "Source Speaker")
    target = add_speaker(session_factory, "Target Speaker")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    relabel(client, run_id, segs[0], token, action="assign", speaker_id=source)
    with session_factory() as session:
        merge_speakers(session, source, target)
        session.commit()
    export = client.get(f"/review/{run_id}/export.txt").text
    # The override renders the MERGE TARGET's name, exactly like label scope.
    assert "Target Speaker: hello there" in export
    assert "Source Speaker" not in export


def test_html_transcript_and_export_agree(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)
    relabel(client, run_id, segs[0], token, action="assign", speaker_id=other)

    export = client.get(f"/review/{run_id}/export.txt").text
    html = client.get(f"/runs/{run_id}/transcript").text
    # Both surfaces resolve the overridden segment to the same speaker.
    assert "Other Person: hello there" in export
    assert "Other Person" in html


def test_speaker_search_stays_label_scoped(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """Documented v1 limitation: a segment-only assignment does not surface a
    speaker in the run search facet (which stays a label-grain fact)."""
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    only_seg = add_speaker(session_factory, "Segment Only Speaker")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)
    relabel(client, run_id, segs[1], token, action="assign", speaker_id=only_seg)

    # The run must NOT match a speaker facet for a speaker present only via a
    # segment override — search is deliberately label-scoped in v1.
    resp = client.get(f"/runs?speaker={only_seg}")
    assert resp.status_code == 200
    assert f"/runs/{run_id}" not in resp.text


def test_segment_scope_is_part_of_replay_identity(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    """Same idempotency key + different scope is a conflict, not a silent adopt."""
    from voxint.adjudication.ledger import ConflictingReplayError, record_decision
    from voxint.db.models import Decision

    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)

    key = uuid.uuid4().hex
    with session_factory() as session:
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key=key,
            speaker_id=other,
        )
        session.commit()
    with session_factory() as session, pytest.raises(ConflictingReplayError):
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.ASSIGN,
            operator="op",
            idempotency_key=key,  # same key...
            speaker_id=other,
            transcript_segment_id=segs[0],  # ...but now segment scope -> conflict
        )


def test_inherit_is_db_constrained_to_segment_scope(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    from sqlalchemy.exc import IntegrityError

    from voxint.adjudication.ledger import record_decision
    from voxint.db.models import Decision

    with session_factory() as session:
        run_id = seed_run(session, media_root)
    # A label-scope (transcript_segment_id NULL) INHERIT violates the CHECK.
    with session_factory() as session, pytest.raises(IntegrityError):
        record_decision(
            session,
            pipeline_run_id=run_id,
            diarization_label="S0",
            decision=Decision.INHERIT,
            operator="op",
            idempotency_key=uuid.uuid4().hex,
        )
        session.flush()


def test_workbench_renders_two_scope_controls(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    token = claim_token(client, run_id)

    # Before any override: the per-segment "reassign" control is offered.
    page = client.get(f"/review/{run_id}?token={token}").text
    assert "reassign segment" in page
    assert f"/review/{run_id}/segments/" in page

    # After an override: the workbench shows the this-segment attribution + reset.
    relabel(client, run_id, segs[0], token, action="assign", speaker_id=other)
    fragment = client.post(
        f"/review/{run_id}/segments/{segs[0]}/relabel",
        data={
            "token": token,
            "nonce": uuid.uuid4().hex,
            "action": "assign",
            "speaker_id": str(other),
        },
        headers=HX,
    ).text
    assert "Other Person (this segment)" in fragment
    assert "reset to label" in fragment


def test_segment_relabel_validation_and_idempotency(
    client: TestClient, session_factory: sessionmaker[Session], media_root: Path
) -> None:
    with session_factory() as session:
        run_id = seed_run(session, media_root)
    other = add_speaker(session_factory, "Other Person")
    segs = segment_ids(session_factory, run_id)
    foreign_seg = bare_run_segment(session_factory)
    token = claim_token(client, run_id)

    # Segment scope allows only assign / inherit.
    assert (
        client.post(
            f"/review/{run_id}/segments/{segs[0]}/relabel",
            data={"token": token, "nonce": uuid.uuid4().hex, "action": "exclude"},
            headers=HX,
        ).status_code
        == 422
    )
    # A segment from ANOTHER run is not addressable here.
    assert (
        relabel(client, run_id, foreign_seg, token, action="assign", speaker_id=other).status_code
        == 404
    )
    # Idempotent replay: same nonce -> the same single row, no duplicate.
    nonce = uuid.uuid4().hex
    first = client.post(
        f"/review/{run_id}/segments/{segs[0]}/relabel",
        data={"token": token, "nonce": nonce, "action": "assign", "speaker_id": str(other)},
        headers=HX,
    )
    replay = client.post(
        f"/review/{run_id}/segments/{segs[0]}/relabel",
        data={"token": token, "nonce": nonce, "action": "assign", "speaker_id": str(other)},
        headers=HX,
    )
    assert first.status_code == 200 and replay.status_code == 200
