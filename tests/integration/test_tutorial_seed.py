"""`voxint tutorial seed` and its shared fixture builder against real Postgres.

Covers the three-state resolution, assignment-shape constraints, idempotency
(seed-twice, rebuild-after-delete, repair-missing-WAV), and the media-serve +
export paths the tutorial UI drives — all through the committed WAV/fixtures.
"""

import html
import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.adjudication.resolver import Resolution, label_states
from voxint.api.app import create_app
from voxint.api.csrf import (
    CSRF_MEDIA_EMPTY_TRASH,
    CSRF_MEDIA_TRASH,
    CSRF_SETTINGS,
    mint_csrf_token,
)
from voxint.app_settings import get_app_settings, ready_tutorial_run_id
from voxint.cli import main
from voxint.config import Settings
from voxint.db.models import (
    AudioArtifact,
    DiarizationTurn,
    MatchCandidate,
    MediaItem,
    MediaOperation,
    OperationState,
    OperationType,
    PipelineRun,
    RunStatus,
    Speaker,
    SpeakerAssignment,
    SpeakerEmbedding,
    TranscriptSegment,
)
from voxint.db.session import session_scope
from voxint.speakers.matching import gates_from_settings
from voxint.tutorial import resources
from voxint.tutorial import seed as seed_module
from voxint.tutorial.seed import TUTORIAL_SOURCE_PATH, seed_tutorial_run

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "tutorial-seed-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def settings(media_root: Path) -> Settings:
    return Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=media_root,
        csrf_secret=_CSRF_KEY,
        review_claim_ttl_seconds=600,
    )


def _seed(session_factory: sessionmaker[Session], settings: Settings) -> uuid.UUID:
    with session_scope(session_factory) as session:
        return seed_tutorial_run(
            session, media_root=settings.media_root, settings=settings
        )


def _count(session: Session, model: type) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def _purge_tutorial(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    """Trash then permanently delete the tutorial item through the real routes."""
    with session_factory() as session:
        media_id = session.execute(
            select(MediaItem.id).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
    trash = client.post(
        "/media/trash",
        data={
            "media_id": str(media_id),
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_TRASH),
        },
        follow_redirects=False,
    )
    assert trash.status_code == 303
    assert "trash_done=1" in trash.headers["location"]
    purge = client.post(
        "/media/empty-trash",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_MEDIA_EMPTY_TRASH)},
        follow_redirects=False,
    )
    assert purge.status_code == 303
    assert "empty_trash_done=1" in purge.headers["location"]
    with session_factory() as session:
        media = session.get(MediaItem, media_id)
        assert media is not None and media.purged_at is not None


def _editor_props(client: TestClient, media_id: uuid.UUID) -> dict[str, Any]:
    resp = client.get(f"/media/{media_id}/editor", follow_redirects=False)
    assert resp.status_code == 200
    text = resp.text
    anchor = text.find('data-island="media-editor"')
    start = text.find("data-props='", anchor) + len("data-props='")
    end = text.find("'", start)
    props: dict[str, Any] = json.loads(html.unescape(text[start:end]))
    return props


def test_seed_records_match_candidates_for_every_label(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # #675: the rail reports "Voice matching did not run" when no label has
    # match evidence, so the seed must record what the live matcher would.
    run_id = _seed(session_factory, settings)
    layout = resources.load_layout()
    roster_label = layout["roster_speaker"]["label"]
    others = {layout["heard_name"]["label"], layout["unresolved_label"]}
    with session_factory() as session:
        rows = {
            r.diarization_label: r
            for r in session.execute(
                select(MatchCandidate).where(MatchCandidate.pipeline_run_id == run_id)
            ).scalars()
        }
        speaker = session.execute(
            select(Speaker).where(
                Speaker.display_name == layout["roster_speaker"]["display_name"]
            )
        ).scalar_one()
        states = {
            s.label: s
            for s in label_states(session, run_id, gates=gates_from_settings(settings))
        }

    assert set(rows) == {roster_label, *others}
    accepted = rows[roster_label]
    assert accepted.decision == "accepted"
    assert accepted.grounded is True
    assert accepted.top_speaker_id == speaker.id
    assert accepted.similarity == pytest.approx(0.95)
    assert accepted.margin is None  # one-speaker roster
    assert accepted.roster_size == 1
    for label in others:
        assert rows[label].decision == "rejected"
        assert rows[label].reason == "below_cosine"
        assert rows[label].grounded is None

    assert all(s.match_decision is not None for s in states.values())
    assert states[roster_label].resolution is Resolution.GROUNDED_COSINE
    for label in others:
        assert states[label].resolution is Resolution.UNRESOLVED
        assert states[label].candidate_prompt_allowed is False
        assert states[label].match_reason not in {
            "too_few_turns",
            "too_little_speech",
            "no_eligible_turns",
        }


def test_reseed_backfills_match_candidates_on_an_older_tutorial(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # A tutorial seeded by 0.49.0 or earlier has no match_candidates rows;
    # re-seeding (Settings set-up, CLI) repairs it in place.
    run_id = _seed(session_factory, settings)
    with session_scope(session_factory) as session:
        session.execute(
            delete(MatchCandidate).where(MatchCandidate.pipeline_run_id == run_id)
        )

    assert _seed(session_factory, settings) == run_id
    with session_factory() as session:
        decisions = sorted(
            session.execute(
                select(MatchCandidate.decision).where(
                    MatchCandidate.pipeline_run_id == run_id
                )
            ).scalars()
        )
    assert decisions == ["accepted", "rejected", "rejected"]


def test_seed_creates_three_states(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _seed(session_factory, settings)
    layout = resources.load_layout()
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == RunStatus.COMPLETED.value
        assert run.current_stage is None
        states = {s.label: s for s in label_states(session, run_id)}

    grounded = states[layout["roster_speaker"]["label"]]
    assert grounded.resolution is Resolution.GROUNDED_COSINE
    assert grounded.speaker_name == layout["roster_speaker"]["display_name"]
    assert grounded.cosine_grounded is True

    heard = states[layout["heard_name"]["label"]]
    assert heard.resolution is Resolution.UNRESOLVED
    assert heard.llm_hint_name == layout["heard_name"]["name"]

    plain = states[layout["unresolved_label"]]
    assert plain.resolution is Resolution.UNRESOLVED
    assert plain.llm_hint_name is None
    assert plain.cosine_speaker_id is None


def test_seed_assignment_shapes(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _seed(session_factory, settings)
    layout = resources.load_layout()
    roster_label = layout["roster_speaker"]["label"]
    heard_label = layout["heard_name"]["label"]
    unresolved_label = layout["unresolved_label"]
    with session_factory() as session:
        rows = (
            session.execute(
                select(SpeakerAssignment).where(
                    SpeakerAssignment.pipeline_run_id == run_id
                )
            )
            .scalars()
            .all()
        )
    by_key = {(r.diarization_label, r.method): r for r in rows}

    cosine = by_key[(roster_label, "cosine")]
    assert cosine.grounded is True
    assert cosine.speaker_id is not None
    assert cosine.proposed_name is None
    assert cosine.confidence == pytest.approx((0.95 + 1.0) / 2.0)

    hint = by_key[(heard_label, "llm_hint")]
    assert hint.grounded is False
    assert hint.speaker_id is None
    assert hint.proposed_name == layout["heard_name"]["name"]
    assert hint.confidence is None

    # The unresolved label carries no proposal at all.
    assert (unresolved_label, "cosine") not in by_key
    assert (unresolved_label, "llm_hint") not in by_key


def test_seed_twice_is_idempotent(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    first = _seed(session_factory, settings)
    second = _seed(session_factory, settings)
    assert first == second
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1
        assert _count(session, MediaItem) == 1
        assert _count(session, DiarizationTurn) == len(
            resources.load_layout()["utterances"]
        )
        assert _count(session, TranscriptSegment) == len(
            resources.load_layout()["utterances"]
        )
        assert _count(session, SpeakerEmbedding) == 1
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == first


def test_reseed_rebuilds_after_run_deleted(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    first = _seed(session_factory, settings)
    # Simulate a deleted run: remove its child rows in FK-safe order, then the run.
    # The app_settings.tutorial_run_id FK (ON DELETE SET NULL) nulls itself.
    with session_scope(session_factory) as session:
        for model in (
            SpeakerAssignment,
            TranscriptSegment,
            DiarizationTurn,
            AudioArtifact,
        ):
            session.execute(
                delete(model).where(model.pipeline_run_id == first)
            )
        from voxint.db.models import StageRun

        session.execute(delete(StageRun).where(StageRun.pipeline_run_id == first))
        session.execute(delete(PipelineRun).where(PipelineRun.id == first))

    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id is None  # SET NULL fired

    second = _seed(session_factory, settings)
    assert second != first
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1  # old gone, one fresh run
        assert _count(session, MediaItem) == 1  # sentinel MediaItem reused
        assert _count(session, Speaker) == 1  # roster speaker reused
        assert _count(session, SpeakerEmbedding) == 1  # centroid not duplicated
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == second


def test_reseed_repairs_missing_wav(
    session_factory: sessionmaker[Session], settings: Settings, media_root: Path
) -> None:
    run_id = _seed(session_factory, settings)
    wav = media_root / "artifacts" / str(run_id) / "normalized.wav"
    assert wav.is_file()
    wav.unlink()

    again = _seed(session_factory, settings)
    assert again == run_id  # same run, no rebuild
    assert wav.is_file()
    assert wav.read_bytes() == resources.load_sample_wav_bytes()
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1


def test_seed_writes_the_original_the_media_library_checks(
    session_factory: sessionmaker[Session], settings: Settings, media_root: Path
) -> None:
    # #646: the Media library (now the default page) flags any item whose live
    # file is absent, and trash/restore move that file. The tutorial item must
    # have real bytes at its source_path, and a wiped media_root is repaired.
    _seed(session_factory, settings)
    original = media_root / TUTORIAL_SOURCE_PATH
    assert original.read_bytes() == resources.load_sample_wav_bytes()

    original.unlink()
    _seed(session_factory, settings)
    assert original.read_bytes() == resources.load_sample_wav_bytes()


def test_failed_seed_leaves_no_original_for_a_folder_scan(
    session_factory: sessionmaker[Session],
    settings: Settings,
    media_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A seed that fails rolls its rows back; a tutorial original left behind with
    # no MediaItem row would look like new media to a folder scan. The file is
    # written only after every step that can fail.
    def boom(*_args: object, **_kwargs: object) -> None:
        raise seed_module.TutorialSeedError("forced")

    monkeypatch.setattr(seed_module, "_verify_states", boom)
    with pytest.raises(seed_module.TutorialSeedError), session_scope(session_factory) as session:
        seed_tutorial_run(session, media_root=media_root, settings=settings)
    assert not (media_root / TUTORIAL_SOURCE_PATH).exists()


def test_reseed_repairs_the_original_where_it_was_moved(
    session_factory: sessionmaker[Session], settings: Settings, media_root: Path
) -> None:
    # A trashed or archived item's file lives at current_path; the repair follows
    # the pointer instead of resurrecting a copy at the old source_path.
    _seed(session_factory, settings)
    moved = "trash/tutorial/sample-3speaker.wav"
    with session_scope(session_factory) as session:
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
        media.current_path = moved
    (media_root / TUTORIAL_SOURCE_PATH).unlink()

    _seed(session_factory, settings)
    assert (media_root / moved).read_bytes() == resources.load_sample_wav_bytes()
    assert not (media_root / TUTORIAL_SOURCE_PATH).exists()


def test_reseed_after_purge_with_cleared_run_id_revives_the_item(
    session_factory: sessionmaker[Session], settings: Settings, media_root: Path
) -> None:
    # #676: the run row survives a purge, but if tutorial_run_id was also
    # cleared the fresh-build path finds the purged MediaItem (UNIQUE
    # source_path). It must revive that item deliberately and archive the old
    # run, never attach a new run to a purged tombstone.
    first = _seed(session_factory, settings)
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    _purge_tutorial(client, session_factory)
    with session_scope(session_factory) as session:
        row = get_app_settings(session)
        assert row is not None
        row.tutorial_run_id = None

    second = _seed(session_factory, settings)
    assert second != first
    with session_factory() as session:
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
        assert media.purged_at is None
        assert media.trashed_at is None
        assert media.current_path == TUTORIAL_SOURCE_PATH
        old = session.get(PipelineRun, first)
        assert old is not None and old.archived_at is not None
        new = session.get(PipelineRun, second)
        assert new is not None and new.archived_at is None
        assert new.media_item_id == media.id
    assert (media_root / TUTORIAL_SOURCE_PATH).read_bytes() == (
        resources.load_sample_wav_bytes()
    )
    assert (media_root / "artifacts" / str(second) / "normalized.wav").is_file()


def test_reseed_refuses_to_revive_a_purged_item_mid_operation(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # Reviving rewrites the item's lifecycle columns; it must never race a
    # media operation that still owns them.
    _seed(session_factory, settings)
    with session_scope(session_factory) as session:
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
        media.purged_at = func.now()
        media.current_path = None
        session.add(
            MediaOperation(
                media_id=media.id,
                operation_type=OperationType.TRASH.value,
                state=OperationState.PLANNED.value,
                origin_path=TUTORIAL_SOURCE_PATH,
            )
        )

    with pytest.raises(seed_module.TutorialSeedError, match="operation in progress"):
        _seed(session_factory, settings)
    with session_factory() as session:
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
        assert media.purged_at is not None  # rolled back, still purged


def test_backfill_fails_loud_without_the_grounded_assignment(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _seed(session_factory, settings)
    with session_scope(session_factory) as session:
        session.execute(
            delete(MatchCandidate).where(MatchCandidate.pipeline_run_id == run_id)
        )
        session.execute(
            delete(SpeakerAssignment).where(
                SpeakerAssignment.pipeline_run_id == run_id,
                SpeakerAssignment.method == "cosine",
            )
        )

    with pytest.raises(seed_module.TutorialSeedError, match="grounded cosine"):
        _seed(session_factory, settings)


def test_seed_does_not_disturb_existing_roster(
    session_factory: sessionmaker[Session], settings: Settings
) -> None:
    # A real user's roster identity (different name, real embedding space) must be
    # left untouched: the reserved tutorial name never adopts someone else's row,
    # and no synthetic embedding is grafted onto it.
    with session_scope(session_factory) as session:
        real = Speaker(display_name="Dana Real")
        session.add(real)
        session.flush()
        session.add(
            SpeakerEmbedding(
                speaker_id=real.id,
                embedding_space="titanet-large-v2",
                embedding=[1.0] + [0.0] * 191,
            )
        )

    _seed(session_factory, settings)
    tutorial_name = resources.load_layout()["roster_speaker"]["display_name"]
    assert tutorial_name != "Dana Real"
    with session_factory() as session:
        names = {s.display_name for s in session.execute(select(Speaker)).scalars()}
        assert {"Dana Real", tutorial_name} <= names
        real_row = session.execute(
            select(Speaker).where(Speaker.display_name == "Dana Real")
        ).scalar_one()
        real_embs = (
            session.execute(
                select(SpeakerEmbedding).where(
                    SpeakerEmbedding.speaker_id == real_row.id
                )
            )
            .scalars()
            .all()
        )
        assert len(real_embs) == 1  # untouched
        assert real_embs[0].embedding_space == "titanet-large-v2"


# --- media-serve + export through the real API --------------------------------


@pytest.fixture()
def client_and_run(
    session_factory: sessionmaker[Session], settings: Settings
) -> tuple[TestClient, uuid.UUID]:
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    seed_onboarded(session_factory)
    run_id = _seed(session_factory, settings)
    return client, run_id


def test_media_serves_seeded_wav(
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    client, run_id = client_and_run
    size = len(resources.load_sample_wav_bytes())

    head = client.head(f"/media/{run_id}")
    assert head.status_code == 200
    assert head.headers["accept-ranges"] == "bytes"
    assert int(head.headers["content-length"]) == size

    full = client.get(f"/media/{run_id}")
    assert full.status_code == 200
    assert full.content[:4] == b"RIFF"
    assert len(full.content) == size


def test_media_library_finds_the_tutorial_original(
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    client, _ = client_and_run
    library = client.get("/media")
    assert library.status_code == 200
    assert "sample-3speaker" in library.text
    assert "Original file not found" not in library.text


def test_export_attributes_grounded_but_not_heard_name(
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    client, run_id = client_and_run
    layout = resources.load_layout()
    export = client.get(f"/review/{run_id}/export.txt")
    assert export.status_code == 200
    body = export.text

    # The grounded label is attributed to its roster speaker.
    assert f"] {layout['roster_speaker']['display_name']}:" in body
    # Both unresolved labels render as humanized "Voice N" display names.
    heard_idx = int(layout["heard_name"]["label"].split("_")[1])
    assert f"] Voice {heard_idx + 1}:" in body
    unresolved_idx = int(layout["unresolved_label"].split("_")[1])
    assert f"] Voice {unresolved_idx + 1}:" in body
    # The heard name is never promoted to an attribution.
    assert f"] {layout['heard_name']['name']}:" not in body
    # Every utterance's text is present.
    for utt in layout["utterances"]:
        assert utt["text"] in body


def test_tutorial_editor_rail_inputs_report_matching_ran(
    session_factory: sessionmaker[Session],
    client_and_run: tuple[TestClient, uuid.UUID],
) -> None:
    # #675 pin: the speaker rail's coverage()/summary() read these props. With
    # no matchDecision on any label the rail says "Voice matching did not run"
    # while listing a matched voice; a later seed change must not bring that back.
    client, run_id = client_and_run
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None
        media_id = run.media_item_id
    states = _editor_props(client, media_id)["labelStates"]
    assert len(states) == 3
    assert all(state["matchDecision"] is not None for state in states)
    grounded = [s for s in states if s["resolution"] == "grounded_cosine"]
    assert len(grounded) == 1
    needs_you = [s for s in states if s["resolution"] == "unresolved"]
    assert len(needs_you) == 2
    assert all(s["candidatePromptAllowed"] is False for s in needs_you)


def test_purged_tutorial_reads_as_not_set_up_and_reseeds_fresh(
    session_factory: sessionmaker[Session],
    client_and_run: tuple[TestClient, uuid.UUID],
    media_root: Path,
) -> None:
    # #676: "Empty trash permanently" on the tutorial recording must leave the
    # tutorial "not set up", and setting it up again must build a playable run.
    client, old_run = client_and_run
    _purge_tutorial(client, session_factory)

    with session_factory() as session:
        assert ready_tutorial_run_id(session) is None
    page = client.get("/settings")
    assert page.status_code == 200
    assert 'action="/settings/tutorial/seed"' in page.text
    assert 'action="/settings/tutorial/replay"' not in page.text
    replay = client.post(
        "/settings/tutorial/replay",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert replay.status_code == 409

    seeded = client.post(
        "/settings/tutorial/seed",
        data={"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS)},
        follow_redirects=False,
    )
    assert seeded.status_code == 303
    new_run = uuid.UUID(seeded.headers["location"].split("/runs/")[1].split("?")[0])
    assert new_run != old_run

    with session_factory() as session:
        assert ready_tutorial_run_id(session) == new_run
        old = session.get(PipelineRun, old_run)
        assert old is not None and old.archived_at is not None
        media = session.execute(
            select(MediaItem).where(MediaItem.source_path == TUTORIAL_SOURCE_PATH)
        ).scalar_one()
        assert media.purged_at is None and media.trashed_at is None
        assert _count(session, MediaItem) == 1
    assert (media_root / TUTORIAL_SOURCE_PATH).read_bytes() == (
        resources.load_sample_wav_bytes()
    )
    served = client.get(f"/media/{new_run}")
    assert served.status_code == 200
    assert served.content[:4] == b"RIFF"


# --- CLI end to end -----------------------------------------------------------


def test_cli_tutorial_seed_creates_completed_run(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    assert main(["tutorial", "seed"]) == 0
    run_id = uuid.UUID(capsys.readouterr().out.strip())
    with session_factory() as session:
        run = session.get(PipelineRun, run_id)
        assert run is not None and run.status == RunStatus.COMPLETED.value
        row = get_app_settings(session)
        assert row is not None and row.tutorial_run_id == run_id
    assert (tmp_path / "artifacts" / str(run_id) / "normalized.wav").is_file()


def test_cli_tutorial_seed_is_idempotent(
    session_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("MEDIA_ROOT", str(tmp_path))
    assert main(["tutorial", "seed"]) == 0
    first = capsys.readouterr().out.strip()
    assert main(["tutorial", "seed"]) == 0
    second = capsys.readouterr().out.strip()
    assert first == second
    with session_factory() as session:
        assert _count(session, PipelineRun) == 1
