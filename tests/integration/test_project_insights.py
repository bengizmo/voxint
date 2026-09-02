"""Project overview insights against the migrated PostgreSQL schema (#336)."""

import html
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.project_insights import get_project_insights
from voxint.api.projects_query import project_detail
from voxint.config import Settings
from voxint.db.models import (
    AdjudicationDecision,
    CorpusAnalysisArtifact,
    CorpusAnalysisArtifactKind,
    DiarizationTurn,
    MediaFolder,
    MediaItem,
    PipelineRun,
    Project,
    RunAssetKind,
    RunEnrichmentAsset,
    RunStatus,
    Speaker,
)

CREDS = ("reviewer", "s3cret")
SOURCE_HASH = "a" * 64


@pytest.fixture()
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> TestClient:
    settings = Settings(
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        media_root=tmp_path,
        console_projects_enabled=True,
    )
    test_client = TestClient(
        create_app(settings=settings, session_factory=session_factory)
    )
    test_client.auth = CREDS
    seed_onboarded(session_factory)
    return test_client


def _project_and_folder(
    session: Session, *, name: str = "Project insights", path: str = "project"
) -> tuple[Project, MediaFolder]:
    project = Project(name=name)
    session.add(project)
    session.flush()
    folder = MediaFolder(path=path, project_id=project.id)
    session.add(folder)
    session.flush()
    return project, folder


def _media(session: Session, folder: MediaFolder, path: str) -> MediaItem:
    item = MediaItem(source_path=path, media_folder_id=folder.id)
    session.add(item)
    session.flush()
    return item


def _run(
    session: Session,
    media: MediaItem,
    *,
    status: RunStatus = RunStatus.COMPLETED,
    created_at: datetime | None = None,
    archived: bool = False,
) -> PipelineRun:
    run = PipelineRun(
        media_item_id=media.id,
        status=status.value,
        created_at=created_at or datetime.now(UTC),
        archived_at=datetime.now(UTC) if archived else None,
    )
    session.add(run)
    session.flush()
    return run


def _asset(
    session: Session,
    run: PipelineRun,
    kind: RunAssetKind,
    payload: dict[str, object],
    *,
    generation: int = 1,
) -> RunEnrichmentAsset:
    now = datetime.now(UTC)
    asset = RunEnrichmentAsset(
        pipeline_run_id=run.id,
        asset_kind=kind.value,
        generation=generation,
        payload=payload,
        payload_schema_version=1,
        producer="integration-test",
        producer_version="1",
        model="test-model",
        source_content_hash=SOURCE_HASH,
        idempotency_key=str(uuid.uuid4()),
        started_at=now,
        completed_at=now,
    )
    session.add(asset)
    session.flush()
    return asset


def _entity_payload(*surfaces: str) -> dict[str, object]:
    return {
        "mentions": [
            {"surface": surface, "kind": "organization", "occurrences": [{}]}
            for surface in surfaces
        ]
    }


def _assign_speaker(
    session: Session,
    run: PipelineRun,
    speaker: Speaker,
    *,
    label: str,
    turn_index: int = 0,
) -> None:
    session.add(
        DiarizationTurn(
            pipeline_run_id=run.id,
            turn_index=turn_index,
            start_seconds=0.0,
            end_seconds=5.0,
            label=label,
            skip_reason="test-seed",
        )
    )
    session.add(
        AdjudicationDecision(
            pipeline_run_id=run.id,
            diarization_label=label,
            decision="assign",
            speaker_id=speaker.id,
            operator="reviewer",
            idempotency_key=str(uuid.uuid4()),
        )
    )


def test_insights_use_only_current_assets_on_canonical_project_runs(
    session_factory: sessionmaker[Session],
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        first = _media(session, folder, "project/first.wav")
        second = _media(session, folder, "project/second.wav")
        old_run = _run(session, first, created_at=now - timedelta(days=2))
        current_run = _run(session, first, created_at=now - timedelta(days=1))
        _run(session, second, created_at=now)

        _asset(
            session,
            old_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("StaleCorp"),
        )
        superseded = _asset(
            session,
            current_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("SupersededCorp"),
            generation=1,
        )
        current = _asset(
            session,
            current_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("Acme Corp"),
            generation=2,
        )
        superseded.superseded_by_asset_id = current.id
        _asset(
            session,
            current_run,
            RunAssetKind.TOPICS,
            {"topics": [{"label": "Heat pumps", "confidence": 0.9}]},
        )

        _, outside_folder = _project_and_folder(
            session, name="Outside", path="outside"
        )
        outside_run = _run(
            session, _media(session, outside_folder, "outside/leak.wav")
        )
        _asset(
            session,
            outside_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("OutsideCorp"),
        )
        archived_run = _run(
            session,
            _media(session, folder, "project/archived.wav"),
            archived=True,
        )
        _asset(
            session,
            archived_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("ArchivedCorp"),
        )
        queued_run = _run(
            session,
            _media(session, folder, "project/queued.wav"),
            status=RunStatus.QUEUED,
        )
        _asset(
            session,
            queued_run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("QueuedCorp"),
        )
        session.commit()

        insights = get_project_insights(session, project.id)

    assert insights is not None
    assert [entity["surface"] for entity in insights["entities"]] == ["Acme Corp"]
    assert [topic["label"] for topic in insights["topics"]] == ["Heat pumps"]
    assert insights["enrichment_coverage"] == {
        "total_runs": 2,
        "entity_runs": 1,
        "topic_runs": 1,
    }


def test_cache_reuses_one_artifact_and_invalidates_for_a_new_asset_generation(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        run = _run(session, _media(session, folder, "project/recording.wav"))
        old = _asset(
            session,
            run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("Old Corp"),
        )
        project_id = project.id
        session.commit()

        first = get_project_insights(session, project_id)
        first_artifact = session.execute(
            select(CorpusAnalysisArtifact).where(
                CorpusAnalysisArtifact.scope_kind == "project",
                CorpusAnalysisArtifact.scope_id == project_id,
                CorpusAnalysisArtifact.artifact_kind
                == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            )
        ).scalar_one()
        second = get_project_insights(session, project_id)
        artifacts = session.execute(
            select(CorpusAnalysisArtifact).where(
                CorpusAnalysisArtifact.scope_kind == "project",
                CorpusAnalysisArtifact.scope_id == project_id,
                CorpusAnalysisArtifact.artifact_kind
                == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            )
        ).scalars().all()

        assert first == second
        assert len(artifacts) == 1
        assert artifacts[0].id == first_artifact.id
        first_artifact_id = first_artifact.id

        new = _asset(
            session,
            run,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("New Corp"),
            generation=2,
        )
        old.superseded_by_asset_id = new.id
        session.flush()

        refreshed = get_project_insights(session, project_id)
        refreshed_artifacts = session.execute(
            select(CorpusAnalysisArtifact).where(
                CorpusAnalysisArtifact.scope_kind == "project",
                CorpusAnalysisArtifact.scope_id == project_id,
                CorpusAnalysisArtifact.artifact_kind
                == CorpusAnalysisArtifactKind.PROJECT_INSIGHTS.value,
            )
        ).scalars().all()

    assert refreshed is not None
    assert [entity["surface"] for entity in refreshed["entities"]] == ["New Corp"]
    assert len(refreshed_artifacts) == 1
    assert refreshed_artifacts[0].id != first_artifact_id


def test_detail_renders_links_coverage_and_escaped_values(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        enriched = _run(session, _media(session, folder, "project/enriched.wav"))
        hostile_run = _run(
            session,
            _media(session, folder, "<script>x</script>"),
        )
        _asset(
            session,
            enriched,
            RunAssetKind.ENTITY_MENTIONS,
            _entity_payload("Acme Corp", "<b>Evil</b>"),
        )
        speaker = Speaker(display_name="Presenter")
        session.add(speaker)
        session.flush()
        _assign_speaker(session, hostile_run, speaker, label="SPEAKER_00")
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "Acme Corp" in response.text
    assert (
        f'/explore?q=%22Acme%20Corp%22&amp;project={project_id}' in response.text
    )
    assert "Enrichment covers 1 of 2 recordings for entities" in response.text
    assert "<b>Evil</b>" not in response.text
    assert "&lt;b&gt;Evil&lt;/b&gt;" in response.text
    assert "<script>x</script>" not in response.text
    assert "&lt;script&gt;x&lt;/script&gt;" in response.text


def test_detail_fallback_when_no_completed_recordings(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, _ = _project_and_folder(session)
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    normalized_text = " ".join(response.text.split())
    assert "No completed recordings yet" in normalized_text
    assert "finish processing" in normalized_text


def test_detail_explains_when_run_assets_are_absent(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        _run(session, _media(session, folder, "project/recording.wav"))
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    assert "Enable run enrichment" in response.text
    assert 'href="/settings"' in response.text


def test_detail_distinguishes_assets_with_empty_results(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        run = _run(session, _media(session, folder, "project/recording.wav"))
        _asset(
            session,
            run,
            RunAssetKind.ENTITY_MENTIONS,
            {
                "mentions": [],
                "diagnostics": {
                    "dropped_unlocatable": 0,
                    "dropped_out_of_run": 0,
                },
            },
        )
        project_id = project.id
        session.commit()

    response = client.get(f"/projects/{project_id}")

    assert response.status_code == 200
    normalized_text = " ".join(response.text.split())
    assert "found no topics" in normalized_text
    assert "entity mentions to report" in normalized_text


def test_speakers_and_matrix_use_only_canonical_runs(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    now = datetime.now(UTC)
    with session_factory() as session:
        project, folder = _project_and_folder(session)
        repeated_media = _media(session, folder, "project/repeated.wav")
        other_media = _media(session, folder, "project/other.wav")
        historical = _run(
            session, repeated_media, created_at=now - timedelta(days=2)
        )
        canonical = _run(
            session, repeated_media, created_at=now - timedelta(days=1)
        )
        other = _run(session, other_media, created_at=now)
        current = Speaker(display_name="Current")
        ghost = Speaker(display_name="Ghost")
        second = Speaker(display_name="Second")
        session.add_all([current, ghost, second])
        session.flush()
        _assign_speaker(session, historical, current, label="SPEAKER_00")
        _assign_speaker(session, historical, ghost, label="SPEAKER_01", turn_index=1)
        _assign_speaker(session, canonical, current, label="SPEAKER_00")
        _assign_speaker(session, other, second, label="SPEAKER_00")
        project_id = project.id
        session.commit()

        detail = project_detail(session, project_id)
        insights = get_project_insights(session, project_id)

    assert detail is not None
    assert {speaker.name: speaker.run_count for speaker in detail.speakers} == {
        "Current": 1,
        "Second": 1,
    }
    assert insights is not None
    assert [speaker["name"] for speaker in insights["coverage"]["speakers"]] == [
        "Current",
        "Second",
    ]
    assert insights["coverage"]["cells"] == [[0, 1], [1, 0]]

    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    assert "Ghost" not in response.text
    # Two speakers by two recordings is below the 3x3 matrix threshold (#385):
    # the page renders the plain speaker list, one row per canonical speaker.
    assert 'class="pi-speaker-list"' in response.text
    assert 'class="pi-cov-dot"' not in response.text
    assert response.text.count("1 of 2 recordings") == 2


# ---------------------------------------------------------------------------
# Low-data rendering thresholds (#385): boundary tests at each cut-over.
# ---------------------------------------------------------------------------


def _seed_entities(session: Session, count: int) -> uuid.UUID:
    project, folder = _project_and_folder(session)
    run = _run(session, _media(session, folder, "project/enriched.wav"))
    _asset(
        session,
        run,
        RunAssetKind.ENTITY_MENTIONS,
        _entity_payload(*[f"Entity {index}" for index in range(count)]),
    )
    session.commit()
    return project.id


@pytest.mark.parametrize(
    ("count", "expect_bars"),
    [(4, False), (5, True)],
    ids=["four-entities-list", "five-entities-bars"],
)
def test_entity_widget_switches_from_list_to_bars_at_five(
    client: TestClient,
    session_factory: sessionmaker[Session],
    count: int,
    expect_bars: bool,
) -> None:
    with session_factory() as session:
        project_id = _seed_entities(session, count)

    body = client.get(f"/projects/{project_id}").text

    assert ('class="pi-entity-bars"' in body) is expect_bars
    assert ('class="pi-entity-list"' in body) is not expect_bars
    assert ('class="pi-entity-fill"' in body) is expect_bars
    # Never hides data: every entity is still listed and linked either way.
    for index in range(count):
        assert f"Entity {index}" in body
    assert body.count(f"&amp;project={project_id}") >= count
    if not expect_bars:
        assert '<span class="pi-entity-rank" aria-hidden="true">1.</span>' in body
        assert 'title="Entity 0: 1 mention across 1 recording"' in body


def _seed_coverage(session: Session, speakers: int, recordings: int) -> uuid.UUID:
    project, folder = _project_and_folder(session)
    people = [Speaker(display_name=f"Speaker {index}") for index in range(speakers)]
    session.add_all(people)
    session.flush()
    for rec_index in range(recordings):
        run = _run(session, _media(session, folder, f"project/rec-{rec_index}.wav"))
        for spk_index, person in enumerate(people):
            _assign_speaker(
                session, run, person, label=f"SPEAKER_{spk_index:02d}", turn_index=spk_index
            )
    session.commit()
    return project.id


@pytest.mark.parametrize(
    ("speakers", "recordings", "expect_matrix"),
    [(2, 3, False), (3, 2, False), (3, 3, True)],
    ids=["2x3-list", "3x2-list", "3x3-matrix"],
)
def test_coverage_widget_switches_from_list_to_matrix_at_three_by_three(
    client: TestClient,
    session_factory: sessionmaker[Session],
    speakers: int,
    recordings: int,
    expect_matrix: bool,
) -> None:
    with session_factory() as session:
        project_id = _seed_coverage(session, speakers, recordings)

    body = client.get(f"/projects/{project_id}").text

    assert ('class="pi-coverage"' in body) is expect_matrix
    assert ('class="pi-speaker-list"' in body) is not expect_matrix
    for index in range(speakers):
        assert f"Speaker {index}" in body
    if expect_matrix:
        assert body.count('class="pi-cov-dot"') == speakers * recordings
    else:
        plural = "s" if recordings != 1 else ""
        assert body.count(f"{recordings} of {recordings} recording{plural}") == speakers
        # Truncated-label safety: the row title names every recording.
        # Recording names are visible text (not tooltip-only), with the full
        # list repeated as the title for the truncating case.
        titles = ", ".join(f"project/rec-{index}.wav" for index in range(recordings))
        assert f'<span class="pi-speaker-titles" title="{titles}">{titles}</span>' in body


def _seed_dated(session: Session, days: list[int]) -> uuid.UUID:
    base = datetime(2026, 3, 1, 12, tzinfo=UTC)
    project, folder = _project_and_folder(session)
    for index, day in enumerate(days):
        media = _media(session, folder, f"project/day-{index}.wav")
        media.created_at = base + timedelta(days=day)
        _run(session, media)
    session.commit()
    return project.id


@pytest.mark.parametrize(
    ("days", "expect_mode"),
    [([0, 0], "single_date"), ([0, 1], "chart"), ([], "empty")],
    ids=["one-date-summary", "two-dates-chart", "no-dates-empty"],
)
def test_temporal_widget_mode_follows_distinct_dates(
    client: TestClient,
    session_factory: sessionmaker[Session],
    days: list[int],
    expect_mode: str,
) -> None:
    with session_factory() as session:
        project_id = _seed_dated(session, days)

    body = client.get(f"/projects/{project_id}").text

    assert ('data-island="temporal-trends"' in body) is (expect_mode == "chart")
    if expect_mode == "chart":
        assert '"display_mode": "chart"' in html.unescape(body)
    else:
        assert f'data-temporal-mode="{expect_mode}"' in body
    if expect_mode == "single_date":
        assert "All 2 dated recordings are from 2026-03-01." in body
        assert "Trends appear once recordings span more than one day." in body
    if expect_mode == "empty":
        assert "No dated recordings are available yet." in body
