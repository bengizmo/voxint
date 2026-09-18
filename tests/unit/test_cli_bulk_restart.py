"""Bulk restart selector, transaction isolation, and dispatch contracts."""

import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from voxint import cli
from voxint.db import session as db_session
from voxint.db.models import Stage
from voxint.ingest.service import RestartImpact


@pytest.mark.parametrize(
    "args",
    [
        [],
        [str(uuid.uuid4()), "--all"],
        [str(uuid.uuid4()), "--status", "failed"],
        [str(uuid.uuid4()), "--since", "2026-09-01"],
        ["--all", "--status", "completed"],
        [str(uuid.uuid4()), "--dry-run"],
        ["--since", "bad-date"],
        ["--since", ""],
    ],
)
def test_invalid_selectors_do_not_connect(monkeypatch, args):
    engine = MagicMock(side_effect=AssertionError("unexpected DB access"))
    monkeypatch.setattr(db_session, "build_engine", engine)
    assert cli.main(["restart", *args]) == 2
    engine.assert_not_called()


@pytest.fixture
def bulk(monkeypatch):
    import voxint.ingest as ingest

    candidates = [(uuid.uuid4(), i + 1) for i in range(4)]
    sessions = [MagicMock() for _ in candidates]
    for session in sessions:
        session.get.return_value = SimpleNamespace(status="completed", archived_at=None)
    factory = MagicMock(side_effect=sessions)
    impact = MagicMock(return_value=RestartImpact(0, 0, 0))
    restart = MagicMock()
    publish = MagicMock(return_value=True)
    monkeypatch.setattr(ingest, "restart_impact", impact)
    monkeypatch.setattr(ingest, "restart_run", restart)
    monkeypatch.setattr(cli, "_publish_or_defer", publish)
    monkeypatch.setenv("RERUN_PUBLISH_BATCH_SIZE", "2")
    return SimpleNamespace(
        candidates=candidates,
        sessions=sessions,
        factory=factory,
        impact=impact,
        restart=restart,
        publish=publish,
    )


def test_commit_failure_isolated_and_only_committed_runs_published(bulk, capsys):
    bulk.sessions[1].commit.side_effect = SQLAlchemyError("commit failed")

    def publish(run_id, *, stage):
        assert stage == Stage.ENHANCE_MATCH
        for session in bulk.sessions:
            session.close.assert_called_once()
        assert run_id != bulk.candidates[1][0]
        return True

    bulk.publish.side_effect = publish
    rc = cli._restart_bulk_execute(
        bulk.candidates,
        bulk.factory,
        Stage.ENHANCE_MATCH,
        False,
        False,
    )
    assert rc == 1
    bulk.sessions[1].rollback.assert_called_once()
    assert bulk.publish.call_count == 2
    out = capsys.readouterr().out
    assert "restarted 3, skipped 1" in out
    assert bulk.restart.call_args_list[0].kwargs["expected_revision"] == 1


def test_broker_outage_stops_publish_attempts(bulk, capsys):
    bulk.publish.return_value = False
    assert cli._restart_bulk_execute(bulk.candidates, bulk.factory, None, False, False) == 0
    bulk.publish.assert_called_once_with(bulk.candidates[0][0], stage=None)
    assert "published 0, deferred 4" in capsys.readouterr().out
    for session in bulk.sessions:
        session.commit.assert_called_once()


def test_dry_run_reports_impact_without_mutation_or_dispatch(bulk, capsys):
    bulk.sessions[-1].get.return_value = None
    bulk.impact.side_effect = [
        RestartImpact(0, 0, 0),
        RestartImpact(0, 2, 1),
        RestartImpact(3, 0, 0),
    ]
    assert cli._restart_bulk_dry_run(bulk.candidates, bulk.factory, None, False, False) == 0
    assert "1 restart-ready, 1 void-required, 1 label-risk, 1 ineligible" in capsys.readouterr().out
    bulk.restart.assert_not_called()
    bulk.publish.assert_not_called()


def test_selection_filters_and_closes_before_preview(monkeypatch, bulk):
    engine = MagicMock()
    selection = MagicMock()
    selection.execute.return_value.all.return_value = bulk.candidates
    monkeypatch.setattr(db_session, "build_engine", lambda: engine)
    monkeypatch.setattr(db_session, "build_session_factory", lambda _: lambda: selection)

    def preview(candidates, factory, stage, acknowledge, acknowledge_void):
        selection.close.assert_called_once()
        assert candidates == bulk.candidates
        assert stage == Stage.FINALIZE
        assert not acknowledge
        assert not acknowledge_void
        return 0

    monkeypatch.setattr(cli, "_restart_bulk_dry_run", preview)
    assert (
        cli.main(
            [
                "restart",
                "--status",
                "failed",
                "--status",
                "failed",
                "--since",
                "2026-09-01",
                "--from-stage",
                "finalize",
                "--dry-run",
            ]
        )
        == 0
    )
    query = selection.execute.call_args.args[0].compile()
    assert ["failed"] in query.params.values()
    assert "pipeline_runs.archived_at IS NULL" in str(query)
    assert "ORDER BY pipeline_runs.created_at, pipeline_runs.id" in str(query)
    cutoff = next(value for value in query.params.values() if hasattr(value, "tzinfo"))
    assert cutoff.isoformat() == "2026-09-01T00:00:00+00:00"
    engine.dispose.assert_called_once()


def test_stale_revision_counted_as_stale_not_error(bulk, capsys):
    from voxint.pipeline.transitions import StaleRevisionError

    call_count = [0]

    def restart_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise StaleRevisionError(bulk.candidates[1][0], 1)
        return MagicMock()

    bulk.restart.side_effect = restart_side_effect
    rc = cli._restart_bulk_execute(bulk.candidates, bulk.factory, None, False, False)
    assert rc == 0
    out = capsys.readouterr()
    assert "restarted 3" in out.out
    assert "1 stale" in out.out
    assert "0 errors" in out.out
    assert "concurrently modified" in out.err


def test_confirmation_refusal_returns_2(monkeypatch, bulk):
    engine = MagicMock()
    selection = MagicMock()
    selection.execute.return_value.all.return_value = bulk.candidates
    monkeypatch.setattr(db_session, "build_engine", lambda: engine)
    monkeypatch.setattr(db_session, "build_session_factory", lambda _: lambda: selection)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.main(["restart", "--all"]) == 2
    bulk.restart.assert_not_called()


def test_confirmation_eof_returns_2(monkeypatch, bulk):
    engine = MagicMock()
    selection = MagicMock()
    selection.execute.return_value.all.return_value = bulk.candidates
    monkeypatch.setattr(db_session, "build_engine", lambda: engine)
    monkeypatch.setattr(db_session, "build_session_factory", lambda _: lambda: selection)
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=EOFError))
    assert cli.main(["restart", "--all"]) == 2
    bulk.restart.assert_not_called()
