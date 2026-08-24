import uuid

import pytest
from celery.exceptions import OperationalError
from fastapi.testclient import TestClient

from voxint import __version__
from voxint.api.app import app
from voxint.api.routers import deps as deps_module


def test_healthz() -> None:
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


# --- _publish_or_defer: broker-down degradation vs. real-bug propagation ------
# Monkeypatching _publish_run keeps the broker out entirely — these prove the
# catch is precise: a broker outage defers (never raises), a genuine bug does not.


def test_publish_or_defer_returns_true_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(deps_module, "_publish_run", lambda _run_id, **_kwargs: None)
    assert deps_module._publish_or_defer(uuid.uuid4()) is True


def test_publish_or_defer_swallows_broker_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _down(_run_id: uuid.UUID, **_kwargs: object) -> None:
        raise OperationalError("Error 111 connecting to redis. Connection refused.")

    monkeypatch.setattr(deps_module, "_publish_run", _down)
    # A broker outage is non-fatal: deferred, not raised — the committed QUEUED
    # run is left for the recovery sweep.
    assert deps_module._publish_or_defer(uuid.uuid4()) is False


def test_publish_or_defer_reraises_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _bug(_run_id: uuid.UUID, **_kwargs: object) -> None:
        raise RuntimeError("a real bug in the publish path")

    monkeypatch.setattr(deps_module, "_publish_run", _bug)
    # Only OperationalError is swallowed; anything else must surface, not be
    # silently deferred as if the broker were down.
    with pytest.raises(RuntimeError, match="real bug"):
        deps_module._publish_or_defer(uuid.uuid4())


# --- ignore_result is load-bearing, not cosmetic ------------------------------
# The above tests monkeypatch _publish_run, so they never exercise the real
# apply_async call. These guard the actual config: with a Redis result backend,
# dropping ignore_result makes a dead-broker publish raise a vague RuntimeError
# from the result consumer instead of the kombu OperationalError _publish_or_defer
# catches — so a silent removal would break degradation while the mocked tests
# stayed green.


def test_run_pipeline_task_ignores_result() -> None:
    from voxint.worker.tasks import run_pipeline

    assert run_pipeline.ignore_result is True


def test_publish_run_enqueues_with_ignore_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from voxint.worker.tasks import run_pipeline

    captured: dict[str, object] = {}

    def _capture(args: tuple[object, ...], **kwargs: object) -> None:
        captured["args"] = args
        captured["kwargs"] = kwargs

    # Patch apply_async (not _publish_run) so the REAL _publish_run body runs.
    monkeypatch.setattr(run_pipeline, "apply_async", _capture)
    run_id = uuid.uuid4()
    deps_module._publish_run(run_id)
    assert captured["args"] == (str(run_id),)
    assert captured["kwargs"] == {"ignore_result": True}
