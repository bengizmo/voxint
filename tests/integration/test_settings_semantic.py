"""Settings → Semantic search section (issue #121): tri-state toggles, end to end.

The two embedding flags (``semantic_index_enabled``, ``semantic_index_autogenerate``)
are a section of their own, not folded into Features: they depend on nothing else
and validate through their OWN self-contained invariant (``semantic_index_flags_ok``:
autogenerate rides on the feature), deliberately outside the EffectiveFlags web.
This covers the ``POST /settings/semantic`` candidate -> validate -> ONE mutation
contract against real Postgres: a UI On/Off applies with no restart, "use
installation setting" writes NULL, the one reachable invariant violation is refused
server-side with the operator's choices preserved and NOTHING written, CSRF is
required, and the drift guard that every rendered switch round-trips through the
route (a flag missing from the reconcile list silently reverts to inherit on save).
"""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.api.routers.settings import _SEMANTIC_FLAG_NAMES
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-semantic-test-csrf-key"


def make_client(
    session_factory: sessionmaker[Session],
    tmp_path: Path,
    *,
    onboarded: bool = True,
    **overrides: object,
) -> tuple[TestClient, Settings]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=tmp_path,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory)
    return client, settings


def _form(
    rendered: list[str] | None = None, **fields: str
) -> dict[str, str | list[str]]:
    """Build form data supporting repeated ``_rendered`` keys."""
    result: dict[str, str | list[str]] = {
        "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS),
    }
    if rendered:
        result["_rendered"] = rendered
    result.update(fields)
    return result


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _seed_flags(session_factory: sessionmaker[Session], **columns: object) -> None:
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def test_semantic_section_renders_tristate(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    body = client.get("/settings/ai").text
    assert 'id="semantic-search"' in body
    # Unset columns render as switches (effective state from env default).
    assert 'id="sw-semantic_index_enabled"' in body
    assert 'id="sw-semantic_index_autogenerate"' in body
    assert 'role="switch"' in body
    # No overrides → no "Changed" badge rendered in content.
    assert ">Changed</span>" not in body


def test_semantic_route_declares_every_flag(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Behavioral drift guard: every radio the Semantic section RENDERS must
    # round-trip through POST /settings/semantic. Seed off first so turning on
    # is a real change (idempotency would keep inherit for a no-op save).
    client, _ = make_client(session_factory, tmp_path)
    _seed_flags(session_factory, **{name: False for name in _SEMANTIC_FLAG_NAMES})
    resp = client.post(
        "/settings/semantic",
        data=_form(
            rendered=list(_SEMANTIC_FLAG_NAMES),
            **{name: "on" for name in _SEMANTIC_FLAG_NAMES},
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    row = _row(session_factory)
    assert row is not None
    for name in _SEMANTIC_FLAG_NAMES:
        assert getattr(row, name) is True, f"{name} did not round-trip through the route"


def test_off_stores_false(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    resp = client.post(
        "/settings/semantic",
        data=_form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate"],
            semantic_index_enabled="off",
            semantic_index_autogenerate="off",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.semantic_index_enabled is False
    assert row.semantic_index_autogenerate is False


def test_unchecking_stored_overrides_stores_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Checkbox encoding: unchecked = rendered but no value → "off".
    client, _ = make_client(session_factory, tmp_path)
    _seed_flags(
        session_factory,
        semantic_index_enabled=True,
        semantic_index_autogenerate=True,
    )
    resp = client.post(
        "/settings/semantic",
        data=_form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate"],
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.semantic_index_enabled is False
    assert row.semantic_index_autogenerate is False


def test_autogenerate_without_feature_is_refused_and_writes_nothing(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # The one reachable invariant: autogenerate on while the feature is off.
    # env defaults are both True, so force the feature off in the same POST.
    client, _ = make_client(session_factory, tmp_path)
    _seed_flags(session_factory, semantic_index_enabled=False)
    resp = client.post(
        "/settings/semantic",
        data=_form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate"],
            semantic_index_enabled="off",
            semantic_index_autogenerate="on",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Turn semantic search on before auto-indexing" in resp.text
    # Nothing was written: the pre-existing off override is untouched, autogen NULL.
    row = _row(session_factory)
    assert row is not None
    assert row.semantic_index_enabled is False
    assert row.semantic_index_autogenerate is None
    # The operator's rejected choices are preserved in the re-rendered form:
    # the switch for the submitted "on" renders checked, with a "Changed" badge.
    assert 'id="sw-semantic_index_autogenerate"' in resp.text
    assert ">Changed</span>" in resp.text


def test_malformed_choice_treated_as_off(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # _reconcile_switches sanitises rendered values: anything != "on" is off.
    # Also render autogenerate unchecked to avoid the autogenerate-on invariant.
    client, _ = make_client(session_factory, tmp_path)
    resp = client.post(
        "/settings/semantic",
        data=_form(
            rendered=["semantic_index_enabled", "semantic_index_autogenerate"],
            semantic_index_enabled="maybe",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.semantic_index_enabled is False


def test_stored_override_round_trips_in_the_rendered_form(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # A full-form replace must re-check the stored state, or the next unrelated
    # save resubmits "inherit" and quietly clears the override.
    client, _ = make_client(session_factory, tmp_path)
    _seed_flags(session_factory, semantic_index_enabled=True)
    body = client.get("/settings/ai").text
    # The stored override renders the switch checked AND shows the "Changed" badge.
    assert 'id="sw-semantic_index_enabled"' in body
    assert "checked" in body
    assert "switch-badge" in body


def test_semantic_requires_csrf(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    client, _ = make_client(session_factory, tmp_path)
    resp = client.post(
        "/settings/semantic",
        data={"semantic_index_enabled": "off"},  # no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    row = _row(session_factory)
    assert row is not None and row.semantic_index_enabled is None


def test_weights_absent_note_shows_when_feature_on(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # The env default enables the feature but this env has no weights installed,
    # so the honest "weights are not installed" note renders.
    client, _ = make_client(session_factory, tmp_path)
    body = client.get("/settings/ai").text
    assert "The embedding model weights are not installed" in body


def test_weights_absent_note_hidden_when_effectively_off_via_inherit(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # "Use installation setting" with an installation default of OFF is
    # effectively off, so the weights-missing note must NOT claim that an enabled
    # search cannot answer — the note gates on effective enablement, not the raw
    # tri-state.
    # Both env flags go off together: the Settings invariant refuses autogenerate
    # riding on a disabled feature at construction (the same rule the route checks).
    client, _ = make_client(
        session_factory,
        tmp_path,
        semantic_index_enabled=False,
        semantic_index_autogenerate=False,
    )
    body = client.get("/settings/ai").text
    # Inherited with env default off → switch unchecked, no badge.
    assert 'id="sw-semantic_index_enabled"' in body
    assert ">Changed</span>" not in body
    assert "The embedding model weights are not installed" not in body


def test_weights_absent_note_shows_when_explicitly_on_over_off_default(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # An explicit On override beats the OFF installation default, so the feature
    # is effectively on and the honest weights-missing note renders again.
    client, _ = make_client(
        session_factory,
        tmp_path,
        semantic_index_enabled=False,
        semantic_index_autogenerate=False,
    )
    _seed_flags(session_factory, semantic_index_enabled=True)
    body = client.get("/settings/ai").text
    # Explicit On over OFF env default → switch checked + "Changed" badge.
    assert 'id="sw-semantic_index_enabled"' in body
    assert ">Changed</span>" in body
    assert "The embedding model weights are not installed" in body
