"""Settings → Features section (issue #62): tri-state runtime toggles, end to end.

Covers the ``POST /settings/features`` candidate → validate → ONE mutation
contract against real Postgres: a UI enable/disable applies to the capability
gates with no restart, "use installation setting" writes ``NULL`` (revert to
env), an invariant-violating combo is refused server-side with the operator's
choices preserved and NOTHING written, an independent flag (yt-dlp) toggles free
of the LLM invariants, CSRF is required, and saving Features never disturbs the
LLM section's stored state.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.enrichment.asset_jobs import run_asset_gates_open

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-features-test-csrf-key"


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
    seed_llm_enabled: bool = False,
    **overrides: object,
) -> tuple[TestClient, Settings]:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0],
        voxint_password=CREDS[1],
        csrf_secret=_CSRF_KEY,
        media_root=media_root,
        **overrides,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    client.auth = CREDS
    if onboarded:
        seed_onboarded(session_factory, llm_enabled=seed_llm_enabled)
    return client, settings


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _seed_flags(session_factory: sessionmaker[Session], **columns: object) -> None:
    """Force feature-flag columns on the row directly (bypassing the form/validator)
    to set up a pre-existing override for the write-nothing regression."""
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        for name, value in columns.items():
            setattr(row, name, value)
        session.commit()


def _switch_row(body: str, flag_name: str) -> str:
    """Return the rendered switch-row containing ``flag_name``."""
    marker = f'id="sw-{flag_name}"'
    marker_at = body.index(marker)
    row_start = body.rfind('<div class="switch-row', 0, marker_at)
    assert row_start >= 0, flag_name
    next_row = body.find('<div class="switch-row', marker_at)
    section_end = body.find("</section>", marker_at)
    if next_row < 0 or (section_end >= 0 and section_end < next_row):
        next_row = section_end
    return body[row_start:] if next_row < 0 else body[row_start:next_row]


def _checkbox(body: str, flag_name: str) -> str:
    """Return the checkbox input tag for ``flag_name``."""
    marker = f'id="sw-{flag_name}"'
    marker_at = body.index(marker)
    input_start = body.rfind("<input", 0, marker_at)
    assert input_start >= 0, flag_name
    return body[input_start : body.index(">", marker_at) + 1]


def test_features_section_renders_tristate(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.get("/settings")
    assert resp.status_code == 200
    body = resp.text
    assert 'id="features"' in body
    for name in (
        "enrichment_names_enabled",
        "enrichment_names_llm_enabled",
        "enrichment_run_assets_enabled",
        "enrichment_run_assets_autogenerate",
        "ytdlp_enabled",
    ):
        assert f'name="{name}"' in body
    # No stored override anywhere → every flag renders as a switch with no badge.
    assert 'role="switch"' in body
    # No overrides → no "Changed" badges (check for the rendered badge element,
    # not the CSS class name which appears in the stylesheet).
    assert ">Changed</span>" not in body
    assert 'id="sw-enrichment_run_assets_enabled"' in body
    # The gated fragments no longer send operators to raw env vars.
    assert "ENRICHMENT_RUN_ASSETS_ENABLED" not in body


def test_enable_run_assets_applies_without_restart(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # env run-assets off; onboard with LLM on so run_assets ⇒ llm is satisfiable.
    client, settings = make_client(session_factory, media_root, seed_llm_enabled=True)
    row = _row(session_factory)
    assert run_asset_gates_open(settings, row) is False  # closed to start

    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_enabled="on"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/settings#features"
    row = _row(session_factory)
    assert row is not None and row.enrichment_run_assets_enabled is True
    # Same app instance — no restart — and the gate is now open.
    assert run_asset_gates_open(settings, row) is True

    # Disabling closes it again, still no restart.
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_enabled="off"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None and row.enrichment_run_assets_enabled is False
    assert run_asset_gates_open(settings, row) is False


def test_invariant_violation_writes_nothing_and_preserves_input(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # env LLM off → enabling the LLM name pass violates names_llm ⇒ llm_enabled.
    # Seed an unrelated stored override first: a rejected save must leave the WHOLE
    # row untouched, not just the rejected column.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=False)
    _seed_flags(session_factory, ytdlp_enabled=False, enrichment_names_enabled=False)
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_names_llm_enabled="on"),
        follow_redirects=False,
    )
    assert resp.status_code == 200  # re-render, not a redirect
    # Plain-language message — NOT the raw invariant identifier string.
    assert "LLM name pass needs LLM transcript enhancement" in resp.text
    assert "requires llm_enabled=true" not in resp.text
    # Nothing written: the rejected column stays NULL AND the seeded overrides survive.
    row = _row(session_factory)
    assert row is not None
    assert row.enrichment_names_llm_enabled is None
    assert row.ytdlp_enabled is False
    assert row.enrichment_names_enabled is False
    # The operator's rejected choice is rendered back for correction: the switch
    # for the submitted "on" renders checked (the submitted value is "on", so
    # effective = True → checked), and the "Changed" badge appears.
    assert 'id="sw-enrichment_names_llm_enabled"' in resp.text
    assert ">Changed</span>" in resp.text


def test_names_llm_requires_names_enabled(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # names_llm on + names off in the same POST violates names_llm ⇒ names.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/features",
        data=_form(
            enrichment_names_enabled="off",
            enrichment_names_llm_enabled="on",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "needs speaker name suggestions turned on" in resp.text
    row = _row(session_factory)
    assert row is not None
    assert row.enrichment_names_llm_enabled is None  # nothing written
    assert row.enrichment_names_enabled is None


def test_autogenerate_requires_run_assets(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # autogenerate on while run assets inherit env-off violates autogenerate ⇒ run_assets.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_autogenerate="on"),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Auto-generating run assets needs run assets turned on" in resp.text
    assert _row(session_factory).enrichment_run_assets_autogenerate is None  # type: ignore[union-attr]


def test_all_inherit_leaves_every_column_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Saving with every control on "use installation setting" must not pin any env
    # default onto the row (all columns stay NULL).
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/features",
        data=_form(
            enrichment_names_enabled="inherit",
            enrichment_names_llm_enabled="inherit",
            enrichment_run_assets_enabled="inherit",
            enrichment_run_assets_autogenerate="inherit",
            ytdlp_enabled="inherit",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    for name in (
        "enrichment_names_enabled",
        "enrichment_names_llm_enabled",
        "enrichment_run_assets_enabled",
        "enrichment_run_assets_autogenerate",
        "ytdlp_enabled",
    ):
        assert getattr(row, name) is None, name


def test_malformed_choice_is_rejected_without_writing(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A value outside {on,off,inherit} (stale client / hand-crafted POST) is
    # rejected, not silently coerced to off/inherit.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_enabled="yes-please"),
        follow_redirects=False,
    )
    assert resp.status_code == 200
    assert "Unrecognized feature setting" in resp.text
    assert _row(session_factory).enrichment_run_assets_enabled is None  # type: ignore[union-attr]


def test_valid_dependent_enable_succeeds(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    resp = client.post(
        "/settings/features",
        data=_form(
            enrichment_names_enabled="on",
            enrichment_names_llm_enabled="on",
        ),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.enrichment_names_enabled is True
    assert row.enrichment_names_llm_enabled is True


def test_inherit_reverts_a_stored_override_to_null(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    client.post("/settings/features", data=_form(enrichment_run_assets_enabled="on"))
    assert _row(session_factory).enrichment_run_assets_enabled is True  # type: ignore[union-attr]
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_enabled="inherit"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).enrichment_run_assets_enabled is None  # type: ignore[union-attr]


def test_off_stores_false(session_factory: sessionmaker[Session], media_root: Path) -> None:
    # env names default on; storing an explicit Off override must persist False.
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_names_enabled="off"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).enrichment_names_enabled is False  # type: ignore[union-attr]


def test_ytdlp_toggles_independently_of_llm(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # yt-dlp has no cross-flag invariant, so it toggles with LLM off.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=False)
    resp = client.post(
        "/settings/features",
        data=_form(ytdlp_enabled="off"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).ytdlp_enabled is False  # type: ignore[union-attr]


def test_features_route_accepts_every_rendered_flag(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Behavioral drift guard (#67 regression): every flag the Features section
    # RENDERS (_FEATURE_FLAG_META → the template radios) MUST round-trip through
    # POST /settings/features. A rendered radio whose name the route does not
    # declare as a Form param is silently dropped by Starlette and reverts to
    # inherit (NULL) — exactly how llm_bundled_enabled shipped broken. Submitting
    # an explicit Off for every flag (which violates no cross-flag invariant) and
    # asserting each column stored False catches any flag the route fails to accept.
    from voxint.api.routers.settings import _FEATURE_FLAG_NAMES

    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/features",
        data=_form(**{name: "off" for name in _FEATURE_FLAG_NAMES}),
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    row = _row(session_factory)
    assert row is not None
    for name in _FEATURE_FLAG_NAMES:
        assert getattr(row, name) is False, f"{name} did not round-trip through the route"


def test_enable_bundled_local_model_persists(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Regression (#67): the Features form renders the bundled radio, so its POST
    # must persist. Before the fix the route did not declare the field, Starlette
    # dropped it, and it read back NULL no matter what the operator chose.
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/features",
        data=_form(llm_bundled_enabled="on"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).llm_bundled_enabled is True  # type: ignore[union-attr]

    # And an explicit Off persists False (not NULL/inherit).
    resp = client.post(
        "/settings/features",
        data=_form(llm_bundled_enabled="off"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert _row(session_factory).llm_bundled_enabled is False  # type: ignore[union-attr]


def test_bundled_override_round_trips_in_the_rendered_form(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # The Features form is a full-form replace, so the render must re-check the
    # stored bundled state — otherwise the browser would resubmit "inherit" and
    # quietly clear the override on the next unrelated save.
    client, _ = make_client(session_factory, media_root)
    _seed_flags(session_factory, llm_bundled_enabled=True)
    body = client.get("/settings").text
    # The stored override renders the switch checked AND shows the "Changed" badge.
    assert 'id="sw-llm_bundled_enabled"' in body
    assert "checked" in body
    assert ">Changed</span>" in body


def test_features_requires_csrf(session_factory: sessionmaker[Session], media_root: Path) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/features",
        data={"enrichment_run_assets_enabled": "on"},  # no csrf_token
        follow_redirects=False,
    )
    assert resp.status_code == 403
    # No write happened.
    row = _row(session_factory)
    assert row is not None and row.enrichment_run_assets_enabled is None


def test_saving_features_preserves_llm_section(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # AC4: each section's save is independent — a Features save must not disturb
    # the LLM section's stored key/model/enablement.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    client.post(
        "/settings/llm",
        data=_form(enabled="true", llm_api_key="sk-KEEP-me", llm_model="kept-model"),
    )
    resp = client.post(
        "/settings/features",
        data=_form(enrichment_run_assets_enabled="on"),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    row = _row(session_factory)
    assert row is not None
    assert row.llm_api_key == "sk-KEEP-me"  # untouched by the Features save
    assert row.llm_model == "kept-model"
    assert row.llm_enabled is True
    assert row.enrichment_run_assets_enabled is True


def test_llm_off_disables_dependent_controls(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=False)
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.text

    disabled_flags = (
        "enrichment_names_llm_enabled",
        "enrichment_run_assets_enabled",
        "enrichment_run_assets_autogenerate",
    )
    enabled_flags = ("enrichment_names_enabled", "ytdlp_enabled", "llm_bundled_enabled")

    for name in disabled_flags:
        assert f'id="sw-{name}"' in body
        assert "disabled" in _checkbox(body, name)
        row = _switch_row(body, name)
        assert "switch-disabled" in row
        assert f'name="_rendered" value="{name}"' not in row

    for name in enabled_flags:
        assert f'id="sw-{name}"' in body
        assert "disabled" not in _checkbox(body, name)
        row = _switch_row(body, name)
        assert "switch-disabled" not in row
        assert f'name="_rendered" value="{name}"' in row


def test_llm_on_names_off_disables_only_llm_name_pass(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(
        session_factory,
        media_root,
        seed_llm_enabled=True,
        enrichment_names_enabled=False,
    )
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.text

    assert "disabled" in _checkbox(body, "enrichment_names_llm_enabled")
    assert "disabled" not in _checkbox(body, "enrichment_run_assets_enabled")
    assert "speaker name suggestions" in _switch_row(body, "enrichment_names_llm_enabled")


def test_run_assets_off_disables_autogenerate(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=True)
    response = client.get("/settings")
    assert response.status_code == 200
    assert "disabled" in _checkbox(response.text, "enrichment_run_assets_autogenerate")
    assert "disabled" not in _checkbox(response.text, "enrichment_run_assets_enabled")

    _seed_flags(session_factory, enrichment_run_assets_enabled=True)
    response = client.get("/settings")
    assert response.status_code == 200
    assert "disabled" not in _checkbox(response.text, "enrichment_run_assets_autogenerate")


def test_same_section_children_get_dependent_class(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.text

    dependent_flags = (
        "enrichment_names_llm_enabled",
        "enrichment_run_assets_autogenerate",
    )
    other_flags = (
        "enrichment_names_enabled",
        "enrichment_run_assets_enabled",
        "llm_bundled_enabled",
        "ytdlp_enabled",
    )
    for name in dependent_flags:
        assert "switch-dependent" in _switch_row(body, name)
    for name in other_flags:
        assert "switch-dependent" not in _switch_row(body, name)


def test_disabled_flag_preserved_on_save(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    make_client(session_factory, media_root, seed_llm_enabled=True)
    _seed_flags(
        session_factory,
        enrichment_names_enabled=True,
        enrichment_names_llm_enabled=True,
    )

    # Turning LLM off makes the stored-on name pass disabled. The canonical
    # General-tab form omits its _rendered marker, so reconciling a save of an
    # unrelated standalone flag must retain the raw stored override.
    client, _ = make_client(session_factory, media_root, seed_llm_enabled=False)
    body = client.get("/settings").text
    assert "disabled" in _checkbox(body, "enrichment_names_llm_enabled")

    response = client.post(
        "/settings",
        data=_form(_rendered="ytdlp_enabled", ytdlp_enabled="off"),
        follow_redirects=False,
    )
    # The preserved pre-existing value still violates the LLM invariant, so the
    # tab re-renders instead of committing the unrelated edit.
    assert response.status_code == 200
    row = _row(session_factory)
    assert row is not None
    assert row.enrichment_names_llm_enabled is True
