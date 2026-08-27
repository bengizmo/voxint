"""Settings → Media folders + the folder browser (issue #63), end to end.

Covers the shared folder-panel routes on the Settings mount (``/settings/folders``,
``/settings/folders/browse``) against real Postgres: the directory browser
(containment-safe, never disclosing outside MEDIA_ROOT), registering/unregistering
folders, the per-folder domain-pack picker (set / clear-to-default / reject unknown
/ reject unregistered), the overlap refusal that keeps folder membership
unambiguous (issue #153), honest rendering when a stored pack or the whole registry
is unavailable, CSRF scope, onboarding gating, input bounds, that concurrent
mutations serialise on the registration advisory lock (no lost update), and that a
UI-set mapping actually reaches the ingest read path that freezes a run's pack.

Since #153 the panel writes first-class ``media_folders`` rows, not the legacy
``app_settings.media_folders`` / ``folder_domain_packs`` columns; these tests read
back through the same relation the writes target.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import create_app
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.api.routers.settings import _folder_panel_context
from voxint.config import Settings
from voxint.db.models import MediaFolder, PipelineRun
from voxint.db.session import session_scope
from voxint.ingest.service import submit_media_item
from voxint.media.registration import (
    PACK_DEFAULT_SENTINEL,
    folder_pack_map,
    register_folder,
    registered_folder_paths,
)

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-folders-test-csrf-key"
_HTMX = {"HX-Request": "true"}


@pytest.fixture()
def media_root(tmp_path: Path) -> Path:
    return tmp_path


def make_client(
    session_factory: sessionmaker[Session],
    media_root: Path,
    *,
    onboarded: bool = True,
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
        seed_onboarded(session_factory)
    return client, settings


def _form(**fields: str) -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS), **fields}


def _folders(session_factory: sessionmaker[Session]) -> list[str]:
    with session_factory() as session:
        return registered_folder_paths(session)


def _packs(session_factory: sessionmaker[Session]) -> dict[str, str]:
    with session_factory() as session:
        return folder_pack_map(session)


def _seed_folder(
    session_factory: sessionmaker[Session],
    path: str,
    *,
    pack: str | None = None,
    watch: bool = True,
) -> None:
    """Insert a media_folders row directly, bypassing the route validators.

    Some honest-degradation tests need a stored pack the current registry cannot
    resolve — a state the route's own validator refuses to create.
    """
    with session_factory() as session:
        session.add(MediaFolder(path=path, domain_pack=pack, watch=watch))
        session.commit()


def _add(client: TestClient, folder: str, *, path: str = ".") -> None:
    resp = client.post(
        "/settings/folders", data=_form(action="add", folder=folder, path=path), headers=_HTMX
    )
    assert resp.status_code == 200, resp.text


def _make_packs_dir(tmp: Path, *names: str) -> Path:
    """A DOMAIN_PACKS_DIR holding one manifest-bearing pack per name."""
    packs = tmp / "packs"
    packs.mkdir(exist_ok=True)
    for name in names:
        (packs / name).mkdir()
        (packs / name / "manifest.yaml").write_text(f"name: {name}\n")
    return packs


# --------------------------------------------------------------- render + browse


def test_folders_section_renders_on_settings(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    body = client.get("/settings").text
    assert 'id="folders"' in body and 'id="folder-panel"' in body
    assert 'action="/settings/folders"' in body


def test_browse_requires_auth_and_is_readonly(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    assert client.get("/settings/folders/browse", auth=None).status_code == 401
    resp = client.get("/settings/folders/browse")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    assert _folders(session_factory) == []  # a browse never registers anything


def test_browse_lists_subdirs_with_add_controls(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    (media_root / "interviews").mkdir()
    client, _ = make_client(session_factory, media_root)
    body = client.get("/settings/folders/browse").text
    assert "podcasts" in body and "interviews" in body
    assert 'value="add"' in body  # each unregistered dir offers an Add


def test_browse_navigates_into_subdir(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "interviews" / "2026").mkdir(parents=True)
    client, _ = make_client(session_factory, media_root)
    body = client.get("/settings/folders/browse", params={"path": "interviews"}).text
    assert "2026" in body  # the child of interviews/


def test_browse_bad_path_recovers_without_disclosure(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "safe").mkdir()
    client, _ = make_client(session_factory, media_root)
    body = client.get("/settings/folders/browse", params={"path": "../../etc"}).text
    assert "isn't available" in body  # honest recovery notice
    assert "safe" in body  # recovered to the root listing
    assert "/etc" not in body  # never discloses the attempted outside path


# ------------------------------------------------------------ add / remove / cap


def test_add_persists_and_is_idempotent(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "podcasts")
    _add(client, "podcasts")  # idempotent no-op
    assert _folders(session_factory) == ["podcasts"]


def test_add_rejects_escape_and_writes_nothing(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/folders", data=_form(action="add", folder="/etc", path="."), headers=_HTMX
    )
    assert resp.status_code == 200
    assert "media folder" in resp.text
    assert _folders(session_factory) == []  # the failed add wrote nothing


def test_add_refuses_overlapping_registration(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A file under a nested pair would belong to two folders; membership (ADR 0002)
    # would be ambiguous. Registering a child of an existing folder is refused, and
    # nothing is written — register only the parent or only the child.
    (media_root / "audio" / "podcasts").mkdir(parents=True)
    client, _ = make_client(session_factory, media_root)
    _add(client, "audio")
    resp = client.post(
        "/settings/folders",
        data=_form(action="add", folder="audio/podcasts", path="audio"),
        headers=_HTMX,
    )
    assert resp.status_code == 200
    assert "overlaps an already-registered folder" in resp.text
    assert _folders(session_factory) == ["audio"]  # the child was not added
    # The reverse direction (registering a parent of an existing folder) is refused too.
    (media_root / "clips").mkdir()
    (media_root / "clips" / "raw").mkdir()
    _add(client, "clips/raw")
    resp = client.post(
        "/settings/folders", data=_form(action="add", folder="clips", path="."), headers=_HTMX
    )
    assert resp.status_code == 200
    assert "overlaps an already-registered folder" in resp.text
    assert set(_folders(session_factory)) == {"audio", "clips/raw"}


def test_remove_drops_folder_and_its_pack_mapping(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "podcasts").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "podcasts")
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="podcasts", pack="generic", path="."),
        headers=_HTMX,
    )
    assert _packs(session_factory) == {"podcasts": "generic"}
    client.post(
        "/settings/folders", data=_form(action="remove", folder="podcasts", path="."), headers=_HTMX
    )
    assert _folders(session_factory) == []
    assert _packs(session_factory) == {}  # no orphan mapping survives


def test_remove_unregistered_is_noop(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/folders", data=_form(action="remove", folder="ghost", path="."), headers=_HTMX
    )
    assert resp.status_code == 200


# ----------------------------------------------------------------- pack picker


def test_pack_set_then_default_clears(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "pods")
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", pack="generic", path="."),
        headers=_HTMX,
    )
    assert _packs(session_factory) == {"pods": "generic"}
    # The "Default" option submits an explicit sentinel (not "") — it clears the pack.
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", pack=PACK_DEFAULT_SENTINEL, path="."),
        headers=_HTMX,
    )
    assert _packs(session_factory) == {}


def test_pack_rejects_unknown_name(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "pods")
    resp = client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", pack="does-not-exist", path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 200
    assert "Unknown domain pack" in resp.text
    assert _packs(session_factory) == {}


def test_pack_rejects_unregistered_folder(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/folders",
        data=_form(action="pack", folder="ghost", pack="generic", path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 200
    assert "not registered" in resp.text
    assert _packs(session_factory) == {}  # nothing was written


def test_pack_then_remove_leaves_no_orphan(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "pods")
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", pack="generic", path="."),
        headers=_HTMX,
    )
    client.post(
        "/settings/folders", data=_form(action="remove", folder="pods", path="."), headers=_HTMX
    )
    # Deleting the folder row drops its pack with it — a pack can only exist on a
    # registered folder (it is a column on the row, not a separate mapping).
    assert set(_packs(session_factory)).issubset(set(_folders(session_factory)))


# ------------------------------------------------------- honest pack degradation


def test_stale_pack_renders_as_unavailable(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    # Seed a folder mapped to a pack the (default) registry does not offer.
    _seed_folder(session_factory, "pods", pack="ghostpack")
    body = client.get("/settings").text
    assert "ghostpack (unavailable)" in body  # honest, not a false "Default"


def test_registry_failure_disables_pack_selection(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # DOMAIN_PACKS_DIR pointing at a non-directory makes available_domain_packs
    # raise DomainPackError; the page must degrade honestly, not 500.
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("x")
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root, domain_packs_dir=not_a_dir)
    # Seed a folder + pack directly (the route's validator can't run with a broken
    # registry) so the render must handle a stored pack under total registry failure.
    _seed_folder(session_factory, "pods", pack="somepack")
    body = client.get("/settings").text
    assert "Domain packs can't be listed" in body
    assert "disabled" in body  # the select is disabled
    # The stored pack is still shown honestly, not silently replaced by "Default".
    assert "somepack (unavailable)" in body


def test_registry_down_pack_submit_preserves_mapping(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # With the registry down the panel disables the <select> AND the Set button, and
    # a disabled select submits no `pack` field. A submit that still reaches the route
    # (belt-and-braces / no-JS) must be a no-op, NEVER read as "select Default" and
    # silently wipe the stored pack — the honest-degradation contract.
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("x")
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root, domain_packs_dir=not_a_dir)
    _seed_folder(session_factory, "pods", pack="somepack")
    # action=pack with NO pack field (exactly what a disabled select submits).
    resp = client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 200
    assert _packs(session_factory) == {"pods": "somepack"}  # pack preserved
    # The disabled control renders on the button too, not only the select.
    assert "<button type=\"submit\" disabled>Set</button>" in resp.text


def test_nonhtmx_pack_error_rerenders_page_with_message_and_path(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # A no-JS (non-HTMX) mutation failure must re-render the whole Settings page with
    # the error inline — not a silent 303 that discards it and looks like success —
    # and keep the operator's browse position instead of snapping back to the root.
    packs = _make_packs_dir(tmp_path, "interview")
    (media_root / "interviews").mkdir()
    (media_root / "interviews" / "sub").mkdir()
    client, _ = make_client(session_factory, media_root, domain_packs_dir=packs)
    _add(client, "interviews")
    resp = client.post(
        "/settings/folders",
        data=_form(action="pack", folder="interviews", pack="ghostpack", path="interviews"),
        follow_redirects=False,
    )
    assert resp.status_code == 200  # full page, not a 303 redirect that drops the error
    assert "Unknown domain pack: ghostpack." in resp.text
    assert 'id="folders"' in resp.text  # the whole settings page rendered
    assert ">sub/<" in resp.text  # panel is at path=interviews (its child), not root
    assert _packs(session_factory) == {}  # the rejected pack never landed


# --------------------------------------------------------- CSRF / gating / bounds


def test_wrong_csrf_scope_is_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    # A CSRF_SETUP token must not authorise a CSRF_SETTINGS route.
    resp = client.post(
        "/settings/folders",
        data={
            "csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETUP),
            "action": "add",
            "folder": "pods",
            "path": ".",
        },
    )
    assert resp.status_code == 403
    assert _folders(session_factory) == []


def test_settings_folders_are_onboarding_gated(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root, onboarded=False)
    # Un-onboarded, the protected settings routes 303 back to the wizard.
    assert client.get("/settings/folders/browse", follow_redirects=False).status_code == 303
    posted = client.post(
        "/settings/folders",
        data=_form(action="add", folder="x", path="."),
        follow_redirects=False,
    )
    assert posted.status_code == 303


def test_oversized_folder_field_is_rejected(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/folders",
        data=_form(action="add", folder="a" * 5000, path="."),
        headers=_HTMX,
    )
    assert resp.status_code == 422  # Form(max_length=4096) bound
    assert _folders(session_factory) == []


def test_non_hx_mutation_redirects(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    resp = client.post(
        "/settings/folders",
        data=_form(action="add", folder="pods", path="."),
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/settings?")


# ------------------------------------------------------------------ concurrency


def test_concurrent_adds_preserve_both_folders(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Without the registration advisory lock, two overlapping adds could each read
    # the same base list and one clobber the other (or slip a nesting pair past the
    # overlap check). Prove they serialise: while A holds the lock mid-add, B blocks,
    # and once A commits B is admitted and appends — both folders survive.
    for name in ("seed", "a", "b"):
        (media_root / name).mkdir()
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_scope(session_factory) as pre:
        assert register_folder(pre, settings, "seed") is None  # pre-create a row

    error: dict[str, BaseException] = {}

    def _add_b() -> None:
        try:
            with session_scope(session_factory) as sb:
                assert register_folder(sb, settings, "b") is None
        except BaseException as exc:  # surface to the assertion
            error["b"] = exc

    with session_scope(session_factory) as sa:
        assert register_folder(sa, settings, "a") is None  # A takes the advisory lock
        thread = threading.Thread(target=_add_b)
        thread.start()
        for _ in range(500):  # wait until B is genuinely blocked on the lock
            with session_factory() as w:
                blocked = w.execute(
                    text("SELECT count(*) FROM pg_locks WHERE NOT granted")
                ).scalar_one()
            if blocked >= 1:
                break
            time.sleep(0.01)
        else:
            thread.join(timeout=5)
            pytest.fail("second add never blocked on the registration lock")
        # Exiting the `with` commits A and releases the lock; B is then admitted.

    thread.join(timeout=10)
    assert not error, f"concurrent add raised: {error.get('b')!r}"
    assert set(_folders(session_factory)) == {"seed", "a", "b"}  # no lost update


# ---------------------------------------------------------- reaches ingest read


def test_mapping_reaches_ingest_read_path(
    session_factory: sessionmaker[Session], media_root: Path, tmp_path: Path
) -> None:
    # A UI-written folder→pack mapping must actually freeze that pack onto a run
    # submitted under the folder — proving the write reaches the #11 read path,
    # not just the database.
    packs = _make_packs_dir(tmp_path, "interview")
    (media_root / "interviews").mkdir()
    client, settings = make_client(session_factory, media_root, domain_packs_dir=packs)
    _add(client, "interviews")
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="interviews", pack="interview", path="."),
        headers=_HTMX,
    )
    with session_scope(session_factory) as session:
        result = submit_media_item(session, "interviews/clip.wav", settings=settings)
        run = session.get(PipelineRun, result.run_id)
        assert run.domain_pack is not None
        pack_name = run.domain_pack["name"]
    assert pack_name == "interview"  # the mapped pack, not the default


def test_submit_assigns_folder_membership(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A run submitted under a registered folder records its folder membership
    # (issue #153): the MediaItem's media_folder_id points at the deepest containing
    # folder, and a file outside every registration stays NULL.
    (media_root / "interviews").mkdir()
    client, settings = make_client(session_factory, media_root)
    _add(client, "interviews")
    with session_scope(session_factory) as session:
        folder_id = session.execute(
            select(MediaFolder.id).where(MediaFolder.path == "interviews")
        ).scalar_one()
        inside_result = submit_media_item(session, "interviews/clip.wav", settings=settings)
        outside_result = submit_media_item(session, "loose/elsewhere.wav", settings=settings)
        inside = session.get(PipelineRun, inside_result.run_id)
        outside = session.get(PipelineRun, outside_result.run_id)
        assert inside.media_item.media_folder_id == folder_id
        assert outside.media_item.media_folder_id is None


def test_registered_missing_folder_is_flagged(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "gone").mkdir()
    client, _ = make_client(session_factory, media_root)
    _add(client, "gone")
    (media_root / "gone").rmdir()  # remove it on disk after registering
    body = client.get("/settings").text
    assert "no longer on disk" in body


def test_folder_panel_context_loads_packs_once(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # A light guard that the panel context builder is self-contained and returns
    # the expected keys (the per-request render path other tests exercise).
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_factory() as session:
        ctx = _folder_panel_context(
            session, settings, action_prefix="/settings/folders", csrf="tok", path="."
        )
    assert ctx["folders_mutate_action"] == "/settings/folders"
    assert ctx["folder_packs_available"] is True
    assert "generic" in ctx["folder_pack_names"]
