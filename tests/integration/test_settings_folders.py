"""Settings → Media folders + the folder browser (issue #63), end to end.

Covers the shared folder-panel routes on the Settings mount (``/settings/folders``,
``/settings/folders/browse``) against real Postgres: the directory browser
(containment-safe, never disclosing outside MEDIA_ROOT), registering/unregistering
folders, the per-folder domain-pack picker (set / clear-to-default / reject unknown
/ reject unregistered), the load-bearing invariant that every
``folder_domain_packs`` key is a registered ``media_folders`` entry, honest
rendering when a stored pack or the whole registry is unavailable, CSRF scope,
onboarding gating, input bounds, that concurrent mutations serialise on the
singleton row (no lost update), and that a UI-set mapping actually reaches the
ingest read path that freezes a run's pack.
"""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from voxint.api.app import (
    _add_media_folder,
    _folder_panel_context,
    create_app,
)
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.db.session import session_scope
from voxint.ingest.service import submit_media_item

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


def _row(session_factory: sessionmaker[Session]) -> AppSettings | None:
    with session_factory() as session:
        return get_app_settings(session)


def _folders(session_factory: sessionmaker[Session]) -> list[str]:
    row = _row(session_factory)
    return list(row.media_folders) if row and row.media_folders else []


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
    row = _row(session_factory)
    assert row is not None and row.media_folders == ["podcasts"]


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
    assert (_row(session_factory) or AppSettings()).folder_domain_packs == {"podcasts": "generic"}
    client.post(
        "/settings/folders", data=_form(action="remove", folder="podcasts", path="."), headers=_HTMX
    )
    row = _row(session_factory)
    assert row is not None
    assert row.media_folders == []
    assert row.folder_domain_packs == {}  # no orphan mapping survives


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
    assert (_row(session_factory) or AppSettings()).folder_domain_packs == {"pods": "generic"}
    # "" is the sole Default sentinel — it removes the mapping.
    client.post(
        "/settings/folders",
        data=_form(action="pack", folder="pods", pack="", path="."),
        headers=_HTMX,
    )
    assert (_row(session_factory) or AppSettings()).folder_domain_packs == {}


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
    assert (_row(session_factory) or AppSettings()).folder_domain_packs == {}


def test_pack_rejects_unregistered_folder_preserving_invariant(
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
    # The mapping-key ⊆ media_folders invariant holds: nothing was written.
    row = _row(session_factory)
    assert row is None or row.folder_domain_packs == {}


def test_pack_then_remove_leaves_no_orphan(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    # Whichever order pack-set and remove commit in (they serialise on the row
    # lock), the keys ⊆ folders invariant survives.
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
    row = _row(session_factory)
    assert row is not None
    assert set(row.folder_domain_packs).issubset(set(row.media_folders))


# ------------------------------------------------------- honest pack degradation


def test_stale_pack_renders_as_unavailable(
    session_factory: sessionmaker[Session], media_root: Path
) -> None:
    (media_root / "pods").mkdir()
    client, _ = make_client(session_factory, media_root)
    # Seed a mapping to a pack the (default) registry does not offer.
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.media_folders = ["pods"]
        row.folder_domain_packs = {"pods": "ghostpack"}
        session.commit()
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
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.media_folders = ["pods"]
        row.folder_domain_packs = {"pods": "somepack"}
        session.commit()
    body = client.get("/settings").text
    assert "Domain packs can't be listed" in body
    assert "disabled" in body  # the select is disabled
    # The stored pack is still shown honestly, not silently replaced by "Default".
    assert "somepack (unavailable)" in body


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
    # Without the FOR UPDATE row lock, two overlapping adds each read the same
    # base list and one clobbers the other. Prove they serialise: while A holds
    # the row mid-add, B blocks, and once A commits B rereads and appends — both
    # folders survive.
    for name in ("seed", "a", "b"):
        (media_root / name).mkdir()
    settings = Settings(_env_file=None, media_root=media_root)  # type: ignore[call-arg]
    with session_scope(session_factory) as pre:
        assert _add_media_folder(pre, settings, "seed") is None  # pre-create the row

    error: dict[str, BaseException] = {}

    def _add_b() -> None:
        try:
            with session_scope(session_factory) as sb:
                assert _add_media_folder(sb, settings, "b") is None
        except BaseException as exc:  # surface to the assertion
            error["b"] = exc

    with session_scope(session_factory) as sa:
        assert _add_media_folder(sa, settings, "a") is None  # A locks the row
        thread = threading.Thread(target=_add_b)
        thread.start()
        for _ in range(500):  # wait until B is genuinely blocked on the row lock
            with session_factory() as w:
                blocked = w.execute(
                    text("SELECT count(*) FROM pg_locks WHERE NOT granted")
                ).scalar_one()
            if blocked >= 1:
                break
            time.sleep(0.01)
        else:
            thread.join(timeout=5)
            pytest.fail("second add never blocked on the row lock")
        # Exiting the `with` commits A and releases the lock; B is then admitted.

    thread.join(timeout=10)
    assert not error, f"concurrent add raised: {error.get('b')!r}"
    row = _row(session_factory)
    assert row is not None
    assert set(row.media_folders) == {"seed", "a", "b"}  # no lost update


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
        run = submit_media_item(session, "interviews/clip.wav", settings=settings)
        assert run.domain_pack is not None
        pack_name = run.domain_pack["name"]
    assert pack_name == "interview"  # the mapped pack, not the default


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
