"""Filler settings and reading notes against real Postgres."""

from html import unescape
from pathlib import Path
from secrets import token_urlsafe

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.conftest import seed_onboarded
from tests.integration.test_runs_api import make_run
from voxint.api.app import create_app
from voxint.api.auth import SESSION_COOKIE, create_session, new_session_token
from voxint.api.csrf import CSRF_SETTINGS, CSRF_SETUP, mint_csrf_token
from voxint.api.routers.deps import require_onboarded
from voxint.app_settings import get_app_settings, get_or_create
from voxint.config import Settings
from voxint.db.models import UserRole
from voxint.export.filler_lists import DEFAULT_FILLER_LIST, PRESET_VERSION, TIER_1, TIER_2
from voxint.users import create_user

CREDS = ("reviewer", "s3cret")
_CSRF_KEY = "settings-fillers-test-csrf-key"


@pytest.fixture(params=[True, False])
def console_mode(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def make_client(
    session_factory: sessionmaker[Session], tmp_path: Path, console_mode: bool,
    *, env_add: str = "", env_keep: str = "", onboarded: bool = True,
) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_user=CREDS[0], voxint_password=CREDS[1], csrf_secret=_CSRF_KEY,
        media_root=tmp_path, console_settings_enabled=console_mode,
        voxint_fillers_add=env_add, voxint_fillers_keep=env_keep,
    )
    app = create_app(settings=settings, session_factory=session_factory)
    if onboarded:
        seed_onboarded(session_factory)
    else:
        # Exercise validation before row creation without the first-run redirect.
        app.dependency_overrides[require_onboarded] = lambda: None
    client = TestClient(app)
    client.auth = CREDS
    return client


def form(add: str = "", keep: str = "", suggested: str = "") -> dict[str, str]:
    return {"csrf_token": mint_csrf_token(_CSRF_KEY, CSRF_SETTINGS),
            "add": add, "keep": keep, "suggested": suggested}


def test_section_and_save(
    session_factory: sessionmaker[Session], tmp_path: Path, console_mode: bool,
) -> None:
    client = make_client(session_factory, tmp_path, console_mode)
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.text
    assert '<section id="fillers" aria-labelledby="fillers-heading">' in body
    assert (f"Removed by default (preset {PRESET_VERSION}): {', '.join(TIER_1)}. "
            "A word you keep below comes off this list.") in body
    assert body.count('name="suggested"') == 12
    for entry in TIER_2:
        assert f'value="{entry}">' in body
    assert "Voxint never removes like, so, right or well on its own:" in body
    assert f"Removed now: {', '.join(DEFAULT_FILLER_LIST.words)}" in body
    # The independent form must start after the features form ends.
    assert body.index('</form>', body.index('action="/settings"')) < body.index('id="fillers"')
    response = client.post('/settings/fillers', data=form('you see', 'um', 'I mean'),
                           follow_redirects=False)
    assert response.status_code == 303
    assert response.headers['location'] == '/settings#fillers'
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None
        assert row.fillers_add == {"en": ["I mean", "you see"]}
        assert row.fillers_keep == {"en": ["um"]}
    response = client.get('/settings')
    assert 'value="I mean" checked' in response.text
    assert 'value="you see" checked' in response.text
    assert response.context['fillers_add_text'] == ''
    effective = response.context['fillers_effective']
    assert 'I mean' in effective and 'you see' in effective and 'um' not in effective
    client.post('/settings/fillers', data=form('to be honest', 'hm'))
    response = client.get('/settings')
    assert 'rows="5">to be honest</textarea>' in response.text
    assert ('These kept entries have no effect because they are not on the list: hm.'
            in response.text)


def test_inherited_environment_and_row_precedence(
    session_factory: sessionmaker[Session], tmp_path: Path, console_mode: bool,
) -> None:
    client = make_client(session_factory, tmp_path, console_mode,
                         env_add='I mean', env_keep='um')
    response = client.get('/settings')
    assert "Additions from this installation's environment: I mean." in response.text
    assert "Words kept by this installation's environment: um." in response.text
    assert 'value="I mean">' in response.text
    assert response.context['fillers_keep_text'] == ''
    assert 'I mean' in response.context['fillers_effective']
    client.post('/settings/fillers', data=form())
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.fillers_add is None and row.fillers_keep is None
    client.post('/settings/fillers', data=form('you see', 'uh'))
    response = client.get('/settings')
    assert 'I mean' not in response.context['fillers_effective']
    assert 'um' in response.context['fillers_effective']
    assert not response.context['fillers_env_add'] and not response.context['fillers_env_keep']
    client.post('/settings/fillers', data=form())
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.fillers_add is None and row.fillers_keep is None


def test_other_languages_and_explicit_empty(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    client = make_client(session_factory, tmp_path, True, env_add='I mean')
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.fillers_add = {'en': [], 'fr': ['euh']}
        row.fillers_keep = {'en': ['um'], 'fr': ['ben']}
        session.commit()
    response = client.get('/settings')
    assert not response.context['fillers_env_add']
    assert 'I mean' not in response.context['fillers_effective']
    client.post('/settings/fillers', data=form())
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None
        assert row.fillers_add == {'fr': ['euh']}
        assert row.fillers_keep == {'fr': ['ben']}
    assert 'I mean' in client.get('/settings').context['fillers_effective']


_INVALID = [
    ('add', 'um2'), ('keep', 'a.b'), ('add', 'one two three four five six'),
    ('add', 'x' * 41),
    ('add', '\n'.join('term' + chr(97 + i // 26) + chr(97 + i % 26) for i in range(101))),
]


@pytest.mark.parametrize('field,value', _INVALID,
                         ids=['digit', 'punctuation', 'six-words', '41-chars', '101-entries'])
@pytest.mark.parametrize('existing', [True, False])
def test_invalid_preserves_submission_and_database(
    session_factory: sessionmaker[Session], tmp_path: Path, console_mode: bool,
    field: str, value: str, existing: bool,
) -> None:
    client = make_client(session_factory, tmp_path, console_mode, onboarded=existing)
    if existing:
        with session_factory() as session:
            row = get_or_create(session, llm_enabled_default=False)
            row.fillers_add = {'en': ['old']}
            row.fillers_keep = {'en': ['um']}
            session.commit()
    data = form('submitted', 'uh', 'I mean')
    data[field] = value
    response = client.post('/settings/fillers', data=data)
    assert response.status_code == 422
    prefix = 'Words to remove' if field == 'add' else 'Words to keep'
    assert f'role="alert">{prefix}:' in response.text
    assert response.context[f'fillers_{field}_text'] == value
    assert f'rows="5">{value}</textarea>' in unescape(response.text)
    assert 'value="I mean" checked' in response.text
    if console_mode:
        assert 'role="tab" aria-current="true">General</a>' in response.text
    effective = response.context['fillers_effective']
    assert ('old' in effective) is existing
    assert 'submitted' not in effective and 'I mean' not in effective
    with session_factory() as session:
        row = get_app_settings(session)
        if existing:
            assert row is not None
            assert row.fillers_add == {'en': ['old']}
            assert row.fillers_keep == {'en': ['um']}
        else:
            assert row is None


# Tokens are minted inside the test: a nonce in a parametrize value would give
# each xdist worker a different test id.
@pytest.mark.parametrize('token', ['missing', 'wrong-action'])
def test_csrf(
    session_factory: sessionmaker[Session], tmp_path: Path, token: str,
) -> None:
    client = make_client(session_factory, tmp_path, True)
    data = {'add': 'hello'}
    if token == 'wrong-action':
        data['csrf_token'] = mint_csrf_token(_CSRF_KEY, CSRF_SETUP)
    assert client.post('/settings/fillers', data=data).status_code == 403
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.fillers_add is None and row.fillers_keep is None


@pytest.mark.parametrize('malformed', ['oops', {'en': 'oops'}, {'en': ['um2', 3, 'I MEAN']}])
def test_malformed_row_repair(
    session_factory: sessionmaker[Session], tmp_path: Path, malformed: object,
) -> None:
    client = make_client(session_factory, tmp_path, True)
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.fillers_add = malformed
        session.commit()
    response = client.get('/settings')
    assert response.status_code == 200
    assert 'role="alert">The saved filler word list is not valid.' in response.text
    assert "The current list can't be shown until you save a valid one." in response.text
    if isinstance(malformed, dict) and isinstance(malformed.get('en'), list):
        assert response.context['fillers_add_text'] == 'um2'
        assert 'value="I mean" checked' in response.text
    assert client.get('/settings/ai').status_code == 200
    assert client.get('/settings/media').status_code == 200
    response = client.post('/settings/fillers', data=form('valid'), follow_redirects=False)
    assert response.status_code == 303
    assert not client.get('/settings').context['fillers_error']


@pytest.mark.parametrize('custom,drop,text', [
    (True, True, 'hello world'), (False, True, 'hello world'),
    (True, False, 'hello world'), (True, True, 'hello'), (False, True, 'um'),
])
def test_read_mode_note(
    session_factory: sessionmaker[Session], tmp_path: Path,
    custom: bool, drop: bool, text: str,
) -> None:
    client = make_client(session_factory, tmp_path, True)
    if custom:
        client.post('/settings/fillers', data=form('hello'))
    with session_factory() as session:
        run_id = make_run(session, segments=[('S0', text, None)])
    response = client.get(f'/runs/{run_id}/transcript',
                          params={'read': '1', 'fillers': 'drop' if drop else 'keep'})
    assert response.status_code == 200
    assert ('href="/settings#fillers">your filler list</a>' in response.text) is (custom and drop)
    if drop and text == 'hello world':
        note = ('Left out 1 filler word. The saved transcript is unchanged.' if custom
                else 'Left out no filler words. The saved transcript is unchanged.')
        suffix = (' Filler words are matched using '
                  '<a href="/settings#fillers">your filler list</a>.' if custom else '')
        assert f'<p class="muted">{note}{suffix}</p>' in response.text
    if custom and drop and text == 'hello':
        assert ('The filters left out every word, using '
                '<a href="/settings#fillers">your filler list</a>. '
                'The saved transcript is unchanged.') in response.text


def test_forged_suggestion_and_remove_everything(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    client = make_client(session_factory, tmp_path, True)
    assert client.post('/settings/fillers', data=form(suggested='bad2')).status_code == 422
    response = client.post('/settings/fillers', data=form(suggested='custom'),
                           follow_redirects=False)
    assert response.status_code == 303
    client.post('/settings/fillers', data=form(keep='\n'.join(TIER_1)))
    assert 'Removed now: nothing' in client.get('/settings').text


def test_both_invalid_lists_reported_together(
    session_factory: sessionmaker[Session], tmp_path: Path,
) -> None:
    client = make_client(session_factory, tmp_path, True)
    response = client.post('/settings/fillers', data=form('um2', 'a.b'))
    assert response.status_code == 422
    error = response.context['fillers_error']
    assert error.startswith('Words to remove: ') and ' Words to keep: ' in error
    with session_factory() as session:
        row = get_app_settings(session)
        assert row is not None and row.fillers_add is None and row.fillers_keep is None




@pytest.mark.parametrize('role,linked', [(UserRole.ADMIN, True), (UserRole.REVIEWER, False)])
def test_read_mode_link_only_for_admins(
    session_factory: sessionmaker[Session], tmp_path: Path, role: UserRole, linked: bool,
) -> None:
    seed_onboarded(session_factory)
    with session_factory() as session:
        row = get_or_create(session, llm_enabled_default=False)
        row.fillers_add = {'en': ['hello']}
        # The first account is always made an admin, so seed one before the member.
        # Passwords are random: the test signs in with a session cookie instead.
        create_user(session, username='owner', password=token_urlsafe(16))
        member = create_user(session, username='member', password=token_urlsafe(16),
                             role=role)
        session.commit()
        cookie = new_session_token()
        create_session(session, user_id=member.id, token=cookie, ttl_seconds=3600)
        session.commit()
        run_id = make_run(session, segments=[('S0', 'hello world', None)])
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voxint_multi_user=True, voxint_user='ignored', voxint_password='ignored',
        csrf_secret=_CSRF_KEY, media_root=tmp_path,
    )
    client = TestClient(create_app(settings=settings, session_factory=session_factory),
                        cookies={SESSION_COOKIE: cookie}, follow_redirects=False)
    response = client.get(f'/runs/{run_id}/transcript',
                          params={'read': '1', 'fillers': 'drop'})
    assert response.status_code == 200
    assert 'Filler words are matched using' in response.text
    assert ('href="/settings#fillers"' in response.text) is linked
    if not linked:
        assert 'Filler words are matched using your filler list.' in response.text
