"""Pure-logic unit tests for the native E2E lifecycle tool (no DB, no network).

Guards the tool-neutral pieces: the ``state.env`` parser (which must stay in
parity with the launcher's ``env_value_from_file`` dotenv semantics), the
``NativeConfig`` DSN/URL composer, the Vite-manifest bundle extractor, and the
``Location``-header run-id parser. The DB read-back verifier and the HTTP steps
are exercised in ``tests/integration/test_native_e2e_lifecycle.py``.
"""

from __future__ import annotations

import pytest
from tools.native_e2e_lifecycle import (
    NativeConfig,
    build_parser,
    entry_url,
    manifest_bundles,
    parse_state_env,
    run_id_from_location,
)

# A representative state.env exactly as voxint-native.sh's write_state_env emits it.
_GOOD_STATE = (
    "# Written by voxint-native.sh setup -- ports + secrets (mode 0600).\n"
    "PG_PORT=5433\n"
    "REDIS_PORT=6380\n"
    "API_PORT=8081\n"
    "DB_PASSWORD=db-secret-hex\n"
    "VOXINT_PASSWORD=api-secret-hex\n"
    "CSRF_SECRET=csrf-secret-hex\n"
)


# --------------------------------------------------------------------------- #
# parse_state_env — parity with env_value_from_file (voxint-native.sh:234-249)
# --------------------------------------------------------------------------- #
def test_parse_state_env_reads_all_six_keys() -> None:
    values = parse_state_env(_GOOD_STATE)
    assert values == {
        "PG_PORT": "5433",
        "REDIS_PORT": "6380",
        "API_PORT": "8081",
        "DB_PASSWORD": "db-secret-hex",
        "VOXINT_PASSWORD": "api-secret-hex",
        "CSRF_SECRET": "csrf-secret-hex",
    }


def test_parse_state_env_last_assignment_wins() -> None:
    assert parse_state_env("PG_PORT=5432\nPG_PORT=9999\n")["PG_PORT"] == "9999"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("K=8081\r", "8081"),  # one trailing CR (CRLF file) stripped
        ("K=  spaced  ", "spaced"),  # surrounding blanks stripped
        ("K='single'", "single"),  # one matched single-quote pair
        ('K="double"', "double"),  # one matched double-quote pair
        ('K=  "both"  ', "both"),  # blanks then quotes
        ("K=abc=def", "abc=def"),  # everything after the first '=' kept
        ("K='mismatch\"", "'mismatch\""),  # unmatched quotes untouched
        ("K=", ""),  # empty value
    ],
)
def test_parse_state_env_value_cleaning(line: str, expected: str) -> None:
    assert parse_state_env(line + "\n")["K"] == expected


def test_parse_state_env_ignores_comments_blanks_and_non_kv_lines() -> None:
    text = "# comment\n\n   \nnot a kv line\n#PG_PORT=shadow\nPG_PORT=5433\n"
    assert parse_state_env(text) == {"PG_PORT": "5433"}


# --------------------------------------------------------------------------- #
# NativeConfig
# --------------------------------------------------------------------------- #
def test_native_config_composes_urls_and_auth() -> None:
    cfg = NativeConfig.from_state_text(_GOOD_STATE)
    assert cfg.database_url == "postgresql+psycopg://voxint:db-secret-hex@127.0.0.1:5433/voxint"
    assert cfg.redis_url == "redis://127.0.0.1:6380/0"
    assert cfg.base_url == "http://127.0.0.1:8081"
    assert cfg.auth == ("admin", "api-secret-hex")


@pytest.mark.parametrize("drop", ["PG_PORT", "DB_PASSWORD", "CSRF_SECRET", "VOXINT_PASSWORD"])
def test_native_config_rejects_missing_key(drop: str) -> None:
    text = "\n".join(line for line in _GOOD_STATE.splitlines() if not line.startswith(drop + "="))
    with pytest.raises(ValueError, match=rf"missing required key.*{drop}"):
        NativeConfig.from_state_text(text)


def test_native_config_rejects_empty_valued_key() -> None:
    # An empty value is as unusable as an absent key (the launcher would never
    # write one, but a truncated file could).
    truncated = _GOOD_STATE.replace("DB_PASSWORD=db-secret-hex", "DB_PASSWORD=")
    with pytest.raises(ValueError, match=r"missing required key.*DB_PASSWORD"):
        NativeConfig.from_state_text(truncated)


# --------------------------------------------------------------------------- #
# manifest_bundles / entry_url
# --------------------------------------------------------------------------- #
_MANIFEST = {
    "src/main.ts": {"file": "assets/main-AAA.js", "name": "main", "isEntry": True},
    "src/styles/tailwind.css": {"file": "assets/tailwind-BBB.css", "isEntry": True},
    "src/entries/review-stepper.tsx": {
        "file": "assets/review-stepper-CCC.js",
        "css": ["assets/review-stepper-DDD.css"],
        "isEntry": True,
    },
    "_shared-EEE.js": {"file": "assets/shared-EEE.js"},
}


def test_manifest_bundles_extracts_files_and_css_sorted_and_deduped() -> None:
    bundles = manifest_bundles(_MANIFEST)
    assert bundles == sorted(bundles)  # sorted
    assert len(bundles) == len(set(bundles))  # deduped
    assert "/static/app/assets/main-AAA.js" in bundles
    assert "/static/app/assets/tailwind-BBB.css" in bundles
    assert "/static/app/assets/review-stepper-DDD.css" in bundles  # css array picked up
    assert "/static/app/assets/shared-EEE.js" in bundles


@pytest.mark.parametrize("bad", [{}, {"x": {}}, {"x": "notadict"}, "notamanifest"])
def test_manifest_bundles_rejects_empty_or_invalid(bad: object) -> None:
    with pytest.raises(ValueError):
        manifest_bundles(bad)  # type: ignore[arg-type]


def test_entry_url_keys_by_source_stem() -> None:
    assert entry_url(_MANIFEST, "main") == "/static/app/assets/main-AAA.js"
    assert entry_url(_MANIFEST, "tailwind") == "/static/app/assets/tailwind-BBB.css"
    assert entry_url(_MANIFEST, "does-not-exist") is None


# --------------------------------------------------------------------------- #
# run_id_from_location
# --------------------------------------------------------------------------- #
_UUID = "12345678-1234-1234-1234-1234567890ab"


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        (f"/runs/{_UUID}", _UUID),
        (f"/runs/{_UUID}?enqueue=deferred", _UUID),
        (f"http://127.0.0.1:8081/runs/{_UUID}", _UUID),
        ("/runs/not-a-uuid", None),
        ("/setup", None),
        ("", None),
    ],
)
def test_run_id_from_location(location: str, expected: str | None) -> None:
    assert run_id_from_location(location) == expected


# --------------------------------------------------------------------------- #
# build_parser
# --------------------------------------------------------------------------- #
def test_parser_exposes_every_subcommand() -> None:
    parser = build_parser()
    for cmd in ("env", "smoke", "onboard", "submit", "poll", "verify", "drive"):
        extra: list[str] = []
        if cmd in ("submit", "drive"):
            extra += ["--file", "x.wav"]
        if cmd in ("poll", "verify"):
            extra += ["--run-id", _UUID]
        args = parser.parse_args([cmd, *extra])
        assert args.command == cmd
        assert callable(args.func)


def test_parser_requires_run_id_for_poll_and_verify() -> None:
    parser = build_parser()
    for cmd in ("poll", "verify"):
        with pytest.raises(SystemExit):
            parser.parse_args([cmd])
