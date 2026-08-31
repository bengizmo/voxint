"""Pure-logic unit tests for the browser E2E lifecycle tool (no DB, no browser).

The DB-backed seed + reconcile behaviour is covered in
``tests/integration/test_e2e_browser_lifecycle.py``; this file guards the
tool-neutral pieces: the disposable-DB guard (the fail-closed gate that stops a
copy-pasted live DSN from being dropped), the expectation parser, the admin-URL
derivation, and the build (un)staging that must always preserve ``.gitkeep``.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import pytest
from tools.e2e_browser_lifecycle import (
    _CORRECTED_SEGMENT_INDEX,
    _EDITOR_SEGMENTS,
    _EDITOR_SPLIT_ELIGIBLE,
    FIXTURE_CHOICES,
    GITKEEP,
    Expectation,
    _admin_url,
    _benchmark_segments,
    _faithful_word_timings,
    _guarded,
    _silent_wav_bytes,
    assert_disposable_db,
    build_parser,
    cmd_reconcile,
    cmd_seed,
    cmd_serve,
    cmd_setup,
    cmd_teardown,
    unstage_build,
)

_DISPOSABLE = "postgresql+psycopg://voxint:voxint@localhost:5432/voxint_e2e"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://voxint:voxint@localhost:5432/voxint_e2e",
        "postgresql+psycopg://voxint:voxint@localhost:5432/voxint_dev_test",
        "postgresql+psycopg://u:p@host/E2E_UPPER",  # case-insensitive
    ],
)
def test_assert_disposable_db_accepts_throwaway_names(url: str) -> None:
    assert_disposable_db(url)  # does not raise


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg://voxint:voxint@localhost:5432/voxint",  # the LIVE db
        "postgresql+psycopg://voxint:voxint@localhost:5432/production",
    ],
)
def test_assert_disposable_db_rejects_non_disposable_names(url: str) -> None:
    with pytest.raises(ValueError, match="DISPOSABLE"):
        assert_disposable_db(url)


def test_assert_disposable_db_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="plain identifier"):
        assert_disposable_db("postgresql+psycopg://voxint:voxint@localhost:5432/")


@pytest.mark.parametrize("key", ["dbname", "database", "DBNAME"])
def test_assert_disposable_db_rejects_dbname_query_override(key: str) -> None:
    # A dbname/database query param can override the path at connect time and
    # point destructive ops at the LIVE db — must be refused (codex+kimi).
    url = f"postgresql+psycopg://u:p@host:5432/voxint_e2e?{key}=voxint"
    with pytest.raises(ValueError, match="query parameter"):
        assert_disposable_db(url)


@pytest.mark.parametrize(
    "name",
    ["voxint_e2e; DROP", "voxint-e2e", 'voxint"e2e', "voxint e2e"],
)
def test_assert_disposable_db_rejects_non_identifier_names(name: str) -> None:
    from urllib.parse import quote

    url = f"postgresql+psycopg://u:p@host:5432/{quote(name)}"
    with pytest.raises(ValueError, match="plain identifier"):
        assert_disposable_db(url)


def test_admin_url_repoints_to_maintenance_db() -> None:
    got = _admin_url("postgresql+psycopg://voxint:pw@host:5432/voxint_e2e")
    assert got == "postgresql+psycopg://voxint:pw@host:5432/postgres"


def test_expectation_from_dict_roundtrips() -> None:
    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [0, 2],
            "corrections": {"1": "fixed text"},
            "progress": {"verified": 2, "total": 5},
        }
    )
    assert expect.verified_indexes == frozenset({0, 2})
    assert expect.corrections == {1: "fixed text"}
    assert expect.progress == (2, 5)


def test_expectation_defaults_are_empty() -> None:
    expect = Expectation.from_dict({"progress": {"verified": 0, "total": 3}})
    assert expect.verified_indexes == frozenset()
    assert expect.corrections == {}
    assert expect.progress == (0, 3)


@pytest.mark.parametrize(
    "data",
    [
        {"verified_segment_indexes": "nope", "progress": {"verified": 0, "total": 1}},
        {"corrections": [], "progress": {"verified": 0, "total": 1}},
        {"progress": {"verified": 0}},  # missing total
        {},  # missing progress
        # Strict validation (never coerce): these silently coerced before review.
        {"verified_segment_indexes": [True], "progress": {"verified": 0, "total": 1}},
        {"verified_segment_indexes": [1.9], "progress": {"verified": 0, "total": 1}},
        {"verified_segment_indexes": [-1], "progress": {"verified": 0, "total": 1}},
        {"verified_segment_indexes": [1, 1], "progress": {"verified": 0, "total": 1}},
        {"corrections": {"0": None}, "progress": {"verified": 0, "total": 1}},
        {"corrections": {"0": 5}, "progress": {"verified": 0, "total": 1}},
        {"progress": {"verified": True, "total": 1}},
    ],
)
def test_expectation_from_dict_rejects_malformed(data: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        Expectation.from_dict(data)


def test_expectation_from_dict_rejects_non_object_root() -> None:
    with pytest.raises(ValueError):
        Expectation.from_dict([1, 2])  # type: ignore[arg-type]


def test_silent_wav_bytes_is_a_valid_wav_of_expected_length() -> None:
    payload = _silent_wav_bytes(2.0)
    import io

    with wave.open(io.BytesIO(payload), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getnframes() == 32000  # 16000 * 2.0s


def test_unstage_build_removes_artifacts_but_keeps_gitkeep(tmp_path: Path) -> None:
    static = tmp_path / "app"
    static.mkdir()
    (static / GITKEEP).write_text("")
    (static / "assets").mkdir()
    (static / "assets" / "app-abc123.js").write_text("// bundle")
    (static / ".vite").mkdir()
    (static / ".vite" / "manifest.json").write_text("{}")

    removed = unstage_build(static)

    assert sorted(removed) == [".vite", "assets"]
    assert (static / GITKEEP).exists()
    assert not (static / "assets").exists()
    assert not (static / ".vite").exists()


def test_unstage_build_on_missing_dir_is_a_noop(tmp_path: Path) -> None:
    assert unstage_build(tmp_path / "does-not-exist") == []


def test_guarded_returns_disposable_url_unchanged() -> None:
    assert _guarded(_DISPOSABLE) == _DISPOSABLE


@pytest.mark.parametrize(
    "url",
    [None, "", "postgresql+psycopg://voxint:voxint@localhost:5432/voxint"],
)
def test_guarded_exits_on_missing_or_live_url(url: str | None) -> None:
    # The fail-closed guard must sys.exit(1) rather than let a destructive
    # subcommand run against a missing or live DSN.
    with pytest.raises(SystemExit) as exc:
        _guarded(url)
    assert exc.value.code == 1


def test_cmd_reconcile_exits_on_bad_run_id() -> None:
    args = argparse.Namespace(
        database_url=_DISPOSABLE, run_id="not-a-uuid", expect_file=None, expect=None
    )
    with pytest.raises(SystemExit) as exc:
        cmd_reconcile(args)
    assert exc.value.code == 1


@pytest.mark.parametrize(
    ("argv", "func"),
    [
        (["setup"], cmd_setup),
        (["seed", "--database-url", _DISPOSABLE], cmd_seed),
        (["serve", "--database-url", _DISPOSABLE], cmd_serve),
        (
            ["reconcile", "--database-url", _DISPOSABLE, "--run-id", "x", "--expect", "{}"],
            cmd_reconcile,
        ),
        (["teardown"], cmd_teardown),
    ],
)
def test_parser_wires_each_subcommand(argv: list[str], func: object) -> None:
    args = build_parser().parse_args(argv)
    assert args.func is func


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_cmd_reconcile_exits_when_no_expectation_given() -> None:
    args = argparse.Namespace(
        database_url=_DISPOSABLE,
        run_id="e5fc9e6a-d7bd-4fd7-ad28-72a9e83996d6",
        expect_file=None,
        expect=None,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_reconcile(args)
    assert exc.value.code == 1


# --- Phase 6a: editor fixture + extended reconciliation ---


def test_editor_segments_has_four_speakers() -> None:
    labels = {label for label, _, _ in _EDITOR_SEGMENTS}
    assert labels == {"S0", "S1", "S2", "S3"}


def test_editor_segments_has_varied_confidence() -> None:
    confs = [c for _, _, c in _EDITOR_SEGMENTS]
    assert any(c is not None and c < 0.6 for c in confs)
    assert any(c is None for c in confs)
    assert any(c is not None and c >= 0.6 for c in confs)


def test_benchmark_segments_produces_2000() -> None:
    segs = _benchmark_segments()
    assert len(segs) == 2000


def test_benchmark_segments_custom_count() -> None:
    segs = _benchmark_segments(count=50)
    assert len(segs) == 50


def test_fixture_choices_tuple() -> None:
    assert FIXTURE_CHOICES == ("review", "editor", "benchmark")


def test_parser_seed_fixture_flag() -> None:
    for fixture in FIXTURE_CHOICES:
        args = build_parser().parse_args(
            ["seed", "--database-url", _DISPOSABLE, "--fixture", fixture]
        )
        assert args.fixture == fixture


def test_parser_seed_fixture_default() -> None:
    args = build_parser().parse_args(["seed", "--database-url", _DISPOSABLE])
    assert args.fixture == "review"


def test_expectation_from_dict_with_splits() -> None:
    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": 5},
            "split_parent_indexes": [2, 4],
        }
    )
    assert expect.split_parent_indexes == frozenset({2, 4})


def test_expectation_from_dict_with_annotations() -> None:
    expect = Expectation.from_dict(
        {
            "verified_segment_indexes": [],
            "corrections": {},
            "progress": {"verified": 0, "total": 5},
            "expected_annotations": 3,
        }
    )
    assert expect.expected_annotations == 3


def test_expectation_from_dict_defaults_editor_fields() -> None:
    expect = Expectation.from_dict({"progress": {"verified": 0, "total": 3}})
    assert expect.split_parent_indexes == frozenset()
    assert expect.expected_annotations is None


@pytest.mark.parametrize(
    "data",
    [
        {"split_parent_indexes": "nope", "progress": {"verified": 0, "total": 1}},
        {"split_parent_indexes": [True], "progress": {"verified": 0, "total": 1}},
        {"split_parent_indexes": [-1], "progress": {"verified": 0, "total": 1}},
        {"split_parent_indexes": [1, 1], "progress": {"verified": 0, "total": 1}},
        {"expected_annotations": True, "progress": {"verified": 0, "total": 1}},
        {"expected_annotations": -1, "progress": {"verified": 0, "total": 1}},
        {"expected_annotations": "3", "progress": {"verified": 0, "total": 1}},
    ],
)
def test_expectation_from_dict_rejects_malformed_editor_fields(
    data: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        Expectation.from_dict(data)


def test_faithful_word_timings_reconstructs_text() -> None:
    text = "hello world foo bar"
    words = _faithful_word_timings(text, 0.0, 4.0)
    assert len(words) == 4
    joined = "".join(str(w["word"]) for w in words)
    assert joined == text


def test_faithful_word_timings_monotonic_starts() -> None:
    words = _faithful_word_timings("a b c d e", 10.0, 15.0)
    starts = [float(w["start"]) for w in words]  # type: ignore[arg-type]
    assert starts == sorted(starts)
    assert all(float(w["start"]) >= 10.0 for w in words)  # type: ignore[arg-type]
    assert all(float(w["end"]) <= 15.0 for w in words)  # type: ignore[arg-type]


def test_split_eligible_excludes_corrected_index() -> None:
    assert _CORRECTED_SEGMENT_INDEX not in _EDITOR_SPLIT_ELIGIBLE


def test_split_eligible_indexes_within_editor_range() -> None:
    assert all(0 <= i < len(_EDITOR_SEGMENTS) for i in _EDITOR_SPLIT_ELIGIBLE)


def test_seed_browser_run_rejects_unknown_fixture() -> None:
    with pytest.raises(ValueError, match="unknown fixture"):
        from unittest.mock import MagicMock

        seed = __import__("tools.e2e_browser_lifecycle", fromlist=["seed_browser_run"])
        seed.seed_browser_run(MagicMock(), MagicMock(), fixture="bogus")
