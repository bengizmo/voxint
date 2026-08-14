"""Pure-side unit tests for the stats module: parse_since + the renderers.

The query functions need Postgres and live in the integration suite; everything
here runs against a hand-built ``SystemStats`` with no database.
"""

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from voxint.api.stats_query import (
    StageDurationStat,
    StageFailureCount,
    SystemStats,
    format_stats_text,
    parse_since,
    render_prometheus,
    stats_to_json,
)

_NOW = datetime(2026, 8, 14, 12, 0, 0, tzinfo=UTC)


def _sample() -> SystemStats:
    return SystemStats(
        status_counts={"completed": 20, "failed": 3, "queued": 1},
        stage_failure_counts=(StageFailureCount(stage="transcribe", attempt_count=2),),
        stage_durations=(
            StageDurationStat(stage="transcribe", attempt_count=18, avg_seconds=42.5),
        ),
        roster_size=7,
        runs_created_since=_NOW - timedelta(hours=24),
        runs_created_count=5,
        generated_at=_NOW,
        since=_NOW - timedelta(hours=24),
    )


# ---- parse_since ------------------------------------------------------------


def test_parse_since_relative_hours() -> None:
    assert parse_since("24h", now=_NOW) == _NOW - timedelta(hours=24)


def test_parse_since_relative_days() -> None:
    assert parse_since("7d", now=_NOW) == _NOW - timedelta(days=7)


def test_parse_since_iso_aware_normalized_to_utc() -> None:
    aware = datetime(2026, 8, 14, 6, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    result = parse_since(aware.isoformat(), now=_NOW)
    assert result == aware.astimezone(UTC)
    assert result.tzinfo == UTC


def test_parse_since_rejects_naive_iso() -> None:
    with pytest.raises(ValueError, match="timezone offset"):
        parse_since("2026-08-14T06:00:00", now=_NOW)


def test_parse_since_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="invalid --since"):
        parse_since("last tuesday", now=_NOW)


def test_parse_since_rejects_naive_now() -> None:
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        parse_since("24h", now=datetime(2026, 8, 14, 12, 0, 0))


# ---- text renderer ----------------------------------------------------------


def test_format_stats_text_shows_nonzero_statuses_and_ends_with_newline() -> None:
    text = format_stats_text(_sample())
    lines = text.splitlines()
    assert any(line.startswith("  completed") and line.endswith("20") for line in lines)
    assert any(line.startswith("  failed") and line.endswith("3") for line in lines)
    assert any(line.startswith("  queued") and line.endswith("1") for line in lines)
    assert "Roster size: 7" in text
    assert any(line.startswith("  transcribe") and line.endswith("2") for line in lines)
    assert "42.50s" in text
    assert text.endswith("\n")


def test_format_stats_text_empty_sections() -> None:
    empty = SystemStats(
        status_counts={},
        stage_failure_counts=(),
        stage_durations=(),
        roster_size=0,
        runs_created_since=_NOW,
        runs_created_count=0,
        generated_at=_NOW,
        since=_NOW,
    )
    text = format_stats_text(empty)
    assert "(none)" in text


# ---- json renderer ----------------------------------------------------------


def test_stats_to_json_zero_fills_every_status_and_serialises() -> None:
    payload = stats_to_json(_sample())
    # round-trips through json (no Decimal / datetime leaking through)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["status_counts"]["cancelled"] == 0  # absent status → 0
    assert decoded["status_counts"]["completed"] == 20
    assert decoded["roster_size"] == 7
    assert decoded["runs_created_count"] == 5
    assert decoded["stage_durations"][0]["avg_seconds"] == 42.5


# ---- prometheus renderer ----------------------------------------------------


def test_render_prometheus_zero_fills_all_series() -> None:
    text = render_prometheus(_sample())
    # every RunStatus present, absent ones zero-filled
    assert 'voxint_runs_total{status="completed"} 20' in text
    assert 'voxint_runs_total{status="cancelled"} 0' in text
    # every Stage present for failures + durations, zero-filled when absent
    assert 'voxint_stage_failures_total{stage="transcribe"} 2' in text
    assert 'voxint_stage_failures_total{stage="finalize"} 0' in text
    assert 'voxint_stage_duration_seconds{stage="transcribe"} 42.5' in text
    assert 'voxint_stage_duration_seconds{stage="acquire"} 0.0' in text
    assert "voxint_roster_speakers 7" in text
    assert "voxint_runs_created_24h 5" in text


def test_render_prometheus_has_help_type_and_trailing_newline() -> None:
    text = render_prometheus(_sample())
    assert "# HELP voxint_runs_total" in text
    assert "# TYPE voxint_runs_total gauge" in text
    assert text.endswith("\n")
