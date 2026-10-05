"""Operator counts and restart controls include applied clean-up marks."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from voxint.adjudication import turns, word_marks
from voxint.api.routers.legacy_runs import read_filter_note
from voxint.cli import build_parser
from voxint.export.filler_lists import DEFAULT_FILLER_LIST
from voxint.export.service import filler_report_comment
from voxint.ingest.service import RestartImpact


@pytest.mark.parametrize(
    "omitted,kept,suffix",
    [
        (0, 0, ""),
        (2, 0, "; words you left out: 2"),
        (0, 1, "; fillers you kept: 1"),
        (2, 1, "; words you left out: 2; fillers you kept: 1"),
    ],
)
def test_comment_counts_only_applied_marks(omitted: int, kept: int, suffix: str) -> None:
    assert filler_report_comment(DEFAULT_FILLER_LIST, 3, omitted=omitted, kept=kept) == (
        f"<!-- Filler words left out: 3. Preset {DEFAULT_FILLER_LIST.preset_version}"
        f"{suffix}. The saved transcript is unchanged. -->"
    )


@pytest.mark.parametrize(
    "omitted,kept,copy",
    [
        (2, 0, " Words you left out: 2."),
        (0, 1, " Fillers you kept: 1."),
        (2, 1, " Words you left out: 2. Fillers you kept: 1."),
    ],
)
def test_read_note_counts(omitted: int, kept: int, copy: str) -> None:
    assert read_filter_note(
        drop_fillers=True,
        drop_repeats=False,
        fillers_removed=0,
        repeats_removed=0,
        omitted=omitted,
        kept=kept,
    ) == (f"Left out no filler words.{copy} The saved transcript is unchanged.")


def test_neutral_types_remain_reexported() -> None:
    assert word_marks.WordMarkKey is turns.WordMarkKey
    assert word_marks.EffectiveMarks is turns.EffectiveMarks


def test_mark_only_impact_is_editorial_work() -> None:
    assert RestartImpact(0, 0, 0, word_marks=1).has_editorial_work
    assert not RestartImpact(0, 0, 0).has_editorial_work


def test_cli_restart_help_mentions_marks(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["restart", "--help"])
    assert caught.value.code == 0
    assert "clean-up marks" in capsys.readouterr().out


def test_restart_script_requires_ack_only_when_marks_would_be_lost() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to exercise the template's restart controls")
    path = Path(__file__).resolve().parents[2] / (
        "src/voxint/api/templates/legacy_runs/_run_detail_body.html"
    )
    scripts = re.findall(r"<script>(.*?)</script>", path.read_text(), flags=re.DOTALL)
    assert len(scripts) == 1
    setup = """
const assert = require('node:assert/strict');
const controls = {};
for (const id of ['restart-stage', 'restart-submit', 'restart-void-warn',
  'restart-void-ack', 'restart-void-msg', 'restart-label-warn', 'restart-ack',
  'restart-label-msg', 'restart-editorial-warn', 'restart-editorial-ack',
  'restart-editorial-msg', 'restart-form']) {
  controls[id] = {style: {}, required: false, checked: false, listeners: {},
    addEventListener(event, fn) { this.listeners[event] = fn; }};
}
const sel = controls['restart-stage'];
sel.selectedIndex = 0;
sel.options = [
  {dataset: {corrections: '0', verifications: '0', wordMarks: '2'}},
  {dataset: {corrections: '0', verifications: '0', wordMarks: '0'}}
];
global.document = {getElementById(id) { return controls[id]; }};
"""
    assertions = """
sel.listeners.change();
const box = controls['restart-editorial-ack'];
const warning = controls['restart-editorial-warn'];
assert.equal(box.required, true);
assert.equal(warning.style.display, '');
assert.match(controls['restart-editorial-msg'].textContent, /2 clean-up mark/);
box.checked = true;
sel.selectedIndex = 1;
sel.listeners.change();
assert.equal(box.required, false);
assert.equal(box.checked, false);
assert.equal(warning.style.display, 'none');
sel.selectedIndex = 0;
sel.listeners.change();
assert.equal(box.required, true);
process.stdout.write(JSON.stringify({checked: box.checked, required: box.required}));
"""
    result = subprocess.run(
        [node, "-e", setup + scripts[0] + assertions], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"checked": False, "required": True}
