"""The gated-feature fragments must remediate in plain language, not raw env vars.

Issue #62 AC3: a non-technical operator seeing an off feature must be pointed at
the in-UI Settings toggle, never at an ``ENRICHMENT_*`` / ``VOXINT_*`` /
``LLM_ENABLED`` environment variable as their only remedy. A contract test so a
future edit cannot silently regress the copy back to env-var instructions.
"""

from pathlib import Path

import pytest

_FRAGMENTS = Path(__file__).resolve().parents[2] / "src/voxint/api/templates/fragments"
_FORBIDDEN = ("ENRICHMENT_", "VOXINT_", "LLM_ENABLED", "YTDLP_")


@pytest.mark.parametrize("name", ["run_assets.html", "research.html"])
def test_gated_fragment_has_no_env_var_remediation(name: str) -> None:
    text = (_FRAGMENTS / name).read_text(encoding="utf-8")
    for token in _FORBIDDEN:
        assert token not in text, f"{name} still names env var {token} as remediation"
    # It must instead send the operator to Settings.
    assert "Settings" in text
