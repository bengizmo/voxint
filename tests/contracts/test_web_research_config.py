"""Contract: the web-research settings surface stays documented and gated.

Every web-research Settings field must appear (commented or not) in
``.env.example`` under its env-var name — a field added without documentation
is invisible to operators, and a documented var that no longer exists is a
silent no-op. Also pins the capability's two load-bearing config invariants:
disabled-by-default, and independence from ``llm_enabled`` (no cross-validator
may couple them — issue #39's "configuring an LLM must not imply egress").
"""

import re

from tests.contracts.conftest import REPO_ROOT
from voxint.config import Settings

_WEB_RESEARCH_FIELDS = [
    name
    for name in Settings.model_fields
    if name == "voxint_web_research" or name.startswith(("web_search_", "web_read_"))
]


def test_every_web_research_field_is_documented_in_env_example() -> None:
    env_example = (REPO_ROOT / ".env.example").read_text()
    assert _WEB_RESEARCH_FIELDS, "web-research settings fields disappeared"
    missing = [
        name
        for name in _WEB_RESEARCH_FIELDS
        if not re.search(rf"^#?\s*{name.upper()}=", env_example, re.MULTILINE)
    ]
    assert not missing, f".env.example lacks lines for: {missing}"


def test_web_research_defaults_off() -> None:
    assert Settings(_env_file=None).voxint_web_research is False


def test_llm_and_web_research_stay_independent() -> None:
    # Either capability alone must construct cleanly — a future validator
    # coupling them would break one of these.
    llm_only = Settings(_env_file=None, llm_enabled=True)
    assert llm_only.voxint_web_research is False
    research_only = Settings(
        _env_file=None,
        voxint_web_research=True,
        web_search_base_url="http://searx.lan:8888",
    )
    assert research_only.llm_enabled is False
