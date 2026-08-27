"""Vocabulary contract (epic #205, COPY #209).

A grep-based audit that catches prohibited user-facing terms in template
files. Each screen-refresh issue (R1-R6) is responsible for applying the
vocabulary; this test catches regressions and newly introduced violations.

Allowlist rationale:
- Legacy templates (``legacy_review/``, ``legacy_runs/``) are being
  retired by the P3 editor; they carry an allowlist rather than a
  piecemeal vocabulary rewrite.
- Template comments (``{# ... #}``) are not user-facing.
- Template variable/attribute access (``item.diarization_label``) is an
  internal identifier, not user-facing text.
- Settings model-config templates (``_models.html``) use technical terms
  in ``<details>`` disclosures, which the vocabulary contract explicitly
  permits.
"""

import re

from tests.contracts.conftest import REPO_ROOT
from voxint.api.vocabulary import PROHIBITED_PATTERNS, VOCABULARY

_TEMPLATES_DIR = REPO_ROOT / "src" / "voxint" / "api" / "templates"

_ALLOWLISTED_DIRS = {
    "legacy_review",
    "legacy_runs",
}

_ALLOWLISTED_FILES = {
    "settings/_models.html",
    "settings/setup.html",
    "settings/database.html",
    "fragments/export_menu.html",
}

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_EXPRESSION = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_JINJA_TAG = re.compile(r"\{%.*?%\}", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_CSS_BLOCK = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL)
_SCRIPT_BLOCK = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")


def _user_facing_text(content: str) -> str:
    """Strip everything that is not visible user-facing text: Jinja2
    constructs, HTML comments, CSS, scripts, and HTML tags."""
    content = _JINJA_COMMENT.sub("", content)
    content = _JINJA_EXPRESSION.sub("", content)
    content = _JINJA_TAG.sub("", content)
    content = _HTML_COMMENT.sub("", content)
    content = _CSS_BLOCK.sub("", content)
    content = _SCRIPT_BLOCK.sub("", content)
    content = _HTML_TAG.sub("", content)
    return content


def _is_allowlisted(path_relative: str) -> bool:
    parts = path_relative.split("/")
    if parts[0] in _ALLOWLISTED_DIRS:
        return True
    return path_relative in _ALLOWLISTED_FILES


def test_no_prohibited_terms_in_ops_console_templates() -> None:
    violations: list[str] = []
    for html_file in sorted(_TEMPLATES_DIR.rglob("*.html")):
        rel = str(html_file.relative_to(_TEMPLATES_DIR))
        if _is_allowlisted(rel):
            continue
        user_text = _user_facing_text(html_file.read_text()).lower()
        for term in PROHIBITED_PATTERNS:
            if term.lower() in user_text:
                violations.append(f"{rel}: prohibited term {term!r}")
    assert not violations, (
        "Prohibited vocabulary found in user-facing template text "
        "(see vocabulary.py PROHIBITED_PATTERNS):\n  "
        + "\n  ".join(violations)
    )


def test_vocabulary_mapping_is_nonempty() -> None:
    assert len(VOCABULARY) >= 10


def test_prohibited_patterns_are_subset_of_vocabulary() -> None:
    vocab_keys = set(VOCABULARY.keys())
    for pattern in PROHIBITED_PATTERNS:
        assert pattern in vocab_keys, (
            f"prohibited pattern {pattern!r} has no entry in VOCABULARY"
        )
