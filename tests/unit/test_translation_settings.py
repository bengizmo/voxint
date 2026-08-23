"""Translation settings resolvers + invariants + freshness hash (#133).

Pure tests: resolvers over an in-memory AppSettings row (row-over-env
precedence), the self-contained flag invariant, the language normalizer, and
the freshness hash's content-only properties (text/structure flip it; nothing
else is even part of the snapshot).
"""

import uuid

import pytest

from voxint.app_settings import (
    resolve_effective_translation_autogenerate,
    resolve_effective_translation_target_language,
    translation_flags_ok,
)
from voxint.config import Settings
from voxint.db.models import AppSettings
from voxint.enrichment.translation_jobs import normalized_language
from voxint.enrichment.translations import (
    TranslationLineSource,
    TranslationSource,
    translation_source_hash,
)


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


class TestResolvers:
    def test_target_language_row_wins_over_env(self) -> None:
        settings = make_settings(translation_target_language="fr")
        row = AppSettings(id=1, translation_target_language="  es  ")
        assert resolve_effective_translation_target_language(row, settings) == "es"

    def test_target_language_blank_row_inherits_env(self) -> None:
        settings = make_settings(translation_target_language="fr")
        assert (
            resolve_effective_translation_target_language(
                AppSettings(id=1, translation_target_language="   "), settings
            )
            == "fr"
        )
        assert resolve_effective_translation_target_language(None, settings) == "fr"

    def test_target_language_unset_everywhere_is_none(self) -> None:
        settings = make_settings()
        assert resolve_effective_translation_target_language(None, settings) is None
        assert (
            resolve_effective_translation_target_language(AppSettings(id=1), settings) is None
        )

    def test_autogenerate_tri_state(self) -> None:
        settings = make_settings(
            translation_autogenerate=True, translation_target_language="es"
        )
        assert resolve_effective_translation_autogenerate(None, settings) is True
        assert (
            resolve_effective_translation_autogenerate(
                AppSettings(id=1, translation_autogenerate=False), settings
            )
            is False
        )
        assert (
            resolve_effective_translation_autogenerate(AppSettings(id=1), settings) is True
        )


class TestInvariants:
    def test_autogenerate_requires_target(self) -> None:
        assert translation_flags_ok(autogenerate=True, target_language=None) is not None
        assert translation_flags_ok(autogenerate=True, target_language="  ") is not None
        assert translation_flags_ok(autogenerate=True, target_language="es") is None
        assert translation_flags_ok(autogenerate=False, target_language=None) is None

    def test_unknown_code_rejected_even_without_autogenerate(self) -> None:
        assert translation_flags_ok(autogenerate=False, target_language="klingon") is not None

    def test_env_time_validator_wires_through_settings(self) -> None:
        with pytest.raises(ValueError, match="translation_autogenerate"):
            make_settings(translation_autogenerate=True)
        with pytest.raises(ValueError, match="not a language code"):
            make_settings(translation_target_language="xx-not-a-code")
        # A valid pair boots.
        make_settings(translation_autogenerate=True, translation_target_language="es")

    def test_normalized_language(self) -> None:
        assert normalized_language(" ES ") == "es"
        assert normalized_language(None) is None
        assert normalized_language("   ") is None


def _source(lines: list[TranslationLineSource], *, lang: str | None = "en") -> TranslationSource:
    return TranslationSource(
        pipeline_run_id=uuid.UUID(int=1), source_language=lang, lines=tuple(lines)
    )


def _line(i: int, text: str, *, seg: uuid.UUID | None = None, ws: int | None = None,
          we: int | None = None) -> TranslationLineSource:
    return TranslationLineSource(
        line_index=i, segment_id=seg or uuid.UUID(int=100 + i), word_start=ws,
        word_end=we, text=text,
    )


class TestFreshnessHash:
    def test_deterministic(self) -> None:
        a = _source([_line(0, "Hello."), _line(1, "Bye.")])
        b = _source([_line(0, "Hello."), _line(1, "Bye.")])
        assert translation_source_hash(a) == translation_source_hash(b)

    def test_text_edit_flips(self) -> None:
        base = _source([_line(0, "Hello.")])
        edited = _source([_line(0, "Hello!")])
        assert translation_source_hash(base) != translation_source_hash(edited)

    def test_structure_change_flips(self) -> None:
        seg = uuid.UUID(int=7)
        whole = _source([_line(0, "One two.", seg=seg)])
        split = _source(
            [_line(0, "One", seg=seg, ws=0, we=1), _line(1, "two.", seg=seg, ws=1, we=2)]
        )
        assert translation_source_hash(whole) != translation_source_hash(split)

    def test_source_language_hint_flips(self) -> None:
        lines = [_line(0, "Hello.")]
        assert translation_source_hash(_source(lines, lang="en")) != translation_source_hash(
            _source(lines, lang=None)
        )
