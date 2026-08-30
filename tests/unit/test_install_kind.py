"""install_kind() parsing (#317): validated marker, degrade to unknown."""

import pytest

from voxint.api.settings_view import install_kind
from voxint.config import Settings


def _settings(kind: str | None) -> Settings:
    return Settings(voxint_install_kind=kind)


@pytest.mark.parametrize("kind", ["docker", "native"])
def test_known_kinds_pass_through(kind: str) -> None:
    assert install_kind(_settings(kind)) == kind


@pytest.mark.parametrize("kind", [None, "", "kubernetes", "DOCKER", "nativ e"])
def test_unset_or_unrecognized_degrades_to_unknown(kind: str | None) -> None:
    # Unset (dev-from-source), a typo, or a value this build does not know must
    # degrade to "unknown" rather than guessing or raising.
    assert install_kind(_settings(kind)) == "unknown"
