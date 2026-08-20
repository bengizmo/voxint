"""Pin-parity + isolation contract for the ``eval-quality`` extra (issue #97).

``tools/eval_quality.py`` scores hypothesis diarization/ASR against public
ground truth using ``pyannote.metrics``. That scorer pulls ``pyannote.core>=6``
(numpy>=2.2.2 / scipy>=1.15.1), which conflicts with the diarizer *service*'s
pinned ``pyannote.core==5.0.0``. The two never share an environment, and this
test makes that separation load-bearing rather than a comment:

- the extra pins ``pyannote.metrics`` exactly, and ``uv.lock`` resolves it there
  (so the scorer numerics can't drift under a resolver bump);
- the scorer stays out of ``dev`` and out of the diarizer service image, so a
  future edit can't silently co-resident the incompatible pyannote.core lines.
"""

import re
import tomllib

from tests.contracts.conftest import REPO_ROOT

EXTRA = "eval-quality"
PACKAGE = "pyannote.metrics"
PACKAGE_NORMALIZED = "pyannote-metrics"


def _optional_dependencies() -> dict[str, list[str]]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return data["project"]["optional-dependencies"]


def _extra_pin(spec_list: list[str], dist: str) -> str | None:
    """Return the ``==`` version for ``dist`` in a requirement list, or None."""
    for spec in spec_list:
        name = re.split(r"[<>=!~ \[]", spec, maxsplit=1)[0].strip().lower()
        if name.replace(".", "-") == dist.replace(".", "-").lower() and "==" in spec:
            return spec.split("==", 1)[1].strip()
    return None


def test_eval_quality_extra_pins_pyannote_metrics_exactly() -> None:
    extras = _optional_dependencies()
    assert EXTRA in extras, f"pyproject lost the {EXTRA!r} optional-dependencies extra"
    pin = _extra_pin(extras[EXTRA], PACKAGE)
    assert pin is not None, f"{EXTRA} extra must pin {PACKAGE} with =="


def test_uv_lock_resolves_the_pinned_scorer() -> None:
    pin = _extra_pin(_optional_dependencies()[EXTRA], PACKAGE)
    assert pin is not None
    lock = (REPO_ROOT / "uv.lock").read_text()
    match = re.search(
        rf'name = "{re.escape(PACKAGE_NORMALIZED)}"\nversion = "([0-9][0-9.]*)"', lock
    )
    assert match is not None, f"uv.lock does not resolve {PACKAGE_NORMALIZED}"
    assert match.group(1) == pin, (
        f"uv.lock resolves {PACKAGE_NORMALIZED} {match.group(1)}, extra pins {pin}"
    )


def test_scorer_stays_out_of_dev_extra() -> None:
    # Keeping it out of `dev` is what keeps the incompatible pyannote.core>=6
    # off the default developer/test environment. A `pytest` run must not drag
    # the scorer in; it is opt-in behind `--extra eval-quality`.
    dev = _optional_dependencies().get("dev", [])
    assert _extra_pin(dev, PACKAGE) is None, (
        f"{PACKAGE} leaked into the dev extra — keep it isolated in {EXTRA}"
    )


def test_scorer_never_enters_the_diarizer_service_image() -> None:
    # The diarizer service pins pyannote.core==5.0.0 (the tuned 3.1 clustering
    # hyperparameters live there); pyannote.metrics 4.1 needs pyannote.core>=6.
    # If the scorer were ever added to the service requirements, the image would
    # fail to resolve. Assert the conflicting line stays put and the scorer is
    # absent.
    reqs = (REPO_ROOT / "services" / "pyannote" / "requirements.txt").read_text()
    assert PACKAGE_NORMALIZED not in reqs.replace(".", "-").lower(), (
        f"{PACKAGE} must never be installed into the diarizer service image"
    )
    assert re.search(r"^pyannote\.core==5\.0\.0$", reqs, re.MULTILINE), (
        "diarizer service must keep its pyannote.core==5.0.0 pin (tuned 3.1 "
        "clustering hyperparameters); the eval-quality scorer stays in the host "
        "uv env only"
    )
