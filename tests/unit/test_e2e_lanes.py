"""The real-pipeline lane's expected identities come from ``VOXINT_E2E_LANE``.

An unset, empty or unknown lane must be an error naming the valid lanes, so an
operator who asked for the gate can never run it against a guessed device.
"""

from __future__ import annotations

import pytest

from tests.e2e.lanes import LANE_DEVICES, expected_services

SERVICE_KEYS = ("asr_url", "diarizer_url", "embedder_url")


@pytest.mark.parametrize("lane", [None, "", "   ", "gpu", "nvidia", "metal", "cuda,cpu"])
def test_unset_empty_or_unknown_lane_is_an_error(lane: str | None) -> None:
    with pytest.raises(ValueError) as excinfo:
        expected_services(lane)
    message = str(excinfo.value)
    assert "VOXINT_E2E_LANE" in message
    assert "cuda, rocm, cpu" in message
    assert repr(lane) in message


@pytest.mark.parametrize(
    ("lane", "devices"),
    [
        ("cuda", ("cuda", "cuda", "cuda")),
        ("rocm", ("rocm", "cpu", "cpu")),
        ("cpu", ("cpu", "cpu", "cpu")),
    ],
)
def test_each_lane_expects_its_overlay_devices(lane: str, devices: tuple[str, ...]) -> None:
    expected = expected_services(lane)
    assert tuple(expected) == SERVICE_KEYS
    assert tuple(expected[key]["device"] for key in SERVICE_KEYS) == devices


def test_valid_lanes_are_exactly_the_shipped_overlays() -> None:
    assert set(LANE_DEVICES) == {"cuda", "rocm", "cpu"}


def test_lane_name_is_trimmed_and_case_insensitive() -> None:
    assert expected_services(" CUDA ") == expected_services("cuda")


@pytest.mark.parametrize("lane", sorted(LANE_DEVICES))
def test_model_identity_pins_do_not_vary_by_lane(lane: str) -> None:
    expected = expected_services(lane)
    assert expected["asr_url"]["service"] == "whisper"
    assert expected["asr_url"]["model"] == "large-v2"
    assert expected["diarizer_url"]["service"] == "pyannote"
    assert expected["diarizer_url"]["model"] == "pyannote/speaker-diarization-3.1"
    assert expected["embedder_url"]["service"] == "titanet"
