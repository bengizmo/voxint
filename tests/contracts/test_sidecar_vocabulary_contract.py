"""Sidecar hint bounds and unchanged ASR wire bytes (#741).

The four-path frozen JSON golden lives in test_config_resolution_freeze.py,
where the real database fixtures are available.
"""

import json
from pathlib import Path

import httpx
import pytest
import yaml

from voxint.clients.asr import HttpASRClient
from voxint.ingest import sidecar
from voxint.pipeline.stages.transcribe import INITIAL_PROMPT_MAX_CHARS, _initial_prompt


def test_sidecar_vocabulary_bounds_and_applied_key() -> None:
    assert sidecar.MAX_VOCABULARY_CHARS == INITIAL_PROMPT_MAX_CHARS
    assert "vocabulary" in sidecar._APPLIED_KEYS


@pytest.mark.parametrize("vocabulary, golden", [
    ((), None),
    (("alpha", "beta"), "alpha, beta"),
    (("explicit-term",), "explicit-term"),
    (("project-term",), "project-term"),
    (("folder-term",), "folder-term"),
    (("base-term", "glossary-term"), "base-term, glossary-term"),
    (("x" * 2001, "alpha", "beta"), "alpha, beta"),
    (("x" * 2000, "alpha"), "x" * 2000),
    (("x" * 1997, "a", "b"), "x" * 1997 + ", a"),
])
def test_empty_reserved_preserves_prompt_and_wire_bytes(
    vocabulary: tuple[str, ...], golden: str | None,
) -> None:
    seen: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(200, json={"language": "en", "segments": []})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as http:
        client = HttpASRClient("http://test", Path("/media"), timeout_seconds=5, client=http)
        for prompt in (_initial_prompt(vocabulary), _initial_prompt(vocabulary, ())):
            assert prompt == golden
            client.transcribe(Path("/media/test.wav"), initial_prompt=prompt)
    assert seen[0] == seen[1]
    expected_payload: dict[str, object] = {"path": "test.wav", "language": None}
    if golden is not None:
        expected_payload["initial_prompt"] = golden
    assert json.loads(seen[0]) == expected_payload
    if golden is None:
        assert seen[0] == b'{"path":"test.wav","language":null}'
    elif golden == "alpha, beta":
        assert seen[0] == b'{"path":"test.wav","language":null,"initial_prompt":"alpha, beta"}'


def test_every_accepted_sidecar_term_fits_reserved_budget() -> None:
    terms = [f"{i:02}" + "x" * 118 for i in range(16)] + ["y" * 48]
    parsed = sidecar.parse_sidecar(yaml.safe_dump({"vocabulary": terms}), source_name="x.yaml")
    assert len(", ".join(parsed.vocabulary)) == 2000
    assert _initial_prompt(("pack-term", *parsed.vocabulary), parsed.vocabulary) == ", ".join(terms)
