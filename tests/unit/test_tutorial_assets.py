"""The bundled guided-tutorial assets: validity, provenance, and self-consistency.

These never touch the database or a speech synthesizer — they only read the
committed package data, so they run without Postgres or espeak-ng.
"""

import hashlib
import shutil
import subprocess
import wave

import pytest

from voxint.cli import build_parser
from voxint.tutorial import resources


def test_committed_wav_is_16khz_mono_pcm() -> None:
    with resources.sample_wav_path() as path, wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16-bit => pcm_s16le
        assert w.getnframes() > 0


def test_committed_wav_matches_provenance_sha256() -> None:
    digest = hashlib.sha256(resources.load_sample_wav_bytes()).hexdigest()
    assert digest == resources.load_provenance()["wav_sha256"]


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_committed_wav_probes_as_16khz_mono() -> None:
    with resources.sample_wav_path() as path:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=sample_rate,channels,codec_name",
                "-of", "default=nw=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    assert "sample_rate=16000" in out.stdout
    assert "channels=1" in out.stdout
    assert "codec_name=pcm_s16le" in out.stdout


def test_layout_has_three_labels_and_a_groundable_speaker() -> None:
    layout = resources.load_layout()
    roster_label = layout["roster_speaker"]["label"]
    heard_label = layout["heard_name"]["label"]
    unresolved_label = layout["unresolved_label"]
    assert len({roster_label, heard_label, unresolved_label}) == 3
    assert {u["label"] for u in layout["utterances"]} == {
        roster_label,
        heard_label,
        unresolved_label,
    }
    # The grounded label must honestly clear the grounded gates (>=3 turns, >=10s)
    # so the fixture stays credible even if a future re-match recomputes it.
    grounded_turns = [u for u in layout["utterances"] if u["label"] == roster_label]
    assert len(grounded_turns) >= 3
    assert sum(u["end"] - u["start"] for u in grounded_turns) >= 10.0
    # Timings are monotonic and non-degenerate.
    for utt in layout["utterances"]:
        assert utt["end"] > utt["start"] >= 0.0


def test_expected_transcript_agrees_with_layout() -> None:
    layout = resources.load_layout()
    segments = resources.load_expected_transcript()["segments"]
    utterances = sorted(layout["utterances"], key=lambda u: u["index"])
    assert len(segments) == len(utterances)

    roster_label = layout["roster_speaker"]["label"]
    roster_name = layout["roster_speaker"]["display_name"]
    heard_name = layout["heard_name"]["name"]
    for seg, utt in zip(segments, utterances, strict=True):
        assert seg["segment_index"] == utt["index"]
        assert seg["diarization_label"] == utt["label"]
        assert seg["text"] == utt["text"]
        expected = roster_name if utt["label"] == roster_label else utt["label"]
        assert seg["speaker"] == expected
    # The heard name is a suggestion, never an attribution — it appears nowhere as
    # a resolved speaker in the ground-truth transcript.
    assert all(seg["speaker"] != heard_name for seg in segments)


def test_tutorial_seed_subcommand_requires_a_verb() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tutorial"])  # subcommand required
    args = parser.parse_args(["tutorial", "seed"])
    assert args.tutorial_command == "seed"
    assert hasattr(args, "fn")
