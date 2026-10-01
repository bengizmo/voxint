"""Pure-logic tests for the voiceclone target-voice pack tools (#664).

``tools/voiceclone_target_voice_core.py`` holds the numpy-only maths shared by the
PEP 723 generator (which runs on a rented GPU) and the screen tool (which runs in
the dev venv): spec validation, post-processing to the Chatterbox VC reference
format, WAV encoding and read-back, and the TitaNet screening maths. None of it
touches a model, a network or a GPU.
"""

import copy
import io
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tools import voiceclone_target_voice_core as core

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "tools" / "voiceclone_target_voices.spec.json"
SR = 24000


@pytest.fixture()
def spec() -> dict[str, Any]:
    return core.load_spec(SPEC_PATH)


@pytest.fixture()
def dsp(spec: dict[str, Any]) -> core.Dsp:
    return core.Dsp.from_spec(spec)


def _tone(seconds: float, amplitude: float = 0.5, freq: float = 180.0) -> np.ndarray:
    t = np.arange(round(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _speech_like(lead_s: float, body_s: float, tail_s: float, amplitude: float = 0.5) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(round(lead_s * SR), np.float32),
            _tone(body_s, amplitude),
            np.zeros(round(tail_s * SR), np.float32),
        ]
    )


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# --- spec ---------------------------------------------------------------------


def test_committed_spec_is_valid(spec: dict[str, Any]) -> None:
    assert [v["id"] for v in spec["voices"]] == ["target-a", "target-b", "target-c", "target-d"]
    assert spec["generator"]["reference_recording_used"] is False
    assert spec["release_tag"] == "voiceclone-target-voices-v1"


def test_committed_spec_pitch_bands_are_two_and_two(spec: dict[str, Any]) -> None:
    bands = [v["pitch_band"] for v in spec["voices"]]
    assert bands.count("lower") == 2 and bands.count("higher") == 2


def _mutated(spec: dict[str, Any], path: list[Any], value: Any) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    node = out
    for key in path[:-1]:
        node = node[key]
    if value is _DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return out


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (["schema_version"], 2, "schema_version"),
        (["release_tag"], "voiceclone-target-voices", "release_tag"),
        (["generator", "reference_recording_used"], True, "reference_recording_used"),
        (["generator", "hf_revision"], "main", "hf_revision"),
        (["generator", "code_commit"], "022e286", "code_commit"),
        (["generator", "weights_sha256", "model.safetensors"], "ab" * 31, "sha256"),
        (["generator", "dtype"], "float16", "dtype"),
        (["voices", 0, "id"], "Alice", "neutral"),
        (["voices", 1, "id"], "target-a", "unique"),
        (["voices", 0, "seeds"], [1, 1, 2], "seeds"),
        (["voices", 0, "seeds"], [], "seeds"),
        (["voices", 0, "seeds"], [1, True], "seeds"),
        (["voices", 0, "description"], "  ", "description"),
        (["voices", 0, "pitch_band"], "middle", "pitch_band"),
        (["voices"], [], "four voices"),
        (["text"], "", "text"),
        (["dsp", "sample_rate"], 16000, "sample_rate"),
        (["dsp", "channels"], 2, "channels"),
        (["dsp", "sample_format"], "pcm_f32le", "sample_format"),
        (["dsp", "dither"], True, "dither"),
        (["dsp", "max_samples"], 240001, "max_samples"),
        (["dsp", "min_samples"], 250000, "min_samples"),
        (["dsp", "max_samples"], 239040, "max_samples"),
        (["dsp", "peak_dbfs"], 0.5, "peak_dbfs"),
        (["dsp", "trim_frame_samples"], 0, "trim_frame_samples"),
        (["dsp", "fade_samples"], -1, "fade_samples"),
        (["screening", "reject_at_or_above"], 0.7, "0.6"),
        (["screening", "swap_headroom"], -0.1, "swap_headroom"),
        (["chatterbox", "min_sources_reached"], 16, "min_sources_reached"),
        (["dsp", "peak_dbfs"], _DELETE, "peak_dbfs"),
    ],
)
def test_spec_validation_rejects(
    spec: dict[str, Any], path: list[Any], value: Any, match: str
) -> None:
    with pytest.raises(core.SpecError, match=match):
        core.validate_spec(_mutated(spec, path, value))


def test_spec_rejects_seed_reuse_across_voices(spec: dict[str, Any]) -> None:
    bad = _mutated(spec, ["voices", 1, "seeds"], [1101, 2102, 2103, 2104, 2105])
    with pytest.raises(core.SpecError, match="seed"):
        core.validate_spec(bad)


def test_spec_rejects_description_naming_a_person(spec: dict[str, Any]) -> None:
    bad = _mutated(spec, ["voices", 0, "description"], "A voice that sounds like a newsreader.")
    with pytest.raises(core.SpecError, match="attribute-only"):
        core.validate_spec(bad)


def test_load_spec_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "spec.json"
    p.write_text("[]")
    with pytest.raises(core.SpecError, match="object"):
        core.load_spec(p)


def test_dsp_constants_read_from_spec(dsp: core.Dsp) -> None:
    assert dsp.sample_rate == SR
    assert dsp.max_samples == 240000
    assert dsp.min_samples == 192000
    assert dsp.max_samples % dsp.align_samples == 0


# --- post-processing ------------------------------------------------------------


def test_postprocess_trims_aligns_and_normalizes(dsp: core.Dsp) -> None:
    audio = _speech_like(0.7, 9.3, 0.6)
    pcm = core.postprocess(audio, SR, dsp)
    assert pcm.dtype == np.int16
    assert len(pcm) % dsp.align_samples == 0
    assert dsp.min_samples <= len(pcm) <= dsp.max_samples
    # Trim keeps the configured pad before the onset and drops the rest.
    assert len(pcm) <= round(9.3 * SR) + 2 * dsp.trim_pad_samples
    ceiling = round(10 ** (dsp.peak_dbfs / 20) * 32767)
    assert int(np.max(np.abs(pcm.astype(np.int32)))) == ceiling


def test_postprocess_caps_long_audio_at_max_samples(dsp: core.Dsp) -> None:
    pcm = core.postprocess(_speech_like(0.1, 13.0, 0.1), SR, dsp)
    assert len(pcm) == dsp.max_samples


def test_postprocess_fades_both_edges(dsp: core.Dsp) -> None:
    pcm = core.postprocess(_tone(11.0, amplitude=0.9), SR, dsp)
    assert pcm[0] == 0
    assert abs(int(pcm[-1])) <= 1


def test_postprocess_full_scale_input_never_clips(dsp: core.Dsp) -> None:
    square = np.sign(_tone(10.5, amplitude=1.0)).astype(np.float32)
    square[square == 0] = 1.0
    pcm = core.postprocess(square * 1.7, SR, dsp)
    ceiling = round(10 ** (dsp.peak_dbfs / 20) * 32767)
    assert int(np.max(np.abs(pcm.astype(np.int32)))) == ceiling


def test_postprocess_rejects_short_after_trim(dsp: core.Dsp) -> None:
    with pytest.raises(core.CandidateRejected, match="shorter than"):
        core.postprocess(_speech_like(2.0, 7.0, 2.0), SR, dsp)


def test_postprocess_rejects_exactly_one_alignment_block_short(dsp: core.Dsp) -> None:
    n = dsp.min_samples - dsp.align_samples
    with pytest.raises(core.CandidateRejected):
        core.postprocess(_tone(n / SR), SR, dsp)


def test_postprocess_accepts_exact_minimum(dsp: core.Dsp) -> None:
    pcm = core.postprocess(_tone(dsp.min_samples / SR), SR, dsp)
    assert len(pcm) == dsp.min_samples


def test_postprocess_rejects_all_silence(dsp: core.Dsp) -> None:
    with pytest.raises(core.CandidateRejected, match="silent"):
        core.postprocess(np.zeros(SR * 11, np.float32), SR, dsp)


def test_postprocess_rejects_non_finite(dsp: core.Dsp) -> None:
    audio = _tone(10.5)
    audio[100] = np.nan
    with pytest.raises(core.CandidateRejected, match="non-finite"):
        core.postprocess(audio, SR, dsp)


def test_postprocess_rejects_wrong_rate_and_shape(dsp: core.Dsp) -> None:
    with pytest.raises(ValueError, match="sample rate"):
        core.postprocess(_tone(10.5), 22050, dsp)
    with pytest.raises(ValueError, match="mono"):
        core.postprocess(np.stack([_tone(10.5), _tone(10.5)]), SR, dsp)


def test_trim_silence_keeps_quiet_but_real_speech(dsp: core.Dsp) -> None:
    # A soft passage 30 dB under the peak is speech, not silence (threshold is 40).
    loud = _tone(4.0, amplitude=0.5)
    soft = _tone(4.0, amplitude=0.5 * 10 ** (-30 / 20))
    trimmed = core.trim_silence(np.concatenate([loud, soft]), dsp)
    assert len(trimmed) >= len(loud) + len(soft) - dsp.trim_frame_samples


def test_trim_silence_pad_is_clamped_at_the_edges(dsp: core.Dsp) -> None:
    audio = _tone(9.0)
    assert len(core.trim_silence(audio, dsp)) == len(audio)


# --- WAV encode + read-back -----------------------------------------------------


def test_wav_round_trip_reads_format_from_bytes(dsp: core.Dsp) -> None:
    pcm = core.postprocess(_speech_like(0.3, 9.6, 0.3), SR, dsp)
    data = core.encode_wav(pcm, SR)
    fmt = core.read_wav_format(data)
    assert fmt.sample_rate == SR
    assert fmt.channels == 1
    assert fmt.sample_width_bytes == 2
    assert fmt.n_samples == len(pcm)
    assert fmt.duration_seconds == pytest.approx(len(pcm) / SR)
    assert fmt.peak_dbfs == pytest.approx(dsp.peak_dbfs, abs=0.01)
    assert core.format_problems(fmt, dsp) == []
    assert np.array_equal(core.read_wav_pcm16(data), pcm)


def test_format_problems_names_every_violation(dsp: core.Dsp) -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(2 * 1000, np.int16).tobytes())
    problems = core.format_problems(core.read_wav_format(buf.getvalue()), dsp)
    text = " ".join(problems)
    for needle in ("sample_rate", "channels", "shorter", "multiple of 960"):
        assert needle in text


def test_format_problems_flags_overlong_and_silent(dsp: core.Dsp) -> None:
    fmt = core.WavFormat(SR, 1, 2, dsp.max_samples + dsp.align_samples, 10.04, None)
    text = " ".join(core.format_problems(fmt, dsp))
    assert "longer" in text and "silent" in text


def test_read_wav_pcm16_rejects_non_pcm16() -> None:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(SR)
        w.writeframes(bytes(100))
    with pytest.raises(ValueError, match="16-bit"):
        core.read_wav_pcm16(buf.getvalue())


def test_encode_wav_rejects_non_int16() -> None:
    with pytest.raises(ValueError, match="int16"):
        core.encode_wav(np.zeros(10, np.float32), SR)


def test_sha256_hex_is_lowercase_64() -> None:
    digest = core.sha256_hex(b"voxint")
    assert len(digest) == 64 and digest == digest.lower()


# --- screening maths -------------------------------------------------------------


def test_unit_rejects_degenerate_vectors() -> None:
    assert core.unit(np.zeros(4)) is None
    assert core.unit(np.array([np.nan, 1.0])) is None
    assert core.unit(np.array([np.inf, 1.0])) is None
    assert np.allclose(core.unit(np.array([3.0, 4.0])), [0.6, 0.8])  # type: ignore[arg-type]


def test_centroid_is_unit_mean_of_unit_vectors() -> None:
    c = core.centroid([np.array([2.0, 0.0]), np.array([0.0, 5.0])])
    assert np.allclose(c, [2**-0.5, 2**-0.5])


def test_centroid_skips_degenerate_members_and_fails_closed() -> None:
    c = core.centroid([np.zeros(2), np.array([0.0, 3.0])])
    assert np.allclose(c, [0.0, 1.0])
    with pytest.raises(core.ScreenError, match="no usable"):
        core.centroid([np.zeros(2)])
    with pytest.raises(core.ScreenError, match="no usable"):
        core.centroid([])
    with pytest.raises(core.ScreenError, match="cancel"):
        core.centroid([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])


def test_null_distributions() -> None:
    cents = {"s1": _unit([1, 0, 0]), "s2": _unit([0.6, 0.8, 0]), "s3": _unit([0, 0, 1])}
    pairs = core.cross_speaker_cosines(cents)
    assert pairs.tolist() == pytest.approx(sorted([0.6, 0.0, 0.0]))
    nn = core.nearest_neighbour_cosines(cents)
    assert sorted(nn.tolist()) == pytest.approx([0.0, 0.6, 0.6])
    with pytest.raises(core.ScreenError, match="two speakers"):
        core.cross_speaker_cosines({"s1": _unit([1, 0])})


def test_percentile_rank() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert core.percentile_rank(0.25, null) == pytest.approx(50.0)
    assert core.percentile_rank(0.4, null) == pytest.approx(100.0)
    assert core.percentile_rank(0.0, null) == pytest.approx(0.0)
    with pytest.raises(core.ScreenError):
        core.percentile_rank(0.1, np.array([]))


def _pool() -> dict[str, list[tuple[str, np.ndarray]]]:
    return {
        "spk-1": [("c1a", _unit([1, 0, 0, 0])), ("c1b", _unit([0.8, 0.6, 0, 0]))],
        "spk-2": [("c2a", _unit([0, 0, 1, 0])), ("c2b", _unit([0, 0, 0.6, 0.8]))],
    }


def test_screen_candidate_reports_centroid_and_clip_maxima() -> None:
    pool = _pool()
    cents = {s: core.centroid([v for _, v in clips]) for s, clips in pool.items()}
    rules = core.ScreenRules(reject_at_or_above=0.6, swap_headroom=0.05)
    # Exactly clip c1b: clip max 1.0 (reject) even though it is the clip, not a centroid.
    res = core.screen_candidate(_unit([0.8, 0.6, 0, 0]), pool, cents, rules)
    assert res.max_clip_cosine == pytest.approx(1.0)
    assert res.max_clip_speaker == "spk-1" and res.max_clip_id == "c1b"
    assert res.max_centroid_speaker == "spk-1"
    assert res.rejected
    near = core.screen_candidate(_unit([0, 1, 0, 0]), pool, cents, rules)
    assert near.max_clip_cosine == pytest.approx(0.6)
    assert near.max_clip_id == "c1b"
    far = core.screen_candidate(_unit([-1, 0, -1, 0]), pool, cents, rules)
    assert not far.rejected and not far.swap_candidate
    assert far.headroom == pytest.approx(0.6 - max(far.max_centroid_cosine, far.max_clip_cosine))


def test_screen_candidate_rejects_on_centroid_alone() -> None:
    pool = {
        # Each clip sits ~59.5 degrees off-axis (cosine ~0.507 to [1, 0, 0]); their
        # centroid is exactly [1, 0, 0].
        "spk-1": [("a", _unit([1, 1.7, 0])), ("b", _unit([1, -1.7, 0]))],
        "spk-2": [("c", _unit([0, 0, 1]))],
    }
    cents = {s: core.centroid([v for _, v in clips]) for s, clips in pool.items()}
    rules = core.ScreenRules(0.6, 0.05)
    res = core.screen_candidate(_unit([1, 0, 0]), pool, cents, rules)
    assert res.max_centroid_cosine == pytest.approx(1.0)
    assert res.max_clip_cosine < 0.6
    assert res.rejected


def test_screen_verdict_boundaries() -> None:
    rules = core.ScreenRules(0.6, 0.05)
    assert core.screen_verdict(0.6, rules).rejected  # at the floor rejects
    v = core.screen_verdict(0.5999, rules)
    assert not v.rejected and v.swap_candidate
    v = core.screen_verdict(0.54, rules)
    assert not v.rejected and not v.swap_candidate
    assert core.screen_verdict(0.2, rules).headroom == pytest.approx(0.4)


def test_screen_candidate_swap_band() -> None:
    pool = {"s": [("x", _unit([1, 0]))], "t": [("y", _unit([-1, 0]))]}
    cents = {k: core.centroid([v for _, v in c]) for k, c in pool.items()}
    rules = core.ScreenRules(0.6, 0.05)
    cos = 0.57
    cand = np.array([cos, np.sqrt(1 - cos**2)])
    res = core.screen_candidate(cand, pool, cents, rules)
    assert not res.rejected and res.swap_candidate


def test_screen_candidate_rejects_degenerate_candidate() -> None:
    pool = _pool()
    cents = {s: core.centroid([v for _, v in clips]) for s, clips in pool.items()}
    with pytest.raises(core.ScreenError, match="candidate"):
        core.screen_candidate(np.zeros(4), pool, cents, core.ScreenRules(0.6, 0.05))


def test_screen_candidate_requires_matching_pool_and_centroids() -> None:
    pool = _pool()
    with pytest.raises(core.ScreenError, match="centroid"):
        core.screen_candidate(
            _unit([1, 0, 0, 0]), pool, {"spk-1": _unit([1, 0, 0, 0])}, core.ScreenRules(0.6, 0.05)
        )


def test_pairwise_matrix_is_symmetric_with_unit_diagonal() -> None:
    m = core.pairwise_matrix({"a": _unit([1, 0]), "b": _unit([0.6, 0.8]), "c": _unit([0, 1])})
    assert m["a"]["a"] == pytest.approx(1.0)
    assert m["a"]["b"] == pytest.approx(0.6) == m["b"]["a"]
    assert m["a"]["c"] == pytest.approx(0.0)


def test_select_finalists_maximises_minimum_distance() -> None:
    vectors = {
        "a1": _unit([1, 0, 0]),
        "a2": _unit([0.7, 0.7, 0]),
        "b1": _unit([0.71, 0.7, 0.05]),  # near a2
        "b2": _unit([0, 0, 1]),
    }
    picked = core.select_finalists({"A": ["a1", "a2"], "B": ["b1", "b2"]}, vectors)
    assert picked == {"A": "a1", "B": "b2"}


def test_select_finalists_tie_breaks_deterministically() -> None:
    vectors = {"a1": _unit([1, 0]), "a2": _unit([1, 0]), "b1": _unit([0, 1])}
    assert core.select_finalists({"A": ["a2", "a1"], "B": ["b1"]}, vectors) == {
        "A": "a1",
        "B": "b1",
    }


def test_select_finalists_fails_closed() -> None:
    with pytest.raises(core.ScreenError, match="empty"):
        core.select_finalists({"A": []}, {})
    with pytest.raises(core.ScreenError, match="unknown"):
        core.select_finalists({"A": ["zz"]}, {})
    with pytest.raises(core.ScreenError, match="more than one slot"):
        core.select_finalists({"A": ["x"], "B": ["x"]}, {"x": _unit([1, 0])})


def test_chatterbox_swap_rule() -> None:
    rules = core.ChatterboxRules(target_cosine_floor=0.6, sources=15, min_sources_reached=11)
    reached_11 = [0.61] * 11 + [0.2] * 4
    assert core.reached_count(reached_11, 0.6) == 11
    assert not core.slot_needs_swap(reached_11, rules)
    assert core.slot_needs_swap([0.6] * 10 + [0.59] * 5, rules)  # 0.6 itself counts
    with pytest.raises(core.ScreenError, match="15"):
        core.slot_needs_swap([0.9] * 14, rules)


def test_rules_read_from_spec(spec: dict[str, Any]) -> None:
    assert core.ScreenRules.from_spec(spec) == core.ScreenRules(0.6, 0.05)
    assert core.ChatterboxRules.from_spec(spec) == core.ChatterboxRules(0.6, 15, 11)


def test_screen_result_serializes_to_plain_json() -> None:
    pool = _pool()
    cents = {s: core.centroid([v for _, v in clips]) for s, clips in pool.items()}
    res = core.screen_candidate(_unit([-1, 0, -1, 0]), pool, cents, core.ScreenRules(0.6, 0.05))
    blob = json.dumps(res.as_dict())
    assert "max_centroid_speaker" in blob


# --- screen tool (tools/screen_voiceclone_target_voices.py) ----------------------

from tools import screen_voiceclone_target_voices as scr  # noqa: E402


def _write_wav(path: Path, seconds: float, sr: int = 16000, amp: float = 0.3) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(seconds * sr)) / sr
    pcm = core.to_pcm16(amp * np.sin(2 * np.pi * 150 * t))
    data = core.encode_wav(pcm, sr)
    path.write_bytes(data)
    return core.sha256_hex(data)


def _gate1_manifest(root: Path) -> dict[str, Any]:
    speakers = []
    for key, corpus in (("ami-X1", "ami"), ("vc-y-spk1", "voxconverse")):
        refs = []
        for length, cond in ((10, "clean"), (10, "phone"), (3, "clean"), (3, "phone")):
            rid = f"{key}_{length}s_{cond}"
            sha = _write_wav(root / "gate1" / "refs" / f"{rid}.wav", 1.5)
            refs.append(
                {
                    "condition": cond,
                    "length_s": length,
                    "path": f"refs/{rid}.wav",
                    "ref_id": rid,
                    "sha256": sha,
                }
            )
        held = []
        for i in range(1, 5):
            hid = f"{key}_h{i}"
            sha = _write_wav(root / "gate1" / "heldout" / f"{hid}.wav", 1.5)
            psha = _write_wav(root / "gate1" / "heldout" / f"{hid}_phone.wav", 1.5, amp=0.2)
            held.append(
                {
                    "heldout_id": hid,
                    "path": f"heldout/{hid}.wav",
                    "phone_path": f"heldout/{hid}_phone.wav",
                    "phone_sha256": psha,
                    "recording": "R1",
                    "sha256": sha,
                    "start_s": 1.0,
                    "end_s": 4.0,
                }
            )
        speakers.append(
            {
                "speaker_key": key,
                "corpus": corpus,
                "source_label": key[-3:],
                "ref_window": {"recording": "R0", "start_s": 0.0, "end_s": 10.0},
                "refs": refs,
                "heldout": held,
            }
        )
    manifest = {
        "speakers": speakers,
        "sources": {
            "ami_root": "/somewhere/private",
            "ami_words_zip_sha256": "a" * 64,
            "meetings_xml_sha256": "b" * 64,
            "voxconverse_repo_sha": "c" * 40,
            "voxconverse_rttm_sha256": "d" * 64,
        },
    }
    (root / "gate1" / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _libritts(root: Path) -> Path:
    subset = root / "libritts_r" / "LibriTTS_R" / "test-clean"
    for spk in ("121", "61"):
        for utt, secs in (("0001", 3.5), ("0002", 1.0), ("0003", 3.2), ("0004", 4.0)):
            _write_wav(subset / spk / "100" / f"{spk}_100_{utt}.wav", secs, sr=24000)
    return subset


def test_gate1_clips_lists_twelve_per_speaker(tmp_path: Path) -> None:
    clips = scr.gate1_clips(_gate1_manifest(tmp_path))
    assert len(clips) == 24
    first = [c for c in clips if c.speaker == "ami-X1"]
    assert sum(c.condition == "phone" for c in first) == 6
    assert all(c.path.startswith("gate1/") for c in clips)
    assert "ami-X1_h1_phone" in {c.clip_id for c in first}


def test_libritts_clips_selects_first_long_enough_by_id(tmp_path: Path) -> None:
    subset = _libritts(tmp_path)
    clips, excluded = scr.libritts_clips(subset, tmp_path, per_speaker=2, min_seconds=3.0)
    assert [c.clip_id for c in clips] == [
        "61_100_0001",
        "61_100_0003",
        "121_100_0001",
        "121_100_0003",
    ]
    assert excluded == []
    assert clips[0].speaker == "libritts-r-61"
    assert clips[0].path.startswith("libritts_r/")
    # "Up to" N: a speaker with fewer qualifying clips keeps what it has.
    clips, _ = scr.libritts_clips(subset, tmp_path, per_speaker=9, min_seconds=3.0)
    assert len(clips) == 6
    # A speaker with no qualifying clip is reported, never silently dropped.
    _write_wav(subset / "7" / "100" / "7_100_0001.wav", 1.0, sr=24000)
    clips, excluded = scr.libritts_clips(subset, tmp_path, per_speaker=2, min_seconds=3.0)
    assert excluded == ["libritts-r-7"]
    assert all(c.speaker != "libritts-r-7" for c in clips)
    with pytest.raises(scr.PoolError, match="no LibriTTS-R"):
        scr.libritts_clips(tmp_path / "missing", tmp_path, 1, 3.0)


def test_build_pool_via_cli_round_trips(tmp_path: Path) -> None:
    _gate1_manifest(tmp_path)
    _libritts(tmp_path)
    out = tmp_path / "pool.json"
    rc = scr.main(
        [
            "build-pool",
            "--pool-root",
            str(tmp_path),
            "--libritts-archive-sha256",
            "e" * 64,
            "--per-speaker",
            "2",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["speakers"] == 4
    assert "/somewhere/private" not in out.read_text()  # internal paths never copied
    clips = scr.load_pool_manifest(out)
    assert len(clips) == 28


def test_build_pool_refuses_tampered_clip(tmp_path: Path) -> None:
    _gate1_manifest(tmp_path)
    _libritts(tmp_path)
    (tmp_path / "gate1" / "refs" / "ami-X1_3s_clean.wav").write_bytes(b"RIFF tampered")
    with pytest.raises(scr.PoolError, match="sha256"):
        scr.main(
            [
                "build-pool",
                "--pool-root",
                str(tmp_path),
                "--libritts-archive-sha256",
                "e" * 64,
                "--per-speaker",
                "2",
                "--out",
                str(tmp_path / "pool.json"),
            ]
        )


def test_pool_manifest_rejects_duplicates_and_bad_schema(tmp_path: Path) -> None:
    clip = scr.PoolClip("s", "dup", "p.wav", "0" * 64, "clean", {})
    with pytest.raises(scr.PoolError, match="unique"):
        scr.build_pool_manifest([clip, clip], [])
    p = tmp_path / "pool.json"
    p.write_text(json.dumps({"schema_version": 99, "clips": []}))
    with pytest.raises(scr.PoolError, match="schema_version"):
        scr.load_pool_manifest(p)


def _fake_normalizer(calls: list[Path]) -> scr.NormalizeFn:
    def normalize(src: Path, dest: Path) -> float:
        calls.append(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return scr.wav_seconds(dest)

    return normalize


def test_normalized_copy_verifies_sha_and_caches(tmp_path: Path) -> None:
    sha = _write_wav(tmp_path / "a.wav", 2.0)
    calls: list[Path] = []
    rel, dur = scr.normalized_copy(tmp_path, "a.wav", sha, _fake_normalizer(calls))
    assert rel == f"norm16k/{sha}.wav" and dur == pytest.approx(2.0)
    scr.normalized_copy(tmp_path, "a.wav", sha, _fake_normalizer(calls))
    assert len(calls) == 1
    with pytest.raises(scr.PoolError, match="sha256"):
        scr.normalized_copy(tmp_path, "a.wav", "f" * 64, _fake_normalizer(calls))


def test_embed_pool_records_vectors_and_skips(tmp_path: Path) -> None:
    sha_a = _write_wav(tmp_path / "a.wav", 2.0)
    sha_b = _write_wav(tmp_path / "b.wav", 2.0, amp=0.1)
    clips = [
        scr.PoolClip("s1", "a", "a.wav", sha_a, "clean", {}),
        scr.PoolClip("s1", "b", "b.wav", sha_b, "phone", {}),
    ]

    def embed(rel: str, duration: float) -> np.ndarray | str:
        return "low_snr" if sha_b in rel else np.array([1.0, 0.0])

    records = scr.embed_pool(clips, tmp_path, embed, _fake_normalizer([]))
    assert records[0]["vector"] == [1.0, 0.0]
    assert records[1]["skip_reason"] == "low_snr"
    pool, skipped = scr.pool_vectors(records)
    assert list(pool) == ["s1"] and skipped == ["b"]


def _cand_dir(tmp_path: Path, ids: list[tuple[str, str]]) -> Path:
    d = tmp_path / "cands"
    d.mkdir()
    for cid, voice in ids:
        (d / f"{cid}.json").write_text(
            json.dumps(
                {
                    "candidate_id": cid,
                    "voice_id": voice,
                    "wav": f"{cid}.wav",
                    "wav_sha256": "0" * 64,
                    "median_f0_hz": 120.0,
                }
            )
        )
    return d


def test_load_candidates_fails_closed(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(scr.PoolError, match="no candidate"):
        scr.load_candidates(empty)
    d = _cand_dir(tmp_path, [("a1", "target-a"), ("b1", "target-b")])
    assert [r["candidate_id"] for r in scr.load_candidates(d)] == ["a1", "b1"]
    (d / "zz.json").write_text(
        json.dumps({"candidate_id": "a1", "voice_id": "x", "wav": "w", "wav_sha256": "0" * 64})
    )
    with pytest.raises(scr.PoolError, match="unique"):
        scr.load_candidates(d)
    (d / "zz.json").write_text(json.dumps({"candidate_id": "q"}))
    with pytest.raises(scr.PoolError, match="lacks"):
        scr.load_candidates(d)


def _pool_records() -> list[dict[str, Any]]:
    rows = []
    for spk, vecs in {
        "spk-1": [[1, 0, 0, 0], [0.8, 0.6, 0, 0]],
        "spk-2": [[0, 0, 1, 0], [0, 0, 0.6, 0.8]],
        "spk-3": [[0, -1, 0, 0]],
    }.items():
        for i, v in enumerate(vecs):
            rows.append({"speaker": spk, "clip_id": f"{spk}-{i}", "vector": v})
    rows.append({"speaker": "spk-3", "clip_id": "spk-3-skip", "skip_reason": "too_short"})
    return rows


def _screen_fixture() -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    cands = [
        {"candidate_id": "a1", "voice_id": "target-a", "wav_sha256": "1" * 64, "median_f0_hz": 110},
        {"candidate_id": "a2", "voice_id": "target-a", "wav_sha256": "2" * 64, "median_f0_hz": 115},
        {"candidate_id": "b1", "voice_id": "target-b", "wav_sha256": "3" * 64, "median_f0_hz": 210},
    ]
    vectors = {
        "a1": _unit([0.8, 0.6, 0, 0]),  # equals a pool clip: rejected
        "a2": _unit([-1, 0, -1, 0]),
        "b1": _unit([-1, 0, 0.5, -1]),
    }
    return cands, vectors


def test_screen_report_fixture_with_known_cosines() -> None:
    cands, vectors = _screen_fixture()
    rules = core.ScreenRules(0.6, 0.05)
    report = scr.screen_report(cands, vectors, _pool_records(), rules, {"embedder": {"x": 1}})
    a1 = report["candidates"]["a1"]
    assert a1["rejected"] and a1["max_clip_id"] == "spk-1-1"
    assert a1["max_clip_cosine"] == pytest.approx(1.0)
    a2 = report["candidates"]["a2"]
    assert not a2["rejected"]
    assert a2["max_clip_cosine"] == pytest.approx(
        float(_unit([-1, 0, -1, 0]) @ _unit([0, -1, 0, 0]))
    )
    assert 0.0 <= a2["null_percentile_nearest_neighbour"] <= 100.0
    assert report["pool_summary"] == {
        "speakers": 3,
        "clips_embedded": 5,
        "clips_skipped": ["spk-3-skip"],
    }
    assert set(report["pairwise_passing"]) == {"a2", "b1"}  # rejected a1 excluded
    assert report["null"]["nearest_neighbour"]["n"] == 3
    assert report["null"]["all_pairs"]["n"] == 3
    assert "not proof" in report["limits"]
    assert report["embedder"] == {"x": 1}
    json.dumps(report)


def test_screen_report_needs_two_speakers() -> None:
    cands, vectors = _screen_fixture()
    rows = [r for r in _pool_records() if r["speaker"] == "spk-1"]
    with pytest.raises(scr.PoolError, match="two embedded speakers"):
        scr.screen_report(cands, vectors, rows, core.ScreenRules(0.6, 0.05), {})


def test_select_into_report() -> None:
    cands, vectors = _screen_fixture()
    report = scr.screen_report(cands, vectors, _pool_records(), core.ScreenRules(0.6, 0.05), {})
    out = scr.select_into_report(report, {"target-a": ["a2"], "target-b": ["b1"]}, vectors)
    sel = out["selection"]
    assert sel["finalists"] == {"target-a": "a2", "target-b": "b1"}
    assert sel["finalist_median_f0_hz"] == {"target-a": 115, "target-b": 210}
    assert set(sel["finalist_pairwise"]) == {"a2", "b1"}
    assert "selection" not in report  # input report not mutated
    with pytest.raises(scr.PoolError, match="rejected"):
        scr.select_into_report(report, {"target-a": ["a1"]}, vectors)
    with pytest.raises(scr.PoolError, match="not screened"):
        scr.select_into_report(report, {"target-a": ["zz"]}, vectors)
    with pytest.raises(scr.PoolError, match="belongs to"):
        scr.select_into_report(report, {"target-b": ["a2"]}, vectors)


def test_select_cli_updates_report_in_place(tmp_path: Path) -> None:
    cands, vectors = _screen_fixture()
    report = scr.screen_report(cands, vectors, _pool_records(), core.ScreenRules(0.6, 0.05), {})
    rpath = tmp_path / "screening_report.json"
    rpath.write_text(json.dumps(report))
    (tmp_path / "screening_report.vectors.json").write_text(
        json.dumps({k: v.tolist() for k, v in vectors.items()})
    )
    short = tmp_path / "shortlist.json"
    short.write_text(json.dumps({"target-a": ["a2"], "target-b": ["b1"]}))
    assert scr.main(["select", "--report", str(rpath), "--shortlist", str(short)]) == 0
    assert json.loads(rpath.read_text())["selection"]["finalists"]["target-a"] == "a2"


def _patch_live(
    monkeypatch: pytest.MonkeyPatch, health: dict[str, Any], vectors: dict[str, np.ndarray]
) -> None:
    monkeypatch.setattr(scr, "titanet_health", lambda _url: dict(health))
    monkeypatch.setattr(
        scr, "onnx_graph_identity", lambda _g: {"release": "titanet-onnx-v1", "onnx_sha256": "0"}
    )

    def embedder(_url: str) -> scr.EmbedFn:
        def embed(rel: str, duration: float) -> np.ndarray | str:
            for key, vec in vectors.items():
                if key in rel:
                    return vec
            raise AssertionError(rel)

        return embed

    monkeypatch.setattr(scr, "http_embedder", embedder)
    monkeypatch.setattr(scr, "ffmpeg_normalizer", lambda: _fake_normalizer([]))
    monkeypatch.setattr(scr, "normalizer_provenance", lambda: {"resampler": "fake"})


def test_onnx_graph_identity(tmp_path: Path) -> None:
    graph = tmp_path / "g.onnx"
    graph.write_bytes(b"graph")
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps({"onnx_sha256": core.sha256_hex(b"graph")}))
    assert scr.onnx_graph_identity(graph, prov) == {
        "release": "titanet-onnx-v1",
        "onnx_sha256": core.sha256_hex(b"graph"),
    }
    graph.write_bytes(b"other")
    with pytest.raises(scr.PoolError, match="titanet-onnx-v1"):
        scr.onnx_graph_identity(graph, prov)


def test_embed_pool_and_screen_cli_with_fake_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]
) -> None:
    root = tmp_path / "pool"
    shas = {}
    for name in ("p1", "p2", "p3", "c1"):
        sub = "cands" if name.startswith("c") else "clips"
        shas[name] = _write_wav(root / sub / f"{name}.wav", 2.0, amp=0.1 + 0.05 * len(shas))
    manifest = root / "pool.json"
    clips = [
        scr.PoolClip("s1", "p1", "clips/p1.wav", shas["p1"], "clean", {}),
        scr.PoolClip("s2", "p2", "clips/p2.wav", shas["p2"], "clean", {}),
        scr.PoolClip("s3", "p3", "clips/p3.wav", shas["p3"], "clean", {}),
    ]
    manifest.write_text(json.dumps(scr.build_pool_manifest(clips, [])))
    vectors = {
        shas["p1"]: _unit([1, 0, 0]),
        shas["p2"]: _unit([0, 1, 0]),
        shas["p3"]: _unit([0, 0, 1]),
        shas["c1"]: _unit([-1, -1, -1]),
    }
    health = {"engine": "onnxruntime", "embedding_space": "titanet-large-v2"}
    _patch_live(monkeypatch, health, vectors)
    emb = tmp_path / "pool_embeddings.json"
    scr.main(
        [
            "embed-pool",
            "--manifest",
            str(manifest),
            "--pool-root",
            str(root),
            "--onnx-graph",
            "g.onnx",
            "--out",
            str(emb),
        ]
    )
    (root / "cands" / "c1.json").write_text(
        json.dumps(
            {
                "candidate_id": "c1",
                "voice_id": "target-a",
                "wav": "c1.wav",
                "wav_sha256": shas["c1"],
                "median_f0_hz": 100.0,
            }
        )
    )
    out = tmp_path / "screening_report.json"
    argv = [
        "screen",
        "--spec",
        str(SPEC_PATH),
        "--manifest",
        str(manifest),
        "--pool-embeddings",
        str(emb),
        "--pool-root",
        str(root),
        "--candidates-dir",
        str(root / "cands"),
        "--onnx-graph",
        "g.onnx",
        "--out",
        str(out),
    ]
    scr.main(argv)
    report = json.loads(out.read_text())
    assert not report["candidates"]["c1"]["rejected"]
    assert report["pool_manifest_sha256"] == scr.file_sha256(manifest)
    assert (tmp_path / "screening_report.vectors.json").exists()
    # A different titanet than the one that embedded the pool is refused.
    _patch_live(monkeypatch, {**health, "engine_version": "other"}, vectors)
    with pytest.raises(scr.PoolError, match="titanet changed"):
        scr.main(argv)
