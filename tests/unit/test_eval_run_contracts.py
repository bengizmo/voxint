"""Contract tests for the eval-quality ``run`` step's pure core (issue #97).

``tools/eval_run.py`` is commit 1 of the ``run`` subcommand: every schema and
invariant the live driver (commit 2) will stand on, testable without a worker.
These tests freeze that contract — the AMI WER hypothesis crop (which must match
the frozen reference byte-for-byte), subset/path resolution, the
pipeline-environment + cohort identities that let ``report`` reject a
mislabelled zero-change set, and the write-ahead resume decision — so a later
edit cannot silently move a score or duplicate a run.

``eval_run`` imports only the bakeoff ``_us`` helper (no pyannote/jiwer), so this
runs in the default dev lane rather than the parity lane.
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = REPO / "tools" / "eval_run.py"
    spec = importlib.util.spec_from_file_location("eval_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


er = _load_tool()
CI = er.CohortInput  # terse alias; these tests build many cohort-input records


def _action1(journal: dict, rec: str, *, resume: bool, retry_failed: bool) -> str:
    """The single resume action for one recording (tests select one id at a time)."""
    return er.plan_resume(journal, [rec], resume=resume, retry_failed=retry_failed)[0].action


def _env() -> dict:
    """A minimal VALID pipeline_environment identity."""
    return {
        "schema_version": 1,
        "code": {"git_sha": "abc", "image_digest": "sha256:d"},
        "model_weights": {
            "whisper_ct2_dir_sha256": "w",
            "pyannote_pipeline_sha256": "p",
            "titanet_sha256": "t",
        },
        "gpu": {"name": "RTX 3060", "driver": "550", "cuda": "12.4"},
        "runtime": {"ctranslate2": "4.0", "torch": "2.3", "pyannote_audio": "3.1.1"},
        "decode": {"beam_size": 5, "batch_size": 4, "word_timestamps": True},
        "flags": {"tf32": False, "deterministic": True},
    }


# --------------------------------------------------------------------------- #
# 1. AMI WER hypothesis text
# --------------------------------------------------------------------------- #
class TestAmiHypothesisText:
    def test_keeps_word_whose_midpoint_is_in_uem_half_open(self) -> None:
        # region [1.0s, 3.0s) in microseconds
        uem = [(1_000_000, 3_000_000)]
        segs = [
            [
                {"start": 0.0, "end": 0.5, "word": " before"},  # mid 0.25s out
                {"start": 1.0, "end": 1.4, "word": " Hello"},  # mid 1.2s in
            ],
            [
                {"start": 2.9, "end": 3.2, "word": " edge"},  # mid 3.05s out (upper open)
                {"start": 5.0, "end": 5.2, "word": " after"},  # out
            ],
        ]
        assert er.ami_hypothesis_text(segs, uem) == "Hello"

    def test_lower_edge_is_inclusive_upper_edge_is_exclusive(self) -> None:
        uem = [(1_000_000, 2_000_000)]
        # midpoint exactly at lower bound 1.0s -> kept; exactly at upper 2.0s -> dropped
        segs = [
            [
                {"start": 0.8, "end": 1.2, "word": " low"},  # mid exactly 1.0s
                {"start": 1.8, "end": 2.2, "word": " high"},  # mid exactly 2.0s
            ]
        ]
        assert er.ami_hypothesis_text(segs, uem) == "low"

    def test_preserves_asr_order_and_does_not_timestamp_sort(self) -> None:
        # Deliberately non-monotonic word timings: a correct builder keeps the
        # provider order, NOT a re-sorted-by-time order (which would move WER).
        uem = [(0, 10_000_000)]
        segs = [
            [
                {"start": 5.0, "end": 5.2, "word": " second"},
                {"start": 1.0, "end": 1.2, "word": " first"},
            ]
        ]
        assert er.ami_hypothesis_text(segs, uem) == "second first"

    def test_strips_leading_space_and_skips_empty_tokens(self) -> None:
        uem = [(0, 10_000_000)]
        segs = [
            [{"start": 0.0, "end": 0.1, "word": "   "}, {"start": 0.2, "end": 0.3, "word": " ok"}]
        ]
        assert er.ami_hypothesis_text(segs, uem) == "ok"

    def test_empty_result_is_legitimate(self) -> None:
        uem = [(100_000_000, 200_000_000)]
        segs = [[{"start": 0.0, "end": 0.1, "word": " x"}]]
        assert er.ami_hypothesis_text(segs, uem) == ""

    def test_requires_a_uem(self) -> None:
        with pytest.raises(er.RunError):
            er.ami_hypothesis_text([[{"start": 0.0, "end": 0.1, "word": " x"}]], [])

    def test_rejects_malformed_word(self) -> None:
        uem = [(0, 10_000_000)]
        with pytest.raises(er.RunError):
            er.ami_hypothesis_text([[{"start": 0.0, "word": " x"}]], uem)  # no 'end'
        with pytest.raises(er.RunError):
            er.ami_hypothesis_text([[{"start": "nan", "end": 1.0, "word": " x"}]], uem)
        with pytest.raises(er.RunError):
            er.ami_hypothesis_text([[{"start": 2.0, "end": 1.0, "word": " x"}]], uem)  # end<start

    def test_midpoint_uses_the_same_us_helper_as_the_reference(self) -> None:
        # A word centred on a fractional-microsecond boundary must land the same
        # side as the Decimal-based reference. 1.0000005s -> _us floors to 1000000.
        import prepare_bakeoff_corpus as bake

        assert bake._us(1.0000005) == 1_000_000
        uem = [(1_000_000, 2_000_000)]
        segs = [[{"start": 1.0000005, "end": 1.0000005, "word": " b"}]]
        assert er.ami_hypothesis_text(segs, uem) == "b"


# --------------------------------------------------------------------------- #
# 2. Subset validation + selection
# --------------------------------------------------------------------------- #
def _item(**kw) -> dict:
    base = {"corpus": "ami", "split": "test", "id": "EN2002c", "num_speakers": 4, "extent_s": 100.0}
    base.update(kw)
    return base


class TestSubset:
    def test_valid_item_round_trips(self) -> None:
        items = er.load_subset([_item()], "ami")
        assert items[0].recording_id == "EN2002c"
        assert items[0].num_speakers == 4

    def test_filters_to_the_requested_corpus(self) -> None:
        entries = [_item(), _item(corpus="voxconverse", id="uicid", num_speakers=2)]
        assert [i.recording_id for i in er.load_subset(entries, "ami")] == ["EN2002c"]
        assert [i.recording_id for i in er.load_subset(entries, "voxconverse")] == ["uicid"]

    def test_rejects_duplicate_id_within_corpus(self) -> None:
        with pytest.raises(er.RunError):
            er.load_subset([_item(), _item()], "ami")

    def test_rejects_bad_corpus_and_split(self) -> None:
        with pytest.raises(er.RunError):
            er.load_subset([_item(corpus="nope")], "ami")
        with pytest.raises(er.RunError):
            er.load_subset([_item(split="train")], "ami")

    def test_rejects_bad_speaker_count_and_extent(self) -> None:
        bads = ({"num_speakers": 0}, {"num_speakers": True}, {"extent_s": 0}, {"extent_s": -1.0})
        for bad in bads:
            with pytest.raises(er.RunError):
                er.load_subset([_item(**bad)], "ami")

    def test_rejects_unsafe_recording_id(self) -> None:
        for bad in ("../etc", ".hidden", "a/b", "with space", ""):
            with pytest.raises(er.RunError):
                er.load_subset([_item(id=bad)], "ami")

    def test_errors_when_no_items_for_corpus(self) -> None:
        with pytest.raises(er.RunError):
            er.load_subset([_item()], "voxconverse")

    def test_select_only_filters_and_rejects_unknown(self) -> None:
        items = er.load_subset([_item(), _item(id="IS1009a")], "ami")
        assert [i.recording_id for i in er.select_only(items, ["IS1009a"])] == ["IS1009a"]
        assert er.select_only(items, None) == items
        with pytest.raises(er.RunError):
            er.select_only(items, ["NOPE"])

    def test_select_only_rejects_duplicate_ids(self) -> None:
        # ``--only A,A`` must fail, not expand to two submits of one recording.
        items = er.load_subset([_item(), _item(id="IS1009a")], "ami")
        with pytest.raises(er.RunError, match="duplicate"):
            er.select_only(items, ["EN2002c", "EN2002c"])


# --------------------------------------------------------------------------- #
# 3. Path resolution
# --------------------------------------------------------------------------- #
def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def _write_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nframes = round(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * nframes)
    return path


class TestPathResolution:
    def _ami_layout(self, root: Path, rid: str = "EN2002c", split: str = "test") -> None:
        _write_wav(root / "ami" / "audio" / f"{rid}.Mix-Headset.wav", 1.0)
        setup = root / "ami" / "AMI-diarization-setup-main"
        _touch(setup / "only_words" / "rttms" / split / f"{rid}.rttm")
        _touch(setup / "uems" / split / f"{rid}.uem")
        _touch(root / "ami" / "wer_reference" / f"{rid}.words.txt")

    def test_ami_resolves_all_four_roles(self, tmp_path: Path) -> None:
        self._ami_layout(tmp_path)
        item = er.load_subset([_item()], "ami")[0]
        r = er.resolve_item(tmp_path, item)
        assert r.audio.name == "EN2002c.Mix-Headset.wav"
        assert r.uem is not None and r.wer_reference is not None

    def test_ami_missing_file_errors(self, tmp_path: Path) -> None:
        self._ami_layout(tmp_path)
        (tmp_path / "ami" / "wer_reference" / "EN2002c.words.txt").unlink()
        with pytest.raises(er.RunError):
            er.resolve_item(tmp_path, er.load_subset([_item()], "ami")[0])

    def test_voxconverse_test_and_dev_nesting(self, tmp_path: Path) -> None:
        vc = tmp_path / "voxconverse"
        _write_wav(vc / "audio_test" / "voxconverse_test_wav" / "uicid.wav", 1.0)
        _touch(vc / "voxconverse-master" / "test" / "uicid.rttm")
        item = er.load_subset(
            [_item(corpus="voxconverse", split="test", id="uicid", num_speakers=2)], "voxconverse"
        )[0]
        r = er.resolve_item(tmp_path, item)
        assert r.uem is None and r.wer_reference is None
        assert "audio_test" in r.audio.parts

    def test_voxconverse_ambiguous_audio_is_rejected(self, tmp_path: Path) -> None:
        vc = tmp_path / "voxconverse"
        _write_wav(vc / "audio_test" / "voxconverse_test_wav" / "uicid.wav", 1.0)
        _write_wav(vc / "audio_dev" / "audio" / "uicid.wav", 1.0)  # same id under both nestings
        _touch(vc / "voxconverse-master" / "test" / "uicid.rttm")
        item = er.load_subset(
            [_item(corpus="voxconverse", split="test", id="uicid", num_speakers=2)], "voxconverse"
        )[0]
        with pytest.raises(er.RunError, match="ambiguous"):
            er.resolve_item(tmp_path, item)

    def test_synology_metadata_dir_is_not_an_audio_candidate(self, tmp_path: Path) -> None:
        # A stray @eaDir copy of the audio must be ignored, not counted as a
        # second candidate that would trip the ambiguity guard.
        vc = tmp_path / "voxconverse"
        _write_wav(vc / "audio_test" / "voxconverse_test_wav" / "uicid.wav", 1.0)
        candidates = [
            vc / "audio_test" / "voxconverse_test_wav" / "uicid.wav",
            vc / "audio_test" / "voxconverse_test_wav" / "@eaDir" / "uicid.wav",
        ]
        _write_wav(candidates[1], 1.0)
        assert er._unambiguous_audio(candidates, "uicid") == candidates[0]

    def test_require_within_blocks_traversal(self, tmp_path: Path) -> None:
        with pytest.raises(er.RunError):
            er._require_within(tmp_path / "root", tmp_path / "elsewhere" / "x", "audio")


# --------------------------------------------------------------------------- #
# 4. WAV preflight
# --------------------------------------------------------------------------- #
class TestWavPreflight:
    def test_measures_true_duration(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "a.wav", 2.5)
        assert abs(er.measure_wav_seconds(wav) - 2.5) < 1e-6

    def test_detects_truncation_past_the_header_nframes(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "a.wav", 2.0)
        raw = wav.read_bytes()
        wav.write_bytes(raw[: len(raw) - 16000])  # chop 0.5s of data, header still claims 2.0s
        with pytest.raises(er.RunError, match="truncated"):
            er.measure_wav_seconds(wav)

    def test_rejects_non_wav(self, tmp_path: Path) -> None:
        bad = tmp_path / "b.wav"
        bad.write_bytes(b"not a wav")
        with pytest.raises(er.RunError):
            er.measure_wav_seconds(bad)

    def test_check_duration_flags_mismatch_and_out_of_bounds(self) -> None:
        assert er.check_duration(100.0, 100.0, 99.0, 99.5, tol_s=1.0) == []
        problems = er.check_duration(
            measured_s=100.0,
            extent_s=110.0,
            reference_max_end_s=105.0,
            uem_max_end_s=106.0,
            tol_s=1.0,
        )
        assert len(problems) == 3  # extent mismatch + reference past end + uem past end
        assert er.check_duration(100.0, 100.0, 99.0, None, tol_s=1.0) == []


# --------------------------------------------------------------------------- #
# 5. pipeline_environment identity
# --------------------------------------------------------------------------- #
class TestPipelineEnvironment:
    def test_valid_env_hashes_stably_across_key_order(self) -> None:
        env = _env()
        reordered = {k: env[k] for k in reversed(list(env))}
        assert er.pipeline_environment_hash(env) == er.pipeline_environment_hash(reordered)

    def test_rejects_missing_group(self) -> None:
        env = _env()
        del env["gpu"]
        with pytest.raises(er.RunError):
            er.validate_pipeline_environment(env)

    def test_rejects_missing_key_null_value_and_extra_key(self) -> None:
        env = _env()
        del env["decode"]["beam_size"]
        with pytest.raises(er.RunError):
            er.validate_pipeline_environment(env)
        env = _env()
        env["model_weights"]["titanet_sha256"] = None
        with pytest.raises(er.RunError):
            er.validate_pipeline_environment(env)
        env = _env()
        env["gpu"]["extra"] = "x"
        with pytest.raises(er.RunError):
            er.validate_pipeline_environment(env)

    def test_rejects_wrong_schema_version(self) -> None:
        env = _env()
        env["schema_version"] = 2
        with pytest.raises(er.RunError):
            er.validate_pipeline_environment(env)

    def test_changing_any_identity_field_changes_the_hash(self) -> None:
        base = er.pipeline_environment_hash(_env())
        moved = _env()
        moved["model_weights"]["whisper_ct2_dir_sha256"] = "different"
        assert er.pipeline_environment_hash(moved) != base

    def test_rejects_wrongly_typed_scalars(self) -> None:
        # A string where a bool belongs, a bool where an int belongs, a dict
        # where a scalar digest belongs, an int < 1, and an empty string are all
        # rejected — the hash must bind a real identity, not a mistyped one.
        cases = [
            ("decode", "word_timestamps", "true"),  # str for bool
            ("flags", "tf32", 1),  # int for bool
            ("decode", "beam_size", True),  # bool for int
            ("decode", "batch_size", 0),  # int < 1
            ("code", "image_digest", {"nested": "x"}),  # dict for scalar
            ("code", "git_sha", ""),  # empty string
            ("gpu", "name", 123),  # int for str
        ]
        for group, key, bad in cases:
            env = _env()
            env[group][key] = bad
            with pytest.raises(er.RunError):
                er.validate_pipeline_environment(env)

    def test_accepts_the_declared_types(self) -> None:
        # Sanity: the valid _env() passes with its real bool/int/str fields.
        env = er.validate_pipeline_environment(_env())
        assert env["decode"]["beam_size"] == 5
        assert env["flags"]["deterministic"] is True


# --------------------------------------------------------------------------- #
# 6. Cohort descriptor + hash
# --------------------------------------------------------------------------- #
class TestCohort:
    def _cohort(self, inputs, split_by_id=None) -> dict:
        return er.cohort_descriptor(
            "ami",
            split_by_id or {"EN2002c": "test"},
            inputs,
            er.pipeline_environment_hash(_env()),
            {"strict": {"collar": 0.0, "skip_overlap": False}},
        )

    def test_descriptor_is_order_independent(self) -> None:
        a = [CI("EN2002c", "audio", 10, "aa"), CI("EN2002c", "uem", 5, "uu")]
        h = er.cohort_sha256
        assert h(self._cohort(a)) == h(self._cohort(list(reversed(a))))

    def test_changed_audio_bytes_change_the_cohort(self) -> None:
        a = [CI("EN2002c", "audio", 10, "aa")]
        b = [CI("EN2002c", "audio", 10, "bb")]
        assert er.cohort_sha256(self._cohort(a)) != er.cohort_sha256(self._cohort(b))

    def test_explicit_null_role_distinguishes_corpora(self) -> None:
        with_uem = [CI("EN2002c", "audio", 10, "aa"), CI("EN2002c", "uem", 5, "uu")]
        null_uem = [CI("EN2002c", "audio", 10, "aa"), CI("EN2002c", "uem", None, None)]
        assert er.cohort_sha256(self._cohort(with_uem)) != er.cohort_sha256(self._cohort(null_uem))

    def test_rejects_wrong_schema_version(self) -> None:
        d = self._cohort([CI("EN2002c", "audio", 10, "aa")])
        d["schema_version"] = 99
        with pytest.raises(er.RunError):
            er.cohort_sha256(d)


# --------------------------------------------------------------------------- #
# 7. Journal + resume decision
# --------------------------------------------------------------------------- #
class TestJournalResume:
    def _cohort_hash(self) -> str:
        return er.cohort_sha256(
            er.cohort_descriptor(
                "ami",
                {"EN2002c": "test"},
                [er.CohortInput("EN2002c", "audio", 1, "a")],
                er.pipeline_environment_hash(_env()),
                {"strict": {"collar": 0.0}},
            )
        )

    def test_new_journal_binds_corpus_cohort_and_env(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        assert j["corpus"] == "ami"
        er.validate_journal(j, corpus="ami", cohort_hash=self._cohort_hash())

    def test_validate_journal_rejects_corpus_or_cohort_mismatch(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        with pytest.raises(er.RunError):
            er.validate_journal(j, corpus="voxconverse", cohort_hash=self._cohort_hash())
        with pytest.raises(er.RunError):
            er.validate_journal(j, corpus="ami", cohort_hash="different")

    def test_fresh_item_submits(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        [d] = er.plan_resume(j, ["EN2002c"], resume=False, retry_failed=False)
        assert d.action == er.ACTION_SUBMIT

    def test_completed_with_artifacts_skips_without_them_stops(self) -> None:
        # AMI requires BOTH the hypothesis RTTM sha and the WER text sha before a
        # completed item is safe to skip.
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["done"] = {
            "status": "completed",
            "artifacts": {"hypothesis_rttm_sha256": "h", "wer_text_sha256": "w"},
        }
        j["items"]["empty"] = {"status": "completed", "artifacts": {}}
        decisions = er.plan_resume(j, ["done", "empty"], resume=True, retry_failed=False)
        by_id = {d.recording_id: d.action for d in decisions}
        assert by_id == {"done": er.ACTION_SKIP_DONE, "empty": er.ACTION_STOP}

    def test_ami_completed_missing_wer_sha_stops(self) -> None:
        # RTTM present but WER text sha absent: an AMI run is NOT done, so a
        # resume must STOP rather than skip it into a fake zero-change pass.
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["rttm_only"] = {
            "status": "completed",
            "artifacts": {"hypothesis_rttm_sha256": "h"},
        }
        assert _action1(j, "rttm_only", resume=True, retry_failed=False) == er.ACTION_STOP

    def test_voxconverse_completed_needs_only_rttm(self) -> None:
        # VoxConverse has no WER reference, so the RTTM sha alone is complete.
        j = er.new_journal("voxconverse", self._cohort_hash(), _env())
        j["items"]["v"] = {"status": "completed", "artifacts": {"hypothesis_rttm_sha256": "h"}}
        assert _action1(j, "v", resume=True, retry_failed=False) == er.ACTION_SKIP_DONE

    def test_failure_retried_only_with_flag(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["f"] = {"status": "failed"}
        assert _action1(j, "f", resume=True, retry_failed=False) == er.ACTION_STOP
        assert _action1(j, "f", resume=True, retry_failed=True) == er.ACTION_RETRY

    def test_resumable_state_polls_only_with_uuid_and_resume(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["r"] = {"status": "running", "run_uuid": "u"}
        assert _action1(j, "r", resume=True, retry_failed=False) == er.ACTION_POLL
        assert _action1(j, "r", resume=False, retry_failed=False) == er.ACTION_STOP
        j["items"]["r"] = {"status": "running"}  # no uuid
        assert _action1(j, "r", resume=True, retry_failed=False) == er.ACTION_SUBMIT

    def test_awaiting_adjudication_is_resumable_not_a_failure(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["a"] = {"status": "awaiting_adjudication", "run_uuid": "u"}
        assert _action1(j, "a", resume=True, retry_failed=False) == er.ACTION_POLL

    def test_unknown_submission_reconciles_by_resubmit(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["u"] = {"status": er.UNKNOWN_SUBMISSION}
        assert _action1(j, "u", resume=True, retry_failed=False) == er.ACTION_SUBMIT

    def test_queued_status_is_resumable(self) -> None:
        # ``queued`` is the real first DB status (there is no ``pending``); a
        # queued run with a uuid must POLL under --resume, not fall through.
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["items"]["q"] = {"status": "queued", "run_uuid": "u"}
        assert _action1(j, "q", resume=True, retry_failed=False) == er.ACTION_POLL

    def test_unknown_corpus_journal_is_rejected(self) -> None:
        j = er.new_journal("ami", self._cohort_hash(), _env())
        j["corpus"] = "mystery"
        with pytest.raises(er.RunError, match="required-artifact"):
            er.plan_resume(j, ["x"], resume=True, retry_failed=False)


# --------------------------------------------------------------------------- #
# 8. DB status mapping (fail-closed)
# --------------------------------------------------------------------------- #
class TestMapDbStatus:
    def test_maps_every_real_run_status_to_itself(self) -> None:
        for status in ("queued", "running", "awaiting_adjudication", "completed",
                       "failed", "cancelled"):
            assert er.map_db_status(status) == status

    def test_rejects_journal_only_and_unknown_states(self) -> None:
        # The write-ahead journal-only states are never emitted by the DB, and a
        # misspelled/None status must fail closed rather than be copied verbatim.
        for bad in ("submitting", "submission_unknown", "pending", "submitted", "", None):
            with pytest.raises(er.RunError):
                er.map_db_status(bad)


# --------------------------------------------------------------------------- #
# 9. Journal durability (atomic write + exclusive out-dir lock)
# --------------------------------------------------------------------------- #
class TestJournalDurability:
    def test_write_json_atomic_round_trips_and_leaves_no_temp(self, tmp_path: Path) -> None:
        import json as _json

        target = tmp_path / "sub" / "journal.json"
        payload = {"b": 2, "a": 1, "items": {"x": {"status": "completed"}}}
        er.write_json_atomic(target, payload)
        assert _json.loads(target.read_text(encoding="utf-8")) == payload
        # canonical bytes: sorted keys, compact
        assert target.read_text(encoding="utf-8").startswith('{"a":1')
        leftovers = [p.name for p in target.parent.iterdir() if p.name != "journal.json"]
        assert leftovers == []

    def test_write_json_atomic_overwrites_in_place(self, tmp_path: Path) -> None:
        target = tmp_path / "j.json"
        er.write_json_atomic(target, {"v": 1})
        er.write_json_atomic(target, {"v": 2})
        import json as _json

        assert _json.loads(target.read_text(encoding="utf-8")) == {"v": 2}

    def test_out_dir_lock_is_exclusive(self, tmp_path: Path) -> None:
        # Deliberately nested: hold the first lock, THEN prove a second
        # acquisition of the same out-dir fails closed.
        with er.out_dir_lock(tmp_path), pytest.raises(er.RunError, match="lock"):  # noqa: SIM117
            with er.out_dir_lock(tmp_path):
                pass  # pragma: no cover

    def test_out_dir_lock_releases_on_exit(self, tmp_path: Path) -> None:
        with er.out_dir_lock(tmp_path):
            pass
        # A second acquisition after the first releases must succeed.
        with er.out_dir_lock(tmp_path):
            pass
