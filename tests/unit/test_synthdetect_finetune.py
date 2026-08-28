from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="synthdetect fine-tuning tests require torch")
import torch.nn as nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from synthdetect_corpus import ClipEntry, GeneratorProvenance  # noqa: E402
from synthdetect_finetune import (  # noqa: E402
    FEATURE_DIM,
    LABEL_BONAFIDE,
    LABEL_SPOOF,
    MODEL_WIDTH_SAMPLES,
    BalancedBatchSampler,
    FeatureAASIST,
    FeatureDataset,
    SpeakerSplit,
    _freeze_batch_norm,
    _repeat_pad,
    score_clips,
)


def _clip(
    clip_id: str,
    *,
    speaker_id: str,
    source: str,
    label: str,
    generator_name: str | None = None,
) -> ClipEntry:
    generator = None
    if generator_name is not None:
        generator = GeneratorProvenance(
            name=generator_name,
            version="unit",
            checkpoint_sha=None,
            voice="unit",
            seed="0",
            text_source="unit",
        )
    return ClipEntry(
        clip_id=clip_id,
        rel_path=f"clips/{clip_id}.wav",
        sha256="0" * 64,
        duration_s=4.1,
        label=label,
        language="en",
        license_spdx="CC0-1.0",
        stratum="unit",
        source=source,
        speaker_id=speaker_id,
        split="calibration",
        generator=generator,
        degradation=None,
        parent_clip_id=None,
        acquire=None,
    )


@pytest.fixture
def calibration_clips() -> list[ClipEntry]:
    """Calibration clips covering every group, with two clips per speaker."""
    groups = (
        ("voxconverse", "bona_fide", None),
        ("librispeech", "bona_fide", None),
        ("synthetic-chatterbox", "spoof", "Chatterbox TTS"),
        ("synthetic-piper", "spoof", "Piper TTS"),
    )
    clips: list[ClipEntry] = []
    for group_index, (source, label, generator) in enumerate(groups):
        for speaker_index in range(4):
            speaker = f"speaker-{group_index}-{speaker_index}"
            for take in range(2):
                clip_id = f"clip-{group_index}-{speaker_index}-{take}"
                clips.append(
                    _clip(
                        clip_id,
                        speaker_id=speaker,
                        source=source,
                        label=label,
                        generator_name=generator,
                    )
                )
    return clips


@pytest.fixture
def feature_dataset(tmp_path: Path, calibration_clips: list[ClipEntry]) -> FeatureDataset:
    return FeatureDataset(
        [clip.clip_id for clip in calibration_clips], tmp_path, calibration_clips
    )


def test_repeat_pad_exact_width_returns_same_samples() -> None:
    samples = np.arange(MODEL_WIDTH_SAMPLES, dtype=np.int16)
    np.testing.assert_array_equal(_repeat_pad(samples), samples)


def test_repeat_pad_tiles_short_input_to_width() -> None:
    samples = np.array([1, 2, 3], dtype=np.int16)
    result = _repeat_pad(samples, width=8)
    np.testing.assert_array_equal(result, [1, 2, 3, 1, 2, 3, 1, 2])
    assert result.flags.c_contiguous


def test_repeat_pad_truncates_long_input() -> None:
    samples = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(_repeat_pad(samples, width=4), samples[:4])


def test_repeat_pad_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty clip"):
        _repeat_pad(np.array([], dtype=np.int16))


def test_speaker_split_is_disjoint_source_aware_and_deterministic(
    calibration_clips: list[ClipEntry],
) -> None:
    splitter = SpeakerSplit(dev_fraction=0.2, seed="unit-split")
    train, dev = splitter.partition(calibration_clips)
    train_again, dev_again = splitter.partition(calibration_clips)

    assert train and dev
    assert (train, dev) == (train_again, dev_again)
    assert set(train).isdisjoint(dev)

    by_id = {clip.clip_id: clip for clip in calibration_clips}
    train_speakers = {by_id[clip_id].speaker_id for clip_id in train}
    dev_speakers = {by_id[clip_id].speaker_id for clip_id in dev}
    assert train_speakers.isdisjoint(dev_speakers)
    assert {by_id[clip_id].source for clip_id in dev} == {
        clip.source for clip in calibration_clips
    }


def _batch_groups(dataset: FeatureDataset, batch: list[int]) -> dict[str, int]:
    counts = {"spoof_chatterbox": 0, "spoof_piper": 0, "bf_vox": 0, "bf_other": 0}
    for index in batch:
        clip = dataset.manifest_clips[dataset.clip_ids[index]]
        if clip.label == "bona_fide":
            group = "bf_vox" if "voxconverse" in clip.source.lower() else "bf_other"
        else:
            name = clip.generator.name.lower() if clip.generator else ""
            group = "spoof_piper" if "piper" in name else "spoof_chatterbox"
        counts[group] += 1
    return counts


def test_balanced_batch_sampler_composition(feature_dataset: FeatureDataset) -> None:
    sampler = BalancedBatchSampler(
        feature_dataset, batch_size=40, seed="unit-batches", batches_per_epoch=3
    )

    for batch in sampler:
        counts = _batch_groups(feature_dataset, batch)
        spoof = counts["spoof_chatterbox"] + counts["spoof_piper"]
        bona_fide = counts["bf_vox"] + counts["bf_other"]
        assert len(batch) == 40
        assert spoof == bona_fide == 20
        assert counts["spoof_chatterbox"] / spoof == pytest.approx(0.70)
        assert counts["spoof_piper"] / spoof == pytest.approx(0.30)
        assert counts["bf_vox"] / bona_fide == pytest.approx(0.25)


def test_balanced_batch_sampler_is_deterministic_per_epoch(
    feature_dataset: FeatureDataset,
) -> None:
    sampler = BalancedBatchSampler(
        feature_dataset, batch_size=16, seed="unit-batches", batches_per_epoch=4
    )
    sampler.set_epoch(7)
    first = list(sampler)
    sampler.set_epoch(7)
    assert list(sampler) == first
    sampler.set_epoch(8)
    assert list(sampler) != first


def test_feature_aasist_matches_vendored_backend_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backend classes do not use fairseq; only the unused SSLModel import does.
    monkeypatch.setitem(sys.modules, "fairseq", types.ModuleType("fairseq"))
    model = FeatureAASIST()

    expected_modules = {
        "LL",
        "first_bn",
        "first_bn1",
        "drop",
        "drop_way",
        "selu",
        "encoder",
        "attention",
        "GAT_layer_S",
        "GAT_layer_T",
        "HtrgGAT_layer_ST11",
        "HtrgGAT_layer_ST12",
        "HtrgGAT_layer_ST21",
        "HtrgGAT_layer_ST22",
        "pool_S",
        "pool_T",
        "pool_hS1",
        "pool_hT1",
        "pool_hS2",
        "pool_hT2",
        "out_layer",
    }
    assert set(dict(model.named_children())) == expected_modules
    keys = set(model.state_dict())
    assert not any(key.startswith("ssl_model.") for key in keys)
    assert {
        "pos_S",
        "master1",
        "master2",
        "LL.weight",
        "encoder.0.0.conv1.weight",
        "attention.0.weight",
        "GAT_layer_S.att_weight",
        "HtrgGAT_layer_ST11.att_weight11",
        "pool_hT2.proj.weight",
        "out_layer.weight",
    } <= keys


def test_feature_dataset_uses_upstream_label_indices(
    tmp_path: Path, calibration_clips: list[ClipEntry]
) -> None:
    bona_fide = next(clip for clip in calibration_clips if clip.label == "bona_fide")
    spoof = next(clip for clip in calibration_clips if clip.label == "spoof")
    for clip in (bona_fide, spoof):
        torch.save(torch.ones((2, 3)), tmp_path / f"{clip.clip_id}.pt")
    dataset = FeatureDataset([bona_fide.clip_id, spoof.clip_id], tmp_path, calibration_clips)

    assert dataset[0][1] == LABEL_BONAFIDE == 1
    assert dataset[1][1] == LABEL_SPOOF == 0


def test_balanced_batch_sampler_rejects_small_batch_size(
    feature_dataset: FeatureDataset,
) -> None:
    with pytest.raises(ValueError, match="at least eight"):
        BalancedBatchSampler(feature_dataset, batch_size=4, seed="unit")


def test_score_clips_returns_negated_column_1_per_clip(
    tmp_path: Path, calibration_clips: list[ClipEntry]
) -> None:
    bf = next(c for c in calibration_clips if c.label == "bona_fide")
    sp = next(c for c in calibration_clips if c.label == "spoof")
    for clip in (bf, sp):
        torch.save(torch.randn(3, FEATURE_DIM), tmp_path / f"{clip.clip_id}.pt")
    dataset = FeatureDataset([bf.clip_id, sp.clip_id], tmp_path, calibration_clips)

    class StubModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[0.5, -2.0]])

    model = StubModel()
    scores = score_clips(model, dataset, "cpu")

    assert set(scores) == {bf.clip_id, sp.clip_id}
    for score in scores.values():
        assert score == pytest.approx(2.0)


def test_freeze_batch_norm_keeps_all_batch_norm_layers_in_eval_mode() -> None:
    model = nn.Sequential(
        nn.BatchNorm1d(2),
        nn.Sequential(nn.BatchNorm2d(2), nn.BatchNorm3d(2)),
        nn.Linear(2, 2),
    )
    model.train()
    _freeze_batch_norm(model)

    assert model.training
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    assert batch_norms
    assert all(not module.training for module in batch_norms)
