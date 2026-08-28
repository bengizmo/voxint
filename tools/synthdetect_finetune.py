#!/usr/bin/env python3
"""Feature-cached fine-tuning for the w2v2-AASIST backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from synthdetect_corpus import ClipEntry, Manifest, load_manifest
from synthdetect_eval import compute_eer
from synthdetect_infer import read_canonical_pcm, verify_clip_sha
from synthdetect_sources import MODELS, SELECTION_SEED
from torch.utils.data import DataLoader, Dataset, Sampler

MODEL_WIDTH_SAMPLES = 64_600
FEATURE_DIM = 1_024
DEFAULT_DECISION_THRESHOLD = 0.0
LABEL_SPOOF = 0
LABEL_BONAFIDE = 1
_VENDOR_PATH = Path(__file__).resolve().parent / "synthdetect_vendor" / "ssl_antispoofing_model.py"


def _load_vendor() -> Any:
    name = "synthdetect_finetune_vendor"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _VENDOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load vendored model at {_VENDOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FeatureAASIST(nn.Module):
    """The checkpoint-compatible AASIST backend without its XLS-R front end."""

    def __init__(self) -> None:
        super().__init__()
        vendor = _load_vendor()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.LL = nn.Linear(FEATURE_DIM, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(vendor.Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(vendor.Residual_block(nb_filts=filts[2])),
            nn.Sequential(vendor.Residual_block(nb_filts=filts[3])),
            nn.Sequential(vendor.Residual_block(nb_filts=filts[4])),
            nn.Sequential(vendor.Residual_block(nb_filts=filts[4])),
            nn.Sequential(vendor.Residual_block(nb_filts=filts[4])),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        last_filt: int = filts[-1][-1]  # type: ignore[index]
        self.pos_S = nn.Parameter(torch.randn(1, 42, last_filt))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = vendor.GraphAttentionLayer(
            last_filt, gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = vendor.GraphAttentionLayer(
            last_filt, gat_dims[0], temperature=temperatures[1]
        )
        self.HtrgGAT_layer_ST11 = vendor.HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = vendor.HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = vendor.HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = vendor.HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.pool_S = vendor.GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = vendor.GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = vendor.GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = vendor.GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = vendor.GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = vendor.GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        self.decision_threshold = DEFAULT_DECISION_THRESHOLD

    def forward(self, x_ssl_feat: torch.Tensor) -> torch.Tensor:
        x = self.LL(x_ssl_feat)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output: torch.Tensor = self.out_layer(last_hidden)
        return output

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device | str,
        key_prefix: str | None = None,
    ) -> FeatureAASIST:
        model = cls()
        loaded = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        state = loaded.get("state_dict", loaded) if isinstance(loaded, Mapping) else loaded
        if not isinstance(state, Mapping):
            raise TypeError("checkpoint must contain a state dict")
        stripped: dict[str, torch.Tensor] = {}
        for raw_key, value in state.items():
            key = str(raw_key)
            if key_prefix is not None:
                if not key.startswith(key_prefix):
                    raise RuntimeError(
                        f"checkpoint key {key!r} lacks declared prefix {key_prefix!r}"
                    )
                key = key[len(key_prefix) :]
            if not key.startswith("ssl_model."):
                stripped[key] = value
        model.load_state_dict(stripped, strict=True)
        return model.to(device)


def _manifest(value: Manifest | Path | str) -> Manifest:
    if isinstance(value, Manifest):
        return value
    return load_manifest(json.loads(Path(value).read_text(encoding="utf-8")))


def _weight_for(model_id: str, role: str) -> Any:
    try:
        model = MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model_id {model_id!r}") from exc
    matches = [weight for weight in model.weights if weight.role == role]
    if len(matches) != 1:
        raise ValueError(f"model {model_id!r} must have exactly one {role!r} weight")
    return matches[0]


def _repeat_pad(samples: np.ndarray, width: int = MODEL_WIDTH_SAMPLES) -> np.ndarray:
    if samples.size == 0:
        raise ValueError("cannot repeat-pad an empty clip")
    if samples.size >= width:
        return np.ascontiguousarray(samples[:width])
    repeats = math.ceil(width / samples.size)
    return np.ascontiguousarray(np.tile(samples, repeats)[:width])


def extract_and_cache_features(
    manifest: Manifest | Path | str,
    corpus_root: Path | str,
    weights_dir: Path | str,
    model_id: str,
    cache_dir: Path | str,
    device: torch.device | str,
    split: str,
) -> None:
    from synthdetect_infer import _load_real_engine, parse_device_index

    loaded_manifest = _manifest(manifest)
    clips = [clip for clip in loaded_manifest.clips if clip.split == split]
    if not clips:
        raise ValueError(f"manifest has no clips in split {split!r}")
    weights_root = Path(weights_dir)
    device_obj = torch.device(device)
    if device_obj.type != "cuda" or device_obj.index is None:
        device_obj = torch.device(f"cuda:{parse_device_index(str(device_obj))}")
    loaded = _load_real_engine(MODELS[model_id], weights_root, device_obj.index)
    full_model = loaded.engine._net  # type: ignore[attr-defined]
    full_model.eval()
    full_model.ssl_model.model.eval()
    output_root = Path(cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_shape: list[int] | None = None
    with torch.no_grad(), torch.inference_mode():
        for clip in clips:
            destination = output_root / f"{clip.clip_id}.pt"
            audio = read_canonical_pcm(Path(corpus_root) / clip.rel_path)
            verify_clip_sha(clip, audio)
            samples = _repeat_pad(audio.samples).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples).unsqueeze(0).to(device_obj)
            full_model.ssl_model.model.eval()
            features = full_model.ssl_model.extract_feat(tensor).squeeze(0)
            features = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
                raise RuntimeError(
                    f"{clip.clip_id}: expected features (T, {FEATURE_DIM}), "
                    f"got {tuple(features.shape)}"
                )
            torch.save(features, destination)
            shape = list(features.shape)
            if feature_shape is not None and shape != feature_shape:
                raise RuntimeError(f"inconsistent feature shapes: {feature_shape} and {shape}")
            feature_shape = shape
    xlsr_weight = _weight_for(model_id, "xlsr_ssl_base")
    meta = {
        "model_id": model_id,
        "split": split,
        "xlsr_checkpoint_sha256": xlsr_weight.sha256,
        "clip_count": len(clips),
        "feature_shape": feature_shape,
    }
    (output_root / "cache_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class SpeakerSplit:
    """Deterministic, source-aware, speaker-disjoint calibration partition."""

    def __init__(self, dev_fraction: float = 0.2, seed: str = SELECTION_SEED) -> None:
        if not 0.0 < dev_fraction < 1.0:
            raise ValueError("dev_fraction must be between zero and one")
        self.dev_fraction = dev_fraction
        self.seed = seed

    def partition(self, clips: Sequence[ClipEntry]) -> tuple[list[str], list[str]]:
        calibration = [clip for clip in clips if clip.split == "calibration"]
        if not calibration:
            raise ValueError("manifest has no calibration clips")
        by_speaker: dict[str, list[ClipEntry]] = defaultdict(list)
        for clip in calibration:
            by_speaker[clip.speaker_id].append(clip)

        def rank(speaker_id: str) -> int:
            payload = f"{self.seed}:{speaker_id}".encode()
            return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

        cutoff = int(self.dev_fraction * (1 << 64))
        dev_speakers = {speaker for speaker in by_speaker if rank(speaker) < cutoff}
        all_speakers = set(by_speaker)
        if not dev_speakers:
            dev_speakers.add(min(all_speakers, key=rank))
        if dev_speakers == all_speakers:
            dev_speakers.remove(max(dev_speakers, key=rank))
        sources = {clip.source for clip in calibration}
        for source in sorted(sources):
            if not any(
                clip.source == source for speaker in dev_speakers for clip in by_speaker[speaker]
            ):
                candidates = {
                    clip.speaker_id for clip in calibration if clip.source == source
                } - dev_speakers
                if candidates and len(all_speakers - dev_speakers) > 1:
                    dev_speakers.add(min(candidates, key=rank))
        train = [clip.clip_id for clip in calibration if clip.speaker_id not in dev_speakers]
        dev = [clip.clip_id for clip in calibration if clip.speaker_id in dev_speakers]
        if not train or not dev:
            raise ValueError("speaker partition needs at least two speakers")
        train_clips = [clip for clip in calibration if clip.speaker_id not in dev_speakers]
        train_groups = self._sampler_groups(train_clips)
        empty = [g for g, ids in train_groups.items() if not ids]
        if empty:
            raise ValueError(
                f"speaker partition leaves training groups empty: {sorted(empty)}; "
                f"add more speakers or reduce dev_fraction"
            )
        return train, dev

    @staticmethod
    def _sampler_groups(clips: list[ClipEntry]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            "bf_vox": [], "bf_other": [], "spoof_chatterbox": [], "spoof_piper": [],
        }
        for clip in clips:
            if clip.label == "bona_fide":
                key = "bf_vox" if "voxconverse" in clip.source.lower() else "bf_other"
            else:
                gen = clip.generator.name.lower() if clip.generator is not None else ""
                key = "spoof_piper" if "piper" in gen else "spoof_chatterbox"
            groups[key].append(clip.clip_id)
        return groups

    def __call__(self, clips: Sequence[ClipEntry]) -> tuple[list[str], list[str]]:
        return self.partition(clips)


class FeatureDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        clip_ids: Sequence[str],
        cache_dir: Path | str,
        manifest_clips: Sequence[ClipEntry] | Mapping[str, ClipEntry],
    ) -> None:
        self.clip_ids = list(clip_ids)
        self.cache_dir = Path(cache_dir)
        self.manifest_clips = (
            dict(manifest_clips)
            if isinstance(manifest_clips, Mapping)
            else {clip.clip_id: clip for clip in manifest_clips}
        )
        missing_meta = set(self.clip_ids) - set(self.manifest_clips)
        if missing_meta:
            raise ValueError(f"missing manifest metadata for {sorted(missing_meta)[:3]}")

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        clip_id = self.clip_ids[index]
        features = torch.load(
            self.cache_dir / f"{clip_id}.pt", map_location="cpu", weights_only=True
        )
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError(f"invalid cached feature tensor for {clip_id!r}")
        features = features.to(dtype=torch.float32)
        label = (
            LABEL_BONAFIDE
            if self.manifest_clips[clip_id].label == "bona_fide"
            else LABEL_SPOOF
        )
        return features, label


class BalancedBatchSampler(Sampler[list[int]]):
    """Oversampled 50/50 batches with fixed spoof-generator and bona-fide mixes."""

    def __init__(
        self,
        dataset: FeatureDataset,
        batch_size: int,
        seed: str = SELECTION_SEED,
        batches_per_epoch: int | None = None,
    ) -> None:
        if batch_size < 8 or batch_size % 2:
            raise ValueError("batch_size must be even and at least eight")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, clip_id in enumerate(dataset.clip_ids):
            clip = dataset.manifest_clips[clip_id]
            if clip.label == "bona_fide":
                key = "bf_vox" if "voxconverse" in clip.source.lower() else "bf_other"
            else:
                generator = clip.generator.name.lower() if clip.generator is not None else ""
                key = "spoof_piper" if "piper" in generator else "spoof_chatterbox"
            self.groups[key].append(index)
        required = {"bf_vox", "bf_other", "spoof_piper", "spoof_chatterbox"}
        absent = required - {key for key, values in self.groups.items() if values}
        if absent:
            raise ValueError(f"balanced sampling groups are empty: {sorted(absent)}")
        half = batch_size // 2
        per_batch = {
            "spoof_chatterbox": round(half * 0.7),
            "spoof_piper": half - round(half * 0.7),
            "bf_vox": round(half * 0.25),
            "bf_other": half - round(half * 0.25),
        }
        default_batches = max(
            math.ceil(len(self.groups[key]) / per_batch[key])
            for key in required
            if per_batch[key] > 0
        )
        self.batches_per_epoch = batches_per_epoch or default_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        half = self.batch_size // 2
        chatterbox_n = round(half * 0.7)
        piper_n = half - chatterbox_n
        vox_n = round(half * 0.25)
        other_n = half - vox_n

        def sample(group: str, count: int) -> list[int]:
            return [rng.choice(self.groups[group]) for _ in range(count)]

        for _ in range(self.batches_per_epoch):
            batch = (
                sample("spoof_chatterbox", chatterbox_n)
                + sample("spoof_piper", piper_n)
                + sample("bf_vox", vox_n)
                + sample("bf_other", other_n)
            )
            rng.shuffle(batch)
            yield batch


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    cache_dir: Path
    checkpoint_path: Path
    output_dir: Path
    device: str = "cuda:0"
    key_prefix: str | None = None
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    patience: int = 3
    num_workers: int = 0
    decision_threshold: float = DEFAULT_DECISION_THRESHOLD
    baseline_piper_eer: float | None = None
    baseline_bf_fpr: float | None = None
    piper_max_regression: float = 0.02
    bf_fpr_max_regression: float = 0.01
    seed: str = SELECTION_SEED


def _freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _linear_warmup_cosine(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _generator_name(clip: ClipEntry) -> str | None:
    if clip.generator is None:
        return None
    name = clip.generator.name.lower()
    if "chatterbox" in name:
        return "chatterbox"
    if "piper" in name:
        return "piper"
    return name


def evaluate_dev(
    model: nn.Module,
    dev_dataset: FeatureDataset,
    dev_clips_meta: Sequence[ClipEntry] | Mapping[str, ClipEntry],
    device: torch.device | str,
) -> dict[str, float]:
    metadata = (
        dict(dev_clips_meta)
        if isinstance(dev_clips_meta, Mapping)
        else {clip.clip_id: clip for clip in dev_clips_meta}
    )
    loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
    scores: dict[str, float] = {}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for clip_id, (features, _) in zip(dev_dataset.clip_ids, loader, strict=True):
            logits = model(features.to(device))
            scores[clip_id] = float((-logits[:, 1]).item())
    if was_training:
        model.train()
        _freeze_batch_norm(model)
    metrics: dict[str, float] = {}
    bona_fide_ids = [
        clip_id for clip_id in dev_dataset.clip_ids if metadata[clip_id].label == "bona_fide"
    ]
    for generator in ("chatterbox", "piper"):
        spoof_ids = [
            clip_id
            for clip_id in dev_dataset.clip_ids
            if _generator_name(metadata[clip_id]) == generator
        ]
        if not spoof_ids or not bona_fide_ids:
            raise ValueError(f"dev EER for {generator} requires spoof and bona-fide clips")
        ids = bona_fide_ids + spoof_ids
        labels = [0] * len(bona_fide_ids) + [1] * len(spoof_ids)
        eer, _ = compute_eer(labels, [scores[clip_id] for clip_id in ids])
        metrics[f"{generator}_dev_eer"] = float(eer)
        for source in sorted({metadata[clip_id].source for clip_id in ids}):
            source_bf = [clip_id for clip_id in bona_fide_ids if metadata[clip_id].source == source]
            source_spoof = [clip_id for clip_id in spoof_ids if metadata[clip_id].source == source]
            if source_bf and source_spoof:
                source_ids = source_bf + source_spoof
                source_labels = [0] * len(source_bf) + [1] * len(source_spoof)
                source_eer, _ = compute_eer(
                    source_labels, [scores[clip_id] for clip_id in source_ids]
                )
                metrics[f"{generator}_dev_eer_source_{source}"] = float(source_eer)
    threshold = float(getattr(model, "decision_threshold", DEFAULT_DECISION_THRESHOLD))
    metrics["bf_fpr"] = float(np.mean([scores[clip_id] >= threshold for clip_id in bona_fide_ids]))
    for source in sorted({metadata[clip_id].source for clip_id in dev_dataset.clip_ids}):
        source_ids = [clip_id for clip_id in bona_fide_ids if metadata[clip_id].source == source]
        if source_ids:
            metrics[f"bf_fpr_source_{source}"] = float(
                np.mean([scores[clip_id] >= threshold for clip_id in source_ids])
            )
    return metrics


def train(config: TrainConfig) -> Path:
    manifest = _manifest(config.manifest)
    train_ids, dev_ids = SpeakerSplit(seed=config.seed)(manifest.clips)
    metadata = {clip.clip_id: clip for clip in manifest.clips}
    train_dataset = FeatureDataset(train_ids, config.cache_dir, metadata)
    dev_dataset = FeatureDataset(dev_ids, config.cache_dir, metadata)
    sampler = BalancedBatchSampler(train_dataset, config.batch_size, config.seed)
    loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
    )
    device = torch.device(config.device)
    model = FeatureAASIST.from_checkpoint(
        config.checkpoint_path, device, key_prefix=config.key_prefix
    )
    model.decision_threshold = config.decision_threshold
    model.train()
    _freeze_batch_norm(model)
    initial_metrics = evaluate_dev(model, dev_dataset, metadata, device)
    baseline_piper_eer = (
        config.baseline_piper_eer
        if config.baseline_piper_eer is not None
        else initial_metrics["piper_dev_eer"]
    )
    baseline_bf_fpr = (
        config.baseline_bf_fpr if config.baseline_bf_fpr is not None else initial_metrics["bf_fpr"]
    )
    print(json.dumps({"baseline": True, **initial_metrics}, sort_keys=True), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, config.epochs * len(loader))
    scheduler = _linear_warmup_cosine(optimizer, config.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.output_dir / "best_feature_aasist.pt"
    initial_piper_ok = (
        initial_metrics["piper_dev_eer"] <= baseline_piper_eer + config.piper_max_regression
    )
    initial_bf_ok = initial_metrics["bf_fpr"] <= baseline_bf_fpr + config.bf_fpr_max_regression
    if initial_piper_ok and initial_bf_ok:
        best_eer = initial_metrics["chatterbox_dev_eer"]
        torch.save(model.state_dict(), best_path)
        (config.output_dir / "best_metrics.json").write_text(
            json.dumps({"baseline": True, **initial_metrics}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        best_eer = math.inf
    stale_epochs = 0
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        _freeze_batch_norm(model)
        total_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach())
        metrics = evaluate_dev(model, dev_dataset, metadata, device)
        piper_ok = metrics["piper_dev_eer"] <= baseline_piper_eer + config.piper_max_regression
        bf_ok = metrics["bf_fpr"] <= baseline_bf_fpr + config.bf_fpr_max_regression
        constrained = piper_ok and bf_ok
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(loader)),
            "learning_rate": scheduler.get_last_lr()[0],
            "regression_gates_pass": constrained,
            **metrics,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if constrained and metrics["chatterbox_dev_eer"] < best_eer:
            best_eer = metrics["chatterbox_dev_eer"]
            stale_epochs = 0
            torch.save(model.state_dict(), best_path)
            (config.output_dir / "best_metrics.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("no checkpoint passed the Piper EER and bona-fide FPR gates")
    return best_path


def _add_manifest_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cache = subparsers.add_parser("cache-features")
    _add_manifest_cache_args(cache)
    cache.add_argument("--corpus-root", type=Path, required=True)
    cache.add_argument("--weights-dir", type=Path, required=True)
    cache.add_argument("--model-id", choices=sorted(MODELS), default="w2v2-aasist")
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--split", default="calibration")

    fit = subparsers.add_parser("train")
    _add_manifest_cache_args(fit)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--key-prefix", choices=["module."], default=None)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--warmup-steps", type=int, default=500)
    fit.add_argument("--patience", type=int, default=3)
    fit.add_argument("--num-workers", type=int, default=0)
    fit.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    fit.add_argument("--baseline-piper-eer", type=float)
    fit.add_argument("--baseline-bf-fpr", type=float)
    fit.add_argument("--seed", default=SELECTION_SEED)

    evaluate = subparsers.add_parser("evaluate")
    _add_manifest_cache_args(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--split", default="calibration")
    evaluate.add_argument("--decision-threshold", type=float, default=DEFAULT_DECISION_THRESHOLD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "cache-features":
        extract_and_cache_features(
            args.manifest,
            args.corpus_root,
            args.weights_dir,
            args.model_id,
            args.cache_dir,
            args.device,
            args.split,
        )
        return 0
    if args.command == "train":
        config = TrainConfig(
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            key_prefix=args.key_prefix,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            patience=args.patience,
            num_workers=args.num_workers,
            decision_threshold=args.decision_threshold,
            baseline_piper_eer=args.baseline_piper_eer,
            baseline_bf_fpr=args.baseline_bf_fpr,
            seed=args.seed,
        )
        print(train(config))
        return 0
    manifest = _manifest(args.manifest)
    clip_ids = [clip.clip_id for clip in manifest.clips if clip.split == args.split]
    dataset = FeatureDataset(clip_ids, args.cache_dir, manifest.clips)
    model = FeatureAASIST.from_checkpoint(args.checkpoint, args.device)
    model.decision_threshold = args.decision_threshold
    metrics = evaluate_dev(model, dataset, manifest.clips, args.device)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
