"""File-based adapters behind ``voxint score`` — parse, validate, score, report.

Everything here is file-in/file-out: no settings, no database, no worker. Input
contracts (schema_version 1) are documented in ``docs/harness.md``. Malformed
input is reported with its file and line number and exits with code 2; reports
are written atomically (temp file + rename) with deterministic serialization.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from voxint.harness import agreement, ensemble, name_accuracy
from voxint.harness.vectors import InvalidVectorError, TaggedVector, tagged_vector

SCHEMA_VERSION = 1

KIND_CURATED = "curated"
KIND_NEGATIVE_CONTROL = "negative_control"


class InputError(Exception):
    """A user-input problem (bad file, bad schema). Message is user-facing."""


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for each non-blank line of a JSONL file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path}: {exc.strerror or exc}") from exc
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise InputError(f"{path}:{lineno}: expected a JSON object")
        yield lineno, record


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"{path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{path}: expected a JSON object")
    return payload


def _check_schema_version(payload: Mapping[str, Any], where: str) -> None:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise InputError(
            f"{where}: schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )


def _write_atomic(path: Path | None, text: str) -> None:
    """Write ``text`` to ``path`` atomically, or to stdout when path is None."""
    if path is None:
        sys.stdout.write(text)
        return
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _require_number(payload: Mapping[str, Any], key: str, where: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where}: {key!r} must be a number")
    return float(value)


def _require_int(payload: Mapping[str, Any], key: str, where: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{where}: {key!r} must be an integer")
    return value


def _require_str(record: Mapping[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: {key!r} must be a non-empty string")
    return value


def _optional_number(record: Mapping[str, Any], key: str, where: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where}: {key!r} must be a number")
    if not math.isfinite(value):
        raise InputError(f"{where}: {key!r} must be finite")
    return float(value)


def _optional_non_negative(record: Mapping[str, Any], key: str, where: str) -> float | None:
    value = _optional_number(record, key, where)
    if value is not None and value < 0.0:
        raise InputError(f"{where}: {key!r} must be >= 0")
    return value


# --------------------------------------------------------------------------- #
# name-accuracy
# --------------------------------------------------------------------------- #
def _load_aliases(path: Path | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    payload = _read_json(path)
    _check_schema_version(payload, str(path))
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict):
        raise InputError(f"{path}: 'aliases' must be an object of canonical -> [alts]")
    out: dict[str, list[str]] = {}
    for canonical, alts in aliases.items():
        if not isinstance(alts, list) or not all(isinstance(a, str) for a in alts):
            raise InputError(f"{path}: aliases[{canonical!r}] must be a list of strings")
        out[canonical] = list(alts)
    return out


def _parse_name_items(
    path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Parse a name-accuracy JSONL file into ``{item_id: {slot: fields}}``."""
    items: dict[str, dict[str, dict[str, Any]]] = {}
    for lineno, record in _read_jsonl(path):
        where = f"{path}:{lineno}"
        item_id = _require_str(record, "item_id", where)
        if item_id in items:
            raise InputError(f"{where}: duplicate item_id {item_id!r}")
        slots = record.get("slots")
        if not isinstance(slots, dict) or not slots:
            raise InputError(f"{where}: 'slots' must be a non-empty object")
        parsed: dict[str, dict[str, Any]] = {}
        for slot_name, fields in slots.items():
            slot_where = f"{where} slot {slot_name!r}"
            if not isinstance(fields, dict):
                raise InputError(f"{slot_where}: must be an object")
            assigned = fields.get("assigned_name")
            if assigned is not None and not isinstance(assigned, str):
                raise InputError(f"{slot_where}: 'assigned_name' must be a string or null")
            truth = fields.get("truth")
            if truth is not None and not isinstance(truth, str):
                raise InputError(f"{slot_where}: 'truth' must be a string or null")
            parsed[str(slot_name)] = {
                "assigned_name": assigned,
                "truth": truth,
                "confidence": _optional_number(fields, "confidence", slot_where),
                "duration": _optional_non_negative(fields, "duration", slot_where),
            }
        items[item_id] = parsed
    if not items:
        raise InputError(f"{path}: no records")
    return items


def _score_name_items(
    items: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aliases: Mapping[str, list[str]] | None,
) -> tuple[list[dict[str, Any]], list[str | tuple[str, float]], list[tuple[float | None, bool]]]:
    """Per-item verdicts + flat (weighted) verdict list + risk-coverage pairs."""
    per_item: list[dict[str, Any]] = []
    flat: list[str | tuple[str, float]] = []
    rc_pairs: list[tuple[float | None, bool]] = []
    for item_id in sorted(items):
        verdicts: dict[str, str] = {}
        for slot_name in sorted(items[item_id]):
            fields = items[item_id][slot_name]
            verdict = name_accuracy.slot_verdict(
                fields["assigned_name"], fields["truth"], aliases=aliases
            )
            verdicts[slot_name] = verdict
            duration = fields["duration"]
            flat.append(verdict if duration is None else (verdict, float(duration)))
            if verdict != name_accuracy.EXCLUDED:
                rc_pairs.append(
                    (
                        fields["confidence"],
                        verdict in (name_accuracy.TP, name_accuracy.TN),
                    )
                )
        per_item.append({"item_id": item_id, "verdicts": verdicts})
    return per_item, flat, rc_pairs


def _paired_correctness(
    baseline: Mapping[str, Mapping[str, Mapping[str, Any]]],
    candidate: Mapping[str, Mapping[str, Mapping[str, Any]]],
    aliases: Mapping[str, list[str]] | None,
) -> list[list[tuple[bool, bool]]]:
    """Per-item clusters of (baseline_correct, candidate_correct) slot pairs."""
    if set(baseline) != set(candidate):
        only_base = sorted(set(baseline) - set(candidate))[:5]
        only_cand = sorted(set(candidate) - set(baseline))[:5]
        raise InputError(
            "baseline and input item_ids differ "
            f"(baseline-only: {only_base}, input-only: {only_cand})"
        )
    clusters: list[list[tuple[bool, bool]]] = []
    ok = (name_accuracy.TP, name_accuracy.TN)
    for item_id in sorted(baseline):
        base_slots, cand_slots = baseline[item_id], candidate[item_id]
        if set(base_slots) != set(cand_slots):
            raise InputError(f"item {item_id!r}: baseline and input slot labels differ")
        pairs: list[tuple[bool, bool]] = []
        for slot_name in sorted(base_slots):
            if base_slots[slot_name]["truth"] != cand_slots[slot_name]["truth"]:
                raise InputError(
                    f"item {item_id!r} slot {slot_name!r}: baseline and input disagree "
                    "on ground truth — paired statistics require one shared truth"
                )
            bv = name_accuracy.slot_verdict(
                base_slots[slot_name]["assigned_name"],
                base_slots[slot_name]["truth"],
                aliases=aliases,
            )
            cv = name_accuracy.slot_verdict(
                cand_slots[slot_name]["assigned_name"],
                cand_slots[slot_name]["truth"],
                aliases=aliases,
            )
            if bv == name_accuracy.EXCLUDED or cv == name_accuracy.EXCLUDED:
                continue
            pairs.append((bv in ok, cv in ok))
        if pairs:
            clusters.append(pairs)
    return clusters


def cmd_name_accuracy(args: argparse.Namespace) -> int:
    aliases = _load_aliases(Path(args.aliases) if args.aliases else None)
    items = _parse_name_items(Path(args.input))
    per_item, flat, rc_pairs = _score_name_items(items, aliases)
    agg = name_accuracy.aggregate(flat)

    scored_n = agg.tp + agg.fp_wrong + agg.fp_overname + agg.fn + agg.tn
    accuracy_ci = name_accuracy.wilson_ci(agg.tp + agg.tn, scored_n)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "name_accuracy_report",
        "n_items": len(items),
        "n_slots_scored": scored_n,
        "aggregate": dataclasses.asdict(agg),
        "accuracy": (agg.tp + agg.tn) / scored_n if scored_n else 0.0,
        "accuracy_ci95": list(accuracy_ci),
        "per_item": per_item,
    }

    if any(conf is not None for conf, _ in rc_pairs):
        rc = name_accuracy.risk_coverage(rc_pairs, target_accuracy=args.target_accuracy)
        report["risk_coverage"] = dataclasses.asdict(rc)

    if args.baseline:
        base_items = _parse_name_items(Path(args.baseline))
        clusters = _paired_correctness(base_items, items, aliases)
        pairs = [pair for cluster in clusters for pair in cluster]
        mn = name_accuracy.mcnemar([p for p, _ in pairs], [c for _, c in pairs])
        boot = name_accuracy.clustered_bootstrap_delta(clusters, seed=args.seed)
        report["paired"] = {
            "n_slots": len(pairs),
            "mcnemar": dataclasses.asdict(mn),
            "bootstrap": dataclasses.asdict(boot),
        }

    _write_atomic(Path(args.out) if args.out else None, _dumps(report) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# agreement
# --------------------------------------------------------------------------- #
def _load_thresholds(path: Path) -> agreement.Thresholds:
    payload = _read_json(path)
    _check_schema_version(payload, str(path))
    required = (
        "tau",
        "margin",
        "min_duration",
        "min_segments",
        "low_band",
        "neg_min_total_duration",
        "min_enrollment_items",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise InputError(f"{path}: missing threshold fields {missing}")
    try:
        return agreement.Thresholds(
            tau=_require_number(payload, "tau", str(path)),
            margin=_require_number(payload, "margin", str(path)),
            min_duration=_require_number(payload, "min_duration", str(path)),
            min_segments=_require_int(payload, "min_segments", str(path)),
            low_band=_require_number(payload, "low_band", str(path)),
            neg_min_total_duration=_require_number(
                payload, "neg_min_total_duration", str(path)
            ),
            min_enrollment_items=_require_int(payload, "min_enrollment_items", str(path)),
        )
    except (TypeError, ValueError) as exc:
        raise InputError(f"{path}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class Voiceprint:
    """One enrolled voiceprint + the provenance the leakage gates read."""

    vector: TaggedVector
    enrollment_items: int
    held_out: bool
    source_item_ids: frozenset[str]


def _load_enrollment(path: Path) -> tuple[str, dict[str, Voiceprint]]:
    payload = _read_json(path)
    _check_schema_version(payload, str(path))
    space = payload.get("embedding_space")
    if not isinstance(space, str) or not space.strip():
        raise InputError(f"{path}: 'embedding_space' must be a non-empty string")
    dims = payload.get("dims")
    if not isinstance(dims, int) or isinstance(dims, bool) or dims <= 0:
        raise InputError(f"{path}: 'dims' must be a positive integer")
    raw = payload.get("voiceprints")
    if not isinstance(raw, dict) or not raw:
        raise InputError(f"{path}: 'voiceprints' must be a non-empty object")

    voiceprints: dict[str, Voiceprint] = {}
    for host_id, fields in raw.items():
        where = f"{path} voiceprint {host_id!r}"
        if not isinstance(fields, dict):
            raise InputError(f"{where}: must be an object")
        try:
            vector = tagged_vector(space, fields.get("embedding"))
        except InvalidVectorError as exc:
            raise InputError(f"{where}: {exc}") from exc
        if vector.dims != dims:
            raise InputError(f"{where}: embedding has {vector.dims} dims, expected {dims}")
        n_items = fields.get("enrollment_items")
        if not isinstance(n_items, int) or isinstance(n_items, bool) or n_items < 0:
            raise InputError(f"{where}: 'enrollment_items' must be a non-negative integer")
        held_out = fields.get("held_out")
        if not isinstance(held_out, bool):
            raise InputError(f"{where}: 'held_out' must be a boolean")
        sources = fields.get("source_item_ids", [])
        if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
            raise InputError(f"{where}: 'source_item_ids' must be a list of strings")
        voiceprints[str(host_id)] = Voiceprint(
            vector=vector,
            enrollment_items=n_items,
            held_out=held_out,
            source_item_ids=frozenset(sources),
        )
    return space, voiceprints


def _parse_agreement_slots(
    fields: Mapping[str, Any], space: str, dims: int, where: str
) -> dict[str, agreement.Slot]:
    slots_raw = fields.get("slots")
    if not isinstance(slots_raw, dict):
        raise InputError(f"{where}: 'slots' must be an object")
    slots: dict[str, agreement.Slot] = {}
    for slot_name, slot_fields in slots_raw.items():
        slot_where = f"{where} slot {slot_name!r}"
        if not isinstance(slot_fields, dict):
            raise InputError(f"{slot_where}: must be an object")
        try:
            vector = tagged_vector(space, slot_fields.get("embedding"))
        except InvalidVectorError as exc:
            raise InputError(f"{slot_where}: {exc}") from exc
        if vector.dims != dims:
            raise InputError(
                f"{slot_where}: embedding has {vector.dims} dims, expected {dims}"
            )
        duration = _optional_non_negative(slot_fields, "duration", slot_where)
        segments = slot_fields.get("segments")
        if not isinstance(segments, int) or isinstance(segments, bool) or segments < 0:
            raise InputError(f"{slot_where}: 'segments' must be a non-negative integer")
        slots[str(slot_name)] = agreement.Slot(
            vector=vector,
            duration=duration if duration is not None else 0.0,
            segments=segments,
        )
    return slots


def _enrollment_gate(
    voiceprint: Voiceprint, item_id: str, thresholds: agreement.Thresholds
) -> tuple[bool, str]:
    """Derive (enrollment_ok, reason) from provenance. Leakage beats weakness."""
    if item_id in voiceprint.source_item_ids:
        return False, agreement.REASON_SESSION_LEAKAGE_RISK
    if not voiceprint.held_out:
        return False, agreement.REASON_SESSION_LEAKAGE_RISK
    if voiceprint.enrollment_items < thresholds.min_enrollment_items:
        return False, agreement.REASON_WEAK_ENROLLMENT
    return True, agreement.REASON_WEAK_ENROLLMENT


def cmd_agreement(args: argparse.Namespace) -> int:
    thresholds = _load_thresholds(Path(args.thresholds))
    space, voiceprints = _load_enrollment(Path(args.enrollment))
    dims = next(iter(voiceprints.values())).vector.dims

    lines: list[str] = []
    seen: set[str] = set()
    for lineno, record in _read_jsonl(Path(args.slots)):
        where = f"{args.slots}:{lineno}"
        item_id = _require_str(record, "item_id", where)
        if item_id in seen:
            raise InputError(f"{where}: duplicate item_id {item_id!r}")
        seen.add(item_id)
        kind = _require_str(record, "kind", where)
        if kind not in (KIND_CURATED, KIND_NEGATIVE_CONTROL):
            raise InputError(
                f"{where}: 'kind' must be '{KIND_CURATED}' or '{KIND_NEGATIVE_CONTROL}'"
            )
        # The record must PROVE its space, not inherit the enrollment file's tag —
        # equal-dimensional vectors from another model must never slip through.
        record_space = _require_str(record, "embedding_space", where)
        if record_space != space:
            raise InputError(
                f"{where}: embedding_space {record_space!r} does not match "
                f"enrollment space {space!r}"
            )
        slots = _parse_agreement_slots(record, space, dims, where)
        total_speech = _optional_non_negative(record, "total_speech", where)

        if kind == KIND_CURATED:
            host_id = _require_str(record, "host_id", where)
            voiceprint = voiceprints.get(host_id)
            if voiceprint is None:
                raise InputError(f"{where}: unknown host_id {host_id!r}")
            enrollment_ok, reason = _enrollment_gate(voiceprint, item_id, thresholds)
            result = agreement.label_positive(
                slots,
                voiceprint.vector,
                thresholds,
                enrollment_ok=enrollment_ok,
                total_speech=total_speech,
                enrollment_reason=reason,
            )
        else:
            usable = {
                host_id: vp.vector
                for host_id, vp in voiceprints.items()
                if _enrollment_gate(vp, item_id, thresholds)[0]
            }
            result = agreement.label_negative_control(
                slots, usable, thresholds, total_speech=total_speech
            )

        payload = dataclasses.asdict(result)
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "item_id": item_id,
                "kind": kind,
                "embedding_space": space,
            }
        )
        lines.append(_dumps(payload))

    if not lines:
        raise InputError(f"{args.slots}: no records")
    _write_atomic(Path(args.out) if args.out else None, "\n".join(lines) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
def _load_verdicts(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load an agreement-output JSONL file into ``(space, {item_id: record})``."""
    space: str | None = None
    records: dict[str, dict[str, Any]] = {}
    for lineno, record in _read_jsonl(path):
        where = f"{path}:{lineno}"
        _check_schema_version(record, where)
        item_id = _require_str(record, "item_id", where)
        if item_id in records:
            raise InputError(f"{where}: duplicate item_id {item_id!r}")
        record_space = _require_str(record, "embedding_space", where)
        if space is None:
            space = record_space
        elif record_space != space:
            raise InputError(
                f"{where}: mixed embedding spaces in one voter file "
                f"({space!r} vs {record_space!r})"
            )
        verdict = _require_str(record, "verdict", where)
        if verdict not in (
            agreement.CONFIDENT_HOST_PRESENT,
            agreement.NO_CURATED_HOST_DETECTED,
            agreement.ABSTAIN,
        ):
            raise InputError(f"{where}: unknown verdict {verdict!r}")
        kind = _require_str(record, "kind", where)
        if kind not in (KIND_CURATED, KIND_NEGATIVE_CONTROL):
            raise InputError(
                f"{where}: 'kind' must be '{KIND_CURATED}' or '{KIND_NEGATIVE_CONTROL}'"
            )
        # Verdict/kind consistency: a positive only makes sense on a curated
        # item, a silver absence only on a negative control.
        if verdict == agreement.CONFIDENT_HOST_PRESENT and kind != KIND_CURATED:
            raise InputError(f"{where}: {verdict} is only valid on a curated item")
        if verdict == agreement.NO_CURATED_HOST_DETECTED and kind != KIND_NEGATIVE_CONTROL:
            raise InputError(f"{where}: {verdict} is only valid on a negative control")
        # A confident-present without its winning slot cannot be fused — two such
        # records would agree on host_slot None and mint a false silver label.
        if verdict == agreement.CONFIDENT_HOST_PRESENT:
            _require_str(record, "host_slot", where)
        contradiction = record.get("contradiction", False)
        if not isinstance(contradiction, bool):
            raise InputError(f"{where}: 'contradiction' must be a boolean")
        reason = record.get("reason")
        if reason is not None and reason not in agreement.ABSTAIN_REASONS:
            raise InputError(f"{where}: unknown reason {reason!r}")
        records[item_id] = record
    if space is None:
        raise InputError(f"{path}: no records")
    return space, records


def _label_result_from(record: Mapping[str, Any]) -> agreement.LabelResult:
    return agreement.LabelResult(
        verdict=str(record["verdict"]),
        reason=record.get("reason") if isinstance(record.get("reason"), str) else None,
        host_slot=(
            record.get("host_slot") if isinstance(record.get("host_slot"), str) else None
        ),
        contradiction=bool(record.get("contradiction", False)),
    )


def cmd_ensemble(args: argparse.Namespace) -> int:
    space_a, voter_a = _load_verdicts(Path(args.voter_a))
    space_b, voter_b = _load_verdicts(Path(args.voter_b))
    if space_a == space_b:
        raise InputError(
            f"both voters are in embedding space {space_a!r} — the ensemble "
            "AND-gate requires two independent models"
        )
    if set(voter_a) != set(voter_b):
        only_a = sorted(set(voter_a) - set(voter_b))[:5]
        only_b = sorted(set(voter_b) - set(voter_a))[:5]
        raise InputError(
            f"voter files cover different item_ids (a-only: {only_a}, b-only: {only_b})"
        )

    lines: list[str] = []
    for item_id in sorted(voter_a):
        rec_a, rec_b = voter_a[item_id], voter_b[item_id]
        if rec_a["kind"] != rec_b["kind"]:
            raise InputError(
                f"item {item_id!r}: voters disagree on kind "
                f"({rec_a['kind']!r} vs {rec_b['kind']!r})"
            )
        la, lb = _label_result_from(rec_a), _label_result_from(rec_b)
        if rec_a["kind"] == KIND_CURATED:
            decision = ensemble.combine_curated(
                la, lb, voter_a_name=space_a, voter_b_name=space_b
            )
        else:
            decision = ensemble.combine_neg_control(la, lb)
        lines.append(
            _dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "item_id": item_id,
                    "kind": rec_a["kind"],
                    "verdict": decision.verdict,
                    "reason": decision.reason,
                    "agreement": decision.agreement,
                    "voter_a": {"embedding_space": space_a, "verdict": la.verdict},
                    "voter_b": {"embedding_space": space_b, "verdict": lb.verdict},
                }
            )
        )

    _write_atomic(Path(args.out) if args.out else None, "\n".join(lines) + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Parser registration (called lazily from voxint.cli)
# --------------------------------------------------------------------------- #
def register(subparsers: Any) -> None:
    """Attach the ``score`` command group to the top-level CLI subparsers."""
    score_p = subparsers.add_parser(
        "score",
        help="offline speaker-attribution scoring harness (file-based, no DB)",
        description=(
            "Score speaker-attribution quality over curated golden vectors: "
            "name accuracy, acoustic agreement, and ensemble fusion. "
            "ASR accuracy / WER measurement is out of scope."
        ),
    )
    score_sub = score_p.add_subparsers(dest="score_command", required=True)

    na = score_sub.add_parser(
        "name-accuracy", help="score assigned speaker names against ground truth"
    )
    na.add_argument("input", help="items JSONL (see docs/harness.md)")
    na.add_argument("--baseline", help="baseline items JSONL for paired comparison")
    na.add_argument("--aliases", help="aliases JSON (canonical -> [alternates])")
    na.add_argument("--out", help="report path (default: stdout)")
    na.add_argument(
        "--target-accuracy",
        type=float,
        help="Chow-point target for the risk-coverage curve",
    )
    na.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    na.set_defaults(fn=_run(cmd_name_accuracy))

    ag = score_sub.add_parser(
        "agreement", help="acoustic agreement verdicts for one embedding voter"
    )
    ag.add_argument("--slots", required=True, help="per-item slot embeddings JSONL")
    ag.add_argument("--enrollment", required=True, help="voiceprint enrollment JSON")
    ag.add_argument("--thresholds", required=True, help="decision thresholds JSON")
    ag.add_argument("--out", help="verdicts JSONL path (default: stdout)")
    ag.set_defaults(fn=_run(cmd_agreement))

    en = score_sub.add_parser(
        "ensemble", help="fuse two voters' agreement verdicts (verdict-level only)"
    )
    en.add_argument("voter_a", help="first voter's agreement verdicts JSONL")
    en.add_argument("voter_b", help="second voter's agreement verdicts JSONL")
    en.add_argument("--out", help="fused verdicts JSONL path (default: stdout)")
    en.set_defaults(fn=_run(cmd_ensemble))


def _run(handler: Any) -> Any:
    """Wrap a command handler: InputError -> stderr diagnostic + exit code 2."""

    def wrapped(args: argparse.Namespace) -> int:
        try:
            result: int = handler(args)
        except InputError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return result

    return wrapped
