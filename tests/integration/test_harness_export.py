"""DB -> harness exporter (issue #113 step 4) against real Postgres/pgvector.

Covers the two export families and the evidence snapshot, the fail-closed error
paths, merge canonicalization, the grounded-vs-resolution separation (a label a
human adjudicated still reports the machine's grounded prediction), and full
round trips of the emitted artifacts through the real ``voxint score`` CLI.
"""

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from voxint.cli import main
from voxint.config import Settings
from voxint.db.models import (
    EMBEDDING_DIM,
    AdjudicationDecision,
    Decision,
    DiarizationTurn,
    MediaItem,
    PipelineRun,
    Speaker,
    SpeakerEmbedding,
)
from voxint.harness_export import (
    ABSTAIN,
    NEITHER_DETERMINABLE,
    ExportError,
    TruthAnchoring,
    agreement_enrollment,
    agreement_slots,
    evidence_snapshot,
    name_accuracy_items,
)
from voxint.speakers.matching import (
    MatchingGates,
    evaluate_run,
    match_speakers,
    replace_run_match_candidates,
    replace_run_proposals,
    roster_centroids,
)

SPACE = "titanet-large-v2"
GATES = MatchingGates()


def unit(*components: tuple[int, float]) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for dim, value in components:
        vector[dim] = value
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


E0 = unit((0, 1.0))
E1 = unit((1, 1.0))
OFF = unit((0, 0.5), (5, math.sqrt(0.75)))  # cos to E0 = 0.5 < 0.60 -> rejected


@pytest.fixture()
def session(session_factory: sessionmaker[Session]):  # type: ignore[no-untyped-def]
    with session_factory() as s:
        yield s


def make_run(session: Session) -> uuid.UUID:
    media = MediaItem(source_path=f"incoming/{uuid.uuid4()}.wav")
    session.add(media)
    session.flush()
    run = PipelineRun(media_item_id=media.id)
    session.add(run)
    session.flush()
    return run.id


def add_turn(
    session: Session,
    run_id: uuid.UUID,
    index: int,
    label: str,
    start: float,
    end: float,
    embedding: list[float] | None,
    overlap_seconds: float = 0.0,
    space: str | None = SPACE,
) -> None:
    session.add(
        DiarizationTurn(
            pipeline_run_id=run_id,
            turn_index=index,
            start_seconds=start,
            end_seconds=end,
            label=label,
            overlap=overlap_seconds > 0,
            overlap_seconds=overlap_seconds,
            skip_reason=None if embedding is not None else "too_short",
            embedding=embedding,
            embedding_space=space if embedding is not None else None,
        )
    )


def add_speaker(
    session: Session,
    name: str,
    embeddings: list[list[float]],
    space: str = SPACE,
    source_runs: list[uuid.UUID | None] | None = None,
) -> uuid.UUID:
    speaker = Speaker(display_name=name)
    session.add(speaker)
    session.flush()
    sources = source_runs if source_runs is not None else [None] * len(embeddings)
    for embedding, source in zip(embeddings, sources, strict=True):
        session.add(
            SpeakerEmbedding(
                speaker_id=speaker.id,
                embedding_space=space,
                embedding=embedding,
                source_pipeline_run_id=source,
            )
        )
    return speaker.id


def add_decision(
    session: Session,
    run_id: uuid.UUID,
    label: str,
    decision: str,
    speaker_id: uuid.UUID | None = None,
) -> None:
    session.add(
        AdjudicationDecision(
            pipeline_run_id=run_id,
            diarization_label=label,
            decision=decision,
            speaker_id=speaker_id,
            operator="reviewer",
            idempotency_key=uuid.uuid4().hex,
            created_at=datetime.now(tz=UTC),
        )
    )


def run_matcher(session: Session, run_id: uuid.UUID) -> None:
    """Populate speaker_assignments + match_candidates as the stage does."""
    decisions = evaluate_run(session, run_id, GATES)
    replace_run_proposals(session, run_id, match_speakers(session, run_id, GATES), ())
    replace_run_match_candidates(session, run_id, decisions)
    session.flush()


def _grounded_run(session: Session, speaker_name: str = "Alice") -> tuple[uuid.UUID, uuid.UUID]:
    """A run whose single label grounds onto ``speaker_name`` (Alice=E0)."""
    run_id = make_run(session)
    alice = add_speaker(session, speaker_name, [E0])
    add_speaker(session, "Bob", [E1])
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    run_matcher(session, run_id)
    return run_id, alice


# ----------------------------------------------------------- name-accuracy


def test_name_accuracy_grounded_no_human_truth_is_excluded(session: Session) -> None:
    run_id, _alice = _grounded_run(session)
    items = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
    )
    assert len(items) == 1
    item = items[0]
    assert item["item_id"] == str(run_id)
    assert item["truth_anchoring"] == "independent"
    slot = item["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["assigned_name"] == "Alice"
    assert slot["truth"] is None  # no human ruling -> unscoreable
    assert slot["confidence"] is not None
    assert slot["match"]["decision"] == "accepted"
    assert slot["match"]["grounded"] is True
    assert slot["match"]["top_speaker_name"] == "Alice"


def test_grounded_prediction_survives_human_assign(session: Session) -> None:
    """The load-bearing separation: a human ASSIGN must NOT erase the machine's
    grounded prediction. assigned_name is read from the machine evidence, not the
    read-time resolution (which a human ruling would flip to HUMAN_ASSIGN)."""
    run_id, alice = _grounded_run(session)
    add_decision(session, run_id, "SPEAKER_00", Decision.ASSIGN.value, alice)
    session.flush()
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.POST_PROPOSAL
    )[0]["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["assigned_name"] == "Alice"  # not None
    assert slot["truth"] == "Alice"  # scores TP


def test_grounded_prediction_wrong_against_human_truth(session: Session) -> None:
    run_id, _alice = _grounded_run(session)
    carol = add_speaker(session, "Carol", [unit((7, 1.0))])
    add_decision(session, run_id, "SPEAKER_00", Decision.ASSIGN.value, carol)
    session.flush()
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.POST_PROPOSAL
    )[0]["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["assigned_name"] == "Alice"
    assert slot["truth"] == "Carol"  # scores FP_WRONG


def test_human_exclude_and_unknown_truth_sentinels(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    for i in range(3):
        add_turn(session, run_id, i, "EXC", i * 5.0, i * 5.0 + 4.0, OFF)  # rejected
        add_turn(session, run_id, i + 3, "UNK", 40 + i * 5.0, 44 + i * 5.0, OFF)
    session.flush()
    run_matcher(session, run_id)
    add_decision(session, run_id, "EXC", Decision.EXCLUDE.value)
    add_decision(session, run_id, "UNK", Decision.UNKNOWN.value)
    session.flush()
    slots = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
    )[0]["slots"]
    assert slots["EXC"]["truth"] == ABSTAIN  # type: ignore[index]
    assert slots["EXC"]["assigned_name"] is None  # type: ignore[index]
    assert slots["UNK"]["truth"] == NEITHER_DETERMINABLE  # type: ignore[index]


def test_ungrounded_proposal_reports_no_assigned_name(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    for i in range(2):  # accepted but not grounded (2 turns, 3.5s)
        add_turn(session, run_id, i, "SPEAKER_00", i * 4.0, i * 4.0 + 3.5, E0)
    session.flush()
    run_matcher(session, run_id)
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
    )[0]["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["assigned_name"] is None
    assert slot["confidence"] is None  # no name -> no confidence for risk-coverage
    assert slot["match"]["decision"] == "accepted"
    assert slot["match"]["grounded"] is False


def test_rejected_single_speaker_roster_margin_is_null(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])  # one-speaker roster -> margin undefined
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, OFF)
    session.flush()
    run_matcher(session, run_id)
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
    )[0]["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["assigned_name"] is None
    assert slot["match"]["decision"] == "rejected"
    assert slot["match"]["reason"] == "below_cosine"
    assert slot["match"]["margin"] is None  # never Infinity
    assert slot["match"]["top_speaker_name"] == "Alice"  # near-miss name resolved


def test_ineligible_label_provenance(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    add_turn(session, run_id, 0, "SKIP", 0.0, 1.0, None)  # skipped window
    session.flush()
    run_matcher(session, run_id)
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
    )[0]["slots"]["SKIP"]  # type: ignore[index]
    assert slot["match"]["decision"] == "ineligible"
    assert slot["match"]["similarity"] is None
    assert slot["match"]["top_speaker_id"] is None


def test_name_accuracy_canonicalizes_merged_speaker(session: Session) -> None:
    run_id, alice = _grounded_run(session)
    # Merge Alice into a new canonical speaker; the ledger keeps Alice's id.
    canonical = add_speaker(session, "Alice Canonical", [unit((9, 1.0))])
    add_decision(session, run_id, "SPEAKER_00", Decision.ASSIGN.value, alice)
    session.flush()
    alice_row = session.get(Speaker, alice)
    assert alice_row is not None
    alice_row.merged_into_id = canonical
    alice_row.merged_at = datetime.now(tz=UTC)
    session.flush()
    slot = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.POST_PROPOSAL
    )[0]["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert slot["truth"] == "Alice Canonical"


def test_name_accuracy_deterministic_and_sorted(session: Session) -> None:
    run_id = make_run(session)
    add_speaker(session, "Alice", [E0])
    for i in range(3):
        add_turn(session, run_id, i, "ZEBRA", i * 5.0, i * 5.0 + 4.0, E0)
        add_turn(session, run_id, i + 3, "ALPHA", 40 + i * 5.0, 44 + i * 5.0, E0)
    session.flush()
    run_matcher(session, run_id)
    first = name_accuracy_items(session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT)
    second = name_accuracy_items(session, [run_id], truth_anchoring=TruthAnchoring.INDEPENDENT)
    assert first == second
    assert sorted(cast(dict[str, Any], first[0]["slots"])) == ["ALPHA", "ZEBRA"]


def test_name_accuracy_rejects_bad_selection(session: Session) -> None:
    run_id, _ = _grounded_run(session)
    with pytest.raises(ExportError, match="duplicate run id"):
        name_accuracy_items(
            session, [run_id, run_id], truth_anchoring=TruthAnchoring.INDEPENDENT
        )
    with pytest.raises(ExportError, match="no runs selected"):
        name_accuracy_items(session, [], truth_anchoring=TruthAnchoring.INDEPENDENT)


def test_name_accuracy_round_trips_through_score_cli(
    session: Session, tmp_path: Path
) -> None:
    run_id, alice = _grounded_run(session)
    add_decision(session, run_id, "SPEAKER_00", Decision.ASSIGN.value, alice)
    session.flush()
    items = name_accuracy_items(
        session, [run_id], truth_anchoring=TruthAnchoring.POST_PROPOSAL
    )
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(
        "\n".join(json.dumps(rec, sort_keys=True) for rec in items) + "\n"
    )
    out_path = tmp_path / "report.json"
    assert main(["score", "name-accuracy", str(items_path), "--out", str(out_path)]) == 0
    report = json.loads(out_path.read_text())
    assert report["aggregate"]["tp"] == 1


# ----------------------------------------------------------- agreement enrollment


def test_enrollment_matches_production_centroid(session: Session) -> None:
    src = make_run(session)
    add_speaker(session, "Alice", [E0, E0], source_runs=[src, src])
    enroll = agreement_enrollment(session, SPACE)
    assert enroll["schema_version"] == 1
    assert enroll["embedding_space"] == SPACE
    assert enroll["dims"] == EMBEDDING_DIM
    (host_id, vp), = enroll["voiceprints"].items()  # type: ignore[attr-defined]
    centroid = roster_centroids(session, SPACE)[uuid.UUID(host_id)]
    assert np.allclose(vp["embedding"], centroid.tolist())
    assert vp["enrollment_items"] == 2
    assert vp["held_out"] is True
    assert vp["source_item_ids"] == [str(src)]


def test_enrollment_held_out_false_without_source(session: Session) -> None:
    add_speaker(session, "Alice", [E0], source_runs=[None])
    vp = next(iter(agreement_enrollment(session, SPACE)["voiceprints"].values()))  # type: ignore[attr-defined]
    assert vp["held_out"] is False
    assert vp["source_item_ids"] == []


def test_enrollment_zero_vector_row_not_counted(session: Session) -> None:
    src = make_run(session)
    zero = [0.0] * EMBEDDING_DIM
    add_speaker(session, "Alice", [E0, zero], source_runs=[src, src])
    vp = next(iter(agreement_enrollment(session, SPACE)["voiceprints"].values()))  # type: ignore[attr-defined]
    assert vp["enrollment_items"] == 1  # zero vector excluded from the centroid


def test_enrollment_filter_and_missing(session: Session) -> None:
    alice = add_speaker(session, "Alice", [E0])
    add_speaker(session, "Bob", [E1])
    only = agreement_enrollment(session, SPACE, roster_speaker_ids=[alice])
    assert sorted(cast(dict[str, Any], only["voiceprints"])) == [str(alice)]
    with pytest.raises(ExportError, match="no usable centroid"):
        agreement_enrollment(session, SPACE, roster_speaker_ids=[uuid.uuid4()])


def test_enrollment_empty_roster_errors(session: Session) -> None:
    with pytest.raises(ExportError, match="no active roster voiceprints"):
        agreement_enrollment(session, SPACE)


# ----------------------------------------------------------- agreement slots


def test_agreement_slots_curated(session: Session) -> None:
    run_id = make_run(session)
    for i in range(3):
        add_turn(session, run_id, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    host = uuid.uuid4()
    session.flush()
    records = agreement_slots(
        session,
        [run_id],
        kind_by_run={run_id: "curated"},
        host_by_run={run_id: host},
        gates=GATES,
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["item_id"] == str(run_id)
    assert rec["kind"] == "curated"
    assert rec["host_id"] == str(host)
    assert rec["embedding_space"] == SPACE
    slot = rec["slots"]["SPEAKER_00"]  # type: ignore[index]
    assert len(slot["embedding"]) == EMBEDDING_DIM
    assert slot["segments"] == 3
    assert slot["duration"] == pytest.approx(12.0)


def test_agreement_total_speech_is_interval_union(session: Session) -> None:
    run_id = make_run(session)
    # Overlapping turns: [0,10) and [5,15) union to 15s, not 20s.
    add_turn(session, run_id, 0, "A", 0.0, 10.0, E0)
    add_turn(session, run_id, 1, "B", 5.0, 15.0, E1)
    session.flush()
    rec = agreement_slots(
        session, [run_id], kind_by_run={run_id: "negative_control"}, gates=GATES
    )[0]
    assert rec["total_speech"] == pytest.approx(15.0)
    assert "host_id" not in rec  # negative controls carry no host


def test_agreement_slots_errors(session: Session) -> None:
    run_id = make_run(session)
    for i in range(3):
        add_turn(session, run_id, i, "S", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()
    with pytest.raises(ExportError, match="kind must be"):
        agreement_slots(session, [run_id], kind_by_run={run_id: "bogus"}, gates=GATES)
    with pytest.raises(ExportError, match="no host speaker id"):
        agreement_slots(session, [run_id], kind_by_run={run_id: "curated"}, gates=GATES)


def test_agreement_rejects_mixed_embedding_spaces(session: Session) -> None:
    run_id = make_run(session)
    add_turn(session, run_id, 0, "A", 0.0, 4.0, E0, space=SPACE)
    add_turn(session, run_id, 1, "B", 5.0, 9.0, E1, space="other-space")
    session.flush()
    with pytest.raises(ExportError, match="multiple embedding spaces"):
        agreement_slots(
            session, [run_id], kind_by_run={run_id: "negative_control"}, gates=GATES
        )


def test_agreement_no_usable_slots_errors(session: Session) -> None:
    run_id = make_run(session)
    add_turn(session, run_id, 0, "S", 0.0, 1.0, None)  # only a skipped window
    session.flush()
    with pytest.raises(ExportError, match="no embeddings in any embedding space"):
        agreement_slots(
            session, [run_id], kind_by_run={run_id: "negative_control"}, gates=GATES
        )


def test_agreement_round_trips_through_score_cli(
    session: Session, tmp_path: Path
) -> None:
    # Enroll Alice from run A; score run B (curated) -> present; and run A itself
    # -> the leakage gate must abstain because A is in Alice's source_item_ids.
    run_a = make_run(session)
    run_b = make_run(session)
    for i in range(4):
        add_turn(session, run_b, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    alice = add_speaker(session, "Alice", [E0, E0, E0], source_runs=[run_a, run_a, run_a])
    for i in range(4):  # give run A eligible speech too
        add_turn(session, run_a, i, "SPEAKER_00", i * 5.0, i * 5.0 + 4.0, E0)
    session.flush()

    enroll = agreement_enrollment(session, SPACE, roster_speaker_ids=[alice])
    slots = agreement_slots(
        session,
        [run_a, run_b],
        kind_by_run={run_a: "curated", run_b: "curated"},
        host_by_run={run_a: alice, run_b: alice},
        gates=GATES,
    )
    enroll_path = tmp_path / "enrollment.json"
    enroll_path.write_text(json.dumps(enroll))
    slots_path = tmp_path / "slots.jsonl"
    slots_path.write_text("\n".join(json.dumps(r) for r in slots) + "\n")
    thr_path = tmp_path / "thresholds.json"
    thr_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tau": 0.6,
                "margin": 0.05,
                "min_duration": 10.0,
                "min_segments": 3,
                "low_band": 0.3,
                "neg_min_total_duration": 5.0,
                "min_enrollment_items": 2,
            }
        )
    )
    out_path = tmp_path / "verdicts.jsonl"
    assert (
        main(
            [
                "score",
                "agreement",
                "--slots",
                str(slots_path),
                "--enrollment",
                str(enroll_path),
                "--thresholds",
                str(thr_path),
                "--out",
                str(out_path),
            ]
        )
        == 0
    )
    verdicts = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in out_path.read_text().splitlines()
    }
    assert verdicts[str(run_b)]["verdict"] == "CONFIDENT_HOST_PRESENT"
    assert verdicts[str(run_a)]["verdict"] == "ABSTAIN"
    assert verdicts[str(run_a)]["reason"] == "session_leakage_risk"


# ----------------------------------------------------------- evidence snapshot


def test_evidence_snapshot(session: Session) -> None:
    run_id, _alice = _grounded_run(session)
    settings = Settings()
    snap = evidence_snapshot(
        session,
        settings,
        [run_id],
        exported_at="2026-08-20T00:00:00Z",
        git_sha="deadbeef",
    )
    assert snap["kind"] == "match_evidence_snapshot"
    assert snap["exported_at"] == "2026-08-20T00:00:00Z"
    assert snap["code"]["git_sha"] == "deadbeef"  # type: ignore[index]
    assert snap["embedding"]["space"] == SPACE  # type: ignore[index]
    assert snap["embedding"]["dims"] == EMBEDDING_DIM  # type: ignore[index]
    assert snap["gates_at_export"]["min_cosine"] == settings.match_min_cosine  # type: ignore[index]
    assert snap["run_ids"] == [str(run_id)]
    assert len(cast(list[Any], snap["roster_digest_at_export"])) == 2  # Alice + Bob
    # Deterministic for fixed inputs.
    again = evidence_snapshot(
        session, settings, [run_id], exported_at="2026-08-20T00:00:00Z", git_sha="deadbeef"
    )
    assert snap == again
