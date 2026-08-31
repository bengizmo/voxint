"""The print/PDF evidence pack page (issue #331 Phase 7).

A server-rendered, print-optimized page of a run's highlights with their
provenance; the browser's Print / Save as PDF is the PDF engine (no PDF
dependency, by doctrine). The contract pinned here that DIFFERS from the
exports: a STALE highlight renders with a visible warning instead of failing
the document with a 409 — an evidence pack degrades honestly rather than
refusing to print.

Reuses the seeding helpers from ``test_export_manifest``.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session, sessionmaker

from tests.integration.test_export_manifest import (
    CREDS,
    _extract_clip,
    _make_tag,
    _seed_run,
    _seed_stage_runs,
    _word_annotation,
    client,
    media_root,
)
from voxint.db.models import AnnotationTagLink, TranscriptSegment

__all__ = ["CREDS", "client", "media_root"]


def test_evidence_pack_renders_quotes_and_provenance(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
) -> None:
    with session_factory() as session:
        run_id = _seed_run(session, media_root, media_sha256="ab" * 32)
        _seed_stage_runs(session, run_id)
        ann_id = _word_annotation(session, run_id)
    clip_id = _extract_clip(client, run_id, ann_id)

    resp = client.get(f"/review/{run_id}/annotations/evidence-pack")
    assert resp.status_code == 200
    html = resp.text
    assert "Evidence pack" in html
    assert "world" in html  # the quote text
    assert str(ann_id) in html
    assert "ab" * 32 in html  # source media sha256
    assert "Source text hash" in html
    assert "large-v2" in html  # stage provenance
    assert f"voxint-{run_id.hex[:8]}-clip-{uuid.UUID(clip_id).hex[:8]}.wav" in html
    assert "window.print()" in html
    assert "re-anchor before citing" not in html


def test_evidence_pack_stale_renders_flagged_not_409(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
) -> None:
    """The deliberate divergence from the exports: stale renders with a
    warning and the captured copy, never a 409 and never a silent omission."""
    with session_factory() as session:
        run_id = _seed_run(session, media_root)
        stale_ann = _word_annotation(session, run_id, seg_index=0)
        live_ann = _word_annotation(session, run_id, seg_index=1)
        session.execute(
            update(TranscriptSegment)
            .where(
                TranscriptSegment.pipeline_run_id == run_id,
                TranscriptSegment.segment_index == 0,
            )
            .values(raw_text="MUTATED TEXT HERE!")
        )
        session.commit()

    resp = client.get(f"/review/{run_id}/annotations/evidence-pack")
    assert resp.status_code == 200
    html = resp.text
    assert "re-anchor before citing" in html
    assert str(stale_ann) in html  # flagged, not omitted
    assert str(live_ann) in html
    assert "world" in html  # the stale card still shows the captured copy


def test_evidence_pack_tag_filter(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
) -> None:
    with session_factory() as session:
        run_id = _seed_run(session, media_root)
        tagged = _word_annotation(session, run_id, seg_index=0)
        untagged = _word_annotation(session, run_id, seg_index=1)
    tag_id = _make_tag(client, "Lead")
    with session_factory() as session:
        session.add(AnnotationTagLink(annotation_id=tagged, tag_id=tag_id))
        session.commit()

    resp = client.get(f"/review/{run_id}/annotations/evidence-pack?tag={tag_id}")
    assert resp.status_code == 200
    assert str(tagged) in resp.text
    assert str(untagged) not in resp.text
    assert "Tag-filtered subset" in resp.text

    unknown = client.get(
        f"/review/{run_id}/annotations/evidence-pack?tag={uuid.uuid4()}"
    )
    assert unknown.status_code == 404


def test_evidence_pack_empty_and_unknown_run(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
) -> None:
    with session_factory() as session:
        run_id = _seed_run(session, media_root)
    resp = client.get(f"/review/{run_id}/annotations/evidence-pack")
    assert resp.status_code == 200
    assert "No highlights on this run" in resp.text

    missing = client.get(f"/review/{uuid.uuid4()}/annotations/evidence-pack")
    assert missing.status_code == 404


def test_evidence_pack_requires_auth(
    client: TestClient,
    session_factory: sessionmaker[Session],
    media_root: Path,
) -> None:
    with session_factory() as session:
        run_id = _seed_run(session, media_root)
    resp = client.get(
        f"/review/{run_id}/annotations/evidence-pack", auth=("wrong", "creds")
    )
    assert resp.status_code == 401
