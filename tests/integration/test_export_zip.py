"""Bundled quote ZIP export routes (issue #281).

The single and bulk ``export.zip`` routes compose the existing pull-quote
Markdown, provenance manifest, and clip machinery into one archive. The
contract here is BYTE-IDENTITY: the ``.md`` and ``.json`` members must equal
the standalone endpoints' responses (drift would make the bundle a second,
subtly different export). Error semantics mirror the standalone exports: 404
deleted/foreign, 409 stale (atomic in bulk), clip member omitted when no clip
exists, empty bulk = a ZIP holding only the empty bundle manifest.

Reuses the seeding helpers from ``test_export_manifest`` (same fixture DNA on
purpose: whatever seeds a manifest test seeds a zip test).
"""

from __future__ import annotations

import io
import uuid
import zipfile
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
from voxint.db.models import TranscriptSegment

__all__ = ["CREDS", "client", "media_root"]


def _zip_names(resp_content: bytes) -> list[str]:
    with zipfile.ZipFile(io.BytesIO(resp_content)) as archive:
        return sorted(archive.namelist())


def _zip_member(resp_content: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(resp_content)) as archive:
        return archive.read(name)


class TestSingleZip:
    def test_members_and_byte_identity(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        """The .md and .json members are byte-identical to the standalone
        endpoints, and the clip member carries the clip route's exact bytes."""
        with session_factory() as session:
            run_id = _seed_run(session, media_root, media_sha256="ab" * 32)
            _seed_stage_runs(session, run_id)
            ann_id = _word_annotation(session, run_id)
        clip_id = _extract_clip(client, run_id, ann_id)

        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        run8, ann_hex = run_id.hex[:8], ann_id.hex
        assert (
            resp.headers["content-disposition"]
            == f'attachment; filename="voxint-{run8}-quote-{ann_id.hex[:8]}.zip"'
        )
        md_name = f"voxint-{run8}-quote-{ann_hex}.md"
        json_name = f"voxint-{run8}-manifest-{ann_hex}.json"
        clip_name = f"voxint-{run8}-clip-{uuid.UUID(clip_id).hex[:8]}.wav"
        assert _zip_names(resp.content) == sorted([md_name, json_name, clip_name])

        md_standalone = client.get(f"/review/{run_id}/annotations/{ann_id}/export.md")
        assert _zip_member(resp.content, md_name) == md_standalone.content

        json_standalone = client.get(f"/review/{run_id}/annotations/{ann_id}/export.json")
        zip_manifest = _zip_member(resp.content, json_name)
        # exported_at differs per request; compare with it normalized out.
        import json as jsonlib

        a = jsonlib.loads(zip_manifest)
        b = jsonlib.loads(json_standalone.content)
        a.pop("exported_at", None)
        b.pop("exported_at", None)
        assert a == b

        clip_standalone = client.get(f"/runs/{run_id}/clips/{clip_id}")
        assert _zip_member(resp.content, clip_name) == clip_standalone.content

    def test_clip_absent_member_omitted(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.zip")
        assert resp.status_code == 200
        names = _zip_names(resp.content)
        assert len(names) == 2
        assert not any(n.endswith(".wav") for n in names)

    def test_unservable_clip_fails_closed(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        """A manifest-referenced clip whose file is gone fails the export with
        the clip service's honest status instead of shipping a bundle that
        silently lacks its audio."""
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
        _extract_clip(client, run_id, ann_id)
        for wav in (media_root / "artifacts" / str(run_id) / "clips").glob("*.wav"):
            wav.unlink()
        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.zip")
        assert resp.status_code == 404

    def test_stale_409(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
            session.execute(
                update(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
                .values(raw_text="MUTATED TEXT HERE!")
            )
            session.commit()
        resp = client.get(f"/review/{run_id}/annotations/{ann_id}/export.zip")
        assert resp.status_code == 409
        assert resp.headers.get("x-voxint-conflict") == "stale"

    def test_unknown_annotation_404(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        resp = client.get(f"/review/{run_id}/annotations/{uuid.uuid4()}/export.zip")
        assert resp.status_code == 404

    def test_requires_auth(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            ann_id = _word_annotation(session, run_id)
        resp = client.get(
            f"/review/{run_id}/annotations/{ann_id}/export.zip",
            auth=("wrong", "creds"),
        )
        assert resp.status_code == 401


class TestBulkZip:
    def test_members_and_bundle_identity(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            first = _word_annotation(session, run_id, seg_index=0)
            second = _word_annotation(session, run_id, seg_index=1)
        clip_id = _extract_clip(client, run_id, first)

        resp = client.get(f"/review/{run_id}/annotations/export.zip")
        assert resp.status_code == 200
        run8 = run_id.hex[:8]
        assert (
            resp.headers["content-disposition"]
            == f'attachment; filename="voxint-{run8}-quotes.zip"'
        )
        expected = sorted(
            [
                f"voxint-{run8}-quote-{first.hex}.md",
                f"voxint-{run8}-quote-{second.hex}.md",
                f"voxint-{run8}-manifests.json",
                f"voxint-{run8}-clip-{uuid.UUID(clip_id).hex[:8]}.wav",
            ]
        )
        assert _zip_names(resp.content) == expected

        # The per-annotation .md members equal the standalone single exports.
        for ann in (first, second):
            standalone = client.get(f"/review/{run_id}/annotations/{ann}/export.md")
            assert (
                _zip_member(resp.content, f"voxint-{run8}-quote-{ann.hex}.md")
                == standalone.content
            )

        # The bundle member equals the standalone bulk manifest (exported_at
        # normalized: it is stamped per request).
        import json as jsonlib

        bundle = jsonlib.loads(_zip_member(resp.content, f"voxint-{run8}-manifests.json"))
        standalone_bundle = client.get(f"/review/{run_id}/annotations/export.json").json()
        bundle.pop("exported_at", None)
        standalone_bundle.pop("exported_at", None)
        assert bundle == standalone_bundle

    def test_tag_filter(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            tagged = _word_annotation(session, run_id, seg_index=0)
            _word_annotation(session, run_id, seg_index=1)
        tag_id = _make_tag(client, "Key")
        # Link directly: the PATCH route needs a live review claim, and the
        # filter contract under test is the export's, not the tagging route's.
        from voxint.db.models import AnnotationTagLink

        with session_factory() as session:
            session.add(AnnotationTagLink(annotation_id=tagged, tag_id=tag_id))
            session.commit()
        resp = client.get(f"/review/{run_id}/annotations/export.zip?tag={tag_id}")
        assert resp.status_code == 200
        run8 = run_id.hex[:8]
        md_names = [n for n in _zip_names(resp.content) if n.endswith(".md")]
        assert md_names == [f"voxint-{run8}-quote-{tagged.hex}.md"]

    def test_unknown_filter_tag_404(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        resp = client.get(f"/review/{run_id}/annotations/export.zip?tag={uuid.uuid4()}")
        assert resp.status_code == 404

    def test_stale_aborts_atomically(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
            _word_annotation(session, run_id, seg_index=0)
            _word_annotation(session, run_id, seg_index=1)
            session.execute(
                update(TranscriptSegment)
                .where(
                    TranscriptSegment.pipeline_run_id == run_id,
                    TranscriptSegment.segment_index == 0,
                )
                .values(raw_text="MUTATED TEXT HERE!")
            )
            session.commit()
        resp = client.get(f"/review/{run_id}/annotations/export.zip")
        assert resp.status_code == 409
        assert resp.headers.get("x-voxint-conflict") == "stale"

    def test_empty_run_yields_empty_bundle_zip(
        self,
        client: TestClient,
        session_factory: sessionmaker[Session],
        media_root: Path,
    ) -> None:
        with session_factory() as session:
            run_id = _seed_run(session, media_root)
        resp = client.get(f"/review/{run_id}/annotations/export.zip")
        assert resp.status_code == 200
        run8 = run_id.hex[:8]
        names = _zip_names(resp.content)
        assert names == [f"voxint-{run8}-manifests.json"]
        import json as jsonlib

        bundle = jsonlib.loads(_zip_member(resp.content, names[0]))
        assert bundle == {
            "schema_version": 1,
            "kind": "quote_provenance_bundle",
            "quotes": [],
        }
