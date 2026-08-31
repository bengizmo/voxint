"""Explore: corpus-wide evidence browser (issue #331, #333)."""

from __future__ import annotations

import csv
import dataclasses
import io
import math
import uuid
from collections.abc import Callable
from datetime import date
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from voxint.api.csrf import CSRF_QUOTE_SAVE, mint_csrf_token
from voxint.api.explore_query import KWICFilters, KWICRow, corpus_stats, kwic_search, term_stats
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    get_session_factory,
    require_onboarded,
    templates,
)
from voxint.api.semantic_layout import semantic_layout
from voxint.api.similar_query import similar_passages
from voxint.db.models import Project, Speaker

explore_router = APIRouter(dependencies=[Depends(require_onboarded)])
router = explore_router

_PAGE_SIZE = 50


def _filters(
    *,
    q: str,
    project: uuid.UUID | None,
    speaker: uuid.UUID | None,
    date_from: date | None,
    date_to: date | None,
    confidence: bool,
    suspect: bool,
) -> KWICFilters:
    return KWICFilters(
        query=q,
        project_id=project,
        speaker_id=speaker,
        date_from=date_from,
        date_to=date_to,
        low_confidence_only=confidence,
        suspect_only=suspect,
    )


def _row_to_dict(row: KWICRow) -> dict[str, Any]:
    return {
        "left_context": row.left_context,
        "hit": row.hit,
        "right_context": row.right_context,
        "speaker_name": row.speaker_name,
        "speaker_id": str(row.speaker_id) if row.speaker_id else None,
        "run_id": str(row.run_id),
        "media_title": row.media_title,
        "segment_id": str(row.segment_id),
        "start_seconds": row.start_seconds,
        "confidence": row.confidence,
        "suspect": row.suspect,
    }


def _filter_projects(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(select(Project.id, Project.name).order_by(Project.name)).all()
    return [{"id": str(r.id), "name": r.name} for r in rows]


def _filter_speakers(session: Session) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Speaker.id, Speaker.display_name)
        .where(Speaker.merged_into_id.is_(None), Speaker.deleted_at.is_(None))
        .order_by(Speaker.display_name)
    ).all()
    return [{"id": str(r.id), "display_name": r.display_name} for r in rows]


def _pagination_qs_factory(request: Request) -> Callable[[int], str]:
    """Return a callable that builds a pagination query string preserving filters."""
    base_params = dict(request.query_params)

    def pagination_qs(target_page: int) -> str:
        params = dict(base_params)
        params["page"] = str(target_page)
        return urlencode(params)

    return pagination_qs


@explore_router.get("/explore", name="explore")
def explore(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    q: str = "",
    project: uuid.UUID | None = None,
    speaker: uuid.UUID | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    confidence: bool = False,
    suspect: bool = False,
    page: int = Query(default=1, ge=1),
) -> Response:
    filters = _filters(
        q=q,
        project=project,
        speaker=speaker,
        date_from=date_from,
        date_to=date_to,
        confidence=confidence,
        suspect=suspect,
    )
    results = kwic_search(
        session,
        filters,
        limit=_PAGE_SIZE,
        offset=(page - 1) * _PAGE_SIZE,
    )
    stats = corpus_stats(session, project)
    ts = term_stats(session, project)
    explore_props = {
        "rows": [_row_to_dict(r) for r in results.rows],
        "total": results.total,
        "query": results.query,
        "page": page,
        "pageSize": _PAGE_SIZE,
        "stats": dataclasses.asdict(stats),
        "filters": {
            "project_id": str(filters.project_id) if filters.project_id else None,
            "speaker_id": str(filters.speaker_id) if filters.speaker_id else None,
            "date_from": filters.date_from.isoformat() if filters.date_from else None,
            "date_to": filters.date_to.isoformat() if filters.date_to else None,
            "low_confidence_only": filters.low_confidence_only,
            "suspect_only": filters.suspect_only,
        },
        "termStats": ts.terms[:200],
        "csrfQuoteSave": mint_csrf_token(
            request.app.state.csrf_secret, CSRF_QUOTE_SAVE,
        ),
    }
    context = {
        "request": request,
        "active_nav": "explore",
        "query": results.query,
        "total": results.total,
        "rows": results.rows,
        "stats": stats,
        "filters": filters,
        "page": page,
        "page_size": _PAGE_SIZE,
        "total_pages": max(1, math.ceil(results.total / _PAGE_SIZE)),
        "explore_props": explore_props,
        "projects": _filter_projects(session),
        "speakers": _filter_speakers(session),
        "pagination_qs": _pagination_qs_factory(request),
    }
    return templates.TemplateResponse(request, "explore/explore.html", context)


@explore_router.get("/explore/csv", name="explore_csv")
def explore_csv(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    q: str = "",
    project: uuid.UUID | None = None,
    speaker: uuid.UUID | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    confidence: bool = False,
    suspect: bool = False,
) -> StreamingResponse:
    filters = _filters(
        q=q,
        project=project,
        speaker=speaker,
        date_from=date_from,
        date_to=date_to,
        confidence=confidence,
        suspect=suspect,
    )
    def _csv_safe(value: str) -> str:
        if value and value[0] in ("=", "+", "-", "@"):
            return "'" + value
        return value

    results = kwic_search(session, filters, limit=500, offset=0)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        (
            "left_context",
            "hit",
            "right_context",
            "speaker_name",
            "run_id",
            "media_title",
            "start_seconds",
            "confidence",
            "suspect",
        )
    )
    for row in results.rows:
        writer.writerow(
            (
                _csv_safe(row.left_context),
                _csv_safe(row.hit),
                _csv_safe(row.right_context),
                _csv_safe(row.speaker_name or ""),
                row.run_id,
                _csv_safe(row.media_title),
                row.start_seconds,
                "" if row.confidence is None else f"{row.confidence:.3f}",
                row.suspect,
            )
        )
    headers = {"Content-Disposition": 'attachment; filename="voxint-explore.csv"'}
    return StreamingResponse(
        iter((output.getvalue(),)),
        media_type="text/csv",
        headers=headers,
    )


@explore_router.get("/explore/segments/{segment_id}/similar", name="explore_similar")
def explore_similar(
    request: Request,
    operator: OperatorDep,
    segment_id: uuid.UUID,
) -> JSONResponse:
    """Passages nearest this segment in the embedding space (#357).

    Read-only GET; the segment id is the only client input — text and run are
    resolved server-side. Uses the app session factory directly (not
    SessionDep) because the scan controls its own REPEATABLE READ snapshot.
    """
    page = similar_passages(
        get_session_factory(request),
        settings=request.app.state.settings,
        segment_id=segment_id,
    )
    return JSONResponse(
        {
            "state": page.state.value,
            "items": [
                {
                    "run_id": str(item.run_id),
                    "title": item.title,
                    "speaker_label": item.speaker_label,
                    "start_seconds": item.start_seconds,
                    "end_seconds": item.end_seconds,
                    "preview": item.preview,
                    "jump_url": item.jump_url,
                }
                for item in page.items
            ],
        }
    )


@explore_router.get("/explore/meaning-map", name="explore_meaning_map")
def explore_meaning_map(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project: uuid.UUID | None = None,
) -> JSONResponse:
    """The semantic meaning map for this scope, computed and cached on read (#357).

    This GET intentionally materializes derived cache (an advisory-locked,
    idempotent artifact write an attacker cannot influence) — not a pattern to
    copy for operator-owned state, which belongs on CSRF-checked POSTs.
    """
    result = semantic_layout(session, request.app.state.settings, project)
    return JSONResponse({"state": result.state, **result.payload})


@explore_router.get("/explore/terms", name="explore_terms")
def explore_terms(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project: uuid.UUID | None = None,
) -> JSONResponse:
    """Return precomputed term statistics as JSON."""
    ts = term_stats(session, project)
    return JSONResponse({"terms": ts.terms, "stale": ts.stale})
