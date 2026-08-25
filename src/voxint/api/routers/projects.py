"""The projects area (Console 2.0 P2b, #153): the project list and detail pages.

A project groups media folders and carries project-scoped configuration (the
vocabulary/corrections editors land in a later phase). These pages let an
operator create a project, see its folders and the speakers its runs resolve to,
and assign an unassigned folder to it. Reads live in
:mod:`voxint.api.projects_query`; the two mutating routes (create, assign) follow
the console's per-action CSRF idiom.

The routes are always registered so the console route inventory is stable across
the dark-ship flip; :func:`require_projects_enabled` 404s them until
``console_projects_enabled`` is on. Registering ``/projects`` is what flips
``app.state.projects_routed``, so the sidebar's Projects link appears only once
these pages exist and the flag is set (no base.html change).
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from sqlalchemy.exc import IntegrityError

from voxint.api.csrf import (
    CSRF_PROJECT_ASSIGN,
    CSRF_PROJECT_CORRECTIONS,
    CSRF_PROJECT_CREATE,
    CSRF_PROJECT_VOCAB,
    mint_csrf_token,
)
from voxint.api.projects_query import list_projects, project_detail
from voxint.api.routers.deps import (
    OperatorDep,
    SessionDep,
    _require_csrf,
    require_onboarded,
    require_projects_enabled,
    templates,
)
from voxint.api.setup_wizard import SetupValidationError, normalize_vocabulary
from voxint.db.models import MediaFolder, Project
from voxint.domain_packs.corrections import (
    MAX_MATCH_CHARS,
    MAX_REPLACEMENT_CHARS,
    MAX_RULES_PER_PACK,
    OperatorCorrectionError,
    normalize_operator_corrections,
)

router = APIRouter(
    dependencies=[Depends(require_onboarded), Depends(require_projects_enabled)]
)


def _list_context(
    request: Request, session: SessionDep, *, error: str | None = None, name: str = ""
) -> dict[str, Any]:
    return {
        "request": request,
        "active_nav": "projects",
        "now": datetime.now(UTC),
        "projects": list_projects(session),
        "csrf_create": mint_csrf_token(request.app.state.csrf_secret, CSRF_PROJECT_CREATE),
        "error": error,
        "name_value": name,
    }


@router.get("/projects", name="projects")
def projects_page(
    request: Request, operator: OperatorDep, session: SessionDep
) -> Response:
    return templates.TemplateResponse(
        request, "projects/projects.html", _list_context(request, session)
    )


@router.post("/projects")
def create_project(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    name: Annotated[str, Form(max_length=200)] = "",
    description: Annotated[str, Form(max_length=2000)] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_PROJECT_CREATE, csrf_token)
    cleaned = name.strip()
    if not cleaned:
        return templates.TemplateResponse(
            request,
            "projects/projects.html",
            _list_context(request, session, error="A project name is required.", name=name),
            status_code=400,
        )
    description_value = description.strip() or None
    project = Project(name=cleaned, description=description_value)
    session.add(project)
    try:
        session.commit()
    except IntegrityError:
        # The unique-name constraint: a friendly 409 re-render, not a 500.
        session.rollback()
        return templates.TemplateResponse(
            request,
            "projects/projects.html",
            _list_context(
                request,
                session,
                error=f"A project named “{cleaned}” already exists.",
                name=name,
            ),
            status_code=409,
        )
    return RedirectResponse(f"/projects/{project.id}", status_code=303)


def _detail_context(
    request: Request,
    session: SessionDep,
    detail: Any,
    *,
    error: str | None = None,
    assigned_id: uuid.UUID | None = None,
    vocabulary_error: str | None = None,
    vocabulary_submitted: str | None = None,
    corrections_error: str | None = None,
) -> dict[str, Any]:
    # The just-assigned folder, echoed as a confirmation banner that names the
    # supersede relationship when the folder carries a pack.
    assigned = None
    if assigned_id is not None:
        assigned = next((f for f in detail.folders if f.id == assigned_id), None)
    secret = request.app.state.csrf_secret
    corrections_token = mint_csrf_token(secret, CSRF_PROJECT_CORRECTIONS)
    # The corrections-editor island (issue #84) hydrates over the read-only
    # fallback; it submits the full ordered rule list to POST
    # /projects/{id}/corrections, which validates through the same #80 gate and,
    # unlike the folder/global layers, replaces (never unions) the project's set.
    corrections_props = {
        "rules": detail.corrections or [],
        "action": f"/projects/{detail.id}/corrections",
        "csrfToken": corrections_token,
        "limits": {
            "maxRules": MAX_RULES_PER_PACK,
            "maxMatchChars": MAX_MATCH_CHARS,
            "maxReplacementChars": MAX_REPLACEMENT_CHARS,
        },
    }
    # The vocabulary textarea body: the operator's own submitted text on a rejected
    # save (never lose their edit), else the project's stored terms.
    if vocabulary_submitted is not None:
        vocabulary_text = vocabulary_submitted
    else:
        vocabulary_text = "\n".join(detail.vocabulary) if detail.vocabulary else ""
    return {
        "request": request,
        "active_nav": "projects",
        "now": datetime.now(UTC),
        "detail": detail,
        "csrf_assign": mint_csrf_token(secret, CSRF_PROJECT_ASSIGN),
        "csrf_vocab": mint_csrf_token(secret, CSRF_PROJECT_VOCAB),
        "csrf_corrections": corrections_token,
        "corrections_props": corrections_props,
        "vocabulary_text": vocabulary_text,
        "vocabulary_error": vocabulary_error,
        "corrections_error": corrections_error,
        "error": error,
        "assigned": assigned,
    }


@router.get("/projects/{project_id}", name="project_detail")
def project_detail_page(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project_id: uuid.UUID,
    assigned: uuid.UUID | None = None,
) -> Response:
    detail = project_detail(session, project_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id}")
    return templates.TemplateResponse(
        request,
        "projects/project_detail.html",
        _detail_context(request, session, detail, assigned_id=assigned),
    )


@router.post("/projects/{project_id}/folders")
def assign_folder(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project_id: uuid.UUID,
    folder_id: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    _require_csrf(request, CSRF_PROJECT_ASSIGN, csrf_token)
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id}")

    def _reject(message: str) -> Response:
        detail = project_detail(session, project_id)
        return templates.TemplateResponse(
            request,
            "projects/project_detail.html",
            _detail_context(request, session, detail, error=message),
            status_code=400,
        )

    try:
        parsed = uuid.UUID(folder_id)
    except ValueError:
        return _reject("Choose a folder to assign.")
    folder = session.get(MediaFolder, parsed)
    if folder is None:
        return _reject("That folder no longer exists.")
    if folder.project_id is not None:
        # The assignable list only offers unassigned folders; a stale form could
        # still submit one that was assigned meanwhile. Refuse rather than
        # silently move it out of its current project.
        where = "this project" if folder.project_id == project_id else "another project"
        return _reject(f"That folder is already assigned to {where}.")

    folder.project_id = project_id
    session.commit()
    return RedirectResponse(
        f"/projects/{project_id}?assigned={folder.id}", status_code=303
    )


@router.post("/projects/{project_id}/vocabulary")
def set_project_vocabulary(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project_id: uuid.UUID,
    mode: Annotated[str, Form()] = "set",
    vocabulary: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Set or clear this project's vocabulary override (issue #153, ADR 0002).

    ``mode="inherit"`` writes NULL — the project inherits the folder pack / global
    baseline. ``mode="set"`` writes the normalized terms (through the SAME
    ``normalize_vocabulary`` gate the glossary uses), which may be an empty list:
    "set to none" is explicit and wins over the lower layers, distinct from
    inherit. On a bounds violation NOTHING is written and the page re-renders with
    the message and the operator's own submitted text.
    """
    _require_csrf(request, CSRF_PROJECT_VOCAB, csrf_token)
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id}")
    if mode == "inherit":
        project.vocabulary = None
        session.commit()
        return RedirectResponse(f"/projects/{project_id}", status_code=303)
    try:
        terms = normalize_vocabulary(vocabulary)
    except SetupValidationError as exc:
        detail = project_detail(session, project_id)
        return templates.TemplateResponse(
            request,
            "projects/project_detail.html",
            _detail_context(
                request,
                session,
                detail,
                vocabulary_error=str(exc),
                vocabulary_submitted=vocabulary,
            ),
            status_code=422,
        )
    project.vocabulary = terms
    session.commit()
    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@router.post("/projects/{project_id}/corrections")
def set_project_corrections(
    request: Request,
    operator: OperatorDep,
    session: SessionDep,
    project_id: uuid.UUID,
    mode: Annotated[str, Form()] = "set",
    rules: Annotated[str, Form()] = "",
    csrf_token: Annotated[str | None, Form()] = None,
) -> Response:
    """Set or clear this project's corrections override (issue #153, ADR 0002).

    ``mode="inherit"`` writes NULL (inherit the folder pack / global baseline);
    the plain reset form takes this path. ``mode="set"`` is the corrections-editor
    island's save: the full ordered rule list arrives as a JSON string in
    ``rules`` and is validated INTERNALLY through the #80 gate — NOT unioned
    against the default/folder pack, whose collisions are irrelevant to a layer
    the project replaces. An empty list is "explicitly none" and wins. On any
    violation NOTHING is written: the island (Accept: application/json) gets a 422
    with the message and offending row; a JS-off submit re-renders the page.
    """
    _require_csrf(request, CSRF_PROJECT_CORRECTIONS, csrf_token)
    wants_json = "application/json" in (request.headers.get("accept") or "")
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"no project {project_id}")

    def _reject(message: str, row: int | None) -> Response:
        if wants_json:
            return JSONResponse(
                {"ok": False, "error": message, "row": row}, status_code=422
            )
        detail = project_detail(session, project_id)
        return templates.TemplateResponse(
            request,
            "projects/project_detail.html",
            _detail_context(request, session, detail, corrections_error=message),
            status_code=422,
        )

    if mode == "inherit":
        project.corrections = None
        session.commit()
        if wants_json:
            return JSONResponse({"ok": True, "corrections": None})
        return RedirectResponse(f"/projects/{project_id}", status_code=303)
    try:
        raw_items = json.loads(rules) if rules else []
    except json.JSONDecodeError:
        return _reject("The corrections payload was not valid JSON.", None)
    try:
        normalized = normalize_operator_corrections(raw_items)
    except OperatorCorrectionError as exc:
        return _reject(exc.message, exc.row)
    project.corrections = normalized
    session.commit()
    if wants_json:
        return JSONResponse({"ok": True, "corrections": normalized})
    return RedirectResponse(f"/projects/{project_id}", status_code=303)
