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

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.exc import IntegrityError

from voxint.api.csrf import (
    CSRF_PROJECT_ASSIGN,
    CSRF_PROJECT_CREATE,
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
from voxint.db.models import MediaFolder, Project

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
) -> dict[str, Any]:
    # The just-assigned folder, echoed as a confirmation banner that names the
    # supersede relationship when the folder carries a pack.
    assigned = None
    if assigned_id is not None:
        assigned = next((f for f in detail.folders if f.id == assigned_id), None)
    return {
        "request": request,
        "active_nav": "projects",
        "now": datetime.now(UTC),
        "detail": detail,
        "csrf_assign": mint_csrf_token(request.app.state.csrf_secret, CSRF_PROJECT_ASSIGN),
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
