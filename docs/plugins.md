# Plugin author guide

Voxint plugins add optional features without coupling them to the pipeline or
shared console pages. This guide uses the built-in `synthdetect` plugin as the
worked example.

## 1. When to write a plugin

Use a plugin for a greenfield feature with a standalone surface. Apply one
boundary test: does a core page render the feature's state? If yes, keep the
feature in core.

Translation, LLM enrichment, and semantic search are native features.
Synthetic-speech detection (`synthdetect`) is a plugin. It owns a model service,
a post-completion job, and a standalone report. [ADR 0006](adr/0006-plugin-scope-native-vs-greenfield.md)
records this scope decision.

## 2. Plugin anatomy

Create the package at `src/voxint/plugins/<id>/`. A full plugin normally has:

```text
src/voxint/plugins/<id>/
├── __init__.py       # VoxintPlugin subclass and manifest
├── jobs.py           # job persistence and lifecycle
├── routes.py         # APIRouter builder
├── tasks.py          # Celery task entry points
├── client.py         # external service client, when needed
└── templates/        # exposed through the <id>/ Jinja namespace
    └── *.html
```

Keep the subclass small. Delegate route, task, job, and client logic to their
modules. Use [`synthdetect`](../src/voxint/plugins/synthdetect/) as the reference
implementation. Its templates live in `templates/` and are addressed as
`synthdetect/<template>.html`.

## 3. The contract

Subclass `VoxintPlugin` and define a class-level `PluginManifest`. The manifest
contains:

- `id`: matches `^[a-z][a-z0-9_]{1,31}$`.
- `name`: non-empty display name.
- `description`: short operator-facing description.
- `settings_prefixes`: prefixes used by config-parity contract tests.
- `task_names`: Celery names owned by the plugin. Greenfield names start with
  `voxint.plugin.<id>.`.

[`base.py`](../src/voxint/plugins/base.py) is the canonical API reference. It
defines the hook types, contribution records, and lifecycle rules. Every hook
has a safe no-op default.

| Hook | Signature | Default | Override when |
|---|---|---|---|
| `enabled` | `(row, settings) -> bool` | `False` | The plugin has an effective feature gate. `synthdetect` calls `resolve_effective_synthdetect_enabled`. |
| `invariant_errors` | `(row, settings) -> list[str]` | `[]` | Flags have cross-field constraints. `synthdetect` requires its base flag before autogeneration. |
| `settings_section` | `() -> SettingsSection \| None` | `None` | The settings UI needs a plugin section. `synthdetect` returns its template and order. |
| `run_detail_panels` | `() -> Sequence[PanelContribution]` | `()` | The run detail page needs a plugin-owned fragment. `synthdetect` contributes its result panel. |
| `run_detail_context` | `(run_id, session, settings) -> dict[str, Any]` | `{}` | A panel needs plugin-prefixed context. `synthdetect` loads the latest useful job. |
| `build_router` | `(deps) -> APIRouter \| None` | `None` | The plugin owns HTTP routes. `synthdetect` delegates to `build_synthdetect_router`. |
| `task_modules` | `() -> Sequence[str]` | `()` | Celery must import plugin task modules. `synthdetect` returns its `tasks` module. |
| `task_routes` | `() -> Mapping[str, Mapping[str, str]]` | `{}` | Plugin tasks need `post` queue routes. `synthdetect` routes its scoring task. |
| `on_run_completed` | `(event) -> None` | no-op | Completed runs should enqueue idempotent work. `synthdetect` creates and dispatches missing jobs. |
| `job_lanes` | `() -> Sequence[JobLaneSpec]` | `()` | The recovery sweep must redispatch stale queued work. `synthdetect` exposes its lookup and task name. |
| `add_cli_commands` | `(subparsers) -> None` | no-op | The plugin provides top-level CLI commands. `synthdetect` registers its commands. |

Plugins import core. Core imports concrete plugin classes only in
`src/voxint/plugins/discover.py`. Code under `src/voxint/pipeline/` never imports
`voxint.plugins`.

## 4. Registration and gates

Import the plugin class in `src/voxint/plugins/discover.py` and append it to the
`BUILTIN` tuple. Registration is static. The registry validates every built-in,
then sorts active plugins by manifest id.

Feature settings use three layers:

1. Add the environment default to the Pydantic `Settings` model.
2. Add a nullable override column to `AppSettings`. `NULL` means inherit.
3. Add `resolve_effective_*` functions that select the DB override when present
   and otherwise use the environment value.

Call the resolver from `enabled()` and re-check it at execution time. The base
implementation returns `False`, so an omitted gate fails closed.

Operators can disable registered plugins with the comma-separated
`VOXINT_PLUGINS_DISABLED` environment variable. The kill switch removes matching
ids from the active registry. `voxint doctor` reports unknown ids.

## 5. Routes, templates, and CLI

`build_router()` receives `PluginRouteDeps`. It contains four fields:
`templates`, `get_session`, `verify_csrf`, and `render_settings_page`. Use these
dependencies instead of importing application construction internals. Mutation
routes for installation-wide state must also depend on core `AdminDep`.

Do not add `from __future__ import annotations` to a routes module when handlers
inside a `build_router` closure use `Depends()`. FastAPI must evaluate those
annotations eagerly to see the dependency expressions. The plugin class and
other modules can use deferred annotations.

Store templates in `src/voxint/plugins/<id>/templates/`. Refer to them as
`<id>/template.html`. The application adds each active plugin directory to a
prefix loader.

Override `add_cli_commands(subparsers)` to add top-level `voxint` subcommands.
Keep command imports inside the hook when they pull in optional runtime code.

## 6. Jobs and media

Use a dedicated job lane for asynchronous work. Claim each job with one compare
and swap update:

```sql
UPDATE <id>_jobs
SET status = 'running'
WHERE id = :id AND status = 'queued' AND cancel_requested = FALSE
```

Return without work when the claim affects no row. Expose stale queued ids with
a `JobLaneSpec` so the recovery sweep can redispatch them. Make creation and
post-completion enqueue paths idempotent.

Fence cancellation at every terminal write. A success update must require the
job to remain `running` with `cancel_requested = FALSE`. Failure and cancellation
updates must use guarded terminal compare and swap logic so a late worker cannot
overwrite a winner.

Audio-consuming plugins obtain a confined descriptor with
`run_audio_descriptor()` from `voxint.plugins.media`. Handle its fail-closed
errors and the descriptor's reclaimed state. They must also add a
`NOT EXISTS` predicate for their active jobs to
`media/reclaim.py._select_eligible()`. This prevents garbage collection while a
queued or running job still needs the audio.

## 7. Database and compose

Namespace plugin tables with the id, such as `<id>_jobs` and `<id>_scores`.
Mirror every `Settings` field selected by `settings_prefixes` on `AppSettings`.
Use nullable columns for settings that support DB inheritance.

Add schema changes to the main Alembic chain. Do not create a plugin-specific
migration history.

For a companion service, add two overlays:

- `compose.plugin-<id>.yaml` defines the service and wires its environment into
  the API and worker.
- `compose.plugin-<id>.build.yaml` overrides the image with a source build.

Layer a release overlay after the base and hardware files:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml \
  -f compose.plugin-<id>.yaml up -d
```

See `compose.plugin-synthdetect.yaml` and
`compose.plugin-synthdetect.build.yaml` for a complete pair.

## 8. Testing and shipping

`tests/contracts/test_plugin_framework.py` enforces import direction, the kill
switch, manifest and settings parity, task naming, and the fail-closed gate.
`tests/contracts/test_settings_plugin_parity.py` covers plugin list and detail
rendering plus settings-section contribution. Add plugin-specific unit and
integration coverage for gates, routes, jobs, cancellation, and recovery.

Routes are a pinned contract. Update these goldens when routes change:

- `tests/contracts/fixtures/console2_route_order.json`
- `tests/contracts/fixtures/console2_route_characterization.json`
- `tests/contracts/fixtures/route_inventory.json`

Ship in this order:

1. Create `src/voxint/plugins/<id>/` and its tests.
2. Add the class to `discover.py` `BUILTIN`.
3. Add the main-chain Alembic migration.
4. Add settings fields and document environment variables in `.env.example`.
5. Add release and build compose overlays when the plugin has a service.
6. Update contract tests and route inventory goldens.
7. Add the user-visible change to `CHANGELOG.md`.
