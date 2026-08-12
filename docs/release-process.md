# Release process (maintainers)

How a Voxint release is cut. Development happens on the private origin; the
public mirror is `github.com/bengizmo/voxint` — releases are cut there,
because the image workflow only runs on GitHub (`release.yml` is guarded by
`github.server_url` and stays inert elsewhere).

## What a release produces

| Artifact | Where | How |
|---|---|---|
| `ghcr.io/bengizmo/voxint:X.Y.Z` (app: api/worker/beat/migrate) | GHCR | `release.yml` on the `vX.Y.Z` tag |
| `ghcr.io/bengizmo/voxint-{whisper,pyannote,titanet}:X.Y.Z` | GHCR | same workflow, matrix entries |
| `voxint X.Y.Z` sdist + wheel | PyPI | manual `uv build` + `uv publish` (not in CI) |
| Release notes | GitHub Releases | manual `gh release create` |

`docker/metadata-action` also emits a mutable `X.Y` tag for full releases;
compose files must keep referencing the exact `X.Y.Z`. Pre-release versions
(e.g. `v0.0.0-test`) get only their exact tag — useful for exercising the
workflow without polluting `X.Y`.

## Cutting a release

1. **Release commit** on `main`: bump the version in `pyproject.toml` AND
   `src/voxint/__init__.py`, and bump the `VOXINT_IMAGE_TAG` default pin in
   `compose.yaml` + `compose.gpu.yaml` (and its `.env.example` comment) to the
   new version — the default stack always runs the release this checkout
   documents. Run the gates (`ruff` / `mypy` / `pytest` with the pgvector test
   DB) and both gitleaks scans (`gitleaks dir .` and `gitleaks git .` with
   `.gitleaks.toml`).
2. Push to both remotes; wait for `ci` to go green on GitHub.
3. **Tag**: `git tag -a vX.Y.Z -m "Voxint vX.Y.Z" && git push github vX.Y.Z`
   (push the tag to the private origin too). The tag must point at the release
   commit so images are built from exactly what the compose files pin.
4. **Watch `release.yml`** (4 matrix jobs; whisper is the slow one, 25–45 min).
   `fail-fast` is off, so one failed matrix entry leaves the others published —
   a failure in `docker/metadata-action` *before* the build step has been
   transient GitHub infrastructure: re-run failed jobs
   (`gh api -X POST repos/…/actions/runs/<id>/rerun-failed-jobs`; this `gh`
   version's `run rerun` has no `--failed` flag).
5. **Verify anonymous pull** of all four images (no login):
   the packages inherit public visibility from the repo via the
   `org.opencontainers.image.source` label, but confirm — fetch each manifest
   with an anonymous GHCR token and expect 200. Optionally
   `docker run --rm ghcr.io/bengizmo/voxint:X.Y.Z python -c "import voxint; print(voxint.__version__)"`.
6. **PyPI**: `rm -rf dist && uv build && uv publish --token <pypi-token> dist/*`,
   then check `https://pypi.org/pypi/voxint/json` reports the new version.
7. **GitHub Release**: `gh release create vX.Y.Z --title "Voxint vX.Y.Z" --notes …`
   and update `CHANGELOG.md` in the next commit if it wasn't part of the
   release commit.
8. Optional: mirror the app image to Docker Hub
   (`docker tag … bengizmo/voxint:X.Y.Z && docker push …`). GHCR is canonical;
   the multi-GB GPU images are GHCR-only.

## Gotchas

- **Workflow smoke-testing**: a pre-release tag like `v0.0.0-test` runs the
  whole pipeline safely (no `X.Y` mutable tag). GHCR versions of public
  container packages can NOT be deleted through the REST API (422) — only via
  the web UI, so don't mint test tags casually.
- The compose pin means a release is *self-referential*: the images the tag
  builds are the ones the tagged compose files pull. Step 1's pin bump is what
  keeps that true — tagging without it ships compose files that run the
  previous release.
- PyPI publishing is deliberately manual (no long-lived token in CI). If that
  changes, prefer PyPI trusted publishing over a stored secret.
