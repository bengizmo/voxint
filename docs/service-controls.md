# Model service controls

Restart Transcriber (`whisper`), Voice separation (`pyannote`), and Voice identity
(`titanet`) from **Settings > Status** with the opt-in Docker overlay.

## Deployment support

| Deployment | Web restart controls |
|---|---|
| Docker (GPU, CPU, ROCm) | Supported with `compose.service-controls.yaml`. |
| Native macOS | Not yet implemented; controls are disabled and terminal hints are shown. |
| Metal hybrid (Docker core, native models) | Not supported; use the host terminal. The API container cannot control host launchd jobs. |

## Enable on Docker

Run commands from the repository root. On Linux, find the host Docker group GID:

```bash
getent group docker | cut -d: -f3
```

Set `DOCKER_GID` in `.env` to that number if it differs from `999`. The API runs
as the non-root `voxint` user; the overlay adds this supplementary group so it
can access the socket.

Include the service-controls overlay with your existing Compose files:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml \
  -f compose.service-controls.yaml up -d
```

For CPU or ROCm, replace `compose.gpu.yaml` with `compose.cpu.yaml` or
`compose.rocm.yaml`. Retain any other overlays your deployment uses.
The overlay mounts `/var/run/docker.sock` into the API and sets
`VOXINT_SERVICE_CONTROL=docker`. Open **Settings > Status** to restart a model
service. A restart interrupts requests using that service; allow time for its
model to load before submitting more work.

| Variable | Default | Purpose |
|---|---|---|
| `VOXINT_SERVICE_CONTROL` | Unset (disabled) | `docker` enables controls; `launchd` is not yet implemented. |
| `VOXINT_DOCKER_SOCKET` | `/var/run/docker.sock` | Socket path inside the API container. |
| `VOXINT_COMPOSE_PROJECT` | `voxint` | Compose project label used to locate model containers. |
| `DOCKER_GID` | `999` in the overlay | Host Docker group GID added to the API container. |

For a custom Compose project, set `VOXINT_COMPOSE_PROJECT` in `.env` to the
project name used by your stack. For a custom socket, adjust the overlay's bind
mount and set `VOXINT_DOCKER_SOCKET` to its container-side path; the variable
alone does not change the mount. Recreate the API with `up -d` after changing
these settings. To disable controls, remove the overlay and leave
`VOXINT_SERVICE_CONTROL` unset, then recreate the API.

## Security

The Docker socket grants the API host-equivalent power over Docker and its
containers. Enable this only when you trust everyone who can log in to the
console. Use a strong, unique console password and restrict console access.

## Manual restarts

Docker, using the same Compose files and project as your running stack:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml restart whisper
docker compose -f compose.yaml -f compose.gpu.yaml restart pyannote
docker compose -f compose.yaml -f compose.gpu.yaml restart titanet
```

Substitute your CPU or ROCm overlay as above. If your shell already selects the
stack through `COMPOSE_FILE`, `docker compose restart whisper` is sufficient.

For native macOS and Metal hybrid, run these on the Mac as the user who started
the services:

```bash
launchctl kickstart -k gui/$(id -u)/com.voxint.metal.whisper
launchctl kickstart -k gui/$(id -u)/com.voxint.metal.pyannote
launchctl kickstart -k gui/$(id -u)/com.voxint.metal.titanet
```

Both modes use the Metal launcher for model services. If the jobs are not
loaded, start them with `scripts/metal/voxint-metal.sh up`.

## Troubleshooting

- **Socket missing or permission denied:** Confirm the overlay is included and
  `DOCKER_GID` matches the socket's group. Check the API container with
  `docker compose exec api id` and
  `docker compose exec api ls -ln /var/run/docker.sock`. Recreate the API after
  correcting the group or mount; do not make the socket world-writable.
- **Container not found:** Check `VOXINT_COMPOSE_PROJECT` against the running
  stack's project name and run `docker compose ls`. Controls match both project
  and service labels and require exactly one matching container per service.
  Use `docker compose -f compose.yaml -f compose.gpu.yaml ps -a` to check that
  the model containers exist, substituting your deployment's files.
- **Service does not come back:** Inspect logs with
  `docker compose -f compose.yaml -f compose.gpu.yaml logs --tail=100 whisper`
  (substitute your files and service). Check for model-loading failures, memory
  exhaustion, or GPU errors. A successful restart request does not guarantee
  model readiness; check Settings > Status again after startup. Web controls
  require a running container; start a stopped one with the same Compose files
  and `up -d whisper`. On macOS, use
  `scripts/metal/voxint-metal.sh status` and
  `scripts/metal/voxint-metal.sh logs whisper`.

See [operations](operations.md) for deployment details,
[native macOS](native-macos-preview.md) for the native launcher, and the
[documentation index](README.md) for other references.
