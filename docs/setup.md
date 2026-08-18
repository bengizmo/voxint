# Setup

Everything you need to get Voxint running on your operating system and hardware.
This page takes you from "nothing installed" to "the console is up"; the
in-browser first-run walkthrough that follows (setup wizard + guided tutorial)
is covered in [onboarding.md](onboarding.md), and running it day to day is in
[operations.md](operations.md).

If you just want the fast path, the [README quickstart](../README.md#quickstart)
is the two-command version. This page is the full reference.

## Before you start

Voxint is **self-hosted**: it runs as a small set of containers on a computer you
control (your laptop, a home server, a workstation). For the **standard install**
(every tier below except the native preview) the one hard requirement is:

- **[Docker](https://docs.docker.com/get-started/get-docker/) Engine with the
  Compose plugin, version ≥ 2.24.** Check with `docker compose version`. The old
  standalone `docker-compose` (v1) command cannot read this stack — you need the
  `docker compose` (two words) plugin.

Everything else — the database, the AI models, all their weights — is installed
for you. There is **no Hugging Face account or token** to create.

> Two Apple-Silicon paths need a little more: the **metal tier** additionally
> needs Homebrew and `uv` (see §3), and the separate docker-free
> **[native macOS preview](native-macos-preview.md)** has its own prerequisites.

## 1. Install Docker on your operating system

Follow the official Docker instructions for your platform, then come back here.

| Platform | Install guide | Notes |
|---|---|---|
| **Ubuntu / Debian** | [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) · [Debian](https://docs.docker.com/engine/install/debian/) | Install *Docker Engine* + the Compose plugin (not the old `docker-compose`). |
| **Fedora / RHEL** | [Fedora](https://docs.docker.com/engine/install/fedora/) · [RHEL](https://docs.docker.com/engine/install/rhel/) | Same: Docker Engine + Compose plugin. |
| **Arch Linux** | [ArchWiki: Docker](https://wiki.archlinux.org/title/Docker) | `docker` and `docker-compose` (the plugin) from the official repos. |
| **macOS** | [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) | Docker Desktop runs containers inside a VM. For the CPU tier, raise the VM's **memory limit** to ≥ 8 GB in Docker Desktop → Settings → Resources. Apple Silicon Macs can also run the faster **metal** tier — see below. |
| **Windows** | [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) | Use the **WSL 2** backend. Run the commands below from your WSL 2 Linux shell. |

On Linux you may also want to
[run Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/)
so you don't need `sudo` for every command.

## 2. Get Voxint and install it

### Guided install (recommended)

```bash
git clone https://github.com/bengizmo/voxint.git && cd voxint
./scripts/install.sh
```

The installer (a plain Bash script — nothing extra to install for the Docker
tiers; the metal tier adds Homebrew + `uv`, see §3) checks your Docker setup,
then asks for three things:

- an **admin password** for the console,
- a **media folder** for your recordings, and
- a **compute tier** — which hardware runs the models (it suggests the right one:
  NVIDIA GPU, AMD GPU, Apple Silicon, plain CPU, or "none for now").

It generates everything else, pulls the pinned release images, starts the stack,
waits until the API reports healthy, and prints the console URL. It is safe to
re-run — an existing configuration is kept (and backed up before any change), and
your tier choice is remembered. Then continue with the in-browser
[first-run walkthrough](onboarding.md).

### Manual install

If you'd rather set it up by hand:

```bash
cp .env.example .env          # then edit at least VOXINT_PASSWORD
mkdir -p media                # the media folder; pre-create so it isn't root-owned
docker compose pull           # prebuilt release images from GHCR
docker compose up -d          # database, Redis, one-shot migrate, console, worker, scheduler
```

A one-shot `migrate` step brings the database up to date before the console and
worker start — seeing it report `Exited (0)` in `docker compose ps -a` is
**success, not a crash**. If a default port is already taken on your machine,
override the published side in `.env` (`API_PORT`, `POSTGRES_PORT`,
`REDIS_PORT`). More detail: [operations.md](operations.md#deployment).

The commands above start the **core stack** (console + database + worker). To
actually transcribe anything you also need a **compute tier** for the model
services — pick yours below.

## 3. Choose your compute tier

The three model services (transcription, speaker separation, voice identity) run
in whichever tier fits your hardware. Layer the matching overlay on top of the
core stack. Per-service details and tunables live in each
`services/*/README.md`.

**Which one is mine?** Match your hardware to a compute tier:

| Your hardware | Compute tier | Overlay / guide |
|---|---|---|
| No GPU, or any Mac via Docker | **CPU** (the default) | `compose.cpu.yaml` — below |
| NVIDIA GPU | **NVIDIA** | `compose.gpu.yaml` — below |
| AMD GPU | **AMD / ROCm** | `compose.rocm.yaml` — below |
| Apple Silicon Mac (fastest on a Mac) | **metal** | `voxint-metal.sh` — below |

All four run the **core stack in Docker**. Separately, Apple-Silicon users who
can't or won't run Docker Desktop can use the docker-free
**[native preview](native-macos-preview.md)** — that's a *deployment mode*, not a
fifth compute tier (it still runs the metal model services under the hood).

### CPU — runs anywhere (the default)

No graphics card, no special drivers. Works on ordinary servers and Apple Silicon
Macs (via Docker Desktop).

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

(`up -d` pulls the images on first run — no separate `pull` step needed.)

Expect it to be **much slower** than a GPU — a long recording can take hours
rather than minutes — but the results are identical. **8 GB of memory is the
tight floor** (on Docker Desktop that's the *VM* memory limit, not your machine's
total) — below it the services are OOM-killed with an opaque exit, not a clear
message, so a long recording can fail with no diagnosis. **16 GB is comfortable.**
More: [operations.md](operations.md#running-without-an-nvidia-gpu-cpu-tier).

### NVIDIA GPU — the fast path

Needs an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

All three services share one GPU. Their loaded weights total roughly 3.5–4.5 GB
of VRAM (Whisper large-v2 ~1.5 GB, pyannote ~1–2 GB, TitaNet ~1 GB); budget about
**6–8 GB** in practice for decoding headroom. An **8 GB card is comfortable**.
Wire contracts: [docs/gpu-contracts.md](gpu-contracts.md).

### AMD GPU — ROCm tier

A hybrid tier for an AMD GPU: transcription runs on the GPU, while speaker
separation and voice identity use the CPU images. The host needs only the
**amdgpu kernel driver** — no ROCm install (the image carries its own).

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d
```

Some AMD consumer GPUs still hit a known convolution issue
([#4](https://github.com/bengizmo/voxint/issues/4)). Details:
[operations.md](operations.md#running-on-an-amd-gpu-rocm-tier).

### Apple Silicon Mac — metal tier

Docker Desktop can't pass the Apple GPU into a container, so on a Mac the metal
tier keeps the core stack in Docker but runs the model services **natively** so
speaker separation can use the Apple GPU. This tier needs **Docker Desktop**
specifically — Colima, OrbStack, and plain `dockerd` can't route the containers to
the native services (they break the `host.docker.internal` loopback). It also needs
**[Homebrew](https://brew.sh) and [`uv`](https://docs.astral.sh/uv/)**
(`brew install uv`) for the native model environments.

```bash
./scripts/install.sh                   # choose the [M]etal tier
./scripts/metal/voxint-metal.sh setup  # native environments + verified model weights
./scripts/metal/voxint-metal.sh up     # start the services
./scripts/metal/voxint-metal.sh status # confirm: whisper cpu / pyannote mps / titanet cpu
```

Weights come from the same verified release assets the images use — still no
Hugging Face account. Details:
[operations.md](operations.md#running-on-apple-silicon-metal-tier).

**Most Mac users want the metal tier.** There is also a docker-free
**[native preview](native-macos-preview.md)** that runs the *whole* stack without
Docker — choose it only if you can't or won't run Docker Desktop; it's a hands-on
technical preview, not the packaged release.

### Optional: bundled local LLM (no API key)

Voxint's transcript enhancement and run-asset summaries can call a language
model. You can point them at your own OpenAI-compatible endpoint (Settings →
LLM), **or** run the opt-in bundled model — a vendored, Apache-2.0
Qwen3-4B-Instruct served locally, so those features work with **no external
key**. It layers on top of *any* tier above:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.llm.yaml up -d
```

Then turn on **Settings → Features → "Use the bundled local model"**. It powers
**only** enhancement and run-asset summaries + entities — web research and LLM
speaker-name suggestions still need your own endpoint. On CPU it is slow for a
4B model, so a GPU is recommended (see the note in `compose.llm.yaml`). Details:
[operations.md](operations.md#bundled-local-llm-issue-67-optional-no-api-key).

## 4. Build from source (developers)

To run the code you checked out instead of the pinned release images, layer the
build overlays (build first, then start):

```bash
docker compose -f compose.yaml -f compose.build.yaml build api
docker compose -f compose.yaml -f compose.build.yaml up -d
```

Exactly one service owns each build overlay; see
[operations.md](operations.md#release-images-vs-building-from-source). For working
without Docker at all, see the [README's developer notes](../README.md#for-developers).

## 5. Verify it's running

```bash
curl http://127.0.0.1:8080/healthz     # 200 = ready (use your API_PORT if you changed it)
```

Then open **`http://127.0.0.1:8080/`** in a browser and sign in with the username
and password you set. A fresh install holds you at the **setup wizard** until
you finish it — that's expected. Continue with the
[first-run walkthrough](onboarding.md), which covers the wizard and the guided
tutorial.

Something not working? See
[Settings & troubleshooting](how-to/settings-and-troubleshooting.md) and the
troubleshooting notes in [onboarding.md](onboarding.md#troubleshooting).
