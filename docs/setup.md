# Setup

*Getting Voxint running on your operating system and hardware, from nothing installed to the console being up.*

This page takes you from "nothing installed" to "the console is up". The
in-browser first-run walkthrough that follows (setup wizard and guided tutorial)
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
  standalone `docker-compose` (v1) command cannot read this stack; you need the
  `docker compose` (two words) plugin.

Everything else (the database, the AI models, all their weights) is installed
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
| **macOS** | [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/) | Docker Desktop runs containers inside a VM. For the CPU tier, raise the VM's **memory limit** to ≥ 8 GB in Docker Desktop → Settings → Resources. Apple Silicon Macs can also run the faster **metal** tier (see below). |
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

The installer (a plain Bash script, with nothing extra to install for the Docker
tiers; the metal tier adds Homebrew + `uv`, see §3) checks your Docker setup,
then asks for three things:

- an **admin password** for the console,
- a **media folder** for your recordings, and
- a **compute tier**, which hardware runs the models (it suggests the right one:
  NVIDIA GPU, AMD GPU, Apple Silicon, plain CPU, or "none for now").

It generates everything else, pulls the pinned release images, starts the stack,
waits until the API reports healthy, and prints the console URL. It is safe to
re-run: an existing configuration is kept (and backed up before any change), and
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
worker start; seeing it report `Exited (0)` in `docker compose ps -a` is
**success, not a crash**. If a default port is already taken on your machine,
override the published side in `.env` (`API_PORT`, `POSTGRES_PORT`,
`REDIS_PORT`). More detail: [operations.md](operations.md#deployment).

The commands above start the **core stack** (console + database + worker). To
actually transcribe anything you also need a **compute tier** for the model
services. Pick yours below.

## 3. Choose your compute tier

The three model services (transcription, speaker separation, voice identity) run
in whichever tier fits your hardware. Layer the matching overlay on top of the
core stack. Per-service details and tunables live in each
`services/*/README.md`.

**Which one is mine?** Match your hardware to a compute tier:

| Your hardware | Compute tier | Overlay / guide |
|---|---|---|
| No GPU, or any Mac via Docker | **CPU** (the default) | `compose.cpu.yaml` (below) |
| NVIDIA GPU | **NVIDIA** | `compose.gpu.yaml` (below) |
| AMD GPU | **AMD / ROCm** | `compose.rocm.yaml` (below) |
| Apple Silicon Mac (fastest on a Mac) | **metal** | `voxint-metal.sh` (below) |

All four run the **core stack in Docker**. Separately, Apple-Silicon users who
can't or won't run Docker Desktop can use the docker-free
**[native preview](native-macos-preview.md)**. That is a *deployment mode* rather
than a fifth compute tier: it still runs the metal model services under the hood.

### CPU: runs anywhere (the default)

No graphics card, no special drivers. Works on ordinary servers and Apple Silicon
Macs (via Docker Desktop).

```bash
docker compose -f compose.yaml -f compose.cpu.yaml up -d
```

(`up -d` pulls the images on first run, so no separate `pull` step is needed.)

Expect it to be **much slower** than a GPU (a long recording can take hours
rather than minutes), but the results are identical. **8 GB of memory is the
tight floor** (on Docker Desktop that's the *VM* memory limit, not your machine's
total). Below it the services are OOM-killed with an opaque exit rather than a
clear message, so a long recording can fail with no diagnosis. **16 GB is
comfortable.** More:
[operations.md](operations.md#running-without-an-nvidia-gpu-cpu-tier).

### NVIDIA GPU: the fast path

Needs an NVIDIA GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f compose.yaml -f compose.gpu.yaml pull
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

All three services share one GPU. Whisper is pinned to `int8`, so the loaded
weights are lean; budget for decoding headroom on top:

| What you run | Resident VRAM | Practical budget | Comfortable on |
|---|---|---|---|
| **Transcription suite** (Whisper large-v2 int8 ~1.5 GB + pyannote ~1–2 GB + TitaNet ~1 GB) | ~3.5–4.5 GB | **6–8 GB** | an **8 GB** card |
| **+ synthdetect** (w2v2-AASIST, `compose.plugin-synthdetect.yaml`) | +~1.5–2 GB | **~8–10 GB** | a **12 GB** card |
| **+ bundled local LLM** (Qwen3-4B Q5_K_M, `compose.llm.yaml`, GPU-offloaded) | +~4.5–5 GB | **~10–11 GB** | a **12 GB** card |

The pipeline stages run **one at a time**, but each service holds its model
resident for the whole session, so the budget is the *sum of resident weights
plus one stage's decode spike*, not all peaks at once. Enabling both synthdetect
and a GPU-offloaded bundled LLM on one card usually needs **16 GB+**; on a
12 GB card, prefer CPU LLM mode or a second GPU.

**Compatible consumer cards** (NVIDIA):

- **8 GB** (RTX 3050/3060 Ti/4060): transcription suite, comfortably.
- **12 GB** (RTX 3060 12 GB, 4070): transcription suite **plus** the bundled
  Qwen3-4B LLM overlay on the same card, with a thin margin.
- **16 GB+** (RTX 4060 Ti 16 GB, 4070 Ti Super, 4080/4090): the same workload
  with comfortable headroom for longer LLM context or concurrent runs.

> The bundled LLM is opt-in and GPU offload is off by default; enable it by
> uncommenting the GPU `command:`/`deploy:` blocks in `compose.llm.yaml`. It
> covers transcript **enhancement** and run-asset **summary/entities** only;
> web research and speaker-name attribution still need a BYO endpoint.

> **Installed with the guided installer?** On the GPU tier it inventories every
> GPU on the host, measures free VRAM, and recommends the tier that fits. On a
> multi-GPU host it lets you pick which card to target, then writes a
> conservative baseline that pins the model services to that card and works one
> recording at a time so a modest GPU does not thrash. If a card's VRAM is
> occupied (for example by a co-resident local LLM), the installer explains why
> and suggests the CPU tier instead. Two read-only diagnostics preview the
> detection without changing anything: `./scripts/install.sh --gpu-check` (quick
> inventory and classification) and `./scripts/install.sh --hardware-dry-run`
> (full preview including the compose override it would write). Details and how
> to tune the levers back up are in
> [operations.md](operations.md#gpu-memory-on-a-single-modest-gpu-issue-96).

> **Safe defaults have landed; measured per-GPU profiles have not.** The VRAM
> figures above are estimates, and the conservative caps the installer writes are
> deliberately generic. A per-GPU speed profile with a tuned `BATCH_SIZE` (a
> numerics setting, so it only ships once it clears the parity gate) is still to
> come ([#96](https://github.com/bengizmo/voxint/issues/96)), alongside a quality
> assessment against an expertly annotated reference dataset
> ([#97](https://github.com/bengizmo/voxint/issues/97)).

Wire contracts: [docs/gpu-contracts.md](gpu-contracts.md).

### AMD GPU: ROCm tier

A hybrid tier for an AMD GPU: transcription runs on the GPU, while speaker
separation and voice identity use the CPU images. The host needs only the
**amdgpu kernel driver**, with no ROCm install (the image carries its own).

```bash
docker compose -f compose.yaml -f compose.rocm.yaml up -d
```

Some AMD consumer GPUs still hit a known convolution issue
([#4](https://github.com/bengizmo/voxint/issues/4)). Details:
[operations.md](operations.md#running-on-an-amd-gpu-rocm-tier).

Because only transcription runs on the GPU here, a single card is not competing
with two other models for memory the way a small NVIDIA card can. The ROCm tier
keeps the image default `BATCH_SIZE=16` and ships no automatic per-GPU batch
profile. The tuned profiles noted above are NVIDIA-specific; on ROCm you can
still lower `BATCH_SIZE` by hand if a smaller card runs short of memory
(see [operations.md](operations.md#gpu-memory-on-a-single-modest-gpu-issue-96)).

> **Guided installer and AMD GPUs.** The installer discovers AMD GPUs via sysfs,
> reads their VRAM, and classifies each device. The ROCm budget threshold is
> higher than NVIDIA's (14 GiB recommended vs. 8 GiB) because the ROCm whisper
> image peaks at ~13 GiB VRAM under the default batch size. On a multi-GPU host,
> the installer lets you pick which AMD renderD node to target and pins it in the
> generated `compose.hardware.yaml`.

### Apple Silicon Mac: metal tier

Docker Desktop can't pass the Apple GPU into a container, so on a Mac the metal
tier keeps the core stack in Docker but runs the model services **natively** so
speaker separation can use the Apple GPU. This tier needs **Docker Desktop**
specifically: Colima, OrbStack, and plain `dockerd` can't route the containers to
the native services (they break the `host.docker.internal` loopback). It also needs
**[Homebrew](https://brew.sh) and [`uv`](https://docs.astral.sh/uv/)**
(`brew install uv`) for the native model environments.

```bash
./scripts/install.sh                   # choose the [M]etal tier
./scripts/metal/voxint-metal.sh setup  # native environments + verified model weights
./scripts/metal/voxint-metal.sh up     # start the services
./scripts/metal/voxint-metal.sh status # confirm: whisper cpu / pyannote mps / titanet cpu
```

Weights come from the same verified release assets the images use, still with no
Hugging Face account. Details:
[operations.md](operations.md#running-on-apple-silicon-metal-tier).

**Most Mac users want the metal tier.** There is also a docker-free
**[native preview](native-macos-preview.md)** that runs the *whole* stack without
Docker; choose it only if you can't or won't run Docker Desktop. It is a hands-on
technical preview rather than the packaged release.

### Optional: bundled local LLM (no API key)

Voxint's transcript enhancement and run-asset summaries can call a language
model. You can point them at your own OpenAI-compatible endpoint (Settings →
LLM), **or** run the opt-in bundled model: a vendored, Apache-2.0
Qwen3-4B-Instruct served locally, so those features work with **no external
key**. It layers on top of *any* tier above:

```bash
docker compose -f compose.yaml -f compose.cpu.yaml -f compose.llm.yaml up -d
```

Then turn on **Settings → Features → "Use the bundled local model"**. It powers
**only** enhancement and run-asset summaries plus entities; web research and LLM
speaker-name suggestions still need your own endpoint. On CPU it is slow for a
4B model, so a GPU is recommended (see the note in `compose.llm.yaml`). Details:
[operations.md](operations.md#bundled-local-llm-issue-67-optional-no-api-key).

### Optional: synthetic-speech detection (deepfake scoring)

Voxint can check whether the speech in a recording was generated by an AI voice
tool (a "deepfake"). This is opt-in: it adds a separate GPU service that scores
each recording and shows a risk level in the review console. It requires an
NVIDIA GPU (tested on an RTX 3060 12 GB). Layer it on top of your GPU stack:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml \
               -f compose.plugin-synthdetect.yaml up -d
```

Then go to **Settings → Synthetic-speech detection** and turn it on. You can
also turn on **autogenerate** so every new recording is scored automatically. To
score a recording that was processed before you enabled it, open that recording's
detail page and click the **Score** button.

The scores appear on each recording's detail page as a risk chip (low, medium, or
high). Each recording's report page shows all its scored turns in one view.

> ⚠ The detector has known blind spots: some AI voice generators (notably
> Chatterbox) partially evade it, and certain recording conditions can shift
> scores. The report page documents these limitations. See
> [gpu-contracts.md](gpu-contracts.md#synthetic-speech-detection-synthdetect) for
> the full technical contract.

Details:
[operations.md](operations.md#synthetic-speech-detection-plugin-issue-145-optional).

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
you finish it, which is expected. Continue with the
[first-run walkthrough](onboarding.md), which covers the wizard and the guided
tutorial.

> If multiple people will review transcripts on this instance, see
> [Multi-user authentication](operations.md#multi-user-authentication) in
> the operations guide.

Something not working? See
[Settings & troubleshooting](how-to/settings-and-troubleshooting.md) and the
troubleshooting notes in [onboarding.md](onboarding.md#troubleshooting).
