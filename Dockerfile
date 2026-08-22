# Voxint api/worker image (CPU). GPU model services have their own images under services/.

# --- frontend build (Node exists ONLY at image-build time, never at runtime) ---
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci                              # --frozen semantics via the committed lockfile; never npm install
COPY frontend ./
RUN npm run build                       # tsc --noEmit && vite build && check-no-cdn-urls
RUN test -f dist/.vite/manifest.json    # fail fast if the manifest the Python route needs is missing

# --- python api/worker image (no Node) ---
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

# Prebuilt, hashed island bundle baked in — copied BEFORE uv sync so the
# non-editable wheel packages the static tree into site-packages, where
# app_asset resolves it. No Node in the runtime image.
COPY --from=frontend /app/frontend/dist ./src/voxint/api/static/app

# Vendored transcript-search embedding weights (issue #121): the ~470 MB FP32
# MiniLM ONNX backbone + its tokenizer, baked at the embedder's default path
# (src/voxint/embeddings/onnx_embedder.py). Not in git (weights doctrine).
# Before building, place model.onnx + tokenizer.json under vendor/minilm/. CI
# fetches them from the minilm-onnx-v1 asset release; the maintainer supplies
# them from the pinned upstream revision (see src/voxint/embeddings/models/
# provenance.json). The build FAILS unless their sha256s match the ARGs below.
# Overriding the ARGs produces a NON-PROVENANCE build (dev-only escape hatch);
# release CI never overrides them.
ARG MINILM_ONNX_SHA256=10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a
ARG MINILM_TOKENIZER_SHA256=2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8
COPY vendor/minilm/model.onnx /app/models/minilm/model.onnx
COPY vendor/minilm/tokenizer.json /app/models/minilm/tokenizer.json
RUN printf '%s\n' \
      "${MINILM_ONNX_SHA256}  /app/models/minilm/model.onnx" \
      "${MINILM_TOKENIZER_SHA256}  /app/models/minilm/tokenizer.json" \
    | sha256sum -c - \
    || { echo "vendored MiniLM embedding weights do not match the committed provenance sha256s — \
fetch the matching minilm-onnx-v1 release assets (see src/voxint/embeddings/models/provenance.json)" >&2; exit 1; }

# Honor the committed lockfile exactly; no dev deps, no editable install.
RUN uv sync --frozen --no-dev --no-editable --no-cache
ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home voxint
USER voxint

EXPOSE 8080
# `voxint serve` binds from settings, so the password validator sees the real
# bind address. The image default stays loopback (unpublishable but safe);
# a deployment that wants the port published sets API_HOST=0.0.0.0 AND a real
# VOXINT_PASSWORD — compose.yaml does exactly that for the api service.
CMD ["voxint", "serve"]
