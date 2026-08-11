# Voxint api/worker image (CPU). GPU model services have their own images under services/.
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

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
