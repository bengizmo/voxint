# Voxint titanet service — ONNX Runtime GPU flavor (CUDA EP, Blackwell-ready).
# Same embedding space id (titanet-large-v2) and .onnx graph as Dockerfile.cpu,
# running on GPU via onnxruntime-gpu's CUDAExecutionProvider. No NeMo, no torch.
# Contract: docs/gpu-contracts.md (POST /v1/embed, GET /healthz).
#
# The ~100 MB .onnx graph is NOT in git. Before building, place the exported
# artifact at services/titanet/models/titanet-large.onnx — either by running
# tools/export_titanet_onnx.py inside a NeMo container (maintainer path)
# or by downloading the checksummed release asset (CI path). The build FAILS
# unless its sha256 matches TITANET_ONNX_SHA256, which must equal
# "onnx_sha256" in tests/parity/fixtures/onnx/provenance.json (a contract
# test pins the two together).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python 3.11 from deadsnakes (matching Dockerfile.cpu's python:3.11-slim).
# onnxruntime-gpu >= 1.24 requires Python >= 3.11; Ubuntu 22.04 ships 3.10.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv libgomp1 libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r voxint && useradd -r -g voxint -s /bin/bash voxint

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --no-cache-dir --upgrade pip wheel

COPY requirements.txt /tmp/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r /tmp/requirements.txt
# onnxruntime-gpu 1.28+ is not on PyPI; the CUDA 12 build lives on
# Microsoft's feed. Installed separately so the extra index does not
# affect the other packages.
RUN python3.11 -m pip install --no-cache-dir onnxruntime-gpu==1.28.0 \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Overriding this ARG produces a NON-PROVENANCE build (dev-only escape hatch);
# release CI never overrides it.
ARG TITANET_ONNX_SHA256=f397fedc81e77d7f6a42e4d5bc78e2a5f81b9fd663ac82415e423467221545bf
COPY models/titanet-large.onnx /app/models/titanet-large.onnx
RUN echo "${TITANET_ONNX_SHA256}  /app/models/titanet-large.onnx" | sha256sum -c - \
    || { echo "titanet-large.onnx does not match the committed provenance sha256 — \
re-export with tools/export_titanet_onnx.py or fetch the matching release asset" >&2; exit 1; }

WORKDIR /app
COPY --chown=voxint:voxint app/ /app/app/
RUN chown -R voxint:voxint /app

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
# "utility" mounts libnvidia-ml so the /healthz telemetry sampler can read
# NVML. Losing it degrades telemetry to "unsupported" but never affects
# readiness.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER voxint
ENV HOME=/app

ENV EMBED_ENGINE=onnx
ENV TITANET_ONNX_PATH=/app/models/titanet-large.onnx
ENV TITANET_ORT_PROVIDERS=CUDAExecutionProvider
ENV MEDIA_ROOT=/data/media
ENV PORT=8021

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

EXPOSE 8021

CMD ["sh", "-c", "python3.11 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
