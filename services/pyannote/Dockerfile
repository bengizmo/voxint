# Voxint pyannote service — pyannote/speaker-diarization-3.1 on pyannote.audio 3.1.1.
# Contract: docs/gpu-contracts.md (POST /v1/diarize, GET /healthz).
# The image bakes the vendored diarization weights (~33 MB): before building,
# place the two checkpoints at services/pyannote/models/*.bin — download them
# from the pyannote-models-v1 asset release (CI path) or from the upstream HF
# repos at the revisions pinned in models/provenance.json (maintainer path).
# The build FAILS unless their sha256s match the ARGs below. HF_TOKEN is no
# longer needed for the default pipeline; DIARIZER_MODEL_NAME + HF_TOKEN at
# runtime restore the online Hugging Face path.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r voxint && useradd -r -g voxint -s /bin/bash voxint

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# numpy first, then torch from the cu118 index, then everything else — the
# order prevents the resolver from dragging in incompatible torch builds.
RUN pip3 install --no-cache-dir "numpy==1.24.3"
RUN pip3 install --no-cache-dir \
    torch==2.5.0+cu118 torchaudio==2.5.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
# Weights cache for the optional online HF path (DIARIZER_MODEL_NAME override);
# the default vendored pipeline below needs no cache and no token.
RUN mkdir -p /app/models && chown -R voxint:voxint /app

# Vendored pipeline: sha256s must match models/provenance.json. Overriding the
# ARGs produces a NON-PROVENANCE build (dev-only escape hatch); release CI
# never overrides them. The checkpoints live under a "pyannote"-named
# directory on purpose — pyannote.audio 3.1.1 dispatches embedding loaders on
# path substrings (see models/config.vendored.yaml).
ARG SEGMENTATION_SHA256=da85c29829d4002daedd676e012936488234d9255e65e86dfab9bec6b1729298
ARG WESPEAKER_SHA256=366edf44f4c80889a3eb7a9d7bdf02c4aede3127f7dd15e274dcdb826b143c56
COPY models/config.vendored.yaml /app/vendored/config.yaml
COPY models/segmentation-3.0.bin /app/vendored/pyannote/segmentation-3.0.bin
COPY models/wespeaker-voxceleb-resnet34-LM.bin /app/vendored/pyannote/wespeaker-voxceleb-resnet34-LM.bin
RUN printf '%s\n' \
      "${SEGMENTATION_SHA256}  /app/vendored/pyannote/segmentation-3.0.bin" \
      "${WESPEAKER_SHA256}  /app/vendored/pyannote/wespeaker-voxceleb-resnet34-LM.bin" \
    | sha256sum -c - \
    || { echo "vendored pyannote checkpoints do not match the committed provenance sha256s — \
fetch the matching pyannote-models-v1 release assets (see models/provenance.json)" >&2; exit 1; }
ENV VOXINT_VENDORED_PIPELINE=/app/vendored/config.yaml

COPY --chown=voxint:voxint app/ /app/app/

ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models
ENV HUGGINGFACE_HUB_CACHE=/app/models
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8"

USER voxint
ENV HOME=/app

ENV MEDIA_ROOT=/data/media
ENV PORT=8024

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

EXPOSE 8024

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
