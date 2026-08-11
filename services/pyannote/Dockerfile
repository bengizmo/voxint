# Voxint pyannote service — pyannote/speaker-diarization-3.1 on pyannote.audio 3.1.1.
# Contract: docs/gpu-contracts.md (POST /v1/diarize, GET /healthz).
# The image ships NO model weights: they are HF-gated. Supply HF_TOKEN at runtime
# after accepting the conditions of BOTH pyannote/speaker-diarization-3.1 and
# pyannote/segmentation-3.0 on Hugging Face.
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
# HF weights cache — populated at first startup using the runtime HF_TOKEN.
RUN mkdir -p /app/models && chown -R voxint:voxint /app
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
