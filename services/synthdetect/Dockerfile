# Synthdetect GPU service — w2v2-AASIST synthetic speech detection.
#
# Weights must be staged at build time under weights/ (not committed to git).
# See docs/gpu-contracts.md for the authoritative contract.
#
# Build: docker build -t voxint-synthdetect:dev .
# Run:   docker run --gpus device=0 -v /path/to/media:/data/media:ro \
#            -p 8025:8025 voxint-synthdetect:dev

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MEDIA_ROOT=/data/media
ENV PORT=8025
ENV HF_HUB_OFFLINE=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev curl libsndfile1 git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r voxint && useradd -r -g voxint -d /app -s /sbin/nologin voxint

WORKDIR /app

# Python deps: install order matters (numpy before torch so fairseq's build
# sees a stable numpy; numpy==1.23.5 is the last version with np.float).
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir \
        torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    && pip3 install --no-cache-dir \
        fairseq@git+https://github.com/facebookresearch/fairseq.git@a5402130

# Weights layer (changes rarely, cache-friendly).
# Stage weights/ before building:
#   cp ~/synthdetect-weights/finetuned_aasist.pth weights/
#   cp ~/synthdetect-weights/xlsr2_300m.pt weights/
COPY weights/ /app/weights/

# Vendored upstream model (MIT, commit 4acaa61d)
COPY app/vendor/ /app/app/vendor/

# Application code
COPY app/ /app/app/

# Verify weight shas at build time
ARG AASIST_SHA="e178446b640b8e9f9cf6dd359428b2243f49e24e613e1ae952cd706216b8111e"
ARG XLSR_SHA="b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9"
ENV SYNTHDETECT_AASIST_SHA=${AASIST_SHA}
ENV SYNTHDETECT_XLSR_SHA=${XLSR_SHA}

RUN chown -R voxint:voxint /app
USER voxint

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
