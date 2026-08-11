# Voxint titanet service — NeMo TitaNet-Large speaker embeddings (192-dim).
# Contract: docs/gpu-contracts.md (POST /v1/embed, GET /healthz).
# devel base: NeMo's youtokentome dependency compiles against Python headers.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip gcc g++ libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r voxint && useradd -r -g voxint -s /bin/bash voxint

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Strict install order: numpy → Cython → torch cu118 → audio libs → NeMo deps
# → youtokentome (needs Cython, no build isolation) → NeMo → the rest.
RUN pip3 install --no-cache-dir "numpy==1.24.3"
RUN pip3 install --no-cache-dir "Cython==0.29.36"
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install --no-cache-dir \
    "librosa==0.10.1" "soundfile==0.12.1" "scipy==1.11.4" "numba==0.58.1"
RUN pip3 install --no-cache-dir \
    "hydra-core==1.3.2" "omegaconf==2.3.0" "pytorch-lightning==2.0.7" \
    "transformers==4.36.0" "webdataset==0.2.48" "pyarrow==14.0.1" \
    "scikit-learn==1.3.2"
RUN pip3 install --no-cache-dir --no-build-isolation "youtokentome>=1.0.5"
RUN pip3 install --no-cache-dir "nemo_toolkit[asr]==1.22.0"
# NeMo drags in an incompatible huggingface-hub; repin after.
RUN pip3 install --no-cache-dir "huggingface-hub==0.23.5"
RUN pip3 install --no-cache-dir \
    "fastapi==0.104.1" "uvicorn[standard]==0.24.0" "pydantic==2.5.0" \
    "pyloudnorm==0.1.0" "noisereduce==2.0.1"

# Bake the (ungated) TitaNet-Large weights so runtime needs no network.
ENV NEMO_CACHE_DIR=/app/models
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models
RUN mkdir -p /app/models && \
    python3 -c "import nemo.collections.asr as nemo_asr; nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='nvidia/speakerverification_en_titanet_large', map_location='cpu')"

WORKDIR /app
COPY --chown=voxint:voxint app/ /app/app/
RUN chown -R voxint:voxint /app

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True,garbage_collection_threshold:0.7"

USER voxint
ENV HOME=/app

ENV MEDIA_ROOT=/data/media
ENV PORT=8021

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/healthz || exit 1

EXPOSE 8021

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
