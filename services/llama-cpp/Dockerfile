# Voxint bundled local-LLM service — llama.cpp server + a vendored, sha-pinned
# Qwen3-4B-Instruct-2507 GGUF (Q5_K_M, Apache-2.0). This is the OPTIONAL,
# no-API-key model that powers transcript ENHANCEMENT + run-asset
# SUMMARY/ENTITIES only (issue #67); it does NOT power web research or LLM
# speaker-name attribution — those stay on a BYO endpoint (see #66 for why a
# small local model is not safe as an unrestricted default).
#
# Weights are NOT in git. Before building, place the vendored GGUF at
# services/llama-cpp/models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf — either from the
# checksummed release asset (CI path: gh release download
# qwen3-4b-instruct-2507-q5-v1) or a maintainer-staged copy. The build FAILS
# unless its sha256 matches QWEN_GGUF_SHA256, which must equal "sha256" in
# services/llama-cpp/provenance.json (a contract test pins the two together,
# mirroring the titanet/pyannote vendoring precedent).
#
# The base image is digest-pinned: the serving engine is part of the numerics
# contract, so it may not float. Regenerate/refresh only via a new asset
# release + provenance bump (docs/release-process.md).
FROM ghcr.io/ggml-org/llama.cpp:server@sha256:092d1291f2bcf59ff727fa3af855fb9bd4759d6bff860f6fbfd5e3e377e12625

# Attribution + provenance travel inside the image (Apache-2.0 redistribution).
COPY provenance.json /app/provenance.json
COPY NOTICE /app/NOTICE

# Overriding this ARG produces a NON-PROVENANCE build (dev-only escape hatch for
# local quant experiments); release CI never overrides it.
ARG QWEN_GGUF_SHA256=5bde5e9d883622acb02bf77fe7dcbc56a8b9a9ad4be78a72ca23a532658b4ecb
COPY models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf /models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf
RUN echo "${QWEN_GGUF_SHA256}  /models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf" | sha256sum -c - \
    || { echo "Qwen3-4B-Instruct-2507-Q5_K_M.gguf does not match the committed provenance \
sha256 — fetch the matching qwen3-4b-instruct-2507-q5-v1 release asset" >&2; exit 1; }

# The pinned serving profile (measured under #67 Phase A) ships WITH the weights
# as the default command, so a single-operator `up` needs no serving knobs and
# the profile can't drift out from under the qualification. The base image's
# entrypoint is /app/llama-server, so these are appended as its arguments.
# Greedy sampling (temperature 0) is enforced CLIENT-side (SamplingProfile), so
# it is deliberately not pinned here; these flags pin context/KV/flash-attn and
# the chat/reasoning handling, which do shape the output.
# GPU is strongly recommended for anything but short clips — append `-ngl 99`
# (and reserve a device in compose.llm.yaml) to offload; CPU is a slow backstop.
CMD ["-m", "/models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--alias", "qwen3-4b-instruct-2507", \
     "-c", "32768", "-np", "1", \
     "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", \
     "--jinja", "--reasoning", "off"]
