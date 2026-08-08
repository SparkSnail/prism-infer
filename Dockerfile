# syntax=docker/dockerfile:1.7

# The worker image is linux/amd64-only because its pinned FlashAttention wheel
# targets x86_64.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PRISM_MODEL_PROFILE=week12-qwen3-0.6b \
    PRISM_MODEL=/opt/models/Qwen3-0.6B \
    PRISM_MODEL_ID=Qwen/Qwen3-0.6B \
    PRISM_MODEL_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_TOKENIZER_REVISION=9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439 \
    PRISM_MODEL_CONFIG_SHA256=660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd \
    PRISM_DTYPE=bfloat16 \
    PRISM_TP_SIZE=1 \
    PRISM_TOKENS_PER_BLOCK=256 \
    PRISM_KV_BLOCK_BYTES=29360128 \
    PRISM_KV_LAYOUT=NHDC \
    PRISM_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19

WORKDIR /app

# torch.compile uses the system C++ compiler when it materializes Inductor
# kernels at runtime. Keep that toolchain in the immutable worker image instead
# of installing it in live GPU Pods.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# This exact wheel matches the base image's Python 3.11, CUDA 12, Torch 2.6
# and pre-CXX11 ABI. A mismatch must fail the build rather than compile a
# different implementation silently.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade --retries 20 --timeout 120 \
      "pip==25.1.1" "setuptools==82.0.1" "wheel==0.47.0" && \
    python -m pip install --resume-retries 20 --retries 20 --timeout 120 \
      "fastapi==0.139.0" \
      "uvicorn[standard]==0.51.0" \
      "pydantic==2.13.4" \
      "httpx==0.28.1" \
      "nats-py==2.15.0" \
      "transformers==4.51.3" \
      "xxhash==3.7.0" \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl#sha256=58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05"

# Bake the public, immutable model snapshot once during the image build. The
# four workers therefore start without reaching Hugging Face independently.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python - <<'PY'
import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path(os.environ["PRISM_MODEL"])
snapshot_download(
    repo_id=os.environ["PRISM_MODEL_ID"],
    revision=os.environ["PRISM_MODEL_REVISION"],
    local_dir=target,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.model",
        "*.txt",
        "merges.txt",
        "vocab.json",
    ],
)
actual = hashlib.sha256((target / "config.json").read_bytes()).hexdigest()
expected = os.environ["PRISM_MODEL_CONFIG_SHA256"]
if actual != expected:
    raise SystemExit(f"config.json SHA-256 mismatch: expected {expected}, got {actual}")
if not (target / "tokenizer.json").is_file():
    raise SystemExit("pinned snapshot is missing tokenizer.json")
if not (target / "model.safetensors").is_file() and not (
    target / "model.safetensors.index.json"
).is_file():
    raise SystemExit("pinned snapshot is missing safetensors weights")
(target / ".prism-model-revision").write_text(
    os.environ["PRISM_MODEL_REVISION"] + "\n", encoding="utf-8"
)
PY

COPY pyproject.toml README.md LICENSE ./
COPY prism_infer ./prism_infer

ARG GIT_SHA
ARG SOURCE_URL=https://github.com/SparkSnail/prism-infer

ENV PRISM_IMAGE_GIT_SHA=${GIT_SHA}

RUN python -c "import re,sys; assert re.fullmatch(r'[0-9a-f]{40}', sys.argv[1]), 'GIT_SHA must be a full lowercase commit SHA'" "${GIT_SHA}" && \
    python -m pip install --no-build-isolation --no-deps . && \
    python -c "import flash_attn, httpx, torch, triton; from prism_infer.server.process_identity import assert_pidfd_support; from prism_infer.server.worker import main; assert_pidfd_support(); print(torch.__version__, torch.version.cuda, triton.__version__, flash_attn.__version__, httpx.__version__)" && \
    mkdir -p /opt/prism/build && \
    python -m pip freeze --all > /tmp/prism-pip-freeze.txt && \
    LC_ALL=C sort /tmp/prism-pip-freeze.txt > /opt/prism/build/pip-freeze.txt && \
    rm /tmp/prism-pip-freeze.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="prism-infer experimental 2P2D worker" \
      ai.sparksnail.prism.model.id="Qwen/Qwen3-0.6B" \
      ai.sparksnail.prism.model.revision="9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439" \
      ai.sparksnail.prism.model.config-sha256="660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"

EXPOSE 8001 29500
STOPSIGNAL SIGTERM
ENTRYPOINT ["prism-infer-worker"]
