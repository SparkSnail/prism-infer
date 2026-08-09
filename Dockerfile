# syntax=docker/dockerfile:1.7

ARG PRISM_IMAGE_VARIANT=correctness

# The pinned FlashAttention wheel requires Python 3.11, CUDA 12, Torch 2.6,
# the pre-CXX11 ABI, and x86_64.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS common

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# torch.compile invokes the system C++ compiler when Inductor materializes
# kernels at runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

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

# Each stage owns a complete profile. Model fields are not exposed as
# independent build arguments because hybrid profiles are invalid.
FROM common AS profile-correctness
ENV PRISM_IMAGE_VARIANT=correctness \
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
    PRISM_KV_COMPATIBILITY_ID=a305c48442086f050a0f703c9e79e6e4596c52eaf5dd9f9015cc1a744c1b5b19 \
    PRISM_MODEL_NUM_HIDDEN_LAYERS=28 \
    PRISM_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_MODEL_HEAD_DIM=128 \
    PRISM_MODEL_ROPE_THETA=1000000.0 \
    PRISM_MAX_MODEL_LEN=4096 \
    PRISM_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_MAX_NUM_SEQS=512 \
    PRISM_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_ENFORCE_EAGER=false

FROM common AS profile-performance
ENV PRISM_IMAGE_VARIANT=performance \
    PRISM_MODEL_PROFILE=qwen3-8b-bf16-tp1 \
    PRISM_MODEL=/opt/models/Qwen3-8B \
    PRISM_MODEL_ID=Qwen/Qwen3-8B \
    PRISM_MODEL_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_TOKENIZER_REVISION=b968826d9c46dd6066d109eabc6255188de91218 \
    PRISM_MODEL_CONFIG_SHA256=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30 \
    PRISM_DTYPE=bfloat16 \
    PRISM_TP_SIZE=1 \
    PRISM_TOKENS_PER_BLOCK=256 \
    PRISM_KV_BLOCK_BYTES=37748736 \
    PRISM_KV_LAYOUT=NHDC \
    PRISM_KV_COMPATIBILITY_ID=2647222531d143800a56551dbe8d030c535a1bb2e0e47ff2a12f0781964f4c6c \
    PRISM_MODEL_NUM_HIDDEN_LAYERS=36 \
    PRISM_MODEL_NUM_KEY_VALUE_HEADS=8 \
    PRISM_MODEL_HEAD_DIM=128 \
    PRISM_MODEL_ROPE_THETA=1000000.0 \
    PRISM_MAX_MODEL_LEN=4096 \
    PRISM_MAX_NUM_BATCHED_TOKENS=16384 \
    PRISM_MAX_NUM_SEQS=128 \
    PRISM_GPU_MEMORY_UTILIZATION=0.9 \
    PRISM_ENFORCE_EAGER=false

FROM profile-${PRISM_IMAGE_VARIANT} AS model-download

# Workers load the same pinned snapshot from the image and never contact the
# Hub during Pod startup.
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
if not (target / "tokenizer.json").is_file() or not (
    target / "tokenizer_config.json"
).is_file():
    raise SystemExit("pinned snapshot is missing tokenizer files")
if not (target / "model.safetensors").is_file() and not (
    target / "model.safetensors.index.json"
).is_file():
    raise SystemExit("pinned snapshot is missing safetensors weights")
(target / ".prism-model-revision").write_text(
    os.environ["PRISM_MODEL_REVISION"] + "\n", encoding="utf-8"
)
PY

# The download stage remains the content authority. Each original weight shard
# becomes a separate image layer so registries can upload the shards in parallel.
FROM profile-correctness AS model-files-correctness
COPY --link --from=model-download \
    /opt/models/Qwen3-0.6B/.prism-model-revision \
    /opt/models/Qwen3-0.6B/config.json \
    /opt/models/Qwen3-0.6B/generation_config.json \
    /opt/models/Qwen3-0.6B/merges.txt \
    /opt/models/Qwen3-0.6B/tokenizer.json \
    /opt/models/Qwen3-0.6B/tokenizer_config.json \
    /opt/models/Qwen3-0.6B/vocab.json \
    /opt/models/Qwen3-0.6B/
COPY --link --from=model-download /opt/models/Qwen3-0.6B/model.safetensors /opt/models/Qwen3-0.6B/

FROM profile-performance AS model-files-performance
COPY --link --from=model-download \
    /opt/models/Qwen3-8B/.prism-model-revision \
    /opt/models/Qwen3-8B/config.json \
    /opt/models/Qwen3-8B/generation_config.json \
    /opt/models/Qwen3-8B/merges.txt \
    /opt/models/Qwen3-8B/model.safetensors.index.json \
    /opt/models/Qwen3-8B/tokenizer.json \
    /opt/models/Qwen3-8B/tokenizer_config.json \
    /opt/models/Qwen3-8B/vocab.json \
    /opt/models/Qwen3-8B/
COPY --link --from=model-download /opt/models/Qwen3-8B/model-00001-of-00005.safetensors /opt/models/Qwen3-8B/
COPY --link --from=model-download /opt/models/Qwen3-8B/model-00002-of-00005.safetensors /opt/models/Qwen3-8B/
COPY --link --from=model-download /opt/models/Qwen3-8B/model-00003-of-00005.safetensors /opt/models/Qwen3-8B/
COPY --link --from=model-download /opt/models/Qwen3-8B/model-00004-of-00005.safetensors /opt/models/Qwen3-8B/
COPY --link --from=model-download /opt/models/Qwen3-8B/model-00005-of-00005.safetensors /opt/models/Qwen3-8B/

FROM model-files-${PRISM_IMAGE_VARIANT} AS selected-profile

COPY pyproject.toml README.md LICENSE ./
COPY prism_infer ./prism_infer

ARG GIT_SHA
ARG SOURCE_URL=https://github.com/SparkSnail/prism-infer

ENV PRISM_IMAGE_GIT_SHA=${GIT_SHA}

RUN python -c "import re,sys; assert re.fullmatch(r'[0-9a-f]{40}', sys.argv[1]), 'GIT_SHA must be a full lowercase commit SHA'" "${GIT_SHA}" && \
    python -m pip install --no-build-isolation --no-deps . && \
    python -c "import flash_attn, httpx, torch, triton; from prism_infer.server.process_identity import assert_pidfd_support; from prism_infer.server.unified_baseline import main as baseline_main; from prism_infer.server.worker import main; assert callable(baseline_main); assert_pidfd_support(); print(torch.__version__, torch.version.cuda, triton.__version__, flash_attn.__version__, httpx.__version__)" && \
    mkdir -p /opt/prism/build && \
    python -m pip freeze --all > /tmp/prism-pip-freeze.txt && \
    LC_ALL=C sort /tmp/prism-pip-freeze.txt > /opt/prism/build/pip-freeze.txt && \
    rm /tmp/prism-pip-freeze.txt

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="prism-infer 2P2D worker" \
      ai.sparksnail.prism.image.variant="${PRISM_IMAGE_VARIANT}" \
      ai.sparksnail.prism.model.profile="${PRISM_MODEL_PROFILE}" \
      ai.sparksnail.prism.model.id="${PRISM_MODEL_ID}" \
      ai.sparksnail.prism.model.revision="${PRISM_MODEL_REVISION}" \
      ai.sparksnail.prism.model.tokenizer-revision="${PRISM_TOKENIZER_REVISION}" \
      ai.sparksnail.prism.model.config-sha256="${PRISM_MODEL_CONFIG_SHA256}" \
      ai.sparksnail.prism.kv.compatibility-id="${PRISM_KV_COMPATIBILITY_ID}"

EXPOSE 8001 29500
STOPSIGNAL SIGTERM
ENTRYPOINT ["prism-infer-worker"]
