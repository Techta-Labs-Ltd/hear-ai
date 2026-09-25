ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE} AS validation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git python3.12 python3-pip libsndfile1 libsox3 libsox-fmt-all && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY . .
RUN uv sync --frozen
RUN uv run --no-sync python -m hear.tools.dependency_patches
RUN uv run --no-sync python -m hear.tools.dependency_patches --check
RUN uv run --no-sync python -m pytest tests/test_dependency_patches.py tests/test_four_job_contracts.py tests/test_new_fastapi_health.py tests/test_model_asset_manifest.py tests/test_job_executor.py tests/test_serverless_runtime.py -q

FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg python3.12 python3-pip libsndfile1 libsox3 libsox-fmt-all && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY . .
RUN uv sync --frozen --no-dev
RUN uv run --no-sync python -m hear.tools.dependency_patches
RUN uv run --no-sync python -m hear.tools.dependency_patches --check
RUN rm -rf tests docs deploy/cleaner/evidence

FROM runtime AS pipeline
ENV HEAR_WORKER_ROLE=pipeline

FROM runtime AS transcription
ENV HEAR_WORKER_ROLE=transcription

FROM runtime AS reconstruction
ENV HEAR_WORKER_ROLE=reconstruction

FROM runtime AS magic-clean-natural
ENV HEAR_WORKER_ROLE=magic_clean_natural

FROM runtime AS magic-clean-voice-focus
ENV HEAR_WORKER_ROLE=magic_clean_voice_focus

FROM runtime AS magic-clean-music-atmosphere
ENV HEAR_WORKER_ROLE=magic_clean_music_atmosphere

FROM runtime AS magic-clean-stem-mix
ENV HEAR_WORKER_ROLE=magic_clean_stem_mix
