ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS base
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libsndfile1 libsox-dev portaudio19-dev python3.12 python3.12-dev python3-pip && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY deploy/runtime/pyproject.toml deploy/runtime/uv.lock /app/deploy/runtime/
COPY pyproject.toml uv.lock /app/
COPY hear /app/hear
COPY patches /app/patches
COPY scripts/validate_image.py /app/scripts/validate_image.py
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription
RUN python -m hear.tools.dependency_patches
RUN python -m hear.tools.dependency_patches --check
FROM base AS transcription-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group pod
ENV HEAR_WORKER_ROLE=transcription
CMD ["python", "-m", "hear.entrypoints.pod"]
FROM base AS transcription-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group serverless
ENV HEAR_WORKER_ROLE=transcription
CMD ["python", "-m", "hear.entrypoints.serverless"]