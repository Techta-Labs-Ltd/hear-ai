ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS runtime-base
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libsndfile1 libsox-dev portaudio19-dev python3.12 python3.12-dev python3-pip && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY deploy/runtime/pyproject.toml deploy/runtime/uv.lock /app/deploy/runtime/
COPY hear /app/hear
COPY patches /app/patches

FROM runtime-base AS transcription-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=transcription
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS transcription-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=transcription
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS pipeline-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS pipeline-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS pipeline-llm-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS pipeline-llm-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS reconstruction-base
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech && cd /opt/fish-speech && git checkout 214da3cd841bda85da2496b96cd3c4d7edb1337e
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/fish-speech
RUN python -c "from fish_speech.inference_engine import TTSInferenceEngine; from fish_speech.models.dac.inference import load_model; from fish_speech.models.text2semantic.inference import launch_thread_safe_queue; from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest"

FROM reconstruction-base AS reconstruction-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group pod
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM reconstruction-base AS reconstruction-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group serverless
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS magic-clean-natural-pod-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-natural --group pod

FROM runtime-base AS magic-clean-natural-pod
COPY --from=magic-clean-natural-pod-builder /opt/venv /opt/venv
ENV HEAR_WORKER_ROLE=magic_clean_natural
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS magic-clean-natural-serverless-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-natural --group serverless

FROM runtime-base AS magic-clean-natural-serverless
COPY --from=magic-clean-natural-serverless-builder /opt/venv /opt/venv
ENV HEAR_WORKER_ROLE=magic_clean_natural
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS magic-clean-voice-focus-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-voice-focus --group pod
ENV HEAR_WORKER_ROLE=magic_clean_voice_focus
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS magic-clean-voice-focus-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-voice-focus --group serverless
ENV HEAR_WORKER_ROLE=magic_clean_voice_focus
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS magic-clean-music-atmosphere-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-music-atmosphere --group pod
ENV HEAR_WORKER_ROLE=magic_clean_music_atmosphere
CMD ["python", "-m", "hear.entrypoints.pod"]

FROM runtime-base AS magic-clean-music-atmosphere-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-music-atmosphere --group serverless
ENV HEAR_WORKER_ROLE=magic_clean_music_atmosphere
CMD ["python", "-m", "hear.entrypoints.serverless"]
