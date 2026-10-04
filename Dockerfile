ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS runtime-base
LABEL org.opencontainers.image.source="https://github.com/Techta-Labs-Ltd/hear-ai"
ARG HEAR_BUILD_REVISION=unknown
ENV HEAR_IMAGE_REVISION=$HEAR_BUILD_REVISION
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH=/opt/venv/bin:/root/.local/bin:${PATH}
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
RUN if [ -f /var/lib/dpkg/statoverride ]; then while read -r user group mode path; do getent group "$group" >/dev/null || groupadd --system "$group"; getent passwd "$user" >/dev/null || useradd --system --no-create-home --shell /usr/sbin/nologin "$user"; done < /var/lib/dpkg/statoverride; fi
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git libsndfile1 libsox-dev portaudio19-dev python3.12 python3.12-dev python3-pip && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/0.10.9/install.sh | sh
WORKDIR /app
COPY deploy/runtime/pyproject.toml deploy/runtime/uv.lock /app/deploy/runtime/
COPY deploy/cleaner/deepfilter3.ini /app/deploy/cleaner/deepfilter3.ini
COPY hear/__init__.py /app/hear/__init__.py
COPY hear/tools /app/hear/tools
COPY patches /app/patches
ENV HEAR_PROJECT_ROOT=/app
ENV PYTHONPATH=/app

FROM runtime-base AS runtime-serverless-base
ENV HEAR_RUNTIME_MODE=production
ENV HEAR_MODEL_ROOT=/models
ENV FISH_SPEECH_MODEL_ROOT=/models
ENV HEAR_TEMP_DIR=/audio
ENV HEAR_SERVERLESS_PRELOAD_MODELS=true
ENV HEAR_GPU_IDLE_EVICTION_ENABLED=false
ENV HEAR_SERVERLESS_MAX_CONCURRENT_JOBS=1
ENV WHISPER_BATCH_SIZE=8
ENV WHISPER_LONG_AUDIO_BATCH_SIZE=8
ENV WHISPER_CHUNK_SECONDS=240
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=2

FROM runtime-base AS runtime-pod-base
RUN apt-get update && apt-get install -y --no-install-recommends rabbitmq-server && rm -rf /var/lib/apt/lists/*
COPY deploy/runtime/rabbitmq.conf /etc/rabbitmq/rabbitmq.conf
COPY scripts/run_pod.sh /usr/local/bin/run_pod.sh
RUN chmod 0755 /usr/local/bin/run_pod.sh
ENV HEAR_RABBITMQ_URL=amqp://guest:guest@127.0.0.1:5672/%2F
ENV HEAR_PYTHON_BIN=/opt/venv/bin/python

FROM runtime-pod-base AS transcription-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=transcription
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS transcription-serverless-builder
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=transcription
COPY hear /app/hear
COPY scripts /app/scripts
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -m hear.tools.model_provisioning --role transcription --model-root /models --cache-dir /tmp/hear-hf && rm -rf /tmp/hear-hf /models/.hub-cache /models/*/.cache

FROM runtime-serverless-base AS transcription-serverless
COPY --from=transcription-serverless-builder /opt/venv /opt/venv
COPY --from=transcription-serverless-builder /models /models
ENV HEAR_WORKER_ROLE=transcription
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["bash", "/app/scripts/run_serverless.sh"]

FROM runtime-pod-base AS pipeline-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS pipeline-serverless-builder
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
COPY hear /app/hear
COPY scripts /app/scripts
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -m hear.tools.model_provisioning --role pipeline --model-root /models --cache-dir /tmp/hear-hf && rm -rf /tmp/hear-hf /models/.hub-cache /models/*/.cache

FROM runtime-serverless-base AS pipeline-serverless
COPY --from=pipeline-serverless-builder /opt/venv /opt/venv
COPY --from=pipeline-serverless-builder /models /models
ENV HEAR_WORKER_ROLE=pipeline
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["bash", "/app/scripts/run_serverless.sh"]

FROM runtime-pod-base AS pipeline-llm-pod
# vLLM compiles Triton kernels at engine start-up and needs a C compiler at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS pipeline-llm-serverless-builder
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
COPY hear /app/hear
COPY scripts /app/scripts
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -m hear.tools.model_provisioning --role pipeline --feature qwen_llm --model-root /models --cache-dir /tmp/hear-hf && rm -rf /tmp/hear-hf /models/.hub-cache /models/*/.cache

FROM runtime-serverless-base AS pipeline-llm-serverless
# vLLM compiles Triton kernels at engine start-up and needs a C compiler at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY --from=pipeline-llm-serverless-builder /opt/venv /opt/venv
COPY --from=pipeline-llm-serverless-builder /models /models
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["bash", "/app/scripts/run_serverless.sh"]

FROM runtime-base AS reconstruction-base
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech && git -C /opt/fish-speech checkout 214da3cd841bda85da2496b96cd3c4d7edb1337e
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/fish-speech
RUN python -c "from fish_speech.inference_engine import TTSInferenceEngine; from fish_speech.models.dac.inference import load_model; from fish_speech.models.text2semantic.inference import launch_thread_safe_queue; from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest"

FROM runtime-pod-base AS reconstruction-pod
COPY --from=reconstruction-base /opt/venv /opt/venv
COPY --from=reconstruction-base /opt/fish-speech /opt/fish-speech
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group pod
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/fish-speech
RUN python -c "import aio_pika, uvicorn; from fish_speech.inference_engine import TTSInferenceEngine"
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM reconstruction-base AS reconstruction-serverless-builder
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group serverless
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/fish-speech
RUN python -c "from fish_speech.inference_engine import TTSInferenceEngine"
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
COPY hear /app/hear
COPY scripts /app/scripts
ARG HEAR_FISH_LICENSE_APPROVED=false
RUN mkdir -p /models && if [ "$HEAR_FISH_LICENSE_APPROVED" = "true" ]; then HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 python -m hear.tools.model_provisioning --role reconstruction --model-root /models --cache-dir /tmp/hear-hf --acknowledge-license-review; else echo "Fish model omitted because approval flag is not enabled"; fi && rm -rf /tmp/hear-hf /models/.hub-cache /models/*/*/.cache

FROM runtime-serverless-base AS reconstruction-serverless
COPY --from=reconstruction-serverless-builder /opt/venv /opt/venv
COPY --from=reconstruction-serverless-builder /opt/fish-speech /opt/fish-speech
COPY --from=reconstruction-serverless-builder /models /models
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["bash", "/app/scripts/run_serverless.sh"]

FROM runtime-base AS magic-clean-natural-pod-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-natural --group pod

FROM runtime-pod-base AS magic-clean-natural-pod
COPY --from=magic-clean-natural-pod-builder /opt/venv /opt/venv
ENV HEAR_WORKER_ROLE=magic_clean_natural
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS magic-clean-natural-serverless-builder
RUN apt-get update && apt-get install -y --no-install-recommends cargo rustc && rm -rf /var/lib/apt/lists/*
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-natural --group serverless
COPY scripts/provision_magic_clean_models.py /app/scripts/provision_magic_clean_models.py
RUN python /app/scripts/provision_magic_clean_models.py --model-root /models --engine deepfilter

FROM runtime-base AS sound-cleanup-assets-builder
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group sound-cleanup-provisioning
RUN uv pip install --python /opt/venv/bin/python "huggingface-hub>=0.24,<1" "silero-vad==6.2.1"
COPY scripts /app/scripts
RUN python /app/scripts/provision_release_sound_assets.py --model-root /models

FROM runtime-serverless-base AS magic-clean-natural-serverless
COPY --from=magic-clean-natural-serverless-builder /opt/venv /opt/venv
COPY --from=magic-clean-natural-serverless-builder /models /models
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-v1-runtime /models/sound-cleanup-v1-runtime
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-specialist /models/sound-cleanup-specialist
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-release.env /models/sound-cleanup-release.env
ENV HEAR_WORKER_ROLE=magic_clean_natural
ENV HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
ENV HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["bash", "/app/scripts/run_serverless.sh"]

# Build the complete RunPod runtime and model payload from pinned sources.
# Models are fetched in this disposable build stage, never from /workspace.
FROM runtime-pod-base AS runpod-stack-builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cargo rustc
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
COPY hear /app/hear
COPY scripts /app/scripts
COPY patches /app/patches
RUN UV_PROJECT_ENVIRONMENT=/opt/hear-image-assembly/venvs/pipeline uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pod
RUN /opt/hear-image-assembly/venvs/pipeline/bin/python -m hear.tools.dependency_patches && /opt/hear-image-assembly/venvs/pipeline/bin/python -m hear.tools.dependency_patches --check
RUN git clone https://github.com/fishaudio/fish-speech.git /opt/fish-speech && git -C /opt/fish-speech checkout 214da3cd841bda85da2496b96cd3c4d7edb1337e
RUN UV_PROJECT_ENVIRONMENT=/opt/hear-image-assembly/venvs/reconstruction uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group pod
RUN uv pip install --python /opt/hear-image-assembly/venvs/reconstruction/bin/python --no-deps -e /opt/fish-speech
RUN UV_PROJECT_ENVIRONMENT=/opt/hear-image-assembly/venvs/magic_clean_natural uv sync --project /app/deploy/runtime --frozen --no-dev --group magic-clean-natural --group pod
RUN /opt/hear-image-assembly/venvs/pipeline/bin/python -m hear.tools.model_provisioning --role pipeline --model-root /models --cache-dir /tmp/hear-hf-pipeline
ARG HEAR_FISH_LICENSE_APPROVED=false
RUN if [ "$HEAR_FISH_LICENSE_APPROVED" = "true" ]; then /opt/hear-image-assembly/venvs/reconstruction/bin/python -m hear.tools.model_provisioning --role reconstruction --model-root /models --cache-dir /tmp/hear-hf-fish --acknowledge-license-review && rm -rf /tmp/hear-hf-fish; else echo "Fish model omitted because approval flag is not enabled"; fi
RUN /opt/hear-image-assembly/venvs/magic_clean_natural/bin/python /app/scripts/provision_magic_clean_models.py --model-root /models --engine deepfilter
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-v1-runtime /models/sound-cleanup-v1-runtime
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-specialist /models/sound-cleanup-specialist
COPY --from=sound-cleanup-assets-builder /models/sound-cleanup-release.env /models/sound-cleanup-release.env
RUN HEAR_IMAGE_ASSEMBLY=1 python3.12 /app/scripts/deduplicate_image_dependencies.py

FROM runtime-pod-base AS runpod-stack
ARG HEAR_BUILD_REVISION=unknown
ENV HEAR_IMAGE_REVISION=$HEAR_BUILD_REVISION
ENV HEAR_RUNTIME_MODE=production
COPY --from=runpod-stack-builder /opt/hear-image-assembly/venvs /opt/hear-ai-v11/venvs
COPY --from=runpod-stack-builder /opt/hear-image-assembly/shared /opt/hear-ai-v11/shared
COPY --from=runpod-stack-builder /opt/fish-speech /opt/fish-speech
COPY --from=runpod-stack-builder /models /models
RUN ln -s /opt/hear-ai-v11/venvs/pipeline /opt/hear-ai-v11/venvs/transcription
COPY hear /app/hear
COPY scripts /app/scripts
ENV HEAR_POD_STACK_ROLES=pipeline,magic_clean_natural
ENV HEAR_GATEWAY_PYTHON_BIN=/opt/hear-ai-v11/venvs/pipeline/bin/python
ENV HEAR_MODEL_ROOT=/models
ENV FISH_SPEECH_MODEL_ROOT=/models
ENV FISH_SPEECH_HOME=/opt/fish-speech
ENV HEAR_POD_MAX_CONCURRENT_JOBS=1
ENV HEAR_HOST_MAX_CONCURRENT_JOBS=10
ENV HEAR_POD_ROLE_LIMITS={"pipeline":7,"magic_clean_natural":4}
ENV HEAR_POD_PROCESS_LIMITS={"pipeline":7,"magic_clean_natural":1}
ENV HEAR_WORKER_REPLICAS={"pipeline":1,"magic_clean_natural":4}
ENV WHISPER_BATCH_SIZE=8
ENV WHISPER_LONG_AUDIO_BATCH_SIZE=8
ENV WHISPER_CHUNK_SECONDS=240
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=2
ENV HEAR_GPU_IDLE_EVICTION_ENABLED=true
ENV HEAR_PIPELINE_IDLE_TTL_SECONDS=600
ENV HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS=300
ENV HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS=1200
ENV HEAR_AUDIOSEP_IDLE_TTL_SECONDS=90
ENV HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
ENV HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
ENV HEAR_TEMP_DIR=/root/hear-ai-runtime/scratch
ENV PATH=/opt/hear-ai-v11/venvs/pipeline/bin:/root/.local/bin:${PATH}
RUN /opt/hear-ai-v11/venvs/pipeline/bin/python -c "import torch, aio_pika, uvicorn" && \
    /opt/hear-ai-v11/venvs/reconstruction/bin/python -c "import torch; from fish_speech.inference_engine import TTSInferenceEngine" && \
    /opt/hear-ai-v11/venvs/magic_clean_natural/bin/python -c "import torch; from df.enhance import init_df"
EXPOSE 8000
CMD ["bash", "/app/scripts/run_pod_stack.sh"]
