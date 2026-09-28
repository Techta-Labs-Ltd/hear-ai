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
RUN curl -LsSf https://astral.sh/uv/0.10.9/install.sh | sh
WORKDIR /app
COPY deploy/runtime/pyproject.toml deploy/runtime/uv.lock /app/deploy/runtime/
COPY deploy/cleaner/deepfilter3.ini /app/deploy/cleaner/deepfilter3.ini
COPY hear/__init__.py /app/hear/__init__.py
COPY hear/tools /app/hear/tools
COPY patches /app/patches
ENV HEAR_PROJECT_ROOT=/app

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

FROM runtime-base AS transcription-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group transcription --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=transcription
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-pod-base AS pipeline-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS pipeline-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-pod-base AS pipeline-llm-pod
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group pod
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["/usr/local/bin/run_pod.sh"]

FROM runtime-base AS pipeline-llm-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group pipeline --group pipeline-llm --group serverless
RUN python -m hear.tools.dependency_patches && python -m hear.tools.dependency_patches --check
ENV HEAR_WORKER_ROLE=pipeline
ENV HEAR_MODEL_FEATURES=qwen_llm
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["python", "-m", "hear.entrypoints.serverless"]

FROM runtime-base AS reconstruction-base
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction
RUN git clone https://github.com/groxaxo/fish-speech-int4-patch.git /opt/fish-speech && cd /opt/fish-speech && git checkout fc4e1e24ff3b8d7d28fdd66e6789f23acb63c5bb
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

FROM reconstruction-base AS reconstruction-serverless
RUN uv sync --project /app/deploy/runtime --frozen --no-dev --group reconstruction --group serverless
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /opt/fish-speech
RUN python -c "from fish_speech.inference_engine import TTSInferenceEngine"
ENV HEAR_WORKER_ROLE=reconstruction
ENV FISH_SPEECH_HOME=/opt/fish-speech
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["python", "-m", "hear.entrypoints.serverless"]

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

FROM runtime-base AS magic-clean-natural-serverless
COPY --from=magic-clean-natural-serverless-builder /opt/venv /opt/venv
ENV HEAR_WORKER_ROLE=magic_clean_natural
COPY hear /app/hear
COPY scripts /app/scripts
CMD ["python", "-m", "hear.entrypoints.serverless"]

# Assemble once, remove byte-identical duplicated native libraries before the
# final COPY, and retain separate Python dependency environments for each engine.
FROM runtime-pod-base AS runpod-stack-assembly
COPY --from=pipeline-pod /opt/venv /opt/hear-image-assembly/venvs/pipeline
COPY --from=reconstruction-pod /opt/venv /opt/hear-image-assembly/venvs/reconstruction
COPY --from=magic-clean-natural-pod /opt/venv /opt/hear-image-assembly/venvs/magic_clean_natural
COPY scripts/deduplicate_image_dependencies.py /tmp/deduplicate_image_dependencies.py
RUN HEAR_IMAGE_ASSEMBLY=1 python3.12 /tmp/deduplicate_image_dependencies.py

FROM runtime-pod-base AS runpod-stack
COPY --from=runpod-stack-assembly /opt/hear-image-assembly/venvs /opt/hear-ai-v11/venvs
COPY --from=runpod-stack-assembly /opt/hear-image-assembly/shared /opt/hear-ai-v11/shared
COPY --from=reconstruction-pod /opt/fish-speech /opt/fish-speech
RUN ln -s /opt/hear-ai-v11/venvs/pipeline /opt/hear-ai-v11/venvs/transcription
COPY hear /app/hear
COPY scripts /app/scripts
ENV HEAR_POD_STACK_ROLES=reconstruction,pipeline,magic_clean_natural
ENV HEAR_GATEWAY_PYTHON_BIN=/opt/hear-ai-v11/venvs/pipeline/bin/python
ENV HEAR_MODEL_ROOT=/models
ENV FISH_SPEECH_MODEL_ROOT=/root/hear-ai-v11/models
ENV FISH_SPEECH_HOME=/opt/fish-speech
ENV FISH_SPEECH_BNB_MODE=nf4
ENV HEAR_POD_MAX_CONCURRENT_JOBS=1
ENV HEAR_HOST_MAX_CONCURRENT_JOBS=10
ENV HEAR_POD_ROLE_LIMITS={"pipeline":7,"magic_clean_natural":4,"reconstruction":2}
ENV HEAR_POD_PROCESS_LIMITS={"pipeline":7,"magic_clean_natural":1,"reconstruction":1}
ENV HEAR_WORKER_REPLICAS={"pipeline":1,"magic_clean_natural":4,"reconstruction":2}
ENV WHISPER_BATCH_SIZE=8
ENV WHISPER_LONG_AUDIO_BATCH_SIZE=8
ENV WHISPER_CHUNK_SECONDS=240
ENV OMP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=2
ENV HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
ENV HEAR_SOUND_CLEANUP_BUNDLE_SHA256=f878d14f1d892e142db2c3f582a5092aabc9ac260c9b771b02a09b0ea71389a9
ENV HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
ENV HEAR_SOUND_CLEANUP_SEPARATOR_SHA256=e1227365d076eafde534152c75be2c106302705c578eb8f71969c014f2958546
ENV HEAR_GPU_IDLE_EVICTION_ENABLED=true
ENV HEAR_PIPELINE_IDLE_TTL_SECONDS=600
ENV HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS=300
ENV HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS=1200
ENV HEAR_AUDIOSEP_IDLE_TTL_SECONDS=90
ENV HEAR_TEMP_DIR=/workspace/hear-ai-v11/.runtime-audio
ENV PATH=/opt/hear-ai-v11/venvs/pipeline/bin:/root/.local/bin:${PATH}
RUN /opt/hear-ai-v11/venvs/pipeline/bin/python -c "import torch, aio_pika, uvicorn" && \
    /opt/hear-ai-v11/venvs/reconstruction/bin/python -c "import torch, bitsandbytes; from fish_speech.inference_engine import TTSInferenceEngine" && \
    /opt/hear-ai-v11/venvs/magic_clean_natural/bin/python -c "import torch; from df.enhance import init_df"
EXPOSE 8000
CMD ["bash", "/app/scripts/run_pod_stack.sh"]
