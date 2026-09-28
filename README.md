# Hear AI Runtime

Fish TTS editing and deployment audit: see [Fish reconstruction](docs/FISH_TTS_RECONSTRUCTION.md) and [Pod deployment audit](docs/POD_DEPLOYMENT_AUDIT.md).

Hear AI is a Python 3.12 execution runtime for audio intelligence jobs. It runs as a capability-specific Pod worker or a RunPod Serverless handler. Both entrypoints use the same contracts, executor, workflows, local inference engines, and artifact storage.

See [job intake and deployment checks](docs/JOB_RUNTIME_SETUP.md) before enabling app traffic.

## Runtime layout

```text
Hear Backend
    |
    +-- HTTPS/SSE --> Pod API --> RabbitMQ --> Pod worker --+
    |                                                       |
    +-- RunPod Serverless -------------------------+--> workflow
                                                     --> local inference engines
                                                     --> Backblaze B2 artifacts
Pod worker / Serverless handler -- canonical events as SSE --> Hear Backend
```

The runtime supports four durable job types: `pipeline`, `transcription`, `reconstruction`, and `magic_clean`. Magic Clean uses the pinned DeepFilterNet3 model with four one-click profiles: `natural`, `studio_voice`, `outdoor_mobile`, and `clean_raw`. SAM-Audio is no longer supported. See [DeepFilterNet cleaning profiles](docs/DEEPFILTER_CLEANING_PROFILES.md) for processing, options, validation and migration. Reconstruction generates edited narration with Fish Speech and assembles a disk-backed timeline; it never substitutes pre-rendered audio for TTS. The backend owns durable job state, retries, routing, and client progress streams; this service does not connect to the application database or Redis.

The Pod accepts authenticated `AttemptEnvelope` requests at `POST /v1/attempts` and returns 202 only after confirmed publication to RabbitMQ. Results are reported to the owning backend. `POST /v1/attempts/stream` is an optional preview stream, not a durable result store. Its local worker consumes that queue, claims attempts through the backend, and executes them. RabbitMQ is local to the Pod, and its AMQP listener binds to loopback. Serverless workers use RunPod dispatch and emit the same canonical events; they do not use RabbitMQ. The backend persists events, outcomes, and user-facing progress. The Pod also exposes `/healthz`, `/readyz`, `/capabilities`, `/metrics`, and `/drain`.

## Optional Sound Cleanup

Sound Cleanup adds bounded event-region repair after DeepFilterNet without changing
existing profile defaults. It supports pinned offline event/speech analysis, local
repair, and explicitly selected AudioSep overlap previews. It is off by default;
see [Sound Cleanup v1](docs/SOUND_CLEANUP_V1.md) for contracts, model provisioning,
verified results, and the remaining backend/frontend migration boundary.

## Build runtime images

Docker targets provide separate Pod and Serverless images for each supported role. Optional pipeline LLM images are also available.

```bash
docker build --target pipeline-pod -t hear-ai:pipeline .
docker build --target transcription-serverless -t hear-ai:transcription-serverless .
docker build --target reconstruction-pod -t hear-ai:reconstruction .
docker build --target magic-clean-natural-serverless -t hear-ai:magic-clean-natural .
```

Use a target that matches the worker role and transport. Image builds install the role-specific dependency group, apply and verify required dependency patches, and run workers with offline model loading. Publish immutable image tags tied to the source revision.

## Configure a worker

Start with [.env.example](.env.example) and provide deployment values through the platform's environment or secret store. Required common values include:

- `HEAR_WORKER_ROLE`, `HEAR_IMAGE_REVISION`, and `HEAR_ENGINE_REVISION`; `HEAR_WORKER_ID` may be supplied or generated from the RunPod Pod ID and role
- `HEAR_MODEL_ROOT` and `HEAR_TEMP_DIR`
- `HEAR_BACKEND_INTERNAL_URL` and `HEAR_BACKEND_SERVICE_KEY`
- `HEAR_POD_API_KEY` for authenticated Pod job requests
- `HEAR_RABBITMQ_URL` for the Pod-local RabbitMQ broker
- `HEAR_POD_MAX_CONCURRENT_JOBS` (defaults to `1`)
- `HEAR_SERVERLESS_MAX_CONCURRENT_JOBS` (defaults to `1` per Serverless worker)
- `HEAR_HOST_JOB_LOCK_PATH` for the Pod-wide active job permit
- `HEAR_API_MAX_BODY_BYTES` for bounded attempt request bodies
- `HEAR_OPTIONAL_ENGINE_MODE=available` to run all DeepFilterNet profiles, or `certified` for the legacy Natural certificate-gated workflow
- `HEAR_MAGIC_CLEAN_MODEL_DEVICE=cuda:0` for the installed Magic Clean model engines
- `HEAR_CLEANER_CERTIFICATION_PATH` only for the legacy certified Natural mode

`HEAR_WORKER_ID` identifies the role process for backend leases and heartbeats; it does not select a GPU. GPU selection comes from the Pod's CUDA device, which is `cuda:0` on this one-GPU Pod. `HEAR_IMAGE_REVISION` and `HEAR_ENGINE_REVISION` identify the software and model/runtime versions reported with that worker identity.

Set `HEAR_MODEL_FEATURES=qwen_llm` only when using a pipeline LLM image. Provision model files under `HEAR_MODEL_ROOT`; startup validates local files and does not download weights. Version 1 of `hear/model_manifest.json` records pinned revisions, required files, engine adapters, provenance, and license review status.

### Runpod storage paths

Keep source code and model data in separate directories. On this Pod, source stays in `/workspace/hear-ai-v11`; model weights live under `/models`, Hugging Face/Torch caches under `/root/.cache`, and isolated role environments live under `/opt/hear-ai-v11/venvs/<role>`. `/root`, `/models`, and `/opt` are temporary across Pod replacement. The Serverless profile uses its attached network volume at `/runpod-volume` for persistent weights. The sample path profiles are [runpod-pod.env.example](deploy/runtime/env/runpod-pod.env.example) and [runpod-serverless.env.example](deploy/runtime/env/runpod-serverless.env.example).

Provision models on the Pod after loading the Pod profile:

```bash
python scripts/setup_runtime.py --role transcription --provider pod
set -a
source deploy/runtime/env/runpod-pod.env.example
set +a
uv run --project deploy/runtime --no-sync python -m hear.tools.model_provisioning \
  --role transcription \
  --model-root "$HEAR_MODEL_ROOT" \
  --cache-dir "$HEAR_MODEL_ROOT/.hub-cache"
uv run --project deploy/runtime --no-sync python -m hear.tools.model_provisioning \
  --role transcription \
  --model-root "$HEAR_MODEL_ROOT" \
  --cache-dir "$HEAR_MODEL_ROOT/.hub-cache" \
  --verify-only
```

In Serverless endpoint environment settings, use the values from `runpod-serverless.env.example` and attach a network volume in the same data center. Provision Serverless weights separately under `/runpod-volume/hear-ai-v11/models`; the Pod's `/models` directory is on its temporary container root and is not shared. Model provisioning is an explicit step; the worker image runs in Hugging Face offline mode and will not download weights at startup. Use a network volume for Serverless provisioning because the downloader uses atomic file replacement and Hugging Face cache locks; Runpod documents that its global volume does not provide file locks or atomic rename. Network volumes pin the endpoint to their data center. Keep `/tmp/hear-ai-audio` for per-job scratch; completed workflows remove their attempt directory, and the worker container disk is temporary.

Provision the Magic Clean engine outside worker startup:

```bash
/opt/hear-ai-v11/venvs/magic_clean_natural/bin/python \
  scripts/provision_magic_clean_models.py --model-root /models --engine deepfilter
```

The provisioner verifies pinned file hashes before replacing the model directory.

Set the endpoint’s initial active workers and concurrency to one while measuring GPU and RAM use. Add workers only after confirming the chosen GPU can hold the selected role’s loaded models and concurrent jobs. Pipeline LLM use also reads `QWEN_LLM_GPU_MEMORY_UTILIZATION`; its default is `0.75`.

Provision and verify models before starting a worker:

```bash
python -m hear.tools.model_provisioning --role transcription --model-root /models
python -m hear.tools.model_provisioning --role transcription --model-root /models --verify-only
```

For an image that uses an optional pipeline feature, pass its feature explicitly during provisioning and verification:

```bash
python -m hear.tools.model_provisioning --role pipeline --feature qwen_llm --model-root /models
```

The optional certified Natural workflow requires a pinned Natural-only certificate.
New cleaning presets use available mode and measured output validation; see the
[profile migration guide](docs/DEEPFILTER_CLEANING_PROFILES.md).

## Run locally

Use a Python 3.12 environment with the dependency groups for the selected role. The deployment lock and role groups are in [deploy/runtime/pyproject.toml](deploy/runtime/pyproject.toml). The container targets are the reproducible production build path.

Run one local transcription directly on this Pod without Runpod Serverless or Hear backend credentials:

```bash
python scripts/setup_runtime.py --role transcription --provider pod
set -a
source deploy/runtime/env/runpod-pod.env.example
set +a
uv run --project deploy/runtime --no-sync python -m scripts.run_local_transcription \
  /path/to/audio.wav \
  --output .cache/local-jobs/transcription.json
```

This loads the provisioned Qwen ASR and aligner from `HEAR_MODEL_ROOT`, writes the transcript JSON at the requested output path, and unloads the model afterward. It runs the transcription model and service locally; the normal Pod entrypoint accepts backend attempts over HTTP/SSE.

For local setup, select the worker role and provider explicitly. For example:

```bash
python scripts/setup_runtime.py --role transcription --provider pod
python scripts/setup_runtime.py --role pipeline --provider serverless --feature qwen_llm
```

The repository root `pyproject.toml` contains shared lint, typing, and test configuration. Runtime dependencies and the only dependency lock live under `deploy/runtime/`.

Start the complete Pod gateway and local worker stack:

```bash
cp deploy/runtime/env/runpod-pod.env.example .env
scripts/run_pod_stack.sh
```

Set backend credentials and the Pod API key in `/root/hear-ai-v11/runtime.env` before starting the process. The launcher prefers that root-owned environment file and falls back to the repository `.env` when it is absent. The API is the only web listener on port 8000. It authenticates the request, selects the RabbitMQ queue from `job_type` and the Magic Clean profile, and streams worker events over the same SSE connection. Role processes are queue consumers and do not expose HTTP ports. Pod environments use `/opt/hear-ai-v11/venvs/<role>` and Serverless environments use `/opt/hear-ai-v11/venvs/<role>-serverless`; no backend worker ID is needed to submit a job.

Set `HEAR_POD_STACK_ROLES` explicitly. Each requested role starts its own consumer; transcription is not silently skipped when pipeline is configured. All four cleaning profiles use this one consumer and its existing queue.
Remove the retired SAM worker role from deployment environment settings.

The Magic Clean portion of an attempt request accepts, for example:

```json
{"profile":"studio_voice","auto_level":true,"remove_clicks":false,"trim_silence":false}
```

```json
{"profile":"clean_raw"}
```

Legacy `natural` requests still accept `attenuation_limit_db` as 12, 18 or 24.
The new profiles use real preset DSP and produce a measured FLAC master, MP3 and
validation report. All candidates require approval; originals are retained.
[Full profile/API and deployment guide](docs/DEEPFILTER_CLEANING_PROFILES.md).

Reconstruction is Fish TTS editing in every engine mode. Provide `changes` with
`segment_start`, `segment_end`, `new_text` and aligned `original_text`, or an explicit
source voice reference. Pre-generated replacement URLs are no longer accepted.
See [Fish editing](docs/FISH_TTS_RECONSTRUCTION.md) for duration maps and approval.
Missing Fish assets or licence approval keep reconstruction unavailable.

The backend sends the same versioned attempt envelope used as RunPod Serverless input to `POST /v1/attempts` with `Authorization: Bearer $HEAR_POD_API_KEY`. The Pod queues it in local RabbitMQ, claims it through `HEAR_BACKEND_INTERNAL_URL` when a worker is free, heartbeats and reports events/outcome while it runs, and acknowledges acceptance independently of the caller connection. Optional SSE previews use `/v1/attempts/stream`. The backend deduplicates events by ID, stores job history, and serves reconnectable status/SSE to clients; the Pod owns no job database.

Start a Serverless handler:

```bash
HEAR_WORKER_ROLE=transcription python -m hear.entrypoints.serverless
```

The process requires valid backend credentials and the transport settings for its selected entrypoint. A worker does not accept jobs until local readiness checks pass.

## Project structure

| Path | Responsibility |
| --- | --- |
| `hear/contracts` | Attempt, event, worker, and outcome schemas |
| `hear/entrypoints` | Pod and Serverless process startup |
| `hear/api` | Pod job SSE ingress and operational routes |
| `hear/execution` | Shared executor, lease, and backend reporting |
| `hear/workflows` | Pipeline, transcription, reconstruction, and Magic Clean jobs |
| `hear/inference` | Local engines and model manifest |
| `hear/storage` | Scoped artifact storage |
| `hear/health` and `hear/api` | Readiness probes and Pod operations endpoints |
| `HEAR_AI_FULL_MIGRATION_MASTER_PLAN_V11.md` | Runtime migration architecture and file-by-file plan |

## Test without Docker

Install the local worker and development dependencies with Python 3.12 and `uv`:

```bash
export UV_PROJECT_ENVIRONMENT=/opt/hear-ai-v11/venv
uv sync --project deploy/runtime --locked --group pod --group serverless --group dev
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-cache \
  --index-strategy unsafe-best-match \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  'torch==2.8.0+cpu' 'torchaudio==2.8.0+cpu' 'onnxruntime==1.30.0' \
  'librosa==0.11.0' 'pyloudnorm==0.2.0' 'protobuf==6.33.6'
```

Run all offline unit tests that do not require the optional transcription group:

```bash
export UV_PROJECT_ENVIRONMENT=/opt/hear-ai-v11/venv
uv run --project deploy/runtime python -m hear.tools.check_architecture
uv run --project deploy/runtime ruff check hear tests scripts
uv run --project deploy/runtime mypy
uv run --project deploy/runtime python -m pytest tests -q
uv run --project deploy/runtime python -m pytest tests/integration -q
```

Qwen ASR inference requires the locked `transcription` dependency group and pinned ASR/aligner assets. Tests that use real model checkpoints require those assets to be provisioned locally. Start a worker only after setting its role, credentials, transport, model root, and scratch path in the environment.
