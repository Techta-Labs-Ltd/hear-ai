# RunPod configuration: Pod and Serverless

The live worker configuration as of 2026-10-04 (worker code commit `0736003`).
Secret values are shown as `<secret>`; the owner holds them. The backend side
(what to send, which values the workers accept) is in
[BACKEND_CONNECTION.md](BACKEND_CONNECTION.md).

## Pod

| Setting | Value |
| --- | --- |
| Pod ID | `0as9lqk138vfwz` (RunPod, data centre CA-MTL-1) |
| GPU | 1 × NVIDIA A40 48 GB |
| Container RAM limit | 50 GB |
| Gateway (public) | `https://0as9lqk138vfwz-8000.proxy.runpod.net` (HTTP port 8000 exposed) |
| Roles on the one GPU | pipeline (1 process, up to 7 jobs), magic clean (2 processes, 8 chunk workers each), reconstruction (1 process, 1 job) |
| Job queue | RabbitMQ on the pod, loopback only (`127.0.0.1:5672`) |
| Models | `/models` on the container disk (never on `/workspace`) |
| Env file | `/root/hear-ai-config/production.env` (outside the repo) |
| Start | `HEAR_ENV_FILE=/root/hear-ai-config/production.env nohup bash scripts/run_pod_stack.sh > /root/hear-ai-runtime/pod-stack.log 2>&1 &` |
| Check | `curl https://0as9lqk138vfwz-8000.proxy.runpod.net/readyz` → 200 when every lane is ready |

`production.env`:

```env
ENVIRONMENT="production"
HEAR_RUNTIME_MODE="production"
LOG_LEVEL="INFO"
HEAR_ENABLE_DOCS="false"
HTTP_HOST="0.0.0.0"
HTTP_PORT="8000"
HEAR_WORKER_ROLE="pipeline"
HEAR_ENGINE_REVISION="0736003b590e1e775e751f678f9a112fb85816d9"
HEAR_IMAGE_REVISION="pod-0736003b590e1e775e751f678f9a112fb85816d9"
HEAR_MODEL_ROOT="/models"
FISH_SPEECH_MODEL_ROOT="/models"
FISH_SPEECH_HOME="/opt/fish-speech-upstream"
HEAR_FISH_LICENSE_APPROVED="true"
HEAR_POD_STACK_ROLES="pipeline,magic_clean_natural,reconstruction"
HEAR_POD_API_KEY="<secret>"                      # bearer token the backend sends to /v1/attempts
HEAR_BACKEND_INTERNAL_URL='https://api.hear.media/api/v1'
HEAR_BACKEND_SERVICE_KEY="<secret>"              # X-Service-Key for /internal/ai/runtime/catalog
HEAR_BACKEND_POLICY_JSON="<policy>"              # backend-a policy, see BACKEND_CONNECTION.md
HEAR_RABBITMQ_URL="amqp://guest:guest@127.0.0.1:5672/%2F"
HEAR_TEMP_DIR="/root/hear-ai-runtime/scratch"
HEAR_HOST_JOB_LOCK_PATH="/root/hear-ai-runtime/admission.lock"
HEAR_MIN_FREE_SCRATCH_BYTES="1073741824"
MAGIC_CLEAN_SCRATCH_BYTES="34359738368"
HEAR_POD_MAX_CONCURRENT_JOBS="1"
HEAR_HOST_MAX_CONCURRENT_JOBS="10"
HEAR_POD_ROLE_LIMITS="{\"pipeline\":7,\"magic_clean_natural\":2,\"reconstruction\":1}"
HEAR_POD_PROCESS_LIMITS="{\"pipeline\":7,\"magic_clean_natural\":1,\"reconstruction\":1}"
HEAR_WORKER_REPLICAS="{\"pipeline\":1,\"magic_clean_natural\":2,\"reconstruction\":1}"
HEAR_OPTIONAL_ENGINE_MODE="available"
HEAR_MAGIC_CLEAN_MODEL_DEVICE="cuda:0"
HEAR_GPU_IDLE_EVICTION_ENABLED="true"
HEAR_PIPELINE_IDLE_TTL_SECONDS="600"
HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS="300"
HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS="1200"
HEAR_AUDIOSEP_IDLE_TTL_SECONDS="90"
WHISPER_BATCH_SIZE="16"
WHISPER_LONG_AUDIO_BATCH_SIZE="16"
WHISPER_CHUNK_SECONDS="600"
QWEN_ASR_DTYPE="bfloat16"
QWEN_ASR_DEVICE_MAP="cuda:0"
OMP_NUM_THREADS="2"
OPENBLAS_NUM_THREADS="1"
MKL_NUM_THREADS="2"
HF_HUB_OFFLINE="1"
TRANSFORMERS_OFFLINE="1"
HF_DATASETS_OFFLINE="1"
```

Defaults that apply without being set: `HEAR_MAGIC_CLEAN_PARALLELISM=8`,
`HEAR_MAGIC_CLEAN_CHUNK_SECONDS=300`, `HEAR_MASTERING_PARALLELISM=8`,
`WHISPER_SEGMENT_SECONDS=30`, `WHISPER_VAD_WORKERS=4`,
`RECONSTRUCTION_MAX_SOURCE_SECONDS=14400`, `RECONSTRUCTION_SCRATCH_BYTES=34359738368`.
The LLM pipeline is off on the Pod (`HEAR_MODEL_FEATURES` unset); route pipeline jobs
to the `pipeline-llm` Serverless endpoint for LLM tags and discovery.

Measured on this Pod with a 3-hour track: pipeline 3 min 25 s, transcription
3 min 20 s, Magic Clean 2 min 30 s, reconstruction 2 min 16 s; peak 15.5 GB GPU and
11 GB RAM per job type.

## Serverless endpoints

All four: 1 GPU per worker, `workersMin 0`, idle timeout 180 s, scaler
`QUEUE_DELAY` 4 s, execution timeout 2 h, FlashBoot on, images pulled from GHCR
with registry auth `cmusm0r4r008o7xydkedy1g67`, digest-pinned.

| Endpoint | ID | Template | GPUs | Max workers | Disk | Image |
| --- | --- | --- | --- | --- | --- | --- |
| `hear-ai-pipeline` | `f2rfwwfr8zz51e` | `rdumefwltc` | RTX A5000, RTX A4500 | 3 | 50 GB | `ghcr.io/techta-labs-ltd/hear-ai@sha256:11096346377c55f18b84b6444d66e585b213f9ca59a1a438b330c7a1eeea73ac` |
| `hear-ai-pipeline-llm` | `w4vh65dlnfchs4` | `mx4mashiis` | RTX A5000, RTX 3090 | 2 | 50 GB | `…@sha256:a85f1e3a545cbb7ea8dc044919b3b9fee594d7472f489cbef8efd1bdc2dcb425` |
| `hear-ai-cleaner` | `rhe8iqebqrif70` | `l1swqghx1x` | RTX A4500, RTX A5000 | 3 | 50 GB | `…@sha256:aca00823540e1d4e0905e2bc22afd9ae8f09cf1eb27573ad6873a6edadbb08c1` |
| `hear-ai-reconstruction` | `erkgx070wpn494` | `zq360ynxji` | A40, RTX A6000 | 2 | 60 GB | `…@sha256:14fb5346e2bf51c2fdf6d0915aa1db715a8afba77eb24b28d7e236ef1b1c86a5` |

Env common to all four templates:

```env
ENVIRONMENT=production
HEAR_RUNTIME_MODE=production
LOG_LEVEL=INFO
HEAR_ENABLE_DOCS=false
HTTP_HOST=0.0.0.0
HTTP_PORT=8000
HEAR_BACKEND_INTERNAL_URL=https://api.hear.media/api/v1
HEAR_BACKEND_SERVICE_KEY=<secret>
HEAR_BACKEND_POLICY_JSON=<policy>          # same backend-a policy as the Pod
HEAR_MODEL_ROOT=/models
FISH_SPEECH_HOME=/opt/fish-speech
HEAR_TEMP_DIR=/audio
HEAR_MIN_FREE_SCRATCH_BYTES=1073741824
MAGIC_CLEAN_SCRATCH_BYTES=34359738368
HEAR_MAGIC_CLEAN_MODEL_DEVICE=cuda:0
HEAR_SERVERLESS_PRELOAD_MODELS=true
HEAR_GPU_IDLE_EVICTION_ENABLED=false       # required with preload
HEAR_SERVERLESS_MAX_CONCURRENT_JOBS=1
HEAR_PIPELINE_IDLE_TTL_SECONDS=600
HEAR_MAGIC_CLEAN_IDLE_TTL_SECONDS=300
HEAR_RECONSTRUCTION_IDLE_TTL_SECONDS=1200
HEAR_AUDIOSEP_IDLE_TTL_SECONDS=90
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=2
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

Per endpoint:

```env
# hear-ai-pipeline (f2rfwwfr8zz51e)
HEAR_WORKER_ROLE=pipeline
HEAR_ENGINE_REVISION=11096346377c55f18b84b6444d66e585
QWEN_ASR_DTYPE=bfloat16
QWEN_ASR_DEVICE_MAP=cuda:0
WHISPER_BATCH_SIZE=16
WHISPER_LONG_AUDIO_BATCH_SIZE=16
WHISPER_CHUNK_SECONDS=600

# hear-ai-pipeline-llm (w4vh65dlnfchs4)
HEAR_WORKER_ROLE=pipeline
HEAR_MODEL_FEATURES=qwen_llm
QWEN_LLM_GPU_MEMORY_GIB=8.5                # 4-bit Qwen2.5-7B-Instruct-AWQ
HEAR_ENGINE_REVISION=a85f1e3a545cbb7ea8dc044919b3b9fe
QWEN_ASR_DTYPE=bfloat16
QWEN_ASR_DEVICE_MAP=cuda:0
WHISPER_BATCH_SIZE=16
WHISPER_LONG_AUDIO_BATCH_SIZE=16
WHISPER_CHUNK_SECONDS=600

# hear-ai-cleaner (rhe8iqebqrif70)
HEAR_WORKER_ROLE=magic_clean_natural
HEAR_ENGINE_REVISION=aca00823540e1d4e0905e2bc22afd9ae
HEAR_MAGIC_CLEAN_PARALLELISM=8
HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
HEAR_SOUND_CLEANUP_BUNDLE_SHA256=6cf1259445ef7dc868521e59026007fb594c8006d6790e87f97b9e578751ac7e
HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
HEAR_SOUND_CLEANUP_SEPARATOR_SHA256=2278016f6becb669c5e073ff6b1931a95b03e9f4032f7e5f9ba35c82da22621f

# hear-ai-reconstruction (erkgx070wpn494)
HEAR_WORKER_ROLE=reconstruction
HEAR_ENGINE_REVISION=14fb5346e2bf51c2fdf6d0915aa1db71
HEAR_FISH_LICENSE_APPROVED=true
FISH_SPEECH_MODEL_ROOT=/models
```

The committed plans in `deploy/runpod/*.json` are the source for these values;
`scripts/deploy_serverless.py --plan deploy/runpod/<name>.json --image ghcr.io/techta-labs-ltd/hear-ai@sha256:<digest> --env-from HEAR_BACKEND_SERVICE_KEY,HEAR_BACKEND_POLICY_JSON`
rolls an endpoint (it stamps `HEAR_ENGINE_REVISION` from the digest and refuses a
plan whose workers would not start). Raise `workersMax` in the plan and redeploy to
add throughput; cost is per GPU-second used, not per worker.
