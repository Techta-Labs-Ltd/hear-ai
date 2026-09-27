# Runtime environment

Production images use the role-specific groups in `deploy/runtime/pyproject.toml`. Select a Docker target before supplying environment; Pod workers and RunPod Serverless workers share workflows but use different entrypoints and transport configuration.

## Common settings

| Variable | Purpose |
| --- | --- |
| `HEAR_WORKER_ROLE` | Capability role loaded by the image |
| `HEAR_WORKER_ID` | Optional stable identity for one role worker; generated from the RunPod Pod ID and role when omitted |
| `HEAR_WORKER_GENERATION` | Optional generation; otherwise generated at startup |
| `HEAR_IMAGE_REVISION` | Immutable worker software identifier, normally a container digest or source commit |
| `HEAR_ENGINE_REVISION` | Immutable model and runtime configuration identifier, normally derived from the model manifest or release |
| `HEAR_MODEL_ROOT` | Local model directory; the Pod profile uses `/models` |
| `HEAR_TEMP_DIR` | Writable attempt workspace root |
| `HEAR_MIN_FREE_SCRATCH_BYTES` | Minimum free scratch space for readiness |
| `HEAR_BACKEND_INTERNAL_URL` | Backend API base URL used for claim, heartbeat, event, and outcome callbacks; this repository's configured value is `https://api.hear.media` |
| `HEAR_BACKEND_SERVICE_KEY` | Credential used by `PipelineCatalogClient` to load the pipeline catalog at startup |
| `HEAR_POD_API_KEY` | Bearer credential required by the Pod SSE job endpoint |
| `HEAR_RABBITMQ_URL` | Pod-local RabbitMQ AMQP URL; the broker listens on loopback |
| `HEAR_POD_MAX_CONCURRENT_JOBS` | Maximum concurrent Pod jobs; defaults to one |
| `HEAR_SERVERLESS_MAX_CONCURRENT_JOBS` | Maximum concurrent jobs inside one Serverless worker; defaults to one |
| `HEAR_POD_STACK_ROLES` | Comma-separated Pod lanes; defaults to `pipeline`, which also accepts transcription |
| `HEAR_HOST_JOB_LOCK_PATH` | Shared file lock that limits the Pod to one active model job across role processes |
| `HEAR_API_MAX_BODY_BYTES` | Maximum JSON request body size for the Pod attempt endpoint |
| `HEAR_OPTIONAL_ENGINE_MODE` | `available` uses DeepFilterNet3 for Natural and SAM Audio Base for prompt-driven removal or isolation; `certified` also requires deployment certificates |
| `HEAR_MAGIC_CLEAN_MODEL_DEVICE` | Device used by DeepFilterNet and SAM Audio; defaults to `cuda:0` |
| `AUDIO_DOWNLOAD_MAX_BYTES` | Maximum source download size |
| `AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS` | Source download read timeout |
| `AUDIO_DECODE_TIMEOUT_SECONDS` | Native decode timeout |

The Pod has one authenticated HTTP endpoint at `POST /v1/attempts/stream`. The API selects the local RabbitMQ queue from the request's `job_type` and Magic Clean profile, then returns queued and execution events over the same SSE connection. Role processes consume RabbitMQ directly and expose no HTTP ports. The pipeline lane accepts transcription jobs so a Pod does not load the same Qwen model in separate pipeline and transcription processes. RabbitMQ uses bounded v2 role queues, delayed retries, and dead letter queues. Its AMQP listener binds to loopback and is required by the Pod profile. Serverless uses RunPod dispatch and emits the same canonical events without RabbitMQ. Available Magic Clean workers require their provisioned DeepFilterNet and SAM Audio Base assets. Certified optional engines and the pipeline models are provisioned before startup and validated from `hear/model_manifest.json`. `HEAR_MODEL_FEATURES=qwen_llm` is valid only for the pipeline LLM targets.

Model provisioning and readiness fail closed for entries whose `license_status` is `review_required` or `permission_required`. Update the manifest to `verified` only after the required license review or written permission is recorded through the deployment's review process. The default manifest intentionally keeps Fish Speech, DNSMOS, and SAM gated.

The Serverless entrypoint checks readiness before starting the RunPod worker and registers the same check as a RunPod fitness check. Workers fail startup when role assets, dependency patches, scratch capacity, or loaded engines are not ready.

## Profile settings

Magic Clean workers set `HEAR_CLEANER_CERTIFICATION_PATH` and optionally `HEAR_CLEANER_LOCK_DIR`. Their resource limits are `MAGIC_CLEAN_SCRATCH_BYTES`, `MAGIC_CLEAN_MAX_INPUT_BYTES`, `MAGIC_CLEAN_MAX_FRAMES`, and the GPU budget variables. Natural and SAM Audio use separate worker environments and model assets.

## Dependency patch

The Qwen/WhisperX patch manifest is in `patches/manifest.json`. Image builds apply it with `python -m hear.tools.dependency_patches` and verify it with `--check`. Pipeline and transcription readiness repeat the check without applying changes. A missing or changed patch keeps those workers unready.

## Local development

Use Python 3.12 and select the group needed for the role under test. For example, run `python scripts/setup_runtime.py --role pipeline --provider pod`; add `--feature qwen_llm` for the optional LLM pipeline image. The production dependency contract and sole lock are `deploy/runtime/pyproject.toml` and `deploy/runtime/uv.lock`. CI installs Pod, Serverless, and development groups to run static checks.

Start with `.env.example`. On this Pod, store live runtime values in `/root/hear-ai-v11/runtime.env`; `scripts/run_pod_stack.sh` loads it before the repository `.env`. The Pod stack uses `/opt/hear-ai-v11/venvs/<role>` for isolated role environments, Serverless setup uses `/opt/hear-ai-v11/venvs/<role>-serverless`, and `/models` holds every Pod model asset. Keep credentials in an external secret store; do not copy live service keys, storage keys, or reporting grants into checked-in environment files.
