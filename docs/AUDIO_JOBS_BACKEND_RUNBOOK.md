# Audio jobs runtime runbook

## Worker roles

| Logical job | Worker role | Image targets |
| --- | --- | --- |
| Pipeline | `pipeline` | `pipeline-pod`, `pipeline-serverless` |
| Transcription | `transcription` | `transcription-pod`, `transcription-serverless` |
| Reconstruction | `reconstruction` | `reconstruction-pod`, `reconstruction-serverless` |
| Magic Clean | `magic_clean_natural` | natural Pod and Serverless |
| Magic Clean | `magic_clean_sam_audio` | SAM Audio Pod and Serverless |

Two optional pipeline images enable the `qwen_llm` feature. Magic Clean routes to the `natural` and `sam_audio` profiles.

## Build and provision

Build from the repository root with the target matching the role and provider:

```bash
docker build --target transcription-pod -t hear-ai:transcription-pod .
docker build --target magic-clean-natural-serverless -t hear-ai:magic-clean-natural-serverless .
```

The Pod image serves authenticated HTTP job requests and SSE. The Serverless image uses the native Runpod handler. Both apply and verify the Qwen dependency patch during image build. Provision model files before startup; normal startup must not download models.

On this Pod, set `HEAR_MODEL_ROOT=/models`; model weights and inference caches stay on the root filesystem. In Serverless, use the attached persistent volume under `/runpod-volume/hear-ai-v11/models`.

Magic Clean certification JSON and pinned assets are deployment inputs. Provision them before starting a profile worker and set `HEAR_CLEANER_CERTIFICATION_PATH` plus its externally pinned `HEAR_CLEANER_CERTIFICATION_SHA256`. Each profile’s resource record points to an absolute evidence file and binds its bytes with `evidence_sha256`. Before loading a GPU engine, the worker checks live GPU memory and requires the certified peak plus a 2 GB reserve to fit.

## Pod API

Start Pod-local RabbitMQ first with `scripts/setup_rabbitmq.sh`, then start the Pod API/worker with role settings, `HEAR_RABBITMQ_URL`, `HEAR_POD_API_KEY`, `HEAR_BACKEND_INTERNAL_URL`, `HEAR_WORKER_ID`, `HEAR_IMAGE_REVISION`, and `HEAR_ENGINE_REVISION` configured:

```bash
set -a
source deploy/runtime/env/runpod-pod.env.example
set +a
uv run --project deploy/runtime --no-sync python -m hear.entrypoints.pod
```

Submit a backend-created `AttemptEnvelope` to `POST /v1/attempts/stream` with `Authorization: Bearer <HEAR_POD_API_KEY>` and `Content-Type: application/json`. The Pod publishes it to its durable local RabbitMQ role queue and returns `text/event-stream` beginning with a `queued` event, followed by the same canonical execution events used by RunPod Serverless. The worker claims, heartbeats, and reports events/outcomes through the backend API while the Pod relays SSE; deduplicate both paths by event ID. Serverless bypasses RabbitMQ and emits the same event schema through the RunPod handler.

Set `HEAR_POD_MAX_CONCURRENT_JOBS=1` initially. The API returns 429 with `Retry-After` when the GPU is at capacity. `/healthz`, `/readyz`, `/capabilities`, and `/drain` expose operational state. The Pod does not own durable job state or implement the user-facing job API.

## Scratch and failure handling

Set `HEAR_TEMP_DIR` to writable scratch. Readiness requires the free-space amount configured by `HEAR_MIN_FREE_SCRATCH_BYTES`; Magic Clean roles use at least `MAGIC_CLEAN_SCRATCH_BYTES`. Audio attempts use isolated job/attempt workspaces and are cleaned in workflow finalization. The cleanup command sweeps expired workspaces; destructive purge requires the explicit `--mode purge --yes` flags.

The backend owns retries and reconciles attempts after interrupted HTTP streams. Monitor attempt heartbeats, stream disconnects, outcome acceptance, scratch peaks, B2 upload failures, model load failures, and API admission rejections.

## Release gates

CI runs lint, typing, architecture checks, contract/workflow tests, and builds each supported Docker target. Production release still requires Pod and Serverless golden audio comparison, model/certificate provisioning, B2 interruption checks, stream disconnect/cancellation tests, and performance measurements listed in the V11 plan. The model manifest marks Fish Audio S2 Pro as `permission_required`; its [published license](https://huggingface.co/fishaudio/s2-pro/blob/1de9996b6be38b745688de084d87a5633f714e4e/LICENSE.md) requires a separate written license for commercial use.
