# Backend integration

This page describes the worker contract implemented in this repository. Hear Backend owns durable jobs, provider choice, retries, progress history, client-facing SSE, and final-state reconciliation. Hear AI executes attempts and streams canonical events to the backend over the selected provider connection.

## Job contract

The only durable job types are `pipeline`, `transcription`, `reconstruction`, and `magic_clean`. Reconstruction carries one operation from `replace_segments`, `edit_transcript`, `rebuild`, `remove_segments`, or `preview`. Magic Clean has two production profiles: `natural` for DeepFilterNet3 denoising and `sam_audio` for prompt-driven separation.

For hiss, steady background noise, and speech denoising, use `{"profile":"natural","attenuation_limit_db":24}`. For semantic sound separation, send `options` as `{"profile":"sam_audio","prompt":"background music","action":"remove","prompt_mode":"ambient","seed":0}`. Use `prompt_mode=ambient` for continuous sources such as music, engines, rain, crowds, or speech; it generates two candidates and selects one with the official CLAP text ranker. Use `prompt_mode=event` for bounded events such as a dog bark, cough, door slam, or car horn; it uses the official PE Audio Frame span predictor. The SAM prompt describes one target sound. `remove` publishes everything except the target, while `isolate` publishes only the target and is not expected to retain speech when the prompt describes noise. The action and prompt mode default to `remove` and `ambient`. Prompts are trimmed, converted to lowercase, limited to 160 UTF-8 bytes, and should be short noun or verb phrases. When an isolated target is not audible in the source, the attempt fails with `target_not_detected` and publishes no audio artifact.

Workers validate the versioned `AttemptEnvelope` in `hear/contracts/jobs.py`. It includes job/run/attempt identity, source revision and URL, scoped storage credentials, options, an attempt deadline, and an attempt reporting grant. `HEAR_BACKEND_INTERNAL_URL` configures the reporting destination. The grant and storage application key are secrets and must not be logged.

## Backend claim API

`BackendAttemptClient` in `hear/execution/reporter.py` builds reporting destinations from `HEAR_BACKEND_INTERNAL_URL`, disables redirects, and path-escapes attempt IDs. Each request sends `X-AI-Attempt-Grant`:

| Method | Path suffix | Request |
| --- | --- | --- |
| `POST` | `/internal/ai/attempts/{attempt_id}/claim` | `WorkerIdentity`; response is `AttemptClaim` |
| `POST` | `/internal/ai/attempts/{attempt_id}/heartbeat` | worker ID, generation, heartbeat sequence |
| `POST` | `/internal/ai/attempts/{attempt_id}/events` | canonical `ExecutionEvent` |
| `POST` | `/internal/ai/attempts/{attempt_id}/outcome` | canonical `ExecutionOutcome` |

Claim decisions are `execute`, `already_completed`, `cancelled`, `stale`, `not_current`, and `lease_unavailable`. The worker executes only after an `execute` decision. It sends monotonically increasing event sequence numbers within an attempt. Events and outcomes carry the attempt ID and source revision so the backend can reject stale results.

The backend must make claims, event ingestion, and outcome application idempotent under the current attempt fence, retain durable event/outcome state independently of Redis, and reconcile incomplete attempts after restart. Pod workers report events and outcomes through these internal routes while the Pod API relays the same event stream over SSE. RunPod `/stream` delivers the same canonical event payloads. Deduplicate relayed and reported events by event ID.

## Provider paths

Pod workers expose `POST /v1/attempts/stream`, accept the same `AttemptEnvelope` as Serverless, publish it to local RabbitMQ, claim through the backend when consumed, and return a `queued` SSE event followed by canonical events as `text/event-stream`. RunPod Serverless workers use the native handler in `hear/runtime/serverless.py` and stream the same `ExecutionEvent` values through the provider `/stream` API. Provider retries remain backend-owned.

Both paths call the same `AttemptStream`, `JobExecutor`, and workflow implementations. Artifacts are written to scoped Backblaze B2 locations; the outcome contains manifests with object key, byte size, SHA-256, content type, and optional delivery URL. The browser-facing SSE stream belongs to the backend, which persists and relays Pod or Serverless events.

## Source of truth

The executable contract is in `hear/contracts/`, `hear/execution/`, `hear/api/routers/jobs.py`, and `hear/runtime/`. The migration requirements and open production gates are in `HEAR_AI_FULL_MIGRATION_MASTER_PLAN_V11.md`.
