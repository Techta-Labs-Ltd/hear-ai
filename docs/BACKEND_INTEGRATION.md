# Backend integration

This page describes the worker contract implemented in this repository. Hear Backend owns durable jobs, provider choice, retries, progress history, client-facing SSE, and final-state reconciliation. Hear AI executes attempts and streams canonical events to the backend over the selected provider connection.

## Job contract

The only durable job types are `pipeline`, `transcription`, `reconstruction`, and `magic_clean`. Reconstruction carries one operation from `replace_segments`, `edit_transcript`, `rebuild`, `remove_segments`, or `preview`. Magic Clean has four DeepFilterNet3 profiles: `natural`, `studio_voice`, `outdoor_mobile`, and `clean_raw`, sharing the existing Natural worker route.

For default spoken-word enhancement, send `options` as
`{"profile":"studio_voice","auto_level":true,"remove_clicks":false,"trim_silence":false}`.
Legacy `natural` supports attenuation 12, 18 or 24. The other choices are
`outdoor_mobile` and `clean_raw`. Removed SAM requests are rejected, not remapped.
Read `/capabilities` for labels, options and availability. See the
[profile integration guide](DEEPFILTER_CLEANING_PROFILES.md), including trim offsets,
measured-output metadata and frontend/backend allowlist changes.

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
