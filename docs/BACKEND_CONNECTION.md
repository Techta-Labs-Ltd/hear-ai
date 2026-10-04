# Connecting the backend to the Hear AI workers

Everything a backend needs to send jobs to the Pod and to RunPod Serverless today.
Secrets are named, never written here; the owner hands them over through the
team's secret store.

## Endpoints

| Transport | Role | Address | Card |
| --- | --- | --- | --- |
| Pod (A40, all roles) | gateway | `https://0as9lqk138vfwz-8000.proxy.runpod.net` | A40 48 GB |
| Serverless | `pipeline` (also transcription) | endpoint `f2rfwwfr8zz51e` | RTX A5000 / A4500 |
| Serverless | `pipeline` with LLM tags/discovery | endpoint `w4vh65dlnfchs4` | RTX A5000 / 3090 |
| Serverless | `magic_clean_natural` | endpoint `rhe8iqebqrif70` | RTX A4500 / A5000 |
| Serverless | `reconstruction` (Fish) | endpoint `erkgx070wpn494` | A40 / RTX A6000 |

Serverless API base: `https://api.runpod.ai/v2/{endpoint_id}`. Use one pipeline
endpoint per deployment: `w4vh65dlnfchs4` for LLM-quality tags and discovery,
`f2rfwwfr8zz51e` for the cheaper path.

## Secrets (values held by the owner)

| Name | Used for | Where the owner has it |
| --- | --- | --- |
| Pod API key | `Authorization: Bearer …` on `POST /v1/attempts` | `HEAR_POD_API_KEY` in the Pod's `production.env` |
| RunPod API key | `Authorization: Bearer …` on `api.runpod.ai/v2/*` | RunPod console → Settings → API keys |
| Backend service key | Workers fetch `/internal/ai/runtime/catalog` with `X-Service-Key` | `HEAR_BACKEND_SERVICE_KEY` |
| Grant secret | HMAC for `reporting_grant` (the backend signs and verifies its own grants) | `AI_SERVICE_SECRET` in the backend, ≥ 24 characters |
| Backblaze key pair | `storage.key_id` / `storage.application_key` in every envelope | backend `HEAR_STORAGE_KEY_ID` / `HEAR_STORAGE_APPLICATION_KEY` |

## What every envelope must contain to be accepted

The workers enforce this policy (`HEAR_BACKEND_POLICY_JSON` on the Pod and in every
Serverless template). An envelope that differs is refused before any work starts.

| Envelope field | Required value |
| --- | --- |
| `backend_id` | `backend-a` |
| `backend_base_url` | `https://api.hear.media/api/v1` (callbacks go to `…/internal/ai/attempts/{attempt_id}/{claim,heartbeat,events,outcome}`) |
| `storage.bucket_name` | `OldAlexa` |
| `storage.endpoint_url` | `https://s3.eu-central-003.backblazeb2.com` |
| `storage.public_base_url` | `https://cdn.hear.media` |
| `source.url` host | one of `cdn.hear.media`, `oldalexa.s3.eu-central-003.backblazeb2.com`, `s3.eu-central-003.backblazeb2.com` |

Full envelope, callback and result shapes: `docs/GO_AI_DISPATCH_PLAN.md` §6–7;
Magic Clean options and results: `docs/CLEANING_PROFILES.md`; byte-level test
vectors for the scope digest and grant: `docs/contract-vectors.json`.

## Sending a job

Pod:

```bash
curl -X POST https://0as9lqk138vfwz-8000.proxy.runpod.net/v1/attempts \
  -H "Authorization: Bearer $POD_API_KEY" -H "Content-Type: application/json" \
  -d @envelope.json
# 202 {"status": "accepted", "attempt_id": "...", ...} = durably queued
# 401 wrong key, 422 envelope invalid or deadline passed, 429 queue full (Retry-After), 503 lane not ready
```

Serverless:

```bash
curl -X POST https://api.runpod.ai/v2/rhe8iqebqrif70/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": $(cat envelope.json)}"
# 200 {"id": "<runpod job id>", "status": "IN_QUEUE"}
```

Health before sending: Pod `GET /readyz` (200 = all lanes ready, body lists each
lane) and `GET /capabilities` (concurrency limits, profiles); Serverless
`GET https://api.runpod.ai/v2/{endpoint_id}/health`.

Results never come back in these responses. Workers call the backend: `claim`
(must answer `{"decision": "execute", "lease_seconds": 120, "heartbeat_seconds": 20}`),
`heartbeat`, `events`, `outcome`, each with headers `X-AI-Attempt-Grant`,
`X-AI-Worker-ID`, `X-AI-Worker-Generation`.

## Settings for the current Python backend

The Python backend already has a client for this protocol behind a flag that is
off by default and has not yet carried a live job. To use it until the Go service
replaces it:

```env
HEAR_AI_RUNTIME_V1=true
HEAR_AI_TRANSPORT=pod                       # or serverless; one transport at a time
HEAR_BACKEND_ID=backend-a                   # must match the worker policy (default "hear-backend" is refused)
HEAR_AI_CALLBACK_BASE_URL=https://api.hear.media/api/v1
HEAR_HTTP_URL=https://0as9lqk138vfwz-8000.proxy.runpod.net/api/v1   # client strips /api/v1 and posts to /v1/attempts
HEAR_AI_INGRESS_TOKEN=<Pod API key>
HEAR_RUNPOD_API_KEY=<RunPod API key>
HEAR_RUNPOD_ENDPOINTS_JSON={"pipeline":"w4vh65dlnfchs4","magic_clean_natural":"rhe8iqebqrif70","reconstruction":"erkgx070wpn494"}
AI_SERVICE_SECRET=<grant secret, at least 24 characters>
HEAR_STORAGE_ENDPOINT_URL=https://s3.eu-central-003.backblazeb2.com
HEAR_STORAGE_BUCKET_NAME=OldAlexa
HEAR_STORAGE_PUBLIC_BASE_URL=https://cdn.hear.media
HEAR_STORAGE_KEY_ID=<Backblaze key id>
HEAR_STORAGE_APPLICATION_KEY=<Backblaze application key>
```

Known gaps in that Python path, fixed by the Go plan: one transport at a time (no
Pod/Serverless split), no daily Serverless budget, and Magic Clean options must be
the profile form (`{"profile": "studio_voice"}`); the old slider fields are rejected.

## Before the first real job

1. Start the Pod stack: `HEAR_ENV_FILE=/root/hear-ai-config/production.env nohup bash scripts/run_pod_stack.sh > /root/hear-ai-runtime/pod-stack.log 2>&1 &`,
   then `curl https://0as9lqk138vfwz-8000.proxy.runpod.net/readyz` must return 200.
   (Port 8000 is already exposed; it answers 502 until the stack is running.)
2. Deploy the backend with the settings above and run one job of each type; check
   the claim, events and outcome arrive and the result is applied.
