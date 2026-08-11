# Hear backend integration

This is the current contract between the Hear backend and Hear AI.

## Quick links

- [All job response payloads](#all-job-response-payloads)
- [Submit through `/process`](#submit-through-process)
- [Stage events and automatic continuation](#stage-events-and-continuation)
- [Complete gRPC RPC reference](#complete-grpc-rpc-reference)

## Transport model

Use one submission transport, then gRPC for progress and recovery:

```text
Backend -- POST /process ----------> Hear AI HTTP :8000
Backend <-- Pipeline.Subscribe ----- Hear AI gRPC :50051
Backend -- Pipeline.GetResult -----> Hear AI gRPC :50051
```

Hear AI persists and queues the job, runs every applicable stage in order, and
automatically continues after each stage. The backend must not submit a new
request when a stage completes.

`Pipeline.SubmitJob` is an alternative for a backend that submits entirely
through gRPC. Never call both `/process` and `SubmitJob` for the same job.

## Configuration and authentication

```dotenv
HEAR_HTTP_URL=http://hear-ai:8000
HEAR_GRPC_TARGET=hear-ai:50051
HEAR_BACKEND_ID=<registered-backend-id>
HEAR_SERVICE_KEY=<service key whose SHA-256 hash is registered for that backend>
HEAR_GRPC_APPLICATION=hear
```

| Transport | Authentication |
| --- | --- |
| HTTP `POST /process` | `X-Service-Key: <registered backend service key>` |
| gRPC | metadata `x-api-key: <registered backend service key>` and `application: hear` |

Use TLS at the ingress or a secure gRPC channel outside a trusted network.

## Health and readiness

```bash
curl --fail "$HEAR_HTTP_URL/health"
curl --fail "$HEAR_HTTP_URL/ready"
```

Submit production work only after `/ready` returns `200`. `/health` reports
GPU, queue, pipeline, and resolver state.

## Submit through `/process`

The endpoint is `/process`, not `/`. A `POST / 422` log means the caller used
the base URL instead of the submission endpoint.

The backend creates and persists a unique `job_id` before submission. Every
job requires non-empty `job_id`, `track_id`, `user_id`, and `job_type`. Audio
jobs require a directly downloadable `audio_url`.

```bash
curl --fail-with-body \
  -X POST "$HEAR_HTTP_URL/process" \
  -H "Content-Type: application/json" \
  -H "X-Service-Key: $HEAR_SERVICE_KEY" \
  -d '{
    "job_id": "01JBACKENDGENERATEDULID",
    "backend_id": "backend-a",
    "storage": {
      "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
      "bucket_name": "backend-a-bucket",
      "key_id": "temporary-key-id",
      "application_key": "temporary-application-key",
      "folder_prefix": "users/user-456/jobs/01JBACKENDGENERATEDULID",
      "public_base_url": "https://media.example.com/backend-a-bucket",
      "expires_at": "2026-07-27T12:00:00Z"
    },
    "track_id": "track-123",
    "user_id": "user-456",
    "job_type": "pipeline",
    "audio_url": "https://storage.example/audio/track-123.mp3",
    "max_tags": 8
  }'
```

Do not send `track_exists`. Hear AI does not own the track database and does
not confirm whether a track exists. `track_id` is a correlation key.

`backend_id` and every `storage` field are mandatory on every `/process`,
`/discovery`, and gRPC `SubmitJob` request. The service key must resolve to the
same backend. The endpoint, bucket, and public base URL must be allowlisted for
that backend, `expires_at` must cover the complete job lifecycle, and
`folder_prefix` must be a relative, traversal-free user/job folder. All shorter
examples below omit the repeated storage block only for readability; callers
must include it. Old requests without `backend_id` and `storage` fail validation.

`audio_url` must return audio bytes directly, not HTML. Hear AI does not fetch
recordings or transcriptions from another backend endpoint.

A new job returns `202`:

```json
{
  "job_id": "01JBACKENDGENERATEDULID",
  "backend_id": "backend-a",
  "run_id": "2c420293-12b1-49b9-9fc7-dba3b38e619b",
  "track_id": "track-123",
  "job_type": "pipeline",
  "status": "queued",
  "replayed": false
}
```

`job_id` is the idempotency key:

- Identical payload and ID: `200`, original `run_id`, `replayed: true`.
- Different payload with an existing ID: `409`.
- Timeout or `503`: retry the identical payload with the same ID.
- `401`, `409`, or `422`: fix the request; do not blindly retry.

## Job types

| Job type | Additional input | Processing |
| --- | --- | --- |
| `pipeline` | `audio_url` | transcription/correction → moderation → categorization → discovery → optimized MP3 variants |
| `transcription` | `audio_url` | transcription, language, segments, and word alignment |
| `magic_clean` | `audio_url`; optional stem levels | download → separation → enhancement → mix → loudness/peak finalization |
| `discovery` | `audio_url` | transcription → standalone discovery profile |
| `categorization` | `audio_url` or `edited_transcript` | transcription when needed → moderation → categorization → discovery |
| `audio_tag` | `audio_url` | short transcription → up to two suggestions |
| `reconstruct` | `audio_url`, `changes` | download → replace selected segments → quality validation |
| `edit_transcript` | `audio_url`, `edited_transcript` | download → transcribe → diff → regenerate changed speech |
| `rebuild` | `audio_url`, `edited_transcript` | transcription/correction using supplied text → moderation → categorization → discovery |

`magic-clean` is normalized to `magic_clean`.

For Magic Clean, `speech`, `music`, and `background` are integer percentages
from `0` to `100`. Supply all three or omit all three. `speech` and `music`
control how much of their separated stems is retained. `background` controls
both the retained residual-background level and the cleaning strength: lower
background means stronger noise suppression. The service derives suppression
as `min(90%, 100% - background)`; the 90% ceiling protects speech from
metallic, harmonic, and warbling artifacts.

| `background` | Residual retained | Noise suppression |
| ---: | ---: | ---: |
| `100` | 100% | 0% |
| `50` | 50% | 50% |
| `10` | 10% | 90% |
| `0` | 0% | 90% (safe ceiling) |

When all three values are omitted, the API applies the production default
`speech: 100`, `music: 10`, and `background: 10`. The normalized job sent
through the pipeline therefore always contains explicit Magic Clean levels.
This keeps the complete separation and cleaning process active and applies 90%
background-noise suppression while retaining the full separated speech stem.

Set the optional boolean `cut_silence` to `true` to remove detected quiet gaps
after enhancement and before final loudness normalization. It defaults to
`false`, so existing jobs preserve the source timeline.

Each reconstruct change requires `segment_end > segment_start` and non-empty
`new_text`:

```json
{
  "segment_start": 12.4,
  "segment_end": 15.8,
  "new_text": "replacement speech",
  "original_text": "original speech"
}
```

## Request examples for every job type

Every example below also requires the `X-Service-Key` header. Values named
`job-*` must be unique and persisted by the backend.

### `pipeline`

```json
{"job_id":"job-pipeline","track_id":"track-1","user_id":"user-1","job_type":"pipeline","audio_url":"https://storage.example/track.mp3","max_tags":8}
```

Returns a `JobResult.pipeline` payload containing the source URL,
transcription, moderation, optional categorization/discovery, content
description, compressed MP3 information, and a flagged/report result when no
content is detected.

### `transcription`

```json
{"job_id":"job-transcription","track_id":"track-1","user_id":"user-1","job_type":"transcription","audio_url":"https://storage.example/track.mp3"}
```

Returns `JobResult.transcription` with transcript text, language, confidence,
segments, and aligned words.

### `magic_clean`

Minimal request using the production defaults:

```json
{"job_id":"job-clean","track_id":"track-1","user_id":"user-1","job_type":"magic_clean","audio_url":"https://storage.example/track.mp3"}
```

Equivalent normalized payload processed internally:

```json
{"job_id":"job-clean","track_id":"track-1","user_id":"user-1","job_type":"magic_clean","audio_url":"https://storage.example/track.mp3","speech":100,"music":10,"background":10}
```

Custom levels:

```json
{"job_id":"job-clean","track_id":"track-1","user_id":"user-1","job_type":"magic_clean","audio_url":"https://storage.example/track.mp3","speech":50,"music":10,"background":10}
```

Returns `JobResult.magic_clean` with enhanced audio URL/key, audio-quality
measurements, stage timings, and any transcription/moderation performed by the
cleaning flow. The three mix percentages must be supplied together. In this
example, `background: 10` retains 10% of the residual background and applies
the maximum safe 90% noise suppression to the separated speech stem.

### `discovery`

```json
{"job_id":"job-discovery","track_id":"track-1","user_id":"user-1","job_type":"discovery","audio_url":"https://storage.example/track.mp3","source":"upload"}
```

This may be sent to `/process`, or to the `/discovery` convenience endpoint
without `job_type`. It returns `JobResult.pipeline` with transcription,
discovery profile, and content description.

### `categorization` from audio

```json
{"job_id":"job-category-audio","track_id":"track-1","user_id":"user-1","job_type":"categorization","audio_url":"https://storage.example/track.mp3","max_tags":8}
```

### `categorization` from supplied text

```json
{"job_id":"job-category-text","track_id":"track-1","user_id":"user-1","job_type":"categorization","edited_transcript":"A supplied transcript to categorize.","max_tags":8}
```

Exactly one of usable `audio_url` or `edited_transcript` is sufficient. The
result uses `JobResult.pipeline` with moderation and categorization data.

### `audio_tag`

```json
{"job_id":"job-audio-tag","track_id":"track-1","user_id":"user-1","job_type":"audio_tag","audio_url":"https://storage.example/track.mp3"}
```

Returns `JobResult.audio_tag` with transcript text and at most two suggestions.

### `reconstruct`

```json
{
  "job_id":"job-reconstruct",
  "track_id":"track-1",
  "user_id":"user-1",
  "job_type":"reconstruct",
  "audio_url":"https://storage.example/track.mp3",
  "same_speaker":true,
  "changes":[{"segment_start":12.4,"segment_end":15.8,"new_text":"replacement speech","original_text":"original speech"}]
}
```

Returns `JobResult.reconstruct` with rebuilt audio, regeneration status,
transcription, and moderation.

### `edit_transcript`

```json
{"job_id":"job-edit","track_id":"track-1","user_id":"user-1","job_type":"edit_transcript","audio_url":"https://storage.example/track.mp3","edited_transcript":"The complete corrected transcript.","same_speaker":true}
```

### `rebuild`

```json
{"job_id":"job-rebuild","track_id":"track-1","user_id":"user-1","job_type":"rebuild","audio_url":"https://storage.example/track.mp3","edited_transcript":"The complete text used to rebuild the audio."}
```

`edit_transcript` returns `JobResult.reconstruct`. `rebuild` currently returns
`JobResult.pipeline`. For both jobs, `edited_transcript` must be non-empty.

## All job response payloads

This is the authoritative response contract for **all nine job types**:
`pipeline`, `transcription`, `magic_clean`, `discovery`, `categorization`,
`audio_tag`, `reconstruct`, `edit_transcript`, and `rebuild`.


The HTTP submission response only confirms acceptance. Final data arrives in
the `result` of `job_completed` and is recoverable with `GetResult`. The JSON
below is the protobuf JSON representation; generated clients should read the
typed fields directly. Fields with default/empty values may be omitted.

### Authoritative job-to-payload mapping

The application accepts exactly these nine asynchronous job types. They map
to five protobuf `JobResult.oneof payload` fields:

| Submitted `job_type` | Populated `JobResult` field | Protobuf message |
| --- | --- | --- |
| `pipeline` | `result.pipeline` | `PipelinePayload` |
| `categorization` | `result.pipeline` | `PipelinePayload` |
| `discovery` | `result.pipeline` | `PipelinePayload` |
| `rebuild` | `result.pipeline` | `PipelinePayload` |
| `transcription` | `result.transcription` | `TranscriptionPayload` |
| `audio_tag` | `result.audio_tag` | `AudioTagPayload` |
| `magic_clean` | `result.magic_clean` | `MagicCleanPayload` |
| `reconstruct` | `result.reconstruct` | `ReconstructPayload` |
| `edit_transcript` | `result.reconstruct` | `ReconstructPayload` |

Every `JobResult` always has these envelope fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Durable backend idempotency/correlation ID |
| `backend_id` | string | Registered owner; also enforced for replay and cancellation |
| `run_id` | string | ID for this execution attempt/run |
| `track_id` | string | Backend track correlation ID |
| `job_type` | string | One of the nine values above |
| `status` | string | `queued`, `running`, `completed`, `failed`, or `cancelled` |
| `current_stage` | string | Current/failed stage; empty after successful completion |
| `error` | string | Sanitized terminal error when failed |
| `payload` | protobuf oneof | Exactly one typed success payload listed above |

### Returned nested object schemas

These schemas are reused by the job payloads:

| Object | Returned fields and types |
| --- | --- |
| `TranscriptionObject` | `transcript: string`, `segments: Segment[]`, `language: string`, `confidence: float` |
| `Segment` | `start: double`, `end: double`, `text: string`, `speaker: string`, `words: Word[]` |
| `Word` | `word: string`, `start: double`, `end: double`, `score: float`, `speaker: string` |
| `ModerationReply` | `flagged: bool`, `severity: string`, `intent: string`, `reason: string`, `flagged_categories: string[]`, `blocked_words_found: string[]` |
| `CategorizationReply` | `categories: string[]`, `tags: string[]`, `confidence_scores: Struct`, `sentiment: string`, `new_tags_added: string[]`, `new_categories_added: string[]`, `settings_applied: bool`, `llm_used: bool`, `categorizer_mode: string` |
| `EnhancedAudio` | `backend_id: string`, `bucket_name: string`, `audio_url: string`, `b2_key: string` |
| `AudioQuality` | `quality_score: float`, `snr_db: float`, `peak_db: float`, `lufs: float`, `clipping_detected: bool` |
| `RebuiltAudio` | `backend_id: string`, `bucket_name: string`, `audio_url: string`, `b2_key: string`, `duration: float` |

The five top-level payload messages return:

| Payload | Every defined field |
| --- | --- |
| `PipelinePayload` | `source_audio_url: string`, `transcription: TranscriptionObject`, `moderation: ModerationReply`, optional `categorization: CategorizationReply`, optional `edited_transcript: string`, `discovery: Struct`, `content_description: string`, `compressed_audio: Struct`, `report: Struct`, `flagged: bool` |
| `TranscriptionPayload` | `source_audio_url: string`, `transcription: TranscriptionObject` |
| `AudioTagPayload` | `source_audio_url: string`, `transcription: string`, `suggestions: string[]` |
| `MagicCleanPayload` | `enhanced: bool`, `enhanced_audio: EnhancedAudio`, `quality: AudioQuality`, `stage_times: Struct`, `transcription: TranscriptionObject`, `moderation: ModerationReply` |
| `ReconstructPayload` | optional `edited_transcript: string`, `rebuilt_audio: RebuiltAudio`, `is_regenerated: bool`, `transcription: TranscriptionObject`, `moderation: ModerationReply` |

### Payload sent by `Pipeline.Subscribe`

Every streamed update uses this common `PipelineEvent` envelope:

```json
{
  "event":"stage_changed|stage_result|job_retrying|job_completed|job_failed|job_cancelled",
  "job_id":"job-1",
  "run_id":"run-1",
  "track_id":"track-1",
  "job_type":"pipeline",
  "status":"queued|running|completed|failed|cancelled",
  "current_stage":"transcribing",
  "label":"Transcribing audio",
  "description":"Converting speech to text",
  "progress_pct":12,
  "elapsed_seconds":3.4,
  "estimated_remaining":8.0,
  "error":"",
  "result":{}
}
```

The fields not relevant to a particular event remain empty/default. The
payload placed in `job_completed.result` has these exact top-level keys:

| Job type | Keys sent in `job_completed.result` |
| --- | --- |
| `pipeline` | `job_id`, `run_id`, `job_type`, `track_id`, `source_audio_url`, `transcription`, `moderation`, `categorization`, `edited_transcript`, optional `discovery`, optional `content_description`, optional `compressed_audio` |
| `transcription` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription` |
| `magic_clean` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription`, `moderation`, `categorization`, `enhanced`, `enhanced_audio`, `quality`, `stage_times` |
| `discovery` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription`, `discovery`, optional `content_description`; or `report` and `flagged` when no content exists |
| `categorization` | `job_id`, `run_id`, `job_type`, `track_id`, `source_audio_url`, `transcription`, `moderation`, `categorization`, `edited_transcript`, optional `discovery`, optional `content_description` |
| `audio_tag` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription` (string), `suggestions` (maximum two) |
| `reconstruct` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription`, `moderation`, `categorization`, `rebuilt_audio`, `is_regenerated` |
| `edit_transcript` | `job_id`, `run_id`, `job_type`, `track_id`, `transcription`, `moderation`, `categorization`, `edited_transcript`, `rebuilt_audio`, `is_regenerated` |
| `rebuild` | Same raw pipeline keys as `categorization`, with `job_type: rebuild` |

`stage_result.result` always has this shape:

```json
{"stage":"moderating","data":{"flagged":false,"severity":"none"}}
```

`job_retrying.result` and `job_failed.result` contain a report:

```json
{
  "report":{
    "stage":"transcribing",
    "error":"An unexpected error occurred. Please try again later.",
    "attempt":1,
    "retryable":true
  }
}
```

The backend should synchronize the envelope on every event. On
`job_completed`, persist the result and mark the backend job complete. On
`job_retrying`, keep it active/queued. On `job_failed` or `job_cancelled`, stop
waiting and persist the terminal state.

For `pipeline`, replace the backend track audio only from
`job_completed.result.compressed_audio`. For `magic_clean`, replace it only
from `job_completed.result.enhanced_audio`. Both contain the exact Backblaze
`audio_url` and `b2_key` that passed post-upload size verification. Do not
reuse the submitted source URL as the completed audio URL.

The pipeline `compressed_audio` object also reports the bitrate actually selected. Encoding is adaptive for Alexa delivery: lossless or high-bitrate sources are capped at `PIPELINE_MP3_BITRATE_KBPS` (96 kbps by default), while already-compressed sources use the next standard bitrate at or below 80% of their measured source bitrate. The same adaptive rule is applied when encoding the final Magic Clean enhanced MP3; enhancement processing is unchanged.

The pipeline `compressed_audio` object reports:

```json
{
  "audio_url":"https://storage.example/pipeline-output.mp3",
  "b2_key":"pipeline-source-mp3/track-1/job-1.mp3",
  "format":"mp3",
  "bitrate_kbps":96,
  "duration_seconds":320.73,
  "size_bytes":3849135,
  "source_size_bytes":4026395,
  "size_reduction_bytes":177260,
  "size_reduction_pct":4.402
}
```

A negative `size_reduction_bytes` or `size_reduction_pct` means the chosen
bitrate produced a larger object; the backend must display it honestly rather
than describing it as reduced. MP3 uploads use `Content-Type: audio/mpeg`.
Downloads, encoded duration, and uploaded object size are verified before a
terminal success is emitted.

### `pipeline` response — `JobResult.pipeline`

```json
{
  "job_id":"job-pipeline","run_id":"run-1","track_id":"track-1",
  "job_type":"pipeline","status":"completed",
  "pipeline":{
    "source_audio_url":"https://storage.example/track.mp3",
    "transcription":{"transcript":"Spoken content.","language":"en","confidence":0.98,"segments":[{"start":0.1,"end":2.4,"text":"Spoken content.","words":[{"word":"Spoken","start":0.1,"end":0.8,"score":0.99}]}]},
    "moderation":{"flagged":false,"severity":"none","intent":"safe","reason":"No harmful content detected"},
    "categorization":{"categories":["Technology"],"tags":["ai"],"confidence_scores":{"Technology":0.91},"sentiment":"neutral","categorizer_mode":"trained"},
    "discovery":{"main_topic":"Artificial intelligence","summary_short":"A short summary"},
    "content_description":"A discussion about artificial intelligence.",
    "compressed_audio":{"audio_url":"https://storage.example/output.mp3","b2_key":"audio/output.mp3","format":"mp3"},
    "flagged":false
  }
}
```

If no content is detected, `pipeline.report` contains
`{"code":"content_not_detected","flagged":true,"reason":"..."}` and
`pipeline.flagged` is true. If moderation flags usable content,
`categorization` and `discovery` may be absent.

### `transcription` response — `JobResult.transcription`

```json
{
  "job_id":"job-transcription","run_id":"run-2","track_id":"track-1",
  "job_type":"transcription","status":"completed",
  "transcription":{
    "transcription":{"transcript":"Hello world.","language":"en","confidence":0.99,"segments":[{"start":0.0,"end":1.2,"text":"Hello world.","words":[{"word":"Hello","start":0.0,"end":0.5,"score":0.99},{"word":"world","start":0.6,"end":1.1,"score":0.98}]}]}
  }
}
```

### `magic_clean` response — `JobResult.magic_clean`

```json
{
  "job_id":"job-clean","run_id":"run-3","track_id":"track-1",
  "job_type":"magic_clean","status":"completed",
  "magic_clean":{
    "enhanced":true,
    "enhanced_audio":{"audio_url":"https://storage.example/clean.mp3","b2_key":"audio/clean.mp3"},
    "quality":{"quality_score":0.92,"snr_db":24.5,"peak_db":-1.0,"lufs":-16.0,"clipping_detected":false},
    "stage_times":{"downloading":0.7,"separating":8.2,"enhancing":3.4,"mixing":0.5,"finalizing":0.8}
  }
}
```

### `discovery` response — `JobResult.pipeline`

```json
{
  "job_id":"job-discovery","run_id":"run-4","track_id":"track-1",
  "job_type":"discovery","status":"completed",
  "pipeline":{
    "transcription":{"transcript":"Content to discover.","language":"en","confidence":0.97},
    "discovery":{"main_topic":"News","summary_short":"A news summary","tags":["news"]},
    "content_description":"A spoken news update."
  }
}
```

### `categorization` response — `JobResult.pipeline`

```json
{
  "job_id":"job-category-audio","run_id":"run-5","track_id":"track-1",
  "job_type":"categorization","status":"completed",
  "pipeline":{
    "transcription":{"transcript":"Technology and science content.","language":"en","confidence":0.98},
    "moderation":{"flagged":false,"severity":"none","intent":"safe"},
    "categorization":{"categories":["Technology","Science"],"tags":["ai","research"],"confidence_scores":{"Technology":0.94,"Science":0.82},"sentiment":"neutral","llm_used":false,"categorizer_mode":"trained"},
    "discovery":{"main_topic":"Technology research"}
  }
}
```

### `audio_tag` response — `JobResult.audio_tag`

```json
{
  "job_id":"job-audio-tag","run_id":"run-6","track_id":"track-1",
  "job_type":"audio_tag","status":"completed",
  "audio_tag":{"transcription":"A football match update.","suggestions":["football","sports"]}
}
```

### `reconstruct` response — `JobResult.reconstruct`

```json
{
  "job_id":"job-reconstruct","run_id":"run-7","track_id":"track-1",
  "job_type":"reconstruct","status":"completed",
  "reconstruct":{
    "rebuilt_audio":{"audio_url":"https://storage.example/reconstructed.mp3","b2_key":"audio/reconstructed.mp3","duration":54.2},
    "is_regenerated":true
  }
}
```

### `edit_transcript` response — `JobResult.reconstruct`

```json
{
  "job_id":"job-edit","run_id":"run-8","track_id":"track-1",
  "job_type":"edit_transcript","status":"completed",
  "reconstruct":{
    "edited_transcript":"The complete corrected transcript.",
    "rebuilt_audio":{"audio_url":"https://storage.example/edited.mp3","b2_key":"audio/edited.mp3","duration":61.8},
    "is_regenerated":true,
    "transcription":{"transcript":"The complete corrected transcript.","language":"en","confidence":0.98},
    "moderation":{"flagged":false,"severity":"none","intent":"safe"}
  }
}
```

### `rebuild` response — `JobResult.pipeline`

```json
{
  "job_id":"job-rebuild","run_id":"run-9","track_id":"track-1",
  "job_type":"rebuild","status":"completed",
  "pipeline":{
    "source_audio_url":"https://storage.example/track.mp3",
    "transcription":{"transcript":"The complete text used for processing.","language":"en","confidence":1.0},
    "edited_transcript":"The complete text used for processing.",
    "moderation":{"flagged":false,"severity":"none","intent":"safe"},
    "categorization":{"categories":["Education"],"tags":["learning"]},
    "discovery":{"main_topic":"Education"}
  }
}
```

### Failed or cancelled response — every job type

```json
{"job_id":"job-1","run_id":"run-1","track_id":"track-1","job_type":"pipeline","status":"failed","current_stage":"transcribing","error":"An unexpected error occurred. Please try again later."}
```

Cancelled jobs use `status: "cancelled"`. A failed/cancelled result does not
populate a successful `oneof` payload unless partial durable output exists.

## Stage events and continuation

For each executed stage, `Pipeline.Subscribe` sends:

1. `stage_changed` when work begins, with stage, label, description, progress,
   elapsed time, and estimated remaining time.
2. `stage_result` after that stage commits its output. Data is under
   `result.stage` and `result.data`.
3. Hear AI automatically starts the next applicable stage.

Persist/forward each update and keep reading the same stream. `stage_result`
is not terminal and does not require a continuation request.

### Full pipeline example

```text
job_queued
  ↓
stage_changed: transcribing
stage_result:  transcribing
  ↓  correction occurs inside transcription processing
stage_changed: moderating
stage_result:  moderating
  ↓
stage_changed: categorizing     (only when not flagged)
stage_result:  categorizing
  ↓
stage_changed: discovering      (only when not flagged)
stage_result:  discovering
  ↓
stage_changed: compressing
stage_result:  compressing      (optimized MP3/variants)
  ↓
job_completed
```

Only `job_completed`, `job_failed`, and `job_cancelled` are terminal.

If no usable speech is detected, Hear AI completes with a flagged report
containing `code: content_not_detected`, transcription data, and a reason. It
does not categorize or discover empty content.

If a stage fails after retries, Hear AI persists the failure, removes temporary
job files, emits `job_failed`, and stops the remaining stages. On success, the
final result is persisted, working files are deleted, and pipeline audio is
encoded/uploaded as optimized MP3 variants.

## Subscribe and recover

```python
import os
import uuid
import grpc
import httpx

from hear.proto.pipeline_pb2 import GetResultRequest, SubscribeRequest
from hear.proto.pipeline_pb2_grpc import PipelineStub

HTTP_URL = os.environ["HEAR_HTTP_URL"].rstrip("/")
GRPC_TARGET = os.environ["HEAR_GRPC_TARGET"]
SECRET = os.environ["HEAR_SERVICE_KEY"]
BACKEND_ID = os.environ["HEAR_BACKEND_ID"]
APPLICATION = os.getenv("HEAR_GRPC_APPLICATION", "hear")
METADATA = (("application", APPLICATION), ("x-api-key", SECRET))


def submit(job_id, track_id, user_id, audio_url):
    response = httpx.post(
        f"{HTTP_URL}/process",
        headers={"X-Service-Key": SECRET},
        json={
            "job_id": job_id,
            "backend_id": BACKEND_ID,
            "storage": {
                "endpoint_url": os.environ["HEAR_STORAGE_ENDPOINT_URL"],
                "bucket_name": os.environ["HEAR_STORAGE_BUCKET_NAME"],
                "key_id": os.environ["HEAR_STORAGE_KEY_ID"],
                "application_key": os.environ["HEAR_STORAGE_APPLICATION_KEY"],
                "folder_prefix": os.environ["HEAR_STORAGE_FOLDER_PREFIX"],
                "public_base_url": os.environ["HEAR_STORAGE_PUBLIC_BASE_URL"],
                "expires_at": os.environ["HEAR_STORAGE_EXPIRES_AT"],
            },
            "track_id": track_id,
            "user_id": user_id,
            "job_type": "pipeline",
            "audio_url": audio_url,
            "max_tags": 8,
        },
        timeout=httpx.Timeout(30, connect=5),
    )
    response.raise_for_status()
    return response.json()


def stream_until_terminal(job_id):
    options = [("grpc.max_receive_message_length", 64 * 1024 * 1024)]
    with grpc.insecure_channel(GRPC_TARGET, options=options) as channel:
        grpc.channel_ready_future(channel).result(timeout=10)
        client = PipelineStub(channel)
        try:
            for event in client.Subscribe(
                SubscribeRequest(job_id=job_id),
                metadata=METADATA,
                timeout=60 * 60,
            ):
                # Persist and forward this update from the backend.
                print(event.event, event.current_stage,
                      event.progress_pct, event.status)
                if event.event in {
                    "job_completed", "job_failed", "job_cancelled"
                }:
                    break
        except grpc.RpcError:
            # Reconnect with the same job_id or recover below. Do not create a
            # replacement job just because the stream disconnected.
            pass

        return client.GetResult(
            GetResultRequest(job_id=job_id),
            metadata=METADATA,
            timeout=30,
        )


job_id = str(uuid.uuid4())  # Persist before submission.
accepted = submit(job_id, "track-123", "user-456",
                  "https://storage.example/audio/track-123.mp3")
result = stream_until_terminal(accepted["job_id"])
```

Terminal state is durable. If the stream disconnects, reconnect with the same
`job_id` and call `GetResult`. Do not submit a replacement job.

The final `JobResult` uses a typed protobuf `oneof payload`:
`PipelinePayload`, `TranscriptionPayload`, `MagicCleanPayload`,
`ReconstructPayload`, or `AudioTagPayload`. Word timing data is under
`result.transcription.segments[].words`. Per-stage event data uses
`PipelineEvent.result` (`google.protobuf.Struct`) because stage payloads vary.

## Fair scheduling

`user_id` is mandatory and drives per-user round-robin scheduling so one user
cannot monopolize workers. It is not used to fetch tracks or recordings.

## Complete HTTP surface

HTTP:

- `GET /`, `GET /health`, `GET /ready`
- `POST /process`
- `POST /discovery` (convenience submission for a discovery job)

Pipeline gRPC:

- `SubmitJob`, `Subscribe`, `GetResult`, `CancelJob`, `GetQueueStats`
- `Moderate`, `Categorize`
- `CreatePreview`, `ConfirmPreview`, `RemoveSegment`, `RollbackPreview`,
  `GetPreview`
- `ListDiscovery`, `TrainCategorizer`, `IngestCategoryEvent`,
  `UpdatePlatformSettings`, `Health`

Resolver gRPC: `Resolve`, `ResolverHealth`, `Rebuild`, `Apply`.

There is no `/ws` WebSocket endpoint in Hear AI. The owning backend should
forward gRPC updates to browser/mobile clients.

## Complete Pipeline gRPC reference

Every call requires `METADATA`. RPC names below are methods on
`hear.pipeline.v1.Pipeline`.

| RPC | Request → response | Purpose and required input |
| --- | --- | --- |
| `SubmitJob` | `SubmitJobRequest → SubmitJobResponse` | Alternative job submission. Uses the same fields/rules as `/process`; do not also POST the job. |
| `Subscribe` | `SubscribeRequest → stream PipelineEvent` | Stream queue, stage, and terminal events for `job_id`. |
| `GetResult` | `GetResultRequest → JobResult` | Read durable status/final typed payload by `job_id`. |
| `CancelJob` | `GetResultRequest → JobResult` | Cancel a queued/running `job_id` and return its current state. |
| `GetQueueStats` | `google.protobuf.Empty → QueueStatsReply` | Active/queued/total counts and wait/duration estimates. |
| `Moderate` | `TextRequest → ModerationReply` | Moderate `text`; returns flag, severity, intent, reason, categories, and blocked words. |
| `Categorize` | `CategorizeRequest → CategorizationReply` | Categorize `text` using optional `custom_tags` and `max_tags`. |
| `CreatePreview` | `ReconstructRequest → CreatePreviewReply` | Generate a temporary reconstruction preview from `audio_url`, `track_id`, and `changes` (or legacy single segment fields). |
| `ConfirmPreview` | `PreviewRequest → ConfirmPreviewReply` | Confirm `preview_id`; optional `track_id`/`user_id`; returns final audio object. |
| `RemoveSegment` | `RemoveSegmentRequest → RemoveSegmentReply` | Delete `[segment_start, segment_end]` from `audio_url`; requires `track_id`; optional `user_id`. |
| `RollbackPreview` | `PreviewRequest → RollbackPreviewReply` | Roll back/delete an unconfirmed `preview_id`. |
| `GetPreview` | `PreviewRequest → Preview` | Fetch persisted preview details and quality metrics. |
| `ListDiscovery` | `DiscoveryRequest → ListDiscoveryReply` | List `latest` or `trending` discovery items using `limit`/`offset`. |
| `TrainCategorizer` | `TrainRequest → TrainReply` | Train target `category`, `tags`, or `harm`; result/metrics are encoded in `detail`. |
| `IngestCategoryEvent` | `CategoryEvent → IngestReply` | Store a training event and return its UUID `example_id`. |
| `UpdatePlatformSettings` | `PlatformSettingsRequest → PlatformSettingsReply` | Replace comma-separated blocked and auto-tag keyword sets. |
| `Health` | `google.protobuf.Empty → HealthReply` | GPU identity/memory and active/queued job counts. |

### gRPC job submission

```python
from hear.proto.pipeline_pb2 import StorageContext, SubmitJobRequest

accepted = client.SubmitJob(
    SubmitJobRequest(
        job_id="job-grpc-1",
        backend_id=BACKEND_ID,
        storage=StorageContext(
            endpoint_url=os.environ["HEAR_STORAGE_ENDPOINT_URL"],
            bucket_name=os.environ["HEAR_STORAGE_BUCKET_NAME"],
            key_id=os.environ["HEAR_STORAGE_KEY_ID"],
            application_key=os.environ["HEAR_STORAGE_APPLICATION_KEY"],
            folder_prefix=os.environ["HEAR_STORAGE_FOLDER_PREFIX"],
            public_base_url=os.environ["HEAR_STORAGE_PUBLIC_BASE_URL"],
            expires_at=os.environ["HEAR_STORAGE_EXPIRES_AT"],
        ),
        track_id="track-1",
        user_id="user-1",
        job_type="pipeline",
        audio_url="https://storage.example/track.mp3",
        max_tags=8,
    ),
    metadata=METADATA,
    timeout=30,
)
```

`SubmitJobRequest` also supports `edited_transcript`, repeated `changes`,
`same_speaker`, grouping fields, `source`, `track_count`, speed/playback
fields, Magic Clean percentages, `type`, and `media_file_id`. It does not have
`track_exists`; protobuf field number 17 is reserved.

### Direct moderation and categorization

```python
from hear.proto.pipeline_pb2 import CategorizeRequest, TextRequest

moderation = client.Moderate(
    TextRequest(text="text to inspect"), metadata=METADATA, timeout=30
)
categorization = client.Categorize(
    CategorizeRequest(text="text to classify", custom_tags=["news"], max_tags=8),
    metadata=METADATA,
    timeout=30,
)
```

### Preview, confirm, segment deletion, rollback

```python
from hear.proto.pipeline_pb2 import (
    PreviewRequest, ReconstructRequest, RemoveSegmentRequest, SegmentChange, StorageContext,
)

storage_context = StorageContext(
    endpoint_url=os.environ["HEAR_STORAGE_ENDPOINT_URL"],
    bucket_name=os.environ["HEAR_STORAGE_BUCKET_NAME"],
    key_id=os.environ["HEAR_STORAGE_KEY_ID"],
    application_key=os.environ["HEAR_STORAGE_APPLICATION_KEY"],
    folder_prefix=os.environ["HEAR_STORAGE_FOLDER_PREFIX"],
    public_base_url=os.environ["HEAR_STORAGE_PUBLIC_BASE_URL"],
    expires_at=os.environ["HEAR_STORAGE_EXPIRES_AT"],
)

preview = client.CreatePreview(
    ReconstructRequest(
        audio_url="https://storage.example/track.mp3",
        backend_id=BACKEND_ID,
        storage=storage_context,
        track_id="track-1",
        same_speaker=True,
        changes=[SegmentChange(segment_start=12.4, segment_end=15.8,
                               new_text="replacement")],
    ), metadata=METADATA, timeout=600,
)

saved = client.ConfirmPreview(
    PreviewRequest(preview_id=preview.preview_id, track_id="track-1",
                   user_id="user-1"),
    metadata=METADATA, timeout=600,
)

removed = client.RemoveSegment(
    RemoveSegmentRequest(track_id="track-1", backend_id=BACKEND_ID,
                         storage=storage_context,
                         audio_url="https://storage.example/track.mp3",
                         segment_start=20.0, segment_end=24.5,
                         user_id="user-1"),
    metadata=METADATA, timeout=600,
)

rolled_back = client.RollbackPreview(
    PreviewRequest(preview_id=preview.preview_id),
    metadata=METADATA, timeout=30,
)
```

`RemoveSegment` rejects an end time that is not greater than its start time.

### Discovery catalog

```python
from hear.proto.pipeline_pb2 import DiscoveryRequest

items = client.ListDiscovery(
    DiscoveryRequest(sort="latest", limit=50, offset=0),
    metadata=METADATA, timeout=30,
)
```

### Training data and model training

```python
from hear.proto.pipeline_pb2 import CategoryEvent, TrainRequest

ingested = client.IngestCategoryEvent(
    CategoryEvent(event_type="category_feedback", text="example text",
                  category="Nature", source_id="backend-event-123"),
    metadata=METADATA, timeout=30,
)

trained = client.TrainCategorizer(
    TrainRequest(target="category"), metadata=METADATA, timeout=900
)
```

`CategoryEvent` supports optional `category`, repeated `tags`, optional
`label`, and optional backend `source_id`. Harm labels are `safe` or `harmful`.
Valid training targets are `category`, `tags`, and `harm`.

### Platform settings and queue/health

```python
from google.protobuf.empty_pb2 import Empty
from hear.proto.pipeline_pb2 import PlatformSettingsRequest

settings_reply = client.UpdatePlatformSettings(
    PlatformSettingsRequest(blocked_keywords="word1,word2",
                            auto_tag_keywords="news,sports"),
    metadata=METADATA, timeout=30,
)
queue = client.GetQueueStats(Empty(), metadata=METADATA, timeout=10)
health = client.Health(Empty(), metadata=METADATA, timeout=10)
```

## Complete Resolver gRPC reference

RPCs are methods on `hear.resolver.v1.Resolver` and use the same metadata.

| RPC | Request → response | Purpose |
| --- | --- | --- |
| `Resolve` | `ResolveRequest → ResolveReply` | Resolve an `utterance` for a `country_code` into category, creator, organisation, location, tags, temporal data, free text, action, and candidates. |
| `ResolverHealth` | `HealthRequest → HealthReply` | Return resolver status, taxonomy version, and readiness. |
| `Rebuild` | `RebuildRequest → RebuildReply` | Build/load the requested taxonomy `version` (or latest when supported). |
| `Apply` | `RebuildRequest → RebuildReply` | Apply the requested built taxonomy version. |

```python
from hear.proto.resolver_pb2 import HealthRequest, RebuildRequest, ResolveRequest
from hear.proto.resolver_pb2_grpc import ResolverStub

resolver = ResolverStub(channel)
resolved = resolver.Resolve(
    ResolveRequest(utterance="play jazz music", country_code="US"),
    metadata=METADATA, timeout=10,
)
resolver_health = resolver.ResolverHealth(
    HealthRequest(), metadata=METADATA, timeout=10
)
rebuilt = resolver.Rebuild(
    RebuildRequest(version=19), metadata=METADATA, timeout=600
)
applied = resolver.Apply(
    RebuildRequest(version=19), metadata=METADATA, timeout=60
)
```

## gRPC error policy

| Code | Backend action |
| --- | --- |
| `UNAUTHENTICATED` | Fix `x-api-key`; do not retry blindly |
| `INVALID_ARGUMENT` | Fix the request |
| `NOT_FOUND` | Verify `job_id` or requested resource |
| `ALREADY_EXISTS` | Resolve the operation conflict |
| `UNAVAILABLE` | Retry with exponential backoff and jitter |
| `DEADLINE_EXCEEDED` | Call `GetResult` before resubmitting |
| `INTERNAL` | Log `job_id`/`run_id`; retry only idempotent operations |

## Smoke check

1. Confirm `/ready` returns `200`.
2. Confirm gRPC reaches `READY` on `50051`.
3. Submit one small job to `/process`.
4. Observe ordered `stage_changed` and `stage_result` events.
5. Observe exactly one terminal event.
6. Call `GetResult` and compare it with the terminal event.
7. Resubmit the identical payload and confirm the same `run_id` with
   `replayed: true`.

## Verification

The repository suite currently passes `204` tests covering HTTP/gRPC
contracts, validation, idempotency, typed results, stage reports, fair
scheduling, cancellation, temporary cleanup, Magic Clean, and transcription.
