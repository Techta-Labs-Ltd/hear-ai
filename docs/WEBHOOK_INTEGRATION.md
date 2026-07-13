# Hear AI - Webhook Integration

## Overview

When you submit jobs to Hear AI, results are delivered via webhook to your configured callback URL. This document specifies the exact payloads your backend will receive.

---

## Configuration

| Env Variable | Purpose |
|---|---|
| `HEAR_CALLBACK_URL` | Global fallback webhook URL (used when no per-job URL is set) |
| Individual jobs can set `callback_url` in `POST /api/v1/process` request body |

Authentication: Requests are sent as `POST` with `Content-Type: application/json`. Add `X-Service-Key` header with your service secret for verification.

---

## Job Lifecycle

```
POST /api/v1/process     POST /api/v1/reconstruct     POST /api/v1/edit-transcript
        |                        |                              |
        v                        v                              v
   Job Accepted (202)       Preview Ready (200)          Job Accepted (202)
   status: "accepted"       preview_id + audio_url       status: "accepted"
        |                        |                              |
        v                        |                              v
   Worker processes             |                        Worker processes
   [transcribe/moderate/         |                        [diff → reconstruct]
    categorize/enhance/          |                              |
    reconstruct]                 |                              v
        |                        |                        Webhook fires
        v                        v                        (job completed/failed)
   Webhook fires              User reviews preview
   (job completed/failed)         |
                                  v
                             POST /api/v1/reconstruct/confirm
                                  |
                                  v
                             Splicing + B2 upload
```

---

## 1. Callback Endpoint

Your backend must expose:

```
POST https://your-api.example.com/hear/webhook
Content-Type: application/json
X-Service-Key: <your-secret>
```

**Retry behavior**: If your endpoint returns non-2xx, Hear AI retries every 45 seconds until delivery succeeds. Maximum: 100 retries.

---

## 2. Webhook Payloads

### 2.1 Job Completed (transcription, moderation, categorization, pipeline, magic_clean)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": "your-track-id",
  "user_id": "optional-user-id",
  "job_type": "pipeline",
  "status": "completed",
  "error": null,
  "result": {
    "job_id": "uuid-string",
    "run_id": "uuid-string",
    "job_type": "pipeline",
    "track_id": "your-track-id",
    "source_audio_url": "https://cdn.example.com/audio.mp3",
    "transcription": {
      "transcript": "Full transcript text...",
      "segments": [],
      "language": "en",
      "confidence": 0.98
    },
    "moderation": {
      "flagged": false,
      "severity": "none",
      "intent": "general",
      "reason": "",
      "flagged_categories": [],
      "blocked_words_found": []
    },
    "categorization": {
      "tags": ["news", "politics"],
      "categories": ["News & Politics"],
      "confidence": 0.87
    },
    "edited_transcript": null,
    "compressed_audio": {
      "audio_url": "https://s3.../compressed.mp3",
      "b2_key": "compressed/track-id/uuid.mp3",
      "audio_format": "mp3"
    },
    "speed_layers": [
      {
        "speed": 1.5,
        "audio_url": "https://s3.../speed-1.5x.mp3",
        "b2_key": "speed/track-id/uuid-1.5x.mp3"
      }
    ],
    "playback_speeds_applied": [1.0, 1.25, 1.5, 1.75, 2.0],
    "discovery": {
      "content_description": "Generated description text...",
      "tags": [...],
      "source": "news"
    },
    "content_description": "Generated description text..."
  }
}
```

### 2.2 Job Completed (reconstruct - worker path)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": "your-track-id",
  "user_id": null,
  "job_type": "reconstruct",
  "status": "completed",
  "error": null,
  "result": {
    "job_id": "uuid-string",
    "run_id": "uuid-string",
    "job_type": "reconstruct",
    "track_id": "your-track-id",
    "source_audio_url": "https://cdn.example.com/audio.mp3",
    "same_speaker": true,
    "segments_applied": 3,
    "changes": [
      {
        "segment_start": 1.52,
        "segment_end": 3.22,
        "new_text": "corrected text here",
        "original_text": "original text here"
      }
    ],
    "reconstructed_audio": {
      "audio_url": "https://s3..../reconstructed/track-id/uuid.mp3",
      "b2_key": "reconstructed/track-id/uuid.mp3",
      "duration": 45.67,
      "audio_format": "mp3"
    }
  }
}
```

### 2.3 Job Completed (edit-transcript)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": "your-track-id",
  "user_id": "optional-user-id",
  "job_type": "edit_transcript",
  "status": "completed",
  "error": null,
  "result": {
    "job_id": "uuid-string",
    "run_id": "uuid-string",
    "job_type": "edit_transcript",
    "track_id": "your-track-id",
    "changes_detected": 3,
    "edited_transcript": "full corrected transcript text...",
    "reconstructed_audio": {
      "audio_url": "https://s3..../reconstructed/track-id/uuid.mp3",
      "b2_key": "reconstructed/track-id/uuid.mp3",
      "duration": 120.5,
      "audio_format": "mp3"
    }
  }
}
```

**Note**: If `changes_detected` is `0`, the `reconstructed_audio` field is `null`. No edits needed.

### 2.4 Job Completed (audio_tag)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": null,
  "user_id": null,
  "job_type": "audio_tag",
  "status": "completed",
  "error": null,
  "tags": ["tag1", "tag2", "tag3"],
  "categories": null,
  "media_file_id": "your-media-id",
  "type": "track",
  "result": {
    "job_id": "uuid-string",
    "job_type": "audio_tag",
    "media_file_id": "your-media-id",
    "type": "track",
    "tags": ["tag1", "tag2", "tag3"]
  }
}
```

### 2.5 Job Completed (magic_clean)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": "your-track-id",
  "user_id": null,
  "job_type": "magic_clean",
  "status": "completed",
  "error": null,
  "result": {
    "job_id": "uuid-string",
    "run_id": "uuid-string",
    "job_type": "magic_clean",
    "track_id": "your-track-id",
    "enhancement": {
      "enhanced_url": "https://s3.../enhanced/track-id/uuid.mp3",
      "b2_key": "enhanced/track-id/uuid.mp3",
      "quality_score": 4.2,
      "snr_db": 15.3
    },
    "discovery": {...},
    "content_description": "...",
    "speed_layers": [...],
    "playback_speeds_applied": [1.0, 1.25, 1.5],
    "b2_cleanup": {
      "deleted_keys": ["old-enhanced-key-1", "old-enhanced-key-2"]
    }
  }
}
```

### 2.6 Job Failed (any type)

```json
{
  "job_id": "uuid-string",
  "run_id": "uuid-string",
  "track_id": "your-track-id",
  "user_id": null,
  "job_type": "pipeline",
  "status": "failed",
  "result": null,
  "error": "Audio processing failed. Please check the source file and try again."
}
```

**Error messages are sanitized** -- they never contain file paths, model names, or internal details. Possible values:

| Error Message | Meaning |
|---|---|
| `"Audio processing failed. Please check the source file and try again."` | Download issue, empty/truncated file, format error |
| `"Invalid request. Please check your input and try again."` | Missing/bad parameters |
| `"Speech synthesis failed. Please try again with different text."` | TTS engine failure |
| `"Processing failed. Please try again later."` | Generic runtime error |
| `"Request timed out. Please try with a shorter segment or smaller text."` | Timeout |
| `"An unexpected error occurred. Please try again later."` | Unknown error |

---

## 3. Realtime Events (SSE / WebSocket)

For real-time progress, your frontend subscribes to events. The backend does NOT receive these - they are for the client-facing UI only.

### SSE Endpoint

```
GET /api/v1/events/{job_id}
Headers: X-Service-Key: <your-secret>
Response: text/event-stream
```

### WebSocket Endpoint

```
ws://host:8000/ws/{job_id}
```

### Event Types

| Event | When | Payload |
|---|---|---|
| `queue_position` | On enqueue and whenever position changes | `{ event, job_id, position, total_queued, estimated_wait_s }` |
| `stage_changed` | On each pipeline stage transition | `{ event, job_id, status, current_stage, progress_pct }` |
| `job_completed` | On successful completion | `{ event, job_id, status: "completed", result }` |
| `job_failed` | On unrecoverable failure | `{ event, job_id, status: "failed", error }` |

**Reconstruct-specific events** (broadcast on `track_id` channel):

| Event | When |
|---|---|
| `preview_downloading` | Audio download started |
| `preview_synthesizing` | TTS generation started |
| `preview_quality_check` | Quality assessment started |
| `preview_ready` | Preview available for review |
| `confirm_splicing` | Confirm splice started |
| `confirm_uploading` | Upload to B2 started |
| `confirm_complete` | Final audio ready |
| `preview_rolled_back` | Preview discarded |

---

## 4. REST API Summary

### Submit Job
```
POST /api/v1/process
{
  "job_id": "uuid",
  "track_id": "string",
  "job_type": "pipeline|reconstruct|magic_clean|transcription|categorization|audio_tag|edit_transcript",
  "audio_url": "https://...",
  "callback_url": "https://...",
  "changes": [{"segment_start": 1.0, "segment_end": 2.0, "new_text": "..."}],
  "same_speaker": true,
  "user_id": "optional"
}
Response: 202 { job_id, run_id, status: "accepted" }
```

### Direct Reconstruct (sync preview)
```
POST /api/v1/reconstruct
{
  "audio_url": "https://...",
  "track_id": "string",
  "segment_start": 1.0,
  "segment_end": 2.0,
  "new_text": "hello world",
  "same_speaker": true
}
Response: 200 { preview_id, preview_audio_url, quality_metrics, expires_at }
```

### Confirm Preview
```
POST /api/v1/reconstruct/confirm
{ "preview_id": "uuid", "track_id": "string" }
Response: 200 { audio_url, b2_key, duration, status: "completed" }
```

### Rollback Preview
```
POST /api/v1/reconstruct/rollback?preview_id=uuid
Response: 200 { preview_id, status: "rolled_back" }
```

### Remove Segment (no TTS, just cut)
```
POST /api/v1/reconstruct/remove
{ "track_id": "string", "audio_url": "https://...", "segment_start": 1.0, "segment_end": 2.0 }
Response: 200 { audio_url, b2_key, duration, segments_removed: 1 }
```

### Edit Transcript (auto-diff + reconstruct)
```
POST /api/v1/edit-transcript
{ "job_id": "uuid", "track_id": "string", "edited_transcript": "full corrected text..." }
Response: 202 { job_id, run_id, status: "accepted" }
```

### Check Job Status
```
GET /api/v1/jobs/{job_id}
Response: 200 { job_id, status, current_stage, result, error, track_state }
```

### Cancel Job
```
POST /api/v1/jobs/{job_id}/cancel
Response: 200 { job_id, status: "cancelled", cancelled: true }
```

### Queue Stats
```
GET /api/v1/queue/stats
Response: 200 { queued: 2, active: 3, total: 5, oldest_wait_s: 12.3, estimated_wait_s: 60 }
```

---

## 5. Integration Flow Example

```
1. Your backend calls POST /api/v1/reconstruct
   -> Gets back { preview_id, preview_audio_url, quality_metrics }

2. Your frontend plays preview_audio_url to the user.
   Frontend subscribes to SSE: GET /api/v1/events/{job_id}

3. User clicks "Confirm"
   -> Your backend calls POST /api/v1/reconstruct/confirm { preview_id, track_id }

4. Hear AI splices the TTS into the original track, uploads to B2

5. Your webhook receives:
   POST /your-callback-url
   { status: "completed", job_type: "reconstruct", result: { reconstructed_audio: { audio_url: "..." } } }

6. Your backend stores the new audio_url for the track

If your webhook endpoint is unreachable, Hear AI retries every 45 seconds.
You can also manually trigger: POST /api/v1/jobs/{job_id}/retry-callback
```
