# Hear AI — API Documentation & Integration Guide

## Table of Contents
1.  [Authentication](#1-authentication)
2.  [What's New](#2-whats-new)
3.  [API Endpoints](#3-api-endpoints)
4.  [Job Types & Stage Flows](#4-job-types--stage-flows)
5.  [Webhook / Callback](#5-webhook--callback)
6.  [Realtime Events (SSE & WebSocket)](#6-realtime-events-sse--websocket)
7.  [Regeneration Flow (edit transcript)](#7-regeneration-flow-edit-transcript)
8.  [Queue Visibility](#8-queue-visibility)
9.  [Health & Monitoring](#9-health--monitoring)
10. [Environment Configuration](#10-environment-configuration)
11. [Backend Integration Checklist](#11-backend-integration-checklist)


## 1. Authentication

All API endpoints (except `/health`) require the `X-Service-Key` header:

```
X-Service-Key: <your-ai-service-secret>
```

Configured via `AI_SERVICE_SECRET` in `.env`. Requests without a valid key receive `401 Authorization required`.

---

## 2. What's New

### Redis Queue (Scale)
- Queue positions are **live-tracked** and **crash-safe** via Redis
- O(log n) position lookups — no more in-memory guessing
- Failed jobs are **automatically re-enqueued** through Redis
- Queue state survives server restarts (recovered from PostgreSQL)

### Dynamic Stage Messages
- Every job type now has **human-readable stage labels**
- No model names or internal identifiers exposed to the backend
- Each stage includes `label`, `description`, and `progress_pct`
- Example: `"Transcribing audio"` → `"Checking content safety"` → `"Creating audio variants"`

### Regeneration Pipeline (edit transcript)
- Complete edit-transcript → regenerate → retranscribe → correct flow
- **Punctuation restoration**: Whisper output is aligned with the edited transcript to restore commas, periods, question marks
- **Word correction**: Fuzzy-matched mishearings (e.g., "Havring" → "Havering") are corrected
- **Fallback**: When Whisper fails on TTS audio (< 50% word accuracy), the edited transcript is used as ground truth
- Comma-aware TTS with explicit pause insertion
- Sentence-boundary snapping for clean TTS segments
- `is_regenerated` flag in PostgreSQL and callback payloads

### Comma-Aware TTS
- Text is split on comma boundaries for better prosody
- 150ms explicit silence between comma-separated phrases
- 400ms silence between sentences

### Time-Stretch Protection
- Speed change capped at 10% (0.9x–1.1x), was 50% (0.7x–1.5x)
- No speed-up on retries (removed hard truncation)
- Audio cross-correlation detects and skips overlapping content at splice points

### Scale Configuration (A40)
```
MAX_CONCURRENT_JOBS=3
MAX_CONCURRENT_PIPELINE_JOBS=4
MAX_CONCURRENT_GPU_JOBS=2
MAX_CONCURRENT_EDIT_TRANSCRIPT_JOBS=1
REDIS_URL=redis://localhost:6379/0
```

---

## 3. API Endpoints

### 3.1 Health Check
```
GET /health
No auth required
```

**Response**:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A40",
  "models_loaded": ["whisper", "mossformer2", "categorizer", "moderator", "fishspeech"],
  "active_jobs": 2,
  "queued_jobs": 1,
  "redis_status": "connected"
}
```

### 3.2 Submit Job
```
POST /api/v1/process
Content-Type: application/json
X-Service-Key: <secret>
```

**Request body**:
```json
{
  "job_id": "uuid-string",
  "track_id": "your-track-id",
  "job_type": "pipeline|transcription|categorization|audio_tag|edit_transcript|rebuild|reconstruct|magic_clean",
  "edited_transcript": "optional-full-transcript",
  "audio_url": "optional-override-url",
  "callback_url": "optional-webhook-url",
  "changes": [
    {
      "segment_start": 1.0,
      "segment_end": 2.5,
      "new_text": "corrected text",
      "original_text": "original text"
    }
  ],
  "same_speaker": true,
  "max_tags": 8
}
```

**Response** (202):
```json
{
  "job_id": "uuid",
  "status": "accepted"
}
```

### 3.3 Get Job Status
```
GET /api/v1/jobs/{job_id}
X-Service-Key: <secret>
```

**Response**:
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "job_type": "pipeline",
  "status": "running",
  "current_stage": null,
  "track_id": "track-id",
  "attempts": 0,
  "result": null,
  "error": null,
  "callback_delivered": false,
  "created_at": "2026-06-22T23:56:38",
  "started_at": null,
  "completed_at": null,
  "track_state": null
}
```

### 3.4 Cancel Job
```
POST /api/v1/jobs/{job_id}/cancel
X-Service-Key: <secret>
```

**Response** (200):
```json
{
  "job_id": "uuid",
  "status": "cancelled",
  "cancelled": true
}
```

**Not found** (404):
```json
{
  "detail": "Job not found"
}
```

### 3.5 Queue Stats
```
GET /api/v1/queue/stats
X-Service-Key: <secret>
```

**Response** (200):
```json
{
  "queued": 5,
  "active": 2,
  "total": 7,
  "oldest_wait_s": 45.3,
  "estimated_wait_s": 150.0,
  "avg_job_duration_s": 30.0
}
```

### 3.6 Realtime Events (SSE)
```
GET /api/v1/events/{job_id}
X-Service-Key: <secret>
Accept: text/event-stream
```

Connects a Server-Sent Events stream. Sends `queue_position` immediately on connect, then real-time stage updates.

### 3.7 Realtime Events (WebSocket)
```
WS /ws/{job_id}
X-Service-Key: <secret>
```

WebSocket connection. Sends `queue_position` immediately on connect, then real-time stage updates.

### 3.8 Retry Callback
```
POST /api/v1/jobs/{job_id}/retry-callback
X-Service-Key: <secret>
```

Manually retriggers the webhook callback for a completed job.

---

## 4. Job Types & Stage Flows

### 4.1 pipeline
Stages:
```
"Transcribing audio"       (0–25%)  Converting speech to text
"Correcting transcript"    (25–30%) Applying punctuation and word fixes
"Checking content safety"  (30–45%) Running content moderation checks
"Tagging content"          (45–55%) Categorizing by topic and theme
"Building discovery"       (55–60%) Creating content profile
"Creating audio variants"  (60–100%) Generating MP3 and speed layers
```

### 4.2 edit_transcript
```
"Downloading audio"        (0–10%)  Fetching source audio
"Transcribing audio"       (10–25%) Getting word timestamps from speech
"Finding changes"          (25–35%) Comparing original vs edited text
"Regenerating speech"      (35–100%) Creating new audio for edited parts
```

### 4.3 magic_clean
```
"Enhancing audio quality"  (0–100%) Improving clarity and reducing noise
```

Sub-stages broadcast as nested events with labels like:
- "Removing DC offset"
- "Deep noise reduction"
- "Enhancing voice clarity"
- "Equalizing speech"
- "Reducing sibilance"
- "Normalizing loudness"

### 4.4 rebuild
```
"Rebuilding audio"         (0–70%)  Regenerating entire track from text
"Checking content safety"  (70–80%) Running content moderation checks
"Building discovery"       (80–100%) Creating content profile
```

### 4.5 transcription
```
"Transcribing audio"       (0–100%) Converting speech to text
```

### 4.6 categorization
```
"Transcribing audio"       (0–40%)  Converting speech to text
"Checking content safety"  (40–50%) Running content moderation checks
"Tagging content"          (50–100%) Categorizing by topic and theme
```

### 4.7 reconstruct
```
"Downloading audio"        (0–15%)  Fetching source audio
"Reconstructing segments"  (15–100%) Replacing audio at edited positions
```

### 4.8 audio_tag
```
"Extracting audio tags"    (0–100%) Identifying spoken keywords
```

---

## 5. Webhook / Callback

When a job completes (or fails), Hear AI POSTs to your `callback_url`.

### 5.1 Job Completed (pipeline / transcription / categorization)
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "track_id": "your-track-id",
  "user_id": null,
  "job_type": "pipeline",
  "status": "completed",
  "error": null,
  "result": {
    "job_id": "uuid",
    "run_id": "uuid",
    "job_type": "pipeline",
    "track_id": "your-track-id",
    "source_audio_url": "https://cdn.example.com/audio.mp3",
    "transcription": {
      "transcript": "Full transcript text...",
      "segments": [],
      "language": "en",
      "confidence": 0.98,
      "restored": false
    },
    "moderation": {
      "flagged": false,
      "severity": "none",
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
    "is_regenerated": false,
    "compressed_audio": {
      "audio_url": "https://cdn.example.com/compressed.mp3",
      "b2_key": "compressed/uuid.mp3",
      "audio_format": "mp3"
    }
  }
}
```

### 5.2 Job Completed (edit_transcript)
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "track_id": "your-track-id",
  "job_type": "edit_transcript",
  "status": "completed",
  "result": {
    "changes_detected": 3,
    "edited_transcript": "full corrected transcript text...",
    "is_regenerated": true,
    "reconstructed_audio": {
      "audio_url": "https://cdn.example.com/reconstructed/uuid.mp3",
      "b2_key": "reconstructed/uuid.mp3",
      "duration": 120.5,
      "audio_format": "mp3"
    }
  }
}
```

### 5.3 Job Completed (rebuild)
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "track_id": "your-track-id",
  "job_type": "rebuild",
  "status": "completed",
  "result": {
    "transcription": {
      "transcript": "edited transcript text",
      "edited": true
    },
    "edited_transcript": "edited transcript text",
    "is_regenerated": true,
    "rebuilt_audio": {
      "audio_url": "https://cdn.example.com/rebuild/uuid.mp3",
      "b2_key": "rebuild/uuid.mp3",
      "duration": 97.2,
      "audio_format": "mp3"
    }
  }
}
```

### 5.4 Job Failed (any type)
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "track_id": "your-track-id",
  "job_type": "pipeline",
  "status": "failed",
  "result": null,
  "error": "Audio processing failed. Please check the source file and try again."
}
```

Error messages are sanitized — they never contain file paths, model names, or internal stack traces.

### 5.5 Retry Behavior
- Hear AI retries failed callbacks every 45 seconds
- Maximum: configurable via `CALLBACK_RETRY_POLL_SECONDS`
- Manual retrigger: `POST /api/v1/jobs/{job_id}/retry-callback`

---

## 6. Realtime Events (SSE & WebSocket)

### 6.1 SSE Events

| Event | When | Payload |
|-------|------|---------|
| `queue_position` | On connect and on every queue change | `{event, job_id, position, total_queued, estimated_wait_s}` |
| `stage_changed` | On each pipeline stage transition | `{event, job_id, stage, label, description, progress_pct, status}` |
| `job_completed` | On successful completion | `{event, job_id, status: "completed", result}` |
| `job_failed` | On unrecoverable failure | `{event, job_id, status: "failed", error}` |

### 6.2 stage_changed Event Detail

```json
{
  "event": "stage_changed",
  "job_id": "uuid",
  "run_id": "uuid",
  "track_id": "your-track-id",
  "job_type": "pipeline",
  "status": "running",
  "current_stage": "transcribing",
  "label": "Transcribing audio",
  "description": "Converting speech to text",
  "progress_pct": 15
}
```

### 6.3 queue_position Event Detail

```json
{
  "event": "queue_position",
  "job_id": "uuid",
  "position": 3,
  "total_queued": 7,
  "estimated_wait_s": 45
}
```

Sent **on WebSocket/SSE connect** and on every queue change (enqueue/dequeue/cancel).

---

## 7. Regeneration Flow (edit transcript)

### Full Flow
```
1. Backend calls POST /api/v1/process with job_type="edit_transcript"
   - Pass the full edited_transcript text
   - Pass the track_id

2. Hear AI:
   a. Downloads original audio
   b. Transcribes it to get word timestamps
   c. Diffs original vs edited transcript (sentence-aligned)
   d. Generates TTS for changed segments (comma-aware, emotion-injected)
   e. Splices TTS into original waveform
   f. Retranscribes regenerated audio with Whisper
   g. Runs correction pipeline:
      - restore_punctuation_from_edit() — restores , . ! ? ;
      - correct_whisper_mishearings() — fuzzy-matches misheard words
      - Fallback: if word accuracy < 50%, uses edited transcript as ground truth
   h. Stores is_regenerated=true in PostgreSQL
   i. Sends callback with is_regenerated: true

3. Backend receives callback:
   - Store reconstructed_audio.audio_url as the new track audio
   - Store edited_transcript as the track's transcription
   - Mark track as regenerated

4. Backend sends pipeline job for the track:
   - Hear AI queries DB → finds is_regenerated=true
   - Runs Whisper on the regenerated audio
   - Runs correction pipeline against the edited transcript
   - Result: verified transcript with 100% punctuation and word accuracy
```

### Key Fields in edit_transcript Callback
```json
{
  "result": {
    "is_regenerated": true,
    "edited_transcript": "full corrected text",
    "reconstructed_audio": {
      "audio_url": "https://...",
      "duration": 120.5
    }
  }
}
```

---

## 8. Queue Visibility

### How to show queue position to users:

**Step 1**: Submit job → get `job_id`
```
POST /api/v1/process → {"job_id": "abc", "status": "accepted"}
```

**Step 2**: Connect to SSE or WebSocket
```
SSE:      GET /api/v1/events/abc
WebSocket: WS /ws/abc
```

**Step 3**: Receive live position updates
```
On connect:  {"event": "queue_position", "position": 5, "total_queued": 12}
On dequeue:  {"event": "queue_position", "position": 4, "total_queued": 11}
On running:  {"event": "queue_position", "position": 0, "total_queued": 10}
             {"event": "stage_changed", "stage": "transcribing", "label": "Transcribing audio"}
```

**Step 4**: Poll queue stats for dashboard
```
GET /api/v1/queue/stats
```

---

## 9. Health & Monitoring

### Health Check
```
GET /health
```
Returns GPU status, loaded models, Redis status, active/queued job counts.

### Queue Stats
```
GET /api/v1/queue/stats
```
Returns queued count, active count, oldest wait time, estimated wait time, average job duration.

### Worker Logs
```
make logs       → tail -f logs/hear-ai.out.log
make errors     → tail -f logs/hear-ai.err.log
make status     → supervisorctl status
```

---

## 10. Environment Configuration

### Required in .env
```bash
AI_SERVICE_SECRET=<64-char-secret>
DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db
B2_KEY_ID=<backblaze-key-id>
B2_APPLICATION_KEY=<backblaze-key>
B2_BUCKET_NAME=hear-dev-uploads
REDIS_URL=redis://localhost:6379/0
REDIS_QUEUE_PREFIX=hear
HEAR_BACKEND_URL=https://internal.hear.surf
HEAR_CALLBACK_URL=https://internal.hear.surf/api/v1/internal/ai-callback
```

### Scale Tuning
```bash
MAX_CONCURRENT_GPU_JOBS=2        # A40 can handle 2 concurrent GPU ops
MAX_CONCURRENT_JOBS=3            # Total simultaneous jobs
MAX_CONCURRENT_PIPELINE_JOBS=4   # Pipeline type limit
MAX_CONCURRENT_EDIT_TRANSCRIPT_JOBS=1  # TTS is heavy, one at a time
JOB_MAX_RETRIES=8                # Auto-retry failed jobs
EDIT_PHRASE_EXPANSION_WORDS=1    # Context words for TTS (1=minimal)
```

### Models
```bash
WHISPER_MODEL_SIZE=large-v3       # Faster-Whisper model
WHISPER_BEAM_SIZE=5
WHISPER_WORD_TIMESTAMPS=true
QWEN_LLM_ENABLED=true            # Optional LLM for moderation/categorization
FISH_SPEECH_TTS_ENABLED=true     # TTS for audio regeneration
```

---

## 11. Backend Integration Checklist

### When submitting jobs:
- [ ] Always pass a unique `job_id` (UUID)
- [ ] Pass `track_id` matching your backend's track identifier
- [ ] For `edit_transcript`: pass the full `edited_transcript` text
- [ ] For `reconstruct`: pass `changes[]` array with segment_start, segment_end, new_text
- [ ] Optionally pass `callback_url` for per-job webhook delivery
- [ ] Always validate the `X-Service-Key` header

### When receiving callbacks:
- [ ] Verify `X-Service-Key` on your webhook endpoint
- [ ] Check `result.is_regenerated` flag — if true, the audio was regenerated
- [ ] Store `reconstructed_audio.audio_url` as the track's new audio URL
- [ ] Store `edited_transcript` as the track's transcription (when `is_regenerated: true`)
- [ ] Return 2xx within 30 seconds — Hear AI retries on failure

### When showing queue position to users:
- [ ] Connect to SSE (`/api/v1/events/{job_id}`) after submitting
- [ ] Display `position` and `estimated_wait_s` from `queue_position` events
- [ ] Show `label` and `progress_pct` from `stage_changed` events
- [ ] Poll `/api/v1/queue/stats` for dashboard-level queue overview

### For regenerated audio:
- [ ] Always pass `edited_transcript` in follow-up pipeline jobs for regenerated tracks
- [ ] The correction pipeline will restore punctuation, fix mishearings, and fallback if needed
- [ ] The result's `transcription.restored=true` flag indicates the correction pipeline ran
- [ ] The result's `transcription.whisper_failed=true` flag indicates the edited transcript was used as ground truth (Whisper accuracy < 30%)

### For production deployment:
- [ ] Redis must be running on localhost:6379 (or configured via REDIS_URL)
- [ ] PostgreSQL must be reachable with schema initialized (`make migrate`)
- [ ] Backblaze B2 credentials configured for audio uploads
- [ ] Fish Speech TTS server running on port 8080 (for regeneration)
- [ ] GPU available (A40 recommended) with CUDA
- [ ] `make up` for full bootstrap, `make status` to verify, `make logs` to monitor
