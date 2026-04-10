# Hear AI — Backend Integration Guide

Complete API contract between your NestJS backend and the Hear AI service.

---

## Architecture

```
User → NestJS Backend → Hear AI (RunPod GPU)
                      ← callback with results
```

Your backend makes **1 outbound call** (submit job) and implements **3 inbound endpoints** (serve data + receive results).

---

## Authentication

All requests use the `X-Service-Key` header. Must match `AI_SERVICE_SECRET` on both sides.

```
X-Service-Key: your-shared-secret
```

---

## Outbound: Submit a Job

When a recording is uploaded, submit it for AI processing.

### `POST {HEAR_AI_URL}/api/v1/process`

**Headers:**
```
X-Service-Key: <AI_SERVICE_SECRET>
Content-Type: application/json
```

**Request:**
```json
{
  "recording_id": "rec_abc123",
  "job_id": "unique-uuid-v4",
  "callback_url": "https://api.yourapp.com/api/internal/ai-callback",
  "skip_enhancement": false,
  "skip_transcription": false,
  "existing_transcript": null,
  "max_tags": 8
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `recording_id` | string | ✅ | Your recording ID |
| `job_id` | string | ✅ | UUID you generate to track this job |
| `callback_url` | string | ✅ | Where Hear AI sends results |
| `skip_enhancement` | bool | ❌ | Skip audio enhancement (default `false`) |
| `skip_transcription` | bool | ❌ | Skip transcription (default `false`) |
| `existing_transcript` | string | ❌ | Provide existing transcript instead of running Whisper |
| `max_tags` | int | ❌ | Max tags to return (default `8`) |

> Tags and blocked keywords are automatically fetched from your platform settings endpoint — no need to send them per job.

**Response (202):**
```json
{
  "job_id": "unique-uuid-v4",
  "status": "accepted"
}
```

---

## Inbound: Endpoints Your Backend Must Implement

### 1. `GET /api/internal/recordings/:id`

Hear AI calls this to fetch recording metadata and tracks when a job starts.

**Request headers:**
```
X-Service-Key: <AI_SERVICE_SECRET>
```

**Response (200):**
```json
{
  "id": "rec_abc123",
  "title": "Team Meeting",
  "audio_url": "https://b2.example.com/recordings/rec_abc123.wav",
  "tracks": [
    {
      "id": "track_001",
      "audio_url": "https://b2.example.com/tracks/track_001.wav",
      "name": "Speaker 1",
      "volume": 1.0,
      "is_muted": false,
      "sort_order": 0,
      "duration": 312.5
    },
    {
      "id": "track_002",
      "audio_url": "https://b2.example.com/tracks/track_002.wav",
      "name": "Speaker 2",
      "volume": 0.8,
      "is_muted": false,
      "sort_order": 1,
      "duration": 312.5
    }
  ]
}
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | string | ✅ | Recording ID |
| `title` | string | ❌ | Recording title |
| `audio_url` | string | ❌ | Main audio URL |
| `tracks[].id` | string | ✅ | Track ID |
| `tracks[].audio_url` | string | ✅ | **Must be a downloadable audio file URL** |
| `tracks[].name` | string | ❌ | Track/speaker name |
| `tracks[].volume` | float | ❌ | Volume 0.0–1.0 (default `1.0`) |
| `tracks[].is_muted` | bool | ❌ | Muted tracks are skipped (default `false`) |
| `tracks[].sort_order` | int | ❌ | Track ordering (default `0`) |
| `tracks[].duration` | float | ❌ | Duration in seconds |

---

### 2. `GET /api/internal/platform-settings`

Hear AI calls this once per job for moderation and auto-tagging config.

**Request headers:**
```
X-Service-Key: <AI_SERVICE_SECRET>
```

**Query params (optional):**
```
?organization_id=org_123
```

**Response (200):**
```json
{
  "blocked_keywords": "violence,hate speech,explicit",
  "auto_tag_keywords": "meeting,standup,interview,podcast"
}
```

| Field | Type | Notes |
|-------|------|-------|
| `blocked_keywords` | string | Comma-separated. Used for content moderation |
| `auto_tag_keywords` | string | Comma-separated. Added to categorization |

> If this endpoint is unreachable, Hear AI continues with empty defaults. Non-critical.

---

### 3. `POST /api/internal/ai-callback`

Hear AI sends results here when processing completes or fails.

**Request headers:**
```
X-Service-Key: <AI_SERVICE_SECRET>
Content-Type: application/json
```

> **You must return 2xx to confirm receipt.** Any other status triggers retries (10 attempts over ~30 minutes).

#### Success Payload

```json
{
  "job_id": "unique-uuid-v4",
  "job_type": "pipeline",
  "status": "completed",
  "result": {
    "recording_id": "rec_abc123",

    "tracks": {
      "track_001": {
        "enhanced_url": "https://b2.example.com/enhanced/track_001.wav",
        "b2_key": "enhanced/track_001.wav",
        "quality_score": 0.92,
        "snr_db": 24.5
      },
      "track_002": {
        "enhanced_url": "https://b2.example.com/enhanced/track_002.wav",
        "b2_key": "enhanced/track_002.wav",
        "quality_score": 0.87,
        "snr_db": 21.3
      }
    },

    "master": {
      "master_url": "https://b2.example.com/masters/rec_abc123.wav",
      "b2_key": "masters/rec_abc123.wav"
    },

    "per_track_transcriptions": {
      "track_001": {
        "transcript": "Hello everyone, let's get started with today's standup.",
        "segments": [
          { "start": 0.0, "end": 3.2, "text": "Hello everyone,", "confidence": 0.97 },
          { "start": 3.2, "end": 6.8, "text": "let's get started with today's standup.", "confidence": 0.95 }
        ],
        "language": "en",
        "confidence": 0.96,
        "duration": 312.5
      }
    },

    "transcription": {
      "transcript": "Full combined transcript of all tracks mixed together...",
      "segments": [
        { "start": 0.0, "end": 3.2, "text": "Hello everyone,", "confidence": 0.97 },
        { "start": 3.2, "end": 6.8, "text": "let's get started with today's standup.", "confidence": 0.95 }
      ],
      "language": "en",
      "language_probability": 0.99,
      "duration": 312.5,
      "confidence": 0.95
    },

    "categorization": {
      "tags": ["meeting", "standup", "engineering"],
      "categories": ["Business", "Technology"],
      "sentiment": "neutral",
      "confidence_scores": {
        "meeting": 0.95,
        "standup": 0.88,
        "engineering": 0.72
      }
    },

    "moderation": {
      "flagged": false,
      "severity": "low",
      "intent": "cautionary",
      "reason": "Speaker discusses a robbery incident in a news reporting context, not promoting harmful activity",
      "flagged_categories": [],
      "blocked_words_found": []
    }
  }
}
```

#### Failure Payload

```json
{
  "job_id": "unique-uuid-v4",
  "job_type": "pipeline",
  "status": "failed",
  "error": "Failed to download audio from track_001: HTTP 404"
}
```

---

## Result Fields Reference

### `result.tracks[trackId]`

| Field | Type | Description |
|-------|------|-------------|
| `enhanced_url` | string | B2 URL of the enhanced audio |
| `b2_key` | string | B2 object key |
| `quality_score` | float | Audio quality 0.0–1.0 |
| `snr_db` | float | Signal-to-noise ratio in dB |

### `result.master`

| Field | Type | Description |
|-------|------|-------------|
| `master_url` | string | Mixed master audio URL |
| `b2_key` | string | B2 object key |

Only present when there are multiple tracks.

### `result.transcription`

| Field | Type | Description |
|-------|------|-------------|
| `transcript` | string | Full text |
| `segments` | array | Timestamped segments |
| `segments[].start` | float | Segment start (seconds) |
| `segments[].end` | float | Segment end (seconds) |
| `segments[].text` | string | Segment text |
| `segments[].confidence` | float | Confidence 0.0–1.0 |
| `language` | string | ISO language code |
| `language_probability` | float | Language detection confidence |
| `duration` | float | Total audio duration (seconds) |
| `confidence` | float | Overall confidence |

### `result.categorization`

| Field | Type | Description |
|-------|------|-------------|
| `tags` | string[] | Auto-generated tags |
| `categories` | string[] | Content categories |
| `sentiment` | string | `positive` / `neutral` / `negative` |
| `confidence_scores` | object | Tag → confidence mapping |

### `result.moderation`

| Field | Type | Description |
|-------|------|-------------|
| `flagged` | bool | `true` if content is flagged |
| `categories` | object | Category → flagged boolean |
| `scores` | object | Category → score float |

---

## Other Available Endpoints

### Poll Job Status

```
GET {HEAR_AI_URL}/api/v1/jobs/{job_id}
Header: X-Service-Key: <secret>
```

```json
{
  "job_id": "uuid",
  "status": "completed",
  "recording_id": "rec_abc123",
  "result": { ... },
  "error": null,
  "callback_delivered": true,
  "created_at": "2026-04-10T12:00:00",
  "completed_at": "2026-04-10T12:02:15"
}
```

Status values: `pending` → `enhancing` → `transcribing` → `categorizing` → `moderating` → `completed` / `failed`

### Retry Callback

If your backend missed the callback, trigger re-delivery:

```
POST {HEAR_AI_URL}/api/v1/jobs/{job_id}/retry-callback
Header: X-Service-Key: <secret>
```

```json
{ "status": "delivered", "job_id": "uuid" }
```

### Health Check

```
GET {HEAR_AI_URL}/health
```

```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A40",
  "models_loaded": ["whisper", "demucs", "categorizer", "moderator"],
  "active_jobs": 2,
  "queued_jobs": 5
}
```

### Standalone Endpoints

These don't require a recording — useful for testing:

```
POST {HEAR_AI_URL}/api/v1/categorize
Body: { "text": "transcript text", "custom_tags": [], "max_tags": 5 }

POST {HEAR_AI_URL}/api/v1/moderate
Body: { "text": "text to check" }
```

---

## Environment Variables

Both services need the **same** secret:

```env
# On your NestJS backend
HEAR_AI_URL=https://<pod-id>-8000.proxy.runpod.net
AI_SERVICE_SECRET=your-shared-secret-here

# On Hear AI (.env)
AI_SERVICE_SECRET=your-shared-secret-here
HEAR_BACKEND_URL=https://api.yourapp.com
```

---

## Database Columns to Add

### Recordings Table

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `ai_job_id` | varchar | null | Hear AI job ID |
| `ai_status` | varchar | `pending` | `pending` / `processing` / `completed` / `failed` |
| `ai_error` | text | null | Error message if failed |
| `transcript` | text | null | Combined transcript |
| `transcript_segments` | jsonb | null | Timestamped segments array |
| `language` | varchar | null | Detected language code |
| `tags` | jsonb | null | Auto-generated tags |
| `categories` | jsonb | null | Content categories |
| `sentiment` | varchar | null | Sentiment result |
| `is_flagged` | boolean | false | Moderation flag |
| `moderation_categories` | jsonb | null | Flagged categories detail |

### Tracks Table

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `enhanced_audio_url` | varchar | null | Enhanced audio B2 URL |
| `quality_score` | float | null | Audio quality 0.0–1.0 |
| `snr_db` | float | null | Signal-to-noise ratio |
| `transcript` | text | null | Per-track transcript |
| `transcript_segments` | jsonb | null | Per-track segments |
| `language` | varchar | null | Per-track language |
