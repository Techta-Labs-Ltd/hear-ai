# Hear AI Service

GPU-accelerated audio processing microservice for the [Hear](https://hear.surf) platform. Handles audio enhancement, transcription, categorization, content moderation, and speech reconstruction — all from a single `recording_id`.

---

## Architecture

```
Hear Backend                          Hear AI (RunPod GPU)
┌──────────────┐     POST /process    ┌──────────────────────────────┐
│              │ ───────────────────▶  │  FastAPI + Uvicorn           │
│  Backend API │                      │                              │
│              │  ◀─── webhook ─────  │  Worker (async job queue)    │
└──────────────┘     callback_url     │    ├─ Fetch recording        │
                                      │    ├─ Fetch platform settings │
                                      │    ├─ Enhance (Demucs)       │
                                      │    ├─ Mix tracks → master    │
                                      │    ├─ Transcribe (Whisper)   │
                                      │    ├─ Categorize (ZSC + GPT) │
                                      │    ├─ Moderate (Toxic-BERT)  │
                                      │    └─ Callback result        │
                                      └──────────────────────────────┘
```

## Features

| Feature | Description |
|---|---|
| **Multi-Track Processing** | Fetches all tracks for a recording, enhances each independently, mixes into a master |
| **Audio Enhancement** | Demucs vocal isolation → EQ → de-essing → silence removal → loudness normalization |
| **Transcription** | Faster-Whisper (large-v3) with word-level timestamps, VAD filtering, empty segment removal |
| **Categorization** | 3-layer system: keyword rules + zero-shot classification + OpenAI GPT — merged with weighted scores |
| **Content Moderation** | Toxic-BERT (local) + OpenAI Moderation API + platform blocked keywords |
| **Speech Reconstruction** | Edit transcript → auto-detect gender + accent (US/UK/AU) → re-synthesize with edge-tts |
| **Realtime Streaming** | SSE and WebSocket endpoints for live transcription progress |
| **Platform Integration** | Fetches blocked keywords & auto-tag keywords from Hear backend settings |

---

## Project Structure

```
hear-ai/
├── app/
│   ├── api/
│   │   ├── auth.py              # X-Service-Key header auth
│   │   ├── router.py            # Mounts all v1 routes
│   │   └── v1/
│   │       ├── categorization.py
│   │       ├── enhancement.py
│   │       ├── health.py
│   │       ├── moderation.py
│   │       ├── pipeline.py      # Main pipeline + reconstruct + jobs + SSE/WS
│   │       └── transcription.py
│   ├── core/
│   │   ├── category_loader.py   # Loads categories/tags/keywords from categories.txt
│   │   ├── downloader.py        # Downloads audio from URL to temp file
│   │   ├── gpu.py               # GPU semaphore for concurrent job control
│   │   ├── platform_settings.py # Fetches blocked/auto-tag keywords from Hear API
│   │   ├── recording_fetcher.py # Fetches recording + tracks from Hear API
│   │   └── storage.py           # B2 (S3-compatible) upload/download
│   ├── models/
│   │   ├── database.py          # SQLite job tracking (SQLAlchemy)
│   │   └── schemas.py           # Pydantic request/response models
│   ├── realtime/
│   │   ├── broadcaster.py       # SSE + WebSocket connection manager
│   │   └── orchestrator.py      # Realtime pipeline (enhance → transcribe → categorize)
│   ├── services/
│   │   ├── callback.py          # POST results back to Hear backend
│   │   ├── categorizer.py       # 3-layer categorization engine
│   │   ├── enhancer.py          # Demucs denoising + EQ + silence removal
│   │   ├── mixer.py             # Multi-track audio mixer using torchaudio
│   │   ├── moderator.py         # Toxic-BERT + OpenAI + blocked keyword check
│   │   ├── registry.py          # Singleton service instances
│   │   ├── synthesizer.py       # TTS reconstruction with auto voice/accent detection
│   │   └── transcriber.py       # Faster-Whisper transcription
│   ├── config.py                # Pydantic settings (from .env)
│   ├── main.py                  # FastAPI app, lifespan, Sentry, CORS
│   └── worker.py                # Async job queue + multi-track pipeline
├── data/
│   └── categories.txt           # 65 categories, 160+ tags, 35 keyword rules
├── tests/
│   └── test_api.py              # Integration test suite
├── Dockerfile                   # CUDA 12.1 + Python 3.12 + ffmpeg
├── requirements.txt
├── .env.example
├── .gitignore
└── .dockerignore
```

---

## Pipeline Flow

When the Hear backend sends a `POST /api/v1/process` request, the pipeline executes these steps:

```
1. Fetch recording + tracks from Hear API   (GET /api/internal/recordings/:id)
2. Fetch platform settings                  (GET /api/internal/platform-settings)
3. Download each track's audio
4. Enhance each track independently         (Demucs → EQ → de-ess → silence strip → normalize)
5. Upload enhanced tracks to B2
6. Mix enhanced tracks → master audio       (respects volume/mute per track)
7. Upload master to B2
8. Transcribe each track individually       (Whisper large-v3, per-track segments)
9. Transcribe combined master               (combined transcript for categorization)
10. Categorize combined transcript           (keywords + zero-shot + OpenAI + auto-tag keywords)
11. Moderate combined transcript             (Toxic-BERT + OpenAI + blocked keywords)
12. Callback result to Hear backend          (POST to callback_url)
```

---

## API Endpoints

All endpoints require `X-Service-Key` header matching `AI_SERVICE_SECRET`.

### Pipeline

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/process` | Queue a recording for full pipeline processing |
| `POST` | `/api/v1/process-realtime` | Upload audio file for live streaming results |
| `POST` | `/api/v1/reconstruct` | Re-synthesize an edited transcript segment |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and result |
| `GET` | `/api/v1/events/{job_id}` | SSE stream for realtime progress |
| `WS` | `/ws/{job_id}` | WebSocket stream for realtime progress |

### Standalone

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/transcribe` | Transcribe audio file |
| `POST` | `/api/v1/enhance` | Enhance audio file |
| `POST` | `/api/v1/categorize` | Categorize text |
| `POST` | `/api/v1/moderate` | Moderate text |
| `GET` | `/api/v1/health` | Service health + GPU info |

---

## Pipeline Request

```json
POST /api/v1/process
{
  "recording_id": "e8c71ce2-fcd1-45ab-ae8c-036d8b3dd978",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "callback_url": "https://api.hear.surf/webhook/ai",
  "skip_enhancement": false,
  "skip_transcription": false,
  "max_tags": 8
}
```

## Pipeline Response (via callback)

```json
{
  "job_id": "...",
  "status": "completed",
  "result": {
    "recording_id": "...",
    "tracks": {
      "track-id-1": {
        "enhanced_url": "https://b2.../enhanced/rec/track1.wav",
        "b2_key": "enhanced/rec/track1.wav",
        "quality_score": 0.87,
        "snr_db": 22.5
      }
    },
    "master": {
      "master_url": "https://b2.../masters/rec/job.wav",
      "b2_key": "masters/rec/job.wav"
    },
    "per_track_transcriptions": {
      "track-id-1": {
        "transcript": "Hello world...",
        "segments": [...],
        "language": "en",
        "confidence": 0.94
      }
    },
    "transcription": {
      "transcript": "Full combined transcript...",
      "segments": [...],
      "language": "en",
      "confidence": 0.92
    },
    "categorization": {
      "tags": ["#BreakingNews", "#Politics"],
      "categories": ["News"],
      "confidence_scores": {...},
      "sentiment": "neutral"
    },
    "moderation": {
      "flagged": false,
      "categories": {...},
      "scores": {...},
      "openai": {...},
      "blocked_words": []
    }
  }
}
```

---

## Reconstruction (Auto Voice Detection)

When a user edits a transcript segment, call `/api/v1/reconstruct` to re-synthesize the audio:

```json
POST /api/v1/reconstruct
{
  "audio_url": "https://media.hear.surf/.../recording.wav",
  "recording_id": "e8c71ce2-...",
  "segment_start": 12.5,
  "segment_end": 15.8,
  "new_text": "The corrected sentence goes here"
}
```

The service automatically:
1. Extracts the audio segment at `[start, end]`
2. Analyzes **pitch (F0)** to detect gender (male < 165 Hz, female ≥ 165 Hz)
3. Analyzes **intonation contour** to detect accent:
   - **Australian** — rising terminal pitch (tail > head by 8%+)
   - **British** — wide pitch variability (std/mean > 0.25 or range > 120 Hz)
   - **American** — flat, monotone pitch (default)
4. Selects the matching `edge-tts` neural voice
5. Synthesizes, crossfades, and returns the reconstructed audio

| Detection | Voice |
|---|---|
| `male_us` | en-US-GuyNeural |
| `female_us` | en-US-JennyNeural |
| `male_uk` | en-GB-RyanNeural |
| `female_uk` | en-GB-SoniaNeural |
| `male_au` | en-AU-WilliamNeural |
| `female_au` | en-AU-NatashaNeural |

---

## Audio Enhancement

The enhancer applies the following chain:

```
Raw audio → Demucs vocal isolation → High-pass 80Hz → Low-pass 14kHz
  → Midrange EQ (-2dB @ 200Hz, +2dB @ 3kHz)
  → Compressor (0.1/0.3 attack/release)
  → De-esser (-3dB @ 7kHz, -2dB @ 9kHz)
  → Strip leading/trailing silence
  → Remove internal silence > 500ms
  → Loudness normalize to -16 LUFS
```

Output includes quality metrics: `quality_score`, `snr_db`, `peak_db`, `lufs`, `clipping_detected`.

---

## Categorization Engine

Three classification layers are run in parallel and merged:

| Layer | Method | Weight (with OpenAI) | Weight (without) |
|---|---|---|---|
| 1 — Keywords | Regex pattern matching from `categories.txt` | 25% | 45% |
| 2 — Zero-Shot | `cross-encoder/nli-distilroberta-base` | 35% | 55% |
| 3 — OpenAI | GPT-4o-mini structured prompt | 40% | — |

Tags with score ≥ 0.35 are returned; categories with score ≥ 0.45.

Sentiment analysis runs via `cardiffnlp/twitter-roberta-base-sentiment-latest`.

---

## Content Moderation

Three checks run in parallel:

1. **Toxic-BERT** (`unitary/toxic-bert`) — local GPU, classifies toxicity categories
2. **OpenAI Moderation API** — if `OPENAI_API_KEY` is set
3. **Blocked Keywords** — checks transcript against platform-configured blocked words from Hear settings

If any layer flags the content, `flagged: true` is returned with `blocked_words` listing which platform words matched.

---

## Required Backend Endpoints

The AI service calls these endpoints on the Hear backend:

### `GET /api/internal/recordings/:id`

Returns recording data with tracks.

```json
{
  "id": "...",
  "title": "...",
  "audio_url": "...",
  "tracks": [
    {
      "id": "...",
      "audio_url": "http://media.hear.surf/.../track.wav",
      "name": "Track 1",
      "volume": 1.0,
      "is_muted": false,
      "sort_order": 0,
      "duration": 30.6
    }
  ]
}
```

### `GET /api/internal/platform-settings`

Returns platform moderation and tagging settings.

```json
{
  "blocked_keywords": "spam,scam,fraud",
  "auto_tag_keywords": "news,breaking,exclusive,interview,report"
}
```

Both endpoints are authenticated via `X-Service-Key` header.

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `AI_SERVICE_SECRET` | ✅ | Shared secret for X-Service-Key auth |
| `HEAR_BACKEND_URL` | ✅ | Hear backend base URL (e.g. `https://api.hear.surf`) |
| `B2_KEY_ID` | ✅ | Backblaze B2 key ID |
| `B2_APPLICATION_KEY` | ✅ | Backblaze B2 application key |
| `B2_BUCKET_NAME` | ✅ | B2 bucket name |
| `B2_ENDPOINT_URL` | ✅ | B2 S3-compatible endpoint |
| `WHISPER_MODEL_SIZE` | — | Whisper model size (default: `large-v3`) |
| `MAX_CONCURRENT_GPU_JOBS` | — | Max parallel GPU jobs (default: `2`) |
| `SQLITE_DB_PATH` | — | Job database path (default: `./data/jobs.db`) |
| `DEMUCS_MODEL` | — | Demucs model variant (default: `htdemucs`) |
| `MODEL_CACHE_DIR` | — | Where models are cached (default: `/opt/ml/models`) |
| `CATEGORIES_FILE` | — | Path to categories file (default: `./data/categories.txt`) |
| `OPENAI_API_KEY` | — | OpenAI API key (enables GPT categorization + moderation) |
| `OPENAI_BASE_URL` | — | OpenAI base URL (default: `https://api.openai.com/v1`) |
| `OPENAI_MODEL` | — | OpenAI model for categorization (default: `gpt-4o-mini`) |
| `SENTRY_DSN` | — | Sentry DSN for error tracking |
| `SENTRY_TRACES_SAMPLE_RATE` | — | Sentry trace sampling rate (default: `0.3`) |
| `ENVIRONMENT` | — | Environment name (default: `production`) |

---

## Deployment

### RunPod (Recommended)

This service is designed to run on a RunPod GPU instance with NVIDIA CUDA support.

```bash
# Build
docker build -t hear-ai .

# Run
docker run --gpus all -p 8000:8000 --env-file .env hear-ai
```

### Manual

```bash
# Install dependencies (requires CUDA-capable PyTorch)
pip install -r requirements.txt

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### GPU Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM (RTX 3070 / A10G)
- **Recommended**: 16GB+ VRAM (A100 / RTX 4090) for concurrent jobs
- Models loaded at startup: Whisper large-v3 (~3GB), Demucs htdemucs (~300MB), Toxic-BERT (~250MB), Zero-shot NLI (~250MB), Sentiment (~250MB)

---

## Testing

```bash
# Set the target URL and secret
export AI_SERVICE_URL=http://localhost:8000
export AI_SERVICE_SECRET=your-secret

# Run integration tests
python -m tests.test_api
```

Tests cover: health check, auth validation, moderation, categorization, and the full multi-track pipeline flow.

---

## License

Private — Techta Labs Ltd. All rights reserved.
