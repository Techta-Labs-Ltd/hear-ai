# Hear AI Service

GPU-accelerated audio intelligence service for the [Hear](https://hear.surf) platform.

Current architecture is **track-first**:
- pipeline and standalone jobs run on `track_id`
- realtime payloads are run-scoped (`job_id`, `run_id`, `track_id`)
- `magic_clean` is standalone enhancement only
- rebuild supports edited transcript audio generation (self-hosted Higgs path)

### Database and temp files

- **PostgreSQL** is required. Set `DATABASE_URL` (see `.env.example`), e.g. `postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require` (the port must be a **number**, usually `5432`, never the literal word `PORT`). On first startup, `init_db()` creates tables, enables `pgcrypto`, and applies lightweight migrations.
- **Temp audio** lives under `HEAR_TMP_DIR` (default OS temp `hear-ai/`) with per-job subfolders `jobs/{job_id}/{run_id}/`. Tracked paths are stored in `ai_temp_files` and cleaned when jobs finish, cancel, or fail, plus a periodic sweep.
- **RunPod / bare metal**: use `start.sh` or `make start` so DNS, deps, env checks, Postgres ping, and an initial temp sweep run before Supervisor starts the app. Use `make errors-tail`, `make migrate`, `make clean-temp`, `make psql` as needed.

---

## Core Flow

### Pipeline (`job_type=pipeline`)
1. Fetch track: `GET /api/v1/internal/tracks/{track_id}/for-ai`
2. Transcription (Whisper)
3. Moderation (Toxic-BERT + optional Qwen)
4. Categorization/tagging (keyword + NLI + optional Qwen/OpenAI)
5. Callback delivery

### Rebuild (`job_type=rebuild`)
1. Accept edited transcript
2. Generate rebuilt speech with self-hosted Higgs (`higgs_audio` module)
3. Splice rebuilt speech into original track waveform
4. Re-run moderation and categorization from edited text
5. Return rebuilt audio metadata in result payload

### Magic Clean (`job_type=magic_clean`)
1. Fetch track
2. Enhance via Demucs/noise pipeline
3. Return enhancement payload only

---

## Main Endpoints

All endpoints require `X-Service-Key`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/process` | Submit track-first async job (`pipeline`, `rebuild`, `transcription`, `categorization`) |
| `POST` | `/api/v1/process-realtime` | Submit track-first realtime job |
| `POST` | `/api/v1/transcribe` | Standalone transcription job |
| `POST` | `/api/v1/enhance` | Standalone magic clean job |
| `POST` | `/api/v1/reconstruct` | Segment-level reconstruction on a track audio URL |
| `GET` | `/api/v1/jobs/{job_id}` | Job + run status |
| `GET` | `/api/v1/events/{job_id}` | SSE stream |
| `WS` | `/ws/{job_id}` | WebSocket stream |
| `GET` | `/health` | Service health + model availability |

---

## Request Examples

### Track-first pipeline submit

```json
POST /api/v1/process
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "track_id": "3162d6ce-aa3a-473e-aa25-fc5679eb60c0",
  "job_type": "pipeline",
  "max_tags": 8
}
```

### Rebuild submit

```json
POST /api/v1/process
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "track_id": "3162d6ce-aa3a-473e-aa25-fc5679eb60c0",
  "job_type": "rebuild",
  "edited_transcript": "Corrected wording for this track"
}
```

### Segment reconstruct

```json
POST /api/v1/reconstruct
{
  "audio_url": "https://media.hear.surf/uploads/...wav",
  "track_id": "3162d6ce-aa3a-473e-aa25-fc5679eb60c0",
  "segment_start": 12.5,
  "segment_end": 15.8,
  "new_text": "The corrected sentence goes here"
}
```

---

## Qwen Usage

Qwen is integrated through `app/services/llm_service.py`:
- model id: `Qwen/Qwen2.5-7B-Instruct`
- loaded at startup in `app/main.py`
- used by:
  - `app/services/moderator.py` in borderline/high-confidence moderation paths
  - `app/services/categorizer.py` in primary tagging/categorization paths

If Qwen is unavailable, service falls back to local moderation/tagging pipelines.

---

## Data Files

- `data/categories.txt`:
  - categories
  - tags
  - keyword rules for categorization
- `data/harm_keywords.txt`:
  - moderation keyword list

---

## Environment Variables

Key variables from `.env.example`:

| Variable | Purpose |
|---|---|
| `AI_SERVICE_SECRET` | Request authentication key |
| `HEAR_BACKEND_URL` | Backend base URL |
| `HEAR_CALLBACK_URL` | Job callback target |
| `MAX_CONCURRENT_JOBS` | Global concurrency |
| `MAX_CONCURRENT_PIPELINE_JOBS` | Pipeline/rebuild concurrency |
| `MAX_CONCURRENT_MAGIC_CLEAN_JOBS` | Magic clean concurrency |
| `JOB_MAX_RETRIES` | Re-queue processing after transient errors (before terminal `failed`) |
| `CALLBACK_RETRY_POLL_SECONDS` | Background interval to POST undelivered completed/failed callbacks |
| `WHISPER_MODEL_SIZE` | Whisper model id (default `distil-large-v3` for speed on one GPU) |
| `WHISPER_DUAL_PASS` | Second relaxed pass when first pass is empty (`false` = faster) |
| `WHISPER_BEAM_SIZE` | Decoder beam width (`1` = fastest) |
| `WHISPER_WORD_TIMESTAMPS` | Word-level timestamps (`false` = faster) |
| `QWEN_LLM_ENABLED` | Load Qwen for moderation/categorization (`false` = faster, BERT+NLI only) |
| `DEMUCS_MODEL` | Enhancement model |
| `MODERATION_AUTO_LEARN` | Enable/disable phrase auto-learning in moderation |
| `HIGGS_AUDIO_ENABLED` | Enable self-hosted Higgs rebuild path |
| `HIGGS_AUDIO_VOICE` | Higgs voice id |
| `OPENAI_API_KEY` | Optional OpenAI fallback/enrichment |

---

## Notes

- This codebase is now **track-first**. Any old `recording_id` pipeline contract is deprecated.
- Rebuild requires local `higgs_audio` module installed for self-hosted synthesis.
