# Fish Speech TTS editing: Pod and Serverless

## Implemented scope

Reconstruction means generating edited narration with Fish Speech S2 Pro. It is
not a requirement for the caller to supply already-generated replacement audio.
The former available-engine FFmpeg factory is no longer registered. Magic Clean
keeps its existing DeepFilterNet engine and defaults; its engine-mode switch
cannot disable Fish model/readiness requirements.

One shared FishReconstructionWorkflow runs under the Pod consumer and the
Serverless handler. The provider selects execution transport, not a different
reconstruction implementation. Fish runs in a supervised warm child process,
with bounded startup/inference timeouts and process termination on cancellation.
A failed/hung child makes the worker unhealthy; it must be restarted before reuse.
The pinned upstream source does not accept NF4 configuration; `none` is the
honest default. Unsupported requested quantisation fails rather than being ignored.

## Request contract

The backend creates the usual AttemptEnvelope with backend_id, job/run/attempt
IDs, authorised source revision/hash, deadline, attempt grant and scoped storage.
The application does not receive Pod credentials or Backblaze application keys.
For example, the reconstruction-specific portion is:

```json
{
  "job_type": "reconstruction",
  "operation": "edit_transcript",
  "options": {
    "same_speaker": true,
    "language": "en-GB",
    "changes": [{
      "segment_start": 12.0,
      "segment_end": 16.0,
      "original_text": "The exact words spoken in this source interval.",
      "new_text": "The edited words to generate."
    }]
  }
}
```

This is a partial example, not a complete authenticated request. Pod transport is
POST /v1/attempts/stream. A RunPod Serverless job wraps the same envelope in its
`input` field. Serverless posts canonical events and the outcome to the owning
backend before yielding completion; returning a RunPod stream alone is not a
backend state update. Transient callback failures have bounded retries.

Supported operations are replace_segments, edit_transcript, preview, rebuild,
and explicit remove_segments. Blank text is not an implicit deletion. Inserts
have equal start/end positions and require new text. Times must be finite and
in range; edits cannot intersect. All selections are checked before synthesis.

Same-speaker generation requires matching original_text over an uncut 1–20 second
source interval or an explicit `reference` with start_seconds, end_seconds and
matching text. Rebuild requires an explicit aligned reference when same_speaker
is true. Independent stereo speakers require reference.channel (0 or 1).
References come from the authorised source recording, not another user's global
voice cache. Missing reference data is an error, not silent generic-voice fallback.

Text is split into bounded chunks without adding emotion tags, rewriting or
paraphrasing. This does not prove Fish will pronounce every word correctly; no
ASR/manually checked word-accuracy certification is claimed by the worker.

## Output data

The canonical outcome contains job_id, run_id, attempt_id, track_id, source_revision,
backend ownership, engine=fish_speech_s2_pro, operation and requires_approval=true.
The reconstructed_audio block contains the final duration, output/source frames,
full master/delivery artefacts, bucket/key/URL/hash and per-edit records.

Per-edit records retain source start/end timestamps and provide generated duration,
output start/end timestamps, cumulative timeline position, replacement MP3 and
reference digest. Fish duration is natural, not forcibly stretched into the old
interval. The backend must apply this timeline map to transcript positions when
accepting the candidate. Rejecting it leaves the original recording unchanged.

Assembly is disk-backed at 48 kHz. Blends lie inside generated regions; they do
not silently consume neighbouring source frames. Untouched source samples are
copied before mastering. Final peak-safety gain and lossy MP3 encoding can change
samples globally; do not advertise final MP3 byte identity. Stereo generated
speech is explicitly dual-mono inside edited regions, not spatial reconstruction.

## Correct Backblaze ownership

Production: backend-a, OldAlexa, s3.eu-central-003.backblazeb2.com, cdn.hear.media.
Development: backend-a-dev, hear-dev-uploads, s3.us-east-005.backblazeb2.com,
media.hear.surf. These were inspected from deployed configuration, not guessed.

Organisation candidates belong under:
`localtns/{organisation_slug}/audio/jobs/{job_id}/{attempt_id}/`
Individual candidates belong under:
`creators/{creator_slug}/audio/jobs/{job_id}/{attempt_id}/`

The worker no longer appends jobs/job_id twice when given an already job-scoped
prefix. It rejects expired grants, traversal and incorrect local digests. Upload
HEAD checks verify size and supplied checksum metadata, not an independently
re-downloaded object. Real B2 permission/write/read validation remains unverified
in this audit. A software expiry field does not itself expire a static cloud key.

Configure HEAR_BACKEND_REGISTRY_JSON for multi-backend operation. Each entry has
a policy, a deployment-owned callback_base_url and a DISTINCT SHA-256 digest of
its backend-specific ingress bearer token. A production token cannot authorise a
development identity, and vice versa. Policies bind backend ID, allowed callback
origins, source hosts/hash, bucket, storage endpoint, CDN and exact job/attempt
prefix. Workspaces are partitioned by backend. Never publish the actual tokens
or cloud credentials in logs or API capabilities.

The backend must still verify job ownership and bind its attempt grant to the
source revision, request options and authorised storage. Worker policy validation
is not a replacement for backend authorisation or cloud-level key restrictions.

Pipeline catalogue remains one backend per pipeline worker. Set
HEAR_PIPELINE_CATALOG_BACKEND_ID to its actual catalogue owner. Other environments'
pipeline requests fail instead of being classified with the wrong environment's
catalogue. A separate owned pipeline deployment or per-job catalogue support is
required before mixing both pipeline taxonomies. This restriction does not make
Fish/DeepFilter models environment-specific.

## Deployment and measured limits

Fish uses the new reconstruction.fish_tts queue, separate from the old
reconstruction queue. Old messages are not silently relabelled or consumed by a
new semantic implementation. Drain/reconcile them explicitly during migration.

Configure one Fish job per worker initially; Pod and Serverless reject a higher
in-process reconstruction setting. This is a conservative admission policy, not
a measured maximum. The earlier four-job FFmpeg benchmark was NOT Fish TTS.
After real Fish measurements, scale separate worker instances under a shared Pod
resource budget; do not set every lane to its individual measured maximum at once.

The source installer now installs the pinned Fish package in the reconstruction
venv, and the Serverless image reinstalls it after the final dependency sync.
No model download happens during a job. Provision and verify checkpoints before
startup and retain the existing model licensing approval gate.

## Actual readiness boundary

At this audit, the Pod has the Fish source/package but NO S2 Pro checkpoint in
/models/fish-speech/s2-pro. The manifest still marks Fish permission_required;
no commercial approval was inferred or bypassed. Therefore real Fish audio
quality, cold-start/GPU memory and TTS concurrency were not measured.

Tests use an explicitly synthetic speech double to exercise real decoding,
reference handling, MP3/FLAC mastering, timeline edits, API validation and backend
callbacks. Those are integration tests, not evidence of Fish voice quality.

The deployed backend still calls legacy /process and lacks the v11 runtime
catalogue/attempt-control integration. This worker change does not complete that
separate backend migration. No live backend, live Pod code, RabbitMQ configuration
or concurrency settings were changed. End-to-end release still requires the
compatible backend dispatcher/claim/heartbeat/event/outcome routes, approved Fish
model provisioning, a real TTS canary, actual B2 round-trip and approval tests.
