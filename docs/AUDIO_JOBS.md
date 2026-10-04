# Audio jobs

Send audio options inside the versioned `AttemptEnvelope`. The backend owns the
source revision, storage scope, dispatch, callbacks, and approval described in the
[README](../README.md#jobs-and-callbacks). Read `/capabilities` for supported roles,
profile labels/defaults, and optional engine availability.

## Cleaning

All four presets use `magic_clean_natural`, its `magic_clean.natural` queue, and
the pinned DeepFilterNet3 assets under `HEAR_MAGIC_CLEAN_MODEL_DIR`.
The presets preserve mono/stereo layout and are intended for spoken recordings.

| Profile | Processing |
| --- | --- |
| `natural` | 36 dB maximum denoising attenuation; no EQ/compression |
| `studio_voice` | Full-strength denoising with post-filter, 70 Hz high-pass, gentle presence EQ, linked slow auto-level, 2:1 compression, and mild de-essing |
| `outdoor_mobile` | Full-strength denoising with post-filter, 100 Hz high-pass, linked slow auto-level, 3:1 compression, and mild de-essing |
| `clean_raw` | 18 dB denoising; no EQ, compression, loudness boost, or trimming |

Studio Voice is the recommended spoken-recording preset. Attenuation is a maximum
setting, not a promised reduction. The retired SAM profile is rejected; reconcile
its queued separation jobs separately rather than remapping them to denoising.

Example options:

```json
{
  "profile": "studio_voice",
  "auto_level": true,
  "remove_clicks": true,
  "trim_silence": false
}
```

`attenuation_limit_db` accepts 12, 18, 24, 36, or 60, with a profile default when
omitted; 60 is the full model output and lower values mix that much of the original
noise back in for a more natural sound. The post-filter is fixed per profile.
Boolean controls require JSON booleans. Unknown keys and removed profiles fail.
Studio/Outdoor compression and EQ remain active when auto-level is off.
Clean & Raw rejects enabled auto-level, de-clicking, or trimming.

De-clicking uses conservative FFmpeg impulse interpolation before denoising and
is off by default. Trimming is also off by default: it removes only near-silent
leading/trailing runs of at least 750 ms, keeps 250 ms handles, and preserves
internal pauses. All-silent audio is not reduced to zero frames. The report's
original/retained frame range uses a 48 kHz grid; divide offsets by 48,000 when
updating transcript times after approval.

Processing decodes to 48 kHz float PCM, applies selected pre-processing, denoising,
optional event repair, finishing, and export validation. The only uploaded
artifact is the 128 kbps mono or 192 kbps stereo MP3; the review report travels
inline in the outcome's `result.report`, no lossless master or JSON manifest is
stored. The export is measured for finite samples, duration/channels,
content-loss risk, and true peaks.
A -1 dBTP ceiling is checked after MP3 encoding. Overshoot correction re-renders
from float audio rather than repeatedly transcoding an MP3.

Content checks are two-layered. An energy gate rejects outputs that lose more
than 90% of a channel's energy or add content to silence, and flags blocks within
12 dB of the loudest block that lost over half their energy as
`possible_wanted_content_loss` review hints. When the Sound Cleanup bundle is
provisioned, a Silero VAD comparison of the model input and output fails the job
with `speech_activity_lost` if more than 1% (minimum two) of confidently voiced
frames become unvoiced; the report's `speech_preservation` block records the
counts. Energy alone cannot distinguish quiet speech from noise, so deploy the
bundle wherever strong denoising is enabled.

Auto-level targets -19 LUFS mono or -16 LUFS stereo, with bounded gain and peak
safety taking priority. An unreachable target becomes a warning. Clean & Raw can
apply linear attenuation for clipping prevention but does not boost or compress.
The report records effective options, loudness/peaks, gain, layout, timing, trimming,
and content-risk warnings. Candidates require approval and never replace the
original automatically. Technical validation does not certify audible quality.

## Sound Cleanup

Optional Sound Cleanup runs after DeepFilterNet and before finishing/mastering.
It is disabled unless requested and keeps the selected profile's defaults.
Pinned Silero/PANNs analysis protects likely speech on every channel. The planner
prioritizes manual selections, merges detections, and bounds event coverage.
Eligible nonspeech regions use interpolation, adjacent stable room tone with
crossfades, or bounded attenuation. Uncertain overlaps remain unchanged.

Example options:

```json
{
  "profile": "studio_voice",
  "sound_cleanup": {
    "enabled": true,
    "auto_detect": true,
    "targets": ["handling", "impact", "animal"],
    "remove_coughs": false
  }
}
```

For overlap previews, add `preview_overlaps: true` and one to four `regions` with
`start_seconds`, `end_seconds`, and `kind`. Set `auto_detect: false` for selections
only. Kinds are `handling`, `impact`, `animal`, `cough`, and `click`; cough selections
require `remove_coughs: true`. `confirmed_no_speech: true` is an explicit user
assertion permitting nonspeech repair despite detector disagreement and must
never be supplied silently by the UI.

Each region is at most eight seconds. Selections cannot overlap or exceed 60
seconds in total. Coverage is limited to 25% of the recording and 128 event records.
Clean & Raw rejects Sound Cleanup; trimming cannot be combined with it. Times
refer to the unchanged source timeline, using decoded
48 kHz frames rather than compressed-file padding.

The optional AudioSep specialist estimates a named event per channel in bounded
10-second/32 kHz windows using precomputed queries. Estimates align back to 48 kHz
and affect only selected regions. Previews subtract the complete estimated event
and reject unsafe/insignificant estimates, excessive peaks, speech leakage,
speech-activity loss, and new tonal/DC artifacts. These checks do not certify
words or remove every source of hum. Overlaps remain explicit review candidates.

The nested `sound_cleanup` report contains recipe/model hashes, regions, methods,
scores, outcomes, and review counters. `audio_changed` describes the added repair;
`no_changes` means no repair was accepted, not that noise is absent. A job can
complete with unresolved regions. Rejected edits retain the baseline. Validation
checks frame/channel counts, finite samples, and unchanged PCM outside edits.
Publication is atomic/create-only; cancellation waits for native work to exit
before workspace deletion. Retries restart attempts rather than resuming a
per-region checkpoint.

Install the separate `sound-cleanup-provisioning` dependency group for asset export.
`scripts/provision_sound_cleanup_models.py` verifies the pinned analysis models and
creates a hashed bundle. `scripts/provision_event_separator.py` verifies pinned
AudioSep source/checkpoint, precomputes queries, and checks its exported inference.
`scripts/provision_release_sound_assets.py` assembles verified release metadata.
Preserve [model notices](SOUND_CLEANUP_MODEL_NOTICES.md) with exported bundles.

```text
HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
HEAR_SOUND_CLEANUP_BUNDLE_SHA256=<sha256 of its manifest.json>
HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
HEAR_SOUND_CLEANUP_SEPARATOR_SHA256=<sha256 of its manifest.json>
```

The analysis bundle enables local repair; the optional separator enables selected
overlap previews. `/capabilities` reports each independently. Explicit requests
fail if required assets are missing/corrupt/disabled. Use immutable reviewed asset
paths and deploy source/model versions together. No downloads occur during jobs.

## Reconstruction

Fish Speech S2 Pro generates narration edits using the same supervised workflow
for Pod and Serverless. It runs in a warm child process with bounded startup and
inference timeouts; cancellation terminates native inference. Restart an unhealthy
worker after a failed/hung child. The model runs in bfloat16 with a 4096-token
cache (upstream would otherwise allocate a 32k-token cache); measured on an A40
it holds 15.8 GB after load, peaks at 16.4 GB, and renders a four-second sentence
in about nine seconds after a 75 s cold start. One 48 GB card therefore keeps
Fish, the pipeline (about 10 GB) and four cleaner replicas (about 3 GB each) warm
together.

Example reconstruction-specific payload, inside a complete authenticated envelope:

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

Operations are `replace_segments`, `edit_transcript`, `preview`, `rebuild`, and
explicit `remove_segments`. Blank text does not imply deletion. Inserts use equal
start/end positions and require new text. Edits must have finite in-range times
and cannot overlap. The caller supplies text and authorized references, not
pre-generated replacement audio.

Same-speaker generation requires matching `original_text` over an uncut 1–20 second
source interval or an explicit `reference` with `start_seconds`, `end_seconds`,
and matching text. Rebuild needs an aligned reference when `same_speaker` is true.
Independent stereo speakers require `reference.channel` (0 or 1). References come
from the authorized source. Missing references fail rather than substituting a
generic voice. Text chunks do not add emotion tags or paraphrase the requested text.

Outcomes include identities, source revision, backend ownership,
`engine=fish_speech_s2_pro`, operation, and `requires_approval=true`.
`reconstructed_audio` (inline in the outcome, no manifest upload) contains
final/source frames, duration, delivery
artifacts, storage keys/URLs/hashes, and per-edit records. Those records retain
source times, generated durations, output positions, replacement MP3s, and reference
digests. Generated speech is not stretched into the old interval; the backend must
apply this timeline map when approving transcript updates.

Assembly is disk-backed at 48 kHz with blends inside generated regions. Untouched
source samples are copied before mastering. Final peak gain and MP3 encoding can
change samples globally. Edited stereo speech is dual-mono rather than spatial
reconstruction. Rejecting a candidate preserves the original.

The pinned model is the official `fishaudio/s2-pro` bf16 release at
`1de9996b6be38b745688de084d87a5633f714e4e`; the source is upstream
`fishaudio/fish-speech` at `214da3cd841bda85da2496b96cd3c4d7edb1337e`. Every
weight file is hashed in `hear/model_manifest.json`. Install the pinned source in
the reconstruction environment, then provision:

```bash
HF_HUB_OFFLINE=0 /opt/hear-ai-v11/venvs/reconstruction/bin/python \
  -m hear.tools.model_provisioning --role reconstruction --model-root /models \
  --acknowledge-license-review
```

The flag only permits the download; readiness keeps reporting the licence blocker
until approval is recorded. Set `FISH_SPEECH_HOME` to the pinned source checkout and
`FISH_SPEECH_MODEL_ROOT=/models`. Models must be outside the source checkout and off
network volumes. Quantized third-party builds are not supported; on smaller GPUs
run Fish on its own Serverless endpoint rather than quantizing it.

Serverless reconstruction images include model assets only when
`HEAR_FISH_LICENSE_APPROVED=true` is supplied during the build. Downloading or
quantizing a checkpoint does not grant commercial permission; production retains
the manifest licensing gate until approval is recorded.

Fish consumes `reconstruction.fish_tts`. Reconcile old reconstruction messages
rather than relabeling them. Configure one Fish job per process; measure resource
use before scaling separate workers under the shared Pod admission budget.

## Acceptance

Use full recordings and representative speech/event fixtures. Test channel/layout,
finite samples, timing, silence, quiet voices, long chunk joins, unchanged regions,
rollback, cancellation, and artifact validation. Listen at matched loudness;
extra volume alone does not demonstrate better denoising. Synthetic unit tests
exercise processing/contracts without certifying audible restoration, word
accuracy, or speaker similarity. Review event thresholds and worst-case resource
use before enabling automatic repair by default.

Application controls must use typed attempt options and worker capabilities,
persist previews/approval, and apply trimming/reconstruction timeline maps.
Legacy `/process` sliders do not express this contract. Validate a fresh live
backend-issued attempt and independent B2 read/hash/approval round-trip before
release. Local test results alone do not establish deployed behavior.
