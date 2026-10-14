# Audio jobs

Send audio options inside the versioned `AttemptEnvelope`. The backend owns the
source revision, storage scope, dispatch, callbacks, and approval described in the
[README](../README.md#jobs-and-callbacks). Read `/capabilities` for supported roles,
profile labels/defaults, and optional engine availability.

## Cleaning

The profile table, request rules, the result payload and its warning codes are
documented for backend and frontend work in [CLEANING_PROFILES.md](CLEANING_PROFILES.md).

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
    "seed": 1234,
    "reference": {
      "start_seconds": 40.2,
      "end_seconds": 51.7,
      "text": "The aligner's exact words for this clean stretch of the same speaker."
    },
    "takes": 3,
    "changes": [{
      "segment_start": 13.4,
      "segment_end": 14.1,
      "original_text": "Tuesday",
      "new_text": "Thursday",
      "context_before": "we will ship the update on",
      "context_after": "after the review."
    }]
  }
}
```

Operations are `replace_segments`, `edit_transcript`, `preview`, `rebuild`, and
explicit `remove_segments`. Blank text does not imply deletion. Inserts use equal
start/end positions and require new text. Edits must have finite in-range times
and cannot overlap. The caller supplies text and authorized references, not
pre-generated replacement audio.

### Voice reference

Fish clones the voice from a reference prompt: 5–20 seconds of continuous speech
from the authorized source plus the exact words spoken in it. Send it as an
explicit `reference` with `start_seconds`, `end_seconds`, and `text`; one
reference serves every edit in the job. Only an edit whose own interval is 5–20
seconds with a matching `original_text` may omit it and clone from the words it
replaces; shorter same-speaker edits fail with
`same_speaker_requires_explicit_reference_or_5_to_20_second_aligned_interval`.
Rebuild always needs an explicit reference when `same_speaker` is true.

Choose the reference, not the edit, for voice quality: 8–15 seconds of
confidently transcribed, uninterrupted speech by the same speaker, near the edit,
with the aligner's words verbatim as `text`. The worker rejects references that
are mostly silent (`voice_reference_mostly_silent`), clipped
(`voice_reference_clipped`), or whose text length does not fit the interval
(`voice_reference_text_length_mismatch`), and reports `voice_reference`
measurements with `reference_shorter_than_recommended` or
`reference_has_long_pauses` warnings on each segment. Independent stereo speakers
require `reference.channel` (0 or 1). Missing references fail rather than
substituting a generic voice. Text chunks do not add emotion tags or paraphrase
the requested text.

### Sentence context and word-level splicing

Send the neighbouring words of the same sentence as `context_before` and
`context_after` on each change; the interval `[segment_start, segment_end]`
covers only the replaced words, snapped to the aligner's word boundaries. The
worker generates the whole sentence so the new words carry natural prosody,
force-aligns the take with wav2vec2 CTC, and cuts only the new words out at the
quietest point of the gaps to their neighbours. The source audio for the context
words is never replaced, so everything but the changed words stays the user's own
recording. Each segment's `splice` reports the mode (`words` or
`whole_segment`), the cut points inside the take, and the timed new words. When
the sentence cannot be aligned (digits or symbols in the text, or a take longer
than 60 s) the worker regenerates the new words alone with the chosen seed and
replaces the whole interval; a worker without the verifier does the same from
the start and reports `verifier_unavailable`.

### Verification and takes

Every take is checked before it can be delivered. wav2vec2-base-960h transcribes
the take; the new words must all appear in order (90 % for edits over five
words, fuzzy to CTC spelling), otherwise the take is rejected. WavLM-base-plus-sv
embeds the take and the voice reference; their cosine similarity must reach
`HEAR_RECONSTRUCTION_MIN_VOICE_SIMILARITY` (default 0.75). The worker renders up
to `HEAR_RECONSTRUCTION_TAKES` (default 3, `takes` in the options overrides it,
1–4) candidate takes per edit with consecutive seeds, stops early once a take is
verified and at least 0.1 above the floor, and keeps the most similar verified
take. If no take says the new words the job fails with
`generated_words_unverified`; if none reaches the floor it fails with
`voice_match_below_threshold`. Both mean "regenerate with another seed or a
better reference", not "approve anyway". Segments report `takes`,
`selected_take`, `voice_similarity`, and `words_verified`; the outcome's
`word_accuracy_verified` is true only when every generated segment passed. Words
the CTC model cannot spell (digits, symbols) are reported as unverifiable rather
than failing; spell them out in `new_text` to have them checked.

Both verification models are manifest entries for the reconstruction role and
are provisioned with it; `HEAR_RECONSTRUCTION_VERIFICATION_ENABLED=false`
disables them, in which case takes drop to one and nothing is verified.
`facebook/wav2vec2-base-960h` is Apache-2.0. `microsoft/wavlm-base-plus-sv` is
released under CC BY-SA 3.0 (its model card points to the UniSpeech licence):
commercial use is permitted, attribution to Microsoft's WavLM is required in
product notices, and any fine-tuned derivative must be shared under the same
licence. The worker only runs inference on the unmodified weights.

The verifier loads at worker start (`warmup`) in float16 on CUDA, about 0.2 GB
of weights per model; a typical ten-second take verifies in well under a second
with a transient peak of a few hundred megabytes, and the hard ceiling is the
60 s alignment limit at roughly 1 GB. `SpeechVerifier.last_metrics` reports the
measured seconds and peak CUDA bytes of the last call. The reference embedding
is cached per reference, so the takes and edits of one job compute it once.

### Takes, previews and confirmation

Every edit is generated with one seed, so all of its text chunks are the same
take of the voice. Omit `seed` for a job-derived seed, or send `seed` (0 to
2^31-1): edit *i* uses `seed + 16·i`, its candidate takes add 0–3, and the
chosen take's seed is recorded on each segment. A new seed asks for another take;
the same seed reproduces the previous one.

`preview` generates, matches and uploads only the segment MP3s. Its outcome has
`delivery: null`, `preview: true`, and the same `segments` timeline a full render
would produce, so the UI can play source, segment, source in place. Confirm by
resubmitting the identical `changes`, `reference`, and `seed` as
`replace_segments` or `edit_transcript`; the full track is assembled once.

### Matching the insert to the recording

Generated speech is matched to the two seconds of source on either side of the
edit before assembly: its voiced level is brought to the surrounding speaking
level (bounded to ±12 dB), it is low-passed to the source bandwidth when the
source rolls off below 16 kHz, and the quietest half second of surrounding room
tone is looped under it when the recording has an audible noise floor. Each
segment's `acoustic_match` reports the measurements. Regions blend over 20 ms at
each edge inside the generated interval; `crossfade_frames` records the length.

Outcomes include identities, source revision, backend ownership,
`engine=fish_speech_s2_pro`, operation, and `requires_approval=true`.
`reconstructed_audio` (inline in the outcome, no manifest upload) contains
final/source frames, duration, delivery
artifacts, storage keys/URLs/hashes, and per-edit records. Those records retain
source times, generated durations, output positions, replacement MP3s, reference
digests and measurements, seeds, chunk counts, and acoustic matching. Generated speech is not stretched into the old interval; the backend must
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
