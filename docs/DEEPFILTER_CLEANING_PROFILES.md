# Hear Magic Clean: DeepFilterNet profiles

## Release scope

This release removes SAM-Audio from executable code, worker roles, queue routing,
model provisioning, dependency groups and container targets. It retains the pinned
DeepFilterNet3 checkpoint and its bounded, contextual long-recording implementation.
The presets are intended for spoken recordings, not music restoration.
It does not add Resemble Enhance or claim to remove room reverberation, separate
speakers, eliminate all wind distortion, or reconstruct missing speech perfectly.

All four public profiles use the existing `magic_clean_natural` role and
`magic_clean.natural` queue route. Do not create four GPU workers just to offer four
presets. Pod and Serverless both use the same available-engine workflow.

## Profiles and defaults

| API value | User label | Default processing |
| --- | --- | --- |
| `natural` | Natural | DeepFilterNet3, 24 dB maximum attenuation; no EQ or compression. Existing API default behaviour stays denoise-only. |
| `studio_voice` | Studio Voice | 18 dB denoising, 70 Hz high-pass, gentle presence EQ, linked-channel slow auto-level, 2:1 compression and mild de-essing. Recommended for spoken recordings. |
| `outdoor_mobile` | Outdoor & Mobile | 24 dB denoising, 100 Hz high-pass, linked-channel slow auto-level, 3:1 compression and mild de-essing. |
| `clean_raw` | Clean & Raw | 12 dB denoising, no EQ, no compression, no loudness boost and no trimming. |

Attenuation is a maximum reduction setting, not a guarantee of that much noise
removal. Studio and Outdoor use different processing, not different model names.
All profiles preserve mono/stereo layout. Stereo channels are not mixed to mono.

## Options

```json
{
  "profile": "studio_voice",
  "auto_level": true,
  "remove_clicks": true,
  "trim_silence": false
}
```

`attenuation_limit_db` may explicitly select 12, 18 or 24. The profile supplies the
default when it is omitted. Boolean options require real JSON booleans; strings
such as `"false"`, numeric values, unknown keys and removed profile IDs are rejected.

`auto_level` applies bounded slow gain followed by measured delivery mastering.
It is not speaker diarisation and cannot promise that overlapping speakers become
equally loud. Profile compression/EQ remain active when Studio/Outdoor auto-level
is disabled; choose Clean & Raw for denoise-only output.

`remove_clicks` uses conservative FFmpeg impulse interpolation before denoising.
It is off by default and is not a guarantee to remove all mouth sounds or plosives.

`trim_silence` is off by default. It trims only near-silent beginning/end runs of
at least 750 ms, keeps 250 ms handles, and never removes internal pauses. It uses
low energy thresholds, not speech recognition. All-silent audio is not shortened
to zero frames. The validation report includes the original and retained frame
range at the normalized 48 kHz processing rate. Divide frame offsets by 48,000
when updating transcript times before accepting a trimmed candidate.

Clean & Raw rejects enabled auto-level, de-clicking or trimming rather than
silently changing the meaning of the profile.

## Mastering and output integrity

The pipeline decodes to 48 kHz float PCM, applies selected pre-processing, runs the
pinned denoiser, validates duration/channels/finite samples and content-loss risk,
applies selected finishing, then masters and validates the actual exports.

Outputs are a 24-bit FLAC master, a 128 kbps mono or 192 kbps stereo MP3, and a JSON
report. Master and encoded MP3 are both measured; a true-peak ceiling of -1 dBTP is
checked after encoding. Codec overshoot correction re-renders from float audio,
never repeatedly transcodes an MP3.

When auto-level is enabled, Hear targets -19 LUFS mono or -16 LUFS stereo. These
are product delivery targets, not universal broadcasting standards. Final gain
is bounded and yields to peak safety. A target that cannot be reached is reported
as a warning, not hidden behind a success label. Clean & Raw does not boost or
compress; linear attenuation can still be applied to prevent clipping, and the
report explicitly records it.

The report contains profile/version, effective options, input/output loudness and
true peak, applied final gain, channel/rate/duration, trim range and content-risk
warnings. Technical validation does not certify audible studio quality. Every
candidate still returns `requires_approval: true`; the source is never overwritten.

## API and UI integration

Read `/capabilities` and display `magic_clean.profiles`. Use each profile's label,
description, defaults and availability. Recommended profile is `studio_voice`.
Do not show a selectable Echo/Room or SAM profile. Pass the chosen options inside
the existing authenticated `AttemptEnvelope`; the transport and outcome artifact
contract are unchanged.

The AI runtime change does not modify `hear-backend` or `hear-frontend`. Their
profile allowlists, controls and mapping must use these IDs before the presets
can be selected by platform users. Do not silently map queued SAM separation
jobs to denoising: the operations mean different things.

## Deployment migration

Use `HEAR_OPTIONAL_ENGINE_MODE=available`, the existing `magic_clean_natural` role
and a Natural Pod or Serverless image. Remove `magic_clean_sam_audio` from
`HEAR_POD_STACK_ROLES` and retire its consumer after dealing with queued old jobs.
The shared Pod stack and gateway ignore a stale SAM role with a warning so it
does not stop unrelated workers; they do not remap SAM requests or add a missing
Natural worker. Remove the old setting explicitly.
Old queues and model files are NOT deleted automatically. Retain or archive them
according to your rollback policy; this code change does not erase user audio.

```bash
python scripts/setup_runtime.py --role magic_clean_natural --provider pod
/opt/hear-ai-v11/venvs/magic_clean_natural/bin/python \
  scripts/provision_magic_clean_models.py --model-root /models --engine deepfilter
```

The existing certified Natural workflow is still Natural-only. Old certificates
containing a SAM section must be regenerated as Natural-only documents with the
corresponding pinned digest. A certificate for the previous chain does not certify
the new Studio/Outdoor DSP. New presets require available mode; certificate mode
must not advertise them as available.

Before restarting production, verify pinned model hashes, runtime dependency
patches, FFmpeg, worker readiness and `/capabilities`. Then submit one complete
recording for each profile and inspect/listen to the returned candidate and report.
No production process is restarted by the source migration itself.

## Local full-file comparison

`scripts/run_local_magic_clean.py` runs the same cleaner and writes its master,
delivery MP3 and report without needing backend or storage credentials. It fails
rather than overwriting existing output files. Use the role's Python environment.

```bash
python -m scripts.run_local_magic_clean /path/to/recording.mp3 \
  --profile studio_voice --model-root /models --device cuda:0 \
  --output-dir /path/to/comparison/studio --remove-clicks
```

Use a distinct output directory per profile. Keep the original and compare at
matched playback loudness; increased volume alone is not denoising improvement.

## Tests and acceptance

`tests/test_cleaning_profile_workflow.py` verifies all four outcome routes,
source-digest checks before inference, rejection of unvalidated output and cleanup.
`tests/test_cleaning_profiles.py` covers strict options, routing, availability,
actual FFmpeg filters, silence trimming, finite samples, deadlines, export frames,
bit depth, peaks and raw dynamics. Its pipeline stub deliberately does NOT prove
DeepFilterNet sound quality. Real-checkpoint tests/CLI runs and listening on full
representative user files are separate acceptance steps. Test quiet voices,
sibilants, mobile wind, noise-only intervals, stereo, silence and long chunk joins.

References: official DeepFilterNet project, official FFmpeg filter documentation,
and official Resemble Enhance repository. No additional model is bundled here.
