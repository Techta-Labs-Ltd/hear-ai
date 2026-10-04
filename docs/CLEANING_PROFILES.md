# Magic Clean profiles: what to send, what comes back

Profile catalogue version `deepfilter-presets-v2`. All four profiles run the same
engine (DeepFilterNet3, full model, 48 kHz) on the `magic_clean_natural` worker
role; they differ only in how hard the denoiser is allowed to work and which DSP
runs before and after it. Every profile preserves mono/stereo layout, duration
(unless trimming is requested) and internal pauses, and every result needs the
creator's approval before it replaces the original. The machine-readable form of
this table is `GET /capabilities` on the Pod (`magic_clean.profiles`), which the UI
should read instead of hard-coding it.

## The four profiles

| Profile | Label | Use it for | Denoise cap | Post-filter | Before the denoiser | After the denoiser |
| --- | --- | --- | --- | --- | --- | --- |
| `studio_voice` | Studio Voice (recommended) | Spoken recordings meant for listeners: podcasts, interviews, narration | 60 dB (full model) | on | 70 Hz high-pass | +1.5 dB presence at 3 kHz, slow linked auto-level, 2:1 compression, mild de-essing |
| `outdoor_mobile` | Outdoor & Mobile | Phone and field recordings with wind, traffic, handling rumble | 60 dB (full model) | on | 100 Hz high-pass | slow linked auto-level, 3:1 compression, mild de-essing (no presence EQ) |
| `natural` | Natural | Keep the room's character; reduce noise without "studio" processing | 36 dB (about 1.6 % of the noise left in) | off | none | none |
| `clean_raw` | Clean & Raw | Archival or further editing elsewhere: gentle denoise, nothing else | 18 dB (about 12 % of the noise left in) | off | none | none; never boosts, never compresses |

"Denoise cap" (`attenuation_limit_db`) is the maximum attenuation the model may apply
to noise; lower values mix that much of the original back in for a more natural
sound. It is a ceiling, not a promised reduction. The post-filter is a DeepFilterNet
option that suppresses residual noise more aggressively; it is fixed per profile.

Auto-level targets −19 LUFS for mono and −16 LUFS for stereo with at most +6 dB of
gain; peak safety (−1.2 dBTP before encoding, verified at −1 dBTP after) always wins
over the target. Clean & Raw may *reduce* gain for peak safety but never raises it.

## Request options

```json
{
  "profile": "studio_voice",
  "attenuation_limit_db": 60,
  "auto_level": true,
  "remove_clicks": false,
  "trim_silence": false,
  "sound_cleanup": {"enabled": false},
  "reduce_stationary_noise": false
}
```

| Key | Type | Default | Rules |
| --- | --- | --- | --- |
| `profile` | enum | required | one of the four above; retired profiles (`sam_audio`, `echo_room`) are rejected |
| `attenuation_limit_db` | int | profile default | one of 12, 18, 24, 36, 60 |
| `auto_level` | bool | `studio_voice`/`outdoor_mobile`: true; others: false | `clean_raw` rejects `true` |
| `remove_clicks` | bool | false | conservative impulse repair before denoising (ffmpeg `adeclick`); `clean_raw` rejects `true` |
| `trim_silence` | bool | false | removes only near-silent leading/trailing runs ≥ 750 ms, keeps 250 ms handles, never touches internal pauses; `clean_raw` rejects `true`; cannot be combined with `sound_cleanup` |
| `sound_cleanup` | object | disabled | event detection and repair (coughs, thumps, handling, animals, clicks); see `docs/AUDIO_JOBS.md`; `clean_raw` rejects it; coughs need `remove_coughs: true` **and** `"cough"` in `targets` |
| `reduce_stationary_noise` | bool | false | extra background pass using the analyser; `clean_raw` rejects it |

Booleans must be JSON booleans; unknown keys fail validation (`unsupported_magic_clean_options`).
The old `speech` / `music` / `background` sliders and `cut_silence` from the Python
backend do not exist in this contract; the UI offers the profile plus the toggles.

## What comes back: identical shape for every profile

The outcome (`ExecutionOutcome`, delivered to the backend callback, never in the
transport response) has `status: "completed"`, exactly one artifact, and this `result`:

```json
{
  "profile": "studio_voice",
  "engine": "deepfilternet3",
  "requires_approval": true,
  "source_sha256": "<sha256 of the source object that was cleaned>",
  "delivery_audio": {
    "bucket_name": "hear-media",
    "object_key": "creators/x/audio/jobs/<job_id>/<attempt_id>/delivery_audio.mp3",
    "size_bytes": 1518336,
    "sha256": "<sha256 of the MP3>",
    "content_type": "audio/mpeg",
    "audio_url": "https://cdn.hear.media/creators/x/audio/jobs/<job_id>/<attempt_id>/delivery_audio.mp3",
    "duration_seconds": 94.890208
  },
  "report": {
    "duration_seconds": 94.890208,
    "channels": 1,
    "sample_rate": 48000,
    "gain_db": 3.990928,
    "target_lufs": -19.0,
    "delivery_measurement": {"integrated_lufs": -19.598878, "unavailable_reason": null, "true_peak_dbtp": -1.641017},
    "warnings": ["possible_wanted_content_loss", "wanted_content_requires_review"],
    "speech_preservation": {"status": "passed", "voiced_frames": 2293, "lost_voiced_frames": 0, "allowed_lost_frames": 22, "step_frames": 1536},
    "sound_cleanup": {},
    "background_cleanup": "not_requested"
  }
}
```

The artifact list contains the same object as `delivery_audio` (the backend
validates bucket, key prefix, size and SHA-256 before using it). The MP3 is
48 kHz, 128 kbps for mono and 192 kbps for stereo.

| `report` key | Meaning |
| --- | --- |
| `duration_seconds`, `channels`, `sample_rate` | Of the delivered file; duration equals the source unless `trim_silence` removed edges |
| `gain_db` | Linear gain applied in mastering (positive = louder). With `auto_level` off it is 0 or negative (peak safety only) |
| `target_lufs` | −19 (mono) / −16 (stereo) when `auto_level` is on, otherwise `null` |
| `delivery_measurement` | Measured on the delivered MP3: integrated loudness (LUFS, `null` with `unavailable_reason` `too_short` or `below_measurement_gate`) and true peak (dBTP, always ≤ −1) |
| `warnings` | Review hints, see the table below; `wanted_content_requires_review` is always present |
| `speech_preservation` | VAD comparison of model input vs output: `status` `passed` or `review_required`, counts of confidently voiced frames and how many went unvoiced (allowed: 1 %, minimum 2); `{"status": "analyser_not_provisioned"}` where the Sound Cleanup bundle is not deployed |
| `sound_cleanup` | Only when requested: `status` (`no_changes`, `reduced`, `partial`), `repaired_count`, `unresolved_count`, `preview_count`, `events[]` (kind, time range, method, outcome) |
| `background_cleanup` | `not_requested`, `applied`, or `rejected_speech_activity_loss` |
| `timeline` | Present only when trimming changed the file: `start_frame`, `end_frame`, `original_frames` on a 48 kHz grid (divide by 48 000 for seconds; shift transcript times by `start_frame` after approval) |

Warning codes:

| Code | Meaning | What the UI should do |
| --- | --- | --- |
| `wanted_content_requires_review` | Always: technical checks passed, audible quality is for a human to judge | Show the A/B review |
| `possible_wanted_content_loss` | Some loud blocks lost more than half their energy; usually noise that was loud, sometimes speech | Mention "check quiet passages" |
| `possible_speech_loss` | The VAD saw voiced frames go silent (within the allowed 1 %) | Highlight; suggest `natural` or a lower attenuation if it sounds wrong |
| `speech_activity_unavailable` | No VAD bundle on this worker; only the energy gate ran | Informational |
| `sound_cleanup_some_events_need_review` | Some detected events were left unchanged (overlapping speech) | Offer the event list |
| `loudness_target_limited_by_headroom_or_measurement_gate` | Auto-level could not reach the target (peaks or too quiet to measure) | Informational |
| `linear_attenuation_applied_for_peak_safety` | Gain was reduced to avoid clipping (profiles with auto-level off) | Informational |

## Failed outcomes

`status: "failed"`, no artifacts, `error_code` plus `result.message`:

| `error_code` | Typical cause | Retry? |
| --- | --- | --- |
| `invalid_audio` | Undecodable or non-finite audio; or the safety gates failed: `speech_activity_lost:N_of_M_voiced_frames`, `channel content disappeared`, `generated content on silent source` | Not with the same settings; a lower `attenuation_limit_db` or `natural` may pass |
| `resource_exhausted` | Source longer than 4 h, larger than 4 GB, or the host had no scratch disk | Shorter source |
| `source_mismatch` | Downloaded file's SHA-256 differs from `source.file_sha256` | Re-issue the job with the current source |
| `deadline_exceeded` | The envelope deadline passed before completion | New job |
| `invalid_request` | Options failed validation | Fix the request |
| `engine_unavailable`, `process_failed`, `storage_failed` | Worker-side fault | Transient; the backend never retries automatically, the user may |

## What the backend must do with a completed result

1. Validate the artifact (bucket, prefix, size, SHA-256 re-read from storage).
2. Store the candidate (`delivery_audio`, `report`) and set the job to awaiting
   approval; `preview_audio_url` for the UI is `delivery_audio.audio_url`.
3. On approval, copy the object into the track's own namespace, swap the media,
   delete the old media and the job folder, queue a new transcription; if the track
   was published it becomes `ready` and must be published again
   (`docs/GO_AI_DISPATCH_PLAN.md` §10.2).
4. On rejection or after 24 h, delete the candidate.
