# Sound Cleanup v1

## Release boundary

This change preserves the existing DeepFilterNet3 engine and profile defaults.
Sound Cleanup is optional and disabled unless explicitly requested. It runs on
float PCM after DeepFilterNet and before existing finishing/mastering. It never
runs MossFormer or SAM. The public profile IDs remain unchanged.

This is implemented worker functionality, not a declaration of universal studio
quality. Automatic event thresholds are not calibrated against a representative
held-out Hear corpus. Overlapping-speech separation produces explicitly requested
previews requiring listening approval, not automatically accepted repairs.

## Processing

Pinned Silero analyses original and denoised speech on each channel. PANNs
Cnn14 DecisionLevelMax supplies frame-wise event proposals on both versions.
Any channel's likely speech protects the interval on all channels. This avoids
silently mixing stereo or cancelling anti-phase speech during analysis.

The bounded planner merges intersecting automatic detections, prioritises manual
selections, and limits event count, duration and total editable coverage. Short
selected impulses use interpolation. Eligible nonspeech events use adjacent,
low-energy, stable room tone with crossfades, or bounded attenuation when no
suitable room-tone reference exists. Uncertain overlaps remain unchanged.

An optional AudioSep specialist estimates the named disturbance in a fixed
10-second contextual window at 32 kHz. A pinned, precomputed query avoids loading
its text encoder during jobs. Inference is per-channel; estimates are aligned
back to 48 kHz and applied only inside the selected interval, preserving the
original high-frequency content outside the estimate. This is not proof of
unchanged stereo perception or perfect speech preservation.

Overlap previews reject insignificant or oversized estimates, excessive peak
increases, lost high-confidence speech activity and detected speech leakage into
the removed estimate. VAD agreement does not certify words. The preview uses a
bounded 85% estimated-component subtraction and retains explicit review status.
No recursive or open-ended stronger-processing loop is used.

## Request options

Inside the existing versioned AttemptEnvelope options:

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

For a selected overlapping event, add `preview_overlaps: true` and one to four
`regions`, each containing `start_seconds`, `end_seconds` and `kind`. Set
`auto_detect: false` to work only on those selections. Supported kinds are
`handling`, `impact`, `animal`, `cough`, and `click`. Cough selections require
`remove_coughs: true`. A separate `confirmed_no_speech: true` is an explicit user
assertion that permits local nonspeech repair despite detector disagreement;
it is recorded in the report and must never be silently supplied by the UI.

Each region must be at most eight seconds. Selections cannot overlap or exceed
60 seconds in total. Repair coverage is also limited to 25% of the recording;
there are at most 128 event records. Clean & Raw rejects Sound Cleanup. Combining
Sound Cleanup with trimming or an old certified cleaner ticket is rejected.
Times refer to the unchanged recording timeline; report frames use the decoded
48 kHz grid, not compressed-file padding or metadata duration.

## Results and rollback

The nested `sound_cleanup` report contains version and recipe/model hashes,
regions, methods, score evidence, per-region outcomes and review counters.
`audio_changed` describes only the added repair, not the entire DeepFilterNet job.
`no_changes` does not mean no noise exists; no repair was accepted.
`partial` can include review previews or skipped/rejected regions. The normal job
can complete successfully while some events remain unresolved. The source is
never overwritten and the normal workflow still requires candidate approval.

The stage verifies finite samples, exact frame/channel counts, and unchanged PCM
outside accepted edit windows. Output publication is create-only and atomic.
Rejected candidates retain the baseline region. Cancellation signals native work
and waits for its safe exit before deleting the workspace.

## Provisioning and deployment

Install the optional `sound-cleanup-provisioning` dependency group in a separate
build environment. It is not added to the normal cleaner's runtime group.
`scripts/provision_sound_cleanup_models.py` verifies the author checkpoint and
Silero asset, exports fixed-window frame-wise inference, and creates a hashed
bundle. `scripts/provision_event_separator.py` verifies pinned AudioSep source
and checkpoint, precomputes the constrained event queries, and checks exported
inference against eager zero/noise examples. No download occurs during jobs.
Keep the original model/source notices with provisioned bundles.

Configure only the assets that have been provisioned and reviewed:

```text
HEAR_SOUND_CLEANUP_BUNDLE=/models/sound-cleanup-v1-runtime
HEAR_SOUND_CLEANUP_BUNDLE_SHA256=<sha256 of its manifest.json>
HEAR_SOUND_CLEANUP_SEPARATOR_BUNDLE=/models/sound-cleanup-specialist/runtime
HEAR_SOUND_CLEANUP_SEPARATOR_SHA256=<sha256 of its manifest.json>
```

The first bundle enables local repair. The optional second bundle enables
selected overlap previews. `/capabilities` reports these separately. Missing,
corrupt or disabled assets do not silently downgrade an explicit request. New
requests require available-engine mode, not the older certified Natural workflow.
Use immutable bundle paths and deploy source/model versions together. Model
loading and CUDA calls are bounded by window size; native GPU hangs still need
process/container supervision. Existing job retries restart an attempt; this
change does not introduce a durable per-region checkpoint/cache system.

## Verification performed

The three full user recordings (Tracks 8, 15 and 20) ran with real DeepFilterNet,
Silero and PANNs. No automatic event reached the configured selection threshold;
those runs correctly reported no added edits. They do not demonstrate event
removal. A separate controlled test mixed a CC0 dog recording into a known speech
reference. A selected overlap preview improved waveform error relative to that
reference by 15.85 dB in the five-second affected interval, with zero flagged
high-confidence speech losses or removed-speech windows. This is one synthetic
mixture, not an independent listening score or broad restoration benchmark.

The same selected-bark case also completed the full DeepFilterNet, Sound Cleanup,
mastering and FLAC/MP3 export path. Unit tests cover stereo layout, unchanged
outside regions, options, consent, review status, rollback, asset gating and
cancellation. Human listening, word accuracy, event-class calibration, fresh
real-world page/cough/door examples and new worst-case memory/concurrency testing
remain release gates before enabling automatic cleanup by default.

## Application integration blocker

The checked backend/frontend main branches still use the legacy `/process`
transport and slider-style Magic Clean fields. They cannot invoke the new typed
AttemptEnvelope options without the v11 control-plane migration. An existing
backend migration worktree contains unrelated unfinished changes; this feature
must not overwrite or silently merge those changes. Update the backend contract,
durable dispatch, capability proxy, preview and approval UI together. Until then,
this feature is usable through the local CLI or the versioned worker API, not the
current dashboard controls. The existing backend catalog routing problem is not
fixed by this audio-engine change.
