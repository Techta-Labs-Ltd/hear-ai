# Magic Clean professional-grade remediation plan

**Status:** remediation implemented in the current worktree through containment and the
safety-critical core of Phase 1; **not release-ready or professional-grade** until the
remaining corpus, long-duration GPU soak, PostgreSQL race, storage-integration, and listening
gates pass

**Date:** 2026-09-03

**Primary goal:** stop audio loss on repeated Magic Clean runs and raise delivered quality to
a professional voice-isolation standard without breaking the existing backend wire contract

## 1. Decision and expected outcome

Magic Clean must become a non-destructive transformation of one canonical source revision,
not a filter repeatedly applied to its own lossy output. Here, canonical means the last input
revision not produced by Magic Clean; it may already contain an approved reconstruction or
other user edit and is not necessarily the track's first upload. The existing HTTP/gRPC
request, job type, controls, result payload, storage result, and subscription workflow are a
compatibility requirement.

The replacement will:

- resolve every known previous Magic Clean derivative to the same canonical source, or fail
  safely when that source cannot be recovered;
- preserve the complete timeline when `cut_silence=false`;
- protect quiet words, consonants, breaths, multiple speakers, and speech at chunk edges;
- use the least aggressive enhancement that provides a measurable improvement;
- process neural chunks with context, stitch once, and master once;
- validate the decoded delivery artifact before it is published;
- return real measurements rather than target values or fabricated defaults;
- preserve existing `speech`, `music`, `background`, and `cut_silence` inputs;
- keep `JobResult.magic_clean.enhanced_audio` as the authoritative backend result.

“ElevenLabs-level” is a listening and measurement target, not a claim that can be established
from architecture alone. Parity will only be claimed after a blinded, matched-source
non-inferiority test against legally obtained reference outputs.

### 1.1 Verified implementation checkpoint

This document began as an audit. The current worktree now contains an implementation, but the
distinction between automated containment and perceptual release evidence is intentional.

Implemented locally, with focused coverage for the named safety properties:

- tenant/track-scoped exact-URL and byte/PCM-hash lineage, retry content pins, cycle/depth/
  ambiguity rejection, cross-scope fail-closed behavior, and five-derivative root resolution;
- validated-bitstream reuse for identical canonical hashes, controls, and engine revision,
  with a new job-scoped key and no second model or MP3 encode;
- lossless float ingest with mono/stereo and source-rate preservation through enhancement;
- one context/core disk-backed engine for every duration, RF64-capable float intermediates,
  one global silence decision pass, one global bounded-gain/limited encode pass, and valid MP3
  delivery-rate selection;
- disabled default spectral suppression and fixed tone shaping, corrected compressor state,
  strict MossFormer/Demucs length and finite-output contracts (including exact Demucs
  model-rate length and source-rate-derived round-trip bounds), per-channel speech-collapse
  dry protection, mixture-safe stem dry protection, deterministic zero-shift Demucs inference,
  PyTorch 2.6+ checkpoint allowlisting, linked multichannel limiting, and compensated FFmpeg
  limiter latency;
- source-plus-enhanced activity union using blockwise PCM scans and compact per-frame state,
  conservative quiet/sparse-activity protection, 20 ms non-speech joins, and stable synthetic
  repeated-silence decisions;
- blockwise decode validation of the final MP3, real decoded LUFS/true peak, a documented but
  uncalibrated energy-contrast SNR estimate with an unavailable sentinel, fixed duration
  allowance, short-clip loudness handling, silent-input hallucination and silence-edit activity
  gates, source-relative fragmented-attenuation detection, metric-qualified per-channel
  peak/RMS erasure gates, coarse retained pre-master duration/layout/energy validation for
  intentional component removal, codec-frame/resampling allowance after silence editing,
  clipping/finiteness gates, a bounded codec true-peak corrective re-encode, and exact remote
  byte checksum read-back before completion;
- actor-side source re-hashing before inference, conditional cancel/retry/fail/complete state
  transitions, expected-key cleanup on malformed/failed remote results and post-upload stage
  failures, durable pre-upload cleanup intent plus tombstones with a cancellation-refreshed
  writer grace period and row locking, retained active orchestrator tasks, best-effort Ray
  cancellation plus cooperative cancellation between neural windows, contained download/
  conversion/background/remote writers, periodic reconciliation, and a 24-hour minimum
  storage-credential lifetime at admission and runtime with observable queued parking plus
  authenticated, monotonic credential refresh;
- local startup presence/structure checks for model manifests and referenced Demucs weights,
  plus FFmpeg capability, FFprobe presence, chunk/overlap, bitrate, and engine-revision checks;
  the cached MossFormer and checksum-verified Demucs artifacts are now copied into the intended
  persistent `/workspace/models` layout and an offline dual-model CUDA smoke test executes them.

Verification on 2026-09-03:

```bash
uv run pytest -q
# 457 passed, 53 warnings

uv run ruff check <changed Magic Clean, orchestration, storage, config, and test files>
# All checks passed

uv run mypy <changed Magic Clean implementation files>
# Success: no issues found in the scoped source files

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python main.py --validate-only
# exit 0
```

`git diff --check` also passes. The unrestricted repository-wide MyPy and Ruff runs still
report failures outside these scoped commands (`ruff check .` currently reports 743 findings),
so neither scoped result may be represented as a repository-wide static-check pass.

The latest offline real-model smoke used a local speech fixture plus a deterministic hum, a
96 kHz stereo source, the production `100/10/10` Demucs route, MossFormer, four context windows,
the disk-backed MP3 path, and decoded-delivery validation on CUDA. It preserved all 712,203
pre-encode samples and returned `-16.43 LUFS`, `-2.30 dBTP`, no clipping, and a 7.418792-second
decoded MP3 for the 7.418781-second source. Model loading took 8.038 seconds and processing plus
validation took 8.475 seconds on the available A40. This proves local high-rate/stereo execution
and hard integrity, not real-corpus transparency or perceptual quality.

Still required before release:

- long-duration, repeated-run, and multi-GPU-contention soak tests with the provisioned models;
- repository-pinned model checksum manifests rather than an operator-verified cache copy;
- a provisioned/checksummed production VAD and the full edit-decision/word-retention gate;
- calibrated source-aligned content identity/protected-speech, zero-run, click, polarity,
  phase/image, spectral-change, and attenuation gates beyond the hard integrity checks now
  present; the retained-reference gate does not yet reject equal-energy unrelated content;
- content-aware candidate selection and clean-input transparency proof for the production
  `100/10/10` route; its safety is improved but not yet established by a real-audio corpus;
- PostgreSQL two-session race tests and live object-store delete/checksum/reconciliation tests;
- durable service-owned cleanup credentials; admission/dispatch/runtime lifetime checks reduce
  exposure, but job-scoped credentials still cannot guarantee deletion after a longer outage;
- production disk-backed repeat-stability, mutable-retry pinning, actor hash-mismatch,
  tombstone-creation, and completion-uncertainty integration tests beyond the focused guards;
- disk-capacity/cleanup tests at the supported duration limit; RF64 removes the 4 GiB RIFF
  ceiling, but processing and validation can temporarily retain roughly three float PCM copies;
- bitrate bake-off, versioned real-audio corpus, ASR/perceptual metrics, blinded listening,
  canary, rollback drill, and the Phase 2/3 evidence below.

## 2. Non-negotiable compatibility requirements

No protobuf or request/response shape change is intended. The backend must nevertheless honor
the credential-refresh behavior documented below, and release is blocked until existing
consumers verify these behavioral changes. In particular:

| Contract surface | Required behavior |
|---|---|
| Submission | Keep `POST /process` and gRPC `SubmitJob`, `job_type=magic_clean`, and current authentication/storage context. |
| Controls | Keep `speech`, `music`, `background` as optional `0..100` integers and `cut_silence` as an optional boolean. |
| Defaults | Continue accepting omitted controls. The server may make the internal processing adaptive, but no new field is required. |
| Idempotency | An identical submission with the same `job_id` returns the existing `run_id` and current status; a queued Magic Clean job may also accept a credential-only refresh with a strictly later expiry and unchanged semantic destination/request. `Subscribe`/`GetResult` remain authoritative for the terminal result. A deliberate new run still uses a new `job_id`. |
| Result | Preserve every `MagicCleanPayload` field, including `transcription`, `moderation`, `enhanced`, all nested `enhanced_audio`/`quality` fields, and `stage_times`. |
| Artifact | Keep an MP3 at the returned `bucket_name`, `b2_key`, and `audio_url`; keep the current `enhanced/<job_id>.mp3` key shape. |
| Protobuf | No breaking field change or regenerated client is required for this remediation. |
| Deployment | Keep Magic Clean inside the existing Ray Serve deployment and orchestrator. No sidecar or third-party runtime call is introduced. |

Durable lineage, source hashes, engine revision, and quality diagnostics must live in
PostgreSQL-backed job metadata without changing the public request. Replica-local state is
only for disposable caches; cached stems/masters must be tenant-scoped, encrypted where
stored, bounded by quota and TTL, and covered by deletion policy.

## 3. Audit-baseline production path

The path below describes the 2026-08-29 audit baseline and explains the defect evidence. It is
superseded by the current worktree implementation described in §1.1; it must not be read as the
new deployment path.

```text
submitted audio URL
  -> download and force mono PCM16 WAV
  -> Demucs separation (production defaults always provide stem levels)
  -> MossFormer2 speech enhancement
  -> custom spectral subtraction
  -> fixed speech EQ and de-esser
  -> speech/music/residual mix
  -> per-window compression
  -> optional per-window silence stripping
  -> per-window or global loudness/limiting, depending on file duration
  -> 96 kbps MP3
  -> upload and return enhanced_audio
```

If the backend replaces the track URL with `enhanced_audio.audio_url` and the user starts a
new job, that derived MP3 enters the complete path again. Same-`job_id` replay does not rerun.
Automatic retry reuses the persisted URL, but it redownloads unpinned bytes and is therefore
request-idempotent rather than content-idempotent when a URL is mutable. The usual damaging
case is a deliberate new job ID using the previous enhanced URL.

## 4. Confirmed audit-baseline defects

The evidence and line numbers in this section are historical audit references. Current
disposition is summarized below; “partial” means containment exists but a Definition-of-Done
release gate remains open.

| ID | Current worktree disposition |
|---|---|
| MC-001 | Implemented and locally regression-tested, including identical-control bitstream reuse and cross-scope failure. Live PostgreSQL/storage integration remains. |
| MC-002 | Implemented and regression-tested. |
| MC-003 | Contained: default suppression is disabled; strength-zero identity and arbitrary-tail safety are tested. It is not re-enabled for production. |
| MC-004 | Safety core implemented globally with source/output activity union and conservative joins; production VAD and corpus word-retention proof remain. |
| MC-005 | Implemented with strict typed adapter failures, exact model-rate output validation, and sample-rate-derived resampler-rounding correction. |
| MC-006 | Hard pre-upload/final-MP3 integrity gates implemented; advanced aligned speech/perceptual gates remain. |
| MC-007–009 | Partial: controls remain compatible and speech-collapse protection was added, but content-aware candidate selection and real-corpus default-route proof remain. |
| MC-010 | Implemented: one engine is used for all durations. |
| MC-011 | Partial: spectral suppression and fixed EQ/de-essing are off by default; model-path transparency still needs corpus selection gates. |
| MC-012 | Implemented for supported mono/stereo input, including per-channel collapse protection and delivered-channel energy validation. |
| MC-013 | Partial: output has a dedicated setting and valid delivery-rate policy; the 96/128/192 kbps bake-off remains. |
| MC-014 | Partial: loaded production model-output violations fail closed and startup fails when model loading fails. A directly invoked unloaded MossFormer adapter still returns immutable dry input, so that state must never count as a loaded production service. |
| MC-015 | Partial: LUFS, true peak, clipping, and duration come from the exact decoded delivery artifact, and the unavailable SNR sentinel is excluded from its score contribution. The energy-contrast SNR and aggregate quality score remain uncalibrated. |
| MC-016 | Real subtimings, active-task/Ray/cooperative cancellation, download/conversion writer containment, admission/dispatch/runtime credential-lifetime handling, observable queued monotonic refresh, pre-upload cleanup intent, cancellation-refreshed writer grace, retry-aware tombstone ownership, and status-aware post-upload cleanup are locally tested; streamed progress remains coarse and live database/storage races remain open. |
| MC-017 | Synthetic and local FFmpeg coverage is substantially expanded, including retry-key ownership guards; provisioned GPU, production disk-backed repeat, live integration, real-audio corpus, and listening coverage remain open. |

### P0: audio-integrity blockers

| ID | Defect | Evidence | User-visible effect |
|---|---|---|---|
| MC-001 | A new run can recursively clean a previous Magic Clean output. | `hear/orchestrator.py:1395-1426` downloads the submitted URL without Magic Clean lineage resolution. `docs/BACKEND_INTEGRATION.md:424-428` tells the backend to adopt the enhanced URL. | Each deliberate new job using the prior output repeats separation, suppression, dynamics, mono conversion, and MP3 encoding. Speech and ambience progressively disappear and artifacts accumulate. |
| MC-002 | Compressor gain starts near zero at every processing window. | `hear/services/magic_clean/processing/dynamics.py:62-67` runs `lfilter` with zero initial state. Production stem processing is also finalized as `ContentMode.MUSIC` at `pipeline.py:164`, selecting slower dynamics. | The start of a clip or chunk fades in as if it were cut. Repeated runs attenuate the opening again. |
| MC-003 | The custom spectral suppressor is not length- or amplitude-safe. | `noise.py:83-120` uses unpadded Hann overlap/add, divides by tiny edge weights, and never processes the final partial frame. `noise.py:78-81` forces processable clips shorter than three seconds to at least 95% suppression, even when strength is zero; sub-window inputs can instead throw internally and fall back. | Edge spikes, zeroed tails, pumping after peak scaling, metallic noise, and damaged short clips. |
| MC-004 | Silence removal fades into detected speech and operates independently per chunk. | `silence.py:12-16,73-102` applies a 300 ms fade with only 200 ms pre-pad and 150 ms post-pad. `pipeline.py:87-95` and `streaming.py:51-74,133-171` disable overlap and strip each chunk separately. | Initial consonants, quiet endings, breaths, short words, and speech crossing a chunk boundary can be faded or removed. |
| MC-005 | A MossFormer failure can silently change sample count and effective rate. | `mossformer.py:67-73` overwrites the working tensor with a 48 kHz resample; its broad fallback at `mossformer.py:136-138` can return that tensor to a caller treating it as 44.1 kHz. | Production stem remix will commonly fail on the resulting shape mismatch; a path that accepts it can instead drift or stretch duration. |
| MC-006 | There is no do-no-harm gate before upload. | `audio_utils.py:96-132` encodes without checking speech retention or decoded output duration. `streaming.py:88-96` measures durations but does not compare them. | A truncated, muted, discontinuous, or otherwise degraded result can be published as successful. |

### P1: professional-quality blockers

| ID | Defect | Evidence | Consequence |
|---|---|---|---|
| MC-007 | The tested safe default path is not the production default. | Omitted controls become `100/10/10` in `jobs/submission.py:108-115`; the service creates `StemLevels` at `service.py:128-167`; any levels route to stem mixing at `pipeline.py:54-55`. The no-stem test at `tests/test_magic_clean_pipeline.py:90-107` calls the pipeline directly with `levels=None`. | Normal production input always receives Demucs isolation, 90% non-vocal attenuation, and up to 90% residual suppression. Separator leakage can remove real speech. |
| MC-008 | The default is vocal isolation, not transparent cleanup. | `pipeline.py:143-164` keeps 100% enhanced vocals but only 10% music and 10% algebraic residual. | Music, room tone, overlapping voices, and speech placed in non-vocal stems sound thin or disappear. |
| MC-009 | “Background” is not a separately estimated background stem. | `pipeline.py:146-148` defines it as `waveform - speech - music`; the same control is also mapped to spectral suppression at `pipeline.py:150-155`. | The slider mixes a reconstruction error while also changing denoise strength, so its quality and meaning are unpredictable. |
| MC-010 | Chunk mastering differs by duration. | `pipeline.py:92-115` finalizes every in-memory chunk. `streaming.py:139-146` disables per-chunk finalization and later applies global ffmpeg loudness. The branch changes at `service.py:124-143`. | A 299-second and 300-second file can have different loudness, pumping, and seams. |
| MC-011 | Fixed processing is always cascaded with no clean-input bypass. | Production stem processing applies MossFormer, spectral subtraction, EQ, and de-essing at `pipeline.py:143-164`, followed by dynamics/finalization at `pipeline.py:173-189`. | Already-clean recordings and low-noise speech can become less natural. Repeating fixed EQ/dynamics compounds tonal change. |
| MC-012 | Stereo and source precision are discarded before separation. | `hear/core/downloader.py:67-73` forces mono PCM16. `audio_io.py:23-28` and `streaming.py:60-65` also downmix; `stems.py:23-32` duplicates mono back to artificial stereo for Demucs. | Lost spatial separation cues, possible phase cancellation, weaker stem separation, and an avoidable quality ceiling. |
| MC-013 | Delivery uses a generic Alexa-oriented 96 kbps setting. | `service.py:190-198` uses `PIPELINE_MP3_BITRATE_KBPS`; because orchestration already converted input to WAV, it selects the 96 kbps cap. | Each deliberate rerun adds another lossy MP3 generation. The documentation's adaptive-source-bitrate claim is not true for this path. |
| MC-014 | Processor exceptions can masquerade as success. | MossFormer, spectral suppression, EQ, de-essing, and dynamics catch broad exceptions and return a tensor. | A model/DSP failure can produce partially processed or invalid audio while the result says `enhanced=true`. |
| MC-015 | Quality values do not describe the delivered artifact. | Long-file values are fixed at `streaming.py:90-96` and `service.py:99-109`. Short-file clipping is measured on input at `service.py:145-150`, and peak/LUFS are measured before MP3 encoding. `quality.py:10-15` calls source/output difference a noise estimate. | Quality can appear successful when audio was damaged; short and long jobs are not comparable. |
| MC-016 | Progress and timing do not represent actual processing. | Separation, enhancement, mix, finalization, encode, and upload occur inside the remote call at `orchestrator.py:1413-1427`; `mixing` and `finalizing` are emitted afterward. `EnhancementResult` has no `stage_times` field, although `orchestrator.py:1454` reads it. | Operational diagnosis is weak, stage timings are empty, and a long remote call is difficult to cancel safely. |
| MC-017 | Existing tests prove contracts, not delivered sound. | Current tests use identity processors, constant tensors, or protobuf fixtures. There are no real spectral, silence, streaming, failure-output, stereo, repeated-output, or post-encode quality tests. | Reproducible audio corruption exists while the focused test suite remains green. |

## 5. Reproduction evidence from this audit

These are local synthetic diagnostics, not a substitute for a real listening corpus:

- A two-second 440 Hz tone passed through `spectral_suppress(..., strength=0)` had a
  maximum absolute error of `107.31`; sample 4 was about `107.36` while the input was
  about `0.05`, and the output ended with zeroed samples. Strength zero must be identity.
- With a constant `0.1` input at 44.1 kHz and the current speech/music thresholds and release
  profiles, compressor output/input gain at time zero was about `0.0003` on the speech path
  and `0.0001` on the music path. At 80 ms it was only about `0.56` and `0.35`, respectively.
- A one-second synthetic speech region surrounded by silence retained only 856 of 1,000
  full-level speech samples after the first silence pass. Repeated passes continued reducing
  the strong portion (`856 -> 826 -> 810 -> 800`).
- An unloaded/failed MossFormer call returned 48,000 samples for a 44,100-sample input.
- Thirty-two focused Magic Clean, stage, model-path, and submission tests passed during the
  audit. Their success confirms the coverage gap; it does not validate audio quality.

No live model, private audio, production storage, or competitor API was used for these
diagnostics.

Audit environment: source base `4c08cf2`, Python 3.12.3, PyTorch 2.8.0+cu128, and SciPy
1.16.3. The focused baseline command was:

```bash
uv run pytest tests/test_magic_clean_pipeline.py tests/test_magic_clean_stages.py \
  tests/test_magic_clean_model_path.py tests/test_job_submission.py -q
```

The ad hoc probes above are specifications for permanent tests; release evidence must come
from checked-in deterministic fixtures rather than relying on these one-time shell probes.

## 6. Target internal architecture

```text
request (unchanged)
  -> resolve/pin canonical non-Magic source revision and engine revision
  -> lossless float decode with source channels preserved
  -> source analysis: speech activity, noise, music, reverb, loudness, clipping
  -> generate candidates from that canonical source revision
       A. clean-input bypass / minimal mastering
       B. conservative speech enhancement
       C. stem-assisted mixed-content enhancement when justified
  -> speech-protection and objective candidate selection
  -> context-aware exact-length stitching
  -> optional one-time global silence shortening
  -> one global EQ/dynamics/loudness/true-peak pass
  -> one high-quality MP3 encode
  -> decode the MP3 and enforce hard release gates
  -> upload, persist real measurements, complete job
```

### 6.1 Canonical source and repeat safety

The transformation must be absolute: controls describe the desired result from the canonical
non-Magic source revision, never an incremental edit of the prior clean result.

1. Before download, query completed Magic Clean jobs scoped to the same `backend_id` and
   `track_id`.
2. If the submitted URL exactly equals a previous `enhanced_audio.audio_url`, follow that
   job's persisted root lineage. After download, also compare a cryptographic delivered-file
   hash and deterministic decoded-PCM hash against known outputs for this backend and track.
   This safely covers query-string aliases or byte-identical reuploads; never use perceptual
   fingerprinting to guess across identity boundaries.
3. Persist `magic_clean_root_url`, source hash, parent job, control tuple, and engine revision
   in internal job metadata before inference so automatic retries are stable.
4. Detect cycles, ambiguous matches, tenant/track mismatches, and excessive lineage depth;
   never guess aliases across a security boundary.
5. Cache intermediate source-derived stems by backend, track, canonical PCM hash, and model
   revision, subject to the security/retention controls above.
   Identical controls and engine revision can copy the prior validated bitstream to the
   current job's required key without re-encoding; changed controls remix source-derived
   material.
6. If the canonical source is unavailable, fail closed rather than destructively process a
   known enhanced artifact.

Exact-URL lineage requires the canonical revision URL to remain readable. Guaranteed reruns
after expiry/deletion require a new private canonical-master storage capability with explicit
encryption, future-read authorization, retention, deletion, and quota policy. The current
job-scoped storage flow constructs a public artifact URL and is not sufficient proof of that
capability. This is unresolved internal infrastructure, not a backend request change. Until
it is approved and implemented, unavailable roots must produce a clear job failure rather
than a damaged output.

Known exact URLs, object identities, delivered-byte hashes, and decoded-PCM hashes cover the
normal backend flow and exact aliases. A transcoded reupload that matches none of those is
indistinguishable from a new source without a new trusted identifier; treat it as a new
canonical revision and record this limitation rather than using fuzzy audio identity.

### 6.2 Safe ingest and channel handling

- Stop converting Magic Clean inputs to mono PCM16 in the orchestrator.
- Decode once to float32 and retain the original sample rate and channel layout.
- Analyze channel correlation and per-channel speech activity before downmix. Use a
  deterministic correlated-mid or speech-dominant-channel policy as appropriate; never
  blindly average anti-correlated channels for a model that requires mono.
- Run separation with real stereo where available and reconstruct retained ambience/music
  in the original stereo field.
- Resample only at model boundaries and require every model adapter to return the declared
  rate, channel count, exact core length, finite samples, and bounded energy.
- Keep an immutable canonical-source tensor/file for every fallback. Never return a mutated
  resampled working tensor after failure.

### 6.3 Content-aware enhancement and mixing

- Keep MossFormer2 as an initial candidate, not an unconditional proof of quality.
- Use VAD, estimated noise level, music presence, and separation confidence to select the
  lightest valid path.
- Bypass aggressive denoise on already-clean speech.
- Use context padding around neural windows and retain only the valid center, with a single
  overlap/add implementation shared by short and long files.
- Temporarily disable the current spectral suppressor. Replace it only after its weighted
  overlap/add implementation passes exact identity, length, boundary, attenuation-floor,
  and repeated-pass tests.
- Protect speech using confidence-controlled dry/wet blending. Low-confidence, unvoiced,
  transient, whispered, and overlapping-speech regions receive more canonical-revision signal.
- Apply EQ and de-essing conditionally and once. Do not repeatedly impose fixed tonal curves.
- Build mixture-consistent speech, music, and ambience estimates whose sum reconstructs the
  source. The `background` control continues to mean “less retained ambience/noise,” but it
  is not permission to delete detected speech.
- Apply speech protection to enhancement/separation before final user gains, then honor the
  explicit component gains. An explicit `speech=0` or `0/0/0` is intentional removal and is
  tested under a different quality/loudness policy.

The production default can remain visibly `100/10/10`, but that tuple must be safe whether it
was omitted or explicitly sent by an existing backend. If omitted intent is later used to
select different internal semantics, persist that bit and include it in the request
fingerprint so idempotency comparisons remain correct.

### 6.4 Silence behavior

`cut_silence=false` is a strict timeline-preservation mode. No VAD edit may change its PCM
sample count.

When `cut_silence=true`:

- enhance and stitch the complete timeline first;
- run the already-declared Silero VAD from a provisioned local artifact, with no runtime
  model download; add an explicit model path, artifact manifest/checksum, configuration, and
  startup validation because the dependency alone does not provision a production model;
- combine canonical-source VAD, enhanced-output VAD, and conservative energy/hysteresis
  checks so the detector can notice speech that enhancement accidentally removed;
- protect music, singing, and uncertain mixed-content activity or disable shortening for
  those regions;
- retain generous pre/post speech protection and a natural minimum pause;
- remove or shorten only long, high-confidence non-speech intervals;
- make joins inside non-speech with short 10-30 ms equal-power crossfades;
- create an internal edit-decision map for duration and speech-retention validation;
- bypass silence editing if confidence is low or any speech-preservation gate fails.

Silence decisions must be global. Long recordings may use a disk-backed VAD timeline, but
they must not reset thresholds or cut independently at 60-second boundaries.

### 6.5 Dynamics and mastering

- Initialize compressor/envelope state from the first target value, not zero.
- Implement standard attack/release ballistics in the dB domain and carry state across
  blocks or discard context margins.
- Use speech dynamics for speech, including stem-assisted speech; do not force the music
  profile for the complete mixed result.
- Stitch all enhanced core windows before final mastering.
- Measure eligible program material with gated BS.1770, apply one bounded static/global gain
  toward `-16 LUFS`, and do not boost silence, very short material, or intentionally removed
  `0/0/0` output toward a program-loudness target.
- Apply one linked-multichannel, oversampled true-peak limiter with a bounded gain-reduction
  budget. Track maximum gain change, limiter reduction, loudness range, crest-factor collapse,
  DC offset, channel energy, phase coherence, image width, and unintended channel collapse.
- Use a ceiling that leaves codec headroom. If the decoded MP3 exceeds `-1 dBTP`, perform at
  most two bounded re-encodes with a lower ceiling; fail if it still misses the gate.
- Decode and measure the final MP3. Pre-encode measurements are diagnostic only.

Magic Clean output encoding should be decoupled from the generic Alexa delivery bitrate.
Benchmark 128 kbps mono and 192 kbps stereo candidates against the 96 kbps compatibility
setting before choosing defaults. Pin the encoder, CBR/VBR mode, sample-rate/channel policy,
and delay/padding handling in the release evidence. This changes no request or result field.

### 6.6 Candidate selection and failure policy

Generate multiple mixes from one model/separation pass so safety does not require repeated
expensive inference:

1. minimally mastered canonical source revision;
2. conservative enhanced/source blend;
3. full candidate allowed by the requested controls.

Choose the least aggressive candidate that measurably improves background/noise quality
without reducing the components the controls request to retain. If the full candidate fails,
fall back to the conservative candidate. For zero-backend-change rollout, fail the existing
job when no safe candidate passes. Completing with `enhanced=false` is only an option after
consumer handling is audited, because current guidance tells the backend to adopt
`enhanced_audio` on completion. Never publish damaged audio with `enhanced=true`.

Broad “catch and return input” behavior must be removed from required stages. Failures need
typed internal errors, the correct failed stage, cleanup, and no orphaned committed object.

### 6.7 Delivered-artifact validation

Before upload, and again after decoding the exact MP3 that will be uploaded, enforce:

- decoder readability, non-empty output, finite samples, and valid channels/rate;
- exact pre-encode sample count when `cut_silence=false`;
- decoded duration within one codec frame (or 30 ms, whichever is larger) when no cut is
  requested; with silence cutting, require the retained PCM not to exceed the source timeline
  and allow only that codec/resampling tolerance in the decoded MP3; treat 30 ms as provisional
  until calibrated for the pinned encoder/container;
- source/output alignment after known fixed-delay compensation, with no unexplained local
  time warp, drift, or polarity inversion;
- no source-normalized unexpected zero runs, channel loss, or model length drift in protected
  activity;
- no lost protected VAD regions when speech retention is requested; aligned-word retention
  is enforced in the release corpus and at runtime whenever reference alignment is available;
- no source-normalized derivative/click anomaly at a known processing boundary;
- integrated loudness of `-16 LUFS +/- 1` for eligible speech/program material, with bounded
  gain; short, silent, ambience-only, and intentionally removed output uses peak/RMS guards;
- decoded true peak at or below `-1 dBTP` and zero clipped samples;
- bounded gain, spectral change, and—when speech is retained—speaker-embedding drift on clean
  input;
- a calibrated no-reference perceptual check only for durations and content supported by the
  validated scorer; short/unsupported material uses integrity and corpus-proven safeguards.

The current Phase 1 checkpoint implements decodeability, exact pre-encode timeline handling,
decoded-duration/layout checks, gross silence and per-channel energy-collapse checks, clipping,
true peak, and remote byte integrity. It does **not** yet establish content identity: aligned
content/polarity, local time-warp, click, spectral-change, protected-word, phase/image, and
speaker-drift checks above remain release gates pending calibrated tolerances and corpus proof.

Validate first, upload second. Verify remote size and checksum. If cancellation or terminal
failure wins after upload but before completion, delete the uncommitted object when the
scoped credentials permit it. If deletion cannot be confirmed, write a durable cleanup
tombstone for reconciliation and alert on expiry/repeated cleanup failure.

### 6.8 Measurements and observability

Keep the existing response fields, but make them truthful:

- `peak_db`: measured decoded-output true peak;
- `lufs`: measured decoded-output integrated loudness;
- `clipping_detected`: decoded-output clipping, not input clipping;
- `snr_db`: a defined calibrated VAD/mask-based estimate when enough material exists;
  otherwise the current non-optional scalar uses documented `0` as an unavailable sentinel
  and is excluded from `quality_score`;
- `quality_score`: keep the public scalar/range compatible until consumer thresholds are
  audited; calibrate any semantic migration and version richer scoring internally first;
- `stage_times`: populate real timings while keeping the existing five streamed stage IDs;
  nest analysis, separation, enhancement, mix/stitch, validation, and upload subtimings in
  the existing Struct.

Advanced values such as source hash, engine/model revision, VAD retention, duration delta,
fallback reason, cache hit, and perceptual sub-scores remain internal initially. Log only
safe numeric/identity metadata—never credentials, private URLs, or audio content.

Align the existing streamed stage transitions with the actual work, propagate chunk progress,
check cancellation between bounded operations, and clean temporary files and uncommitted
uploads on every exit path. New public stage IDs require a separate compatibility decision.

## 7. Implementation sequence

### Phase 0 — containment and regression harness

**Checkpoint:** the core containment, local regression harness, and short provisioned-model
execution are implemented, but this phase is not complete. Object-version identity,
calibrated zero-run protection, and live PostgreSQL/object-store integration remain before its
exit gate can be accepted for release.

1. Add synthetic tests that reproduce the spectral edge spike, compressor fade-in,
   MossFormer rate drift, silence erosion, and repeated-output behavior.
2. Bypass the current spectral suppressor until its replacement passes identity tests.
3. Fix compressor initial state and MossFormer fallback invariants.
4. Add pre-upload length, finiteness, peak, zero-run, and decoded-duration checks.
5. Add canonical lineage using exact URL/object identity and scoped byte/PCM hashes for prior
   Magic Clean outputs; pin the first downloaded hash so mutable-URL retries fail safely.

**Exit gate:** no known path may shorten `cut_silence=false`, return a rate/length-mismatched
tensor, or process a known prior Magic Clean artifact recursively.

### Phase 1 — one consistent, speech-safe engine

**Checkpoint:** the safety-critical engine, basic mastering, channel preservation, silence
editing, delivered-artifact validation, metrics plumbing, and retry-aware cleanup are
implemented locally. The bounded decoded-MP3 true-peak re-encode and short dual-model CUDA
smoke now pass. Content-aware candidate selection, provisioned VAD, calibrated speech-retention
checks, real-corpus parity, and long-duration evidence remain open.

1. Preserve source channels/precision through ingest.
2. Replace the split in-memory/streaming behavior with one context/core chunk engine.
3. Stitch once and master once.
4. Move silence shortening to one global post-stitch decision pass.
5. Add content analysis, clean-input bypass, confidence-based blending, and
   mixture-consistent component mixing.
6. Decode the final MP3 for real metrics and hard release gates.
7. Populate real stage timings and cancellation/cleanup behavior.

**Exit gate:** short/long parity, clean-input transparency, speech-retention, mastering, and
post-encode gates pass the versioned fixture corpus.

### Phase 2 — model bake-off and professional quality gate

**Checkpoint:** not started; architecture or synthetic tests cannot substitute for this phase.

Benchmark the current locally hosted MossFormer2/Demucs candidate against alternative
self-hosted enhancement/separation models. A candidate is adopted only if its pinned model
artifact, license, GPU memory, latency, and audio results pass. Models are provisioned during
a controlled build/operator step; nothing downloads at application startup or request time.

Run objective metrics and blinded listening against source, current Magic Clean, the new
candidate, and legally obtained ElevenLabs reference output.

**Exit gate:** statistically supported non-inferiority in speech naturalness/intelligibility,
with required background reduction and no cohort-specific regressions.

### Phase 3 — shadow, canary, and rollout

**Checkpoint:** not started and blocked on Phase 2 evidence.

1. Run the new scorer/pipeline on non-private test traffic or an approved shadow corpus.
2. Compare rejection/fallback rate, speech retention, duration, GPU time, and artifact size.
3. Canary by server-side engine revision; no backend flag is required.
4. Retain immediate rollback to the last safe engine revision.
5. Promote only after the release matrix and listening report are attached to the release.

## 8. Required automated coverage

### Unit and DSP tests

- spectral strength-zero identity for impulses, tones, speech, and noise at arbitrary lengths;
- no unprocessed tail, edge amplification, NaN/Inf, or unexplained zeros;
- compressor unity/target gain from sample zero and state continuity across chunks;
- MossFormer unloaded/error/short/long/NaN output preserving original rate, length, and device;
- exact overlap/add length at `chunk-1`, `chunk`, `chunk+1`, and every overlap remainder;
- exact chunked length, bounded model-specific error in valid core regions, and no boundary
  degradation; nonlinear neural output need not equal monolithic inference;
- conditional EQ/de-esser and correct speech/music dynamics selection;
- mono, stereo, one-sided dialogue, phase-opposed, and multichannel ingest cases;
- quiet consonants, whispers, breaths, short words, and speech spanning chunk boundaries;
- silence-cut removal limits, edit map, joins, and conservative fallback.

### Service, lineage, and contract tests

- omitted-control production routing, explicit controls, and every `0/100` boundary;
- same-job replay performs no second inference;
- five new jobs using each prior result URL all resolve to one canonical source revision;
- changed controls remix the root; identical controls/revision copy a validated bitstream to
  the current job key without another model pass or encode;
- cross-backend, cross-track, ambiguous, cyclic, known URL/hash alias, unavailable root, and
  unrecognized transcoded-reupload behavior follows the documented security policy;
- mutable source content is detected by canonical PCM hash;
- short/long threshold parity around 60 and 300 seconds;
- final MP3 decode, duration, checksum, LUFS, true peak, clipping, and speech retention;
- cancellation/failure after processing leaves no authoritative artifact and either deletes
  the object or creates a durable, alertable cleanup tombstone;
- real `stage_times` and existing REST/gRPC oneof/result layout;
- no protobuf descriptor or backend request change.

### Real-audio corpus

Use versioned, non-private, legally usable fixtures covering:

- clean studio and phone speech;
- stationary, transient, crowd, traffic, wind, and keyboard noise;
- reverberant rooms and far-field microphones;
- speech over music and changing music beds;
- overlapping speakers, singing, whispers, breaths, and non-verbal speech;
- multiple languages, accents, ages, voice ranges, and speaking rates;
- stereo placement, phase edge cases, clipped sources, and low-bitrate sources;
- sub-three-second clips, 60/300-second boundaries, and long recordings;
- leading, trailing, and internal silence around quiet words.

## 9. Initial objective and listening release gates

Use two layers. Per-artifact hard integrity gates reject broken output. Corpus-level
perceptual/intelligibility gates use time-aligned signals, cohort medians, confidence
intervals, and catastrophic-tail limits; they do not require every noisy file to improve every
metric. Numeric cohort thresholds are preregistered after measuring the baseline and before
evaluating the candidate.

| Area | Gate type | Initial gate |
|---|---|---|
| Timeline | Per artifact | Exact PCM core length with `cut_silence=false`, plus fixed-delay-compensated alignment with no local time warp or polarity reversal. The decoded MP3 allowance starts at one codec frame or 30 ms but must be calibrated to the pinned encoder. |
| Speech retention | Per artifact/corpus | When final controls retain speech, at least 99% protected VAD-frame retention at runtime and no deleted aligned word in the release corpus. Intentional speech removal uses a different gate. |
| Intelligibility | Corpus | Ground-truth ASR WER no more than 1 absolute point worse on clean speech and 2 points worse on noisy speech as initial cohort limits; Magic Clean does not add a runtime ASR call in Phase 0. |
| Loudness | Per artifact | Eligible speech/program output targets `-16 LUFS +/- 1` with bounded gain. Silent, short, ambience-only, and intentionally removed output must not be amplified to that target. |
| Peak/dynamics | Per artifact | Delivered true peak `<= -1 dBTP`, zero clipped samples, and calibrated limits for limiter reduction, crest factor, loudness range, and DC offset. |
| Boundaries | Per artifact | No source-normalized derivative/click anomaly at known processing boundaries; calibrate the detector on natural transitions rather than using a blanket energy-step rule. |
| Clean transparency | Corpus | For supported-duration clips with validated SIG/BAK/OVRL extraction, initial DNSMOS SIG decline no worse than 0.1, plus preregistered spectral and speaker-similarity tails. Exempt unsupported short clips. |
| Noisy improvement | Corpus | Preregister numeric DNSMOS background/overall, SI-SDR, and STOI/ESTOI cohort improvements where aligned clean references exist. Report confidence intervals and worst-tail regressions. |
| Repeat stability | Integration | Five known derivative attempts resolve to the same canonical revision or fail safely; identical controls/revision copy the same validated content to each authorized job key. |
| Path parity | Corpus | Short and disk-backed processing meet the same cohort bands and show no boundary-specific preference loss. |

Run ITU-T P.835-style testing for speech distortion, background intrusiveness, and overall
quality. If MUSHRA is also used for wideband artifact/naturalness comparison, run it as a
separate defined study with hidden reference and anchor. Preregister the non-inferiority
margin, sample-size/power calculation, listener screening, playback calibration,
randomization, exclusions, and confidence-interval analysis before examining candidate
results. Cohorts must be examined individually so quiet voices, accents, stereo material, or
music-backed speech cannot regress behind an improved overall mean.

## 10. Files expected to change during implementation

The exact diff should stay focused, but the likely surfaces are:

- `hear/orchestrator.py`: canonical Magic Clean source resolution, truthful stages, cleanup;
- `hear/services/jobs/submission.py`: retain omitted-vs-explicit intent and internal revision;
- `hear/core/downloader.py`: preserve Magic Clean input fidelity;
- `hear/services/magic_clean/service.py`: unified engine, post-encode validation, real metrics;
- `hear/services/magic_clean/pipeline.py`: candidate flow, one stitch/master pass;
- `hear/services/magic_clean/streaming.py`: shared disk-backed implementation;
- `hear/services/magic_clean/processing/`: safe model adapter, dynamics, VAD, mixing, quality;
- `hear/services/magic_clean/models.py`: typed internal reports and stage timings;
- `hear/config.py` and `.env.example`: server-side engine/output/quality defaults only;
- focused unit, integration, contract, and audio-quality tests;
- backend integration/runbook text whose current bitrate and progress claims are inaccurate.

## 11. Risks and decisions that must be recorded

1. **Canonical master retention:** guaranteed repeat safety after source expiry needs an
   approved private/encrypted source-master capability, future-read authorization, and
   retention/deletion policy that current job storage does not provide. Without it, known
   lineage works while the canonical revision URL is readable and otherwise fails safe.
2. **Output bitrate/channel compatibility:** evaluate pinned 128 kbps mono and 192 kbps stereo
   candidates with every playback target; keep the server-side 96 kbps setting unless a
   candidate wins both listening and compatibility gates.
3. **Model selection:** do not promise vendor parity or add a model based on reputation.
   License, artifact provenance, latency, memory, and cohort results all gate adoption.
4. **Silence semantics:** `cut_silence=true` intentionally changes time. The release report
   must state maximum removal, natural-pause policy, and verified speech retention.
5. **Fallback semantics:** fail the job during the zero-backend-change rollout if no safe
   candidate passes. A future safe-master completion with `enhanced=false` requires explicit
   consumer verification even though the field is wire-compatible.
6. **Quality-score semantics:** audit every consumer threshold before changing the public
   scalar's meaning. Keep richer/versioned scoring internal until migration behavior is
   proven compatible.

## 12. Definition of done

Magic Clean is ready only when all of the following are true:

- every P0 and P1 defect above has a focused regression test and verified fix;
- every recognized prior Magic Clean artifact is resolved to its canonical non-Magic source
  revision or fails safely; the documented exact-URL backend flow passes five-run testing;
- `cut_silence=false` preserves timeline and, when requested, protected speech across short
  and long files;
- `cut_silence=true` passes the removal, word-retention, and join-quality gates;
- the delivered MP3—not an intermediate tensor—passes loudness, peak, duration, decode,
  speech-retention, and perceptual checks;
- real quality values (or the documented unavailable sentinel where the wire contract cannot
  express absence) and stage timings are returned consistently;
- REST/gRPC requests and the typed Magic Clean result remain backend-compatible;
- the versioned corpus, five-run stability matrix, GPU integration run, and blinded listening
  report pass in a non-production environment;
- model/startup validation, cancellation, storage cleanup, and rollback are demonstrated;
- documentation matches the actual production path.

Until those gates pass, the service should be described as under remediation rather than
professional-grade or ElevenLabs-equivalent.
