# DeepFilterNet profile verification

The JSON file records a real pinned-DeepFilterNet3 CPU run of all four presets on
a complete 52.628958-second deterministic noisy synthetic-speech recording.
Every output retained all 2,526,190 input frames. Master and delivery peaks passed
-1 dBTP validation. Original source bytes remained unchanged. The model checkpoint
was hash-verified and reused between profiles; there was no model stub in this run.

The RMS comparison uses the known noise-only interval at 0.25–1.25 seconds, after
all selected processing and gain. It is not SNR, a perceptual quality metric or a
speaker-intelligibility test. The quiet, high-crest-factor synthetic fixture did
not reach the Studio/Outdoor loudness targets under bounded gain and peak safety;
that warning is retained, not converted into a claim of perfect mastering.

The source contains eSpeak speech twice, with 2-second outer margins and a
3-second internal gap. Speech peak was scaled to 0.35 before deterministic white
noise and 60 Hz hum were added. This test is not a user's uploaded recording.

The repository tests additionally cover real FFmpeg DSP and export behaviour,
contract/route validation, immutable-source digest checks, failed-result handling,
and approval semantics. Some workflow tests intentionally use a model/storage
stub to isolate transport behaviour; they are separate from the real-model run.

GPU throughput, listening acceptance on representative real recordings, live B2
publication and the existing backend/frontend deployment were not verified here.
The AI Pod was disconnected during this change. Do not describe this report as a
production release certificate or a successful production deployment.
