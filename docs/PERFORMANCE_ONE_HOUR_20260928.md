# Measured one-hour performance tuning

The current real-model simulation API was tested with the same full 3,600-second
mono recording before and after tuning. The fixture repeats Track 8; it is not
an independent hour-long programme. One pipeline and one Studio Voice cleaning
job ran together on a warm A40 Pod. Times include local simulated storage and
validated callbacks, not cold start, queue delay or a real Backblaze round-trip.

| Workflow | Before | After |
| --- | ---: | ---: |
| Pipeline | 294.2794 s | 188.4126 s |
| Studio Voice cleaner | 471.9973 s | 310.2503 s |

ASR settings changed from batches of 2 / long-file batch 1 / 60-second windows to
batch 8 / long-file batch 8 / 240-second windows. The ASR and aligner checkpoints
are unchanged. The tuned transcription stage took 128.35 seconds, including
120.77 seconds for ASR and alignment. This is a stage measurement, not a separate
transcription API benchmark. Different chunk boundaries slightly changed model
text output; manual word-accuracy acceptance was not performed.

Cleaning measurement-only passes now use FFmpeg ebur128 with true-peak analysis
instead of running loudnorm normalization and discarding its output. All final
FLAC/MP3 integrity and peak checks remain. The summary meter has 0.1 LU/dB
resolution; reported peak includes a conservative 0.05 dB rounding margin.
DeepFilterNet and the denoising profile remain unchanged.

Denoising itself took 33.21 seconds. Mastering and export still took 192.10 seconds,
so this is not yet a sub-five-minute complete cleaner: the measured total is
5 minutes 10 seconds. The remaining bottleneck is finishing/export/validation,
not a need to replace DeepFilterNet.

All three tuned audio exports decoded to exactly 172,800,000 mono samples at
48 kHz. Transcript timestamps cover the final minute and stay within the source
timeline. Checksums and returned local-storage URLs were verified.

The active caps are pipeline 7, Magic Clean 4, Fish reconstruction 2 and dedicated
transcription 1, with a shared whole-Pod cap of 10. A mixed 13-job short-input run
completed successfully and observed peaks of 10 total, 4 cleaner and 2 Fish jobs.
A separate seven-pipeline test observed all seven in flight and completed. Seven
pipeline workflows share a single serialized, batched ASR executor; they are not
seven independent simultaneous GPU model copies. These tests do not guarantee
that seven full-hour jobs will each finish in the isolated per-file time.

The full regression run passed 666 tests with 12 skipped, and mypy checked 139
source files successfully. The worker files and environment tuning are active;
no new container image or production backend/B2 deployment is claimed.

Detailed evidence: verification/performance-one-hour-20260928.json.
