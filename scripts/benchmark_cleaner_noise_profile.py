"""Synthetic CPU duration probe, not music/speech-retention certification."""

import argparse
import hashlib
import json
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.noise_reference import SpeechAwareNoiseReferenceAnalyser
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.speech_activity import CpuSpeechActivity, SpeechActivityPolicy
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanPlan, NoiseReference, NoiseReferenceSelection
from hear.services.magic_clean.engines.noise_profile import NoiseProfileEngine


class NoiseProfileSoak:
    @staticmethod
    def run() -> dict:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--speech-model", type=Path, required=True)
        parser.add_argument("--seconds", type=int, default=3600)
        args = parser.parse_args()
        if not 1 <= args.seconds <= 7200:
            parser.error("duration must be between 1 and 7200 seconds")
        frames, rate = args.seconds * 48000 + 17, 48000
        required = frames * 8 * 3 + 16_777_216
        if shutil.disk_usage(tempfile.gettempdir()).free < required:
            raise RuntimeError("insufficient probe scratch space")
        started = time.monotonic()
        root = Path(__file__).resolve().parents[1]
        code = hashlib.sha256()
        for name in (
            "hear/services/magic_clean/engines/noise_profile.py",
            "hear/runtime/cleaner/noise_reference.py",
            "hear/runtime/cleaner/speech_activity.py",
            "hear/runtime/cleaner/resampling.py",
            "hear/runtime/cleaner/resource_guard.py",
        ):
            code.update((root / name).read_bytes())
        with tempfile.TemporaryDirectory(prefix="hear-noise-soak-") as directory:
            workspace = Path(directory)
            source, output = workspace / "source.wav", workspace / "processed.wav"
            guard = ResourceGuard(
                ResourceBudget(required, frames * 8 + 1048576, frames),
                workspace,
                time.monotonic() + 900,
                threading.Event(),
            )
            rng = np.random.default_rng(42)
            with sf.SoundFile(
                source, "w", samplerate=rate, channels=2, format="RF64", subtype="FLOAT"
            ) as audio:
                for start in range(0, frames, rate):
                    guard.check()
                    count = min(rate, frames - start)
                    samples = rng.normal(0, 0.005, (count, 2)).astype(np.float32)
                    if start >= rate:
                        position = start + np.arange(count)
                        samples[:, 0] += (0.1 * np.sin(position * (2 * np.pi * 440 / rate))).astype(
                            np.float32
                        )
                        samples[:, 1] += (
                            0.08 * np.sin(position * (2 * np.pi * 880 / rate))
                        ).astype(np.float32)
                    audio.write(samples)
            speech = CpuSpeechActivity(
                args.speech_model,
                SpeechActivityPolicy(
                    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
                    "1.27.0",
                    "1.26.4",
                ),
                AudioResampler(CancellableProcessRunner()),
                guard,
            )
            analyser = SpeechAwareNoiseReferenceAnalyser(speech)
            selection = NoiseReferenceSelection(
                revision_id="synthetic-soak", start_frame=0, end_frame=rate
            )
            review = analyser.review(source, selection, guard)
            if review.assessment.speech_detected:
                raise RuntimeError("synthetic reference unexpectedly contains speech evidence")
            plan = CleanPlan(
                profile="music_atmosphere",
                profile_version="synthetic-soak-v1",
                catalogue_sha256="0" * 64,
                runtime=NoiseProfileEngine.describe(analyser),
                attenuation_limit_db=None,
                noise_reduction_db=3,
                noise_reference=NoiseReference(
                    revision_id=selection.revision_id,
                    start_frame=selection.start_frame,
                    end_frame=selection.end_frame,
                    confirmed_noise_only=True,
                    analysis_sha256=review.assessment.analysis_sha256,
                ),
                prompt_sha256=None,
                channel_policy="preserve",
                mono_acknowledged=False,
                adjust_loudness=False,
                match_comparison_loudness=True,
                shorten_pauses=False,
                seed=42,
            )
            engine = NoiseProfileEngine(plan.runtime, analyser)
            processing_started = time.monotonic()
            session = engine.open_session(plan, guard)
            try:
                session.process(source, output, plan, guard)
            finally:
                session.close()
            processing_seconds = time.monotonic() - processing_started
            digest, read_frames = hashlib.sha256(), 0
            with sf.SoundFile(output) as audio:
                if (audio.frames, audio.samplerate, audio.channels, audio.format) != (
                    frames,
                    rate,
                    2,
                    "RF64",
                ):
                    raise RuntimeError("output metadata mismatch")
                for chunk in audio.blocks(blocksize=48000, dtype="float32", always_2d=True):
                    guard.check()
                    if not np.isfinite(chunk).all():
                        raise RuntimeError("nonfinite output")
                    digest.update(chunk.tobytes())
                    read_frames += len(chunk)
                audio.seek(frames - 17)
                tail_peak = float(np.max(np.abs(audio.read(17))))
            if read_frames != frames or tail_peak <= 1e-8:
                raise RuntimeError("output truncated or missing tail")
            return {
                "result": "passed",
                "synthetic_audio": True,
                "device": "cpu",
                "frames": frames,
                "channels": 2,
                "sample_rate": rate,
                "processing_seconds": processing_seconds,
                "rtf": processing_seconds / args.seconds,
                "elapsed_seconds": time.monotonic() - started,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "output_bytes": output.stat().st_size,
                "tail_peak": tail_peak,
                "pcm_sha256": digest.hexdigest(),
                "adapter_code_sha256": code.hexdigest(),
                "runtime": plan.runtime.model_dump(mode="json"),
                "speech_policy_sha256": speech.policy.digest,
                "reference_warnings": list(review.warning_codes),
                "torch_imported": "torch" in sys.modules,
            }


if __name__ == "__main__":
    print(json.dumps(NoiseProfileSoak.run()), flush=True)
