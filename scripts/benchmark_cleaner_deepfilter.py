"""Offline DF3 duration probe with synthetic audio, not release certification.

CUDA mode must be externally admitted/supervised and whole-process GPU memory
sampled separately. Torch peak counters alone do not measure CUDA contexts or
non-Torch allocations. CPU remains the default; this script never starts Ray.
"""

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hear.runtime.cleaner.deepfilter_loader import PinnedDeepFilterAssets, PinnedDeepFilterFactory
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanPlan
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterSession


class ProgressBackend:
    def __init__(self, backend):
        self.backend = backend
        self.calls = 0
        self.max_window_frames = 0

    def enhance(self, samples, attenuation_limit_db):
        result = self.backend.enhance(samples, attenuation_limit_db)
        self.calls += 1
        self.max_window_frames = max(self.max_window_frames, samples.shape[1])
        if self.calls % 6 == 0:
            print(
                json.dumps(
                    {"completed_blocks": self.calls, "peak_rss_bytes": DeepFilterSoak.peak_rss()}
                ),
                flush=True,
            )
        return result

    def close(self):
        self.backend.close()


class DeepFilterSoak:
    @staticmethod
    def peak_rss():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Linux memory measurement unavailable")

    @staticmethod
    def run():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--seconds", type=int, default=3600)
        parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
        parser.add_argument("--allocator-cap-bytes", type=int, default=9_000_000_000)
        args = parser.parse_args()
        if not 1 <= args.seconds <= 7200:
            parser.error("duration must be between 1 and 7200 seconds")
        if not 0 < args.allocator_cap_bytes <= 9_000_000_000:
            parser.error("allocator cap must be positive and no greater than 9 GB")
        root = Path(__file__).resolve().parents[1]
        assets = PinnedDeepFilterAssets(
            root / "deploy/cleaner/deepfilter3.ini",
            "0a926b0471793d7ba7446b07a8bdc10eafa5c9e3b93de4d65496e2cbcacc40d3",
            args.checkpoint,
            "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003",
            (
                ("deepfilternet", "0.5.6"),
                ("deepfilterlib", "0.5.6"),
                ("torch", "2.8.0+cu128"),
                ("torchaudio", "2.8.0+cu128"),
                ("numpy", "1.26.4"),
            ),
            args.device,
        )
        policy = ContextualPolicy(480000, 48000)
        factory = PinnedDeepFilterFactory(assets)
        # Record code separately from the asset descriptor; neither certifies an image.
        code_digest = hashlib.sha256()
        for relative in (
            "hear/runtime/cleaner/deepfilter_loader.py",
            "hear/services/magic_clean/engines/deepfilter.py",
        ):
            code_digest.update((root / relative).read_bytes())
        plan = CleanPlan(
            profile="natural",
            profile_version="benchmark-v1",
            catalogue_sha256="0" * 64,
            runtime=factory.identity(policy.digest),
            attenuation_limit_db=18,
            noise_reduction_db=None,
            noise_reference=None,
            prompt_sha256=None,
            channel_policy="preserve",
            mono_acknowledged=False,
            adjust_loudness=False,
            match_comparison_loudness=True,
            shorten_pauses=False,
            seed=42,
        )
        frames = args.seconds * 48000 + 17
        torch.set_num_threads(1)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hear-df3-soak-") as directory:
            workspace = Path(directory)
            source, output = workspace / "source.wav", workspace / "processed.wav"
            guard = ResourceGuard(
                ResourceBudget(
                    frames * 8 * 4 + 1048576,
                    frames * 8 + 1048576,
                    frames,
                    allocator_cap_bytes=args.allocator_cap_bytes,
                ),
                workspace,
                time.monotonic() + 7200,
                threading.Event(),
            )
            rng = np.random.default_rng(42)
            with sf.SoundFile(
                source, "w", samplerate=48000, channels=2, format="RF64", subtype="FLOAT"
            ) as audio:
                for start in range(0, frames, 48000):
                    guard.check()
                    count = min(48000, frames - start)
                    wave = rng.normal(0, 0.02, (count, 2)).astype(np.float32)
                    wave[:, 0] += (
                        0.1 * np.sin((start + np.arange(count)) * (2 * np.pi * 440 / 48000))
                    ).astype(np.float32)
                    audio.write(wave)
            factory.validate_identity(plan.runtime)
            # Counter reset may initialize a context but does not allocate model
            # tensors. The loader sets the cap before model transfer to CUDA.
            if args.device == "cuda:0":
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(0)
            backend = ProgressBackend(factory.open(guard))
            lease = threading.Lock()
            lease.acquire()
            session = DeepFilterSession(plan, backend, policy, lease)
            inference_started = time.monotonic()
            try:
                session.process(source, output, plan, guard)
            finally:
                session.close()
            inference_seconds = time.monotonic() - inference_started
            cuda_metrics = None
            if args.device == "cuda:0":
                torch.cuda.synchronize(0)
                cuda_metrics = {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "allocator_cap_bytes": args.allocator_cap_bytes,
                    "device_name": torch.cuda.get_device_name(0),
                }
            peak_rss = DeepFilterSoak.peak_rss()
            read_frames = 0
            digest = hashlib.sha256()
            with sf.SoundFile(output) as audio:
                if (audio.frames, audio.samplerate, audio.channels, audio.format) != (
                    frames,
                    48000,
                    2,
                    "RF64",
                ):
                    raise RuntimeError("output metadata mismatch")
                while True:
                    guard.check()
                    chunk = audio.read(48000, dtype="float32", always_2d=True)
                    if not len(chunk):
                        break
                    if not np.isfinite(chunk).all():
                        raise RuntimeError("nonfinite output")
                    read_frames += len(chunk)
                    digest.update(chunk.tobytes())
                audio.seek(frames - 17)
                tail_peak = float(np.max(np.abs(audio.read(17))))
            if read_frames != frames or tail_peak <= 1e-8:
                raise RuntimeError("truncated or missing tail")
            print(
                json.dumps(
                    {
                        "result": "passed",
                        "adapter_code_sha256": code_digest.hexdigest(),
                        "synthetic_audio": True,
                        "device": args.device,
                        "pid": os.getpid(),
                        "cuda": cuda_metrics,
                        "frames": frames,
                        "channels": 2,
                        "sample_rate": 48000,
                        "inference_seconds": inference_seconds,
                        "rtf": inference_seconds / args.seconds,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_rss_bytes": peak_rss,
                        "max_window_frames": backend.max_window_frames,
                        "blocks": backend.calls,
                        "output_bytes": output.stat().st_size,
                        "tail_peak": tail_peak,
                        "pcm_sha256": digest.hexdigest(),
                        "runtime": plan.runtime.model_dump(mode="json"),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    DeepFilterSoak.run()
