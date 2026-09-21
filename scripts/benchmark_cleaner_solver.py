"""CPU-only analytic solver RSS probe; not SAM/A40 release certification.

Run in separate Linux processes for each size so peak RSS is comparable:
python -m scripts.benchmark_cleaner_solver --frames 131072
python -m scripts.benchmark_cleaner_solver --frames 1048576
"""

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from hear.runtime.cleaner.longform_sam import SolverPolicy, WindowedMidpointSolver
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard


class ConstantField:
    def evaluate(self, state, *, start_frame, time):
        return np.zeros_like(state)


class SolverBenchmark:
    @staticmethod
    def peak_rss_bytes():
        # /proc VmHWM is scoped to this address space. getrusage ru_maxrss can
        # include a pre-exec high-water mark inherited from a launcher.
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Linux VmHWM measurement unavailable")

    @staticmethod
    def run():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--frames", type=int, required=True)
        args = parser.parse_args()
        if not 1 <= args.frames <= 4194304:
            parser.error("frames must be between 1 and 4194304")
        channels, window = 8, 8192
        size = args.frames * channels * 4
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hear-solver-rss-") as directory:
            workspace = Path(directory)
            noise, output = workspace / "noise", workspace / "output"
            block = np.full((window, channels), 0.125, dtype=np.float32)
            with noise.open("xb") as destination:
                for start in range(0, args.frames, window):
                    destination.write(block[: min(window, args.frames - start)].tobytes())
            baseline = SolverBenchmark.peak_rss_bytes()
            guard = ResourceGuard(
                ResourceBudget(size * 5 + 1048576, size, args.frames),
                workspace,
                time.monotonic() + 300,
                threading.Event(),
            )
            policy = SolverPolicy(window, 2048, 2)
            WindowedMidpointSolver(policy).solve(
                noise,
                output,
                frames=args.frames,
                channels=channels,
                field=ConstantField(),
                guard=guard,
            )
            peak = SolverBenchmark.peak_rss_bytes()
            with output.open("rb") as source:
                while chunk := source.read(window * channels * 4):
                    if not np.all(np.frombuffer(chunk, dtype=np.float32) == 0.125):
                        raise RuntimeError("analytic solver output mismatch")
            print(
                json.dumps(
                    {
                        "frames": args.frames,
                        "channels": channels,
                        "latent_bytes": size,
                        "baseline_peak_rss_bytes": baseline,
                        "peak_rss_bytes": peak,
                        "rss_growth_bytes": peak - baseline,
                        "elapsed_seconds": time.monotonic() - started,
                        "policy_sha256": policy.digest,
                        "actual_sam_model": False,
                    },
                    sort_keys=True,
                )
            )


if __name__ == "__main__":
    SolverBenchmark.run()
