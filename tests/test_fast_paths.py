"""The fast long-audio paths must give the same answers as the slow single passes."""

import re
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.parallel import HANDLE_FRAMES, ChunkPlanner, ChunkResult, ParallelCleaner
from hear.runtime.cleaner.pool import WorkerPool
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.loudness import BlockMeasurement, KWeightedMeter
from hear.services.magic_clean.mp3 import FRAME_SAMPLES, EncodeTask, ParallelMp3
from hear.services.magic_clean.quality import BlockEnergies

RATE = 48000


def guard_for(path: Path) -> ResourceGuard:
    return ResourceGuard(ResourceBudget(2**40, 2**40, 2**40), path, time.monotonic() + 120, threading.Event())


def speechlike(seconds: float, channels: int = 1, seed: int = 1) -> np.ndarray:
    """Bursty band-limited noise with pauses, so gating and peaks are exercised."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = (np.sin(2 * np.pi * 0.7 * t) > -0.2).astype(np.float32) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t))
    noise = rng.standard_normal(len(t)).astype(np.float32)
    tone = np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 3.1 * t)
    signal = 0.15 * envelope * (0.5 * noise + tone).astype(np.float32)
    return np.column_stack([signal * (1 - 0.3 * c) for c in range(channels)]).astype(np.float32)


def ffmpeg_meter(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-threads", "1", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr.rsplit("Summary:", 1)[1]
    return (
        float(re.search(r"I:\s*([-\d.]+) LUFS", out).group(1)),
        float(re.search(r"Peak:\s*([-\d.]+) dBFS", out).group(1)),
    )


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_designed_k_weighting_matches_itu_constants_at_48k():
    for (b, a), (b_ref, a_ref) in zip(KWeightedMeter.design(48000), KWeightedMeter.ITU_48K, strict=True):
        assert np.allclose(b, b_ref, atol=1e-6) and np.allclose(a, a_ref, atol=1e-6)


@needs_ffmpeg
@pytest.mark.parametrize("rate,channels", [(48000, 1), (48000, 2), (44100, 1)])
def test_numpy_meter_agrees_with_ffmpeg(tmp_path, rate, channels):
    samples = speechlike(12, channels)
    if rate != RATE:
        from scipy.signal import resample_poly

        samples = resample_poly(samples, rate, RATE, axis=0).astype(np.float32)
    path = tmp_path / "m.wav"
    sf.write(path, samples, rate, subtype="FLOAT")
    integrated, peak = ffmpeg_meter(path)
    measured = [KWeightedMeter.measure(samples, rate)]
    assert KWeightedMeter.integrate(measured) == pytest.approx(integrated, abs=0.2)
    assert KWeightedMeter.true_peak_dbtp(measured) == pytest.approx(peak + 0.05, abs=0.2)


def test_range_measurements_combine_to_the_single_pass_value():
    samples = speechlike(9)
    whole = [KWeightedMeter.measure(samples, RATE)]
    cuts = [0, len(samples) // 3, 2 * len(samples) // 3, len(samples)]
    parts = [
        KWeightedMeter.measure(samples[a - min(a, RATE) : b], RATE, discard_frames=min(a, RATE), keep_frames=b - a)
        for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    ]
    assert KWeightedMeter.integrate(parts) == pytest.approx(KWeightedMeter.integrate(whole), abs=0.05)
    assert KWeightedMeter.true_peak_dbtp(parts) == KWeightedMeter.true_peak_dbtp(whole)
    assert KWeightedMeter.integrate([BlockMeasurement.empty()]) is None


@needs_ffmpeg
def test_parallel_mp3_pieces_splice_without_seams(tmp_path):
    samples = speechlike(40)
    master = tmp_path / "master.wav"
    sf.write(master, samples, RATE, subtype="FLOAT")
    frames = len(samples)
    pieces = ParallelMp3.plan(frames, piece_seconds=12)
    assert len(pieces) == 4 and pieces[-1].end >= frames
    deadline = (datetime.now(UTC) + timedelta(minutes=5)).timestamp()
    kept = WorkerPool.run(
        ParallelMp3.encode_piece,
        [EncodeTask(str(master), str(tmp_path), deadline, piece, 1, frames, 0.0) for piece in pieces],
        guard_for(tmp_path),
        workers=1,
    )
    spliced = tmp_path / "spliced.mp3"
    assert ParallelMp3.splice([path for _, path in sorted(kept)], spliced) == -(-frames // FRAME_SAMPLES)
    single = tmp_path / "single.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(master), "-c:a", "libmp3lame", "-b:a", "128k", "-reservoir", "0",
         "-write_xing", "0", "-id3v2_version", "0", str(single)],
        check=True,
    )
    a, _ = sf.read(spliced, dtype="float32")
    b, _ = sf.read(single, dtype="float32")
    # Spliced frames sit on the absolute grid (decoder delay only); the single pass
    # also carries LAME's 576-sample encoder delay.
    a, b = a[529:], b[529 + 576 :]
    n = min(len(a), len(b), frames - 1200)
    diff = np.abs(a[:n] - b[:n])
    near = np.zeros(n, dtype=bool)
    for piece in pieces[1:]:
        near[max(0, piece.start - 2400) : piece.start + 2400] = True
    codec_noise = np.sqrt(np.mean((b[:n] - samples[:n, 0]) ** 2))
    assert np.sqrt(np.mean(diff[near] ** 2)) < codec_noise
    assert diff[near].max() <= max(diff[~near].max(), 1e-3) * 1.5


def test_chunk_plan_covers_the_file_once_with_handles():
    frames = 17 * 60 * RATE + 123
    chunks = ChunkPlanner.plan(frames, 300 * RATE)
    assert len(chunks) == 3
    assert chunks[0][0] == 0 and chunks[-1][1] == frames
    assert all(a[1] == b[0] for a, b in zip(chunks[:-1], chunks[1:], strict=True))
    assert chunks[1][2] == chunks[1][0] - HANDLE_FRAMES and chunks[1][3] == chunks[1][1] + HANDLE_FRAMES
    assert ChunkPlanner.plan(90 * RATE, 300 * RATE) == [(0, 90 * RATE, 0, 90 * RATE)]


def test_stitching_identical_chunks_reproduces_the_signal(tmp_path):
    samples = speechlike(20, channels=2)
    frames = len(samples)
    results = []
    for index, (start, end, left, right) in enumerate(ChunkPlanner.plan(frames, 7 * RATE)):
        path = tmp_path / f"chunk-{index}.wav"
        sf.write(path, samples[left:right], RATE, subtype="FLOAT")
        empty = BlockEnergies(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, dtype=np.int64))
        results.append(ChunkResult(index, start, end, left, str(path), empty, 10, 0, BlockMeasurement.empty(), BlockMeasurement.empty()))
    target = tmp_path / "stitched.wav"
    assert ParallelCleaner.stitch(results, target, 2, guard_for(tmp_path)) == frames
    stitched, _ = sf.read(target, dtype="float32", always_2d=True)
    assert stitched.shape == samples.shape
    assert np.abs(stitched - samples).max() < 1e-6
    assert ParallelCleaner.speech_report(results)["status"] == "passed"


def test_batched_block_groups_keep_blocks_in_order_and_equal_length():
    from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterSession

    session = DeepFilterSession.__new__(DeepFilterSession)
    session.policy = ContextualPolicy(480_000, 48_000)
    groups = session._block_groups(480_000 * 20 + 1000)
    flat = [block for group in groups for block in group]
    assert [b[0] for b in flat] == [i * 480_000 for i in range(21)]
    assert all(len(group) <= DeepFilterSession.BATCH_BLOCKS for group in groups)
    assert all(len({b[3] - b[2] for b in group}) == 1 for group in groups)
    assert len(groups[0]) == 1 and len(groups[-1]) == 1  # first/last have shorter context


def test_parameters_on_meta_loads_weights_without_a_cpu_copy():
    import torch

    from hear.inference.fish_speech import FishLoaderPatches

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
            self.register_buffer("scale", torch.arange(3.0), persistent=False)

        def forward(self, value):
            return self.linear(value)

    with FishLoaderPatches.parameters_on_meta():
        model = Tiny()
    assert model.linear.weight.device.type == "meta" and model.scale.device.type == "cpu"
    weights = {"linear.weight": torch.ones(3, 4), "linear.bias": torch.zeros(3)}
    model.load_state_dict(weights, assign=True)
    assert model.linear.weight.device.type == "cpu"
    assert torch.equal(model(torch.ones(1, 4)), torch.full((1, 3), 4.0))
    assert torch.nn.Linear(2, 2).weight.device.type == "cpu"  # patch is undone


def test_precomputed_segments_stand_in_for_the_pipeline_vad():
    from hear.inference.qwen_asr import LocalSileroVad, PrecomputedSegments

    stub = PrecomputedSegments([(0.0, 4.0), (4.5, 20.0), (21.0, 40.0)])
    found = stub({"waveform": np.zeros(10), "sample_rate": 16000})
    merged = stub.merge_chunks(found, 30, onset=0.5, offset=0.3)
    assert merged == LocalSileroVad.merge_chunks(found, 30, 0.5, 0.3)
    assert [(m["start"], m["end"]) for m in merged] == [(0.0, 20.0), (21.0, 40.0)]


def test_safetensors_on_device_redirects_upstream_cpu_loads(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    import torch

    from hear.inference.fish_speech import FishLoaderPatches

    path = tmp_path / "w.safetensors"
    safetensors_torch.save_file({"a": torch.ones(2)}, str(path))
    with FishLoaderPatches.safetensors_on_device("cpu"):
        # Upstream Fish calls load_file(str(shard), device="cpu"); the patch must accept that.
        loaded = safetensors_torch.load_file(str(path), device="cpu")
    assert torch.equal(loaded["a"], torch.ones(2))
    assert safetensors_torch.load_file(str(path))["a"].device.type == "cpu"
