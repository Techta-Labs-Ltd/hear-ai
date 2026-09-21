"""Run with isolated Python and an installed cleaner wheel, not repository imports."""

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import hear.runtime.cleaner.sam_loader as loader_module
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamFactory
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import CleanPlan
from hear.services.magic_clean.engines.sam_audio import SamEngine


class InstalledSamProbe:
    @staticmethod
    def run(args):
        if not sys.flags.isolated or "site-packages" not in Path(loader_module.__file__).parts:
            raise RuntimeError("use isolated Python with the installed cleaner wheel")
        if args.diagnose_cublas and args.device != "cuda:0":
            raise ValueError("cuBLAS diagnostics require CUDA")
        if args.frames != 3841 and args.device != "cuda:0":
            raise ValueError("extended engineering fixtures require CUDA")
        torch.set_num_threads(1)
        if args.device == "cuda:0":
            # This isolated diagnostic owns its process-wide precision settings.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
        fixture_path = Path(args.conditioning)
        if fixture_path.stat().st_size > 1048576:
            raise ValueError("conditioning fixture exceeds bound")
        if hashlib.sha256(fixture_path.read_bytes()).hexdigest() != args.conditioning_sha256:
            raise ValueError("conditioning fixture digest mismatch")
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        prompt = SamPromptIdentity(
            hashlib.sha256(b"speech").hexdigest(),
            "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
            hashlib.sha256(b"engineering-t5-cpu-float32-v1").hexdigest(),
            "2ae37fee196dd64b3fc2229fdbb56d5d454f43bda5606c4b83ec15700e35bf12",
            hashlib.sha256(b"\x01\x01").hexdigest(),
            2,
        )
        # Provision the trusted engineering fixture in the deployment byte format.
        # Only this diagnostic reads a weights-only Torch fixture; serving admission
        # reads raw bounded bytes and checks the independently pinned identities.
        with tempfile.TemporaryDirectory(prefix="sam-conditioning-") as directory:
            asset = Path(directory) / "conditioning.bin"
            payload = (
                fixture["text"].numpy().astype("<f4", copy=False).tobytes()
                + fixture["mask"].numpy().tobytes()
            )
            asset.write_bytes(payload)
            conditioning_digest = hashlib.sha256(payload).hexdigest()
            cache = SamPromptCache.from_files(((prompt, asset),))
        factory = PinnedSamFactory(
            PinnedSamAssets(
                Path(args.sam_source),
                Path(args.codec_source),
                Path(args.config),
                Path(args.checkpoint),
                Path(args.optional_manifest),
            ),
            prompt,
            cache,
            codec_tile_frames=args.codec_tile,
            device=args.device,
        )
        plan = CleanPlan(
            profile="voice_focus",
            profile_version="engineering-probe",
            catalogue_sha256=hashlib.sha256(b"engineering-only").hexdigest(),
            runtime=factory.identity,
            attenuation_limit_db=None,
            noise_reduction_db=None,
            noise_reference=None,
            prompt_sha256=prompt.prompt_sha256,
            channel_policy="mono",
            mono_acknowledged=True,
            adjust_loudness=False,
            match_comparison_loudness=True,
            shorten_pauses=False,
            seed=42,
        )
        engine = SamEngine(factory.identity, factory)
        with tempfile.TemporaryDirectory(prefix="sam-installed-") as directory:
            work = Path(directory)
            guard = ResourceGuard(
                # Include the codec's final 1,920-frame reflected padding block.
                ResourceBudget(2000000000, 32000000, max(8192, args.frames + 1920)),
                work,
                time.monotonic() + 600,
                threading.Event(),
            )
            source, output = work / "input.wav", work / "target.wav"
            values = 0.01 * np.sin(
                np.arange(args.frames, dtype=np.float32) * (2 * np.pi * 440 / 48000)
            )
            sf.write(source, values, 48000, subtype="FLOAT")
            original = hashlib.sha256(source.read_bytes()).hexdigest()
            reference = None
            memory = None
            if args.device == "cuda:0" and args.frames == 3841 and args.codec_tile == 257:
                cpu_factory = PinnedSamFactory(factory.assets, prompt, cache, codec_tile_frames=257)
                cpu_plan = plan.model_copy(update={"runtime": cpu_factory.identity})
                cpu_session = SamEngine(cpu_factory.identity, cpu_factory).open_session(
                    cpu_plan, guard
                )
                reference_path = work / "cpu.wav"
                try:
                    cpu_session.process(source, reference_path, cpu_plan, guard)
                finally:
                    cpu_session.close()
                reference, _ = sf.read(reference_path, dtype="float32")
                assert hashlib.sha256(reference.tobytes()).hexdigest() == (
                    "862bb0f8c69581effe20473b5156d374bf237155e269a44ded7ae694d3b8897c"
                )
                reference_path.unlink()
            if args.device == "cuda:0":
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(0)
            sessions = []
            for index in range(args.repeat):
                if index:
                    output.unlink()
                session = engine.open_session(plan, guard)
                try:
                    session.process(source, output, plan, guard)
                finally:
                    session.close()
                samples, _ = sf.read(output, dtype="float32")
                observation = {
                    "index": index,
                    "target_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                }
                if args.device == "cuda:0":
                    torch.cuda.synchronize(0)
                    observation.update(
                        allocated_after_close_bytes=torch.cuda.memory_allocated(0),
                        reserved_after_close_bytes=torch.cuda.memory_reserved(0),
                    )
                assert original == hashlib.sha256(source.read_bytes()).hexdigest()
                assert set(work.iterdir()) == {source, output}
                assert not any(
                    name == "dacvae" or name.startswith(("dacvae.", "_hear_sam_core_"))
                    for name in sys.modules
                )
                sessions.append(observation)
            if args.device == "cuda:0":
                torch.cuda.synchronize(0)
                memory = {
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                    "allocated_after_close_bytes": torch.cuda.memory_allocated(0),
                    "allocator_cap_bytes": guard.budget.allocator_cap_bytes,
                    "gpu": torch.cuda.get_device_name(0),
                }
                # Diagnose retention only after all sessions. Never mask growth by
                # forcing collection or emptying allocator caches between attempts.
                gc.collect()
                torch.cuda.synchronize(0)
                memory["allocated_after_gc_bytes"] = torch.cuda.memory_allocated(0)
                torch.cuda.empty_cache()
                memory["allocated_after_empty_cache_bytes"] = torch.cuda.memory_allocated(0)
                memory["reserved_after_empty_cache_bytes"] = torch.cuda.memory_reserved(0)
                if args.diagnose_cublas:
                    # Pinned-Torch private diagnostic only. Never used by serving
                    # cleanup and only after synchronized completion of all sessions.
                    torch._C._cuda_clearCublasWorkspaces()
                    torch.cuda.synchronize(0)
                    memory["allocated_after_cublas_clear_bytes"] = torch.cuda.memory_allocated(0)
                    torch.cuda.empty_cache()
                    memory["reserved_after_cublas_clear_bytes"] = torch.cuda.memory_reserved(0)
            actual, rate = sf.read(output, dtype="float32")
            digest = hashlib.sha256(actual.tobytes()).hexdigest()
            exact = digest == "862bb0f8c69581effe20473b5156d374bf237155e269a44ded7ae694d3b8897c"
            if args.device == "cpu" and args.codec_tile == 257:
                assert exact
            assert actual.shape == (args.frames,) and rate == 48000 and np.isfinite(actual).all()
            assert original == hashlib.sha256(source.read_bytes()).hexdigest()
            assert set(work.iterdir()) == {source, output}
            assert not any(
                name == "dacvae" or name.startswith(("dacvae.", "_hear_sam_core_"))
                for name in sys.modules
            )
            assert torch.cuda.is_initialized() == (args.device == "cuda:0")
            cache.close()
            return {
                "runtime": plan.runtime.model_dump(),
                "fixture_frames": args.frames,
                "codec_tile_frames": args.codec_tile,
                "latent_frames": (args.frames + 1919) // 1920,
                "target_sha256": digest,
                "module": loader_module.__file__,
                "isolated_python": True,
                "exact_known_answer": exact,
                "source_preserved": True,
                "scratch_and_namespace_cleanup": True,
                "cuda_initialized": torch.cuda.is_initialized(),
                "device": args.device,
                "torch_memory": memory,
                "sessions": sessions,
                "repeat_output_identical": len({item["target_sha256"] for item in sessions}) == 1,
                "cpu_comparison": None
                if reference is None
                else {
                    "atol": 1e-6,
                    "rtol": 1e-5,
                    "passes": bool(np.allclose(actual, reference, atol=1e-6, rtol=1e-5)),
                    "max_abs_error": float(np.max(np.abs(actual - reference))),
                },
                "conditioning_admission": "pinned-raw-file",
                "conditioning_asset_sha256": conditioning_digest,
                "limitations": (
                    "Short synthetic fixture; not aggregate NVML, long-form "
                    "or quality certification"
                ),
            }

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
        parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
        parser.add_argument("--diagnose-cublas", action="store_true")
        parser.add_argument("--frames", type=int, choices=(3841, 480001, 576001), default=3841)
        parser.add_argument("--codec-tile", type=int, choices=(257, 4096), default=257)
        for name in (
            "conditioning",
            "conditioning-sha256",
            "sam-source",
            "codec-source",
            "config",
            "checkpoint",
            "optional-manifest",
        ):
            parser.add_argument("--" + name, required=True)
        print(json.dumps(InstalledSamProbe.run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    InstalledSamProbe.main()
