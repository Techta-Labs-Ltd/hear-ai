"""Offline CPU core probe; extracts unchanged pinned definitions, not production loading."""

import argparse
import ast
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Optional

import ray

from hear.runtime.cleaner.longform_sam import SolverPolicy, WindowedMidpointSolver
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_checkpoint import SamCheckpointLoader
from hear.runtime.cleaner.sam_codec_loader import SamCodecBuilder
from hear.runtime.cleaner.sam_conditioning import AudioOnlySamForward, SamConditionedField
from hear.runtime.cleaner.sam_core_loader import SamCoreBuilder
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamFactory
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import CleanPlan
from hear.services.magic_clean.engines.sam_audio import SamEngine


class SamCoreProbe:
    @staticmethod
    def factory_case(source_directory, checkpoint, codec_source, text, mask, expected_hash):
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
        prompt = SamPromptIdentity(
            hashlib.sha256(b"speech").hexdigest(),
            "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
            hashlib.sha256(b"engineering-t5-cpu-float32-v1").hexdigest(),
            "2ae37fee196dd64b3fc2229fdbb56d5d454f43bda5606c4b83ec15700e35bf12",
            hashlib.sha256(b"\x01\x01").hexdigest(),
            2,
        )
        cache = SamPromptCache(((prompt, text, mask),))
        assets = PinnedSamAssets(
            Path(source_directory),
            Path(codec_source),
            checkpoint / "config.json",
            checkpoint / "checkpoint.pt",
            Path(__file__).resolve().parents[1] / "deploy/cleaner/sam-small-optional-keys.json",
        )
        factory = PinnedSamFactory(assets, prompt, cache, codec_tile_frames=257)
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
        with tempfile.TemporaryDirectory(prefix="hear-sam-factory-") as directory:
            work = Path(directory)
            guard = ResourceGuard(
                ResourceBudget(32_000_000, 32_000_000, 8192),
                work,
                time.monotonic() + 600,
                threading.Event(),
            )
            source, output = work / "input.wav", work / "target.wav"
            values = 0.01 * np.sin(np.arange(3841, dtype=np.float32) * (2 * np.pi * 440 / 48000))
            sf.write(source, values, 48000, subtype="FLOAT")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            before = {name for name in sys.modules if name.startswith("_hear_sam_core_")}
            session = engine.open_session(plan, guard)
            try:
                session.process(source, output, plan, guard)
            finally:
                session.close()
            actual, rate = sf.read(output, dtype="float32")
            digest = hashlib.sha256(actual.tobytes()).hexdigest()
            assert actual.shape == (3841,) and rate == 48000 and np.isfinite(actual).all()
            assert digest == expected_hash
            assert source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
            assert set(work.iterdir()) == {source, output}
            assert {name for name in sys.modules if name.startswith("_hear_sam_core_")} == before
            assert not any(name == "dacvae" or name.startswith("dacvae.") for name in sys.modules)
            assert cache.get(prompt)[0].shape == (1, 2, 768)
            assert not torch.cuda.is_initialized()
            cache.close()
            return dict(
                runtime=plan.runtime.model_dump(),
                target_sha256=digest,
                matches_pipeline_exactly=True,
                input_unchanged=True,
                intermediate_cleanup=True,
                session_namespace_cleanup=True,
                borrowed_cache_preserved=True,
                cuda_initialized=False,
            )

    @staticmethod
    def pipeline_case(core, reference, config, checkpoint, codec_source, text, mask):
        with tempfile.TemporaryDirectory(prefix="hear-sam-codec-build-") as directory:
            guard = ResourceGuard(
                ResourceBudget(1000000, 1000000, 1000),
                Path(directory),
                time.monotonic() + 600,
                threading.Event(),
            )
            built = SamCodecBuilder.build(Path(codec_source), checkpoint / "config.json", guard)
            try:
                result = SamCoreProbe._pipeline_case(
                    core, reference, checkpoint, text, mask, built.codec
                )
            finally:
                built.close()
            assert not any(name == "dacvae" or name.startswith("dacvae.") for name in sys.modules)
            result["codec_namespace_cleanup"] = True
            return result

    @staticmethod
    def _pipeline_case(core, reference, checkpoint, text, mask, codec):
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
        assert all(value.device.type == "meta" for value in codec.state_dict().values())
        manifest_path = (
            Path(__file__).resolve().parents[1] / "deploy/cleaner/sam-small-optional-keys.json"
        )
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        assert manifest["schema"] == "sam-optional-keys-v1"
        assert len(manifest["excluded_keys"]) == len(set(manifest["excluded_keys"])) == 601
        with tempfile.TemporaryDirectory(prefix="hear-sam-load-") as directory:
            loading_guard = ResourceGuard(
                ResourceBudget(32_000_000, 32_000_000, 8192),
                Path(directory),
                time.monotonic() + 600,
                threading.Event(),
            )
            loaded_core, loaded = SamCheckpointLoader.load(
                checkpoint / "checkpoint.pt",
                sha256=manifest["checkpoint_sha256"],
                core=core,
                codec=codec,
                optional_keys=frozenset(manifest["excluded_keys"]),
                guard=loading_guard,
            )
        assert loaded_core == 247 and loaded == 317
        frames = 3841
        values = (
            0.01 * np.sin(np.arange(frames, dtype=np.float32) * (2 * np.pi * 440 / 48000))
        ).reshape(1, 1, frames)
        # Independently reproduce the pinned stream contract for reference inputs.
        initial = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([42, 0x48454152, 1]))
        ).standard_normal((3, 256), dtype=np.float32)
        message = torch.from_numpy(
            np.random.Generator(np.random.PCG64(np.random.SeedSequence([42, 0x48454152, 2])))
            .integers(0, 2, size=(2, 16), dtype=np.int64)
            .astype(np.float32)
        )
        with torch.inference_mode():
            padded = torch.nn.functional.pad(torch.from_numpy(values), (0, 1919), mode="reflect")
            mean, _ = codec.quantizer.in_proj(codec.encoder(padded)).chunk(2, dim=1)
        # A single full-context window with unit blending is the direct midpoint reference.
        expected_state = SamCoreProbe.reference_midpoint(
            reference, mean.transpose(1, 2), text, mask, initial, SolverPolicy(250, 1, 16)
        )
        with torch.inference_mode():
            generated = torch.from_numpy(expected_state.T.copy()).reshape(2, 128, 3)
            expected = codec.decoder(codec.quantizer.out_proj(generated), message=message)[
                ..., :frames
            ].numpy()
        with tempfile.TemporaryDirectory(prefix="hear-sam-pipeline-") as directory:
            work = Path(directory)
            guard = ResourceGuard(
                ResourceBudget(32_000_000, 32_000_000, 8192),
                work,
                time.monotonic() + 600,
                threading.Event(),
            )
            audio = SamFeatureFile(
                work / "audio", frames=frames, batch=1, channels=1, guard=guard, create=True
            )
            audio.write(0, values)
            noise = SamFeatureFile(
                work / "noise", frames=3, batch=1, channels=256, guard=guard, create=True
            )
            noise.write(0, initial.T[None])
            output = None
            try:
                output = SamSeparationPipeline(core, codec, codec_tile_frames=257).separate(
                    audio, noise, text=text, text_mask=mask, message=message
                )
                actual = output.read(0, output.frames)
                assert actual.shape == expected.shape == (2, 1, frames)
                assert set(work.iterdir()) == {audio.path, noise.path, output.path}
                np.testing.assert_array_equal(audio.read(0, frames), values)
                np.testing.assert_array_equal(noise.read(0, 3), initial.T[None])
                pcm_source, pcm_target = work / "input.wav", work / "target.wav"
                sf.write(pcm_source, values[0, 0], 48000, subtype="FLOAT")
                source_hash = hashlib.sha256(pcm_source.read_bytes()).hexdigest()
                identity = SamSeparationPipeline(core, codec, codec_tile_frames=257).separate_file(
                    pcm_source, pcm_target, seed=42, text=text, text_mask=mask, guard=guard
                )
                with sf.SoundFile(pcm_target) as exported:
                    assert (exported.frames, exported.channels, exported.samplerate) == (
                        frames,
                        1,
                        48000,
                    )
                    assert (exported.format, exported.subtype) == ("RF64", "FLOAT")
                    file_actual = exported.read(dtype="float32")
                assert hashlib.sha256(pcm_source.read_bytes()).hexdigest() == source_hash
                assert set(work.iterdir()) == {
                    audio.path,
                    noise.path,
                    output.path,
                    pcm_source,
                    pcm_target,
                }
                np.testing.assert_array_equal(file_actual, actual[0, 0])
                return dict(
                    codec_builder="SamCodecBuilder",
                    initial_codec_checkpoint_tensors_meta=True,
                    checkpoint_loader="SamCheckpointLoader",
                    optional_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    excluded_vision_keys=len(manifest["excluded_keys"]),
                    seed=42,
                    noise_identity=identity,
                    file_target_maximum_absolute_difference=float(
                        np.abs(file_actual - expected[0, 0]).max()
                    ),
                    file_target_matches_reference_tolerance=bool(
                        np.allclose(file_actual, expected[0, 0], atol=1e-6, rtol=1e-5)
                    ),
                    file_target_matches_feature_pipeline_exactly=True,
                    file_target_sha256=hashlib.sha256(file_actual.tobytes()).hexdigest(),
                    file_source_unchanged=True,
                    file_intermediate_cleanup=True,
                    strict_codec_keys=loaded,
                    frames=frames,
                    latent_frames=3,
                    steps=16,
                    output_shape=list(actual.shape),
                    stream_order=["target", "residual"],
                    maximum_absolute_difference=float(np.abs(actual - expected).max()),
                    matches_reference_tolerance=bool(
                        np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
                    ),
                    finite=bool(np.isfinite(actual).all()),
                    borrowed_inputs_unchanged=True,
                    intermediate_cleanup=True,
                )
            finally:
                if output is not None:
                    output.close(remove=True)
                audio.close(remove=True)
                noise.close(remove=True)

    @staticmethod
    def reference_midpoint(reference, mean, text, mask, initial, policy):
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        current = initial.copy()
        step = 1.0 / policy.steps
        for index in range(policy.steps):
            stage = current
            for half in (0, 1):
                derivative = np.zeros_like(current)
                weights = np.zeros((len(current),), dtype=np.float32)
                for start in range(0, len(current), policy.window_frames - policy.overlap_frames):
                    end = min(len(current), start + policy.window_frames)
                    features = mean[:, start:end]
                    with torch.inference_mode():
                        prediction = reference(
                            torch.from_numpy(stage[start:end]).unsqueeze(0),
                            torch.cat((features, features), dim=2),
                            text,
                            torch.tensor([(index + half * 0.5) * step], dtype=torch.float32),
                            masked_video_features=torch.zeros(1, 1024, end - start),
                            text_mask=mask,
                        )[0].numpy()
                    edge = np.minimum(np.arange(end - start) + 1, np.arange(end - start, 0, -1))
                    blend = np.minimum(edge / policy.overlap_frames, 1).astype(np.float32)
                    derivative[start:end] += prediction * blend[:, None]
                    weights[start:end] += blend
                derivative /= weights[:, None]
                if half == 0:
                    stage = current + (step * 0.5) * derivative
                else:
                    current = current + step * derivative
        return current

    @staticmethod
    def definitions(path, names, namespace):
        tree = ast.parse(path.read_text())
        nodes = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
        ]
        if {node.name for node in nodes} != set(names):
            raise ValueError("pinned source definitions missing")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    @staticmethod
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def run(
        source_directory,
        checkpoint_directory,
        text_directory,
        steps=16,
        codec_source=None,
        pipeline_only=False,
    ):
        if steps not in (2, 16):
            raise ValueError("unsupported diagnostic step count")
        if pipeline_only and codec_source is None:
            raise ValueError("pipeline probe requires the pinned codec source")
        with tempfile.TemporaryDirectory(prefix="hear-sam-build-") as directory:
            guard = ResourceGuard(
                ResourceBudget(1000000, 1000000, 1000),
                Path(directory),
                time.monotonic() + 600,
                threading.Event(),
            )
            built = SamCoreBuilder.build(
                Path(source_directory), Path(checkpoint_directory) / "config.json", guard
            )
            try:
                return SamCoreProbe._run(
                    source_directory,
                    checkpoint_directory,
                    text_directory,
                    steps,
                    codec_source,
                    pipeline_only,
                    built.core,
                )
            finally:
                built.close()

    @staticmethod
    def _run(
        source_directory,
        checkpoint_directory,
        text_directory,
        steps,
        codec_source,
        pipeline_only,
        core,
    ):
        if steps not in (2, 16):
            raise ValueError("unsupported diagnostic step count")
        if pipeline_only and codec_source is None:
            raise ValueError("pipeline probe requires the pinned codec source")
        os.environ["HF_HUB_OFFLINE"] = "1"
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        transformers = importlib.import_module("transformers")
        torch.set_num_threads(1)
        source = Path(source_directory) / "sam_audio" / "model"
        root = Path(checkpoint_directory)
        config = json.loads((root / "config.json").read_text())
        assert all(value.device.type == "meta" for value in core.state_dict().values())
        assert all(value.device.type == "cpu" for value in core.buffers())
        namespace = {"torch": torch, "math": math, "Optional": Optional}
        state = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
        weights = {
            key: value
            for key, value in state.items()
            if not key.startswith(("audio_codec.", "vision_encoder."))
        }
        assert set(weights) == set(core.state_dict())
        core.load_state_dict(weights, strict=True, assign=True)
        loaded = len(weights)
        del state, weights
        core.eval()
        # Use original forward/align_inputs bodies as an independent assembly reference.
        tree = ast.parse((source / "model.py").read_text())
        original = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SAMAudio"
        )
        methods = [
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in ("forward", "align_inputs")
        ]
        assert len(methods) == 2
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(source / "model.py"), "exec"),
            namespace,
        )
        core.align_inputs = types.MethodType(namespace["align_inputs"], core)
        reference_forward = types.MethodType(namespace["forward"], core)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            text_directory, local_files_only=True
        )
        encoder = transformers.T5EncoderModel.from_pretrained(
            text_directory, local_files_only=True, use_safetensors=True
        ).eval()
        tokens = tokenizer(
            ["speech"], return_tensors="pt", max_length=512, truncation=True, padding="longest"
        )
        with torch.inference_mode():
            text = encoder(**tokens).last_hidden_state.detach().clone()
        mask = tokens["attention_mask"].bool()
        del encoder, tokenizer
        gc.collect()
        cases = []
        solver_case = None
        with tempfile.TemporaryDirectory(prefix="hear-sam-core-") as directory:
            guard = ResourceGuard(
                ResourceBudget(1000000, 1000000, 1000),
                Path(directory),
                time.monotonic() + 600,
                threading.Event(),
            )
            adapter = AudioOnlySamForward(core, guard)
            for frames, instant in ((2, 0.0), (7, 0.5), (13, 1.0)):
                noisy = torch.sin(torch.arange(frames * 256).float() * 0.017).reshape(
                    1, frames, 256
                )
                mean = torch.cos(torch.arange(frames * 128).float() * 0.013).reshape(1, frames, 128)
                time_value = torch.tensor([instant], dtype=torch.float32)
                with torch.inference_mode():
                    expected = reference_forward(
                        noisy,
                        torch.cat((mean, mean), dim=2),
                        text,
                        time_value,
                        masked_video_features=torch.zeros(1, 1024, frames),
                        text_mask=mask,
                    )
                actual = adapter.forward(noisy, mean, text, mask, time_value)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                cases.append(
                    dict(
                        frames=frames,
                        time=instant,
                        maximum_absolute_difference=float((actual - expected).abs().max()),
                        output_shape=list(actual.shape),
                        finite=bool(actual.isfinite().all()),
                    )
                )
            if not pipeline_only:
                mean = torch.cos(torch.arange(7 * 128).float() * 0.013).reshape(1, 7, 128)
                mean_file = SamFeatureFile(
                    Path(directory) / "mean",
                    frames=7,
                    batch=1,
                    channels=128,
                    guard=guard,
                    create=True,
                )
                mean_file.write(0, mean.transpose(1, 2).numpy())
                initial = np.sin(np.arange(7 * 256, dtype=np.float32) * 0.017).reshape(7, 256)
                noise = SamFeatureFile(
                    Path(directory) / "noise",
                    frames=7,
                    batch=1,
                    channels=256,
                    guard=guard,
                    create=True,
                )
                noise.write(0, initial.T[None])
                field = SamConditionedField(adapter, mean_file, text, mask)
                policy = SolverPolicy(window_frames=5, overlap_frames=2, steps=steps)
                destination = Path(directory) / "solved"
                try:
                    WindowedMidpointSolver(policy).solve(
                        noise.path, destination, frames=7, channels=256, field=field, guard=guard
                    )
                    actual = np.fromfile(destination, dtype=np.float32).reshape(7, 256)
                    expected = SamCoreProbe.reference_midpoint(
                        reference_forward, mean, text, mask, initial, policy
                    )
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
                    assert set(Path(directory).iterdir()) == {
                        mean_file.path,
                        noise.path,
                        destination,
                    }
                    solver_case = dict(
                        frames=7,
                        channels=256,
                        window_frames=5,
                        overlap_frames=2,
                        steps=steps,
                        native_evaluations=6 * steps,
                        partial_tail_frames=1,
                        maximum_absolute_difference=float(np.abs(actual - expected).max()),
                        finite=bool(np.isfinite(actual).all()),
                        intermediate_cleanup=True,
                    )
                finally:
                    field.close()
                    mean_file.close(remove=True)
                    noise.close(remove=True)
        assert not torch.cuda.is_initialized()
        pipeline_case = (
            SamCoreProbe.pipeline_case(
                core, reference_forward, config, root, codec_source, text, mask
            )
            if codec_source is not None
            else None
        )
        assert not torch.cuda.is_initialized()
        factory_case = (
            SamCoreProbe.factory_case(
                source_directory,
                root,
                codec_source,
                text,
                mask,
                pipeline_case["file_target_sha256"],
            )
            if pipeline_case is not None
            else None
        )
        return dict(
            factory_case=factory_case,
            core_builder="SamCoreBuilder",
            initial_core_checkpoint_tensors_meta=True,
            initial_core_nonpersistent_buffers_cpu=True,
            strict_core_keys=loaded,
            cases=cases,
            solver_case=solver_case,
            pipeline_case=pipeline_case,
            text_shape=list(text.shape),
            text_sha256=hashlib.sha256(text.numpy().tobytes()).hexdigest(),
            prompt="speech (engineering fixture, not approved preset)",
            cuda_initialized=False,
            optional_modules_imported=[
                name
                for name in sys.modules
                if name == "core"
                or name.startswith(("sam_audio.ranking", "core.audio_visual_encoder"))
            ],
            torch=torch.__version__,
            transformers=transformers.__version__,
            upstream_source_sha256={
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in (
                    "model.py",
                    "config.py",
                    "transformer.py",
                    "rope.py",
                    "patcher.py",
                    "align.py",
                )
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--checkpoint-directory", required=True)
    parser.add_argument("--text-directory", required=True)
    parser.add_argument("--steps", type=int, choices=(2, 16), default=16)
    parser.add_argument("--codec-source")
    parser.add_argument("--pipeline-only", action="store_true")
    args = parser.parse_args()
    ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    try:
        task = ray.remote(num_cpus=1, num_gpus=0, memory=8_000_000_000, max_retries=0, max_calls=1)(
            SamCoreProbe.run
        )
        print(
            json.dumps(
                ray.get(
                    task.remote(
                        args.source_directory,
                        args.checkpoint_directory,
                        args.text_directory,
                        args.steps,
                        args.codec_source,
                        args.pipeline_only,
                    )
                )
            ),
            flush=True,
        )
    finally:
        ray.shutdown()
