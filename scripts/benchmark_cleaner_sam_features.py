"""Offline, CPU-only Ray probe of real codec layers through disk-backed features.

This tests individual convolutions, not the complete codec or GPU certification.
The source checkout and checkpoint must already be provisioned and pinned.
Successful process exit means diagnostics completed, not numerical acceptance:
inspect all tolerance-failure lists and per-case matches_full_tolerance fields.
"""

import argparse
import functools
import hashlib
import importlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import ray

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_codec_graph import SamCodecGraph
from hear.runtime.cleaner.sam_convolution import SamConvolutionGeometry
from hear.runtime.cleaner.sam_features import SamFeatureFile, SamFeatureRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class SamFeatureProbe:
    @staticmethod
    def _recurrent_backend_enter(stack, module, args):
        torch = importlib.import_module("torch")
        context = torch.backends.mkldnn.flags(enabled=True)
        context.__enter__()
        stack.append(context)

    @staticmethod
    def _recurrent_backend_exit(stack, module, args, output):
        if stack:
            stack.pop().__exit__(None, None, None)

    @staticmethod
    def _mutate_message(message, *args):
        message.zero_()

    @staticmethod
    def watermark_faults(codec, workspace):
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        results = []
        for scenario in (
            "message_freeze",
            "message_cancel",
            "decoder_early_cancel",
            "decoder_late_cancel",
            "decoder_invalid_message",
        ):
            with tempfile.TemporaryDirectory(prefix="watermark-fault-", dir=workspace) as directory:
                work = Path(directory)
                guard = ResourceGuard(
                    ResourceBudget(32_000_000, 32_000_000, 4096),
                    work,
                    time.monotonic() + 120,
                    threading.Event(),
                )
                is_message = scenario.startswith("message_")
                channels, frames = (128, 103) if is_message else (1024, 2)
                values = np.sin(np.arange(channels * frames, dtype=np.float32) * 0.017).reshape(
                    1, channels, frames
                )
                source = SamFeatureFile(
                    work / "input",
                    frames=frames,
                    batch=1,
                    channels=channels,
                    guard=guard,
                    create=True,
                )
                source.write(0, values)
                message = (torch.arange(16) % 2).float().reshape(1, 16)
                processor = codec.decoder.wm_model.msg_processor
                hook = None
                output = None
                try:
                    if scenario == "message_freeze":
                        with torch.inference_mode():
                            expected = processor(torch.from_numpy(values), message).numpy()
                        hook = processor.register_forward_hook(
                            functools.partial(SamFeatureProbe._mutate_message, message)
                        )
                    elif scenario == "decoder_invalid_message":
                        message[0, 0] = 2
                    else:
                        target = (
                            codec.decoder.model[0]
                            if scenario == "decoder_early_cancel"
                            else processor
                        )
                        hook = target.register_forward_hook(
                            lambda *args, cancelled=guard.cancelled: cancelled.set()
                        )
                    try:
                        if is_message:
                            output = SamFeatureRunner.message(
                                processor, message, source, work / "output", tile_frames=17
                            )
                        else:
                            output = SamCodecGraph(tile_frames=257).decode(
                                codec.decoder, source, message=message
                            )
                    except CleanExecutionError as error:
                        assert scenario.endswith("cancel") and error.code == ErrorCode.CANCELLED
                    except ValueError:
                        assert scenario == "decoder_invalid_message"
                    else:
                        assert scenario == "message_freeze"
                        np.testing.assert_array_equal(output.read(0, output.frames), expected)
                        assert not message.any()
                        output.close(remove=True)
                        output = None
                    assert set(work.iterdir()) == {source.path}
                    # Raw input bytes can be verified even after cancellation forbids new work.
                    raw = np.fromfile(source.path, dtype=np.float32).reshape(frames, 1, channels)
                    np.testing.assert_array_equal(raw, values.transpose(2, 0, 1))
                    results.append(
                        dict(
                            scenario=scenario,
                            passed=True,
                            input_preserved=True,
                            no_partial_outputs=True,
                        )
                    )
                finally:
                    if hook is not None:
                        hook.remove()
                    if output is not None:
                        output.close(remove=True)
                    source.close(remove=True)
        return results

    @staticmethod
    def rounding_probe(layer, values, geometry, actual, expected):
        """Diagnostic oracle uses the already normalized FP32 weights, cast to FP64."""
        torch = importlib.import_module("torch")
        np = importlib.import_module("numpy")
        if geometry.transposed or geometry.auto_padding:
            raise ValueError("rounding diagnostic only supports ordinary Conv1d")
        with torch.inference_mode():
            weight = layer.weight.detach().double()
            bias = layer.bias.detach().double() if layer.bias is not None else None
            inputs = torch.from_numpy(values).double()
            reference = torch.nn.functional.conv1d(
                inputs,
                weight,
                bias,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            ).numpy()
            tiled = np.empty_like(reference)
            for start in range(0, reference.shape[-1], 13):
                window = geometry.window(
                    values.shape[-1], start, min(reference.shape[-1], start + 13)
                )
                block = inputs[..., window.input_start : window.input_end].contiguous()
                result = torch.nn.functional.conv1d(
                    block,
                    weight,
                    bias,
                    layer.stride,
                    layer.padding,
                    layer.dilation,
                    layer.groups,
                ).numpy()
                tiled[..., start : window.output_end] = result[
                    ..., window.crop_start : window.crop_end
                ]
        return dict(
            full_fp32_vs_fp64_max_abs=float(np.abs(expected - reference).max()),
            tiled_fp32_vs_fp64_max_abs=float(np.abs(actual - reference).max()),
            tiled_fp64_vs_full_fp64_max_abs=float(np.abs(tiled - reference).max()),
            fp64_geometry_matches=bool(np.allclose(tiled, reference, atol=1e-12, rtol=1e-12)),
        )

    @staticmethod
    def probe(checkpoint_directory, codec_source, disable_mkldnn=False, mixed_cpu_backend=False):
        sys.path.insert(0, codec_source)
        dacvae = importlib.import_module("dacvae")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")

        torch.set_num_threads(1)
        if disable_mkldnn and mixed_cpu_backend:
            raise ValueError("CPU diagnostic backend options are mutually exclusive")
        if disable_mkldnn or mixed_cpu_backend:
            torch.backends.mkldnn.enabled = False
        root = Path(checkpoint_directory)
        cfg = json.loads((root / "config.json").read_text())["audio_codec"]
        keys = [
            "encoder_dim",
            "encoder_rates",
            "latent_dim",
            "decoder_dim",
            "decoder_rates",
            "n_codebooks",
            "codebook_size",
            "codebook_dim",
            "quantizer_dropout",
            "sample_rate",
        ]
        base = dacvae.DACVAE(**{key: cfg[key] for key in keys}).eval()
        codec = torch.nn.Module()
        codec.encoder, codec.quantizer, codec.decoder = base.encoder, base.quantizer, base.decoder
        del base
        state = torch.load(root / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
        weights = {
            k.removeprefix("audio_codec."): v
            for k, v in state.items()
            if k.startswith("audio_codec.")
        }
        loaded = len(weights)
        codec.load_state_dict(weights, strict=True)
        del weights, state
        codec.eval()
        # Diagnostic-only hooks in a single-use, CPU-only worker. Never serving configuration.
        backend_hooks = []
        backend_stack = []
        if mixed_cpu_backend:
            for layer in codec.modules():
                if isinstance(layer, torch.nn.LSTM):
                    backend_hooks.append(
                        layer.register_forward_pre_hook(
                            functools.partial(
                                SamFeatureProbe._recurrent_backend_enter, backend_stack
                            )
                        )
                    )
                    backend_hooks.append(
                        layer.register_forward_hook(
                            functools.partial(
                                SamFeatureProbe._recurrent_backend_exit, backend_stack
                            ),
                            always_call=True,
                        )
                    )
        cases, modules, maximum = 0, 0, 0.0
        failures = []
        rounding_diagnostics = []
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hear-sam-feature-probe.") as tmp:
            work = Path(tmp)
            guard = ResourceGuard(
                ResourceBudget(32_000_000, 32_000_000, 8192),
                work,
                time.monotonic() + 180,
                threading.Event(),
            )
            for name, layer in codec.named_modules():
                if not isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                    continue
                modules += 1
                transposed = isinstance(layer, torch.nn.ConvTranspose1d)
                geometry = SamConvolutionGeometry(
                    layer.kernel_size[0],
                    layer.stride[0],
                    layer.dilation[0],
                    layer.padding[0],
                    layer.output_padding[0] if transposed else 0,
                    transposed,
                    getattr(layer, "pad_mode", "none") == "auto",
                    getattr(layer, "causal", False),
                )
                for frames in (31, 32):
                    values = np.sin(
                        np.arange(layer.in_channels * frames, dtype=np.float32) * 0.017
                    ).reshape(1, layer.in_channels, frames)
                    source = SamFeatureFile(
                        work / "input",
                        frames=frames,
                        batch=1,
                        channels=layer.in_channels,
                        guard=guard,
                        create=True,
                    )
                    source.write(0, values)
                    with torch.inference_mode():
                        expected = layer(torch.from_numpy(values)).numpy()
                    output = SamFeatureRunner.convolution(
                        layer, source, work / "output", tile_frames=13
                    )
                    actual = output.read(0, output.frames)
                    difference = float(np.abs(actual - expected).max())
                    maximum = max(maximum, difference)
                    if name == "decoder.model.1.block.4.block.1" and frames == 32:
                        rounding_diagnostics.append(
                            SamFeatureProbe.rounding_probe(
                                layer,
                                values,
                                geometry,
                                actual,
                                expected,
                            )
                        )
                    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-5):
                        failures.append(
                            dict(
                                layer=name,
                                input_frames=frames,
                                maximum_absolute_difference=difference,
                            )
                        )
                    # Independently execute the same contiguous tile inputs without disk I/O.
                    for start in range(0, output.frames, 13):
                        window = geometry.window(frames, start, min(output.frames, start + 13))
                        block = np.array(
                            values[..., window.input_start : window.input_end], copy=True, order="C"
                        )
                        with torch.inference_mode():
                            reference = layer(torch.from_numpy(block)).numpy()[
                                ..., window.crop_start : window.crop_end
                            ]
                        np.testing.assert_array_equal(
                            actual[..., start : window.output_end], reference
                        )
                    reader = SamFeatureFile(
                        output.path,
                        frames=output.frames,
                        batch=1,
                        channels=output.channels,
                        guard=guard,
                    )
                    np.testing.assert_array_equal(reader.read(0, reader.frames), actual)
                    reader.close()
                    output.close(remove=True)
                    source.close(remove=True)
                    cases += 1
            activation_cases = []
            snake = importlib.import_module("dacvae.nn.layers").Snake1d
            for name, layer in codec.named_modules():
                if type(layer) not in (snake, torch.nn.ELU, torch.nn.Tanh, torch.nn.Identity):
                    continue
                channels = layer.alpha.shape[1] if type(layer) is snake else 8
                values = np.sin(np.arange(2 * channels * 103, dtype=np.float32) * 0.017).reshape(
                    2, channels, 103
                )
                source = SamFeatureFile(
                    work / "input", frames=103, batch=2, channels=channels, guard=guard, create=True
                )
                source.write(0, values)
                with torch.inference_mode():
                    expected = layer(torch.from_numpy(values)).numpy()
                output = SamFeatureRunner.activation(layer, source, work / "output", tile_frames=17)
                actual = output.read(0, output.frames)
                activation_cases.append(
                    dict(
                        layer=name,
                        kind=type(layer).__name__,
                        matches_full_tolerance=bool(
                            np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
                        ),
                        maximum_absolute_difference=float(np.abs(actual - expected).max()),
                    )
                )
                output.close(remove=True)
                source.close(remove=True)
            graph_cases = []
            graph_layers = [("encoder", codec.encoder, 1, 3840)]
            graph_layers.extend(
                (f"decoder.model.{i}", layer, layer.block[1].in_channels, 31)
                for i, layer in enumerate(codec.decoder.model)
                if i > 0
            )
            for name, layer, channels, frames in graph_layers:
                values = np.sin(np.arange(channels * frames, dtype=np.float32) * 0.017).reshape(
                    1, channels, frames
                )
                source = SamFeatureFile(
                    work / "input",
                    frames=frames,
                    batch=1,
                    channels=channels,
                    guard=guard,
                    create=True,
                )
                source.write(0, values)
                with torch.inference_mode():
                    expected = layer(torch.from_numpy(values)).numpy()
                output = SamCodecGraph(tile_frames=257).run(layer, source)
                actual = output.read(0, output.frames)
                assert actual.shape == expected.shape
                assert set(work.iterdir()) == {source.path, output.path}
                graph_cases.append(
                    dict(
                        layer=name,
                        input_frames=frames,
                        tile_frames=257,
                        output_shape=list(actual.shape),
                        matches_full_tolerance=bool(
                            np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
                        ),
                        maximum_absolute_difference=float(np.abs(actual - expected).max()),
                    )
                )
                output.close(remove=True)
                source.close(remove=True)
            # Complete original decoder, including fixed-message watermarking.
            values = np.sin(np.arange(2048, dtype=np.float32) * 0.017).reshape(1, 1024, 2)
            source = SamFeatureFile(
                work / "input", frames=2, batch=1, channels=1024, guard=guard, create=True
            )
            source.write(0, values)
            message = (torch.arange(16) % 2).float().reshape(1, 16)
            with torch.inference_mode():
                expected = codec.decoder(torch.from_numpy(values), message=message).numpy()
            output = SamCodecGraph(tile_frames=257).decode(codec.decoder, source, message=message)
            actual = output.read(0, output.frames)
            assert actual.shape == expected.shape
            assert set(work.iterdir()) == {source.path, output.path}
            decoder_case = dict(
                input_shape=list(values.shape),
                output_shape=list(actual.shape),
                message=message.tolist(),
                tile_frames=257,
                matches_full_tolerance=bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-5)),
                maximum_absolute_difference=float(np.abs(actual - expected).max()),
                intermediate_cleanup=True,
            )
            output.close(remove=True)
            source.close(remove=True)
            watermark_fault_cases = SamFeatureProbe.watermark_faults(codec, work)
            joint_values = np.sin(np.arange(512, dtype=np.float32) * 0.017).reshape(1, 256, 2)
            joint = SamFeatureFile(
                work / "joint", frames=2, batch=1, channels=256, guard=guard, create=True
            )
            joint.write(0, joint_values)
            paired_message = torch.stack((message[0], 1 - message[0]))
            with torch.inference_mode():
                native_pair = torch.from_numpy(joint_values).reshape(2, 128, 2)
                expected_pair = codec.decoder(
                    codec.quantizer.out_proj(native_pair), message=paired_message
                )[..., :3839].numpy()
            paired_output = SamCodecGraph(tile_frames=257).decode_joint(
                codec, joint, frames=3839, message=paired_message
            )
            actual_pair = paired_output.read(0, paired_output.frames)
            assert actual_pair.shape == expected_pair.shape == (2, 1, 3839)
            assert set(work.iterdir()) == {joint.path, paired_output.path}
            joint_case = dict(
                frames=3839,
                output_shape=list(actual_pair.shape),
                stream_order=["target", "residual"],
                messages=paired_message.tolist(),
                maximum_absolute_difference=float(np.abs(actual_pair - expected_pair).max()),
                matches_full_tolerance=bool(
                    np.allclose(actual_pair, expected_pair, atol=1e-6, rtol=1e-5)
                ),
                intermediate_cleanup=True,
            )
            paired_output.close(remove=True)
            joint.close(remove=True)
            roundtrip_cases = []
            for frames in (3840, 3841):
                values = (
                    0.01 * np.sin(np.arange(frames, dtype=np.float32) * (2 * np.pi * 440 / 48000))
                ).reshape(1, 1, frames)
                source = SamFeatureFile(
                    work / "input", frames=frames, batch=1, channels=1, guard=guard, create=True
                )
                source.write(0, values)
                with torch.inference_mode():
                    padded = torch.nn.functional.pad(
                        torch.from_numpy(values), (0, (-frames) % 1920), mode="reflect"
                    )
                    expected_mean, _ = codec.quantizer.in_proj(codec.encoder(padded)).chunk(
                        2, dim=1
                    )
                    expected = codec.decoder(
                        codec.quantizer.out_proj(expected_mean), message=message
                    )[..., :frames].numpy()
                    # Independent original default-backend reference; never relabel a
                    # same-policy match as parity with the original backend.
                    with torch.backends.mkldnn.flags(enabled=True):
                        default_mean, _ = codec.quantizer.in_proj(codec.encoder(padded)).chunk(
                            2, dim=1
                        )
                        default_waveform = codec.decoder(
                            codec.quantizer.out_proj(default_mean), message=message
                        )[..., :frames].numpy()
                graph = SamCodecGraph(tile_frames=257)
                mean = graph.encode_mean(codec, source)
                actual_mean = mean.read(0, mean.frames)
                output = graph.decode_latents(codec, mean, frames=frames, message=message)
                actual = output.read(0, output.frames)
                assert actual.shape == expected.shape == values.shape
                assert set(work.iterdir()) == {source.path, mean.path, output.path}
                roundtrip_cases.append(
                    dict(
                        frames=frames,
                        padded_frames=frames + (-frames) % 1920,
                        latent_shape=list(actual_mean.shape),
                        mean_matches_original_default=bool(
                            np.allclose(actual_mean, default_mean.numpy(), atol=1e-6, rtol=1e-5)
                        ),
                        mean_vs_original_default_max_abs=float(
                            np.abs(actual_mean - default_mean.numpy()).max()
                        ),
                        waveform_matches_original_default=bool(
                            np.allclose(actual, default_waveform, atol=1e-6, rtol=1e-5)
                        ),
                        waveform_vs_original_default_max_abs=float(
                            np.abs(actual - default_waveform).max()
                        ),
                        mean_matches_tolerance=bool(
                            np.allclose(actual_mean, expected_mean.numpy(), atol=1e-6, rtol=1e-5)
                        ),
                        mean_maximum_absolute_difference=float(
                            np.abs(actual_mean - expected_mean.numpy()).max()
                        ),
                        waveform_matches_tolerance=bool(
                            np.allclose(actual, expected, atol=1e-6, rtol=1e-5)
                        ),
                        waveform_maximum_absolute_difference=float(np.abs(actual - expected).max()),
                        final_sample=float(actual[0, 0, -1]),
                        intermediate_cleanup=True,
                    )
                )
                output.close(remove=True)
                mean.close(remove=True)
                source.close(remove=True)
            recurrent_cases = []
            for name, layer in codec.named_modules():
                if not isinstance(layer, torch.nn.LSTM):
                    continue
                values = np.sin(
                    np.arange(2 * layer.input_size * 103, dtype=np.float32) * 0.017
                ).reshape(2, layer.input_size, 103)
                with torch.inference_mode():
                    sequence = torch.from_numpy(values).permute(2, 0, 1)
                    full, _ = layer(sequence)
                    expected = (full + sequence).permute(1, 2, 0).numpy()
                for tile in (8, 17, 64):
                    source = SamFeatureFile(
                        work / "input",
                        frames=103,
                        batch=2,
                        channels=layer.input_size,
                        guard=guard,
                        create=True,
                    )
                    source.write(0, values)
                    output = SamFeatureRunner.recurrent(
                        layer, source, work / "output", tile_frames=tile
                    )
                    actual = output.read(0, output.frames)
                    recurrent_cases.append(
                        dict(
                            layer=name,
                            matches_full_tolerance=bool(
                                np.allclose(actual, expected, atol=1e-6, rtol=1e-6)
                            ),
                            batch=2,
                            frames=103,
                            tile_frames=tile,
                            maximum_absolute_difference=float(np.abs(actual - expected).max()),
                        )
                    )
                    output.close(remove=True)
                    source.close(remove=True)
            assert not list(work.iterdir())
        assert not torch.cuda.is_initialized()
        assert not backend_stack
        for hook in backend_hooks:
            hook.remove()
        feature_module = importlib.import_module("hear.runtime.cleaner.sam_features")

        paths = [
            Path(feature_module.__file__),
            Path(feature_module.__file__).with_name("mapped_residency.py"),
            Path(feature_module.__file__).with_name("sam_convolution.py"),
            Path(feature_module.__file__).with_name("sam_recurrent.py"),
            Path(feature_module.__file__).with_name("sam_codec_graph.py"),
        ]
        return dict(
            strict_codec_keys=loaded,
            convolution_modules=modules,
            cases=cases,
            input_frames=[31, 32],
            batch=1,
            tile_frames=13,
            atol=1e-6,
            rtol=1e-5,
            maximum_absolute_difference=maximum,
            full_sequence_tolerance_failures=failures,
            rounding_diagnostics=rounding_diagnostics,
            mkldnn_enabled=torch.backends.mkldnn.enabled,
            mixed_cpu_backend=mixed_cpu_backend,
            recurrent_cases=recurrent_cases,
            activation_cases=activation_cases,
            graph_cases=graph_cases,
            decoder_case=decoder_case,
            watermark_fault_cases=watermark_fault_cases,
            roundtrip_cases=roundtrip_cases,
            joint_case=joint_case,
            contiguous_tile_reference_exact=True,
            reopened_bytes_match=True,
            scratch_cleanup=True,
            elapsed_seconds=time.monotonic() - started,
            cuda_initialized=False,
            torch=torch.__version__,
            numpy=np.__version__,
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", required=True)
    parser.add_argument("--codec-source", required=True)
    parser.add_argument(
        "--disable-mkldnn",
        action="store_true",
        help="Diagnostic only: disable MKLDNN in the isolated CPU worker",
    )
    parser.add_argument(
        "--mixed-cpu-backend",
        action="store_true",
        help="Diagnostic only: disable MKLDNN except during LSTM calls",
    )
    args = parser.parse_args()
    ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
    try:
        task = ray.remote(num_cpus=1, num_gpus=0, memory=3_000_000_000, max_retries=0, max_calls=1)(
            SamFeatureProbe.probe
        )
        print(
            json.dumps(
                ray.get(
                    task.remote(
                        args.checkpoint_directory,
                        args.codec_source,
                        args.disable_mkldnn,
                        args.mixed_cpu_backend,
                    )
                )
            ),
            flush=True,
        )
    finally:
        ray.shutdown()
