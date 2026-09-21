import functools

import numpy as np
import pytest
import torch

from hear.runtime.cleaner.sam_convolution import SamConvolutionGeometry
from scripts.benchmark_cleaner_sam_core import SamCoreProbe
from scripts.benchmark_cleaner_sam_features import SamFeatureProbe


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_rounding_oracle_checks_geometry_with_fixed_effective_weights(stride):
    layer = torch.nn.Conv1d(2, 3, 5, stride=stride, padding=2).eval()
    values = np.sin(np.arange(206, dtype=np.float32)).reshape(1, 2, 103)
    with torch.inference_mode():
        expected = layer(torch.from_numpy(values)).numpy()
    geometry = SamConvolutionGeometry(5, stride, padding=2)
    result = SamFeatureProbe.rounding_probe(layer, values, geometry, expected, expected)
    assert result["fp64_geometry_matches"]
    assert result["tiled_fp64_vs_full_fp64_max_abs"] < 1e-12
    assert result["full_fp32_vs_fp64_max_abs"] == result["tiled_fp32_vs_fp64_max_abs"]


def test_rounding_oracle_cannot_silently_ignore_dynamic_padding():
    with pytest.raises(ValueError):
        SamFeatureProbe.rounding_probe(
            None, None, SamConvolutionGeometry(4, 2, auto_padding=True), None, None
        )


@pytest.mark.parametrize("steps", [0, 3, 256])
def test_core_probe_rejects_unsupported_steps_before_loading(steps):
    with pytest.raises(ValueError, match="unsupported diagnostic step count"):
        SamCoreProbe.run("missing-source", "missing-checkpoint", "missing-text", steps=steps)


@pytest.mark.parametrize("fail", [False, True])
def test_diagnostic_recurrent_backend_restored_after_native_call(fail):
    layer = torch.nn.LSTM(2, 2).eval()
    stack = []
    observed = []

    def inspect(module, args):
        observed.append(torch.backends.mkldnn.enabled)
        if fail:
            raise RuntimeError("diagnostic fixture failure")

    hooks = [
        layer.register_forward_pre_hook(
            functools.partial(SamFeatureProbe._recurrent_backend_enter, stack)
        ),
        layer.register_forward_hook(
            functools.partial(SamFeatureProbe._recurrent_backend_exit, stack), always_call=True
        ),
        layer.register_forward_pre_hook(inspect),
    ]
    original = torch.backends.mkldnn.enabled
    try:
        with torch.backends.mkldnn.flags(enabled=False), torch.inference_mode():
            if fail:
                with pytest.raises(RuntimeError, match="diagnostic fixture failure"):
                    layer(torch.zeros(4, 1, 2))
            else:
                layer(torch.zeros(4, 1, 2))
            assert not torch.backends.mkldnn.enabled
            assert not stack
        assert torch.backends.mkldnn.enabled == original
        assert observed == [True]
    finally:
        for hook in hooks:
            hook.remove()
