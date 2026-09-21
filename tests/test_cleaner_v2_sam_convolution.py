import pytest
import torch

from hear.runtime.cleaner.sam_convolution import SamConvolutionGeometry


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "stride,kernel,dilation",
    [(1, 7, 1), (1, 7, 3), (2, 4, 1), (5, 10, 1), (8, 16, 1), (10, 20, 1), (12, 24, 1)],
)
@pytest.mark.parametrize("frames", [31, 32, 103])
def test_standard_convolution_windows_match_full_output(
    transposed, stride, kernel, dilation, frames
):
    padding = (stride + 1) // 2 if transposed else (kernel - stride) * dilation // 2
    output_padding = int(stride % 2) if transposed else 0
    # The actual codec does not use transpose stride 1 output_padding=1.
    if transposed and stride == 1:
        output_padding = 0
    geometry = SamConvolutionGeometry(kernel, stride, dilation, padding, output_padding, transposed)
    cls = torch.nn.ConvTranspose1d if transposed else torch.nn.Conv1d
    kwargs = {"output_padding": output_padding} if transposed else {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        layer = cls(
            2, 3, kernel, stride=stride, padding=padding, dilation=dilation, **kwargs
        ).eval()
    features = torch.sin(torch.arange(2 * frames).float()).reshape(1, 2, frames)
    with torch.inference_mode():
        expected = layer(features)
        assert expected.shape[-1] == geometry.output_frames(frames)
        for step in [1, 7, 19]:
            parts = []
            for start in range(0, expected.shape[-1], step):
                window = geometry.window(frames, start, min(start + step, expected.shape[-1]))
                chunk = layer(features[..., window.input_start : window.input_end])
                parts.append(chunk[..., window.crop_start : window.crop_end])
            torch.testing.assert_close(torch.cat(parts, dim=-1), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "configuration",
    [
        dict(kernel=0, stride=1),
        dict(kernel=3, stride=0),
        dict(kernel=3, stride=2, output_padding=2, transposed=True),
        dict(kernel=7, stride=1, dilation=3, auto_padding=True),
    ],
)
def test_unsupported_geometry_fails_explicitly(configuration):
    with pytest.raises(ValueError):
        SamConvolutionGeometry(**configuration)


def test_out_of_range_windows_fail():
    geometry = SamConvolutionGeometry(7, 1, padding=3)
    for start, end in [(-1, 2), (1, 1), (0, 11)]:
        with pytest.raises(ValueError):
            geometry.window(10, start, end)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("stride", [1, 2, 5, 8, 12])
@pytest.mark.parametrize("frames", [31, 32])
def test_auto_padding_keeps_global_phase(transposed, causal, stride, frames):
    kernel = 7 if stride == 1 else 2 * stride
    geometry = SamConvolutionGeometry(
        kernel, stride, transposed=transposed, auto_padding=True, causal=causal
    )
    weight = torch.sin(torch.arange(2 * 2 * kernel).float()).reshape(2, 2, kernel)
    features = torch.sin(torch.arange(2 * frames).float()).reshape(1, 2, frames)

    def original_forward(value):
        total = kernel - stride
        if transposed:
            value = torch.nn.functional.conv_transpose1d(value, weight, stride=stride)
            right = total if causal else total // 2
            left = total - right
            return value[..., left : value.shape[-1] - right]
        extra = (-value.shape[-1]) % stride
        if causal:
            padding = (total, extra)
        else:
            right = extra // 2
            padding = (total - right, right + extra)
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(value, padding), weight, stride=stride
        )

    full = original_forward(features)
    assert full.shape[-1] == geometry.output_frames(frames)
    parts = []
    for start in range(0, full.shape[-1], 7):
        window = geometry.window(frames, start, min(start + 7, full.shape[-1]))
        if not transposed:
            assert (window.input_end - window.input_start) % stride == frames % stride
        part = original_forward(features[..., window.input_start : window.input_end])
        parts.append(part[..., window.crop_start : window.crop_end])
    torch.testing.assert_close(torch.cat(parts, dim=-1), full, atol=1e-6, rtol=1e-6)


def test_hour_length_only_changes_offsets_not_tile_memory():
    geometry = SamConvolutionGeometry(24, 12, auto_padding=True, causal=True)
    frames = 172800017
    output_frames = geometry.output_frames(frames)
    for start in (0, output_frames // 2, output_frames - 4096):
        window = geometry.window(frames, start, start + 4096)
        assert window.input_end - window.input_start <= 4096 * 12 + 2 * 24
