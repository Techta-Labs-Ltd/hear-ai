"""Exact window geometry for the pinned DACVAE convolution wrappers.

Plans retain global stride/padding alignment while calling the original layer,
including its weight-normalization hooks. No weights or watermarking are removed.
Tensor I/O, scratch limits and execution supervision belong to the codec runner.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConvolutionWindow:
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    crop_start: int
    crop_end: int


@dataclass(frozen=True)
class SamConvolutionGeometry:
    kernel: int
    stride: int
    dilation: int = 1
    padding: int = 0
    output_padding: int = 0
    transposed: bool = False
    auto_padding: bool = False
    causal: bool = False

    def __post_init__(self):
        if (
            min(self.kernel, self.stride, self.dilation) < 1
            or min(self.padding, self.output_padding) < 0
            or self.output_padding >= self.stride
            or (not self.transposed and self.output_padding)
            or (self.auto_padding and (self.dilation != 1 or self.padding or self.output_padding))
            or (self.auto_padding and self.kernel < self.stride)
        ):
            raise ValueError("unsupported SAM convolution geometry")

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return -(-value // divisor)

    def _edges(self, frames: int) -> tuple[int, int]:
        if not self.auto_padding:
            return (self.padding, self.padding)
        total = self.kernel - self.stride
        if self.transposed:
            right = total if self.causal else total // 2
            return total - right, right
        extra = (-frames) % self.stride
        # Matches NormConv1d.pad at the pinned DACVAE revision, including
        # its asymmetric noncausal rule; not generic SAME-padding semantics.
        if self.causal:
            return total, extra
        right = extra // 2
        return total - right, right + extra

    def output_frames(self, input_frames: int) -> int:
        if input_frames < 1:
            raise ValueError("convolution input must be nonempty")
        left, right = self._edges(input_frames)
        effective = (self.kernel - 1) * self.dilation + 1
        if self.transposed:
            return (input_frames - 1) * self.stride - left - right + effective + self.output_padding
        return (input_frames + left + right - effective) // self.stride + 1

    def window(self, input_frames: int, start: int, end: int) -> ConvolutionWindow:
        total_output = self.output_frames(input_frames)
        if not 0 <= start < end <= total_output:
            raise ValueError("invalid convolution output interval")
        left, right = self._edges(input_frames)
        effective = (self.kernel - 1) * self.dilation + 1
        if self.transposed:
            first = max(0, self._ceil_div(start + left - effective + 1, self.stride))
            last = min(input_frames, (end - 1 + left) // self.stride + 1)
            offset = first * self.stride
            # Include enough input to retain the target after the wrapper's
            # right-edge unpadding, including bias-only output-padding samples.
            needed = (
                self._ceil_div(
                    end - offset + left + right - effective - self.output_padding, self.stride
                )
                + 1
            )
            last = min(input_frames, max(last, first + needed))
        else:
            first = max(0, ((start * self.stride - left) // self.stride) * self.stride)
            last = min(input_frames, (end - 1) * self.stride - left + effective)
            if self.auto_padding:
                # The layer sees the same length residue as the full sequence,
                # so its dynamic extra padding is identical on every window.
                last = min(input_frames, last + (input_frames - last) % self.stride)
            offset = first // self.stride
        if not 0 <= first < last <= input_frames:
            raise ValueError("unsupported convolution window support")
        crop_start, crop_end = start - offset, end - offset
        if not 0 <= crop_start < crop_end <= self.output_frames(last - first):
            raise ValueError("convolution window cannot cover output interval")
        return ConvolutionWindow(first, last, start, end, crop_start, crop_end)
