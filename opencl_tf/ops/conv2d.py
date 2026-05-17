"""Python wrappers for the three Conv2D ops."""

from __future__ import annotations

from typing import Sequence

from .._library import raw_ops


def conv2d(x, w, strides: Sequence[int] = (1, 1, 1, 1), padding: str = "SAME"):
    """Forward 2D convolution.

    Args:
        x: rank-4 NHWC float tensor [N, H, W, Cin].
        w: rank-4 float tensor [kH, kW, Cin, Cout].
        strides: NHWC strides; strides[0] and strides[3] must be 1.
        padding: "SAME" or "VALID".

    Returns:
        Output tensor [N, Hout, Wout, Cout].
    """
    return raw_ops.opencl_conv2d(x, w, strides=list(strides), padding=padding)


def conv2d_backprop_input(input_sizes, w, grad_out,
                          strides: Sequence[int], padding: str):
    """Gradient of conv2d wrt the input tensor.

    Args:
        input_sizes: int32 1-D tensor of length 4, the desired output shape.
        w: filter tensor [kH, kW, Cin, Cout].
        grad_out: upstream gradient with the forward op's output shape.
        strides, padding: must match the forward call.
    """
    return raw_ops.opencl_conv2d_backprop_input(
        input_sizes, w, grad_out, strides=list(strides), padding=padding)


def conv2d_backprop_filter(x, filter_sizes, grad_out,
                           strides: Sequence[int], padding: str):
    """Gradient of conv2d wrt the filter weights.

    Args:
        x: original input tensor used in the forward call.
        filter_sizes: int32 1-D tensor of length 4, the filter shape.
        grad_out: upstream gradient.
        strides, padding: must match the forward call.
    """
    return raw_ops.opencl_conv2d_backprop_filter(
        x, filter_sizes, grad_out, strides=list(strides), padding=padding)
