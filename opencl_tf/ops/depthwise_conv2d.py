"""Python wrappers for the DepthwiseConv2D ops."""

from __future__ import annotations

from typing import Sequence

from .._library import raw_ops


def depthwise_conv2d(x, w,
                     strides: Sequence[int] = (1, 1, 1, 1),
                     padding: str = "SAME"):
    """Forward depthwise convolution.

    Args:
        x: NHWC float tensor [N, H, W, C].
        w: float tensor [kH, kW, C, depth_multiplier].
        strides: NHWC strides; strides[0] and strides[3] must be 1.
        padding: "SAME" or "VALID".
    """
    return raw_ops.opencl_depthwise_conv2d(
        x, w, strides=list(strides), padding=padding)


def depthwise_conv2d_backprop_input(input_sizes, w, grad_out,
                                    strides: Sequence[int], padding: str):
    return raw_ops.opencl_depthwise_conv2d_backprop_input(
        input_sizes, w, grad_out, strides=list(strides), padding=padding)


def depthwise_conv2d_backprop_filter(x, filter_sizes, grad_out,
                                     strides: Sequence[int], padding: str):
    return raw_ops.opencl_depthwise_conv2d_backprop_filter(
        x, filter_sizes, grad_out, strides=list(strides), padding=padding)
