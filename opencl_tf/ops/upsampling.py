"""Python wrappers for the UpSampling2D bilinear ops."""

from __future__ import annotations

from typing import List

import tensorflow as tf

from .._library import raw_ops


def upsampling_bilinear_2d(x, size: List[int]):
    """Bilinear 2x upsampling on the OpenCL backend.

    Args:
        x:    Input tensor [N, H, W, C].
        size: [sy, sx] integer scale factors.
    Returns:
        y: [N, H*sy, W*sx, C]
    """
    return raw_ops.opencl_upsampling_bilinear2d(x, size=list(size))


def upsampling_bilinear_2d_grad(grad_out, input_sizes, size: List[int]):
    """Backward pass for bilinear upsampling.

    Args:
        grad_out:    Upstream gradient [N, Hout, Wout, C].
        input_sizes: 1-D int32 tensor [N, H, W, C] of the original input.
        size:        [sy, sx] scale factors (must match the forward call).
    Returns:
        grad_in: [N, H, W, C]
    """
    return raw_ops.opencl_upsampling_bilinear2d_grad(
        grad_out, tf.cast(input_sizes, tf.int32), size=list(size))
