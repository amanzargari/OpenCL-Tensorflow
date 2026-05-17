"""Python wrappers for the Sigmoid ops."""

from __future__ import annotations

from .._library import raw_ops


def sigmoid(x):
    """Elementwise sigmoid(x) executed on the OpenCL backend."""
    return raw_ops.opencl_sigmoid(x)


def sigmoid_grad(y, dy):
    """Gradient of sigmoid wrt input x.

    Args:
        y:  The forward output (sigmoid(x)), NOT the original input.
        dy: Upstream gradient.
    Returns:
        grad_x = dy * y * (1 - y)
    """
    return raw_ops.opencl_sigmoid_grad(y, dy)
