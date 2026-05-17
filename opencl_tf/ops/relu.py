"""Python wrappers for the ReLU ops."""

from __future__ import annotations

from .._library import raw_ops


def relu(x):
    """Elementwise max(0, x) executed on the OpenCL backend."""
    return raw_ops.opencl_relu(x)


def relu_grad(gradients, features):
    """Gradient of relu wrt features. `features` is the original forward input."""
    return raw_ops.opencl_relu_grad(gradients, features)
