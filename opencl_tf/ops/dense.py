"""Python wrappers for the Dense (fully-connected) ops."""

from __future__ import annotations

import tensorflow as tf

from .._library import raw_ops


def dense(x, w, b):
    """y = x @ W + b  executed on the OpenCL backend.

    Args:
        x: [batch, in_features]
        w: [in_features, out_features]
        b: [out_features]
    Returns:
        y: [batch, out_features]
    """
    return raw_ops.opencl_dense(x, w, b)


def dense_backprop_input(grad_y, w):
    """grad_x = grad_y @ W^T.

    Args:
        grad_y: [batch, out_features]
        w:      [in_features, out_features]
    Returns:
        grad_x: [batch, in_features]
    """
    return raw_ops.opencl_dense_backprop_input(grad_y, w)


def dense_backprop_weight(x, grad_y):
    """grad_W = x^T @ grad_y.

    Args:
        x:      [batch, in_features]
        grad_y: [batch, out_features]
    Returns:
        grad_W: [in_features, out_features]
    """
    return raw_ops.opencl_dense_backprop_weight(x, grad_y)


def dense_backprop_bias(grad_y):
    """grad_b = sum_n grad_y[n, :].

    Args:
        grad_y: [batch, out_features]
    Returns:
        grad_b: [out_features]
    """
    return raw_ops.opencl_dense_backprop_bias(grad_y)
