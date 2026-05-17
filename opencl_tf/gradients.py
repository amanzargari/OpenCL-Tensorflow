"""Gradient registrations for every custom op exported by `opencl_tf_ops.so`.

`tf.GradientTape` discovers gradient functions via `tf.RegisterGradient`
keyed on the op's registered name (NOT its Python wrapper name). The
keys here must match `REGISTER_OP(...)` strings in the C++ source.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import ops

from ._library import raw_ops


@ops.RegisterGradient("OpenclConv2d")
def _opencl_conv2d_grad(op, grad):
    """Gradient for OpenclConv2d.

    Forward signature : (input, filter) -> output
    Returns a list of gradients, one per *input* in order.
    """
    x = op.inputs[0]
    w = op.inputs[1]
    strides = list(op.get_attr("strides"))
    padding = op.get_attr("padding")

    grad_input = raw_ops.opencl_conv2d_backprop_input(
        tf.shape(x), w, grad,
        strides=strides, padding=padding,
    )
    grad_filter = raw_ops.opencl_conv2d_backprop_filter(
        x, tf.shape(w), grad,
        strides=strides, padding=padding,
    )
    return [grad_input, grad_filter]
