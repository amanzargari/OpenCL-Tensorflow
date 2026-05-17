"""OpenCL backend for selected TensorFlow ops.

Importing this package:
  - Loads the compiled `opencl_tf_ops.so` shared library.
  - Registers gradients for every custom op so `tf.GradientTape` Just Works.
  - Exposes convenience functions and Keras layers.

Typical usage:

    import opencl_tf
    y  = opencl_tf.conv2d(x, w, strides=(1, 2, 2, 1), padding="SAME")

    # ...or as a Keras layer:
    from opencl_tf.layers import OpenCLConv2D
    out = OpenCLConv2D(32, 3, strides=2, padding="same")(x)
"""

from ._library import raw_ops
from .ops.conv2d import (
    conv2d,
    conv2d_backprop_input,
    conv2d_backprop_filter,
)
from . import gradients  # noqa: F401  -- registers @RegisterGradient hooks
from . import layers

__all__ = [
    "raw_ops",
    "conv2d",
    "conv2d_backprop_input",
    "conv2d_backprop_filter",
    "layers",
]

__version__ = "0.1.0"
