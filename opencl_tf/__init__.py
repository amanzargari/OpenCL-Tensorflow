"""OpenCL backend for selected TensorFlow ops.

Importing this package:
  - Loads the compiled `opencl_tf_ops.so` shared library.
  - Registers gradients for every custom op so `tf.GradientTape` Just Works.
  - Exposes convenience functions and Keras layers.

Phase-3 inventory (forward + gradient unless noted):
    conv2d, depthwise_conv2d, relu,
    batch_norm_training (+ matching grad),
    batch_norm_inference  (no gradient -- inference-only),
    sigmoid (+ grad), dense (+ grads), upsampling_bilinear_2d (+ grad).

Keras layers exported from `opencl_tf.layers`:
    OpenCLConv2D, OpenCLDepthwiseConv2D, OpenCLBatchNormalization,
    OpenCLReLU, OpenCLSigmoid, OpenCLDense, OpenCLUpSampling2D.
"""

from ._library import raw_ops
from .ops import (
    conv2d,
    conv2d_backprop_input,
    conv2d_backprop_filter,
    depthwise_conv2d,
    depthwise_conv2d_backprop_input,
    depthwise_conv2d_backprop_filter,
    relu,
    relu_grad,
    batch_norm_training,
    batch_norm_inference,
    batch_norm_grad,
    sigmoid,
    sigmoid_grad,
    dense,
    dense_backprop_input,
    dense_backprop_weight,
    dense_backprop_bias,
)
from . import gradients  # noqa: F401  -- registers @RegisterGradient hooks
from . import layers

__all__ = [
    "raw_ops",
    "conv2d",
    "conv2d_backprop_input",
    "conv2d_backprop_filter",
    "depthwise_conv2d",
    "depthwise_conv2d_backprop_input",
    "depthwise_conv2d_backprop_filter",
    "relu",
    "relu_grad",
    "batch_norm_training",
    "batch_norm_inference",
    "batch_norm_grad",
    "sigmoid",
    "sigmoid_grad",
    "dense",
    "dense_backprop_input",
    "dense_backprop_weight",
    "dense_backprop_bias",
    "layers",
]

__version__ = "0.2.0"
