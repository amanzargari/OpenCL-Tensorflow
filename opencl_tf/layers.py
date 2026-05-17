"""Keras-friendly wrappers around the raw OpenCL ops.

Exports:
    OpenCLConv2D
    OpenCLDepthwiseConv2D
    OpenCLBatchNormalization
    OpenCLReLU
"""

from __future__ import annotations

from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, initializers

from ._library import raw_ops
from .ops.conv2d import conv2d
from .ops.depthwise_conv2d import depthwise_conv2d
from .ops.relu import relu


def _as_pair(x) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


# ---------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------
class OpenCLConv2D(layers.Layer):
    """Drop-in for `tf.keras.layers.Conv2D(use_bias=False)`."""

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "same",
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = _as_pair(kernel_size)
        self.strides = _as_pair(strides)
        self.padding = padding.upper()
        if self.padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(*self.kernel_size, in_channels, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return conv2d(
            x, self.kernel,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding.lower(),
            kernel_initializer=initializers.serialize(self.kernel_initializer),
        )
        return cfg


# ---------------------------------------------------------------------
# DepthwiseConv2D
# ---------------------------------------------------------------------
class OpenCLDepthwiseConv2D(layers.Layer):
    """Drop-in for `tf.keras.layers.DepthwiseConv2D(use_bias=False)`."""

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "same",
        depth_multiplier: int = 1,
        depthwise_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kernel_size = _as_pair(kernel_size)
        self.strides = _as_pair(strides)
        self.padding = padding.upper()
        if self.padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding!r}")
        self.depth_multiplier = int(depth_multiplier)
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.kernel = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="depthwise_kernel",
            shape=(*self.kernel_size, in_channels, self.depth_multiplier),
            initializer=self.depthwise_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return depthwise_conv2d(
            x, self.kernel,
            strides=(1, self.strides[0], self.strides[1], 1),
            padding=self.padding,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding.lower(),
            depth_multiplier=self.depth_multiplier,
            depthwise_initializer=initializers.serialize(self.depthwise_initializer),
        )
        return cfg


# ---------------------------------------------------------------------
# BatchNormalization
# ---------------------------------------------------------------------
class OpenCLBatchNormalization(layers.Layer):
    """BatchNorm over the last (channel) axis.

    Mirrors a subset of `tf.keras.layers.BatchNormalization`:
      * Channels-last only (NHWC).
      * Always trainable scale (gamma) and offset (beta).
      * Tracks `moving_mean` / `moving_variance` via EMA during training.

    `call(x, training=...)`:
      * training=True  -> compute batch stats on device, update moving stats
      * training=False -> use moving stats; no EMA update
    """

    def __init__(
        self,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.momentum = float(momentum)
        self.epsilon  = float(epsilon)
        self.gamma = self.beta = None
        self.moving_mean = self.moving_var = None

    def build(self, input_shape):
        C = int(input_shape[-1])
        self.gamma = self.add_weight("gamma", shape=(C,),
                                     initializer="ones",  trainable=True)
        self.beta  = self.add_weight("beta",  shape=(C,),
                                     initializer="zeros", trainable=True)
        self.moving_mean = self.add_weight("moving_mean", shape=(C,),
                                           initializer="zeros", trainable=False)
        self.moving_var  = self.add_weight("moving_variance", shape=(C,),
                                           initializer="ones",  trainable=False)
        super().build(input_shape)

    def call(self, x, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        # Resolve to a Python bool when possible; tf.function will retrace
        # per-branch which is fine for our purposes.
        training_bool = bool(training) if isinstance(training, (bool, int)) else True

        if training_bool:
            y, batch_mean, batch_var = raw_ops.opencl_batch_norm_training(
                x, self.gamma, self.beta, epsilon=self.epsilon)
            # EMA update of the moving stats.
            self.moving_mean.assign(
                self.momentum * self.moving_mean
                + (1.0 - self.momentum) * batch_mean)
            self.moving_var.assign(
                self.momentum * self.moving_var
                + (1.0 - self.momentum) * batch_var)
            return y
        else:
            return raw_ops.opencl_batch_norm_inference(
                x, self.gamma, self.beta,
                self.moving_mean, self.moving_var,
                epsilon=self.epsilon)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(momentum=self.momentum, epsilon=self.epsilon)
        return cfg


# ---------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------
class OpenCLReLU(layers.Layer):
    """Drop-in for `tf.keras.layers.ReLU` (no max_value / threshold)."""

    def call(self, x):
        return relu(x)
