"""Keras-friendly wrappers around the raw OpenCL ops.

Currently exports:
    OpenCLConv2D    -- drop-in for tf.keras.layers.Conv2D(use_bias=False)

Phase 2+ will add OpenCLDepthwiseConv2D, OpenCLBatchNormalization, etc.
"""

from __future__ import annotations

from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, initializers

from .ops.conv2d import conv2d


def _as_pair(x) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)


class OpenCLConv2D(layers.Layer):
    """2D convolution executed on our OpenCL backend.

    Mirrors `tf.keras.layers.Conv2D(use_bias=False)`. Bias is intentionally
    omitted here because the model this library targets sets
    `use_bias=False` everywhere and follows every conv with BatchNorm.

    Parameters
    ----------
    filters : int
        Number of output channels (Cout).
    kernel_size : int | tuple[int, int]
        Spatial extent of the filter, (kH, kW).
    strides : int | tuple[int, int]
        Spatial stride.
    padding : str
        "same" or "valid" (case-insensitive).
    kernel_initializer : str | Initializer
        Standard Keras initializer.
    """

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
        self.kernel = None  # filled in build()

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
