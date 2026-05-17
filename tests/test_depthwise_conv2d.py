"""Correctness tests for OpenCL depthwise conv2d vs tf.nn.depthwise_conv2d."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import opencl_tf
from opencl_tf.ops.depthwise_conv2d import depthwise_conv2d

TOL = 1e-4


def _inputs(seed, shape_x, shape_w):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal(shape_x).astype(np.float32),
        rng.standard_normal(shape_w).astype(np.float32),
    )


@pytest.mark.parametrize("dm", [1, 2])
@pytest.mark.parametrize(
    "strides,padding",
    [
        ((1, 1, 1, 1), "SAME"),
        ((1, 2, 2, 1), "SAME"),
        ((1, 1, 1, 1), "VALID"),
        ((1, 2, 2, 1), "VALID"),
    ],
)
def test_dwconv2d_forward(strides, padding, dm):
    x_np, w_np = _inputs(0, (2, 9, 9, 4), (3, 3, 4, dm))

    y_ours = depthwise_conv2d(x_np, w_np, strides=strides, padding=padding).numpy()
    y_ref  = tf.nn.depthwise_conv2d(x_np, w_np,
                                    strides=list(strides), padding=padding).numpy()

    np.testing.assert_allclose(y_ours, y_ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("dm", [1, 2])
@pytest.mark.parametrize(
    "strides,padding",
    [
        ((1, 1, 1, 1), "SAME"),
        ((1, 2, 2, 1), "SAME"),
        ((1, 2, 2, 1), "VALID"),
    ],
)
def test_dwconv2d_gradients(strides, padding, dm):
    x_np, w_np = _inputs(1, (2, 9, 9, 4), (3, 3, 4, dm))

    x = tf.Variable(x_np)
    w = tf.Variable(w_np)

    with tf.GradientTape() as tape:
        y = depthwise_conv2d(x, w, strides=strides, padding=padding)
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_ours, gw_ours = tape.gradient(loss, [x, w])

    with tf.GradientTape() as tape_ref:
        y_ref = tf.nn.depthwise_conv2d(x, w, strides=list(strides), padding=padding)
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_ref, gw_ref = tape_ref.gradient(loss_ref, [x, w])

    np.testing.assert_allclose(gx_ours.numpy(), gx_ref.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(gw_ours.numpy(), gw_ref.numpy(), atol=TOL, rtol=TOL)


def test_keras_depthwise_layer_trains():
    from opencl_tf.layers import OpenCLDepthwiseConv2D

    layer = OpenCLDepthwiseConv2D(kernel_size=3, strides=1, padding="same")
    opt   = tf.keras.optimizers.SGD(1e-2)

    x      = tf.random.normal((2, 8, 8, 4), seed=7)
    target = tf.random.normal((2, 8, 8, 4), seed=11)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y = layer(x)
            loss = tf.reduce_mean((y - target) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        opt.apply_gradients(zip(grads, layer.trainable_variables))
        return loss

    loss0 = float(step())
    for _ in range(5):
        loss_final = float(step())
    assert loss_final < loss0
