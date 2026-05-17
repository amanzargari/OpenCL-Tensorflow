"""Correctness tests: opencl_tf.conv2d vs tf.nn.conv2d (forward + gradients)."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

import opencl_tf


# Tolerance vs cuDNN/Eigen reference. The two implementations sum
# floating-point products in different orders, so 1e-5 is too strict
# for non-trivial sizes; 1e-4 is comfortable.
TOL = 1e-4


def _inputs(seed: int, shape_x, shape_w):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape_x).astype(np.float32)
    w = rng.standard_normal(shape_w).astype(np.float32)
    return x, w


@pytest.mark.parametrize(
    "strides,padding",
    [
        ((1, 1, 1, 1), "SAME"),
        ((1, 2, 2, 1), "SAME"),
        ((1, 1, 1, 1), "VALID"),
        ((1, 2, 2, 1), "VALID"),
    ],
)
def test_conv2d_matches_tf_forward(strides, padding):
    x_np, w_np = _inputs(0, (2, 9, 9, 3), (3, 3, 3, 5))
    x = tf.constant(x_np)
    w = tf.constant(w_np)

    y_ours = opencl_tf.conv2d(x, w, strides=strides, padding=padding).numpy()
    y_ref  = tf.nn.conv2d(x, w, strides=list(strides), padding=padding).numpy()

    np.testing.assert_allclose(y_ours, y_ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize(
    "strides,padding",
    [
        ((1, 1, 1, 1), "SAME"),
        ((1, 2, 2, 1), "SAME"),
        ((1, 1, 1, 1), "VALID"),
        ((1, 2, 2, 1), "VALID"),
    ],
)
def test_conv2d_matches_tf_gradients(strides, padding):
    x_np, w_np = _inputs(1, (2, 9, 9, 3), (3, 3, 3, 5))

    x = tf.Variable(x_np)
    w = tf.Variable(w_np)

    with tf.GradientTape() as tape:
        y = opencl_tf.conv2d(x, w, strides=strides, padding=padding)
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_ours, gw_ours = tape.gradient(loss, [x, w])

    with tf.GradientTape() as tape_ref:
        y_ref = tf.nn.conv2d(x, w, strides=list(strides), padding=padding)
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_ref, gw_ref = tape_ref.gradient(loss_ref, [x, w])

    np.testing.assert_allclose(gx_ours.numpy(), gx_ref.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(gw_ours.numpy(), gw_ref.numpy(), atol=TOL, rtol=TOL)


def test_keras_layer_trains():
    """Sanity-check: an OpenCLConv2D layer participates in a tf.function
    optimizer step and the loss decreases."""
    from opencl_tf.layers import OpenCLConv2D

    layer = OpenCLConv2D(filters=4, kernel_size=3, strides=1, padding="same")
    opt   = tf.keras.optimizers.SGD(learning_rate=1e-3)

    x = tf.random.normal((2, 8, 8, 3), seed=42)
    target = tf.random.normal((2, 8, 8, 4), seed=43)

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
    assert loss_final < loss0, (loss0, loss_final)
