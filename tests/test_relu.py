"""Correctness tests for OpenCL ReLU vs tf.nn.relu."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import opencl_tf
from opencl_tf.ops.relu import relu

TOL = 1e-6


def test_relu_forward():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 5, 5, 3)).astype(np.float32)

    y_ours = relu(x_np).numpy()
    y_ref  = tf.nn.relu(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_relu_gradient():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 5, 5, 3)).astype(np.float32)

    x = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        y = relu(x)
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_ours = tape.gradient(loss, x)

    with tf.GradientTape() as tape_ref:
        y_ref = tf.nn.relu(x)
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_ref = tape_ref.gradient(loss_ref, x)

    np.testing.assert_allclose(gx_ours.numpy(), gx_ref.numpy(), atol=TOL)


def test_relu_at_zero_grad_is_zero():
    """ReLU's gradient at x == 0 is conventionally 0 (subgradient choice)."""
    x = tf.Variable(np.zeros((4,), dtype=np.float32))
    with tf.GradientTape() as tape:
        y = relu(x)
        loss = tf.reduce_sum(y)
    g = tape.gradient(loss, x).numpy()
    np.testing.assert_array_equal(g, np.zeros_like(g))
