"""Correctness tests for OpenCL Sigmoid vs tf.nn.sigmoid."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import pytest

import opencl_tf
from opencl_tf.ops.sigmoid import sigmoid, sigmoid_grad

TOL = 1e-4


def test_sigmoid_forward_small():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 5, 5, 3)).astype(np.float32)
    y_ours = sigmoid(x_np).numpy()
    y_ref  = tf.nn.sigmoid(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


@pytest.mark.parametrize("shape", [
    (1, 1, 1, 1),
    (4, 8, 8, 16),
    (8, 4, 4, 32),
    (3, 1, 1, 64),
])
def test_sigmoid_forward_shapes(shape):
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal(shape).astype(np.float32)
    y_ours = sigmoid(x_np).numpy()
    y_ref  = tf.nn.sigmoid(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_sigmoid_extreme_values():
    """Large-magnitude inputs should saturate to 0 or 1 without overflow."""
    x_np = np.array([[-100.0, 100.0, 0.0, -30.0, 30.0]], dtype=np.float32)
    y_ours = sigmoid(x_np).numpy()
    y_ref  = tf.nn.sigmoid(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_sigmoid_gradient_via_tape():
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((3, 6, 6, 4)).astype(np.float32)

    x_ours = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        y = sigmoid(x_ours)
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_ours = tape.gradient(loss, x_ours)

    x_ref = tf.Variable(x_np)
    with tf.GradientTape() as tape_ref:
        y_ref = tf.nn.sigmoid(x_ref)
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_ref = tape_ref.gradient(loss_ref, x_ref)

    np.testing.assert_allclose(gx_ours.numpy(), gx_ref.numpy(), atol=TOL, rtol=TOL)


def test_sigmoid_backward_uses_forward_output():
    """SigmoidGrad takes the forward output y, not x. Verify the kernel uses y."""
    rng = np.random.default_rng(2)
    y_np  = np.clip(rng.uniform(0.01, 0.99, (2, 4, 4, 3)).astype(np.float32), 0.01, 0.99)
    dy_np = rng.standard_normal((2, 4, 4, 3)).astype(np.float32)

    grad_x_ours = sigmoid_grad(y_np, dy_np).numpy()
    # Reference: grad_x = dy * y * (1 - y)
    grad_x_ref  = (dy_np * y_np * (1.0 - y_np)).astype(np.float32)
    np.testing.assert_allclose(grad_x_ours, grad_x_ref, atol=TOL)


def test_keras_sigmoid_layer_forward():
    from opencl_tf.layers import OpenCLSigmoid
    layer = OpenCLSigmoid()
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((2, 4, 4, 8)).astype(np.float32)
    y_ours = layer(x_np).numpy()
    y_ref  = tf.nn.sigmoid(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_keras_sigmoid_layer_trains():
    """Sanity: loss decreases over 5 gradient steps through a sigmoid layer."""
    from opencl_tf.layers import OpenCLSigmoid, OpenCLConv2D

    conv  = OpenCLConv2D(4, 3, padding="same")
    act   = OpenCLSigmoid()
    opt   = tf.keras.optimizers.SGD(1e-2)

    x      = tf.random.normal((2, 6, 6, 3), seed=10)
    target = tf.random.normal((2, 6, 6, 4), seed=11)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y    = act(conv(x))
            loss = tf.reduce_mean((y - target) ** 2)
        grads = tape.gradient(loss, conv.trainable_variables)
        opt.apply_gradients(zip(grads, conv.trainable_variables))
        return loss

    loss0 = float(step())
    for _ in range(5):
        loss_final = float(step())
    assert loss_final < loss0, f"Loss did not decrease: {loss0} -> {loss_final}"
