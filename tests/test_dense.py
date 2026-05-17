"""Correctness tests for OpenCL Dense (fully-connected) vs tf.keras.layers.Dense."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import pytest

import opencl_tf
from opencl_tf.ops.dense import dense, dense_backprop_input, dense_backprop_weight, dense_backprop_bias

TOL = 1e-4


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------

def test_dense_forward_basic():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    W_np = rng.standard_normal((8, 16)).astype(np.float32)
    b_np = rng.standard_normal((16,)).astype(np.float32)

    y_ours = dense(x_np, W_np, b_np).numpy()
    y_ref  = (x_np @ W_np + b_np).astype(np.float32)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


@pytest.mark.parametrize("batch,in_f,out_f", [
    (1,  1,  1),
    (8,  32, 64),
    (16, 64, 32),
    (3,  128, 10),
])
def test_dense_forward_shapes(batch, in_f, out_f):
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((batch, in_f)).astype(np.float32)
    W_np = rng.standard_normal((in_f, out_f)).astype(np.float32)
    b_np = rng.standard_normal((out_f,)).astype(np.float32)

    y_ours = dense(x_np, W_np, b_np).numpy()
    y_ref  = (x_np @ W_np + b_np).astype(np.float32)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_dense_parity_with_keras():
    """Compare against tf.keras.layers.Dense with fixed weights."""
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((5, 12)).astype(np.float32)
    W_np = rng.standard_normal((12, 24)).astype(np.float32)
    b_np = rng.standard_normal((24,)).astype(np.float32)

    ref_layer = tf.keras.layers.Dense(24, use_bias=True, activation=None)
    ref_layer.build((None, 12))
    ref_layer.kernel.assign(W_np)
    ref_layer.bias.assign(b_np)

    y_ours = dense(x_np, W_np, b_np).numpy()
    y_ref  = ref_layer(x_np).numpy()
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


# ---------------------------------------------------------------------------
# Individual backward ops parity
# ---------------------------------------------------------------------------

def test_dense_backprop_input():
    rng = np.random.default_rng(1)
    batch, in_f, out_f = 4, 8, 12
    grad_y = rng.standard_normal((batch, out_f)).astype(np.float32)
    W_np   = rng.standard_normal((in_f, out_f)).astype(np.float32)

    # dL/dx = grad_y @ W^T
    grad_x_ours = dense_backprop_input(grad_y, W_np).numpy()
    grad_x_ref  = (grad_y @ W_np.T).astype(np.float32)
    np.testing.assert_allclose(grad_x_ours, grad_x_ref, atol=TOL)


def test_dense_backprop_weight():
    rng = np.random.default_rng(2)
    batch, in_f, out_f = 4, 8, 12
    x_np   = rng.standard_normal((batch, in_f)).astype(np.float32)
    grad_y = rng.standard_normal((batch, out_f)).astype(np.float32)

    # dL/dW = x^T @ grad_y
    grad_W_ours = dense_backprop_weight(x_np, grad_y).numpy()
    grad_W_ref  = (x_np.T @ grad_y).astype(np.float32)
    np.testing.assert_allclose(grad_W_ours, grad_W_ref, atol=TOL)


def test_dense_backprop_bias():
    rng = np.random.default_rng(3)
    batch, out_f = 6, 10
    grad_y = rng.standard_normal((batch, out_f)).astype(np.float32)

    # dL/db = sum over batch dim
    grad_b_ours = dense_backprop_bias(grad_y).numpy()
    grad_b_ref  = grad_y.sum(axis=0).astype(np.float32)
    np.testing.assert_allclose(grad_b_ours, grad_b_ref, atol=TOL)


# ---------------------------------------------------------------------------
# End-to-end gradient via GradientTape
# ---------------------------------------------------------------------------

def test_dense_gradient_via_tape():
    rng = np.random.default_rng(4)
    batch, in_f, out_f = 4, 8, 6
    x_np = rng.standard_normal((batch, in_f)).astype(np.float32)
    W_np = rng.standard_normal((in_f, out_f)).astype(np.float32)
    b_np = rng.standard_normal((out_f,)).astype(np.float32)

    # ours
    x_o = tf.Variable(x_np)
    W_o = tf.Variable(W_np)
    b_o = tf.Variable(b_np)
    with tf.GradientTape() as tape:
        y = dense(x_o, W_o, b_o)
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_o, gW_o, gb_o = tape.gradient(loss, [x_o, W_o, b_o])

    # reference: tf matmul
    x_r = tf.Variable(x_np)
    W_r = tf.Variable(W_np)
    b_r = tf.Variable(b_np)
    with tf.GradientTape() as tape_ref:
        y_ref = tf.linalg.matmul(x_r, W_r) + b_r
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_r, gW_r, gb_r = tape_ref.gradient(loss_ref, [x_r, W_r, b_r])

    np.testing.assert_allclose(gx_o.numpy(), gx_r.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(gW_o.numpy(), gW_r.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(gb_o.numpy(), gb_r.numpy(), atol=TOL, rtol=TOL)


# ---------------------------------------------------------------------------
# Keras layer
# ---------------------------------------------------------------------------

def test_keras_dense_layer_forward():
    from opencl_tf.layers import OpenCLDense
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((4, 16)).astype(np.float32)

    layer = OpenCLDense(8)
    layer.build((None, 16))
    W_np = layer.kernel.numpy()
    b_np = layer.bias.numpy()

    y_ours = layer(x_np).numpy()
    y_ref  = (x_np @ W_np + b_np).astype(np.float32)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_keras_dense_layer_trains():
    """Sanity: loss decreases over 5 gradient steps."""
    from opencl_tf.layers import OpenCLDense

    layer = OpenCLDense(8)
    opt   = tf.keras.optimizers.SGD(1e-2)

    rng    = np.random.default_rng(6)
    x      = tf.constant(rng.standard_normal((8, 16)).astype(np.float32))
    target = tf.constant(rng.standard_normal((8, 8)).astype(np.float32))

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            y    = layer(x)
            loss = tf.reduce_mean((y - target) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        opt.apply_gradients(zip(grads, layer.trainable_variables))
        return loss

    loss0 = float(step())
    for _ in range(5):
        loss_final = float(step())
    assert loss_final < loss0, f"Loss did not decrease: {loss0} -> {loss_final}"


def test_keras_dense_no_bias():
    """use_bias=False omits the bias term."""
    from opencl_tf.layers import OpenCLDense
    rng = np.random.default_rng(8)
    x_np = rng.standard_normal((3, 10)).astype(np.float32)

    layer = OpenCLDense(5, use_bias=False)
    layer.build((None, 10))
    W_np = layer.kernel.numpy()

    y_ours = layer(x_np).numpy()
    y_ref  = (x_np @ W_np).astype(np.float32)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)
