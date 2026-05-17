"""Correctness tests for OpenCL UpSampling2D bilinear.

Reference: tf.keras.layers.UpSampling2D(size, interpolation='bilinear')
Cross-check: tf.image.resize(method='bilinear') -- both should be identical.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import pytest

import opencl_tf
from opencl_tf.ops.upsampling import upsampling_bilinear_2d

TOL = 1e-4


def _ref(x_np, sy, sx):
    return tf.keras.layers.UpSampling2D((sy, sx), interpolation='bilinear')(
        tf.constant(x_np)).numpy()


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------

def test_upsampling_forward_2x2():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((2, 4, 4, 3)).astype(np.float32)
    y_ours = upsampling_bilinear_2d(x_np, size=[2, 2]).numpy()
    y_ref  = _ref(x_np, 2, 2)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


@pytest.mark.parametrize("sy,sx", [
    (2, 2),
    (3, 3),
    (2, 3),
    (4, 2),
])
def test_upsampling_forward_factors(sy, sx):
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((2, 4, 6, 8)).astype(np.float32)
    y_ours = upsampling_bilinear_2d(x_np, size=[sy, sx]).numpy()
    y_ref  = _ref(x_np, sy, sx)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_upsampling_agrees_with_image_resize():
    """image.resize and UpSampling2D must produce identical results."""
    rng = np.random.default_rng(1)
    x_np = rng.standard_normal((2, 5, 7, 4)).astype(np.float32)
    x = tf.constant(x_np)

    y_ours   = upsampling_bilinear_2d(x, size=[2, 2]).numpy()
    y_resize = tf.image.resize(x, [10, 14], method='bilinear').numpy()
    np.testing.assert_allclose(y_ours, y_resize, atol=TOL)


def test_upsampling_single_pixel():
    """H=W=1 input: all output pixels should equal the sole input pixel."""
    x_np = np.array([[[[3.0, -1.0, 2.5]]]], dtype=np.float32)  # [1,1,1,3]
    y_ours = upsampling_bilinear_2d(x_np, size=[3, 4]).numpy()
    y_ref  = _ref(x_np, 3, 4)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)
    np.testing.assert_allclose(y_ours, np.broadcast_to(x_np, y_ours.shape), atol=TOL)


def test_upsampling_1x_is_identity():
    """size=(1,1) should reproduce the input exactly."""
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal((2, 6, 8, 4)).astype(np.float32)
    y_ours = upsampling_bilinear_2d(x_np, size=[1, 1]).numpy()
    np.testing.assert_allclose(y_ours, x_np, atol=TOL)


# ---------------------------------------------------------------------------
# Backward via GradientTape
# ---------------------------------------------------------------------------

def test_upsampling_gradient_via_tape():
    rng = np.random.default_rng(2)
    x_np = rng.standard_normal((2, 4, 4, 3)).astype(np.float32)

    x_ours = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        y = upsampling_bilinear_2d(x_ours, size=[2, 2])
        loss = 0.5 * tf.reduce_sum(y * y)
    gx_ours = tape.gradient(loss, x_ours).numpy()

    x_ref = tf.Variable(x_np)
    with tf.GradientTape() as tape_ref:
        y_ref = tf.image.resize(x_ref, [8, 8], method='bilinear')
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    gx_ref = tape_ref.gradient(loss_ref, x_ref).numpy()

    np.testing.assert_allclose(gx_ours, gx_ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("sy,sx", [(2, 2), (3, 2)])
def test_upsampling_gradient_shapes(sy, sx):
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
    x_ours = tf.Variable(x_np)
    with tf.GradientTape() as tape:
        y = upsampling_bilinear_2d(x_ours, size=[sy, sx])
        loss = tf.reduce_sum(y)
    gx = tape.gradient(loss, x_ours)
    assert gx.shape == x_ours.shape


# ---------------------------------------------------------------------------
# Keras layer
# ---------------------------------------------------------------------------

def test_keras_upsampling_layer_forward():
    from opencl_tf.layers import OpenCLUpSampling2D
    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((2, 4, 4, 8)).astype(np.float32)
    layer = OpenCLUpSampling2D(size=(2, 2))
    y_ours = layer(x_np).numpy()
    y_ref  = _ref(x_np, 2, 2)
    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_keras_upsampling_layer_nearest_rejected():
    from opencl_tf.layers import OpenCLUpSampling2D
    with pytest.raises(ValueError, match="bilinear"):
        OpenCLUpSampling2D(size=(2, 2), interpolation='nearest')


def test_keras_upsampling_layer_trains():
    """Sanity: loss decreases when upsampling is in a trainable stack."""
    from opencl_tf.layers import OpenCLUpSampling2D, OpenCLConv2D

    conv1  = OpenCLConv2D(4, 3, padding="same")
    up     = OpenCLUpSampling2D(size=(2, 2))
    conv2  = OpenCLConv2D(4, 3, padding="same")
    opt    = tf.keras.optimizers.SGD(1e-2)

    rng    = np.random.default_rng(6)
    x      = tf.constant(rng.standard_normal((2, 4, 4, 3)).astype(np.float32))
    target = tf.constant(rng.standard_normal((2, 8, 8, 4)).astype(np.float32))

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            h    = up(conv1(x))
            y    = conv2(h)
            loss = tf.reduce_mean((y - target) ** 2)
        vars_ = conv1.trainable_variables + conv2.trainable_variables
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    loss0 = float(step())
    for _ in range(5):
        loss_final = float(step())
    assert loss_final < loss0, f"Loss did not decrease: {loss0} -> {loss_final}"
