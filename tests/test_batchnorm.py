"""Correctness tests for OpenCL BatchNormalization.

The reference is the BN formulas applied directly with TF primitives,
not `tf.nn.fused_batch_norm` (whose API has quirks around biased vs
unbiased variance that just complicate the comparison).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

import opencl_tf
from opencl_tf.ops.batchnorm import (
    batch_norm_training,
    batch_norm_inference,
)

TOL = 1e-4
EPS = 1e-3


def _bn_reference(x, gamma, beta, mean, var, epsilon):
    """Plain BN math, suitable as a reference for both forward and grad."""
    inv_std = tf.math.rsqrt(var + epsilon)
    x_hat   = (x - mean) * inv_std
    return gamma * x_hat + beta


def _batch_stats(x):
    mean = tf.reduce_mean(x, axis=[0, 1, 2])
    var  = tf.reduce_mean(tf.square(x - mean), axis=[0, 1, 2])  # biased
    return mean, var


def test_bn_training_forward():
    rng = np.random.default_rng(0)
    x_np     = rng.standard_normal((4, 7, 7, 8)).astype(np.float32)
    gamma_np = rng.standard_normal((8,)).astype(np.float32) + 1.0
    beta_np  = rng.standard_normal((8,)).astype(np.float32) * 0.1

    x, gamma, beta = tf.constant(x_np), tf.constant(gamma_np), tf.constant(beta_np)

    y_ours, mean_ours, var_ours = batch_norm_training(x, gamma, beta, epsilon=EPS)

    mean_ref, var_ref = _batch_stats(x)
    y_ref = _bn_reference(x, gamma, beta, mean_ref, var_ref, EPS)

    np.testing.assert_allclose(mean_ours.numpy(), mean_ref.numpy(), atol=TOL)
    np.testing.assert_allclose(var_ours.numpy(),  var_ref.numpy(),  atol=TOL)
    np.testing.assert_allclose(y_ours.numpy(),    y_ref.numpy(),    atol=TOL)


def test_bn_inference_forward():
    rng = np.random.default_rng(1)
    x_np      = rng.standard_normal((2, 5, 5, 6)).astype(np.float32)
    gamma_np  = rng.standard_normal((6,)).astype(np.float32) + 1.0
    beta_np   = rng.standard_normal((6,)).astype(np.float32) * 0.1
    mean_np   = rng.standard_normal((6,)).astype(np.float32) * 0.5
    var_np    = np.abs(rng.standard_normal((6,)).astype(np.float32)) + 0.1

    x, gamma, beta = tf.constant(x_np), tf.constant(gamma_np), tf.constant(beta_np)
    mean, var      = tf.constant(mean_np), tf.constant(var_np)

    y_ours = batch_norm_inference(x, gamma, beta, mean, var, epsilon=EPS).numpy()
    y_ref  = _bn_reference(x, gamma, beta, mean, var, EPS).numpy()

    np.testing.assert_allclose(y_ours, y_ref, atol=TOL)


def test_bn_training_gradients():
    """Compare grads of our op vs grads of the reference math, both
    computed by tf.GradientTape. This implicitly verifies the entire
    backward path (dx, dgamma, dbeta)."""
    rng = np.random.default_rng(2)
    x_np     = rng.standard_normal((3, 5, 5, 4)).astype(np.float32)
    gamma_np = rng.standard_normal((4,)).astype(np.float32) + 1.0
    beta_np  = rng.standard_normal((4,)).astype(np.float32) * 0.1

    # ----- ours -----
    x_o     = tf.Variable(x_np)
    gamma_o = tf.Variable(gamma_np)
    beta_o  = tf.Variable(beta_np)
    with tf.GradientTape() as tape:
        y, _, _ = batch_norm_training(x_o, gamma_o, beta_o, epsilon=EPS)
        loss = 0.5 * tf.reduce_sum(y * y)
    g_x_o, g_gam_o, g_bet_o = tape.gradient(loss, [x_o, gamma_o, beta_o])

    # ----- reference -----
    x_r     = tf.Variable(x_np)
    gamma_r = tf.Variable(gamma_np)
    beta_r  = tf.Variable(beta_np)
    with tf.GradientTape() as tape_ref:
        mean_r, var_r = _batch_stats(x_r)
        y_ref = _bn_reference(x_r, gamma_r, beta_r, mean_r, var_r, EPS)
        loss_ref = 0.5 * tf.reduce_sum(y_ref * y_ref)
    g_x_r, g_gam_r, g_bet_r = tape_ref.gradient(loss_ref, [x_r, gamma_r, beta_r])

    np.testing.assert_allclose(g_bet_o.numpy(), g_bet_r.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(g_gam_o.numpy(), g_gam_r.numpy(), atol=TOL, rtol=TOL)
    np.testing.assert_allclose(g_x_o.numpy(),   g_x_r.numpy(),   atol=TOL, rtol=TOL)


def test_keras_bn_layer_trains():
    """Sanity: stack an OpenCLBatchNormalization between two convs and
    confirm the loss drops over a few steps."""
    from opencl_tf.layers import OpenCLConv2D, OpenCLBatchNormalization, OpenCLReLU

    layer1 = OpenCLConv2D(8, 3, padding="same")
    bn     = OpenCLBatchNormalization()
    act    = OpenCLReLU()
    layer2 = OpenCLConv2D(4, 3, padding="same")
    opt    = tf.keras.optimizers.SGD(1e-2)

    x      = tf.random.normal((2, 6, 6, 3), seed=4)
    target = tf.random.normal((2, 6, 6, 4), seed=5)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            h = act(bn(layer1(x), training=True))
            y = layer2(h)
            loss = tf.reduce_mean((y - target) ** 2)
        vars_ = (layer1.trainable_variables
                 + bn.trainable_variables
                 + layer2.trainable_variables)
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients(zip(grads, vars_))
        return loss

    loss0 = float(step())
    for _ in range(5):
        loss_final = float(step())
    assert loss_final < loss0, (loss0, loss_final)


def test_bn_moving_stats_update():
    """After several training-mode passes, moving_mean should drift
    away from its zero initialisation toward the (approximately zero)
    batch mean of unit-variance inputs."""
    from opencl_tf.layers import OpenCLBatchNormalization

    bn = OpenCLBatchNormalization(momentum=0.5)  # fast EMA for the test
    # Build by calling once
    _ = bn(tf.zeros((1, 2, 2, 3)), training=False)
    init_mean = bn.moving_mean.numpy().copy()
    init_var  = bn.moving_var.numpy().copy()

    rng = np.random.default_rng(0)
    for _ in range(4):
        x = tf.constant(rng.standard_normal((4, 3, 3, 3)).astype(np.float32) * 3 + 1)
        _ = bn(x, training=True)

    assert not np.allclose(bn.moving_mean.numpy(), init_mean)
    assert not np.allclose(bn.moving_var.numpy(),  init_var)
