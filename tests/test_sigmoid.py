"""Correctness tests for OpenCL Sigmoid vs tf.nn.sigmoid."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import pytest

import threading
import time

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


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------

def test_sigmoid_concurrent_threads():
    """Fire N_THREADS threads simultaneously, each calling sigmoid on its own
    tensor.  Validates that the cl_command_queue mutex prevents data races
    under TF's inter-op thread pool conditions.  All outputs must be
    numerically correct; any exception in a worker thread is re-raised."""
    N_THREADS  = 16
    ITERATIONS = 10  # each thread runs this many sigmoid calls

    errors   = []
    rng_main = np.random.default_rng(99)
    # Pre-generate inputs so threads don't race on the RNG.
    inputs = [
        rng_main.standard_normal((4, 8, 8, 8)).astype(np.float32)
        for _ in range(N_THREADS)
    ]
    refs = [tf.nn.sigmoid(x).numpy() for x in inputs]

    def worker(tid):
        try:
            x = inputs[tid]
            ref = refs[tid]
            for _ in range(ITERATIONS):
                y = sigmoid(x).numpy()
                if not np.allclose(y, ref, atol=TOL):
                    max_err = float(np.max(np.abs(y - ref)))
                    errors.append(
                        f"thread {tid}: max error {max_err:.2e} exceeds {TOL:.2e}"
                    )
                    return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"thread {tid} raised: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0

    assert not errors, "\n".join(errors)
    total_calls = N_THREADS * ITERATIONS
    print(f"\n  concurrent stress: {total_calls} calls, {elapsed:.2f}s "
          f"({total_calls/elapsed:.0f} calls/s)")


def test_sigmoid_stress_sequential():
    """Run sigmoid 200 times on the same tensor and verify the result is
    identical every time.  Catches any state leak between kernel invocations
    (e.g. a forgotten buffer release or a stale cached kernel arg)."""
    N = 200
    rng = np.random.default_rng(77)
    x_np = rng.standard_normal((8, 16, 16, 4)).astype(np.float32)
    ref  = tf.nn.sigmoid(x_np).numpy()

    t0 = time.perf_counter()
    for i in range(N):
        y = sigmoid(x_np).numpy()
        assert np.allclose(y, ref, atol=TOL), \
            f"Iteration {i}: max error {float(np.max(np.abs(y - ref))):.2e}"
    elapsed = time.perf_counter() - t0
    print(f"\n  sequential stress: {N} calls, {elapsed:.2f}s "
          f"({N/elapsed:.0f} calls/s)")
