"""Python wrappers for the BatchNormalization ops."""

from __future__ import annotations

from .._library import raw_ops


def batch_norm_training(x, gamma, beta, epsilon: float = 1e-3):
    """Training-mode BatchNorm.

    Returns:
        (y, saved_mean, saved_var) -- y has the same shape as x;
        saved_mean and saved_var are per-channel ([C]).
    """
    return raw_ops.opencl_batch_norm_training(x, gamma, beta, epsilon=epsilon)


def batch_norm_inference(x, gamma, beta, mean, var, epsilon: float = 1e-3):
    """Inference-mode BatchNorm using externally supplied stats."""
    return raw_ops.opencl_batch_norm_inference(
        x, gamma, beta, mean, var, epsilon=epsilon)


def batch_norm_grad(grad_y, x, gamma, saved_mean, saved_var,
                    epsilon: float = 1e-3):
    """Gradient of batch_norm_training.

    Returns:
        (grad_x, grad_gamma, grad_beta).
    """
    return raw_ops.opencl_batch_norm_grad(
        grad_y, x, gamma, saved_mean, saved_var, epsilon=epsilon)
