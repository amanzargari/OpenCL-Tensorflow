"""Gradient registrations for every custom op exported by `opencl_tf_ops.so`.

The decorator key must match the `REGISTER_OP(...)` string in the C++
source exactly.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.python.framework import ops

from ._library import raw_ops


# ---------------------------------------------------------------------
# Conv2D
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclConv2d")
def _opencl_conv2d_grad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    strides = list(op.get_attr("strides"))
    padding = op.get_attr("padding")
    grad_input = raw_ops.opencl_conv2d_backprop_input(
        tf.shape(x), w, grad, strides=strides, padding=padding)
    grad_filter = raw_ops.opencl_conv2d_backprop_filter(
        x, tf.shape(w), grad, strides=strides, padding=padding)
    return [grad_input, grad_filter]


# ---------------------------------------------------------------------
# DepthwiseConv2D
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclDepthwiseConv2d")
def _opencl_depthwise_conv2d_grad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    strides = list(op.get_attr("strides"))
    padding = op.get_attr("padding")
    grad_input = raw_ops.opencl_depthwise_conv2d_backprop_input(
        tf.shape(x), w, grad, strides=strides, padding=padding)
    grad_filter = raw_ops.opencl_depthwise_conv2d_backprop_filter(
        x, tf.shape(w), grad, strides=strides, padding=padding)
    return [grad_input, grad_filter]


# ---------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclRelu")
def _opencl_relu_grad(op, grad):
    return [raw_ops.opencl_relu_grad(grad, op.inputs[0])]


# ---------------------------------------------------------------------
# BatchNorm (training).
# The forward op has THREE outputs (y, saved_mean, saved_var); gradients
# arrive in the same order. Only grad_y is meaningful for backprop --
# grad_mean / grad_var are ignored (these are side-output statistics, not
# part of the loss path).
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclBatchNormTraining")
def _opencl_bn_training_grad(op, grad_y, grad_mean, grad_var):
    x          = op.inputs[0]
    gamma      = op.inputs[1]
    # beta     = op.inputs[2]  -- not needed in the backward pass
    saved_mean = op.outputs[1]
    saved_var  = op.outputs[2]
    epsilon    = op.get_attr("epsilon")

    grad_x, grad_gamma, grad_beta = raw_ops.opencl_batch_norm_grad(
        grad_y, x, gamma, saved_mean, saved_var, epsilon=epsilon)
    return [grad_x, grad_gamma, grad_beta]


# OpenclBatchNormInference is inference-only; no gradient is registered.
# If it ever appears inside tf.GradientTape, TF will raise a
# LookupError, which is the desired behaviour.


# ---------------------------------------------------------------------
# Sigmoid
# The forward op has ONE output y = sigmoid(x). The backward takes y
# (not x) to compute grad_x = dy * y * (1 - y).
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclSigmoid")
def _opencl_sigmoid_grad(op, grad):
    y = op.outputs[0]
    return [raw_ops.opencl_sigmoid_grad(y, grad)]


# ---------------------------------------------------------------------
# Dense
# Forward inputs: x [batch, in_f], W [in_f, out_f], b [out_f]
# Returns one gradient per forward input, in order: grad_x, grad_W, grad_b
# ---------------------------------------------------------------------
@ops.RegisterGradient("OpenclDense")
def _opencl_dense_grad(op, grad):
    x = op.inputs[0]
    W = op.inputs[1]
    grad_x = raw_ops.opencl_dense_backprop_input(grad, W)
    grad_W = raw_ops.opencl_dense_backprop_weight(x, grad)
    grad_b = raw_ops.opencl_dense_backprop_bias(grad)
    return [grad_x, grad_W, grad_b]
