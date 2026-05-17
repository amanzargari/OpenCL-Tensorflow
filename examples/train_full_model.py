"""
Full model end-to-end training example on synthetic data.

This script builds the complete uwb_loc_v3 model from docs/ROADMAP.md
using OpenCL layers throughout (Phase 1–3 ops), runs a forward pass,
then trains for a few steps to verify the loss decreases.

Usage:
    cd <repo-root>
    conda run -n opencl-tf python examples/train_full_model.py
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# Import the package to load the .so and register all gradients.
import opencl_tf  # noqa: F401
from opencl_tf.layers import (
    OpenCLConv2D,
    OpenCLDepthwiseConv2D,
    OpenCLBatchNormalization,
    OpenCLReLU,
    OpenCLDense,
    OpenCLUpSampling2D,
    OpenCLSigmoid,
)

# ---------------------------------------------------------------------------
# Model hyper-parameters (reduced for fast synthetic-data demonstration)
# ---------------------------------------------------------------------------
T_CTX      = 8    # temporal context frames (H dimension after reshape)
N_BINS_KEPT = 16  # range bins kept (W dimension after reshape)

# Head output grid: 9 × 6 × 8  (H × W × C after dense + reshape)
HEAD_H = 9
HEAD_W = 6
HEAD_C = 8


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def conv_bn_relu(x, filters, kernel_size, strides=(1, 1), name=None):
    prefix = (name + "_") if name else ""
    x = OpenCLConv2D(filters, kernel_size, strides=strides,
                     padding="same", name=prefix + "conv")(x)
    x = OpenCLBatchNormalization(name=prefix + "bn")(x)
    x = OpenCLReLU(name=prefix + "relu")(x)
    return x


def dsconv_block(x, filters, strides=(1, 1), name=None):
    prefix = (name + "_") if name else ""
    x = OpenCLDepthwiseConv2D(3, strides=strides, padding="same",
                               name=prefix + "dw")(x)
    x = OpenCLBatchNormalization(name=prefix + "dw_bn")(x)
    x = OpenCLReLU(name=prefix + "dw_relu")(x)
    x = OpenCLConv2D(filters, 1, padding="same", name=prefix + "pw")(x)
    x = OpenCLBatchNormalization(name=prefix + "pw_bn")(x)
    x = OpenCLReLU(name=prefix + "pw_relu")(x)
    return x


# ---------------------------------------------------------------------------
# Full model (matches docs/ROADMAP.md build_model() with OpenCL layers)
# ---------------------------------------------------------------------------
def build_model():
    inp = layers.Input(shape=(T_CTX, 6, 3, N_BINS_KEPT, 2), name="stacked_iq")

    # Permute + reshape to 2D spatial representation [T_CTX, N_BINS_KEPT, 36]
    x = layers.Permute((1, 4, 2, 3, 5), name="to_tcrange")(inp)
    x = layers.Reshape((T_CTX, N_BINS_KEPT, 36), name="merge_sensors")(x)

    # Backbone: spatial dims 8×16 → 8×8 → 4×4 → 2×2 → 1×1
    x = conv_bn_relu(x, 48,  (3, 7), strides=(1, 2), name="stem")
    x = dsconv_block(x, 96,  strides=(2, 2), name="b1")
    x = dsconv_block(x, 144, strides=(2, 2), name="b2")
    x = dsconv_block(x, 192, strides=(2, 2), name="b3")
    x = conv_bn_relu(x, 64,  (1, 1),         name="neck")

    # Flatten to 1D feature vector
    trunk = layers.Flatten(name="flatten")(x)

    # Head: dense → reshape → 2× upsample → refine → 2× upsample → refine → heatmap
    h = OpenCLDense(HEAD_H * HEAD_W * HEAD_C, name="hm_dense")(trunk)
    h = OpenCLReLU(name="hm_relu")(h)
    h = layers.Reshape((HEAD_H, HEAD_W, HEAD_C), name="hm_reshape")(h)
    h = OpenCLUpSampling2D((2, 2), name="hm_up1")(h)
    h = conv_bn_relu(h, 16, (3, 3), name="hm_refine1")
    h = OpenCLUpSampling2D((2, 2), name="hm_up2")(h)
    h = conv_bn_relu(h, 16, (3, 3), name="hm_refine2")

    # Final 1×1 conv + sigmoid produces the heatmap
    h = OpenCLConv2D(1, (1, 1), name="heatmap_conv")(h)
    heatmap = OpenCLSigmoid(name="heatmap")(h)

    return Model(inp, heatmap, name="uwb_loc_v3")


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
BATCH = 2

def make_batch(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(
        (BATCH, T_CTX, 6, 3, N_BINS_KEPT, 2)).astype(np.float32)
    # Target heatmap: batch × (HEAD_H*4) × (HEAD_W*4) × 1
    hout = HEAD_H * 4  # 2 upsampling blocks of ×2
    wout = HEAD_W * 4
    y = rng.uniform(0, 1, (BATCH, hout, wout, 1)).astype(np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"TensorFlow {tf.__version__}, OpenCL-Tensorflow {opencl_tf.__version__}")

    model = build_model()
    model.summary(line_length=80)

    # ---- Forward pass ----
    print("\n--- Forward pass (training=True) ---")
    x_test, y_test = make_batch(seed=1)
    out = model(x_test, training=True)
    print(f"Input shape:  {x_test.shape}")
    print(f"Output shape: {out.shape}   (expected ({BATCH}, {HEAD_H*4}, {HEAD_W*4}, 1))")
    assert out.shape == (BATCH, HEAD_H * 4, HEAD_W * 4, 1), \
        f"Unexpected output shape: {out.shape}"
    print("Forward pass OK.")

    # ---- Training loop (manual, to avoid Keras compile quirks) ----
    print("\n--- Training (5 steps, MSE loss) ---")
    opt = tf.keras.optimizers.Adam(1e-3)
    x_train = tf.constant(x_test)
    y_train = tf.constant(y_test)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            pred = model(x_train, training=True)
            loss = tf.reduce_mean((pred - y_train) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    losses = []
    for i in range(5):
        l = float(train_step())
        losses.append(l)
        print(f"  step {i+1}: loss = {l:.6f}")

    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 5 steps: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print(f"\nLoss decreased from {losses[0]:.6f} to {losses[-1]:.6f}  ✓")
    print("\nPhase 3 milestone: full build_model() trains end-to-end on OpenCL.")
