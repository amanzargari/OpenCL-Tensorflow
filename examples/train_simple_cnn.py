"""End-to-end training demo: a tiny CNN with an OpenCLConv2D layer.

Run from the repo root:
    python examples/train_simple_cnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

import opencl_tf  # noqa: F401  -- registers gradients
from opencl_tf.layers import OpenCLConv2D


def main() -> None:
    rng = np.random.default_rng(0)
    N, H, W, Cin = 32, 16, 16, 3
    x_np = rng.standard_normal((N, H, W, Cin)).astype(np.float32)
    # Synthetic regression target: convolve with a fixed random kernel.
    true_w = rng.standard_normal((3, 3, Cin, 4)).astype(np.float32)
    y_np = tf.nn.conv2d(x_np, true_w, strides=1, padding="SAME").numpy()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(H, W, Cin)),
        OpenCLConv2D(8, 3, padding="same"),
        tf.keras.layers.ReLU(),
        OpenCLConv2D(4, 3, padding="same"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss="mse",
    )
    model.fit(x_np, y_np, epochs=5, batch_size=8, verbose=1)


if __name__ == "__main__":
    main()
