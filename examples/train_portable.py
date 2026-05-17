#!/usr/bin/env python3
"""
Portable training script: runs on AMD GPU via OpenCL locally,
or on any CUDA/CPU device (e.g. Colab) without changes.

Workflow
--------
  Local (AMD GPU):
      python examples/train_portable.py --train --epochs 20 --out weights.h5

  Colab / any machine without opencl_tf:
      python examples/train_portable.py --train --epochs 20 --out weights.h5

  Inference (either machine):
      python examples/train_portable.py --infer --weights weights.h5

Key idea
--------
  Weights are just numbers.  They don't care whether a layer was computed
  by an OpenCL kernel or a CUDA kernel.  We save them with
  model.save_weights() and restore with model.load_weights(by_name=True).
  Using by_name=True means the standard-Keras and OpenCL-backed layers can
  differ in Python type, as long as they share the same `name=` string.
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

# ── Detect backend ────────────────────────────────────────────────────────────
try:
    from opencl_tf.layers import (
        OpenCLConv2D,
        OpenCLBatchNormalization,
        OpenCLReLU,
        OpenCLDense,
    )
    USE_OPENCL = True
    print("Backend: OpenCL  (AMD GPU via Rusticl / Intel GPU)")
except ImportError:
    USE_OPENCL = False
    print("Backend: standard TensorFlow  (CPU / CUDA GPU)")


# ── Layer factories (same call-site for both backends) ────────────────────────
def Conv2D(filters, kernel_size, **kw):
    if USE_OPENCL:
        return OpenCLConv2D(filters, kernel_size, **kw)
    return tf.keras.layers.Conv2D(
        filters, kernel_size, use_bias=kw.pop("use_bias", True), **kw
    )


def BN(**kw):
    if USE_OPENCL:
        return OpenCLBatchNormalization(**kw)
    return tf.keras.layers.BatchNormalization(**kw)


def ReLU(**kw):
    if USE_OPENCL:
        return OpenCLReLU(**kw)
    return tf.keras.layers.ReLU(**kw)


def Dense(units, **kw):
    if USE_OPENCL:
        return OpenCLDense(units, **kw)
    return tf.keras.layers.Dense(units, **kw)


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(num_classes: int = 10, input_shape=(32, 32, 3)):
    """
    Small CNN:  Conv→BN→ReLU  ×2  →  GlobalAvgPool  →  Dense→ReLU  →  Dense.

    Every layer has an explicit `name=` so that weights can be transferred
    between the OpenCL and standard-Keras versions with by_name=True.
    """
    inp = tf.keras.Input(shape=input_shape, name="input")

    x = Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(inp)
    x = BN(name="bn1")(x)
    x = ReLU(name="relu1")(x)

    x = Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = BN(name="bn2")(x)
    x = ReLU(name="relu2")(x)

    # Global-average-pool: not (yet) a custom op, use built-in TF op.
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = Dense(128, name="fc1")(x)
    x = ReLU(name="relu3")(x)
    out = Dense(num_classes, name="fc2")(x)

    return tf.keras.Model(inp, out, name="portable_cnn")


# ── Synthetic dataset ─────────────────────────────────────────────────────────
def make_dataset(n_train=1024, n_val=256, img_size=32, num_classes=10, batch=32):
    """
    Synthetic CIFAR-style dataset (random images + labels).
    Replace this with tf.keras.datasets.cifar10 or your own data.
    """
    rng = np.random.default_rng(0)
    X_tr = rng.standard_normal((n_train, img_size, img_size, 3), dtype="float32")
    y_tr = rng.integers(0, num_classes, n_train)
    X_va = rng.standard_normal((n_val,   img_size, img_size, 3), dtype="float32")
    y_va = rng.integers(0, num_classes, n_val)

    tr = (tf.data.Dataset
          .from_tensor_slices((X_tr, y_tr))
          .shuffle(n_train, seed=1)
          .batch(batch)
          .prefetch(tf.data.AUTOTUNE))
    va = (tf.data.Dataset
          .from_tensor_slices((X_va, y_va))
          .batch(batch)
          .prefetch(tf.data.AUTOTUNE))
    return tr, va


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    tr_ds, va_ds = make_dataset(
        n_train=args.n_train, n_val=args.n_val,
        img_size=32, num_classes=10, batch=args.batch,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out, save_weights_only=True, save_best_only=True,
            monitor="val_loss", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1,
        ),
    ]

    history = model.fit(
        tr_ds, validation_data=va_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    print(f"\nWeights saved to: {args.out}")
    return model, history


# ── Inference / weight loading demo ──────────────────────────────────────────
def infer(args):
    if not os.path.exists(args.weights):
        sys.exit(f"Weights file not found: {args.weights}")

    model = build_model()
    # Build the model by running one dummy forward pass so that all weights exist.
    _ = model(tf.zeros((1, 32, 32, 3)))

    # Keras 3 matches weights by object path (layer name= string),
    # so OpenCL-backed and standard-Keras layers are interchangeable
    # as long as every layer has the same explicit name= argument.
    model.load_weights(args.weights)
    print(f"Loaded weights from: {args.weights}")

    rng = np.random.default_rng(42)
    x   = rng.standard_normal((4, 32, 32, 3), dtype="float32")
    logits = model(x, training=False)
    preds  = tf.argmax(logits, axis=-1).numpy()
    print(f"Predictions for 4 random inputs: {preds}")


# ── Colab snippet (printed, not executed) ─────────────────────────────────────
COLAB_SNIPPET = '''
# ──────────────────────────────────────────────────────────────
# Colab / CUDA machine: load weights trained on AMD GPU
# ──────────────────────────────────────────────────────────────
# 1. Upload weights.h5 to Colab (Files panel or Drive mount)
# 2. Copy-paste this block:

import numpy as np
import tensorflow as tf

def build_model(num_classes=10, input_shape=(32, 32, 3)):
    inp = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1")(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, name="fc1")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    out = tf.keras.layers.Dense(num_classes, name="fc2")(x)
    return tf.keras.Model(inp, out)

model = build_model()
model(tf.zeros((1, 32, 32, 3)))          # build weights
model.load_weights("my_model.weights.h5")

# Fine-tune with CUDA or just run inference:
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# model.fit(...)   ← continue training on Colab GPU
# ──────────────────────────────────────────────────────────────
'''


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Portable OpenCL ↔ CUDA training")
    sub = p.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Train model and save weights")
    tr.add_argument("--epochs",  type=int,   default=10)
    tr.add_argument("--batch",   type=int,   default=32)
    tr.add_argument("--lr",      type=float, default=1e-3)
    tr.add_argument("--n-train", type=int,   default=1024, dest="n_train")
    tr.add_argument("--n-val",   type=int,   default=256,  dest="n_val")
    tr.add_argument("--out",     type=str,   default="weights.weights.h5")

    inf = sub.add_parser("infer", help="Load weights and run inference demo")
    inf.add_argument("--weights", type=str, default="weights.weights.h5")

    sub.add_parser("colab", help="Print Colab loading snippet")

    args = p.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer":
        infer(args)
    elif args.cmd == "colab":
        print(COLAB_SNIPPET)
    else:
        p.print_help()
        print()
        print("Quick start:")
        print("  python examples/train_portable.py train --epochs 5 --out my_model.weights.h5")
        print("  python examples/train_portable.py infer --weights my_model.weights.h5")
        print("  python examples/train_portable.py colab   # print Colab paste snippet")


if __name__ == "__main__":
    main()
