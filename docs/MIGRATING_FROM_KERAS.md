# Migrating from Keras layers to OpenCL-TF layers

This guide shows how to swap standard Keras layers for their OpenCL-backed
counterparts in `opencl_tf.layers`. The goal is a minimal, drop-in change that
keeps model semantics the same while running supported ops on OpenCL.

## Quick checklist

1. Import OpenCL layers from `opencl_tf.layers`.
2. Replace supported Keras layers using the mapping table below.
3. Split activations out of `Dense`/`Conv2D` into explicit activation layers.
4. Keep channels-last (NHWC) data format and `padding="same"|"valid"`.
5. Run your tests (or `pytest tests/ -v`) to validate parity.

## Layer mapping

| Keras layer | OpenCL-TF layer | Notes |
| --- | --- | --- |
| `tf.keras.layers.Conv2D` | `OpenCLConv2D` | No `use_bias`, no `activation` arg; `padding` is `same`/`valid` only. |
| `tf.keras.layers.DepthwiseConv2D` | `OpenCLDepthwiseConv2D` | No `use_bias`, no `activation` arg; `depth_multiplier` supported. |
| `tf.keras.layers.BatchNormalization` | `OpenCLBatchNormalization` | Channels-last only; gamma/beta always trainable; `training` flag controls EMA. |
| `tf.keras.layers.Dense` | `OpenCLDense` | `activation` handled separately; `use_bias` supported. |
| `tf.keras.layers.ReLU` / `Activation("relu")` | `OpenCLReLU` | No `max_value`/`threshold` parameters. |
| `tf.keras.layers.Activation("sigmoid")` | `OpenCLSigmoid` | Sigmoid activation only. |
| `tf.keras.layers.UpSampling2D` (bilinear) | `OpenCLUpSampling2D` | Only `interpolation="bilinear"` supported. |

## Before → after example

**Before (Keras):**

```python
import tensorflow as tf

x = tf.keras.Input(shape=(64, 64, 3))

y = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
y = tf.keras.layers.BatchNormalization()(y)
y = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu")(y)
y = tf.keras.layers.Dense(128, activation="relu")(tf.keras.layers.Flatten()(y))
y = tf.keras.layers.UpSampling2D((2, 2), interpolation="bilinear")(y)
outputs = tf.keras.layers.Activation("sigmoid")(y)

model = tf.keras.Model(x, outputs)
```

**After (OpenCL-TF):**

```python
import tensorflow as tf
from opencl_tf.layers import (
    OpenCLConv2D, OpenCLDepthwiseConv2D,
    OpenCLBatchNormalization, OpenCLDense,
    OpenCLReLU, OpenCLSigmoid, OpenCLUpSampling2D,
)

x = tf.keras.Input(shape=(64, 64, 3))

y = OpenCLConv2D(32, 3, padding="same")(x)
y = OpenCLBatchNormalization()(y)
y = OpenCLReLU()(y)

y = OpenCLDepthwiseConv2D(3, padding="same")(y)
y = OpenCLBatchNormalization()(y)
y = OpenCLReLU()(y)

y = tf.keras.layers.Flatten()(y)
y = OpenCLDense(128)(y)
y = OpenCLReLU()(y)

y = OpenCLUpSampling2D((2, 2))(y)
outputs = OpenCLSigmoid()(y)

model = tf.keras.Model(x, outputs)
```

## Behavior differences & limitations

- **NHWC only:** OpenCL layers assume channels-last (`[N, H, W, C]`).
- **Conv2D/DepthwiseConv2D:** no bias and no activation argument. If your Keras
  layer used `use_bias=True`, add a `BiasAdd` or `Dense` layer separately.
- **Dense:** no `activation` argument; use `OpenCLReLU` / `OpenCLSigmoid` after.
- **BatchNorm:** no `axis` argument and no `center=False` / `scale=False` modes.
  `training=True/False` controls whether batch stats or moving stats are used.
- **UpSampling2D:** only bilinear interpolation is supported.
- **Padding:** only `"same"` or `"valid"` are supported.
- **Extra arguments:** only the arguments present on the OpenCL layer wrappers
  are supported; features like dilation or grouped convolutions are not yet
  implemented.

## Weight portability (optional but recommended)

If you plan to train on OpenCL but load the weights elsewhere (e.g. CUDA or CPU),
keep **layer names consistent** across both models. Keras matches weights by
`name=`. This makes weight transfer seamless:

```python
# OpenCL model
x = tf.keras.Input(shape=(64, 64, 3))
y = OpenCLConv2D(32, 3, padding="same", name="conv1")(x)
model = tf.keras.Model(x, y)
model.save_weights("model.weights.h5")

# Standard Keras model (same names)
x2 = tf.keras.Input(shape=(64, 64, 3))
y2 = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv1")(x2)
model2 = tf.keras.Model(x2, y2)
model2.load_weights("model.weights.h5")
```

See [`examples/train_portable.py`](../examples/train_portable.py) for a complete
end-to-end weight transfer workflow.

## When to keep a standard Keras layer

If your model relies on features not yet supported by OpenCL-TF (e.g. grouped
convolutions, dilation, or non-bilinear upsampling), keep those layers as
standard Keras layers and only migrate the supported subset. The OpenCL layers
compose cleanly with standard Keras layers in the same model.
