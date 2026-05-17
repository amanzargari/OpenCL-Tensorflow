# OpenCL-Tensorflow

A lightweight C++ / OpenCL backend for **training and inference** of a custom
TensorFlow / Keras model on AMD GPUs that **do not support ROCm** — specifically
older GCN parts (e.g. Topaz XT) where only the open-source `amdgpu` + Mesa /
Clover / rusticl OpenCL stack is available.

Implemented as a set of **TensorFlow custom ops** (forward + backward), each
backed by hand-written OpenCL kernels. Gradients are registered through
`tf.RegisterGradient` so `tf.GradientTape` flows through transparently.

> **Status — Phase 1 (Conv2D).** Standard 2D convolution forward, backprop-input,
> and backprop-filter are implemented end-to-end and verified against
> `tf.nn.conv2d` on four `(stride, padding)` cases. See [`docs/ROADMAP.md`](docs/ROADMAP.md)
> for the rest of the layer queue.

---

## What's here

| Layer / op                | Forward | dL/dx | dL/dw | Notes                              |
|---------------------------|:-------:|:-----:|:-----:|------------------------------------|
| `Conv2D` (`use_bias=False`) |   ✅    |  ✅   |  ✅   | NHWC, SAME/VALID, asymmetric pad    |
| `DepthwiseConv2D`         |   ⏳    |  ⏳   |  ⏳   | Phase 2                            |
| `BatchNormalization`      |   ⏳    |  ⏳   |  ⏳   | Phase 2                            |
| `ReLU`                    |   ⏳    |  ⏳   |       | Phase 2 (trivial)                  |
| `Dense`                   |   ⏳    |  ⏳   |  ⏳   | Phase 3 (via GEMM kernel)          |
| `UpSampling2D` (bilinear) |   ⏳    |  ⏳   |       | Phase 3 (scatter-add backward)     |

---

## Hardware & software targets

- **GPU:** AMD GCN 3rd gen and similar (Topaz XT, Tonga, Fiji, Polaris).
  Anything that gives you a working OpenCL 1.2 ICD via `clinfo`.
- **OS:** Ubuntu 22.04 / 24.04. Other distros work if you can get `clinfo` to
  see your GPU.
- **OpenCL:** 1.2 target (compiles fine on 2.0+ ICDs too).
- **TensorFlow:** 2.10 – 2.15 (CPU build of TF; we don't use the GPU device).

---

## Install

### 1. System packages

```bash
sudo apt update
sudo apt install -y build-essential cmake \
                    opencl-headers ocl-icd-opencl-dev \
                    mesa-opencl-icd clinfo \
                    python3-dev python3-pip
```

Verify your GPU is visible:

```bash
clinfo -l
# Should list your AMD device under a Mesa/Clover or rusticl platform.
```

### 2. Python deps

```bash
pip install -r requirements.txt
```

### 3. Build the shared library

Either of these works.

**CMake (recommended):**

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
# Produces opencl_tf/opencl_tf_ops.so
```

**Make (simpler):**

```bash
make
```

### 4. Smoke test

```bash
pytest tests/ -v
```

You should see all `(strides, padding)` cases pass with max-abs-diff ≤ 1e-4
against `tf.nn.conv2d`.

---

## Usage

### Raw op (mirrors `tf.nn.conv2d`)

```python
import opencl_tf

y = opencl_tf.conv2d(x, w, strides=(1, 2, 2, 1), padding="SAME")
```

### Keras layer (drop-in for `Conv2D(use_bias=False)`)

```python
from tensorflow.keras import Input, Model
from opencl_tf.layers import OpenCLConv2D

inp = Input(shape=(64, 64, 3))
x   = OpenCLConv2D(32, 3, strides=2, padding="same")(inp)
# ... add the rest of your model
model = Model(inp, x)
```

### Training

Gradients are auto-registered, so `tf.GradientTape` and `model.fit()` Just Work:

```python
import tensorflow as tf
import opencl_tf  # importing the package registers all gradients

w = tf.Variable(tf.random.normal([3, 3, 3, 32]))
x = tf.random.normal([4, 64, 64, 3])

with tf.GradientTape() as tape:
    y    = opencl_tf.conv2d(x, w, strides=(1, 1, 1, 1), padding="SAME")
    loss = tf.reduce_mean(y * y)

gw = tape.gradient(loss, w)   # computed on the OpenCL device
```

---

## Repository layout

```
OpenCL-Tensorflow/
├── kernels/                  # OpenCL .cl source files
│   └── conv2d_kernels.cl
├── src/                      # C++ TF custom-op sources
│   ├── cl_backend.h          # CLBackend singleton + RAII helpers
│   ├── cl_backend.cc
│   ├── padding_utils.h       # SAME / VALID resolver
│   └── ops/
│       └── conv2d_ops.cc     # REGISTER_OP + OpKernel for the 3 conv2d ops
├── opencl_tf/                # Python package
│   ├── __init__.py
│   ├── _library.py           # loads opencl_tf_ops.so
│   ├── gradients.py          # @RegisterGradient bindings
│   ├── layers.py             # Keras-friendly wrappers
│   └── ops/
│       └── conv2d.py
├── tests/
│   ├── conftest.py
│   └── test_conv2d.py
├── examples/
│   └── train_simple_cnn.py
├── docs/
│   ├── ARCHITECTURE.md       # design decisions, why DEVICE_CPU, etc.
│   ├── ADDING_NEW_OPS.md     # cookbook for Phase 2+ contributors
│   └── ROADMAP.md
├── .github/workflows/build.yml
├── CMakeLists.txt
├── Makefile
├── requirements.txt
└── README.md
```

---

## How it works (one-paragraph version)

The C++ ops are registered on `DEVICE_CPU` — TF gives us host-pointer tensors,
we copy them into `cl_mem` device buffers, launch a kernel from
`kernels/conv2d_kernels.cl`, and blocking-read the result back into the output
tensor TF allocated. A singleton `CLBackend` (in `src/cl_backend.cc`) owns the
platform/device/context/queue and caches compiled programs and kernels keyed by
filename. The queue is serialized with a mutex because `cl_command_queue` is
not thread-safe per spec, and TF will call `Compute()` concurrently from its
inter-op thread pool. Gradients are registered in `opencl_tf/gradients.py` and
chain the forward op to two custom backward ops that share the same
`CLBackend` instance. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
the long version.

---

## Roadmap

The model we're ultimately backing is in [`docs/ROADMAP.md`](docs/ROADMAP.md);
short version is: depthwise-separable conv stack → BN/ReLU → flatten → dense
head → bilinear upsample → final 1×1 conv → sigmoid heatmap. Every op needed
for that pipeline is listed there with status.

---

## License

MIT — see [`LICENSE`](LICENSE).
