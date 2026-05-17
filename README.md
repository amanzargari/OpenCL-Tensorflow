# OpenCL-Tensorflow

A lightweight C++ / OpenCL backend for **training and inference** of a custom
TensorFlow / Keras model on AMD GPUs that **do not support ROCm** — specifically
older GCN parts (e.g. Topaz XT) where only the open-source `amdgpu` + Mesa /
Clover / rusticl OpenCL stack is available.

Implemented as a set of **TensorFlow custom ops** (forward + backward), each
backed by hand-written OpenCL kernels. Gradients are registered through
`tf.RegisterGradient` so `tf.GradientTape` flows through transparently.

> **Status — Phase 2 complete.** The four ops needed for `conv_bn_relu` and
> `dsconv_block` (Conv2D, DepthwiseConv2D, BatchNormalization, ReLU) are all
> implemented and tested. The stem + b1 + b2 + b3 + neck portion of the target
> model trains end-to-end on the OpenCL backend.
>
> Phase 3 (Dense, UpSampling2D bilinear, Sigmoid) is up next — see
> [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## What's here

| Layer / op                 | Forward | dL/dx | dL/dw | Notes                            |
|----------------------------|:-------:|:-----:|:-----:|----------------------------------|
| `Conv2D` (`use_bias=False`)|   ✅    |  ✅   |  ✅   | NHWC, SAME/VALID                 |
| `DepthwiseConv2D`          |   ✅    |  ✅   |  ✅   | depth_multiplier supported       |
| `BatchNormalization`       |   ✅    |  ✅   |  ✅   | train + inference; EMA in Keras  |
| `ReLU`                     |   ✅    |  ✅   |  —    | elementwise                      |
| `Dense`                    |   ⏳    |  ⏳   |  ⏳   | Phase 3 (GEMM kernel)            |
| `UpSampling2D` (bilinear)  |   ⏳    |  ⏳   |  —    | Phase 3 (scatter-add backward)   |
| `Sigmoid`                  |   ⏳    |  ⏳   |  —    | Phase 3                          |

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

You should see all conv2d, depthwise_conv2d, batchnorm, and relu tests pass.

---

## Usage

### Raw ops

```python
import opencl_tf as ocl

y = ocl.conv2d(x, w, strides=(1, 2, 2, 1), padding="SAME")
z = ocl.depthwise_conv2d(y, dw, strides=(1, 1, 1, 1), padding="SAME")
b, batch_mean, batch_var = ocl.batch_norm_training(z, gamma, beta, epsilon=1e-3)
a = ocl.relu(b)
```

### Keras layers

```python
from tensorflow.keras import Input, Model
from opencl_tf.layers import (
    OpenCLConv2D, OpenCLDepthwiseConv2D,
    OpenCLBatchNormalization, OpenCLReLU,
)

def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = OpenCLConv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = OpenCLBatchNormalization()(x)
    return OpenCLReLU()(x)

def dsconv_block(x, filters, strides):
    x = OpenCLDepthwiseConv2D(3, strides=strides, padding="same")(x)
    x = OpenCLBatchNormalization()(x)
    x = OpenCLReLU()(x)
    x = OpenCLConv2D(filters, 1, padding="same")(x)
    x = OpenCLBatchNormalization()(x)
    return OpenCLReLU()(x)

inp = Input(shape=(64, 64, 3))
x   = conv_bn_relu(inp, 48, (3, 7), strides=(1, 2))
x   = dsconv_block(x, 96,  strides=(2, 2))
x   = dsconv_block(x, 144, strides=(2, 2))
x   = dsconv_block(x, 192, strides=(2, 2))
x   = conv_bn_relu(x, 64, (1, 1))
model = Model(inp, x)
```

### Training

Gradients are auto-registered, so `tf.GradientTape` and `model.fit()` Just Work:

```python
import tensorflow as tf
import opencl_tf

model.compile(optimizer="adam", loss="mse")
model.fit(x_train, y_train, epochs=10, batch_size=8)
```

---

## Repository layout

```
OpenCL-Tensorflow/
├── kernels/                          # OpenCL .cl source files
│   ├── conv2d_kernels.cl
│   ├── depthwise_conv2d_kernels.cl
│   ├── batchnorm_kernels.cl
│   └── relu_kernels.cl
├── src/                              # C++ TF custom-op sources
│   ├── cl_backend.{h,cc}             # CLBackend singleton + RAII helpers
│   ├── padding_utils.h               # SAME / VALID resolver
│   └── ops/
│       ├── conv2d_ops.cc
│       ├── depthwise_conv2d_ops.cc
│       ├── batchnorm_ops.cc
│       └── relu_ops.cc
├── opencl_tf/                        # Python package
│   ├── __init__.py
│   ├── _library.py                   # loads opencl_tf_ops.so
│   ├── gradients.py                  # @RegisterGradient bindings
│   ├── layers.py                     # OpenCL{Conv2D,DepthwiseConv2D,BN,ReLU}
│   └── ops/
│       ├── conv2d.py
│       ├── depthwise_conv2d.py
│       ├── relu.py
│       └── batchnorm.py
├── tests/                            # pytest suite, parity vs tf.nn.*
├── examples/train_simple_cnn.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ADDING_NEW_OPS.md
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
we copy them into `cl_mem` device buffers, launch a kernel from the
corresponding file in `kernels/`, and blocking-read the result back into the
output tensor TF allocated. A singleton `CLBackend` (`src/cl_backend.{h,cc}`)
owns the platform/device/context/queue and caches compiled programs and
kernels keyed by filename. The queue is serialized with a mutex because
`cl_command_queue` is not thread-safe per spec, and TF will call `Compute()`
concurrently from its inter-op thread pool. Gradients are registered in
`opencl_tf/gradients.py` and chain forward ops to their backward
counterparts, all of which share the same `CLBackend` instance. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the long version.

---

## License

MIT — see [`LICENSE`](LICENSE).
