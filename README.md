# OpenCL-Tensorflow

A lightweight C++ / OpenCL backend for **training and inference** of custom
TensorFlow / Keras models on AMD GPUs that **do not support ROCm** — specifically
older GCN parts where only the open-source `amdgpu` + Mesa / rusticl OpenCL stack
is available.

Implemented as a set of **TF custom ops** (forward + backward), each backed by
hand-written OpenCL 1.2 kernels. Gradients are registered through
`tf.RegisterGradient` so `tf.GradientTape` and `model.fit()` flow through
transparently. Weights saved locally can be loaded on any TF environment
(CUDA, CPU) without changes — see [Portable training](#portable-training--weight-transfer).

> **Status — Phase 3 complete.**  All seven ops are implemented and tested.
> The full `build_model()` graph trains end-to-end on the OpenCL backend —
> see [`examples/train_full_model.py`](examples/train_full_model.py).
> Phase 4 (performance: buffer pool, tiled GEMM, tree-reduction BN) is next —
> see [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Implemented ops

| Layer / op                 | Forward | dL/dx | dL/dw | Notes                                     |
|----------------------------|:-------:|:-----:|:-----:|-------------------------------------------|
| `Conv2D`                   |   ✅    |  ✅   |  ✅   | NHWC, SAME/VALID, any stride              |
| `DepthwiseConv2D`          |   ✅    |  ✅   |  ✅   | `depth_multiplier` supported              |
| `BatchNormalization`       |   ✅    |  ✅   |  ✅   | train + inference, EMA maintained in Keras |
| `ReLU`                     |   ✅    |  ✅   |   —   | elementwise                               |
| `Dense`                    |   ✅    |  ✅   |  ✅   | GEMM: one work-item per output element    |
| `UpSampling2D` (bilinear)  |   ✅    |  ✅   |   —   | half-pixel centres, float atomic scatter-add |
| `Sigmoid`                  |   ✅    |  ✅   |   —   | backward takes forward output y (not x)  |

71 tests pass — run `pytest tests/ -v` to verify.

---

## Hardware & software targets

| Item | Requirement |
|---|---|
| GPU | Any that exposes OpenCL 1.2 via `clinfo` — AMD GCN (Topaz, Tonga, Fiji, Polaris, Vega), Intel integrated, or Mesa CPU fallback |
| OS | Ubuntu 22.04 / 24.04; other distros work if `clinfo -l` sees your device |
| OpenCL | 1.2 target; works on 2.0+ ICDs too |
| TF | 2.10 – 2.15 CPU build (ops are registered on `DEVICE_CPU`; GPU compute happens via OpenCL, not TF's CUDA/ROCm path) |
| Python | 3.8 – 3.13 |

The `CLBackend` automatically probes each OpenCL platform by compiling a trivial
test kernel and picks the first one that succeeds. Broken ICDs (e.g. Mesa rusticl
without the LLVM SPIR-V backend) are silently skipped. If your preferred platform
is not selected, set `OCL_ICD_VENDORS=/etc/OpenCL/vendors/your.icd` before running.

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

Verify your device is visible:

```bash
clinfo -l
# Example output:
# Platform #0: rusticl
#  `-- Device #0: AMD Radeon R5 M445 Series (radeonsi, ...)
# Platform #1: Intel(R) OpenCL Graphics
#  `-- Device #0: Intel(R) UHD Graphics 620
```

### 2. Python deps

```bash
pip install -r requirements.txt
```

### 3. Build the shared library

**Make:**

```bash
make
# Produces opencl_tf/opencl_tf_ops.so
```

**CMake:**

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

The Makefile auto-detects TF include and link flags from the active Python
environment (`python3 -c 'import tensorflow as tf; ...'`).

### 4. Smoke test

```bash
pytest tests/ -v
# Expected: 71 passed
```

---

## Usage

### Import

```python
import opencl_tf as ocl
```

### Raw ops

```python
y = ocl.conv2d(x, w, strides=(1, 2, 2, 1), padding="SAME")
z = ocl.depthwise_conv2d(y, dw, strides=(1, 1, 1, 1), padding="SAME")
b, mean, var = ocl.batch_norm_training(z, gamma, beta, epsilon=1e-3)
a = ocl.relu(b)
s = ocl.sigmoid(a)
o = ocl.dense(s_flat, W, bias)
```

### Keras layers

Every layer is a drop-in for its `tf.keras.layers` counterpart:

```python
from opencl_tf.layers import (
    OpenCLConv2D, OpenCLDepthwiseConv2D,
    OpenCLBatchNormalization, OpenCLReLU,
    OpenCLDense, OpenCLUpSampling2D, OpenCLSigmoid,
)

def conv_bn_relu(x, filters, kernel_size, strides=1):
    x = OpenCLConv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = OpenCLBatchNormalization()(x)
    return OpenCLReLU()(x)

def dsconv_block(x, filters, strides):
    x = OpenCLDepthwiseConv2D(3, strides=strides, padding="same")(x)
    x = OpenCLBatchNormalization()(x)
    x = OpenCLReLU()(x)
    x = OpenCLConv2D(filters, 1)(x)
    x = OpenCLBatchNormalization()(x)
    return OpenCLReLU()(x)

inp = tf.keras.Input(shape=(64, 64, 3))
x   = conv_bn_relu(inp, 48, (3, 7), strides=(1, 2))
x   = dsconv_block(x, 96,  strides=(2, 2))
x   = dsconv_block(x, 144, strides=(2, 2))
x   = OpenCLDense(128)(tf.keras.layers.Flatten()(x))
x   = OpenCLReLU()(x)
x   = OpenCLUpSampling2D((2, 2))(tf.keras.layers.Reshape((4, 4, 8))(x))
out = OpenCLSigmoid()(x)
model = tf.keras.Model(inp, out)
```

### Training

Gradients are auto-registered so `model.fit()` and `tf.GradientTape` just work:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="mse",
)
model.fit(x_train, y_train, epochs=10, batch_size=8)
```

See [`examples/train_full_model.py`](examples/train_full_model.py) for a
complete end-to-end training loop on the `uwb_loc_v3` model.

---

## Portable training / weight transfer

Train locally on your AMD GPU, then load weights on any other machine
(Colab, a cloud CUDA box, a friend's laptop) with zero code changes.

```bash
# Local — uses OpenCL if opencl_tf is importable, otherwise falls back to
# standard TensorFlow automatically.
python examples/train_portable.py train --epochs 30 --out my_model.weights.h5

# Print the Colab paste snippet
python examples/train_portable.py colab
```

The key: save with `model.save_weights()` (weights only, no graph) and load
with `model.load_weights()`. Keras matches weights by layer `name=` string, so
`OpenCLDense(name="fc1")` and `tf.keras.layers.Dense(name="fc1")` share the
same weight slots. See [`examples/train_portable.py`](examples/train_portable.py)
for the full two-sided implementation.

---

## GPU stress test

```bash
# Default: 60 s per phase on whichever GPU OpenCL selects
python tools/stress_gpu.py

# Longer run, bigger matrices
python tools/stress_gpu.py --duration 120 --size 2048

# Force a specific platform
python tools/stress_gpu.py --platform rusticl   # AMD via Mesa
python tools/stress_gpu.py --platform Intel     # Intel integrated GPU
```

Three phases: FP32 SGEMM (compute), buffer copy (bandwidth), and SGEMM + LUT
reads (mixed). Reports GFLOPS and GB/s. Reads GPU temperature via `rocm-smi`
if available.

---

## Repository layout

```
OpenCL-Tensorflow/
├── kernels/                          # OpenCL .cl source files
│   ├── conv2d_kernels.cl
│   ├── depthwise_conv2d_kernels.cl
│   ├── batchnorm_kernels.cl
│   ├── relu_kernels.cl
│   ├── sigmoid_kernels.cl
│   ├── dense_kernels.cl
│   └── upsampling_bilinear_kernels.cl
├── src/                              # C++ TF custom-op sources
│   ├── cl_backend.{h,cc}             # CLBackend singleton + RAII helpers
│   ├── padding_utils.h               # SAME / VALID resolver
│   └── ops/
│       ├── conv2d_ops.cc
│       ├── depthwise_conv2d_ops.cc
│       ├── batchnorm_ops.cc
│       ├── relu_ops.cc
│       ├── sigmoid_ops.cc
│       ├── dense_ops.cc
│       └── upsampling_ops.cc
├── opencl_tf/                        # Python package
│   ├── __init__.py
│   ├── _library.py                   # loads opencl_tf_ops.so + sets KERNELS_PATH
│   ├── gradients.py                  # @RegisterGradient for all 7 ops
│   ├── layers.py                     # OpenCL{Conv2D,DepthwiseConv2D,BN,ReLU,
│   │                                 #         Dense,UpSampling2D,Sigmoid}
│   └── ops/
│       ├── conv2d.py
│       ├── depthwise_conv2d.py
│       ├── batchnorm.py
│       ├── relu.py
│       ├── sigmoid.py
│       ├── dense.py
│       └── upsampling.py
├── tests/                            # pytest suite — parity vs tf.nn.*
│   ├── test_conv2d.py
│   ├── test_depthwise_conv2d.py
│   ├── test_batchnorm.py
│   ├── test_relu.py
│   ├── test_sigmoid.py               # includes concurrent + sequential stress
│   ├── test_dense.py
│   └── test_upsampling.py
├── examples/
│   ├── train_full_model.py           # full uwb_loc_v3 end-to-end training
│   ├── train_portable.py             # local AMD GPU → Colab CUDA weight transfer
│   └── train_simple_cnn.py
├── tools/
│   └── stress_gpu.py                 # OpenCL compute / bandwidth stress test
├── docs/
│   ├── ARCHITECTURE.md               # internals deep-dive
│   ├── ADDING_NEW_OPS.md             # step-by-step guide for new ops
│   ├── TROUBLESHOOTING.md            # common issues and fixes
│   └── ROADMAP.md
├── CMakeLists.txt
├── Makefile
├── requirements.txt
└── README.md
```

---

## How it works

The C++ ops are registered on `DEVICE_CPU` — TF gives us host-pointer tensors,
we copy them into `cl_mem` device buffers, launch an OpenCL kernel, and
blocking-read the result back. A singleton `CLBackend` owns the
platform/device/context/queue and caches compiled programs and kernels.

The command queue is serialized with a mutex because `cl_command_queue` is not
thread-safe per spec, and TF will call `Compute()` concurrently from its inter-op
thread pool. **Both `clSetKernelArg` and `clEnqueue*` calls must be inside that
mutex** — `clSetKernelArg` on a cached kernel is not thread-safe either.

Gradients are registered in `opencl_tf/gradients.py` and chain forward ops to
their backward counterparts, all sharing the same `CLBackend` instance.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical
breakdown and [`docs/ADDING_NEW_OPS.md`](docs/ADDING_NEW_OPS.md) for the
step-by-step guide to adding a new layer.

---

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for common issues
(OpenCL platform selection, `GLIBCXX` version errors, kernel compile failures).

---

## License

MIT — see [`LICENSE`](LICENSE).
