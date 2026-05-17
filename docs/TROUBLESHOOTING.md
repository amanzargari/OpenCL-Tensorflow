# Troubleshooting

---

## `GLIBCXX_3.4.32` not found

```
ImportError: .../opencl_tf_ops.so: version `GLIBCXX_3.4.32' not found
```

**Cause.** The `.so` was compiled with GCC 13 (system default on Ubuntu 22.04+),
which adds `_ZSt21ios_base_library_initv@GLIBCXX_3.4.32`. The Python
environment's `libstdc++.so.6` is an older version (GCC 11, GLIBCXX up to
3.4.30 only) — common in Conda base or older distro installs.

**Fix 1 — update Conda's libstdc++ (recommended if using Conda):**

```bash
conda install -c conda-forge libstdcxx-ng
```

Then reopen the terminal and retry `python -c "import opencl_tf"`.

**Fix 2 — preload the system libstdc++:**

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python ...
```

Warning: this sometimes pulls in a newer libstdc++ that exposes other
symbol conflicts. Prefer Fix 1.

**Fix 3 — compile inside the target environment:**

```bash
conda run -n <your-env> make
```

This compiles against the TF headers in `<your-env>` using that
environment's compiler ABI, so the result never has a version mismatch.

---

## `spirv64-unknown-unknown` / Rusticl fails to compile kernels

```
clBuildProgram failed:
error: unknown target triple 'spirv64-unknown-unknown', please use -triple or -arch
```

**Cause.** Mesa Rusticl requires the LLVM SPIR-V backend (`libLLVMSPIRVLib`)
to compile OpenCL kernels. On Ubuntu 24.04 the packaged LLVM 20 does not
include this backend. Rusticl is listed as a valid OpenCL platform by
`clinfo` but cannot compile anything.

**Behaviour.** `CLBackend` probes each platform by compiling a trivial
`"kernel void _probe() {}"`. Rusticl (usually Platform #0 on AMD machines)
fails the probe and is automatically skipped. The next working platform
(e.g., Intel's OpenCL runtime on Platform #1) is selected instead. You
will see your AMD GPU listed under `clinfo` but the ops will actually run
on the Intel iGPU (or whichever platform passes the probe).

**Fix — force a specific platform:**

```bash
# Intel integrated GPU only:
OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd python ...

# AMD via Rusticl (only works after installing the SPIR-V backend):
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd python ...
```

To install the LLVM SPIR-V backend on Ubuntu 24.04:

```bash
sudo apt install libllvmspirvlib-dev llvm-spirv
```

After installing, Rusticl should pass the probe and the AMD GPU will be
preferred (GPUs rank before other devices in CLBackend's candidate list).

---

## Wrong OpenCL platform selected

If `CLBackend` selects an unexpected platform (e.g., an Intel iGPU when you
wanted AMD), you can check which was chosen:

```bash
clinfo -l
```

Then force the desired platform by pointing to its `.icd` file:

```bash
ls /etc/OpenCL/vendors/
# e.g.: intel.icd  rusticl.icd
OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd python ...
```

Alternatively, set `OPENCL_TF_KERNELS_PATH` to override only the kernel
directory while letting the ICD loader pick the platform normally.

---

## `undefined symbol` when loading `opencl_tf_ops.so`

```
ImportError: .../opencl_tf_ops.so: undefined symbol: _ZN10tensorflow...
```

**Cause.** The `.so` was compiled against TensorFlow headers from a
different Python environment than the one used to import it. TF's C++ ABI
is tied to a specific Python / TF version.

**Fix.** Build inside the environment where you plan to use it:

```bash
# If using Conda:
conda activate <your-env>
make clean && make

# Or explicitly:
conda run -n <your-env> make clean
conda run -n <your-env> make
```

---

## Kernel compile error at runtime

```
clBuildProgram failed for kernels/conv2d_kernels.cl:
<kernel error text>
```

The full build log is printed. Common causes:

| Symptom | Likely cause |
|---------|-------------|
| `implicit function declaration` | Missing a function that OpenCL 1.2 doesn't have |
| `unknown address space qualifier` | Used `__local` / `__global` in wrong context |
| `error: no overload for operator...` | C++ syntax in an OpenCL C kernel |

Isolate the failing kernel with `clinfo -l` to verify the platform version,
then rebuild with `-cl-std=CL1.2` (already the default in `CLBackend::LoadProgram`).

---

## Tests fail with `CL_OUT_OF_RESOURCES` or `CL_MEM_OBJECT_ALLOCATION_FAILURE`

The GPU ran out of VRAM during the test. This is most common on machines
with shared VRAM (iGPU) or when many tests run concurrently.

**Fix.** Run tests with reduced parallelism:

```bash
pytest tests/ -v -n 0          # serial
pytest tests/ -v --timeout=60  # add timeout to catch hangs
```

---

## `pytest` hangs or crashes after OpenCL operations

**Cause.** OpenCL runtimes sometimes hang on `clReleaseContext` at
interpreter shutdown if kernel objects are released in the wrong order.
`CLBackend`'s destructor releases kernels before programs and programs
before the context to avoid this, but some runtime bugs remain.

**Workaround.** If you observe hangs only at teardown (not during tests),
add `--forked` or use `pytest-xdist` with process isolation:

```bash
pip install pytest-forked
pytest tests/ --forked
```

---

## `model.save_weights()` fails with "filepath must end in .weights.h5"

Keras 3 (TF ≥ 2.16) requires the new `SavedModel`-compatible format for
weight checkpoints. Use the `.weights.h5` extension:

```python
model.save_weights("my_model.weights.h5")   # ✓
model.save_weights("my_model.h5")            # ✗ on Keras 3
```

---

## `load_weights` fails with "'by_name' only supports loading legacy .h5 files"

Keras 3 dropped `by_name=True` for the new `.weights.h5` format. It now
matches weights by object path (layer `name=` string) automatically:

```python
# Keras 3 — correct:
model.load_weights("my_model.weights.h5")

# Keras 3 — wrong (raises ValueError):
model.load_weights("my_model.weights.h5", by_name=True)
```

As long as both the saving and loading models give every layer the same
`name=` argument, weights transfer between OpenCL-backed and standard Keras
layers without any extra flags.

---

## GPU stress test (`tools/stress_gpu.py`) picks a different device than the ops

`stress_gpu.py` uses raw ctypes OpenCL calls and its own platform selection
logic (prefers AMD GPU by name string). The C++ `CLBackend` in the ops uses
the probe-based approach described above. They may therefore land on
different platforms in heterogeneous environments.

To align them, pass `--platform` to `stress_gpu.py`:

```bash
python tools/stress_gpu.py --platform Intel    # match the Intel iGPU
python tools/stress_gpu.py --platform rusticl  # match the AMD Rusticl path
```
