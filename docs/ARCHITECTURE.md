# Architecture

## Why `DEVICE_CPU` registration?

TensorFlow has two device types built into the public C++ API: `DEVICE_CPU`
and `DEVICE_GPU`. The `DEVICE_GPU` path is wired to TF's CUDA / ROCm
runtime. On a GPU that has no working CUDA or ROCm support — which is the
whole reason this project exists — registering on `DEVICE_GPU` would either
fail to load or never get scheduled.

So we register on `DEVICE_CPU`. From TF's perspective, our ops are CPU
kernels. Internally each `Compute()` call copies the host tensor into an
`cl_mem` buffer, runs an OpenCL kernel on the actual GPU, and copies the
output back. This costs us host↔device traffic on every op, but it's the
only path that works on the target hardware.

## The `CLBackend` singleton

`src/cl_backend.{h,cc}` owns all OpenCL state:

- One `cl_platform_id`, `cl_device_id`, `cl_context`, `cl_command_queue`.
- A program cache keyed by `.cl` filename.
- A kernel cache keyed by `"file.cl:kernel_name"`.
- Two mutexes:
  - `programs_mu_` protects the program / kernel maps (write-rare,
    read-often).
  - `queue_mu_` serializes command-queue access because `cl_command_queue`
    is **not** thread-safe per the OpenCL spec, and TF will call
    `Compute()` from multiple inter-op threads concurrently.

New ops should never construct their own OpenCL context. They call
`CLBackend::Instance().GetKernel(...)` to fetch a compiled kernel and lock
`QueueMutex()` around enqueue+read.

## Kernel-file path resolution

The `.so` resolves a `.cl` filename in this order:

1. Absolute path → use as-is.
2. `$OPENCL_TF_KERNELS_PATH/<file>`.
3. `./kernels/<file>` relative to the current working directory.
4. `<dir-containing-the-.so>/../kernels/<file>`.
5. `<dir-containing-the-.so>/kernels/<file>`.
6. `<dir-containing-the-.so>/<file>`.

Step 4–6 use `dladdr()` to find the loaded shared object on disk. The
Python `_library.py` also sets `OPENCL_TF_KERNELS_PATH` to the repo's
`kernels/` directory before loading the `.so`, so step 2 catches the
common case.

## Tensor layout

NHWC throughout — matches `tf.nn.conv2d`'s default. Filters are
`[kH, kW, Cin, Cout]`. Kernels assume contiguous tensors; TF guarantees
that for the tensors we receive as inputs and allocate as outputs.

## Padding

TF's `SAME` mode pads asymmetrically: when `pad_total` is odd, the extra
pixel goes on the right / bottom. We only encode `pad_before` (= `pad_total / 2`)
in the kernel args because every kernel bounds-checks each filter tap
against the input dimensions, so the asymmetric right / bottom side is
handled implicitly by the bounds check rejecting the would-be out-of-range
read.

## Per-call buffer allocation

Right now every `Compute()` call allocates three `cl_mem` buffers and
releases them at the end of the call. This is wasteful during training,
where the same tensor shapes recur every step. A buffer pool keyed by
`(size_bytes, role)` is on the Phase 4 list (see `ROADMAP.md`). Adding it
later is a localized change confined to `CLBackend`.

## Why direct convolution and not im2col + GEMM?

For Phase 1, correctness and readability beat speed. The direct kernels
fit on screen, match `tf.nn.conv2d` to 1e-4, and don't allocate the huge
`[N·Hout·Wout, kH·kW·Cin]` im2col matrix. Phase 3 will swap in a tiled
GEMM-style kernel and use the same op shells; the kernel file is the only
thing that needs to change.

## Why register the ops with names like `OpenclConv2d`?

TF's CamelCase → snake_case converter for op-to-Python names treats
consecutive uppercase letters and digit-letter transitions awkwardly.
`OpenCLConv2D` becomes `open_c_l_conv2_d`. `OpenclConv2d` becomes the
clean `opencl_conv2d`. The semantics are identical; only the spelling
changes.
