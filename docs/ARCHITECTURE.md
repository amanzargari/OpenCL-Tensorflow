# Architecture

## Why `DEVICE_CPU` registration?

TensorFlow has two device types built into the public C++ API: `DEVICE_CPU`
and `DEVICE_GPU`. The `DEVICE_GPU` path is wired to TF's CUDA / ROCm
runtime. On a GPU that has no working CUDA or ROCm support — which is the
whole reason this project exists — registering on `DEVICE_GPU` would either
fail to load or never get scheduled.

So we register on `DEVICE_CPU`. From TF's perspective, our ops are CPU
kernels. Internally each `Compute()` call copies the host tensor into a
`cl_mem` device buffer, runs an OpenCL kernel on the actual GPU, and
blocking-reads the result back. This costs host↔device round-trip traffic
on every op, but it's the only path that works on the target hardware.

---

## The `CLBackend` singleton

`src/cl_backend.{h,cc}` owns all OpenCL state:

- One `cl_platform_id`, `cl_device_id`, `cl_context`, `cl_command_queue`.
- A program cache keyed by `.cl` filename.
- A kernel cache keyed by `"file.cl:kernel_name"`.
- Two mutexes:
  - `programs_mu_` — protects the program / kernel maps (written rarely,
    read constantly).
  - `queue_mu_` — serializes command-queue access because `cl_command_queue`
    is **not** thread-safe per the OpenCL spec, and TF will call
    `Compute()` from multiple inter-op threads concurrently.

New ops should never construct their own OpenCL context. They call
`CLBackend::Instance().GetKernel(...)` to fetch a compiled kernel and lock
`QueueMutex()` around all enqueue and read-back calls.

---

## Thread safety — the clSetKernelArg rule

The kernel cache maps `"file.cl:kernel_name"` to a single shared
`cl_kernel` object that is reused across calls. The OpenCL spec (§5.9.1)
states:

> `clSetKernelArg` is not thread-safe when the same `cl_kernel` object is
> modified from multiple threads.

Because TF can issue concurrent `Compute()` calls on the same op, **both
`clSetKernelArg` and `clEnqueue*` calls must be inside `QueueMutex()`**.
Having only the `clEnqueue*` call inside the lock is a race condition that
will corrupt arguments silently at high concurrency. The correct pattern in
every op:

```cpp
// ✓ Correct: args + enqueue + read all inside the same lock.
{ std::lock_guard<std::mutex> lk(cl.QueueMutex());
  int a = 0;
  clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
  clSetKernelArg(k, a++, sizeof(int),    &N);
  err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                               &global, &local, 0, nullptr, nullptr);
  err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, ...);
}

// ✗ Wrong: args outside the lock — race on the shared cl_kernel.
int a = 0;
clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
clSetKernelArg(k, a++, sizeof(int),    &N);
{ std::lock_guard<std::mutex> lk(cl.QueueMutex());
  err = clEnqueueNDRangeKernel(...);
  err = clEnqueueReadBuffer(...);
}
```

Buffer allocation (`CLMem` RAII) is thread-safe and happens before the
lock, so you are not holding the queue mutex any longer than necessary.

---

## Platform and device selection

At first initialization the singleton:

1. Enumerates all installed OpenCL platforms via `clGetPlatformIDs`.
2. Builds a candidate list: GPU devices first (two-pass, GPUs before the
   catch-all `CL_DEVICE_TYPE_ALL`), de-duplicated.
3. For each candidate, creates a throwaway `cl_context`, tries to compile
   `"kernel void _probe() {}"`, and releases the context.
4. Selects the first candidate where compilation succeeds.

This probe catches environments with broken OpenCL runtimes — for example,
Mesa Rusticl on a machine where LLVM was built without the SPIR-V backend.
The broken platform is silently skipped; the next one (e.g., Intel's OpenCL
GPU runtime) is used instead.

To force a specific platform, set `OCL_ICD_VENDORS` to the path of a
single `.icd` file before running:

```bash
OCL_ICD_VENDORS=/etc/OpenCL/vendors/intel.icd python ...
```

---

## Kernel-file path resolution

The `.so` resolves a `.cl` filename in this order:

1. Absolute path → use as-is.
2. `$OPENCL_TF_KERNELS_PATH/<file>` — set by `opencl_tf/_library.py` at
   import time so packaged installs always work.
3. `./kernels/<file>` relative to the current working directory.
4. `<dir-containing-the-.so>/../kernels/<file>`.
5. `<dir-containing-the-.so>/kernels/<file>`.
6. `<dir-containing-the-.so>/<file>`.

Steps 4–6 use `dladdr()` to locate the shared object on disk.

---

## Tensor layout

NHWC throughout — matches `tf.nn.conv2d`'s default. Filters are
`[kH, kW, Cin, Cout]` for Conv2D and `[kH, kW, Cin, depth_multiplier]`
for DepthwiseConv2D. Kernels assume contiguous tensors; TF guarantees
this for the tensors passed to `Compute()`.

---

## Padding

TF's `SAME` mode pads asymmetrically: when `pad_total` is odd the extra
pixel goes on the right / bottom. We only pass `pad_before` (=
`pad_total / 2`, integer division) to the kernel. Each kernel bounds-checks
every filter-tap against the input dimensions, so the asymmetric
right / bottom side is handled implicitly: the tap that would land
out-of-range returns zero, exactly as if the pad pixel were there.

---

## Per-call buffer allocation

Every `Compute()` call allocates `cl_mem` buffers via the `CLMem` RAII
helper and releases them when the function returns. This trades performance
for simplicity; on the hot training path the same shapes repeat every step.

A buffer pool keyed by `(size_bytes, role)` is the first item on the
Phase 4 list (see `ROADMAP.md`). Adding it later is a localized change to
`CLBackend` with no changes required in any op file.

---

## Why direct convolution and not im2col + GEMM?

For Phase 1, correctness and readability beat speed. The direct kernels are
compact, match `tf.nn.conv2d` to 1 × 10⁻⁴, and avoid allocating the large
`[N · Hout · Wout, kH · kW · Cin]` im2col matrix. Phase 4 will add a
tiled GEMM-style kernel inside the same op shell — only the `.cl` file
changes.

---

## Why register ops as `OpenclConv2d` (not `OpenCLConv2D`)?

TF's CamelCase → snake_case converter mangles consecutive uppercase letters
and digit-letter transitions. `OpenCLConv2D` becomes `open_c_l_conv2_d`.
`OpenclConv2d` becomes the clean `opencl_conv2d`. The semantics are
identical; only the spelling changes.
