# Adding a new op

This is the playbook for adding a new layer to the OpenCL backend. We'll
walk through it using `DepthwiseConv2D` as the running example.

## 1. Write the OpenCL kernels

Drop a new `.cl` file in `kernels/`. For most layers you'll need three
kernels: forward, dL/dx, dL/dw. Keep the layout consistent with the rest
of the project: one work-item per output element on the forward pass, one
per input / parameter element on the backward passes, NHWC layout, every
kernel bounds-checks its global ID on entry.

```c
// kernels/depthwise_conv2d_kernels.cl
__kernel void dwconv2d_forward(
    __global const float* restrict input,    /* [N, H, W, C]              */
    __global const float* restrict filter,   /* [kH, kW, C, 1]            */
    __global       float* restrict output,   /* [N, Hout, Wout, C]        */
    /* ... shape + stride + pad params ... */
) {
    int id = get_global_id(0);
    int total = N * Hout * Wout * C;
    if (id >= total) return;   /* ← bounds check always required */
    /* ... */
}
```

Verify it compiles by running `clBuildProgram`; the C++ side will surface
the build log if anything is wrong.

## 2. Add a C++ op file under `src/ops/`

Create `src/ops/depthwise_conv2d_ops.cc`. Use the same skeleton the
Conv2D ops use:

```cpp
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "cl_backend.h"
#include "padding_utils.h"

REGISTER_OP("OpenclDepthwiseConv2d")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float")
    .Attr("strides: list(int) >= 4")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn(/* ... */);

class OpenclDepthwiseConv2dOp : public OpKernel {
 public:
  explicit OpenclDepthwiseConv2dOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* ctx) override {
    auto& cl = opencl_tf::CLBackend::Instance();
    cl_kernel k = cl.GetKernel("depthwise_conv2d_kernels.cl",
                               "dwconv2d_forward");

    /* 1. Read input tensors */
    const Tensor& input_t  = ctx->input(0);
    const Tensor& filter_t = ctx->input(1);

    /* 2. Allocate output tensor */
    Tensor* output_t = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output_t));

    /* 3. Allocate cl_mem buffers (thread-safe, outside the lock) */
    opencl_tf::CLMem d_in (cl, input_t.NumElements()  * sizeof(float));
    opencl_tf::CLMem d_flt(cl, filter_t.NumElements() * sizeof(float));
    opencl_tf::CLMem d_out(cl, output_t->NumElements() * sizeof(float));

    /* 4. Copy inputs to device, set args, enqueue, read back —
          ALL inside a single QueueMutex() lock */
    cl_int err = CL_SUCCESS;
    { std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueWriteBuffer(cl.Queue(), d_in.m, CL_TRUE, 0,
                                 input_t.NumElements() * sizeof(float),
                                 input_t.flat<float>().data(),
                                 0, nullptr, nullptr);
      /* ... write filter ... */
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_flt.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_out.m);
      /* ... int shape args ... */
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, 0,
                                output_t->NumElements() * sizeof(float),
                                output_t->flat<float>().data(),
                                0, nullptr, nullptr);
    }
    OP_REQUIRES(ctx, err == CL_SUCCESS,
                errors::Internal("OpenCL kernel failed: ", err));
  }
 private:
  std::vector<int32> strides_;
  std::string        padding_;
};
REGISTER_KERNEL_BUILDER(
    Name("OpenclDepthwiseConv2d").Device(DEVICE_CPU),
    OpenclDepthwiseConv2dOp);
```

Repeat for the two backward ops.

## 3. Add the new source file to the build

**CMake:** append to `OPENCL_TF_SOURCES` in `CMakeLists.txt`.

**Make:** append to `SRC` in `Makefile`.

## 4. Add a Python wrapper

Create `opencl_tf/ops/depthwise_conv2d.py`:

```python
from .._library import raw_ops

def depthwise_conv2d(x, w, strides=(1, 1, 1, 1), padding="SAME"):
    return raw_ops.opencl_depthwise_conv2d(
        x, w, strides=list(strides), padding=padding)
```

Re-export it from `opencl_tf/ops/__init__.py` and `opencl_tf/__init__.py`.

## 5. Register the gradient

Add to `opencl_tf/gradients.py`:

```python
@ops.RegisterGradient("OpenclDepthwiseConv2d")
def _opencl_dwconv2d_grad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    strides = list(op.get_attr("strides"))
    padding = op.get_attr("padding")
    grad_input = raw_ops.opencl_depthwise_conv2d_backprop_input(
        tf.shape(x), w, grad, strides=strides, padding=padding)
    grad_filter = raw_ops.opencl_depthwise_conv2d_backprop_filter(
        x, tf.shape(w), grad, strides=strides, padding=padding)
    return [grad_input, grad_filter]
```

The decorator key (`"OpenclDepthwiseConv2d"`) must match the
`REGISTER_OP` string exactly.

## 6. Add a Keras layer wrapper (optional but nice)

In `opencl_tf/layers.py`:

```python
class OpenCLDepthwiseConv2D(layers.Layer):
    """Drop-in for tf.keras.layers.DepthwiseConv2D(use_bias=False)."""
    # ... same shape as OpenCLConv2D but the kernel is [kH, kW, C, 1]
```

When using `add_weight()` in `build()`, always use keyword arguments —
Keras 3 requires `name=` to be a keyword argument:

```python
# Correct (Keras 3):
self.kernel = self.add_weight(name="kernel", shape=(...), ...)

# Wrong (breaks on Keras 3):
self.kernel = self.add_weight("kernel", shape=(...), ...)
```

## 7. Tests

Mirror `tests/test_conv2d.py`:

- Forward parity with `tf.nn.depthwise_conv2d` across SAME / VALID and
  a couple of stride values.
- Gradient parity via `tf.GradientTape`.
- A "the loss goes down" sanity check for the Keras layer.
- At least one concurrent-thread test (see `test_sigmoid.py` for the
  pattern) to verify there are no `clSetKernelArg` races.

## 8. Update the docs

Tick the row in the op table in `README.md` and update `ROADMAP.md`.

---

## Gotchas to keep in mind

### clSetKernelArg must be inside QueueMutex()

The kernel cache stores a single shared `cl_kernel` object per
`"file.cl:name"` key. `clSetKernelArg` on a shared kernel is **not**
thread-safe (OpenCL spec §5.9.1). TF schedules `Compute()` calls
concurrently from its inter-op thread pool.

**Both `clSetKernelArg` and `clEnqueue*` must be inside the same
`QueueMutex()` lock.** Putting only the enqueue call inside the lock is a
silent race that corrupts arguments under concurrent load.

```cpp
// ✓ Correct — all argument setting and enqueue in one critical section:
{ std::lock_guard<std::mutex> lk(cl.QueueMutex());
  int a = 0;
  clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
  clSetKernelArg(k, a++, sizeof(int),    &N);
  err = clEnqueueNDRangeKernel(cl.Queue(), k, ...);
  err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, ...);
}

// ✗ Wrong — race on the shared cl_kernel:
int a = 0;
clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);   // outside lock!
{ std::lock_guard<std::mutex> lk(cl.QueueMutex());
  err = clEnqueueNDRangeKernel(...);
}
```

### Op naming

Use `OpenclWhatever` (single capital after `Opencl`). Consecutive uppercase
letters get mangled by TF's snake-caser: `OpenCLConv2D` → `open_c_l_conv2_d`.
`OpenclConv2d` → the clean `opencl_conv2d`.

### Gradient input order

`@RegisterGradient` returns one entry per *forward input*. For a forward op
`(x, w) -> y`, return `[grad_x, grad_w]` in that order. Returning them
swapped will silently train the wrong variables.

### Don't hold the queue mutex across buffer allocation

`clCreateBuffer` is thread-safe; only `clEnqueue*` (and `clSetKernelArg`)
needs serialization. Allocate `CLMem` objects first, then lock.

### Asymmetric padding

TF's SAME pads `pad_total / 2` (integer division, rounding down) on the
left/top. `ResolvePadding` in `padding_utils.h` already handles this —
reuse it rather than computing padding inline.

### Bounds checks in kernels

Don't assume the global work size equals the output element count. The host
rounds it up to a multiple of the local size for wavefront alignment. Every
kernel must early-return on out-of-range global IDs:

```c
int id = get_global_id(0);
if (id >= total_elements) return;
```
