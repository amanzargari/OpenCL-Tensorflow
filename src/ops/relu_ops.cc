// =====================================================================
// relu_ops.cc
//
// Elementwise ReLU forward + backward. Tensor shape is arbitrary; we
// only pass the element count.
// =====================================================================

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

#include "cl_backend.h"

#include <vector>

using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using opencl_tf::CLBackend;
using opencl_tf::ClMem;
using opencl_tf::kDefaultLocalSize;
using opencl_tf::RoundUp;

namespace {
constexpr char kKernelFile[] = "relu_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))
}  // namespace


// ---------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------
REGISTER_OP("OpenclRelu")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class OpenclReluOp : public OpKernel {
 public:
  explicit OpenclReluOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));

    const int total = static_cast<int>(in.NumElements());
    if (total == 0) return;

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "relu_forward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)total * sizeof(float);
    ClMem d_in (clCreateBuffer(cl.Context(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                               const_cast<float*>(in.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc input");
    ClMem d_out(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc output");

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_in.m);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_out.m);
    clSetKernelArg(k, 2, sizeof(int),    &total);

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue relu_forward");
      err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, 0, bytes,
                                out->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read relu_forward output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclRelu").Device(DEVICE_CPU), OpenclReluOp);


// ---------------------------------------------------------------------
// Backward
// Inputs match TF's ReluGrad: (gradients, features). Features is the
// original input to the forward Relu.
// ---------------------------------------------------------------------
REGISTER_OP("OpenclReluGrad")
    .Input("gradients: float")
    .Input("features: float")
    .Output("backprops: float")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class OpenclReluGradOp : public OpKernel {
 public:
  explicit OpenclReluGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& go = ctx->input(0);
    const Tensor& x  = ctx->input(1);
    OP_REQUIRES(ctx, go.shape() == x.shape(),
                errors::InvalidArgument("gradients and features must have equal shape"));

    Tensor* gi = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, go.shape(), &gi));

    const int total = static_cast<int>(go.NumElements());
    if (total == 0) return;

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "relu_backward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)total * sizeof(float);

    ClMem d_go(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<float*>(go.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_out");
    ClMem d_x (clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<float*>(x.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc features");
    ClMem d_gi(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_in");

    clSetKernelArg(k, 0, sizeof(cl_mem), &d_go.m);
    clSetKernelArg(k, 1, sizeof(cl_mem), &d_x.m);
    clSetKernelArg(k, 2, sizeof(cl_mem), &d_gi.m);
    clSetKernelArg(k, 3, sizeof(int),    &total);

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue relu_backward");
      err = clEnqueueReadBuffer(cl.Queue(), d_gi.m, CL_TRUE, 0, bytes,
                                gi->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read relu_backward output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclReluGrad").Device(DEVICE_CPU), OpenclReluGradOp);
