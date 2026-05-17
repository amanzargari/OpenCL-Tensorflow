// =====================================================================
// sigmoid_ops.cc
//
// Elementwise sigmoid forward + backward.
//
//   OpenclSigmoid
//     inputs : x  (any shape)
//     output : y  (same shape)
//
//   OpenclSigmoidGrad
//     inputs : y (forward OUTPUT, not x), dy
//     output : grad_x
//
// The backward takes the forward output y to compute grad_x = dy*y*(1-y),
// matching TF's SigmoidGrad convention.
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

using opencl_tf::CLBackend;
using opencl_tf::ClMem;
using opencl_tf::kDefaultLocalSize;
using opencl_tf::RoundUp;

namespace {
constexpr char kKernelFile[] = "sigmoid_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))
}  // namespace


// ---------------------------------------------------------------------
// Forward: y = sigmoid(x)
// ---------------------------------------------------------------------
REGISTER_OP("OpenclSigmoid")
    .Input("input: float")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class OpenclSigmoidOp : public OpKernel {
 public:
  explicit OpenclSigmoidOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in.shape(), &out));

    const int total = static_cast<int>(in.NumElements());
    if (total == 0) return;

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "sigmoid_forward"); }
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

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      clSetKernelArg(k, 0, sizeof(cl_mem), &d_in.m);
      clSetKernelArg(k, 1, sizeof(cl_mem), &d_out.m);
      clSetKernelArg(k, 2, sizeof(int),    &total);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue sigmoid_forward");
      err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, 0, bytes,
                                out->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read sigmoid_forward output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclSigmoid").Device(DEVICE_CPU), OpenclSigmoidOp);


// ---------------------------------------------------------------------
// Backward: grad_x = dy * y * (1 - y)
// Inputs: y (forward output), dy (upstream gradient)
// ---------------------------------------------------------------------
REGISTER_OP("OpenclSigmoidGrad")
    .Input("y: float")
    .Input("dy: float")
    .Output("grad_x: float")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class OpenclSigmoidGradOp : public OpKernel {
 public:
  explicit OpenclSigmoidGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& y  = ctx->input(0);
    const Tensor& dy = ctx->input(1);
    OP_REQUIRES(ctx, y.shape() == dy.shape(),
                errors::InvalidArgument("y and dy must have equal shape"));

    Tensor* grad_x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y.shape(), &grad_x));

    const int total = static_cast<int>(y.NumElements());
    if (total == 0) return;

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "sigmoid_backward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t bytes = (size_t)total * sizeof(float);

    ClMem d_dy(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<float*>(dy.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc dy");
    ClMem d_y (clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<float*>(y.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc y");
    ClMem d_gx(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_x");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      clSetKernelArg(k, 0, sizeof(cl_mem), &d_dy.m);
      clSetKernelArg(k, 1, sizeof(cl_mem), &d_y.m);
      clSetKernelArg(k, 2, sizeof(cl_mem), &d_gx.m);
      clSetKernelArg(k, 3, sizeof(int),    &total);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue sigmoid_backward");
      err = clEnqueueReadBuffer(cl.Queue(), d_gx.m, CL_TRUE, 0, bytes,
                                grad_x->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read sigmoid_backward output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclSigmoidGrad").Device(DEVICE_CPU),
                        OpenclSigmoidGradOp);
