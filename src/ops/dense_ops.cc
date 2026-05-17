// =====================================================================
// dense_ops.cc
//
// Fully-connected (Dense) layer: forward + three backward ops.
//
//   OpenclDense
//     inputs : x [batch, in_features], W [in_features, out_features],
//              b [out_features]
//     output : y [batch, out_features]
//
//   OpenclDenseBackpropInput
//     inputs : grad_y [batch, out_features], W [in_features, out_features]
//     output : grad_x [batch, in_features]
//
//   OpenclDenseBackpropWeight
//     inputs : x [batch, in_features], grad_y [batch, out_features]
//     output : grad_W [in_features, out_features]
//
//   OpenclDenseBackpropBias
//     inputs : grad_y [batch, out_features]
//     output : grad_b [out_features]
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
using shape_inference::DimensionHandle;

using opencl_tf::CLBackend;
using opencl_tf::ClMem;
using opencl_tf::kDefaultLocalSize;
using opencl_tf::RoundUp;

namespace {
constexpr char kKernelFile[] = "dense_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))
}  // namespace


// =====================================================================
// FORWARD  y = x @ W + b
// =====================================================================
REGISTER_OP("OpenclDense")
    .Input("x: float")
    .Input("w: float")
    .Input("b: float")
    .Output("y: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, w;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &w));
      DimensionHandle batch   = c->Dim(x, 0);
      DimensionHandle out_f   = c->Dim(w, 1);
      c->set_output(0, c->MakeShape({batch, out_f}));
      return absl::OkStatus();
    });

class OpenclDenseOp : public OpKernel {
 public:
  explicit OpenclDenseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& W = ctx->input(1);
    const Tensor& b = ctx->input(2);

    OP_REQUIRES(ctx, x.dims() == 2, errors::InvalidArgument("x must be rank 2"));
    OP_REQUIRES(ctx, W.dims() == 2, errors::InvalidArgument("W must be rank 2"));
    OP_REQUIRES(ctx, b.dims() == 1, errors::InvalidArgument("b must be rank 1"));

    const int batch = x.dim_size(0);
    const int in_f  = x.dim_size(1);
    const int out_f = W.dim_size(1);

    OP_REQUIRES(ctx, W.dim_size(0) == in_f,
                errors::InvalidArgument("W[0] must equal x[1] (in_features)"));
    OP_REQUIRES(ctx, b.dim_size(0) == out_f,
                errors::InvalidArgument("b[0] must equal W[1] (out_features)"));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, out_f}, &y));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dense_forward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t x_bytes = (size_t)batch * in_f  * sizeof(float);
    const size_t W_bytes = (size_t)in_f  * out_f * sizeof(float);
    const size_t b_bytes = (size_t)out_f          * sizeof(float);
    const size_t y_bytes = (size_t)batch * out_f * sizeof(float);

    ClMem d_x(clCreateBuffer(cl.Context(),
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_bytes,
                             const_cast<float*>(x.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc x");
    ClMem d_W(clCreateBuffer(cl.Context(),
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, W_bytes,
                             const_cast<float*>(W.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc W");
    ClMem d_b(clCreateBuffer(cl.Context(),
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b_bytes,
                             const_cast<float*>(b.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc b");
    ClMem d_y(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, y_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc y");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)batch * out_f, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_W.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_b.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_y.m);
      clSetKernelArg(k, a++, sizeof(int),    &batch);
      clSetKernelArg(k, a++, sizeof(int),    &in_f);
      clSetKernelArg(k, a++, sizeof(int),    &out_f);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dense_forward");
      err = clEnqueueReadBuffer(cl.Queue(), d_y.m, CL_TRUE, 0, y_bytes,
                                y->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dense_forward output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclDense").Device(DEVICE_CPU), OpenclDenseOp);


// =====================================================================
// BACKPROP INPUT  grad_x = grad_y @ W^T
// =====================================================================
REGISTER_OP("OpenclDenseBackpropInput")
    .Input("grad_y: float")
    .Input("w: float")
    .Output("grad_x: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle gy, w;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &gy));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &w));
      DimensionHandle batch = c->Dim(gy, 0);
      DimensionHandle in_f  = c->Dim(w, 0);
      c->set_output(0, c->MakeShape({batch, in_f}));
      return absl::OkStatus();
    });

class OpenclDenseBackpropInputOp : public OpKernel {
 public:
  explicit OpenclDenseBackpropInputOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_y = ctx->input(0);
    const Tensor& W      = ctx->input(1);

    OP_REQUIRES(ctx, grad_y.dims() == 2, errors::InvalidArgument("grad_y must be rank 2"));
    OP_REQUIRES(ctx, W.dims()      == 2, errors::InvalidArgument("W must be rank 2"));

    const int batch = grad_y.dim_size(0);
    const int out_f = grad_y.dim_size(1);
    const int in_f  = W.dim_size(0);
    OP_REQUIRES(ctx, W.dim_size(1) == out_f,
                errors::InvalidArgument("W[1] must equal grad_y[1]"));

    Tensor* grad_x = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, in_f}, &grad_x));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dense_backprop_input"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t gy_bytes = (size_t)batch * out_f * sizeof(float);
    const size_t W_bytes  = (size_t)in_f  * out_f * sizeof(float);
    const size_t gx_bytes = (size_t)batch * in_f  * sizeof(float);

    ClMem d_gy(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gy_bytes,
                              const_cast<float*>(grad_y.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_y");
    ClMem d_W (clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, W_bytes,
                              const_cast<float*>(W.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc W");
    ClMem d_gx(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, gx_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_x");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)batch * in_f, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gy.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_W.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gx.m);
      clSetKernelArg(k, a++, sizeof(int),    &batch);
      clSetKernelArg(k, a++, sizeof(int),    &in_f);
      clSetKernelArg(k, a++, sizeof(int),    &out_f);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dense_backprop_input");
      err = clEnqueueReadBuffer(cl.Queue(), d_gx.m, CL_TRUE, 0, gx_bytes,
                                grad_x->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dense_backprop_input output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclDenseBackpropInput").Device(DEVICE_CPU),
                        OpenclDenseBackpropInputOp);


// =====================================================================
// BACKPROP WEIGHT  grad_W = x^T @ grad_y
// =====================================================================
REGISTER_OP("OpenclDenseBackpropWeight")
    .Input("x: float")
    .Input("grad_y: float")
    .Output("grad_w: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x, gy;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &gy));
      DimensionHandle in_f  = c->Dim(x, 1);
      DimensionHandle out_f = c->Dim(gy, 1);
      c->set_output(0, c->MakeShape({in_f, out_f}));
      return absl::OkStatus();
    });

class OpenclDenseBackpropWeightOp : public OpKernel {
 public:
  explicit OpenclDenseBackpropWeightOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x      = ctx->input(0);
    const Tensor& grad_y = ctx->input(1);

    OP_REQUIRES(ctx, x.dims()      == 2, errors::InvalidArgument("x must be rank 2"));
    OP_REQUIRES(ctx, grad_y.dims() == 2, errors::InvalidArgument("grad_y must be rank 2"));

    const int batch = x.dim_size(0);
    const int in_f  = x.dim_size(1);
    const int out_f = grad_y.dim_size(1);
    OP_REQUIRES(ctx, grad_y.dim_size(0) == batch,
                errors::InvalidArgument("x and grad_y must have the same batch size"));

    Tensor* grad_W = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {in_f, out_f}, &grad_W));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dense_backprop_weight"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t x_bytes  = (size_t)batch * in_f  * sizeof(float);
    const size_t gy_bytes = (size_t)batch * out_f * sizeof(float);
    const size_t gW_bytes = (size_t)in_f  * out_f * sizeof(float);

    ClMem d_x (clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x_bytes,
                              const_cast<float*>(x.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc x");
    ClMem d_gy(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gy_bytes,
                              const_cast<float*>(grad_y.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_y");
    ClMem d_gW(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, gW_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_W");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)in_f * out_f, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gy.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gW.m);
      clSetKernelArg(k, a++, sizeof(int),    &batch);
      clSetKernelArg(k, a++, sizeof(int),    &in_f);
      clSetKernelArg(k, a++, sizeof(int),    &out_f);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dense_backprop_weight");
      err = clEnqueueReadBuffer(cl.Queue(), d_gW.m, CL_TRUE, 0, gW_bytes,
                                grad_W->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dense_backprop_weight output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclDenseBackpropWeight").Device(DEVICE_CPU),
                        OpenclDenseBackpropWeightOp);


// =====================================================================
// BACKPROP BIAS  grad_b = sum_n grad_y[n, :]
// =====================================================================
REGISTER_OP("OpenclDenseBackpropBias")
    .Input("grad_y: float")
    .Output("grad_b: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle gy;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &gy));
      c->set_output(0, c->Vector(c->Dim(gy, 1)));
      return absl::OkStatus();
    });

class OpenclDenseBackpropBiasOp : public OpKernel {
 public:
  explicit OpenclDenseBackpropBiasOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_y = ctx->input(0);
    OP_REQUIRES(ctx, grad_y.dims() == 2,
                errors::InvalidArgument("grad_y must be rank 2"));

    const int batch = grad_y.dim_size(0);
    const int out_f = grad_y.dim_size(1);

    Tensor* grad_b = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {out_f}, &grad_b));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dense_backprop_bias"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t gy_bytes = (size_t)batch * out_f * sizeof(float);
    const size_t gb_bytes = (size_t)out_f          * sizeof(float);

    ClMem d_gy(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, gy_bytes,
                              const_cast<float*>(grad_y.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_y");
    ClMem d_gb(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, gb_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_b");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)out_f, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gy.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gb.m);
      clSetKernelArg(k, a++, sizeof(int),    &batch);
      clSetKernelArg(k, a++, sizeof(int),    &out_f);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dense_backprop_bias");
      err = clEnqueueReadBuffer(cl.Queue(), d_gb.m, CL_TRUE, 0, gb_bytes,
                                grad_b->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dense_backprop_bias output");
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("OpenclDenseBackpropBias").Device(DEVICE_CPU),
                        OpenclDenseBackpropBiasOp);
