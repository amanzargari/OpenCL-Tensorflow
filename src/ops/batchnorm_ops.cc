// =====================================================================
// batchnorm_ops.cc
//
// Three custom ops for BatchNormalization in NHWC:
//
//   OpenclBatchNormTraining
//     inputs : x [N,H,W,C], gamma [C], beta [C]
//     attrs  : epsilon (float)
//     outputs: y [N,H,W,C], saved_mean [C], saved_var [C]
//     (saved_mean / saved_var are the batch statistics; needed both as
//      side outputs for the Keras layer's EMA update and as inputs to
//      the backward pass.)
//
//   OpenclBatchNormInference  (no gradient registered)
//     inputs : x, gamma, beta, mean, var
//     attrs  : epsilon
//     outputs: y
//
//   OpenclBatchNormGrad
//     inputs : grad_y, x, gamma, saved_mean, saved_var
//     attrs  : epsilon
//     outputs: grad_x, grad_gamma, grad_beta
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
constexpr char kKernelFile[] = "batchnorm_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))

// Helper: copy a host float pointer into a fresh read-only device buffer.
inline cl_mem CopyHostToDevice(cl_context ctx, const float* host, size_t bytes,
                               cl_int* err) {
  return clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                        const_cast<float*>(host), err);
}
}  // namespace


// =====================================================================
// FORWARD (training)
// =====================================================================
REGISTER_OP("OpenclBatchNormTraining")
    .Input("x: float")
    .Input("gamma: float")
    .Input("beta: float")
    .Output("y: float")
    .Output("saved_mean: float")
    .Output("saved_var: float")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &x));
      c->set_output(0, x);
      c->set_output(1, c->Vector(c->Dim(x, 3)));
      c->set_output(2, c->Vector(c->Dim(x, 3)));
      return absl::OkStatus();
    });

class OpenclBatchNormTrainingOp : public OpKernel {
 public:
  explicit OpenclBatchNormTrainingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x     = ctx->input(0);
    const Tensor& gamma = ctx->input(1);
    const Tensor& beta  = ctx->input(2);
    OP_REQUIRES(ctx, x.dims() == 4, errors::InvalidArgument("x must be NHWC"));
    const int N = x.dim_size(0), H = x.dim_size(1),
              W = x.dim_size(2), C = x.dim_size(3);
    OP_REQUIRES(ctx, gamma.NumElements() == C && beta.NumElements() == C,
                errors::InvalidArgument("gamma/beta size must equal channels"));

    Tensor* y    = nullptr;
    Tensor* mean = nullptr;
    Tensor* var  = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {C}, &mean));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {C}, &var));

    auto& cl = CLBackend::Instance();
    cl_kernel k_reduce, k_norm;
    try {
      k_reduce = cl.GetKernel(kKernelFile, "bn_reduce_stats");
      k_norm   = cl.GetKernel(kKernelFile, "bn_normalize");
    } catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t x_bytes = (size_t)N * H * W * C * sizeof(float);
    const size_t c_bytes = (size_t)C * sizeof(float);

    ClMem d_x   (CopyHostToDevice(cl.Context(), x.flat<float>().data(),     x_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc x");
    ClMem d_gam (CopyHostToDevice(cl.Context(), gamma.flat<float>().data(), c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc gamma");
    ClMem d_bet (CopyHostToDevice(cl.Context(), beta.flat<float>().data(),  c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc beta");
    ClMem d_mean(clCreateBuffer(cl.Context(), CL_MEM_READ_WRITE,  c_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc mean");
    ClMem d_var (clCreateBuffer(cl.Context(), CL_MEM_READ_WRITE,  c_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc var");
    ClMem d_y   (clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY,  x_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc y");

    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());

      // ---- Stage 1: per-channel mean / var -----------------------
      int a = 0;
      clSetKernelArg(k_reduce, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k_reduce, a++, sizeof(cl_mem), &d_mean.m);
      clSetKernelArg(k_reduce, a++, sizeof(cl_mem), &d_var.m);
      clSetKernelArg(k_reduce, a++, sizeof(int),    &N);
      clSetKernelArg(k_reduce, a++, sizeof(int),    &H);
      clSetKernelArg(k_reduce, a++, sizeof(int),    &W);
      clSetKernelArg(k_reduce, a++, sizeof(int),    &C);
      const size_t local_c  = kDefaultLocalSize;
      const size_t global_c = RoundUp((size_t)C, local_c);
      err = clEnqueueNDRangeKernel(cl.Queue(), k_reduce, 1, nullptr,
                                   &global_c, &local_c, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue bn_reduce_stats");

      // ---- Stage 2: elementwise normalize ------------------------
      a = 0;
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_mean.m);
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_var.m);
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_gam.m);
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_bet.m);
      clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_y.m);
      clSetKernelArg(k_norm, a++, sizeof(int),    &N);
      clSetKernelArg(k_norm, a++, sizeof(int),    &H);
      clSetKernelArg(k_norm, a++, sizeof(int),    &W);
      clSetKernelArg(k_norm, a++, sizeof(int),    &C);
      clSetKernelArg(k_norm, a++, sizeof(float),  &epsilon_);

      const size_t local_n  = kDefaultLocalSize;
      const size_t global_n = RoundUp((size_t)N * H * W * C, local_n);
      err = clEnqueueNDRangeKernel(cl.Queue(), k_norm, 1, nullptr,
                                   &global_n, &local_n, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue bn_normalize");

      // ---- Read back all three outputs ---------------------------
      err = clEnqueueReadBuffer(cl.Queue(), d_y.m,    CL_FALSE, 0, x_bytes,
                                y->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read y");
      err = clEnqueueReadBuffer(cl.Queue(), d_mean.m, CL_FALSE, 0, c_bytes,
                                mean->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read mean");
      err = clEnqueueReadBuffer(cl.Queue(), d_var.m,  CL_TRUE,  0, c_bytes,
                                var->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read var (blocking)");
    }
  }

 private:
  float epsilon_ = 1e-3f;
};
REGISTER_KERNEL_BUILDER(Name("OpenclBatchNormTraining").Device(DEVICE_CPU),
                        OpenclBatchNormTrainingOp);


// =====================================================================
// FORWARD (inference)  -- no gradient
// =====================================================================
REGISTER_OP("OpenclBatchNormInference")
    .Input("x: float")
    .Input("gamma: float")
    .Input("beta: float")
    .Input("mean: float")
    .Input("var: float")
    .Output("y: float")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

class OpenclBatchNormInferenceOp : public OpKernel {
 public:
  explicit OpenclBatchNormInferenceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x     = ctx->input(0);
    const Tensor& gamma = ctx->input(1);
    const Tensor& beta  = ctx->input(2);
    const Tensor& mean  = ctx->input(3);
    const Tensor& var   = ctx->input(4);
    OP_REQUIRES(ctx, x.dims() == 4, errors::InvalidArgument("x must be NHWC"));
    const int N = x.dim_size(0), H = x.dim_size(1),
              W = x.dim_size(2), C = x.dim_size(3);
    OP_REQUIRES(ctx,
                gamma.NumElements()==C && beta.NumElements()==C &&
                mean .NumElements()==C && var .NumElements()==C,
                errors::InvalidArgument("gamma/beta/mean/var must have C elements"));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    auto& cl = CLBackend::Instance();
    cl_kernel k_norm;
    try { k_norm = cl.GetKernel(kKernelFile, "bn_normalize"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t x_bytes = (size_t)N * H * W * C * sizeof(float);
    const size_t c_bytes = (size_t)C * sizeof(float);

    ClMem d_x   (CopyHostToDevice(cl.Context(), x.flat<float>().data(),     x_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc x");
    ClMem d_mean(CopyHostToDevice(cl.Context(), mean.flat<float>().data(),  c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc mean");
    ClMem d_var (CopyHostToDevice(cl.Context(), var.flat<float>().data(),   c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc var");
    ClMem d_gam (CopyHostToDevice(cl.Context(), gamma.flat<float>().data(), c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc gamma");
    ClMem d_bet (CopyHostToDevice(cl.Context(), beta.flat<float>().data(),  c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc beta");
    ClMem d_y   (clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, x_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc y");

    int a = 0;
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_x.m);
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_mean.m);
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_var.m);
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_gam.m);
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_bet.m);
    clSetKernelArg(k_norm, a++, sizeof(cl_mem), &d_y.m);
    clSetKernelArg(k_norm, a++, sizeof(int),    &N);
    clSetKernelArg(k_norm, a++, sizeof(int),    &H);
    clSetKernelArg(k_norm, a++, sizeof(int),    &W);
    clSetKernelArg(k_norm, a++, sizeof(int),    &C);
    clSetKernelArg(k_norm, a++, sizeof(float),  &epsilon_);

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)N * H * W * C, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k_norm, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue bn_normalize (inf)");
      err = clEnqueueReadBuffer(cl.Queue(), d_y.m, CL_TRUE, 0, x_bytes,
                                y->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read y (inf)");
    }
  }

 private:
  float epsilon_ = 1e-3f;
};
REGISTER_KERNEL_BUILDER(Name("OpenclBatchNormInference").Device(DEVICE_CPU),
                        OpenclBatchNormInferenceOp);


// =====================================================================
// BACKWARD
// =====================================================================
REGISTER_OP("OpenclBatchNormGrad")
    .Input("grad_y: float")
    .Input("x: float")
    .Input("gamma: float")
    .Input("saved_mean: float")
    .Input("saved_var: float")
    .Output("grad_x: float")
    .Output("grad_gamma: float")
    .Output("grad_beta: float")
    .Attr("epsilon: float = 0.001")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &x));
      c->set_output(0, x);
      c->set_output(1, c->Vector(c->Dim(x, 3)));
      c->set_output(2, c->Vector(c->Dim(x, 3)));
      return absl::OkStatus();
    });

class OpenclBatchNormGradOp : public OpKernel {
 public:
  explicit OpenclBatchNormGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& go    = ctx->input(0);
    const Tensor& x     = ctx->input(1);
    const Tensor& gamma = ctx->input(2);
    const Tensor& mean  = ctx->input(3);
    const Tensor& var   = ctx->input(4);
    OP_REQUIRES(ctx, x.dims() == 4 && go.shape() == x.shape(),
                errors::InvalidArgument("grad_y and x must have matching NHWC shapes"));
    const int N = x.dim_size(0), H = x.dim_size(1),
              W = x.dim_size(2), C = x.dim_size(3);

    Tensor* grad_x = nullptr;
    Tensor* grad_gamma = nullptr;
    Tensor* grad_beta = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &grad_x));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {C}, &grad_gamma));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {C}, &grad_beta));

    auto& cl = CLBackend::Instance();
    cl_kernel k_red, k_dx;
    try {
      k_red = cl.GetKernel(kKernelFile, "bn_backward_reduce");
      k_dx  = cl.GetKernel(kKernelFile, "bn_backward_dx");
    } catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t x_bytes = (size_t)N * H * W * C * sizeof(float);
    const size_t c_bytes = (size_t)C * sizeof(float);

    ClMem d_go  (CopyHostToDevice(cl.Context(), go.flat<float>().data(),    x_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_y");
    ClMem d_x   (CopyHostToDevice(cl.Context(), x.flat<float>().data(),     x_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc x");
    ClMem d_gam (CopyHostToDevice(cl.Context(), gamma.flat<float>().data(), c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc gamma");
    ClMem d_mean(CopyHostToDevice(cl.Context(), mean.flat<float>().data(),  c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc mean");
    ClMem d_var (CopyHostToDevice(cl.Context(), var.flat<float>().data(),   c_bytes, &err));
    OP_REQUIRES_CL(ctx, err, "alloc var");
    ClMem d_gbeta (clCreateBuffer(cl.Context(), CL_MEM_READ_WRITE, c_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_beta");
    ClMem d_ggamma(clCreateBuffer(cl.Context(), CL_MEM_READ_WRITE, c_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_gamma");
    ClMem d_gx    (clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, x_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_x");

    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());

      // ---- Stage 1: reduce sum_dy and sum_dy_xhat per channel ----
      int a = 0;
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_go.m);
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_mean.m);
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_var.m);
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_gbeta.m);
      clSetKernelArg(k_red, a++, sizeof(cl_mem), &d_ggamma.m);
      clSetKernelArg(k_red, a++, sizeof(int),    &N);
      clSetKernelArg(k_red, a++, sizeof(int),    &H);
      clSetKernelArg(k_red, a++, sizeof(int),    &W);
      clSetKernelArg(k_red, a++, sizeof(int),    &C);
      clSetKernelArg(k_red, a++, sizeof(float),  &epsilon_);
      const size_t local_c  = kDefaultLocalSize;
      const size_t global_c = RoundUp((size_t)C, local_c);
      err = clEnqueueNDRangeKernel(cl.Queue(), k_red, 1, nullptr,
                                   &global_c, &local_c, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue bn_backward_reduce");

      // ---- Stage 2: elementwise dx ------------------------------
      a = 0;
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_go.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_x.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_mean.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_var.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_gam.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_gbeta.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_ggamma.m);
      clSetKernelArg(k_dx, a++, sizeof(cl_mem), &d_gx.m);
      clSetKernelArg(k_dx, a++, sizeof(int),    &N);
      clSetKernelArg(k_dx, a++, sizeof(int),    &H);
      clSetKernelArg(k_dx, a++, sizeof(int),    &W);
      clSetKernelArg(k_dx, a++, sizeof(int),    &C);
      clSetKernelArg(k_dx, a++, sizeof(float),  &epsilon_);
      const size_t local_n  = kDefaultLocalSize;
      const size_t global_n = RoundUp((size_t)N * H * W * C, local_n);
      err = clEnqueueNDRangeKernel(cl.Queue(), k_dx, 1, nullptr,
                                   &global_n, &local_n, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue bn_backward_dx");

      // ---- Read back ---------------------------------------------
      err = clEnqueueReadBuffer(cl.Queue(), d_gx.m,     CL_FALSE, 0, x_bytes,
                                grad_x->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read grad_x");
      err = clEnqueueReadBuffer(cl.Queue(), d_ggamma.m, CL_FALSE, 0, c_bytes,
                                grad_gamma->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read grad_gamma");
      err = clEnqueueReadBuffer(cl.Queue(), d_gbeta.m,  CL_TRUE,  0, c_bytes,
                                grad_beta->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read grad_beta (blocking)");
    }
  }

 private:
  float epsilon_ = 1e-3f;
};
REGISTER_KERNEL_BUILDER(Name("OpenclBatchNormGrad").Device(DEVICE_CPU),
                        OpenclBatchNormGradOp);
