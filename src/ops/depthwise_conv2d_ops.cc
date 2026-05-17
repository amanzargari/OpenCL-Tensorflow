// =====================================================================
// depthwise_conv2d_ops.cc
//
// TF custom-op wrappers for depthwise 2D convolution. Pattern mirrors
// conv2d_ops.cc — filter shape is [kH, kW, C, depth_multiplier], the
// only structural difference is the lack of Cin reduction.
// =====================================================================

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

#include "cl_backend.h"
#include "padding_utils.h"

#include <string>
#include <vector>

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using opencl_tf::CLBackend;
using opencl_tf::ClMem;
using opencl_tf::kDefaultLocalSize;
using opencl_tf::ResolvePadding;
using opencl_tf::RoundUp;

namespace {

constexpr char kKernelFile[] = "depthwise_conv2d_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))

}  // namespace


// =====================================================================
// FORWARD
// =====================================================================
REGISTER_OP("OpenclDepthwiseConv2d")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float")
    .Attr("strides: list(int) >= 4")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle in_shape, fil_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &in_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &fil_shape));
      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      std::string padding;
      TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

      DimensionHandle batch  = c->Dim(in_shape, 0);
      DimensionHandle in_h   = c->Dim(in_shape, 1);
      DimensionHandle in_w   = c->Dim(in_shape, 2);
      DimensionHandle in_ch  = c->Dim(in_shape, 3);
      DimensionHandle k_h    = c->Dim(fil_shape, 0);
      DimensionHandle k_w    = c->Dim(fil_shape, 1);
      DimensionHandle dm     = c->Dim(fil_shape, 3);

      DimensionHandle out_ch;
      TF_RETURN_IF_ERROR(c->Multiply(in_ch, dm, &out_ch));

      auto compute = [&](DimensionHandle in, DimensionHandle k, int stride,
                         DimensionHandle* out) -> Status {
        if (c->ValueKnown(in) && c->ValueKnown(k)) {
          int64_t i = c->Value(in), kk = c->Value(k), o;
          if (padding == "SAME") o = (i + stride - 1) / stride;
          else                   o = (i - kk + stride) / stride;
          *out = c->MakeDim(o);
        } else {
          *out = c->UnknownDim();
        }
        return absl::OkStatus();
      };

      DimensionHandle out_h, out_w;
      TF_RETURN_IF_ERROR(compute(in_h, k_h, strides[1], &out_h));
      TF_RETURN_IF_ERROR(compute(in_w, k_w, strides[2], &out_w));

      c->set_output(0, c->MakeShape({batch, out_h, out_w, out_ch}));
      return absl::OkStatus();
    });

class OpenclDepthwiseConv2dOp : public OpKernel {
 public:
  explicit OpenclDepthwiseConv2dOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
    OP_REQUIRES(ctx, strides_.size() == 4,
                errors::InvalidArgument("strides must be length 4 (NHWC)"));
    OP_REQUIRES(ctx, strides_[0] == 1 && strides_[3] == 1,
                errors::InvalidArgument("strides on N and C must be 1"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in  = ctx->input(0);
    const Tensor& fil = ctx->input(1);

    const int N  = in.dim_size(0);
    const int H  = in.dim_size(1);
    const int W  = in.dim_size(2);
    const int C  = in.dim_size(3);
    const int kH = fil.dim_size(0);
    const int kW = fil.dim_size(1);
    OP_REQUIRES(ctx, fil.dim_size(2) == C,
                errors::InvalidArgument("filter[2] must equal input channels"));
    const int dm   = fil.dim_size(3);
    const int Cout = C * dm;
    const int sH = strides_[1], sW = strides_[2];

    int Hout, Wout, padH, padW;
    ResolvePadding(H, kH, sH, padding_, &Hout, &padH);
    ResolvePadding(W, kW, sW, padding_, &Wout, &padW);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {N, Hout, Wout, Cout}, &out));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dwconv2d_forward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t in_bytes  = (size_t)N * H * W * C       * sizeof(float);
    const size_t fil_bytes = (size_t)kH * kW * C * dm    * sizeof(float);
    const size_t out_bytes = (size_t)N * Hout * Wout * Cout * sizeof(float);

    ClMem d_in (clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               in_bytes,
                               const_cast<float*>(in.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc input buffer");
    ClMem d_fil(clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               fil_bytes,
                               const_cast<float*>(fil.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc filter buffer");
    ClMem d_out(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY,
                               out_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc output buffer");

    int a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_fil.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_out.m);
    clSetKernelArg(k, a++, sizeof(int), &N);
    clSetKernelArg(k, a++, sizeof(int), &H);
    clSetKernelArg(k, a++, sizeof(int), &W);
    clSetKernelArg(k, a++, sizeof(int), &C);
    clSetKernelArg(k, a++, sizeof(int), &kH);
    clSetKernelArg(k, a++, sizeof(int), &kW);
    clSetKernelArg(k, a++, sizeof(int), &dm);
    clSetKernelArg(k, a++, sizeof(int), &sH);
    clSetKernelArg(k, a++, sizeof(int), &sW);
    clSetKernelArg(k, a++, sizeof(int), &padH);
    clSetKernelArg(k, a++, sizeof(int), &padW);
    clSetKernelArg(k, a++, sizeof(int), &Hout);
    clSetKernelArg(k, a++, sizeof(int), &Wout);

    const size_t global_raw    = (size_t)N * Hout * Wout * Cout;
    const size_t local         = kDefaultLocalSize;
    const size_t global_padded = RoundUp(global_raw, local);

    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global_padded, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dw-forward");
      err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, 0, out_bytes,
                                out->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dw-forward output");
    }
  }

 private:
  std::vector<int32> strides_;
  std::string        padding_;
};
REGISTER_KERNEL_BUILDER(Name("OpenclDepthwiseConv2d").Device(DEVICE_CPU),
                        OpenclDepthwiseConv2dOp);


// =====================================================================
// BACKPROP-INPUT
// =====================================================================
REGISTER_OP("OpenclDepthwiseConv2dBackpropInput")
    .Input("input_sizes: int32")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("output: float")
    .Attr("strides: list(int) >= 4")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return absl::OkStatus();
    });

class OpenclDepthwiseConv2dBackpropInputOp : public OpKernel {
 public:
  explicit OpenclDepthwiseConv2dBackpropInputOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& sizes_t = ctx->input(0);
    const Tensor& fil     = ctx->input(1);
    const Tensor& go      = ctx->input(2);
    OP_REQUIRES(ctx, sizes_t.NumElements() == 4,
                errors::InvalidArgument("input_sizes must have 4 elements"));
    auto sz = sizes_t.flat<int32>();
    const int N = sz(0), H = sz(1), W = sz(2), C = sz(3);
    const int kH = fil.dim_size(0);
    const int kW = fil.dim_size(1);
    OP_REQUIRES(ctx, fil.dim_size(2) == C,
                errors::InvalidArgument("filter[2] != C"));
    const int dm   = fil.dim_size(3);
    const int Cout = C * dm;
    const int sH = strides_[1], sW = strides_[2];

    int Hout, Wout, padH, padW;
    ResolvePadding(H, kH, sH, padding_, &Hout, &padH);
    ResolvePadding(W, kW, sW, padding_, &Wout, &padW);
    OP_REQUIRES(ctx,
                go.dim_size(0)==N && go.dim_size(1)==Hout &&
                go.dim_size(2)==Wout && go.dim_size(3)==Cout,
                errors::InvalidArgument("out_backprop shape mismatch"));

    Tensor* grad_in = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {N, H, W, C}, &grad_in));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dwconv2d_backprop_input"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t go_bytes  = (size_t)N * Hout * Wout * Cout * sizeof(float);
    const size_t fil_bytes = (size_t)kH * kW * C * dm       * sizeof(float);
    const size_t gi_bytes  = (size_t)N * H * W * C          * sizeof(float);

    ClMem d_go (clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               go_bytes,
                               const_cast<float*>(go.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_out");
    ClMem d_fil(clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               fil_bytes,
                               const_cast<float*>(fil.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc filter");
    ClMem d_gi (clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY,
                               gi_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_in");

    int a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_go.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_fil.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_gi.m);
    clSetKernelArg(k, a++, sizeof(int), &N);
    clSetKernelArg(k, a++, sizeof(int), &H);
    clSetKernelArg(k, a++, sizeof(int), &W);
    clSetKernelArg(k, a++, sizeof(int), &C);
    clSetKernelArg(k, a++, sizeof(int), &kH);
    clSetKernelArg(k, a++, sizeof(int), &kW);
    clSetKernelArg(k, a++, sizeof(int), &dm);
    clSetKernelArg(k, a++, sizeof(int), &sH);
    clSetKernelArg(k, a++, sizeof(int), &sW);
    clSetKernelArg(k, a++, sizeof(int), &padH);
    clSetKernelArg(k, a++, sizeof(int), &padW);
    clSetKernelArg(k, a++, sizeof(int), &Hout);
    clSetKernelArg(k, a++, sizeof(int), &Wout);

    const size_t global_raw    = (size_t)N * H * W * C;
    const size_t local         = kDefaultLocalSize;
    const size_t global_padded = RoundUp(global_raw, local);

    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global_padded, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dw-bp-input");
      err = clEnqueueReadBuffer(cl.Queue(), d_gi.m, CL_TRUE, 0, gi_bytes,
                                grad_in->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dw-bp-input");
    }
  }

 private:
  std::vector<int32> strides_;
  std::string        padding_;
};
REGISTER_KERNEL_BUILDER(
    Name("OpenclDepthwiseConv2dBackpropInput")
        .Device(DEVICE_CPU).HostMemory("input_sizes"),
    OpenclDepthwiseConv2dBackpropInputOp);


// =====================================================================
// BACKPROP-FILTER
// =====================================================================
REGISTER_OP("OpenclDepthwiseConv2dBackpropFilter")
    .Input("input: float")
    .Input("filter_sizes: int32")
    .Input("out_backprop: float")
    .Output("output: float")
    .Attr("strides: list(int) >= 4")
    .Attr("padding: {'SAME', 'VALID'}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return absl::OkStatus();
    });

class OpenclDepthwiseConv2dBackpropFilterOp : public OpKernel {
 public:
  explicit OpenclDepthwiseConv2dBackpropFilterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in      = ctx->input(0);
    const Tensor& sizes_t = ctx->input(1);
    const Tensor& go      = ctx->input(2);
    OP_REQUIRES(ctx, sizes_t.NumElements() == 4,
                errors::InvalidArgument("filter_sizes must have 4 elements"));
    auto sz = sizes_t.flat<int32>();
    const int kH = sz(0), kW = sz(1), C = sz(2), dm = sz(3);

    const int N = in.dim_size(0);
    const int H = in.dim_size(1);
    const int W = in.dim_size(2);
    OP_REQUIRES(ctx, in.dim_size(3) == C,
                errors::InvalidArgument("input channels != filter_sizes[2]"));
    const int Cout = C * dm;
    const int sH = strides_[1], sW = strides_[2];

    int Hout, Wout, padH, padW;
    ResolvePadding(H, kH, sH, padding_, &Hout, &padH);
    ResolvePadding(W, kW, sW, padding_, &Wout, &padW);
    OP_REQUIRES(ctx,
                go.dim_size(0)==N && go.dim_size(1)==Hout &&
                go.dim_size(2)==Wout && go.dim_size(3)==Cout,
                errors::InvalidArgument("out_backprop shape mismatch"));

    Tensor* grad_w = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {kH, kW, C, dm}, &grad_w));

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "dwconv2d_backprop_filter"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what()));
      return;
    }

    cl_int err = CL_SUCCESS;
    const size_t in_bytes = (size_t)N * H * W * C        * sizeof(float);
    const size_t go_bytes = (size_t)N * Hout * Wout * Cout * sizeof(float);
    const size_t gw_bytes = (size_t)kH * kW * C * dm     * sizeof(float);

    ClMem d_in(clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              in_bytes,
                              const_cast<float*>(in.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc input");
    ClMem d_go(clCreateBuffer(cl.Context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              go_bytes,
                              const_cast<float*>(go.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_out");
    ClMem d_gw(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY,
                              gw_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_filter");

    int a = 0;
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_go.m);
    clSetKernelArg(k, a++, sizeof(cl_mem), &d_gw.m);
    clSetKernelArg(k, a++, sizeof(int), &N);
    clSetKernelArg(k, a++, sizeof(int), &H);
    clSetKernelArg(k, a++, sizeof(int), &W);
    clSetKernelArg(k, a++, sizeof(int), &C);
    clSetKernelArg(k, a++, sizeof(int), &kH);
    clSetKernelArg(k, a++, sizeof(int), &kW);
    clSetKernelArg(k, a++, sizeof(int), &dm);
    clSetKernelArg(k, a++, sizeof(int), &sH);
    clSetKernelArg(k, a++, sizeof(int), &sW);
    clSetKernelArg(k, a++, sizeof(int), &padH);
    clSetKernelArg(k, a++, sizeof(int), &padW);
    clSetKernelArg(k, a++, sizeof(int), &Hout);
    clSetKernelArg(k, a++, sizeof(int), &Wout);

    const size_t global_raw    = (size_t)kH * kW * C * dm;
    const size_t local         = kDefaultLocalSize;
    const size_t global_padded = RoundUp(global_raw, local);

    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global_padded, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue dw-bp-filter");
      err = clEnqueueReadBuffer(cl.Queue(), d_gw.m, CL_TRUE, 0, gw_bytes,
                                grad_w->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read dw-bp-filter");
    }
  }

 private:
  std::vector<int32> strides_;
  std::string        padding_;
};
REGISTER_KERNEL_BUILDER(
    Name("OpenclDepthwiseConv2dBackpropFilter")
        .Device(DEVICE_CPU).HostMemory("filter_sizes"),
    OpenclDepthwiseConv2dBackpropFilterOp);
