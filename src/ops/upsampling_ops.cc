// =====================================================================
// upsampling_ops.cc
//
// Bilinear UpSampling2D forward + backward.
//
//   OpenclUpsamplingBilinear2d
//     attrs  : size (list<int>, length 2: [sy, sx])
//     inputs : input [N, H, W, C]
//     output : output [N, H*sy, W*sx, C]
//
//   OpenclUpsamplingBilinear2dGrad
//     attrs  : size
//     inputs : grad_out [N, Hout, Wout, C]
//              input_sizes (int32 vector [N, H, W, C], HostMemory)
//     output : grad_in [N, H, W, C]
//
// Pixel mapping: half-pixel centres (TF default):
//   ih_f = (oh + 0.5) * H / Hout - 0.5
//   iw_f = (ow + 0.5) * W / Wout - 0.5
//
// The backward kernel does scatter-add with float atomic_add (uint CAS)
// and requires the grad_in buffer to be zero-initialised first, which we
// do with clEnqueueFillBuffer before launching the kernel.
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
constexpr char kKernelFile[] = "upsampling_bilinear_kernels.cl";

#define OP_REQUIRES_CL(CTX, ERR, MSG)                                  \
  OP_REQUIRES((CTX), (ERR) == CL_SUCCESS,                              \
              errors::Internal(MSG " (cl_err=", static_cast<int>(ERR), ")"))
}  // namespace


// =====================================================================
// FORWARD
// =====================================================================
REGISTER_OP("OpenclUpsamplingBilinear2d")
    .Input("input: float")
    .Output("output: float")
    .Attr("size: list(int)")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle in;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &in));
      std::vector<int32> size;
      TF_RETURN_IF_ERROR(c->GetAttr("size", &size));
      if (size.size() != 2) {
        return errors::InvalidArgument("size must have exactly 2 elements");
      }
      DimensionHandle batch = c->Dim(in, 0);
      DimensionHandle C     = c->Dim(in, 3);
      DimensionHandle H, W, Hout, Wout;
      H = c->Dim(in, 1);
      W = c->Dim(in, 2);
      if (c->ValueKnown(H))
        Hout = c->MakeDim(c->Value(H) * size[0]);
      else
        Hout = c->UnknownDim();
      if (c->ValueKnown(W))
        Wout = c->MakeDim(c->Value(W) * size[1]);
      else
        Wout = c->UnknownDim();
      c->set_output(0, c->MakeShape({batch, Hout, Wout, C}));
      return absl::OkStatus();
    });

class OpenclUpsamplingBilinear2dOp : public OpKernel {
 public:
  explicit OpenclUpsamplingBilinear2dOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &size_));
    OP_REQUIRES(ctx, size_.size() == 2,
                errors::InvalidArgument("size must have exactly 2 elements"));
    OP_REQUIRES(ctx, size_[0] >= 1 && size_[1] >= 1,
                errors::InvalidArgument("size elements must be >= 1"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in = ctx->input(0);
    OP_REQUIRES(ctx, in.dims() == 4, errors::InvalidArgument("input must be NHWC"));

    const int N    = in.dim_size(0);
    const int H    = in.dim_size(1);
    const int W    = in.dim_size(2);
    const int C    = in.dim_size(3);
    const int Hout = H * size_[0];
    const int Wout = W * size_[1];

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {N, Hout, Wout, C}, &out));

    const int total_out = N * Hout * Wout * C;
    if (total_out == 0) return;

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "upsample_bilinear_forward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t in_bytes  = (size_t)N * H * W * C * sizeof(float);
    const size_t out_bytes = (size_t)total_out      * sizeof(float);

    ClMem d_in (clCreateBuffer(cl.Context(),
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_bytes,
                               const_cast<float*>(in.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc input");
    ClMem d_out(clCreateBuffer(cl.Context(), CL_MEM_WRITE_ONLY, out_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc output");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total_out, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());
      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_in.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_out.m);
      clSetKernelArg(k, a++, sizeof(int),    &N);
      clSetKernelArg(k, a++, sizeof(int),    &H);
      clSetKernelArg(k, a++, sizeof(int),    &W);
      clSetKernelArg(k, a++, sizeof(int),    &C);
      clSetKernelArg(k, a++, sizeof(int),    &Hout);
      clSetKernelArg(k, a++, sizeof(int),    &Wout);
      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue upsample_bilinear_forward");
      err = clEnqueueReadBuffer(cl.Queue(), d_out.m, CL_TRUE, 0, out_bytes,
                                out->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read upsample_bilinear_forward output");
    }
  }

 private:
  std::vector<int32> size_;
};
REGISTER_KERNEL_BUILDER(Name("OpenclUpsamplingBilinear2d").Device(DEVICE_CPU),
                        OpenclUpsamplingBilinear2dOp);


// =====================================================================
// BACKWARD
// =====================================================================
REGISTER_OP("OpenclUpsamplingBilinear2dGrad")
    .Input("grad_out: float")
    .Input("input_sizes: int32")
    .Output("grad_in: float")
    .Attr("size: list(int)")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return absl::OkStatus();
    });

class OpenclUpsamplingBilinear2dGradOp : public OpKernel {
 public:
  explicit OpenclUpsamplingBilinear2dGradOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &size_));
    OP_REQUIRES(ctx, size_.size() == 2,
                errors::InvalidArgument("size must have exactly 2 elements"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_out  = ctx->input(0);
    const Tensor& sizes_t   = ctx->input(1);

    OP_REQUIRES(ctx, sizes_t.NumElements() == 4,
                errors::InvalidArgument("input_sizes must have 4 elements [N,H,W,C]"));
    auto sz = sizes_t.flat<int32>();
    const int N    = sz(0);
    const int H    = sz(1);
    const int W    = sz(2);
    const int C    = sz(3);
    const int Hout = H * size_[0];
    const int Wout = W * size_[1];

    OP_REQUIRES(ctx,
                grad_out.dim_size(0)==N && grad_out.dim_size(1)==Hout &&
                grad_out.dim_size(2)==Wout && grad_out.dim_size(3)==C,
                errors::InvalidArgument("grad_out shape mismatch with input_sizes + size"));

    Tensor* grad_in = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {N, H, W, C}, &grad_in));

    const int total_out = N * Hout * Wout * C;
    if (total_out == 0) {
      // Zero fill the output for the empty case.
      auto flat = grad_in->flat<float>();
      std::fill(flat.data(), flat.data() + flat.size(), 0.0f);
      return;
    }

    auto& cl = CLBackend::Instance();
    cl_kernel k;
    try { k = cl.GetKernel(kKernelFile, "upsample_bilinear_backward"); }
    catch (const std::exception& e) {
      ctx->CtxFailure(errors::Internal(e.what())); return;
    }

    cl_int err = CL_SUCCESS;
    const size_t go_bytes = (size_t)total_out      * sizeof(float);
    const size_t gi_bytes = (size_t)N * H * W * C * sizeof(float);

    ClMem d_go(clCreateBuffer(cl.Context(),
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, go_bytes,
                              const_cast<float*>(grad_out.flat<float>().data()), &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_out");
    ClMem d_gi(clCreateBuffer(cl.Context(), CL_MEM_READ_WRITE, gi_bytes, nullptr, &err));
    OP_REQUIRES_CL(ctx, err, "alloc grad_in");

    const size_t local  = kDefaultLocalSize;
    const size_t global = RoundUp((size_t)total_out, local);
    {
      std::lock_guard<std::mutex> lk(cl.QueueMutex());

      int a = 0;
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_go.m);
      clSetKernelArg(k, a++, sizeof(cl_mem), &d_gi.m);
      clSetKernelArg(k, a++, sizeof(int),    &N);
      clSetKernelArg(k, a++, sizeof(int),    &H);
      clSetKernelArg(k, a++, sizeof(int),    &W);
      clSetKernelArg(k, a++, sizeof(int),    &C);
      clSetKernelArg(k, a++, sizeof(int),    &Hout);
      clSetKernelArg(k, a++, sizeof(int),    &Wout);

      // Zero-fill grad_in before scatter-add. clEnqueueFillBuffer is
      // OpenCL 1.2 core and avoids a separate host memset.
      const float zero_pattern = 0.0f;
      err = clEnqueueFillBuffer(cl.Queue(), d_gi.m, &zero_pattern,
                                sizeof(float), 0, gi_bytes, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "fill grad_in with zeros");

      err = clEnqueueNDRangeKernel(cl.Queue(), k, 1, nullptr,
                                   &global, &local, 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "enqueue upsample_bilinear_backward");
      err = clEnqueueReadBuffer(cl.Queue(), d_gi.m, CL_TRUE, 0, gi_bytes,
                                grad_in->flat<float>().data(), 0, nullptr, nullptr);
      OP_REQUIRES_CL(ctx, err, "read upsample_bilinear_backward output");
    }
  }

 private:
  std::vector<int32> size_;
};
REGISTER_KERNEL_BUILDER(
    Name("OpenclUpsamplingBilinear2dGrad").Device(DEVICE_CPU).HostMemory("input_sizes"),
    OpenclUpsamplingBilinear2dGradOp);
