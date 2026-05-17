/* =====================================================================
 * conv2d_kernels.cl
 *
 * OpenCL 1.2 kernels for standard 2D convolution with NHWC layout.
 *   Input  : [N,  H,  W,  Cin]
 *   Filter : [kH, kW, Cin, Cout]
 *   Output : [N,  Hout, Wout, Cout]
 *
 * Padding is supplied in pre-computed pixel amounts (padH, padW). The host
 * resolves SAME/VALID to integer pad values before launch.
 *
 * One work-item == one output element (forward) / one input element
 * (backprop-input) / one filter element (backprop-filter). The host pads
 * the global size to a multiple of the local size, so every kernel does
 * an in-bounds check on the linear index first.
 *
 * Math:
 *   y[n,ho,wo,co] = sum_{kh,kw,ci}  x[n, ho*sH-padH+kh, wo*sW-padW+kw, ci]
 *                                 * w[kh, kw, ci, co]
 *
 *   dL/dx[n,hi,wi,ci] = sum_{kh,kw,co} dL/dy[n, ho, wo, co] * w[kh,kw,ci,co]
 *     where  ho = (hi + padH - kh) / sH  (must be integer in [0,Hout))
 *            wo = (wi + padW - kw) / sW  (must be integer in [0,Wout))
 *
 *   dL/dw[kh,kw,ci,co] = sum_{n,ho,wo} dL/dy[n,ho,wo,co]
 *                                    * x[n, ho*sH-padH+kh, wo*sW-padW+kw, ci]
 * ===================================================================== */

/* ---------- FORWARD ------------------------------------------------- */
__kernel void conv2d_forward(
    __global const float* restrict input,    /* [N, H, W, Cin]            */
    __global const float* restrict filter,   /* [kH, kW, Cin, Cout]       */
    __global       float* restrict output,   /* [N, Hout, Wout, Cout]     */
    const int N, const int H, const int W, const int Cin,
    const int kH, const int kW, const int Cout,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid   = (int)get_global_id(0);
    const int total = N * Hout * Wout * Cout;
    if (gid >= total) return;

    /* Decode flat index -> (n, ho, wo, co), NHWC inner-to-outer. */
    const int co =  gid                                   % Cout;
    const int wo = (gid / Cout)                           % Wout;
    const int ho = (gid / (Cout * Wout))                  % Hout;
    const int n  =  gid / (Cout * Wout * Hout);

    const int hi0 = ho * sH - padH;
    const int wi0 = wo * sW - padW;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        const int hi = hi0 + kh;
        if ((unsigned)hi >= (unsigned)H) continue;   /* signed-safe bounds test */
        for (int kw = 0; kw < kW; ++kw) {
            const int wi = wi0 + kw;
            if ((unsigned)wi >= (unsigned)W) continue;

            const int x_base = ((n * H + hi) * W + wi) * Cin;
            const int w_base = ((kh * kW + kw) * Cin) * Cout + co;

            for (int ci = 0; ci < Cin; ++ci) {
                acc += input[x_base + ci] * filter[w_base + ci * Cout];
            }
        }
    }
    output[gid] = acc;
}

/* ---------- BACKPROP-INPUT  (dL/dx) -------------------------------- */
__kernel void conv2d_backprop_input(
    __global const float* restrict grad_out, /* [N, Hout, Wout, Cout]     */
    __global const float* restrict filter,   /* [kH, kW, Cin, Cout]       */
    __global       float* restrict grad_in,  /* [N, H, W, Cin]            */
    const int N, const int H, const int W, const int Cin,
    const int kH, const int kW, const int Cout,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid   = (int)get_global_id(0);
    const int total = N * H * W * Cin;
    if (gid >= total) return;

    const int ci =  gid                          % Cin;
    const int wi = (gid / Cin)                   % W;
    const int hi = (gid / (Cin * W))             % H;
    const int n  =  gid / (Cin * W * H);

    float acc = 0.0f;
    /* For each filter tap, see if it lands on this input pixel for some
       output coordinate (ho, wo). The relation is:
          hi = ho*sH - padH + kh  =>  ho = (hi + padH - kh) / sH
       which must be a non-negative integer < Hout. */
    for (int kh = 0; kh < kH; ++kh) {
        const int ho_num = hi + padH - kh;
        if (ho_num < 0) continue;
        if ((ho_num % sH) != 0) continue;
        const int ho = ho_num / sH;
        if (ho >= Hout) continue;

        for (int kw = 0; kw < kW; ++kw) {
            const int wo_num = wi + padW - kw;
            if (wo_num < 0) continue;
            if ((wo_num % sW) != 0) continue;
            const int wo = wo_num / sW;
            if (wo >= Wout) continue;

            const int g_base = ((n * Hout + ho) * Wout + wo) * Cout;
            const int w_base = ((kh * kW + kw) * Cin + ci) * Cout;

            for (int co = 0; co < Cout; ++co) {
                acc += grad_out[g_base + co] * filter[w_base + co];
            }
        }
    }
    grad_in[gid] = acc;
}

/* ---------- BACKPROP-FILTER  (dL/dw) ------------------------------- */
__kernel void conv2d_backprop_filter(
    __global const float* restrict input,    /* [N, H, W, Cin]            */
    __global const float* restrict grad_out, /* [N, Hout, Wout, Cout]     */
    __global       float* restrict grad_w,   /* [kH, kW, Cin, Cout]       */
    const int N, const int H, const int W, const int Cin,
    const int kH, const int kW, const int Cout,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid   = (int)get_global_id(0);
    const int total = kH * kW * Cin * Cout;
    if (gid >= total) return;

    const int co =  gid                          % Cout;
    const int ci = (gid / Cout)                  % Cin;
    const int kw = (gid / (Cout * Cin))          % kW;
    const int kh =  gid / (Cout * Cin * kW);

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Hout; ++ho) {
            const int hi = ho * sH - padH + kh;
            if ((unsigned)hi >= (unsigned)H) continue;
            for (int wo = 0; wo < Wout; ++wo) {
                const int wi = wo * sW - padW + kw;
                if ((unsigned)wi >= (unsigned)W) continue;

                const int x_idx = ((n * H + hi) * W + wi) * Cin + ci;
                const int g_idx = ((n * Hout + ho) * Wout + wo) * Cout + co;
                acc += input[x_idx] * grad_out[g_idx];
            }
        }
    }
    grad_w[gid] = acc;
}
