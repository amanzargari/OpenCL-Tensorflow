/* =====================================================================
 * depthwise_conv2d_kernels.cl
 *
 * Depthwise 2D convolution with NHWC layout, matching tf.nn.depthwise_conv2d.
 *   Input  : [N, H, W, C]
 *   Filter : [kH, kW, C, dm]               (dm == depth_multiplier)
 *   Output : [N, Hout, Wout, C * dm]
 *
 * Output channel co = ci * dm + d, where ci is the input channel and
 * d is the depth-multiplier index. There is NO sum across input channels;
 * each output channel co only sees its corresponding ci. That's the
 * distinguishing feature vs full Conv2D.
 * ===================================================================== */

/* ---------- FORWARD ------------------------------------------------- */
__kernel void dwconv2d_forward(
    __global const float* restrict input,    /* [N, H, W, C]              */
    __global const float* restrict filter,   /* [kH, kW, C, dm]           */
    __global       float* restrict output,   /* [N, Hout, Wout, C*dm]     */
    const int N, const int H, const int W, const int C,
    const int kH, const int kW, const int dm,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid  = (int)get_global_id(0);
    const int Cout = C * dm;
    const int total = N * Hout * Wout * Cout;
    if (gid >= total) return;

    const int co =  gid                          % Cout;
    const int wo = (gid / Cout)                  % Wout;
    const int ho = (gid / (Cout * Wout))         % Hout;
    const int n  =  gid / (Cout * Wout * Hout);

    const int ci = co / dm;
    const int d  = co % dm;

    const int hi0 = ho * sH - padH;
    const int wi0 = wo * sW - padW;

    float acc = 0.0f;
    for (int kh = 0; kh < kH; ++kh) {
        const int hi = hi0 + kh;
        if ((unsigned)hi >= (unsigned)H) continue;
        for (int kw = 0; kw < kW; ++kw) {
            const int wi = wi0 + kw;
            if ((unsigned)wi >= (unsigned)W) continue;

            const int x_idx = ((n * H + hi) * W + wi) * C + ci;
            const int w_idx = ((kh * kW + kw) * C + ci) * dm + d;
            acc += input[x_idx] * filter[w_idx];
        }
    }
    output[gid] = acc;
}

/* ---------- BACKPROP-INPUT  (dL/dx) -------------------------------- */
__kernel void dwconv2d_backprop_input(
    __global const float* restrict grad_out, /* [N, Hout, Wout, C*dm]     */
    __global const float* restrict filter,   /* [kH, kW, C, dm]           */
    __global       float* restrict grad_in,  /* [N, H, W, C]              */
    const int N, const int H, const int W, const int C,
    const int kH, const int kW, const int dm,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid = (int)get_global_id(0);
    const int total = N * H * W * C;
    if (gid >= total) return;

    const int ci =  gid                  % C;
    const int wi = (gid / C)             % W;
    const int hi = (gid / (C * W))       % H;
    const int n  =  gid / (C * W * H);

    const int Cout = C * dm;

    float acc = 0.0f;
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

            const int g_base = ((n * Hout + ho) * Wout + wo) * Cout + ci * dm;
            const int w_base = ((kh * kW + kw) * C + ci) * dm;

            for (int d = 0; d < dm; ++d) {
                acc += grad_out[g_base + d] * filter[w_base + d];
            }
        }
    }
    grad_in[gid] = acc;
}

/* ---------- BACKPROP-FILTER  (dL/dw) ------------------------------- */
__kernel void dwconv2d_backprop_filter(
    __global const float* restrict input,    /* [N, H, W, C]              */
    __global const float* restrict grad_out, /* [N, Hout, Wout, C*dm]     */
    __global       float* restrict grad_w,   /* [kH, kW, C, dm]           */
    const int N, const int H, const int W, const int C,
    const int kH, const int kW, const int dm,
    const int sH, const int sW,
    const int padH, const int padW,
    const int Hout, const int Wout)
{
    const int gid = (int)get_global_id(0);
    const int total = kH * kW * C * dm;
    if (gid >= total) return;

    const int d  =  gid                  % dm;
    const int ci = (gid / dm)            % C;
    const int kw = (gid / (dm * C))      % kW;
    const int kh =  gid / (dm * C * kW);

    const int Cout = C * dm;
    const int co   = ci * dm + d;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Hout; ++ho) {
            const int hi = ho * sH - padH + kh;
            if ((unsigned)hi >= (unsigned)H) continue;
            for (int wo = 0; wo < Wout; ++wo) {
                const int wi = wo * sW - padW + kw;
                if ((unsigned)wi >= (unsigned)W) continue;

                const int x_idx = ((n * H + hi) * W + wi) * C + ci;
                const int g_idx = ((n * Hout + ho) * Wout + wo) * Cout + co;
                acc += input[x_idx] * grad_out[g_idx];
            }
        }
    }
    grad_w[gid] = acc;
}
