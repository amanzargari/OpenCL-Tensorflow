/* =====================================================================
 * batchnorm_kernels.cl
 *
 * BatchNormalization in NHWC layout. Statistics are per-channel.
 *
 * Forward (training):
 *   mean_c    = (1/NHW) * sum_{n,h,w} x[n,h,w,c]
 *   var_c     = (1/NHW) * sum_{n,h,w} x[n,h,w,c]^2  -  mean_c^2      (biased)
 *   inv_std_c = 1 / sqrt(var_c + eps)
 *   y[i]      = gamma_c * (x[i] - mean_c) * inv_std_c + beta_c
 *
 * Forward (inference):
 *   Same as training but with externally supplied mean/var (typically
 *   the EMA moving stats).
 *
 * Backward (full derivation):
 *   Let NHW = N*H*W, x_hat = (x - mean) * inv_std.
 *
 *     dbeta_c  = sum_i dy_i
 *     dgamma_c = sum_i dy_i * x_hat_i
 *
 *     dx_i = (gamma_c * inv_std_c / NHW)
 *          * (NHW * dy_i  -  dbeta_c  -  x_hat_i * dgamma_c)
 *
 *   This compact form sidesteps explicit dmean / dvar terms.
 *
 * Reduction layout:
 *   Per-channel reductions use one work-item per channel and iterate
 *   over all NHW elements. Naive but adequate for the channel counts
 *   in the target model (48..192). A tree reduction with local-memory
 *   staging is a Phase-4 optimisation.
 * ===================================================================== */

/* ---------- forward: per-channel mean + biased variance ----------- */
__kernel void bn_reduce_stats(
    __global const float* restrict input,        /* [N, H, W, C]        */
    __global       float* restrict batch_mean,   /* [C]                 */
    __global       float* restrict batch_var,    /* [C]                 */
    const int N, const int H, const int W, const int C)
{
    const int c = (int)get_global_id(0);
    if (c >= C) return;

    const int NHW = N * H * W;
    float sum   = 0.0f;
    float sumsq = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const float v = input[((n * H + h) * W + w) * C + c];
                sum   += v;
                sumsq += v * v;
            }
        }
    }
    const float mean = sum / (float)NHW;
    batch_mean[c] = mean;
    batch_var [c] = sumsq / (float)NHW - mean * mean;
}

/* ---------- forward: elementwise normalize ------------------------ */
__kernel void bn_normalize(
    __global const float* restrict input,    /* [N, H, W, C]            */
    __global const float* restrict mean,     /* [C]                     */
    __global const float* restrict var,      /* [C]                     */
    __global const float* restrict gamma,    /* [C]                     */
    __global const float* restrict beta,     /* [C]                     */
    __global       float* restrict output,   /* [N, H, W, C]            */
    const int N, const int H, const int W, const int C,
    const float epsilon)
{
    const int gid   = (int)get_global_id(0);
    const int total = N * H * W * C;
    if (gid >= total) return;

    const int   c       = gid % C;
    const float inv_std = rsqrt(var[c] + epsilon);
    const float x_hat   = (input[gid] - mean[c]) * inv_std;
    output[gid] = gamma[c] * x_hat + beta[c];
}

/* ---------- backward: per-channel dbeta and dgamma --------------- */
__kernel void bn_backward_reduce(
    __global const float* restrict grad_out, /* [N, H, W, C]            */
    __global const float* restrict input,    /* [N, H, W, C]            */
    __global const float* restrict mean,     /* [C]                     */
    __global const float* restrict var,      /* [C]                     */
    __global       float* restrict grad_beta,  /* [C] = sum(dy)         */
    __global       float* restrict grad_gamma, /* [C] = sum(dy * x_hat) */
    const int N, const int H, const int W, const int C,
    const float epsilon)
{
    const int c = (int)get_global_id(0);
    if (c >= C) return;

    const float inv_std = rsqrt(var[c] + epsilon);
    const float m       = mean[c];

    float s_dy     = 0.0f;
    float s_dy_xh  = 0.0f;

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const int   idx  = ((n * H + h) * W + w) * C + c;
                const float dy   = grad_out[idx];
                const float xhat = (input[idx] - m) * inv_std;
                s_dy    += dy;
                s_dy_xh += dy * xhat;
            }
        }
    }
    grad_beta [c] = s_dy;
    grad_gamma[c] = s_dy_xh;
}

/* ---------- backward: elementwise dx ----------------------------- */
__kernel void bn_backward_dx(
    __global const float* restrict grad_out,    /* [N, H, W, C]         */
    __global const float* restrict input,       /* [N, H, W, C]         */
    __global const float* restrict mean,        /* [C]                  */
    __global const float* restrict var,         /* [C]                  */
    __global const float* restrict gamma,       /* [C]                  */
    __global const float* restrict sum_dy,      /* [C] = dbeta          */
    __global const float* restrict sum_dy_xhat, /* [C] = dgamma         */
    __global       float* restrict grad_in,     /* [N, H, W, C]         */
    const int N, const int H, const int W, const int C,
    const float epsilon)
{
    const int gid   = (int)get_global_id(0);
    const int total = N * H * W * C;
    if (gid >= total) return;

    const int   c       = gid % C;
    const float inv_std = rsqrt(var[c] + epsilon);
    const float xhat    = (input[gid] - mean[c]) * inv_std;
    const float NHW_f   = (float)(N * H * W);
    const float dy      = grad_out[gid];

    grad_in[gid] = (gamma[c] * inv_std / NHW_f)
                 * (NHW_f * dy - sum_dy[c] - xhat * sum_dy_xhat[c]);
}
