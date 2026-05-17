/* =====================================================================
 * dense_kernels.cl
 *
 * Fully-connected (Dense) layer forward + three backward passes.
 *
 * Tensor conventions:
 *   x      : [batch, in_features]
 *   W      : [in_features, out_features]   (column-major in math terms)
 *   b      : [out_features]
 *   y      : [batch, out_features]
 *   grad_y : [batch, out_features]
 *
 * One work-item per output element on all four kernels. This is a
 * straightforward GEMM loop; Phase 4 will swap in a tiled version.
 * ===================================================================== */

/* y[n, o] = sum_i x[n,i] * W[i,o] + b[o] */
__kernel void dense_forward(
    __global const float* restrict x,    /* [batch * in_f] row-major  */
    __global const float* restrict W,    /* [in_f * out_f] row-major  */
    __global const float* restrict b,    /* [out_f]                   */
    __global       float* restrict y,    /* [batch * out_f] row-major */
    const int batch,
    const int in_f,
    const int out_f)
{
    const int gid = (int)get_global_id(0);
    if (gid >= batch * out_f) return;

    const int n = gid / out_f;
    const int o = gid % out_f;

    float acc = b[o];
    for (int i = 0; i < in_f; ++i) {
        acc += x[n * in_f + i] * W[i * out_f + o];
    }
    y[n * out_f + o] = acc;
}

/* grad_x[n, i] = sum_o grad_y[n,o] * W[i,o]
 * One work-item per (n, i) pair. */
__kernel void dense_backprop_input(
    __global const float* restrict grad_y, /* [batch * out_f] */
    __global const float* restrict W,      /* [in_f  * out_f] */
    __global       float* restrict grad_x, /* [batch * in_f]  */
    const int batch,
    const int in_f,
    const int out_f)
{
    const int gid = (int)get_global_id(0);
    if (gid >= batch * in_f) return;

    const int n = gid / in_f;
    const int i = gid % in_f;

    float acc = 0.0f;
    for (int o = 0; o < out_f; ++o) {
        acc += grad_y[n * out_f + o] * W[i * out_f + o];
    }
    grad_x[n * in_f + i] = acc;
}

/* grad_W[i, o] = sum_n x[n,i] * grad_y[n,o]
 * One work-item per (i, o) pair. */
__kernel void dense_backprop_weight(
    __global const float* restrict x,      /* [batch * in_f]  */
    __global const float* restrict grad_y, /* [batch * out_f] */
    __global       float* restrict grad_W, /* [in_f  * out_f] */
    const int batch,
    const int in_f,
    const int out_f)
{
    const int gid = (int)get_global_id(0);
    if (gid >= in_f * out_f) return;

    const int i = gid / out_f;
    const int o = gid % out_f;

    float acc = 0.0f;
    for (int n = 0; n < batch; ++n) {
        acc += x[n * in_f + i] * grad_y[n * out_f + o];
    }
    grad_W[i * out_f + o] = acc;
}

/* grad_b[o] = sum_n grad_y[n, o]
 * One work-item per output feature. */
__kernel void dense_backprop_bias(
    __global const float* restrict grad_y, /* [batch * out_f] */
    __global       float* restrict grad_b, /* [out_f]         */
    const int batch,
    const int out_f)
{
    const int gid = (int)get_global_id(0);
    if (gid >= out_f) return;

    float acc = 0.0f;
    for (int n = 0; n < batch; ++n) {
        acc += grad_y[n * out_f + gid];
    }
    grad_b[gid] = acc;
}
