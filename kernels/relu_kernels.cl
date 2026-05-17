/* =====================================================================
 * relu_kernels.cl
 *
 * Elementwise ReLU and its gradient. Tensor shape is arbitrary; only
 * the total element count matters.
 * ===================================================================== */

/* y = max(0, x) */
__kernel void relu_forward(
    __global const float* restrict input,
    __global       float* restrict output,
    const int total)
{
    const int gid = (int)get_global_id(0);
    if (gid >= total) return;
    output[gid] = fmax(input[gid], 0.0f);
}

/* dx = dy * (x > 0)   (mask taken from the forward input, matching TF's
 *                       ReluGrad which uses "features" = the input to ReLU). */
__kernel void relu_backward(
    __global const float* restrict grad_out,
    __global const float* restrict input,
    __global       float* restrict grad_in,
    const int total)
{
    const int gid = (int)get_global_id(0);
    if (gid >= total) return;
    grad_in[gid] = (input[gid] > 0.0f) ? grad_out[gid] : 0.0f;
}
