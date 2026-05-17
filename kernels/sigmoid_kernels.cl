/* =====================================================================
 * sigmoid_kernels.cl
 *
 * Elementwise sigmoid and its gradient. Tensor shape is arbitrary;
 * only the total element count is passed.
 * ===================================================================== */

/* y = 1 / (1 + exp(-x)) */
__kernel void sigmoid_forward(
    __global const float* restrict input,
    __global       float* restrict output,
    const int total)
{
    const int gid = (int)get_global_id(0);
    if (gid >= total) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

/* grad_x = dy * y * (1 - y)
 * Note: this kernel takes the FORWARD OUTPUT y, not x, matching TF's
 * SigmoidGrad convention and avoiding a redundant sigmoid recomputation. */
__kernel void sigmoid_backward(
    __global const float* restrict grad_out,   /* dy        */
    __global const float* restrict fwd_output, /* y = sigmoid(x) */
    __global       float* restrict grad_in,    /* dx        */
    const int total)
{
    const int gid = (int)get_global_id(0);
    if (gid >= total) return;
    const float y = fwd_output[gid];
    grad_in[gid] = grad_out[gid] * y * (1.0f - y);
}
