/* =====================================================================
 * upsampling_bilinear_kernels.cl
 *
 * Bilinear UpSampling2D forward and backward.
 *
 * Pixel mapping: half-pixel centres, no align_corners.
 *   ih_f = (oh + 0.5f) * H / Hout - 0.5f
 *   iw_f = (ow + 0.5f) * W / Wout - 0.5f
 * This matches tf.image.resize(method='bilinear') default behaviour
 * (half_pixel_centers=True, align_corners=False) and therefore also
 * tf.keras.layers.UpSampling2D(interpolation='bilinear').
 *
 * Edge handling: clamp ih0/ih1 to [0, H-1] and iw0/iw1 to [0, W-1]
 * (edge-replication, same as TF).
 * ===================================================================== */


/* ----- float atomic add via uint CAS (OpenCL 1.2, GCN-safe) ----------
 * There is no native float atomic add in OpenCL 1.2.  We alias the
 * float pointer as uint and use the guaranteed atomic_cmpxchg.
 * cl_khr_global_int32_base_atomics is core in OpenCL 1.2.            */
inline void atomic_add_float(volatile __global float* addr, float val) {
    union { uint u; float f; } old_v, new_v;
    do {
        old_v.f = *addr;
        new_v.f = old_v.f + val;
    } while (atomic_cmpxchg(
                 (volatile __global uint*)addr,
                 old_v.u, new_v.u) != old_v.u);
}


/* ----- Forward: one work-item per output element ----------------------
 * output[n, oh, ow, c] = bilinear(input, ih_f, iw_f)                  */
__kernel void upsample_bilinear_forward(
    __global const float* restrict input,   /* [N, H,    W,    C] */
    __global       float* restrict output,  /* [N, Hout, Wout, C] */
    const int N,
    const int H,
    const int W,
    const int C,
    const int Hout,
    const int Wout)
{
    const int gid = (int)get_global_id(0);
    if (gid >= N * Hout * Wout * C) return;

    /* Decompose linear index into (n, oh, ow, c) */
    const int c  =  gid % C;
    const int ow = (gid / C) % Wout;
    const int oh = (gid / C / Wout) % Hout;
    const int n  =  gid / C / Wout / Hout;

    /* Half-pixel centre mapping */
    const float ih_f = (oh + 0.5f) * (float)H / (float)Hout - 0.5f;
    const float iw_f = (ow + 0.5f) * (float)W / (float)Wout - 0.5f;

    const int ih0 = (int)floor(ih_f);
    const int iw0 = (int)floor(iw_f);
    const int ih1 = ih0 + 1;
    const int iw1 = iw0 + 1;

    const float dh = ih_f - (float)ih0;
    const float dw = iw_f - (float)iw0;

    /* Clamp to [0, H-1] / [0, W-1] for edge replication */
    const int ih0c = max(0, min(ih0, H - 1));
    const int ih1c = max(0, min(ih1, H - 1));
    const int iw0c = max(0, min(iw0, W - 1));
    const int iw1c = max(0, min(iw1, W - 1));

    /* Bilinear interpolation */
    const int base = n * H * W * C + c;
    const float v00 = input[base + (ih0c * W + iw0c) * C];
    const float v01 = input[base + (ih0c * W + iw1c) * C];
    const float v10 = input[base + (ih1c * W + iw0c) * C];
    const float v11 = input[base + (ih1c * W + iw1c) * C];

    output[n * Hout * Wout * C + oh * Wout * C + ow * C + c] =
        (1.0f - dh) * (1.0f - dw) * v00 +
        (1.0f - dh) *          dw * v01 +
                 dh * (1.0f - dw) * v10 +
                 dh *          dw * v11;
}


/* ----- Backward: scatter-add of four weighted contributions -----------
 * For each output pixel (n, oh, ow, c) scatter grad_out[...] * weight
 * back to the four source corners in grad_in.
 *
 * IMPORTANT: grad_in must be zero-filled by the host BEFORE this kernel
 * runs (use clEnqueueFillBuffer).  Multiple work-items may write the same
 * grad_in location, so we use atomic_add_float.                         */
__kernel void upsample_bilinear_backward(
    __global const float* restrict grad_out, /* [N, Hout, Wout, C] */
    __global       float* restrict grad_in,  /* [N, H,    W,    C] */
    const int N,
    const int H,
    const int W,
    const int C,
    const int Hout,
    const int Wout)
{
    const int gid = (int)get_global_id(0);
    if (gid >= N * Hout * Wout * C) return;

    const int c  =  gid % C;
    const int ow = (gid / C) % Wout;
    const int oh = (gid / C / Wout) % Hout;
    const int n  =  gid / C / Wout / Hout;

    const float ih_f = (oh + 0.5f) * (float)H / (float)Hout - 0.5f;
    const float iw_f = (ow + 0.5f) * (float)W / (float)Wout - 0.5f;

    const int ih0 = (int)floor(ih_f);
    const int iw0 = (int)floor(iw_f);
    const int ih1 = ih0 + 1;
    const int iw1 = iw0 + 1;

    const float dh = ih_f - (float)ih0;
    const float dw = iw_f - (float)iw0;

    const int ih0c = max(0, min(ih0, H - 1));
    const int ih1c = max(0, min(ih1, H - 1));
    const int iw0c = max(0, min(iw0, W - 1));
    const int iw1c = max(0, min(iw1, W - 1));

    const float g = grad_out[n * Hout * Wout * C + oh * Wout * C + ow * C + c];
    const int base = n * H * W * C + c;

    atomic_add_float(grad_in + base + (ih0c * W + iw0c) * C, (1.0f - dh) * (1.0f - dw) * g);
    atomic_add_float(grad_in + base + (ih0c * W + iw1c) * C, (1.0f - dh) *          dw  * g);
    atomic_add_float(grad_in + base + (ih1c * W + iw0c) * C,          dh * (1.0f - dw) * g);
    atomic_add_float(grad_in + base + (ih1c * W + iw1c) * C,          dh *          dw  * g);
}
