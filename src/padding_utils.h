// =====================================================================
// padding_utils.h
//
// SAME / VALID padding resolution, matching TF's convention.
// Header-only so any new op file can `#include "padding_utils.h"`.
// =====================================================================

#ifndef OPENCL_TF_PADDING_UTILS_H_
#define OPENCL_TF_PADDING_UTILS_H_

#include <algorithm>
#include <string>

namespace opencl_tf {

// Compute the output size and the *left/top* padding amount for one
// spatial dimension.
//
// TF's SAME mode pads asymmetrically: when pad_total is odd, the extra
// pixel goes on the right/bottom. The kernels in conv2d_kernels.cl
// check bounds on every tap, so encoding only pad_before is sufficient.
inline void ResolvePadding(int in_size, int k, int stride,
                           const std::string& mode,
                           int* out_size, int* pad_before) {
    if (mode == "SAME") {
        *out_size   = (in_size + stride - 1) / stride;
        const int pad_total = std::max((*out_size - 1) * stride + k - in_size, 0);
        *pad_before = pad_total / 2;
    } else {  // VALID
        *out_size   = (in_size - k + stride) / stride;
        *pad_before = 0;
    }
}

}  // namespace opencl_tf

#endif  // OPENCL_TF_PADDING_UTILS_H_
