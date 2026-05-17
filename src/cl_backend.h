// =====================================================================
// cl_backend.h
//
// Singleton OpenCL backend shared by every custom op in this project.
// New ops should:
//
//   1. Drop their .cl source under kernels/.
//   2. Call CLBackend::Instance().GetKernel("file.cl", "kernel_name")
//      to retrieve a compiled, cached cl_kernel.
//   3. Allocate cl_mem buffers using the ClMem RAII wrapper.
//   4. Wrap the enqueue+read in {std::lock_guard lk(cl.QueueMutex()); ...}.
//
// The backend is thread-safe across program/kernel lookups (programs_mu_)
// and serializes command-queue access (queue_mu_) because cl_command_queue
// is not thread-safe per the OpenCL spec.
// =====================================================================

#ifndef OPENCL_TF_CL_BACKEND_H_
#define OPENCL_TF_CL_BACKEND_H_

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace opencl_tf {

// ----- RAII wrapper for cl_mem ---------------------------------------
struct ClMem {
    cl_mem m = nullptr;

    ClMem() = default;
    explicit ClMem(cl_mem mm) : m(mm) {}
    ~ClMem() { if (m) clReleaseMemObject(m); }

    ClMem(const ClMem&) = delete;
    ClMem& operator=(const ClMem&) = delete;

    ClMem(ClMem&& other) noexcept : m(other.m) { other.m = nullptr; }
    ClMem& operator=(ClMem&& other) noexcept {
        if (this != &other) {
            if (m) clReleaseMemObject(m);
            m = other.m;
            other.m = nullptr;
        }
        return *this;
    }
};

// ----- The backend ---------------------------------------------------
class CLBackend {
 public:
    static CLBackend& Instance();

    cl_context    Context()  const { return ctx_;   }
    cl_device_id  Device()   const { return dev_;   }
    cl_command_queue Queue()       { return queue_; }
    std::mutex& QueueMutex()       { return queue_mu_; }

    // Retrieve a compiled kernel. Programs and kernels are cached.
    // `cl_file` is a filename only (e.g. "conv2d_kernels.cl"); the
    // backend resolves it via the search order documented in cl_backend.cc.
    //
    // Throws std::runtime_error on failure (missing file, compile error,
    // unknown kernel name). Callers should let the exception propagate
    // out of CLBackend::Instance() the first time they call it, or wrap
    // in try/catch and report via OP_REQUIRES_OK / ctx->CtxFailure().
    cl_kernel GetKernel(const std::string& cl_file,
                        const std::string& kernel_name);

 private:
    CLBackend();
    ~CLBackend();
    CLBackend(const CLBackend&) = delete;
    CLBackend& operator=(const CLBackend&) = delete;

    cl_program LoadProgram(const std::string& cl_file);
    static std::string ResolveKernelPath(const std::string& cl_file);

    cl_platform_id   plat_  = nullptr;
    cl_device_id     dev_   = nullptr;
    cl_context       ctx_   = nullptr;
    cl_command_queue queue_ = nullptr;

    std::mutex                                   programs_mu_;
    std::unordered_map<std::string, cl_program>  programs_;     // key: cl_file
    std::unordered_map<std::string, cl_kernel>   kernels_;      // key: cl_file + ":" + kernel_name

    std::mutex queue_mu_;
};

// ----- helpers used in every op kernel -------------------------------
constexpr size_t kDefaultLocalSize = 64;   // AMD GCN wavefront

inline size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }

}  // namespace opencl_tf

#endif  // OPENCL_TF_CL_BACKEND_H_
