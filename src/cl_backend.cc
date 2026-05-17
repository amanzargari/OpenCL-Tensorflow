// =====================================================================
// cl_backend.cc
//
// CLBackend implementation. See cl_backend.h for the contract.
// =====================================================================

#include "cl_backend.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace opencl_tf {

// ----- ctor / dtor ---------------------------------------------------
CLBackend::CLBackend() {
    cl_int err = CL_SUCCESS;

    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) {
        throw std::runtime_error("OpenCL: no platforms found (is an ICD installed?)");
    }
    std::vector<cl_platform_id> platforms(nplat);
    clGetPlatformIDs(nplat, platforms.data(), nullptr);

    // Prefer the first platform that exposes a GPU.
    for (cl_platform_id p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) == CL_SUCCESS
            && nd > 0) {
            plat_ = p;
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &dev_, nullptr);
            break;
        }
    }
    // Fall back to any device type if no GPU is found.
    if (!plat_) {
        for (cl_platform_id p : platforms) {
            cl_uint nd = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) == CL_SUCCESS
                && nd > 0) {
                plat_ = p;
                clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 1, &dev_, nullptr);
                break;
            }
        }
    }
    if (!plat_ || !dev_) {
        throw std::runtime_error("OpenCL: no usable device found");
    }

    ctx_ = clCreateContext(nullptr, 1, &dev_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateContext failed: " + std::to_string(err));
    }

    // clCreateCommandQueue is OpenCL 1.2 (deprecated in 2.0 but still
    // works with CL_USE_DEPRECATED_OPENCL_1_2_APIS defined).
    queue_ = clCreateCommandQueue(ctx_, dev_, 0, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));
    }
}

CLBackend::~CLBackend() {
    for (auto& kv : kernels_)  if (kv.second) clReleaseKernel(kv.second);
    for (auto& kv : programs_) if (kv.second) clReleaseProgram(kv.second);
    if (queue_) clReleaseCommandQueue(queue_);
    if (ctx_)   clReleaseContext(ctx_);
}

CLBackend& CLBackend::Instance() {
    static CLBackend inst;   // Meyers singleton, thread-safe init since C++11
    return inst;
}

// ----- kernel-file path resolution -----------------------------------
//
// Search order:
//   1. If `cl_file` is absolute, use as-is.
//   2. $OPENCL_TF_KERNELS_PATH/<cl_file>
//   3. ./kernels/<cl_file>   (relative to current working dir)
//   4. <dir-containing-this-.so>/../kernels/<cl_file>
//   5. <dir-containing-this-.so>/kernels/<cl_file>
//   6. <dir-containing-this-.so>/<cl_file>
//
// Step 4-6 use dladdr() to locate the loaded shared object so that the
// kernels can be packaged alongside it inside the Python package.
// ---------------------------------------------------------------------
std::string CLBackend::ResolveKernelPath(const std::string& cl_file) {
    if (cl_file.empty()) throw std::runtime_error("empty kernel file name");
    if (cl_file[0] == '/') return cl_file;

    auto try_path = [](const std::string& p) -> std::string {
        return (access(p.c_str(), R_OK) == 0) ? p : std::string();
    };

    std::string r;

    if (const char* env = std::getenv("OPENCL_TF_KERNELS_PATH")) {
        r = try_path(std::string(env) + "/" + cl_file);
        if (!r.empty()) return r;
    }

    r = try_path("kernels/" + cl_file);
    if (!r.empty()) return r;

    Dl_info info{};
    // Use the address of a function in this translation unit.
    if (dladdr(reinterpret_cast<void*>(&CLBackend::ResolveKernelPath), &info)
        && info.dli_fname) {
        std::string so_path(info.dli_fname);
        size_t slash = so_path.find_last_of('/');
        std::string so_dir = (slash == std::string::npos) ? "."
                                                          : so_path.substr(0, slash);
        for (const std::string& sub : {"/../kernels/", "/kernels/", "/"}) {
            r = try_path(so_dir + sub + cl_file);
            if (!r.empty()) return r;
        }
    }

    throw std::runtime_error("Cannot locate OpenCL kernel file: " + cl_file
                             + " (set OPENCL_TF_KERNELS_PATH or place kernels/"
                             + cl_file + " in CWD)");
}

// ----- program / kernel loading --------------------------------------
cl_program CLBackend::LoadProgram(const std::string& cl_file) {
    const std::string path = ResolveKernelPath(cl_file);

    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string src = ss.str();
    const char*  src_ptr = src.c_str();
    const size_t src_len = src.size();

    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(ctx_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("clCreateProgramWithSource failed for " + path);
    }
    err = clBuildProgram(prog, 1, &dev_,
                         "-cl-std=CL1.2 -cl-mad-enable",
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz + 1, 0);
        clGetProgramBuildInfo(prog, dev_, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        clReleaseProgram(prog);
        throw std::runtime_error("clBuildProgram failed for " + path
                                 + ":\n" + log.data());
    }
    return prog;
}

cl_kernel CLBackend::GetKernel(const std::string& cl_file,
                               const std::string& kernel_name) {
    const std::string key = cl_file + ":" + kernel_name;

    std::lock_guard<std::mutex> lk(programs_mu_);

    auto kit = kernels_.find(key);
    if (kit != kernels_.end()) return kit->second;

    auto pit = programs_.find(cl_file);
    if (pit == programs_.end()) {
        cl_program prog = LoadProgram(cl_file);
        pit = programs_.emplace(cl_file, prog).first;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(pit->second, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || !k) {
        throw std::runtime_error("Kernel '" + kernel_name + "' not found in " + cl_file
                                 + " (cl_err=" + std::to_string(err) + ")");
    }
    kernels_.emplace(key, k);
    return k;
}

}  // namespace opencl_tf
