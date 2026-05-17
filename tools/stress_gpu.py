#!/usr/bin/env python3
"""
OpenCL GPU stress test.

Selects an AMD GPU if present, otherwise uses the first available GPU.
Runs three back-to-back workloads:
  1. Compute   – SGEMM (matrix multiply): saturates ALUs
  2. Bandwidth – buffer copy:             saturates memory bus
  3. Mixed     – SGEMM + reads:           realistic shader load

Usage:
    python tools/stress_gpu.py [--duration SEC] [--size N] [--platform NAME]

Examples:
    python tools/stress_gpu.py                          # 60 s default
    python tools/stress_gpu.py --duration 120 --size 2048
    python tools/stress_gpu.py --platform rusticl       # force a platform
"""

import argparse
import ctypes
import ctypes.util
import struct
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal raw-OpenCL bindings (no pyopencl dependency)
# ---------------------------------------------------------------------------
_lib = ctypes.CDLL("libOpenCL.so")

cl_int    = ctypes.c_int
cl_uint   = ctypes.c_uint
cl_ulong  = ctypes.c_ulong
cl_size_t = ctypes.c_size_t
cl_mem    = ctypes.c_void_p
cl_plat   = ctypes.c_void_p
cl_dev    = ctypes.c_void_p
cl_ctx    = ctypes.c_void_p
cl_prog   = ctypes.c_void_p
cl_kern   = ctypes.c_void_p
cl_queue  = ctypes.c_void_p

def _cl(func, restype, *argtypes):
    f = getattr(_lib, func)
    f.restype  = restype
    f.argtypes = list(argtypes)
    return f

clGetPlatformIDs     = _cl("clGetPlatformIDs",     cl_int, cl_uint, ctypes.POINTER(cl_plat), ctypes.POINTER(cl_uint))
clGetPlatformInfo    = _cl("clGetPlatformInfo",    cl_int, cl_plat, cl_uint, cl_size_t, ctypes.c_void_p, ctypes.POINTER(cl_size_t))
clGetDeviceIDs       = _cl("clGetDeviceIDs",       cl_int, cl_plat, cl_ulong, cl_uint, ctypes.POINTER(cl_dev), ctypes.POINTER(cl_uint))
clGetDeviceInfo      = _cl("clGetDeviceInfo",      cl_int, cl_dev, cl_uint, cl_size_t, ctypes.c_void_p, ctypes.POINTER(cl_size_t))
clCreateContext      = _cl("clCreateContext",      cl_ctx, ctypes.c_void_p, cl_uint, ctypes.POINTER(cl_dev), ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(cl_int))
clCreateCommandQueue = _cl("clCreateCommandQueue", cl_queue, cl_ctx, cl_dev, cl_ulong, ctypes.POINTER(cl_int))
clCreateBuffer       = _cl("clCreateBuffer",       cl_mem, cl_ctx, cl_ulong, cl_size_t, ctypes.c_void_p, ctypes.POINTER(cl_int))
clCreateProgramWithSource = _cl("clCreateProgramWithSource", cl_prog, cl_ctx, cl_uint, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(cl_size_t), ctypes.POINTER(cl_int))
clBuildProgram       = _cl("clBuildProgram",       cl_int, cl_prog, cl_uint, ctypes.POINTER(cl_dev), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)
clGetProgramBuildInfo= _cl("clGetProgramBuildInfo",cl_int, cl_prog, cl_dev, cl_uint, cl_size_t, ctypes.c_void_p, ctypes.POINTER(cl_size_t))
clCreateKernel       = _cl("clCreateKernel",       cl_kern, cl_prog, ctypes.c_char_p, ctypes.POINTER(cl_int))
clSetKernelArg       = _cl("clSetKernelArg",       cl_int, cl_kern, cl_uint, cl_size_t, ctypes.c_void_p)
clEnqueueNDRangeKernel = _cl("clEnqueueNDRangeKernel", cl_int, cl_queue, cl_kern, cl_uint, ctypes.POINTER(cl_size_t), ctypes.POINTER(cl_size_t), ctypes.POINTER(cl_size_t), cl_uint, ctypes.c_void_p, ctypes.c_void_p)
clEnqueueWriteBuffer = _cl("clEnqueueWriteBuffer", cl_int, cl_queue, cl_mem, cl_int, cl_size_t, cl_size_t, ctypes.c_void_p, cl_uint, ctypes.c_void_p, ctypes.c_void_p)
clEnqueueReadBuffer  = _cl("clEnqueueReadBuffer",  cl_int, cl_queue, cl_mem, cl_int, cl_size_t, cl_size_t, ctypes.c_void_p, cl_uint, ctypes.c_void_p, ctypes.c_void_p)
clFinish             = _cl("clFinish",             cl_int, cl_queue)
clReleaseMemObject   = _cl("clReleaseMemObject",   cl_int, cl_mem)
clReleaseKernel      = _cl("clReleaseKernel",      cl_int, cl_kern)
clReleaseProgram     = _cl("clReleaseProgram",     cl_int, cl_prog)
clReleaseCommandQueue= _cl("clReleaseCommandQueue",cl_int, cl_queue)
clReleaseContext     = _cl("clReleaseContext",     cl_int, cl_ctx)

CL_SUCCESS          = 0
CL_MEM_READ_WRITE   = 1
CL_MEM_READ_ONLY    = 4
CL_MEM_WRITE_ONLY   = 2
CL_DEVICE_TYPE_GPU  = 4
CL_DEVICE_TYPE_ALL  = 0xFFFFFFFF
CL_DEVICE_NAME      = 0x102B
CL_DEVICE_VENDOR    = 0x102C
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
CL_DEVICE_GLOBAL_MEM_SIZE    = 0x101F
CL_PLATFORM_NAME    = 0x0902
CL_PROGRAM_BUILD_LOG= 0x1183
CL_TRUE             = 1

def _get_str(fn, obj, param):
    sz = cl_size_t(0)
    fn(obj, param, 0, None, ctypes.byref(sz))
    buf = ctypes.create_string_buffer(sz.value)
    fn(obj, param, sz.value, buf, None)
    return buf.value.decode(errors="replace").strip()

def _get_uint(fn, obj, param):
    val = ctypes.c_uint(0)
    fn(obj, param, ctypes.sizeof(val), ctypes.byref(val), None)
    return val.value

def _get_ulong(fn, obj, param):
    val = ctypes.c_ulong(0)
    fn(obj, param, ctypes.sizeof(val), ctypes.byref(val), None)
    return val.value

# ---------------------------------------------------------------------------
# OpenCL kernel source
# ---------------------------------------------------------------------------
KERNELS = r"""
/* ---- SGEMM: C = A @ B, one work-item per (row, col) of C ---- */
kernel void sgemm(
    global const float* A,   /* M x K */
    global const float* B,   /* K x N */
    global       float* C,   /* M x N */
    int M, int N, int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

/* ---- Bandwidth: copy src -> dst ---- */
kernel void bandwidth_copy(
    global const float* src,
    global       float* dst,
    int n)
{
    int i = get_global_id(0);
    if (i < n) dst[i] = src[i];
}

/* ---- Mixed: SGEMM + random read (simulates shader work) ---- */
kernel void sgemm_mixed(
    global const float* A,
    global const float* B,
    global       float* C,
    global const float* lut,   /* look-up table read per element */
    int M, int N, int K, int lut_n)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += A[row * K + k] * B[k * N + col];
    /* scatter read from LUT to prevent compiler from eliding the multiply */
    acc += lut[(row * N + col) % lut_n] * 1e-9f;
    C[row * N + col] = acc;
}
"""

# ---------------------------------------------------------------------------
# Platform / device discovery
# ---------------------------------------------------------------------------
def list_platforms():
    n = cl_uint(0)
    clGetPlatformIDs(0, None, ctypes.byref(n))
    plats = (cl_plat * n.value)()
    clGetPlatformIDs(n.value, plats, None)
    return list(plats)

def platform_name(p):
    return _get_str(clGetPlatformInfo, p, CL_PLATFORM_NAME)

def get_devices(p, dtype=CL_DEVICE_TYPE_ALL):
    n = cl_uint(0)
    if clGetDeviceIDs(p, dtype, 0, None, ctypes.byref(n)) != CL_SUCCESS or n.value == 0:
        return []
    devs = (cl_dev * n.value)()
    clGetDeviceIDs(p, dtype, n.value, devs, None)
    return list(devs)

def device_info(d):
    name   = _get_str(clGetDeviceInfo, d, CL_DEVICE_NAME)
    vendor = _get_str(clGetDeviceInfo, d, CL_DEVICE_VENDOR)
    cu     = _get_uint(clGetDeviceInfo, d, CL_DEVICE_MAX_COMPUTE_UNITS)
    mhz    = _get_uint(clGetDeviceInfo, d, CL_DEVICE_MAX_CLOCK_FREQUENCY)
    vram   = _get_ulong(clGetDeviceInfo, d, CL_DEVICE_GLOBAL_MEM_SIZE)
    return dict(name=name, vendor=vendor, cu=cu, mhz=mhz, vram_mb=vram//1024//1024)

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_program(ctx, dev, src):
    err = cl_int(0)
    src_enc = src.encode()
    src_ptr = ctypes.cast(ctypes.c_char_p(src_enc), ctypes.c_char_p)
    arr     = (ctypes.c_char_p * 1)(src_ptr)
    prog    = clCreateProgramWithSource(ctx, 1, arr, None, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        raise RuntimeError(f"clCreateProgramWithSource: {err.value}")

    devs = (cl_dev * 1)(dev)
    rc   = clBuildProgram(prog, 1, devs, None, None, None)
    if rc != CL_SUCCESS:
        log_sz = cl_size_t(0)
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_sz))
        log_buf = ctypes.create_string_buffer(log_sz.value)
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_sz.value, log_buf, None)
        raise RuntimeError(f"clBuildProgram failed:\n{log_buf.value.decode()}")
    return prog

def make_kernel(prog, name):
    err  = cl_int(0)
    kern = clCreateKernel(prog, name.encode(), ctypes.byref(err))
    if err.value != CL_SUCCESS:
        raise RuntimeError(f"clCreateKernel {name}: {err.value}")
    return kern

def alloc(ctx, size, flags=CL_MEM_READ_WRITE):
    err = cl_int(0)
    buf = clCreateBuffer(ctx, flags, size, None, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        raise RuntimeError(f"clCreateBuffer: {err.value}")
    return buf

def write_buf(queue, buf, data):
    arr = data.astype("float32")
    rc  = clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, arr.nbytes,
                               arr.ctypes.data_as(ctypes.c_void_p),
                               0, None, None)
    if rc != CL_SUCCESS:
        raise RuntimeError(f"clEnqueueWriteBuffer: {rc}")

def set_arg_mem(kern, idx, cl_mem_ptr):
    """Set a cl_mem buffer argument (pointer-sized value returned by clCreateBuffer)."""
    v = ctypes.c_void_p(cl_mem_ptr)
    clSetKernelArg(kern, idx, ctypes.sizeof(v), ctypes.byref(v))

def set_arg_int(kern, idx, n):
    """Set a scalar int argument."""
    v = ctypes.c_int(n)
    clSetKernelArg(kern, idx, ctypes.sizeof(v), ctypes.byref(v))

def run_nd(queue, kern, global_sz, local_sz):
    g = (cl_size_t * len(global_sz))(*global_sz)
    l = (cl_size_t * len(local_sz))(*local_sz)
    rc = clEnqueueNDRangeKernel(queue, kern, len(global_sz), None, g, l, 0, None, None)
    if rc != CL_SUCCESS:
        raise RuntimeError(f"clEnqueueNDRangeKernel: {rc}")
    clFinish(queue)

# ---------------------------------------------------------------------------
# Stress workloads
# ---------------------------------------------------------------------------
import numpy as np

def _roundup(n, m):
    return ((n + m - 1) // m) * m

def stress_compute(ctx, queue, prog, info, N, duration):
    """SGEMM loop: N×N matrix multiply repeated until time is up."""
    kern = make_kernel(prog, "sgemm")
    M = K = N
    flops_per_iter = 2 * M * N * K  # multiply-add per element

    rng = np.random.default_rng(42)
    A_h = rng.standard_normal((M, K), dtype="float32")
    B_h = rng.standard_normal((K, N), dtype="float32")

    d_A = alloc(ctx, A_h.nbytes, CL_MEM_READ_ONLY)
    d_B = alloc(ctx, B_h.nbytes, CL_MEM_READ_ONLY)
    d_C = alloc(ctx, M * N * 4,  CL_MEM_WRITE_ONLY)

    write_buf(queue, d_A, A_h)
    write_buf(queue, d_B, B_h)

    wg = 16
    gx = _roundup(M, wg)
    gy = _roundup(N, wg)

    print(f"\n{'─'*62}")
    print(f"  COMPUTE STRESS  –  SGEMM  {M}×{K} × {K}×{N}")
    print(f"{'─'*62}")

    iters = 0
    total_flops = 0
    t_start = time.perf_counter()
    t_print  = t_start

    set_arg_mem(kern, 0, d_A); set_arg_mem(kern, 1, d_B); set_arg_mem(kern, 2, d_C)
    set_arg_int(kern, 3, M);   set_arg_int(kern, 4, N);   set_arg_int(kern, 5, K)

    while time.perf_counter() - t_start < duration:
        run_nd(queue, kern, (gx, gy), (wg, wg))
        iters       += 1
        total_flops += flops_per_iter
        now = time.perf_counter()
        if now - t_print >= 2.0:
            elapsed = now - t_start
            gflops  = total_flops / elapsed / 1e9
            print(f"  {elapsed:6.1f}s  iter={iters:5d}  {gflops:7.2f} GFLOPS", flush=True)
            t_print = now

    elapsed = time.perf_counter() - t_start
    gflops  = total_flops / elapsed / 1e9
    print(f"\n  Done: {iters} iters in {elapsed:.1f}s  →  {gflops:.2f} GFLOPS peak")

    for b in (d_A, d_B, d_C): clReleaseMemObject(b)
    clReleaseKernel(kern)
    return gflops


def stress_bandwidth(ctx, queue, prog, info, N, duration):
    """Buffer copy loop: measures memory bandwidth."""
    kern = make_kernel(prog, "bandwidth_copy")
    n_elem = N * N
    n_bytes = n_elem * 4

    rng  = np.random.default_rng(7)
    src_h = rng.standard_normal(n_elem, dtype="float32")

    d_src = alloc(ctx, n_bytes, CL_MEM_READ_ONLY)
    d_dst = alloc(ctx, n_bytes, CL_MEM_WRITE_ONLY)
    write_buf(queue, d_src, src_h)

    wg = 256
    g  = _roundup(n_elem, wg)

    print(f"\n{'─'*62}")
    print(f"  BANDWIDTH STRESS  –  {n_bytes/1024/1024:.0f} MB copy")
    print(f"{'─'*62}")

    iters   = 0
    total_b = 0
    t_start = time.perf_counter()
    t_print  = t_start

    set_arg_mem(kern, 0, d_src)
    set_arg_mem(kern, 1, d_dst)
    set_arg_int(kern, 2, n_elem)

    while time.perf_counter() - t_start < duration:
        run_nd(queue, kern, (g,), (wg,))
        iters   += 1
        total_b += n_bytes * 2  # read + write
        now = time.perf_counter()
        if now - t_print >= 2.0:
            elapsed = now - t_start
            gbps    = total_b / elapsed / 1e9
            print(f"  {elapsed:6.1f}s  iter={iters:5d}  {gbps:7.2f} GB/s", flush=True)
            t_print = now

    elapsed = time.perf_counter() - t_start
    gbps    = total_b / elapsed / 1e9
    print(f"\n  Done: {iters} iters in {elapsed:.1f}s  →  {gbps:.2f} GB/s peak")

    for b in (d_src, d_dst): clReleaseMemObject(b)
    clReleaseKernel(kern)
    return gbps


def stress_mixed(ctx, queue, prog, info, N, duration):
    """SGEMM + LUT reads: simulates a real shader workload."""
    kern  = make_kernel(prog, "sgemm_mixed")
    M = K = N
    lut_n = 65536

    rng   = np.random.default_rng(99)
    A_h   = rng.standard_normal((M, K), dtype="float32")
    B_h   = rng.standard_normal((K, N), dtype="float32")
    lut_h = rng.standard_normal(lut_n,  dtype="float32")

    d_A   = alloc(ctx, A_h.nbytes, CL_MEM_READ_ONLY)
    d_B   = alloc(ctx, B_h.nbytes, CL_MEM_READ_ONLY)
    d_C   = alloc(ctx, M * N * 4,  CL_MEM_WRITE_ONLY)
    d_lut = alloc(ctx, lut_h.nbytes, CL_MEM_READ_ONLY)

    write_buf(queue, d_A,   A_h)
    write_buf(queue, d_B,   B_h)
    write_buf(queue, d_lut, lut_h)

    wg = 16
    gx = _roundup(M, wg)
    gy = _roundup(N, wg)

    print(f"\n{'─'*62}")
    print(f"  MIXED STRESS  –  SGEMM {M}×{N} + {lut_n//1024}K LUT reads")
    print(f"{'─'*62}")

    iters = 0
    t_start = time.perf_counter()
    t_print  = t_start

    set_arg_mem(kern, 0, d_A);  set_arg_mem(kern, 1, d_B);  set_arg_mem(kern, 2, d_C)
    set_arg_mem(kern, 3, d_lut)
    set_arg_int(kern, 4, M);    set_arg_int(kern, 5, N);    set_arg_int(kern, 6, K)
    set_arg_int(kern, 7, lut_n)

    while time.perf_counter() - t_start < duration:
        run_nd(queue, kern, (gx, gy), (wg, wg))
        iters += 1
        now = time.perf_counter()
        if now - t_print >= 2.0:
            elapsed = now - t_start
            print(f"  {elapsed:6.1f}s  iter={iters:5d}", flush=True)
            t_print = now

    elapsed = time.perf_counter() - t_start
    print(f"\n  Done: {iters} iters in {elapsed:.1f}s")

    for b in (d_A, d_B, d_C, d_lut): clReleaseMemObject(b)
    clReleaseKernel(kern)


# ---------------------------------------------------------------------------
# rocm-smi temperature poll (optional, non-blocking)
# ---------------------------------------------------------------------------
def _rocm_temp():
    try:
        import subprocess
        r = subprocess.run(["rocm-smi", "--showtemp", "--csv"],
                           capture_output=True, text=True, timeout=2)
        for line in r.stdout.splitlines():
            if "GPU" in line and "Edge" in line:
                parts = line.split(",")
                return f"  GPU temp: {parts[-1].strip()} °C"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OpenCL GPU stress test")
    parser.add_argument("--duration", type=float, default=60,
                        help="Seconds per workload phase (default 60)")
    parser.add_argument("--size", type=int, default=1024,
                        help="Matrix dimension N for NxN SGEMM (default 1024)")
    parser.add_argument("--platform", type=str, default=None,
                        help="Substring to match against platform name (e.g. 'rusticl', 'Intel')")
    args = parser.parse_args()

    # ---- Discover platforms / devices ------------------------------------
    plats = list_platforms()
    if not plats:
        sys.exit("No OpenCL platforms found. Install an OpenCL ICD (intel-opencl-icd, mesa-opencl-icd, or ROCm).")

    chosen_plat = None
    chosen_dev  = None

    # Priority: prefer AMD GPU; fall back to any GPU; last resort any device.
    for priority in ("amd", "gpu", "any"):
        for p in plats:
            pname = platform_name(p)
            if args.platform and args.platform.lower() not in pname.lower():
                continue
            dtype = CL_DEVICE_TYPE_GPU if priority != "any" else CL_DEVICE_TYPE_ALL
            devs  = get_devices(p, dtype)
            for d in devs:
                info = device_info(d)
                is_amd = "AMD" in info["vendor"] or "Radeon" in info["name"] or "amd" in pname.lower()
                if priority == "amd" and not is_amd:
                    continue
                chosen_plat = p
                chosen_dev  = d
                break
            if chosen_dev: break
        if chosen_dev: break

    if not chosen_dev:
        sys.exit("No suitable GPU found.")

    info = device_info(chosen_dev)
    pname = platform_name(chosen_plat)

    print("=" * 62)
    print(f"  OpenCL GPU Stress Test")
    print("=" * 62)
    print(f"  Platform : {pname}")
    print(f"  Device   : {info['name']}")
    print(f"  Vendor   : {info['vendor']}")
    print(f"  CUs      : {info['cu']}")
    print(f"  Freq     : {info['mhz']} MHz")
    print(f"  VRAM     : {info['vram_mb']} MB")
    print(f"  Matrix N : {args.size}")
    print(f"  Duration : {args.duration:.0f}s per phase")
    t = _rocm_temp()
    if t: print(t)
    print("=" * 62)

    # ---- OpenCL setup ---------------------------------------------------
    err  = cl_int(0)
    devs = (cl_dev * 1)(chosen_dev)
    ctx  = clCreateContext(None, 1, devs, None, None, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        sys.exit(f"clCreateContext failed: {err.value}")

    queue = clCreateCommandQueue(ctx, chosen_dev, 0, ctypes.byref(err))
    if err.value != CL_SUCCESS:
        sys.exit(f"clCreateCommandQueue failed: {err.value}")

    print("\nCompiling kernels ...", end=" ", flush=True)
    try:
        prog = build_program(ctx, chosen_dev, KERNELS)
    except RuntimeError as e:
        print(f"\n\nKernel compile failed:\n{e}")
        print("\nFor AMD GPU via Mesa Rusticl, you may need:")
        print("  sudo apt install spirv-tools llvm-spirv-14")
        sys.exit(1)
    print("OK")

    t_wall = time.perf_counter()

    # ---- Run phases -----------------------------------------------------
    gflops = stress_compute  (ctx, queue, prog, info, args.size, args.duration)
    gbps   = stress_bandwidth(ctx, queue, prog, info, args.size, args.duration)
    stress_mixed             (ctx, queue, prog, info, args.size, args.duration)

    # ---- Summary --------------------------------------------------------
    total_s = time.perf_counter() - t_wall
    print(f"\n{'═'*62}")
    print(f"  SUMMARY")
    print(f"{'═'*62}")
    print(f"  Device       : {info['name']}")
    print(f"  Compute      : {gflops:.2f} GFLOPS  (FP32 SGEMM)")
    print(f"  Bandwidth    : {gbps:.2f} GB/s")
    print(f"  Total time   : {total_s:.0f}s")
    t = _rocm_temp()
    if t: print(t)
    print(f"{'═'*62}")

    clReleaseProgram(prog)
    clReleaseCommandQueue(queue)
    clReleaseContext(ctx)


if __name__ == "__main__":
    main()
