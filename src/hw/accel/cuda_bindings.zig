const std = @import("std");

pub const cudaError_t = c_uint;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaErrorInvalidValue: cudaError_t = 1;
pub const cudaErrorMemoryAllocation: cudaError_t = 2;
pub const cudaErrorInitializationError: cudaError_t = 3;
pub const cudaErrorLaunchFailure: cudaError_t = 4;
pub const cudaErrorLaunchTimeout: cudaError_t = 6;
pub const cudaErrorLaunchOutOfResources: cudaError_t = 7;
pub const cudaErrorInvalidDeviceFunction: cudaError_t = 8;
pub const cudaErrorInvalidConfiguration: cudaError_t = 9;
pub const cudaErrorInvalidDevice: cudaError_t = 10;
pub const cudaErrorInvalidMemcpyDirection: cudaError_t = 21;
pub const cudaErrorOutOfMemory: cudaError_t = 2;
pub const cudaErrorUnknown: cudaError_t = 999;

pub const cudaDevAttrComputeCapabilityMajor: c_int = 75;
pub const cudaDevAttrComputeCapabilityMinor: c_int = 76;
pub const cudaDevAttrMaxThreadsPerBlock: c_int = 1;
pub const cudaDevAttrMultiProcessorCount: c_int = 16;
pub const cudaDevAttrTotalConstantMemory: c_int = 11;
pub const cudaDevAttrL2CacheSize: c_int = 38;
pub const cudaDevAttrMaxSharedMemoryPerBlockOptin: c_int = 97;

pub const DeviceMemoryInfo = struct {
    free_bytes: usize,
    total_bytes: usize,
};

pub const DeviceIdentity = struct {
    compute_capability_major: c_int = 0,
    compute_capability_minor: c_int = 0,
    multiprocessor_count: c_int = 0,
    max_threads_per_block: c_int = 0,
    l2_cache_bytes: c_int = 0,
};

pub const AllocationFailure = struct {
    requested_bytes: usize,
    free_bytes: usize,
    total_bytes: usize,
    estimated_peak_bytes: usize,
    reserve_bytes: usize,
};

pub const cudaHostAllocDefault: c_uint = 0;
pub const cudaHostAllocPortable: c_uint = 1;
pub const cudaHostAllocMapped: c_uint = 2;
pub const cudaHostAllocWriteCombined: c_uint = 4;

pub const cudaMemcpyHostToHost: c_uint = 0;
pub const cudaMemcpyHostToDevice: c_uint = 1;
pub const cudaMemcpyDeviceToHost: c_uint = 2;
pub const cudaMemcpyDeviceToDevice: c_uint = 3;
pub const cudaMemcpyDefault: c_uint = 4;

pub const cudaStream_t = ?*anyopaque;

pub const cublasHandle_t = ?*anyopaque;
pub const cublasStatus_t = c_uint;
pub const cublasStatus_t_success: cublasStatus_t = 0;

pub const cublasOperation_t = c_uint;
pub const CUBLAS_OP_N: cublasOperation_t = 0;
pub const CUBLAS_OP_T: cublasOperation_t = 1;
pub const CUBLAS_OP_C: cublasOperation_t = 2;

pub const cudaDataType_t = c_uint;
pub const CUDA_R_16F: cudaDataType_t = 2;
pub const CUDA_R_32F: cudaDataType_t = 0;
pub const CUDA_R_16BF: cudaDataType_t = 14;

pub const cublasComputeType_t = c_uint;
pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 0;
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 1;
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 2;
pub const CUBLAS_COMPUTE_32F_FAST_16BF: cublasComputeType_t = 3;

pub const cublasGemmAlgo_t = c_int;
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: cublasGemmAlgo_t = 99;
pub const CUBLAS_GEMM_DFALT_TENSOR_OP: cublasGemmAlgo_t = 99;

pub const CudaError = error{
    InvalidValue,
    MemoryAllocation,
    InitializationError,
    LaunchFailure,
    LaunchTimeout,
    LaunchOutOfResources,
    InvalidDeviceFunction,
    InvalidConfiguration,
    InvalidDevice,
    InvalidMemcpyDirection,
    HostAllocFailed,
    Unknown,
    CublasInitFailed,
    CublasMatmulFailed,
};

const RealApi = struct {
    pub extern "c" fn cudaHostAlloc(ptr: *?*anyopaque, size: usize, flags: c_uint) cudaError_t;
    pub extern "c" fn cudaFreeHost(ptr: ?*anyopaque) cudaError_t;
    pub extern "c" fn cudaMalloc(devPtr: *?*anyopaque, size: usize) cudaError_t;
    pub extern "c" fn cudaFree(devPtr: ?*anyopaque) cudaError_t;
    pub extern "c" fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint) cudaError_t;
    pub extern "c" fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_uint, stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaMemset(devPtr: ?*anyopaque, value: c_int, count: usize) cudaError_t;
    pub extern "c" fn cudaDeviceSynchronize() cudaError_t;
    pub extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaGetLastError() cudaError_t;
    pub extern "c" fn cudaPeekAtLastError() cudaError_t;
    pub extern "c" fn cudaGetErrorString(err: cudaError_t) [*:0]const u8;
    pub extern "c" fn cudaGetErrorName(err: cudaError_t) [*:0]const u8;
    pub extern "c" fn cudaStreamCreate(pStream: *cudaStream_t) cudaError_t;
    pub extern "c" fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
    pub extern "c" fn cudaGetDeviceCount(count: *c_int) cudaError_t;
    pub extern "c" fn cudaSetDevice(device: c_int) cudaError_t;
    pub extern "c" fn cudaGetDevice(device: *c_int) cudaError_t;
    pub extern "c" fn cudaMemGetInfo(free: *usize, total: *usize) cudaError_t;
    pub extern "c" fn cudaDeviceGetAttribute(value: *c_int, attr: c_int, device: c_int) cudaError_t;

    pub extern "c" fn cublasCreate_v2(handle: *cublasHandle_t) cublasStatus_t;
    pub extern "c" fn cublasDestroy_v2(handle: cublasHandle_t) cublasStatus_t;
    pub extern "c" fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) cublasStatus_t;
    pub extern "c" fn cublasGetStream_v2(handle: cublasHandle_t, stream: *cudaStream_t) cublasStatus_t;

    pub extern "c" fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int, n: c_int, k: c_int,
        alpha: *const f32,
        A: ?*const f32, lda: c_int,
        B: ?*const f32, ldb: c_int,
        beta: *const f32,
        C: ?*f32, ldc: c_int,
    ) cublasStatus_t;

    pub extern "c" fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int, n: c_int, k: c_int,
        alpha: ?*const anyopaque,
        A: ?*const anyopaque, Atype: cudaDataType_t, lda: c_int,
        B: ?*const anyopaque, Btype: cudaDataType_t, ldb: c_int,
        beta: ?*const anyopaque,
        C: ?*anyopaque, Ctype: cudaDataType_t, ldc: c_int,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) cublasStatus_t;

    pub extern "c" fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int, n: c_int, k: c_int,
        alpha: ?*const anyopaque,
        A: ?*const anyopaque, Atype: cudaDataType_t, lda: c_int, strideA: i64,
        B: ?*const anyopaque, Btype: cudaDataType_t, ldb: c_int, strideB: i64,
        beta: ?*const anyopaque,
        C: ?*anyopaque, Ctype: cudaDataType_t, ldc: c_int, strideC: i64,
        batchCount: c_int,
        computeType: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) cublasStatus_t;

    pub extern "c" fn cublasSgemv_v2(
        handle: cublasHandle_t,
        trans: cublasOperation_t,
        m: c_int, n: c_int,
        alpha: *const f32,
        A: ?*const f32, lda: c_int,
        x: ?*const f32, incx: c_int,
        beta: *const f32,
        y: ?*f32, incy: c_int,
    ) cublasStatus_t;

    pub extern "c" fn cublasSnrm2_v2(handle: cublasHandle_t, n: c_int, x: ?*const f32, incx: c_int, result: *f32) cublasStatus_t;
    pub extern "c" fn cublasSscal_v2(handle: cublasHandle_t, n: c_int, alpha: *const f32, x: ?*f32, incx: c_int) cublasStatus_t;
    pub extern "c" fn cublasSdot_v2(handle: cublasHandle_t, n: c_int, x: ?*const f32, incx: c_int, y: ?*const f32, incy: c_int, result: *f32) cublasStatus_t;
    pub extern "c" fn cublasSaxpy_v2(handle: cublasHandle_t, n: c_int, alpha: *const f32, x: ?*const f32, incx: c_int, y: ?*f32, incy: c_int) cublasStatus_t;
};

pub const cudaHostAlloc = RealApi.cudaHostAlloc;
pub const cudaFreeHost = RealApi.cudaFreeHost;
pub const cudaMalloc = RealApi.cudaMalloc;
pub const cudaFree = RealApi.cudaFree;
pub const cudaMemcpy = RealApi.cudaMemcpy;
pub const cudaMemcpyAsync = RealApi.cudaMemcpyAsync;
pub const cudaMemset = RealApi.cudaMemset;
pub const cudaDeviceSynchronize = RealApi.cudaDeviceSynchronize;
pub const cudaStreamSynchronize = RealApi.cudaStreamSynchronize;
pub const cudaGetLastError = RealApi.cudaGetLastError;
pub const cudaPeekAtLastError = RealApi.cudaPeekAtLastError;
pub const cudaGetErrorString = RealApi.cudaGetErrorString;
pub const cudaGetErrorName = RealApi.cudaGetErrorName;
pub const cudaStreamCreate = RealApi.cudaStreamCreate;
pub const cudaStreamDestroy = RealApi.cudaStreamDestroy;
pub const cudaGetDeviceCount = RealApi.cudaGetDeviceCount;
pub const cudaSetDevice = RealApi.cudaSetDevice;
pub const cudaGetDevice = RealApi.cudaGetDevice;
pub const cudaMemGetInfo = RealApi.cudaMemGetInfo;
pub const cudaDeviceGetAttribute = RealApi.cudaDeviceGetAttribute;

pub fn queryMemoryInfo() CudaError!DeviceMemoryInfo {
    var free_bytes: usize = 0;
    var total_bytes: usize = 0;
    const err = RealApi.cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) return classifyCudaError(err);
    return DeviceMemoryInfo{ .free_bytes = free_bytes, .total_bytes = total_bytes };
}

pub fn queryDeviceIdentity(device: c_int) CudaError!DeviceIdentity {
    var identity = DeviceIdentity{};
    const attrs = [_]struct { attr: c_int, field: []const u8 }{
        .{ .attr = cudaDevAttrComputeCapabilityMajor, .field = "compute_capability_major" },
        .{ .attr = cudaDevAttrComputeCapabilityMinor, .field = "compute_capability_minor" },
        .{ .attr = cudaDevAttrMultiProcessorCount, .field = "multiprocessor_count" },
        .{ .attr = cudaDevAttrMaxThreadsPerBlock, .field = "max_threads_per_block" },
        .{ .attr = cudaDevAttrL2CacheSize, .field = "l2_cache_bytes" },
    };
    inline for (attrs) |entry| {
        var value: c_int = 0;
        const err = RealApi.cudaDeviceGetAttribute(&value, entry.attr, device);
        if (err != cudaSuccess) return classifyCudaError(err);
        if (comptime std.mem.eql(u8, entry.field, "compute_capability_major")) identity.compute_capability_major = value;
        if (comptime std.mem.eql(u8, entry.field, "compute_capability_minor")) identity.compute_capability_minor = value;
        if (comptime std.mem.eql(u8, entry.field, "multiprocessor_count")) identity.multiprocessor_count = value;
        if (comptime std.mem.eql(u8, entry.field, "max_threads_per_block")) identity.max_threads_per_block = value;
        if (comptime std.mem.eql(u8, entry.field, "l2_cache_bytes")) identity.l2_cache_bytes = value;
    }
    return identity;
}

pub fn classifyCudaError(err: cudaError_t) CudaError {
    return switch (err) {
        cudaSuccess => unreachable,
        cudaErrorInvalidValue => CudaError.InvalidValue,
        cudaErrorMemoryAllocation => CudaError.MemoryAllocation,
        cudaErrorInitializationError => CudaError.InitializationError,
        cudaErrorLaunchFailure => CudaError.LaunchFailure,
        cudaErrorLaunchTimeout => CudaError.LaunchTimeout,
        cudaErrorLaunchOutOfResources => CudaError.LaunchOutOfResources,
        cudaErrorInvalidDeviceFunction => CudaError.InvalidDeviceFunction,
        cudaErrorInvalidConfiguration => CudaError.InvalidConfiguration,
        cudaErrorInvalidDevice => CudaError.InvalidDevice,
        cudaErrorInvalidMemcpyDirection => CudaError.InvalidMemcpyDirection,
        else => CudaError.Unknown,
    };
}
pub const cublasCreate_v2 = RealApi.cublasCreate_v2;
pub const cublasDestroy_v2 = RealApi.cublasDestroy_v2;
pub const cublasSetStream_v2 = RealApi.cublasSetStream_v2;
pub const cublasGetStream_v2 = RealApi.cublasGetStream_v2;
pub const cublasGemmEx = RealApi.cublasGemmEx;
pub const cublasGemmStridedBatchedEx = RealApi.cublasGemmStridedBatchedEx;
pub const cublasSgemm_v2 = RealApi.cublasSgemm_v2;
pub const cublasSgemv_v2 = RealApi.cublasSgemv_v2;
pub const cublasSnrm2_v2 = RealApi.cublasSnrm2_v2;
pub const cublasSscal_v2 = RealApi.cublasSscal_v2;
pub const cublasSdot_v2 = RealApi.cublasSdot_v2;
pub const cublasSaxpy_v2 = RealApi.cublasSaxpy_v2;

pub fn checkCuda(err: cudaError_t) CudaError!void {
    if (err != cudaSuccess) return classifyCudaError(err);
}

pub fn checkCublas(err: cublasStatus_t) CudaError!void {
    if (err != cublasStatus_t_success) return CudaError.CublasMatmulFailed;
}
