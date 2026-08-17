const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");
const core_tensor = @import("../../core/tensor.zig");
const core_memory = @import("../../core/memory.zig");

pub const gpu_enabled: bool = @import("build_options").gpu_acceleration;

pub const AccelError = error{
    FutharkConfigFailed,
    FutharkContextFailed,
    FutharkSyncFailed,
    FutharkArrayNewFailed,
    FutharkValuesFailed,
    FutharkForwardFailed,
    FutharkTrainingStepFailed,
    FutharkScaleWeightsFailed,
    FutharkShapeFailed,
    FutharkComputeLossFailed,
    FutharkBackwardFailed,
    FutharkSFDUpdateFailed,
    CudaHostAllocFailed,
    CudaFreeFailed,
    NullPointer,
    InvalidDimensions,
    InvalidHyperparameter,
    InvalidClipRange,
    InvalidToken,
    AllocationFailed,
    PartialRowCleanup,
};

pub const RSFOptimizerState = struct {
    master_weights_s: []f32,
    master_weights_t: []f32,
    momentum_s: []f32,
    momentum_t: []f32,
    fisher_s: []f32,
    fisher_t: []f32,
    step: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RSFOptimizerState) void {
        self.allocator.free(self.master_weights_s);
        self.allocator.free(self.master_weights_t);
        self.allocator.free(self.momentum_s);
        self.allocator.free(self.momentum_t);
        self.allocator.free(self.fisher_s);
        self.allocator.free(self.fisher_t);
        self.* = undefined;
    }
};

pub const EmbeddingOptimizerState = struct {
    master_weights: []f32,
    momentum: []f32,
    fisher: []f32,
    step: u64,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *EmbeddingOptimizerState) void {
        self.allocator.free(self.master_weights);
        self.allocator.free(self.momentum);
        self.allocator.free(self.fisher);
        self.* = undefined;
    }
};

pub const DeviceBufferF32 = struct {
    ptr: *anyopaque,
    count: usize,
};

fn checkedDimensionI64(value: usize) AccelError!i64 {
    if (value == 0 or value > @as(usize, @intCast(std.math.maxInt(i64)))) return AccelError.InvalidDimensions;
    return @intCast(value);
}

fn checkedElementCount2(first: usize, second: usize) AccelError!usize {
    if (first == 0 or second == 0) return AccelError.InvalidDimensions;
    return std.math.mul(usize, first, second) catch return AccelError.InvalidDimensions;
}

fn checkedElementCount3(first: usize, second: usize, third: usize) AccelError!usize {
    if (third == 0) return AccelError.InvalidDimensions;
    return std.math.mul(usize, try checkedElementCount2(first, second), third) catch return AccelError.InvalidDimensions;
}

fn freeFutharkError(message: ?[*:0]const u8) void {
    if (message) |ptr| std.c.free(@ptrCast(@constCast(ptr)));
}

pub const FutharkContext = struct {
    ctx: ?*futhark.struct_futhark_context,
    cfg: ?*futhark.struct_futhark_context_config,
    mutex: std.Thread.Mutex = .{},

    const Self = @This();

    pub fn init() AccelError!Self {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return AccelError.FutharkConfigFailed;

        if (comptime gpu_enabled) {
            const cache_file: ?[*:0]const u8 = if (std.posix.getenv("JAIDE_FUTHARK_CACHE")) |cache_path| blk: {
                std.debug.print("[FutharkContext] GPU kernel cache: {s}\n", .{cache_path});
                break :blk @as([*:0]const u8, @ptrCast(cache_path.ptr));
            } else blk: {
                std.debug.print("[FutharkContext] WARN: JAIDE_FUTHARK_CACHE not set — NVRTC will recompile on every container start\n", .{});
                break :blk null;
            };
            futhark.configureGpuContext(cfg, cache_file) catch return AccelError.FutharkConfigFailed;
        }

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) {
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkContextFailed;
        }

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            futhark.futhark_context_config_free(cfg);
            return AccelError.FutharkSyncFailed;
        }

        return Self{ .ctx = ctx, .cfg = cfg, .mutex = .{} };
    }

    pub fn deinit(self: *Self) void {
        if (self.ctx) |ctx| {
            _ = futhark.futhark_context_clear_caches(ctx);
            futhark.futhark_context_free(ctx);
            self.ctx = null;
        }
        if (self.cfg) |cfg| {
            futhark.futhark_context_config_free(cfg);
            self.cfg = null;
        }
    }

    fn syncUnlocked(self: *Self) AccelError!void {
        if (self.ctx == null) return AccelError.NullPointer;
        if (futhark.futhark_context_sync(self.ctx) != 0) return AccelError.FutharkSyncFailed;
    }

    pub fn sync(self: *Self) AccelError!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.syncUnlocked();
    }

    pub fn getDataPointerF32_2D(self: *Self, array: *FutharkArray2DF32) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_f32_2d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }

    pub fn getDataPointerF32_3D(self: *Self, array: *FutharkArray3DF32) AccelError!*anyopaque {
        if (self.ctx == null) return AccelError.NullPointer;
        if (array.arr == null) return AccelError.NullPointer;
        const raw_ptr = futhark.futhark_values_raw_f32_3d(self.ctx, array.arr);
        if (raw_ptr == null) return AccelError.NullPointer;
        return raw_ptr.?;
    }
};

pub const PinnedMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    fallback_slice: ?[]align(64) u8,

    const Self = @This();

    pub fn alloc(size: usize) AccelError!Self {
        if (size == 0) {
            return Self{ .ptr = null, .size = 0, .fallback_slice = null };
        }

        if (comptime gpu_enabled) {
            var ptr: ?*anyopaque = null;
            const err = cuda.cudaHostAlloc(&ptr, size, cuda.cudaHostAllocDefault);
            if (err != cuda.cudaSuccess) {
                return AccelError.CudaHostAllocFailed;
            }
            return Self{
                .ptr = ptr,
                .size = size,
                .fallback_slice = null,
            };
        }

        const slice = std.heap.page_allocator.alignedAlloc(u8, 64, size) catch return AccelError.CudaHostAllocFailed;
        return Self{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .fallback_slice = slice,
        };
    }

    pub fn free(self: *Self) void {
        if (self.fallback_slice) |slice| {
            std.heap.page_allocator.free(slice);
            self.fallback_slice = null;
            self.ptr = null;
            self.size = 0;
            return;
        }
        if (self.ptr) |p| {
            if (comptime gpu_enabled) {
                _ = cuda.cudaFreeHost(p);
            }
            self.ptr = null;
            self.size = 0;
        }
    }

    pub fn asSlice(self: *Self, comptime T: type) ?[]T {
        if (self.ptr == null) return null;
        const count = self.size / @sizeOf(T);
        const aligned: [*]T = @ptrCast(@alignCast(self.ptr.?));
        return aligned[0..count];
    }
};

pub const FutharkArray2DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (flat_data.len != try checkedElementCount2(rows, cols)) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(flat_data.ptr),
            try checkedDimensionI64(rows),
            try checkedDimensionI64(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;

        const total = try checkedElementCount2(rows, cols);
        const zeros = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(
            ctx.ctx,
            @ptrCast(zeros.ptr),
            try checkedDimensionI64(rows),
            try checkedDimensionI64(cols),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_2d(ctx.ctx, self.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }
};

pub const FutharkArray3DF16 = struct {
    arr: ?*futhark.struct_futhark_f16_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, flat: []const f16, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        if (flat.len != try checkedElementCount3(d0, d1, d2)) return AccelError.InvalidDimensions;

        const arr = futhark.futhark_new_f16_3d(
            ctx.ctx,
            @ptrCast(flat.ptr),
            try checkedDimensionI64(d0),
            try checkedDimensionI64(d1),
            try checkedDimensionI64(d2),
        );
        if (arr == null) return AccelError.FutharkArrayNewFailed;

        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f16 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f16_3d(ctx.ctx, self.arr, @ptrCast(buf.ptr)) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f16_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }
};

pub const FutharkArray2DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_2d,
    rows: usize,
    cols: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = tensor.shape.dims[0];
        const cols = tensor.shape.dims[1];
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, tensor.data.ptr, try checkedDimensionI64(rows), try checkedDimensionI64(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, rows: usize, cols: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        if (data.len != try checkedElementCount2(rows, cols)) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, data.ptr, try checkedDimensionI64(rows), try checkedDimensionI64(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (rows == 0 or cols == 0) return AccelError.InvalidDimensions;
        const zeros = allocator.alloc(f32, try checkedElementCount2(rows, cols)) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_2d(ctx.ctx, zeros.ptr, try checkedDimensionI64(rows), try checkedDimensionI64(cols));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .rows = rows, .cols = cols };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_2d(ctx.ctx, arr);
            self.arr = null;
            self.rows = 0;
            self.cols = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{ self.rows, self.cols };
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) {
            tensor.deinit();
            return AccelError.FutharkSyncFailed;
        }
        return tensor;
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null or self.arr == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const values = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(values);
        if (futhark.futhark_values_f32_2d(ctx.ctx, self.arr, values.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return values;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF32 {
        const total = std.math.mul(usize, self.rows, self.cols) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointerF32_2D(self), .count = total };
    }
};

pub const FutharkArray3DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_3d,
    dim0: usize,
    dim1: usize,
    dim2: usize,

    const Self = @This();

    pub fn newFromFlat(ctx: *FutharkContext, data: []const f32, d0: usize, d1: usize, d2: usize) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, d0, d1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, d2) catch return AccelError.InvalidDimensions;
        if (data.len != total) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_3d(ctx.ctx, data.ptr, try checkedDimensionI64(d0), try checkedDimensionI64(d1), try checkedDimensionI64(d2));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn newZeros(ctx: *FutharkContext, d0: usize, d1: usize, d2: usize, allocator: std.mem.Allocator) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (d0 == 0 or d1 == 0 or d2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, d0, d1) catch return AccelError.AllocationFailed;
        const total = std.math.mul(usize, d01, d2) catch return AccelError.AllocationFailed;
        const zeros = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const arr = futhark.futhark_new_f32_3d(ctx.ctx, zeros.ptr, try checkedDimensionI64(d0), try checkedDimensionI64(d1), try checkedDimensionI64(d2));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .dim0 = d0, .dim1 = d1, .dim2 = d2 };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_3d(ctx.ctx, arr);
            self.arr = null;
            self.dim0 = 0;
            self.dim1 = 0;
            self.dim2 = 0;
        }
    }

    pub fn valuesFlat(self: *const Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        if (total == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_3d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn deviceBuffer(self: *Self, ctx: *FutharkContext) AccelError!DeviceBufferF32 {
        if (self.dim0 == 0 or self.dim1 == 0 or self.dim2 == 0) return AccelError.InvalidDimensions;
        const d01 = std.math.mul(usize, self.dim0, self.dim1) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, d01, self.dim2) catch return AccelError.InvalidDimensions;
        return .{ .ptr = try ctx.getDataPointerF32_3D(self), .count = total };
    }
};

pub const FutharkArray1DF32 = struct {
    arr: ?*futhark.struct_futhark_f32_1d,
    len: usize,

    const Self = @This();

    pub fn fromTensor(ctx: *FutharkContext, tensor: *const core_tensor.Tensor) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (tensor.shape.dims.len != 1) return AccelError.InvalidDimensions;
        const n = tensor.shape.dims[0];
        if (n == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, tensor.data.ptr, try checkedDimensionI64(n));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = n };
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_f32_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }

    pub fn toTensor(self: *Self, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        const shape = [_]usize{self.len};
        var tensor = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, tensor.data.ptr) != 0) {
            tensor.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(ctx.ctx) != 0) {
            tensor.deinit();
            return AccelError.FutharkSyncFailed;
        }
        return tensor;
    }

    pub fn newFromSlice(ctx: *FutharkContext, data: []const f32) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_f32_1d(ctx.ctx, data.ptr, try checkedDimensionI64(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DF32, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]f32 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(f32, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_f32_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }
};

pub const FutharkArray1DU64 = struct {
    arr: ?*futhark.struct_futhark_u64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_u64_1d(ctx.ctx, data.ptr, try checkedDimensionI64(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DU64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]u64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(u64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_u64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_u64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const RSFLayer = struct {
    weights_s: FutharkArray2DF16,
    weights_t: FutharkArray2DF16,

    pub fn free(self: *RSFLayer, ctx: *FutharkContext) void {
        self.weights_t.free(ctx);
        self.weights_s.free(ctx);
    }
};

pub const FusedStepScalars = struct {
    loss: f32,
    reconstruction_loss: f32,
    logdet_mean: f32,
};

pub const FusedStepResult = struct {
    stack_gradient_s: FutharkArray3DF32,
    stack_gradient_t: FutharkArray3DF32,
    input_delta: FutharkArray3DF16,
    pending: ?*futhark.struct_futhark_opaque_tup6_fused_stack_gradients,
    finalized: bool,
    scalars: FusedStepScalars,

    pub fn gradientDeviceBuffers(self: *FusedStepResult, ctx: *FutharkContext) AccelError![2]DeviceBufferF32 {
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        return .{
            try self.stack_gradient_s.deviceBuffer(ctx),
            try self.stack_gradient_t.deviceBuffer(ctx),
        };
    }

    pub fn finalize(self: *FusedStepResult, ctx: *FutharkContext) AccelError!FusedStepScalars {
        if (self.finalized) return self.scalars;
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        const tup = self.pending orelse {
            self.finalized = true;
            return self.scalars;
        };
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        var loss_out: f32 = 0.0;
        var recon_out: f32 = 0.0;
        var logdet_out: f32 = 0.0;
        const p3 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_3(ctx.ctx, &loss_out, tup);
        const p4 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_4(ctx.ctx, &recon_out, tup);
        const p5 = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_5(ctx.ctx, &logdet_out, tup);
        _ = futhark.futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(ctx.ctx, tup);
        self.pending = null;
        if (p3 != 0 or p4 != 0 or p5 != 0) return AccelError.FutharkTrainingStepFailed;
        if (!std.math.isFinite(loss_out) or !std.math.isFinite(recon_out) or !std.math.isFinite(logdet_out)) return AccelError.FutharkTrainingStepFailed;
        self.scalars = .{ .loss = loss_out, .reconstruction_loss = recon_out, .logdet_mean = logdet_out };
        self.finalized = true;
        return self.scalars;
    }

    pub fn deinit(self: *FusedStepResult, ctx: *FutharkContext) void {
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        self.stack_gradient_t.free(ctx);
        self.stack_gradient_s.free(ctx);
        self.input_delta.free(ctx);
        if (self.pending) |tup| {
            _ = futhark.futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(ctx.ctx, tup);
            self.pending = null;
        }
    }
};

pub const RSFAccelerator = struct {
    ctx: FutharkContext,
    layers: []RSFLayer,
    layers_owner: std.mem.Allocator,
    allocator: std.mem.Allocator,
    model_dim: usize,
    num_layers: usize,
    clip_min: f16,
    clip_max: f16,
    initialized: bool,
    scratch_lengths_buf: []i64 = &[_]i64{},
    scratch_lengths_cap: usize = 0,
    stack_weights_s: ?FutharkArray3DF16 = null,
    stack_weights_t: ?FutharkArray3DF16 = null,
    stack_master_weights_s: ?FutharkArray3DF32 = null,
    stack_master_weights_t: ?FutharkArray3DF32 = null,
    stack_momentum_s: ?FutharkArray3DF32 = null,
    stack_momentum_t: ?FutharkArray3DF32 = null,
    stack_fisher_s: ?FutharkArray3DF32 = null,
    stack_fisher_t: ?FutharkArray3DF32 = null,
    stack_arrays_valid: bool = false,
    layers_mirror_valid: bool = true,
    optimizer_step: u64 = 0,
    last_spectral_before: f32 = 0.0,
    last_spectral_after: f32 = 0.0,

    const Self = @This();

    pub fn init(model_dim: usize) AccelError!Self {
        return initMultiLayer(model_dim, 1, std.heap.page_allocator);
    }

    pub fn initMultiLayer(model_dim: usize, num_layers: usize, allocator: std.mem.Allocator) AccelError!Self {
        return initMultiLayerWithDepthScale(model_dim, num_layers, allocator, true);
    }

    pub fn initMultiLayerWithDepthScale(
        model_dim: usize,
        num_layers: usize,
        allocator: std.mem.Allocator,
        depth_compensation: bool,
    ) AccelError!Self {
        if (model_dim == 0) return AccelError.InvalidDimensions;
        if (model_dim % 2 != 0) return AccelError.InvalidDimensions;
        if (num_layers == 0) return AccelError.InvalidDimensions;
        const half: usize = model_dim / 2;

        var ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const base_seed: u64 = 0x4A41494445204E4F;
        const depth_scale: f32 = if (depth_compensation)
            1.0 / @sqrt(@as(f32, @floatFromInt(num_layers)))
        else
            1.0;
        const init_stddev: f32 = depth_scale * 0.25 / @sqrt(@as(f32, @floatFromInt(half)));

        var layers = allocator.alloc(RSFLayer, num_layers) catch return AccelError.AllocationFailed;
        errdefer allocator.free(layers);

        const columns = std.math.add(usize, half, 1) catch return AccelError.InvalidDimensions;
        const per_layer = std.math.mul(usize, half, columns) catch return AccelError.InvalidDimensions;
        const stack_count = std.math.mul(usize, num_layers, per_layer) catch return AccelError.InvalidDimensions;
        const master_s_data = allocator.alloc(f32, stack_count) catch return AccelError.AllocationFailed;
        defer allocator.free(master_s_data);
        const master_t_data = allocator.alloc(f32, stack_count) catch return AccelError.AllocationFailed;
        defer allocator.free(master_t_data);
        const shadow_s_data = allocator.alloc(f16, stack_count) catch return AccelError.AllocationFailed;
        defer allocator.free(shadow_s_data);
        const shadow_t_data = allocator.alloc(f16, stack_count) catch return AccelError.AllocationFailed;
        defer allocator.free(shadow_t_data);
        const zeros = allocator.alloc(f32, stack_count) catch return AccelError.AllocationFailed;
        defer allocator.free(zeros);
        @memset(zeros, 0.0);

        var layers_built: usize = 0;
        errdefer {
            var index: usize = 0;
            while (index < layers_built) : (index += 1) layers[index].free(&ctx);
        }

        var layer_index: usize = 0;
        while (layer_index < num_layers) : (layer_index += 1) {
            const layer_seed = base_seed +% (@as(u64, @intCast(layer_index)) *% 0x9E3779B97F4A7C15);
            var rng = std.Random.DefaultPrng.init(layer_seed);
            const random = rng.random();
            const base = layer_index * per_layer;
            var index: usize = 0;
            while (index < per_layer) : (index += 1) {
                const value_s = random.floatNorm(f32) * init_stddev;
                const value_t = random.floatNorm(f32) * init_stddev;
                master_s_data[base + index] = value_s;
                master_t_data[base + index] = value_t;
                shadow_s_data[base + index] = @floatCast(value_s);
                shadow_t_data[base + index] = @floatCast(value_t);
            }
            var row: usize = 0;
            while (row < half) : (row += 1) {
                const bias_index = base + row * columns + half;
                master_s_data[bias_index] = 0.0;
                master_t_data[bias_index] = 0.0;
                shadow_s_data[bias_index] = 0.0;
                shadow_t_data[bias_index] = 0.0;
            }
            var layer_s = try FutharkArray2DF16.newFromFlat(&ctx, shadow_s_data[base .. base + per_layer], half, columns);
            errdefer layer_s.free(&ctx);
            const layer_t = try FutharkArray2DF16.newFromFlat(&ctx, shadow_t_data[base .. base + per_layer], half, columns);
            layers[layer_index] = .{ .weights_s = layer_s, .weights_t = layer_t };
            layers_built += 1;
        }

        var stack_master_s = try FutharkArray3DF32.newFromFlat(&ctx, master_s_data, num_layers, half, columns);
        errdefer stack_master_s.free(&ctx);
        var stack_master_t = try FutharkArray3DF32.newFromFlat(&ctx, master_t_data, num_layers, half, columns);
        errdefer stack_master_t.free(&ctx);
        var stack_shadow_s = try FutharkArray3DF16.newFromFlat(&ctx, shadow_s_data, num_layers, half, columns);
        errdefer stack_shadow_s.free(&ctx);
        var stack_shadow_t = try FutharkArray3DF16.newFromFlat(&ctx, shadow_t_data, num_layers, half, columns);
        errdefer stack_shadow_t.free(&ctx);
        var momentum_s = try FutharkArray3DF32.newFromFlat(&ctx, zeros, num_layers, half, columns);
        errdefer momentum_s.free(&ctx);
        var momentum_t = try FutharkArray3DF32.newFromFlat(&ctx, zeros, num_layers, half, columns);
        errdefer momentum_t.free(&ctx);
        var fisher_s = try FutharkArray3DF32.newFromFlat(&ctx, zeros, num_layers, half, columns);
        errdefer fisher_s.free(&ctx);
        var fisher_t = try FutharkArray3DF32.newFromFlat(&ctx, zeros, num_layers, half, columns);
        errdefer fisher_t.free(&ctx);

        const max_batch: usize = 2048;
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);

        return .{
            .ctx = ctx,
            .layers = layers,
            .layers_owner = allocator,
            .allocator = allocator,
            .model_dim = model_dim,
            .num_layers = num_layers,
            .clip_min = -5.0,
            .clip_max = 5.0,
            .initialized = true,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
            .stack_weights_s = stack_shadow_s,
            .stack_weights_t = stack_shadow_t,
            .stack_master_weights_s = stack_master_s,
            .stack_master_weights_t = stack_master_t,
            .stack_momentum_s = momentum_s,
            .stack_momentum_t = momentum_t,
            .stack_fisher_s = fisher_s,
            .stack_fisher_t = fisher_t,
            .stack_arrays_valid = true,
            .layers_mirror_valid = true,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        if (self.scratch_lengths_buf.len > 0) {
            self.allocator.free(self.scratch_lengths_buf);
            self.scratch_lengths_buf = &[_]i64{};
        }
        self.freeStackArrays();

        var i: usize = self.layers.len;
        while (i > 0) {
            i -= 1;
            self.layers[i].free(&self.ctx);
        }
        self.layers_owner.free(self.layers);
        self.ctx.deinit();
        self.initialized = false;
    }

    fn freeStackArrays(self: *Self) void {
        if (self.stack_fisher_t) |*a| a.free(&self.ctx);
        if (self.stack_fisher_s) |*a| a.free(&self.ctx);
        if (self.stack_momentum_t) |*a| a.free(&self.ctx);
        if (self.stack_momentum_s) |*a| a.free(&self.ctx);
        if (self.stack_weights_t) |*a| a.free(&self.ctx);
        if (self.stack_weights_s) |*a| a.free(&self.ctx);
        if (self.stack_master_weights_t) |*a| a.free(&self.ctx);
        if (self.stack_master_weights_s) |*a| a.free(&self.ctx);
        self.stack_fisher_t = null;
        self.stack_fisher_s = null;
        self.stack_momentum_t = null;
        self.stack_momentum_s = null;
        self.stack_weights_t = null;
        self.stack_weights_s = null;
        self.stack_master_weights_t = null;
        self.stack_master_weights_s = null;
        self.stack_arrays_valid = false;
    }

    fn markStackDirty(self: *Self) void {
        self.stack_arrays_valid = false;
    }

    pub fn numLayers(self: *const Self) usize {
        return self.num_layers;
    }

    pub fn layerPtr(self: *Self, layer_idx: usize) AccelError!*RSFLayer {
        if (!self.initialized) return AccelError.NullPointer;
        if (layer_idx >= self.layers.len) return AccelError.InvalidDimensions;
        return &self.layers[layer_idx];
    }

    fn stackStateDimensionsMatch(state: FutharkArray3DF32, layers: usize, half: usize, cols: usize) bool {
        return state.dim0 == layers and state.dim1 == half and state.dim2 == cols and state.arr != null;
    }

    fn ensureStackPacked(self: *Self) AccelError!void {
        if (self.stack_arrays_valid and self.stack_weights_s != null and self.stack_weights_t != null and
            self.stack_master_weights_s != null and self.stack_master_weights_t != null and
            self.stack_momentum_s != null and self.stack_momentum_t != null and
            self.stack_fisher_s != null and self.stack_fisher_t != null) return;

        const l_count = self.layers.len;
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, l_count, per_layer) catch return AccelError.InvalidDimensions;
        const ws_flat = self.allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(ws_flat);
        const wt_flat = self.allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(wt_flat);

        for (self.layers, 0..) |*layer, layer_index| {
            const ws = layer.weights_s.valuesFlat(&self.ctx, self.allocator) catch return AccelError.FutharkValuesFailed;
            defer self.allocator.free(ws);
            const wt = layer.weights_t.valuesFlat(&self.ctx, self.allocator) catch return AccelError.FutharkValuesFailed;
            defer self.allocator.free(wt);
            if (ws.len != per_layer or wt.len != per_layer) return AccelError.InvalidDimensions;
            @memcpy(ws_flat[layer_index * per_layer .. (layer_index + 1) * per_layer], ws);
            @memcpy(wt_flat[layer_index * per_layer .. (layer_index + 1) * per_layer], wt);
        }

        var replacement_s = try FutharkArray3DF16.newFromFlat(&self.ctx, ws_flat, l_count, half, cols);
        errdefer replacement_s.free(&self.ctx);
        var replacement_t = try FutharkArray3DF16.newFromFlat(&self.ctx, wt_flat, l_count, half, cols);
        errdefer replacement_t.free(&self.ctx);
        const ws_master_flat = self.allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(ws_master_flat);
        const wt_master_flat = self.allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer self.allocator.free(wt_master_flat);
        for (ws_flat, ws_master_flat) |value, *master| master.* = @floatCast(value);
        for (wt_flat, wt_master_flat) |value, *master| master.* = @floatCast(value);
        var replacement_master_s = try FutharkArray3DF32.newFromFlat(&self.ctx, ws_master_flat, l_count, half, cols);
        errdefer replacement_master_s.free(&self.ctx);
        var replacement_master_t = try FutharkArray3DF32.newFromFlat(&self.ctx, wt_master_flat, l_count, half, cols);
        errdefer replacement_master_t.free(&self.ctx);

        const momentum_s = self.stack_momentum_s orelse return AccelError.NullPointer;
        const momentum_t = self.stack_momentum_t orelse return AccelError.NullPointer;
        const fisher_s = self.stack_fisher_s orelse return AccelError.NullPointer;
        const fisher_t = self.stack_fisher_t orelse return AccelError.NullPointer;
        if (!stackStateDimensionsMatch(momentum_s, l_count, half, cols) or
            !stackStateDimensionsMatch(momentum_t, l_count, half, cols) or
            !stackStateDimensionsMatch(fisher_s, l_count, half, cols) or
            !stackStateDimensionsMatch(fisher_t, l_count, half, cols)) return AccelError.InvalidDimensions;

        if (self.stack_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_weights_t) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_t) |*old| old.free(&self.ctx);
        self.stack_weights_s = replacement_s;
        self.stack_weights_t = replacement_t;
        self.stack_master_weights_s = replacement_master_s;
        self.stack_master_weights_t = replacement_master_t;
        self.stack_arrays_valid = true;
        self.layers_mirror_valid = true;
    }

    fn assignLayerWeightsDirect(self: *Self, layer_idx: usize, is_s: bool, data: []const f16, rows: usize, cols: usize) AccelError!void {
        var replacement = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        errdefer replacement.free(&self.ctx);
        const layer = try self.layerPtr(layer_idx);
        if (is_s) {
            layer.weights_s.free(&self.ctx);
            layer.weights_s = replacement;
        } else {
            layer.weights_t.free(&self.ctx);
            layer.weights_t = replacement;
        }
    }

    pub fn syncLayersFromStack(self: *Self) AccelError!void {
        if (!self.stack_arrays_valid) return;
        if (self.layers_mirror_valid) return;
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const l_count = self.layers.len;
        const total = std.math.mul(usize, l_count, per_layer) catch return AccelError.InvalidDimensions;

        if (self.stack_weights_s) |*sws| {
            const flat = try sws.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(flat);
            if (flat.len != total) return AccelError.InvalidDimensions;
            var li: usize = 0;
            while (li < l_count) : (li += 1) {
                try self.assignLayerWeightsDirect(li, true, flat[li * per_layer .. (li + 1) * per_layer], half, cols);
            }
        }
        if (self.stack_weights_t) |*swt| {
            const flat = try swt.valuesFlat(&self.ctx, self.allocator);
            defer self.allocator.free(flat);
            if (flat.len != total) return AccelError.InvalidDimensions;
            var li: usize = 0;
            while (li < l_count) : (li += 1) {
                try self.assignLayerWeightsDirect(li, false, flat[li * per_layer .. (li + 1) * per_layer], half, cols);
            }
        }
        self.layers_mirror_valid = true;
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!RSFOptimizerState {
        try self.ensureStackPacked();
        const master_s = try self.stack_master_weights_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(master_s);
        const master_t = try self.stack_master_weights_t.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(master_t);
        const ms = try self.stack_momentum_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(ms);
        const mt = try self.stack_momentum_t.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(mt);
        const fs = try self.stack_fisher_s.?.valuesFlat(&self.ctx, allocator);
        errdefer allocator.free(fs);
        const ft = try self.stack_fisher_t.?.valuesFlat(&self.ctx, allocator);
        return .{ .master_weights_s = master_s, .master_weights_t = master_t, .momentum_s = ms, .momentum_t = mt, .fisher_s = fs, .fisher_t = ft, .step = self.optimizer_step, .allocator = allocator };
    }

    pub fn setOptimizerState(
        self: *Self,
        master_weights_s: []const f32,
        master_weights_t: []const f32,
        momentum_s: []const f32,
        momentum_t: []const f32,
        fisher_s: []const f32,
        fisher_t: []const f32,
        step: u64,
    ) AccelError!void {
        try self.ensureStackPacked();
        const half = self.model_dim / 2;
        const cols = half + 1;
        const per_layer = std.math.mul(usize, half, cols) catch return AccelError.InvalidDimensions;
        const total = std.math.mul(usize, self.num_layers, per_layer) catch return AccelError.InvalidDimensions;
        if (master_weights_s.len != total or master_weights_t.len != total or momentum_s.len != total or momentum_t.len != total or fisher_s.len != total or fisher_t.len != total) return AccelError.InvalidDimensions;
        for (master_weights_s) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (master_weights_t) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum_s) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum_t) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (fisher_s) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;
        for (fisher_t) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;

        var new_master_s = try FutharkArray3DF32.newFromFlat(&self.ctx, master_weights_s, self.num_layers, half, cols);
        errdefer new_master_s.free(&self.ctx);
        var new_master_t = try FutharkArray3DF32.newFromFlat(&self.ctx, master_weights_t, self.num_layers, half, cols);
        errdefer new_master_t.free(&self.ctx);
        var new_ms = try FutharkArray3DF32.newFromFlat(&self.ctx, momentum_s, self.num_layers, half, cols);
        errdefer new_ms.free(&self.ctx);
        var new_mt = try FutharkArray3DF32.newFromFlat(&self.ctx, momentum_t, self.num_layers, half, cols);
        errdefer new_mt.free(&self.ctx);
        var new_fs = try FutharkArray3DF32.newFromFlat(&self.ctx, fisher_s, self.num_layers, half, cols);
        errdefer new_fs.free(&self.ctx);
        var new_ft = try FutharkArray3DF32.newFromFlat(&self.ctx, fisher_t, self.num_layers, half, cols);
        errdefer new_ft.free(&self.ctx);
        var forward_s: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &forward_s, new_master_s.arr) != 0 or forward_s == null) return AccelError.FutharkScaleWeightsFailed;
        var new_forward_s = FutharkArray3DF16{ .arr = forward_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        errdefer new_forward_s.free(&self.ctx);
        var forward_t: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &forward_t, new_master_t.arr) != 0 or forward_t == null) return AccelError.FutharkScaleWeightsFailed;
        var new_forward_t = FutharkArray3DF16{ .arr = forward_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        errdefer new_forward_t.free(&self.ctx);
        self.freeStackArraysForReplacement();
        self.stack_weights_s = new_forward_s;
        self.stack_weights_t = new_forward_t;
        self.stack_master_weights_s = new_master_s;
        self.stack_master_weights_t = new_master_t;
        self.stack_momentum_s = new_ms;
        self.stack_momentum_t = new_mt;
        self.stack_fisher_s = new_fs;
        self.stack_fisher_t = new_ft;
        self.stack_arrays_valid = true;
        self.layers_mirror_valid = false;
        self.optimizer_step = step;
    }

    pub fn forward(self: *Self, input: *FutharkArray2DF16) AccelError!FutharkArray2DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (input.arr == null) return AccelError.NullPointer;
        if (self.layers.len == 0) return AccelError.NullPointer;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();

        try self.syncLayersFromStack();

        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var current_arr: ?*futhark.struct_futhark_f16_2d = input.arr;
        const rows = input.rows;
        const cols = input.cols;

        var li: usize = 0;
        while (li < self.layers.len) : (li += 1) {
            const layer = &self.layers[li];
            if (layer.weights_s.arr == null or layer.weights_t.arr == null) return AccelError.NullPointer;

            var next_arr: ?*futhark.struct_futhark_f16_2d = null;
            const result = futhark.futhark_entry_rsf_forward(
                self.ctx.ctx,
                &next_arr,
                current_arr,
                layer.weights_s.arr,
                layer.weights_t.arr,
                clip_min_bits,
                clip_max_bits,
            );
            if (result != 0) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.FutharkForwardFailed;
            }
            if (next_arr == null) {
                if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
                return AccelError.NullPointer;
            }

            if (li > 0) _ = futhark.futhark_free_f16_2d(self.ctx.ctx, current_arr);
            current_arr = next_arr;
        }

        return FutharkArray2DF16{ .arr = current_arr, .rows = rows, .cols = cols };
    }

    pub fn stackForward(self: *Self, inputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null) return AccelError.NullPointer;
        if (inputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_forward(
            self.ctx.ctx,
            &out,
            inputs.arr,
            sws.arr,
            swt.arr,
            clip_min_bits,
            clip_max_bits,
        );
        if (rc != 0 or out == null) {
            if (out) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
            return AccelError.FutharkForwardFailed;
        }
        return FutharkArray3DF16{ .arr = out, .dim0 = inputs.dim0, .dim1 = inputs.dim1, .dim2 = inputs.dim2 };
    }

    pub fn stackInverse(self: *Self, outputs: *FutharkArray3DF16) AccelError!FutharkArray3DF16 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (outputs.arr == null) return AccelError.NullPointer;
        if (outputs.dim2 != self.model_dim) return AccelError.InvalidDimensions;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const clip_min_bits: u16 = @bitCast(self.clip_min);
        const clip_max_bits: u16 = @bitCast(self.clip_max);

        var out: ?*futhark.struct_futhark_f16_3d = null;
        const rc = futhark.futhark_entry_rsf_stack_inverse(
            self.ctx.ctx,
            &out,
            outputs.arr,
            sws.arr,
            swt.arr,
            clip_min_bits,
            clip_max_bits,
        );
        if (rc != 0 or out == null) {
            if (out) |o| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, o);
            return AccelError.FutharkBackwardFailed;
        }
        return FutharkArray3DF16{ .arr = out, .dim0 = outputs.dim0, .dim1 = outputs.dim1, .dim2 = outputs.dim2 };
    }

    pub fn fusedTrainingStep(
        self: *Self,
        inputs: *FutharkArray3DF16,
        targets: *FutharkArray3DF16,
        sequence_lengths: []const usize,
        grad_mean: bool,
        gradient_scale: f32,
        reconstruction_alpha: f32,
        forward_scale: f32,
        logdet_weight: f32,
    ) AccelError!FusedStepResult {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (inputs.arr == null or targets.arr == null) return AccelError.NullPointer;
        if (inputs.dim0 != targets.dim0 or inputs.dim1 != targets.dim1 or inputs.dim2 != targets.dim2) return AccelError.InvalidDimensions;
        if (inputs.dim2 != self.model_dim or sequence_lengths.len != inputs.dim0) return AccelError.InvalidDimensions;
        if (!std.math.isFinite(gradient_scale) or gradient_scale < 0.0 or gradient_scale > 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(reconstruction_alpha) or !std.math.isFinite(forward_scale) or !std.math.isFinite(logdet_weight)) return AccelError.InvalidHyperparameter;

        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > inputs.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }
        var lengths_array = try FutharkArray1DI64.newFromSlice(&self.ctx, lengths_i64);
        defer lengths_array.free(&self.ctx);

        const sws = self.stack_weights_s orelse return AccelError.NullPointer;
        const swt = self.stack_weights_t orelse return AccelError.NullPointer;
        const clip_min_f32: f32 = @floatCast(self.clip_min);
        const clip_max_f32: f32 = @floatCast(self.clip_max);

        var final_outputs: ?*futhark.struct_futhark_f16_3d = null;
        var out_tuple: ?*futhark.struct_futhark_opaque_tup6_fused_stack_gradients = null;
        const forward_result = futhark.futhark_entry_rsf_stack_forward(
            self.ctx.ctx,
            &final_outputs,
            inputs.arr,
            sws.arr,
            swt.arr,
            @bitCast(self.clip_min),
            @bitCast(self.clip_max),
        );
        if (forward_result != 0 or final_outputs == null) {
            if (final_outputs) |output| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, output);
            return AccelError.FutharkForwardFailed;
        }

        const backward_result = futhark.futhark_entry_rsf_stack_backward_gradients_fused(
            self.ctx.ctx,
            &out_tuple,
            final_outputs,
            targets.arr,
            inputs.arr,
            lengths_array.arr,
            sws.arr,
            swt.arr,
            grad_mean,
            gradient_scale,
            clip_min_f32,
            clip_max_f32,
            reconstruction_alpha,
            forward_scale,
            logdet_weight,
        );
        _ = futhark.futhark_free_f16_3d(self.ctx.ctx, final_outputs);
        if (backward_result != 0 or out_tuple == null) {
            if (out_tuple) |tuple| _ = futhark.futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(self.ctx.ctx, tuple);
            const message = futhark.futhark_context_get_error(self.ctx.ctx);
            defer freeFutharkError(message);
            if (message) |text| std.debug.print("[Futhark rsf_stack_backward_gradients_fused error] {s}\n", .{std.mem.span(text)});
            return AccelError.FutharkTrainingStepFailed;
        }

        const tuple = out_tuple.?;
        var gradient_s: ?*futhark.struct_futhark_f32_3d = null;
        var gradient_t: ?*futhark.struct_futhark_f32_3d = null;
        var delta: ?*futhark.struct_futhark_f16_3d = null;
        const projection_s = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_0(self.ctx.ctx, &gradient_s, tuple);
        const projection_t = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_1(self.ctx.ctx, &gradient_t, tuple);
        const projection_delta = futhark.futhark_project_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32_2(self.ctx.ctx, &delta, tuple);
        if (projection_s != 0 or projection_t != 0 or projection_delta != 0 or gradient_s == null or gradient_t == null or delta == null) {
            if (gradient_s) |array| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
            if (gradient_t) |array| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
            if (delta) |array| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, array);
            _ = futhark.futhark_free_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32(self.ctx.ctx, tuple);
            return AccelError.FutharkTrainingStepFailed;
        }

        const half = self.model_dim / 2;
        const columns = std.math.add(usize, half, 1) catch return AccelError.InvalidDimensions;
        return .{
            .stack_gradient_s = .{ .arr = gradient_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns },
            .stack_gradient_t = .{ .arr = gradient_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns },
            .input_delta = .{ .arr = delta, .dim0 = inputs.dim0, .dim1 = inputs.dim1, .dim2 = inputs.dim2 },
            .pending = out_tuple,
            .finalized = false,
            .scalars = .{ .loss = 0.0, .reconstruction_loss = 0.0, .logdet_mean = 0.0 },
        };
    }

    pub fn applyStackGradientsSFD(
        self: *Self,
        gradient_s: *FutharkArray3DF32,
        gradient_t: *FutharkArray3DF32,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (!std.math.isFinite(learning_rate) or learning_rate < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(momentum_beta) or momentum_beta < 0.0 or momentum_beta >= 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(fisher_gamma) or fisher_gamma < 0.0 or fisher_gamma >= 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(epsilon) or epsilon <= 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(trust_ratio) or trust_ratio <= 0.0 or trust_ratio > 1.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(weight_floor) or weight_floor <= 0.0) return AccelError.InvalidHyperparameter;

        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();
        const half = self.model_dim / 2;
        const columns = std.math.add(usize, half, 1) catch return AccelError.InvalidDimensions;
        if (gradient_s.arr == null or gradient_t.arr == null or
            gradient_s.dim0 != self.num_layers or gradient_t.dim0 != self.num_layers or
            gradient_s.dim1 != half or gradient_t.dim1 != half or
            gradient_s.dim2 != columns or gradient_t.dim2 != columns) return AccelError.InvalidDimensions;

        const master_s = self.stack_master_weights_s orelse return AccelError.NullPointer;
        const master_t = self.stack_master_weights_t orelse return AccelError.NullPointer;
        const momentum_s = self.stack_momentum_s orelse return AccelError.NullPointer;
        const momentum_t = self.stack_momentum_t orelse return AccelError.NullPointer;
        const fisher_s = self.stack_fisher_s orelse return AccelError.NullPointer;
        const fisher_t = self.stack_fisher_t orelse return AccelError.NullPointer;
        const next_step: i64 = @intCast(@min(self.optimizer_step +| 1, @as(u64, std.math.maxInt(i64))));

        var tuple_s: ?*futhark.struct_futhark_opaque_tup3_stack_sfd = null;
        var tuple_t: ?*futhark.struct_futhark_opaque_tup3_stack_sfd = null;
        if (futhark.futhark_entry_stack_update_sfd_master(self.ctx.ctx, &tuple_s, master_s.arr, gradient_s.arr, momentum_s.arr, fisher_s.arr, learning_rate, momentum_beta, fisher_gamma, next_step, epsilon, trust_ratio, weight_floor) != 0 or tuple_s == null) return AccelError.FutharkTrainingStepFailed;
        defer if (tuple_s) |tuple| {
            _ = futhark.futhark_free_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32(self.ctx.ctx, tuple);
        };
        if (futhark.futhark_entry_stack_update_sfd_master(self.ctx.ctx, &tuple_t, master_t.arr, gradient_t.arr, momentum_t.arr, fisher_t.arr, learning_rate, momentum_beta, fisher_gamma, next_step, epsilon, trust_ratio, weight_floor) != 0 or tuple_t == null) return AccelError.FutharkTrainingStepFailed;
        defer if (tuple_t) |tuple| {
            _ = futhark.futhark_free_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32(self.ctx.ctx, tuple);
        };

        var new_master_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_master_t: ?*futhark.struct_futhark_f32_3d = null;
        var new_momentum_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_momentum_t: ?*futhark.struct_futhark_f32_3d = null;
        var new_fisher_s: ?*futhark.struct_futhark_f32_3d = null;
        var new_fisher_t: ?*futhark.struct_futhark_f32_3d = null;
        errdefer if (new_master_s) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };
        errdefer if (new_master_t) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };
        errdefer if (new_momentum_s) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };
        errdefer if (new_momentum_t) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };
        errdefer if (new_fisher_s) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };
        errdefer if (new_fisher_t) |array| {
            _ = futhark.futhark_free_f32_3d(self.ctx.ctx, array);
        };

        const projections = [_]c_int{
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(self.ctx.ctx, &new_master_s, tuple_s.?),
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(self.ctx.ctx, &new_momentum_s, tuple_s.?),
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(self.ctx.ctx, &new_fisher_s, tuple_s.?),
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_0(self.ctx.ctx, &new_master_t, tuple_t.?),
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_1(self.ctx.ctx, &new_momentum_t, tuple_t.?),
            futhark.futhark_project_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32_2(self.ctx.ctx, &new_fisher_t, tuple_t.?),
        };
        for (projections) |projection| if (projection != 0) return AccelError.FutharkTrainingStepFailed;
        if (new_master_s == null or new_master_t == null or new_momentum_s == null or new_momentum_t == null or new_fisher_s == null or new_fisher_t == null) return AccelError.FutharkTrainingStepFailed;

        var new_weights_s: ?*futhark.struct_futhark_f16_3d = null;
        var new_weights_t: ?*futhark.struct_futhark_f16_3d = null;
        errdefer if (new_weights_s) |array| {
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, array);
        };
        errdefer if (new_weights_t) |array| {
            _ = futhark.futhark_free_f16_3d(self.ctx.ctx, array);
        };
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &new_weights_s, new_master_s) != 0 or new_weights_s == null) return AccelError.FutharkScaleWeightsFailed;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &new_weights_t, new_master_t) != 0 or new_weights_t == null) return AccelError.FutharkScaleWeightsFailed;

        self.freeStackArraysForReplacement();
        self.stack_weights_s = .{ .arr = new_weights_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_weights_t = .{ .arr = new_weights_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_master_weights_s = .{ .arr = new_master_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_master_weights_t = .{ .arr = new_master_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_momentum_s = .{ .arr = new_momentum_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_momentum_t = .{ .arr = new_momentum_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_fisher_s = .{ .arr = new_fisher_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_fisher_t = .{ .arr = new_fisher_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = columns };
        self.stack_arrays_valid = true;
        self.layers_mirror_valid = false;
        self.optimizer_step +|= 1;
    }

    fn freeStackArraysForReplacement(self: *Self) void {
        if (self.stack_fisher_t) |*a| a.free(&self.ctx);
        self.stack_fisher_t = null;
        if (self.stack_fisher_s) |*a| a.free(&self.ctx);
        self.stack_fisher_s = null;
        if (self.stack_momentum_t) |*a| a.free(&self.ctx);
        self.stack_momentum_t = null;
        if (self.stack_momentum_s) |*a| a.free(&self.ctx);
        self.stack_momentum_s = null;
        if (self.stack_master_weights_t) |*a| a.free(&self.ctx);
        self.stack_master_weights_t = null;
        if (self.stack_master_weights_s) |*a| a.free(&self.ctx);
        self.stack_master_weights_s = null;
        if (self.stack_weights_t) |*a| a.free(&self.ctx);
        self.stack_weights_t = null;
        if (self.stack_weights_s) |*a| a.free(&self.ctx);
        self.stack_weights_s = null;
        self.stack_arrays_valid = false;
    }

    pub fn sync(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        return self.ctx.sync();
    }

    pub fn setLayerWeightsS(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const total = std.math.mul(usize, rows, cols) catch return AccelError.InvalidDimensions;
        if (rows == 0 or cols == 0 or data.len != total) return AccelError.InvalidDimensions;
        try self.syncLayersFromStack();
        const replacement = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        const layer = try self.layerPtr(layer_idx);
        layer.weights_s.free(&self.ctx);
        layer.weights_s = replacement;
        self.markStackDirty();
        self.layers_mirror_valid = true;
    }

    pub fn setLayerWeightsT(self: *Self, layer_idx: usize, data: []const f16, rows: usize, cols: usize) AccelError!void {
        const total = std.math.mul(usize, rows, cols) catch return AccelError.InvalidDimensions;
        if (rows == 0 or cols == 0 or data.len != total) return AccelError.InvalidDimensions;
        try self.syncLayersFromStack();
        const replacement = try FutharkArray2DF16.newFromFlat(&self.ctx, data, rows, cols);
        const layer = try self.layerPtr(layer_idx);
        layer.weights_t.free(&self.ctx);
        layer.weights_t = replacement;
        self.markStackDirty();
        self.layers_mirror_valid = true;
    }

    fn normalizeMasterStackLocked(self: *Self, master: FutharkArray3DF32, target: f32, iterations: usize) AccelError!struct { array: FutharkArray3DF32, before: f32, after: f32 } {
        var tuple: ?*futhark.struct_futhark_opaque_tup3_stack_spectral = null;
        const rc = futhark.futhark_entry_stack_spectral_normalize(
            self.ctx.ctx,
            &tuple,
            master.arr,
            target,
            @intCast(iterations),
        );
        if (rc != 0 or tuple == null) return AccelError.FutharkForwardFailed;
        var array: ?*futhark.struct_futhark_f32_3d = null;
        var before: f32 = 0.0;
        var after: f32 = 0.0;
        const p0 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_0(self.ctx.ctx, &array, tuple);
        const p1 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_1(self.ctx.ctx, &before, tuple);
        const p2 = futhark.futhark_project_opaque_tup3_arr3d_f32_f32_f32_2(self.ctx.ctx, &after, tuple);
        _ = futhark.futhark_free_opaque_tup3_arr3d_f32_f32_f32(self.ctx.ctx, tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or array == null or !std.math.isFinite(before) or !std.math.isFinite(after)) {
            if (array) |value| _ = futhark.futhark_free_f32_3d(self.ctx.ctx, value);
            return AccelError.FutharkForwardFailed;
        }
        return .{
            .array = .{ .arr = array, .dim0 = master.dim0, .dim1 = master.dim1, .dim2 = master.dim2 },
            .before = before,
            .after = after,
        };
    }

    pub fn spectralNormalizeLayers(self: *Self, target: f32, iterations: usize) AccelError!void {
        if (!std.math.isFinite(target) or target <= 0.0 or iterations == 0) return AccelError.InvalidHyperparameter;
        self.ctx.mutex.lock();
        defer self.ctx.mutex.unlock();
        try self.ensureStackPacked();
        const master_s = self.stack_master_weights_s orelse return AccelError.NullPointer;
        const master_t = self.stack_master_weights_t orelse return AccelError.NullPointer;
        var normalized_s = try self.normalizeMasterStackLocked(master_s, target, iterations);
        errdefer normalized_s.array.free(&self.ctx);
        var normalized_t = try self.normalizeMasterStackLocked(master_t, target, iterations);
        errdefer normalized_t.array.free(&self.ctx);
        var shadow_s: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &shadow_s, normalized_s.array.arr) != 0 or shadow_s == null) return AccelError.FutharkScaleWeightsFailed;
        errdefer _ = futhark.futhark_free_f16_3d(self.ctx.ctx, shadow_s);
        var shadow_t: ?*futhark.struct_futhark_f16_3d = null;
        if (futhark.futhark_entry_master_weights_to_f16_3d(self.ctx.ctx, &shadow_t, normalized_t.array.arr) != 0 or shadow_t == null) return AccelError.FutharkScaleWeightsFailed;
        errdefer _ = futhark.futhark_free_f16_3d(self.ctx.ctx, shadow_t);
        if (self.stack_master_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_master_weights_t) |*old| old.free(&self.ctx);
        if (self.stack_weights_s) |*old| old.free(&self.ctx);
        if (self.stack_weights_t) |*old| old.free(&self.ctx);
        self.stack_master_weights_s = normalized_s.array;
        self.stack_master_weights_t = normalized_t.array;
        const half = self.model_dim / 2;
        const cols = half + 1;
        self.stack_weights_s = .{ .arr = shadow_s, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_weights_t = .{ .arr = shadow_t, .dim0 = self.num_layers, .dim1 = half, .dim2 = cols };
        self.stack_arrays_valid = true;
        self.layers_mirror_valid = false;
        self.last_spectral_before = @max(normalized_s.before, normalized_t.before);
        self.last_spectral_after = @max(normalized_s.after, normalized_t.after);
    }

    pub fn setClipRange(self: *Self, clip_min_val: f16, clip_max_val: f16) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        const minimum: f32 = @floatCast(clip_min_val);
        const maximum: f32 = @floatCast(clip_max_val);
        if (!std.math.isFinite(minimum) or !std.math.isFinite(maximum)) return AccelError.InvalidClipRange;
        if (minimum >= maximum or minimum < -20.0 or maximum > 20.0) return AccelError.InvalidClipRange;
        self.clip_min = clip_min_val;
        self.clip_max = clip_max_val;
    }

    pub fn forwardFromTensor(self: *Self, input: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        if (!self.initialized) return AccelError.NullPointer;
        if (input.shape.dims.len != 2) return AccelError.InvalidDimensions;
        const rows = input.shape.dims[0];
        const cols = input.shape.dims[1];
        const f16_data = allocator.alloc(f16, try checkedElementCount2(rows, cols)) catch return AccelError.AllocationFailed;
        defer allocator.free(f16_data);
        {
            var i: usize = 0;
            while (i < input.data.len) : (i += 1) {
                const v = input.data[i];
                f16_data[i] = @floatCast(v);
            }
        }
        var f16_input = try FutharkArray2DF16.newFromFlat(&self.ctx, f16_data, rows, cols);
        defer f16_input.free(&self.ctx);
        var output = try self.forward(&f16_input);
        defer output.free(&self.ctx);
        const shape = [_]usize{ output.rows, output.cols };
        var result = core_tensor.Tensor.init(allocator, &shape) catch return AccelError.AllocationFailed;
        const out_f16 = allocator.alloc(f16, try checkedElementCount2(output.rows, output.cols)) catch {
            result.deinit();
            return AccelError.AllocationFailed;
        };
        defer allocator.free(out_f16);
        if (futhark.futhark_values_f16_2d(self.ctx.ctx, output.arr, @ptrCast(out_f16.ptr)) != 0) {
            result.deinit();
            return AccelError.FutharkValuesFailed;
        }
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) {
            result.deinit();
            return AccelError.FutharkSyncFailed;
        }
        {
            var i: usize = 0;
            while (i < out_f16.len) : (i += 1) {
                const v = out_f16[i];
                result.data[i] = @floatCast(v);
            }
        }
        return result;
    }
};

pub const GPUOps = struct {
    ctx: FutharkContext,

    const Self = @This();

    pub fn init() AccelError!Self {
        return Self{ .ctx = try FutharkContext.init() };
    }

    pub fn deinit(self: *Self) void {
        self.ctx.deinit();
    }

    pub fn matmul(self: *Self, a: *const core_tensor.Tensor, b: *const core_tensor.Tensor, allocator: std.mem.Allocator) AccelError!core_tensor.Tensor {
        var fa = try FutharkArray2DF32.fromTensor(&self.ctx, a);
        defer fa.free(&self.ctx);
        var fb = try FutharkArray2DF32.fromTensor(&self.ctx, b);
        defer fb.free(&self.ctx);

        var out_arr: ?*futhark.struct_futhark_f32_2d = null;
        if (futhark.futhark_entry_matmul(self.ctx.ctx, &out_arr, fa.arr, fb.arr) != 0) {
            return AccelError.FutharkForwardFailed;
        }
        if (out_arr == null) return AccelError.NullPointer;

        var result = FutharkArray2DF32{ .arr = out_arr, .rows = a.shape.dims[0], .cols = b.shape.dims[1] };
        defer result.free(&self.ctx);
        return result.toTensor(&self.ctx, allocator);
    }
};

pub const FutharkArray1DI64 = struct {
    arr: ?*futhark.struct_futhark_i64_1d,
    len: usize,

    const Self = @This();

    pub fn newFromSlice(ctx: *FutharkContext, data: []const i64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (data.len == 0) return AccelError.InvalidDimensions;
        const arr = futhark.futhark_new_i64_1d(ctx.ctx, data.ptr, try checkedDimensionI64(data.len));
        if (arr == null) return AccelError.FutharkArrayNewFailed;
        return Self{ .arr = arr, .len = data.len };
    }

    pub fn valuesSlice(self: *FutharkArray1DI64, ctx: *FutharkContext, allocator: std.mem.Allocator) AccelError![]i64 {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (self.arr == null) return AccelError.NullPointer;
        if (self.len == 0) return AccelError.InvalidDimensions;
        const buf = allocator.alloc(i64, self.len) catch return AccelError.AllocationFailed;
        errdefer allocator.free(buf);
        if (futhark.futhark_values_i64_1d(ctx.ctx, self.arr, buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        return buf;
    }

    pub fn free(self: *Self, ctx: *FutharkContext) void {
        if (self.arr) |arr| {
            _ = futhark.futhark_free_i64_1d(ctx.ctx, arr);
            self.arr = null;
            self.len = 0;
        }
    }
};

pub const EmbeddingAccelerator = struct {
    ctx: *FutharkContext,
    weight: FutharkArray2DF16,
    master_weight: FutharkArray2DF32,
    grad_weight: FutharkArray2DF32,
    vocab_size: usize,
    dim: usize,
    initialized: bool,
    allocator: std.mem.Allocator,
    optimizer_step: u64 = 0,
    scratch_token_buf: []i64 = &[_]i64{},
    scratch_token_cap: usize = 0,
    scratch_lengths_buf: []i64 = &[_]i64{},
    scratch_lengths_cap: usize = 0,
    scratch_positions_buf: []i64 = &[_]i64{},
    scratch_positions_cap: usize = 0,
    momentum_state: ?FutharkArray2DF32 = null,
    fisher_state: ?FutharkArray2DF32 = null,

    last_spectral_before: f32 = 0.0,
    last_spectral_after: f32 = 0.0,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, ctx: *FutharkContext, vocab_size: usize, dim: usize, seed: u64) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;

        var rng = std.Random.DefaultPrng.init(seed);
        const rnd = rng.random();
        const total = vocab_size * dim;
        const weight_data = allocator.alloc(f16, total) catch return AccelError.AllocationFailed;
        defer allocator.free(weight_data);
        for (weight_data) |*v| {
            v.* = @floatCast((rnd.float(f32) - 0.5) * 0.02);
        }

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_data, vocab_size, dim);
        errdefer weight.free(ctx);
        const master_data = allocator.alloc(f32, total) catch return AccelError.AllocationFailed;
        defer allocator.free(master_data);
        for (weight_data, master_data) |value, *master| master.* = @floatCast(value);
        var master_weight = try FutharkArray2DF32.newFromFlat(ctx, master_data, vocab_size, dim);
        errdefer master_weight.free(ctx);
        var grad_weight = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer grad_weight.free(ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const scratch_token_buf = allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_token_buf);
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);
        const scratch_positions_buf = allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = ctx,
            .weight = weight,
            .master_weight = master_weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .allocator = allocator,
            .scratch_token_buf = scratch_token_buf,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = scratch_positions_buf,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn deinit(self: *Self) void {
        if (!self.initialized) return;
        if (self.fisher_state) |*s| s.free(self.ctx);
        if (self.momentum_state) |*s| s.free(self.ctx);
        self.fisher_state = null;
        self.momentum_state = null;
        self.grad_weight.free(self.ctx);
        self.master_weight.free(self.ctx);
        self.weight.free(self.ctx);
        if (self.scratch_positions_buf.len > 0) self.allocator.free(self.scratch_positions_buf);
        if (self.scratch_lengths_buf.len > 0) self.allocator.free(self.scratch_lengths_buf);
        if (self.scratch_token_buf.len > 0) self.allocator.free(self.scratch_token_buf);
        self.initialized = false;
    }

    pub fn cloneDevice(self: *Self) AccelError!Self {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        const total_elements = try checkedElementCount2(self.vocab_size, self.dim);
        const flat = self.allocator.alloc(f16, total_elements) catch return AccelError.AllocationFailed;
        defer self.allocator.free(flat);
        const rc_values = futhark.futhark_values_f16_2d(self.ctx.ctx, self.weight.arr, @ptrCast(flat.ptr));
        if (rc_values != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        var weight_copy = FutharkArray2DF16.newFromFlat(self.ctx, flat, self.vocab_size, self.dim) catch return AccelError.FutharkArrayNewFailed;
        errdefer weight_copy.free(self.ctx);
        const master_flat = try self.master_weight.valuesFlat(self.ctx, self.allocator);
        defer self.allocator.free(master_flat);
        var master_copy = try FutharkArray2DF32.newFromFlat(self.ctx, master_flat, self.vocab_size, self.dim);
        errdefer master_copy.free(self.ctx);
        var grad_copy = FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator) catch return AccelError.FutharkArrayNewFailed;
        errdefer grad_copy.free(self.ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const st = self.allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer self.allocator.free(st);
        const sl = self.allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer self.allocator.free(sl);
        const sp = self.allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = self.ctx,
            .weight = weight_copy,
            .master_weight = master_copy,
            .grad_weight = grad_copy,
            .vocab_size = self.vocab_size,
            .dim = self.dim,
            .initialized = true,
            .allocator = self.allocator,
            .scratch_token_buf = st,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = sl,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = sp,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn initWithWeights(ctx: *FutharkContext, allocator: std.mem.Allocator, vocab_size: usize, dim: usize, weight_f16: []const f16) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        if (vocab_size == 0 or dim == 0) return AccelError.InvalidDimensions;
        if (weight_f16.len != vocab_size * dim) return AccelError.InvalidDimensions;

        var weight = try FutharkArray2DF16.newFromFlat(ctx, weight_f16, vocab_size, dim);
        errdefer weight.free(ctx);
        const master_data = allocator.alloc(f32, weight_f16.len) catch return AccelError.AllocationFailed;
        defer allocator.free(master_data);
        for (weight_f16, master_data) |value, *master| master.* = @floatCast(value);
        var master_weight = try FutharkArray2DF32.newFromFlat(ctx, master_data, vocab_size, dim);
        errdefer master_weight.free(ctx);
        var grad_weight = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer grad_weight.free(ctx);

        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const scratch_token_buf = allocator.alloc(i64, max_batch * max_seq) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_token_buf);
        const scratch_lengths_buf = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths_buf);
        const scratch_positions_buf = allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;

        return Self{
            .ctx = ctx,
            .weight = weight,
            .master_weight = master_weight,
            .grad_weight = grad_weight,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .allocator = allocator,
            .scratch_token_buf = scratch_token_buf,
            .scratch_token_cap = max_batch * max_seq,
            .scratch_lengths_buf = scratch_lengths_buf,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = scratch_positions_buf,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn initWithMasterWeights(ctx: *FutharkContext, allocator: std.mem.Allocator, vocab_size: usize, dim: usize, master_values: []const f32) AccelError!Self {
        if (ctx.ctx == null) return AccelError.NullPointer;
        const total = std.math.mul(usize, vocab_size, dim) catch return AccelError.InvalidDimensions;
        if (vocab_size == 0 or dim == 0 or master_values.len != total) return AccelError.InvalidDimensions;
        for (master_values) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        var master = try FutharkArray2DF32.newFromFlat(ctx, master_values, vocab_size, dim);
        errdefer master.free(ctx);
        var shadow_pointer: ?*futhark.struct_futhark_f16_2d = null;
        if (futhark.futhark_entry_master_weights_to_f16_2d(ctx.ctx, &shadow_pointer, master.arr) != 0 or shadow_pointer == null) return AccelError.FutharkArrayNewFailed;
        var shadow = FutharkArray2DF16{ .arr = shadow_pointer, .rows = vocab_size, .cols = dim };
        errdefer shadow.free(ctx);
        var gradient = try FutharkArray2DF32.newZeros(ctx, vocab_size, dim, allocator);
        errdefer gradient.free(ctx);
        const max_batch: usize = 2048;
        const max_seq: usize = 1024;
        const scratch_count = std.math.mul(usize, max_batch, max_seq) catch return AccelError.InvalidDimensions;
        const scratch_tokens = allocator.alloc(i64, scratch_count) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_tokens);
        const scratch_lengths = allocator.alloc(i64, max_batch) catch return AccelError.AllocationFailed;
        errdefer allocator.free(scratch_lengths);
        const scratch_positions = allocator.alloc(i64, max_seq) catch return AccelError.AllocationFailed;
        return .{
            .ctx = ctx,
            .weight = shadow,
            .master_weight = master,
            .grad_weight = gradient,
            .vocab_size = vocab_size,
            .dim = dim,
            .initialized = true,
            .allocator = allocator,
            .scratch_token_buf = scratch_tokens,
            .scratch_token_cap = scratch_count,
            .scratch_lengths_buf = scratch_lengths,
            .scratch_lengths_cap = max_batch,
            .scratch_positions_buf = scratch_positions,
            .scratch_positions_cap = max_seq,
        };
    }

    pub fn forwardPadded(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        sequence_length: usize,
    ) AccelError!FutharkArray3DF16 {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (sequence_lengths.len == 0 or sequence_length == 0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, sequence_lengths.len, sequence_length) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > sequence_length) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        const positions_i64 = if (sequence_length <= self.scratch_positions_cap) self.scratch_positions_buf[0..sequence_length] else (self.allocator.alloc(i64, sequence_length) catch return AccelError.AllocationFailed);
        defer if (sequence_length > self.scratch_positions_cap) self.allocator.free(positions_i64);
        for (positions_i64, 0..) |*position, index| {
            position.* = @intCast(index);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);
        var position_array = try FutharkArray1DI64.newFromSlice(self.ctx, positions_i64);
        defer position_array.free(self.ctx);

        var output: ?*futhark.struct_futhark_f16_3d = null;
        const result = futhark.futhark_entry_embedding_forward_padded(
            self.ctx.ctx,
            &output,
            token_array.arr,
            length_array.arr,
            position_array.arr,
            self.weight.arr,
        );
        if (result != 0 or output == null) {
            if (output) |value| _ = futhark.futhark_free_f16_3d(self.ctx.ctx, value);
            return AccelError.FutharkForwardFailed;
        }
        return FutharkArray3DF16{
            .arr = output,
            .dim0 = sequence_lengths.len,
            .dim1 = sequence_length,
            .dim2 = self.dim,
        };
    }

    pub fn backwardPaddedAccumulate(
        self: *Self,
        tokens: []const u32,
        sequence_lengths: []const usize,
        gradient_output: *FutharkArray3DF16,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (gradient_output.arr == null or gradient_output.dim2 != self.dim) return AccelError.InvalidDimensions;
        if (sequence_lengths.len != gradient_output.dim0) return AccelError.InvalidDimensions;
        const expected_tokens = std.math.mul(usize, gradient_output.dim0, gradient_output.dim1) catch return AccelError.InvalidDimensions;
        if (tokens.len != expected_tokens) return AccelError.InvalidDimensions;

        const token_i64s = if (tokens.len <= self.scratch_token_cap) self.scratch_token_buf[0..tokens.len] else (self.allocator.alloc(i64, tokens.len) catch return AccelError.AllocationFailed);
        defer if (tokens.len > self.scratch_token_cap) self.allocator.free(token_i64s);
        for (tokens, 0..) |token, index| {
            if (@as(usize, token) >= self.vocab_size) return AccelError.InvalidDimensions;
            token_i64s[index] = @intCast(token);
        }

        const lengths_i64 = if (sequence_lengths.len <= self.scratch_lengths_cap) self.scratch_lengths_buf[0..sequence_lengths.len] else (self.allocator.alloc(i64, sequence_lengths.len) catch return AccelError.AllocationFailed);
        defer if (sequence_lengths.len > self.scratch_lengths_cap) self.allocator.free(lengths_i64);
        for (sequence_lengths, 0..) |length, index| {
            if (length > gradient_output.dim1) return AccelError.InvalidDimensions;
            lengths_i64[index] = @intCast(length);
        }

        var token_array = try FutharkArray1DI64.newFromSlice(self.ctx, token_i64s);
        defer token_array.free(self.ctx);
        var length_array = try FutharkArray1DI64.newFromSlice(self.ctx, lengths_i64);
        defer length_array.free(self.ctx);

        var new_gradient: ?*futhark.struct_futhark_f32_2d = null;
        const result = futhark.futhark_entry_embedding_backward_padded(
            self.ctx.ctx,
            &new_gradient,
            token_array.arr,
            length_array.arr,
            gradient_output.arr,
            self.grad_weight.arr,
        );
        if (result != 0 or new_gradient == null) {
            if (new_gradient) |value| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, value);
            return AccelError.FutharkBackwardFailed;
        }
        const old_gradient = self.grad_weight.arr;
        self.grad_weight.arr = new_gradient;
        _ = futhark.futhark_free_f32_2d(self.ctx.ctx, old_gradient);
    }

    pub fn getGradientDevicePtrF32(self: *Self) AccelError!DeviceBufferF32 {
        if (!self.initialized or self.grad_weight.arr == null) return AccelError.NullPointer;
        return self.grad_weight.deviceBuffer(self.ctx);
    }

    pub fn clipGradient(self: *Self, clip_norm: f32) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null or self.grad_weight.arr == null) return AccelError.NullPointer;
        if (!std.math.isFinite(clip_norm) or clip_norm <= 0.0) return AccelError.InvalidHyperparameter;
        var clipped: ?*futhark.struct_futhark_f32_2d = null;
        const result = futhark.futhark_entry_clip_matrix_global_norm_f32(
            self.ctx.ctx,
            &clipped,
            self.grad_weight.arr,
            clip_norm,
        );
        if (result != 0 or clipped == null) {
            if (clipped) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            return AccelError.FutharkScaleWeightsFailed;
        }
        self.grad_weight.free(self.ctx);
        self.grad_weight = .{ .arr = clipped, .rows = self.vocab_size, .cols = self.dim };
    }

    pub fn scaleGradient(self: *Self, scale_factor: f32) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null or self.grad_weight.arr == null) return AccelError.NullPointer;
        if (!std.math.isFinite(scale_factor) or scale_factor < 0.0) return AccelError.InvalidHyperparameter;
        var scaled: ?*futhark.struct_futhark_f32_2d = null;
        const result = futhark.futhark_entry_scale_matrix_f32(
            self.ctx.ctx,
            &scaled,
            self.grad_weight.arr,
            scale_factor,
        );
        if (result != 0 or scaled == null) {
            if (scaled) |value| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, value);
            return AccelError.FutharkScaleWeightsFailed;
        }
        self.grad_weight.free(self.ctx);
        self.grad_weight = .{ .arr = scaled, .rows = self.vocab_size, .cols = self.dim };
    }

    pub fn ensureFisherState(self: *Self) AccelError!void {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.momentum_state != null and self.fisher_state != null) return;
        if (self.momentum_state == null) {
            self.momentum_state = try FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator);
        }
        if (self.fisher_state == null) {
            self.fisher_state = try FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator);
        }
    }

    pub fn applyUpdateFusedSFD(
        self: *Self,
        learning_rate: f32,
        momentum_beta: f32,
        fisher_gamma: f32,
        epsilon: f32,
        trust_ratio: f32,
        weight_floor: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (!std.math.isFinite(learning_rate) or learning_rate < 0.0) return AccelError.InvalidHyperparameter;
        if (!std.math.isFinite(momentum_beta) or momentum_beta < 0.0 or momentum_beta >= 1.0 or
            !std.math.isFinite(fisher_gamma) or fisher_gamma < 0.0 or fisher_gamma >= 1.0 or
            !std.math.isFinite(epsilon) or epsilon <= 0.0 or
            !std.math.isFinite(trust_ratio) or trust_ratio <= 0.0 or trust_ratio > 1.0 or
            !std.math.isFinite(weight_floor) or weight_floor <= 0.0) return AccelError.InvalidHyperparameter;
        try self.ensureFisherState();
        const ms = &(self.momentum_state orelse return AccelError.NullPointer);
        const fs = &(self.fisher_state orelse return AccelError.NullPointer);

        var tuple: ?*futhark.struct_futhark_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32 = null;
        const rc = futhark.futhark_entry_embedding_update_sfd_master(
            self.ctx.ctx,
            &tuple,
            self.master_weight.arr,
            self.grad_weight.arr,
            ms.arr,
            fs.arr,
            learning_rate,
            momentum_beta,
            fisher_gamma,
            @intCast(@min(self.optimizer_step +| 1, @as(u64, std.math.maxInt(i64)))),
            epsilon,
            trust_ratio,
            weight_floor,
        );
        if (rc != 0 or tuple == null) return AccelError.FutharkSFDUpdateFailed;
        var new_master: ?*futhark.struct_futhark_f32_2d = null;
        var new_momentum: ?*futhark.struct_futhark_f32_2d = null;
        var new_fisher: ?*futhark.struct_futhark_f32_2d = null;
        const p0 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_0(self.ctx.ctx, &new_master, tuple);
        const p1 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_1(self.ctx.ctx, &new_momentum, tuple);
        const p2 = futhark.futhark_project_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32_2(self.ctx.ctx, &new_fisher, tuple);
        _ = futhark.futhark_free_opaque_tup3_arr2d_f32_arr2d_f32_arr2d_f32(self.ctx.ctx, tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or new_master == null or new_momentum == null or new_fisher == null) {
            if (new_master) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            if (new_momentum) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            if (new_fisher) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            return AccelError.FutharkSFDUpdateFailed;
        }
        var new_forward: ?*futhark.struct_futhark_f16_2d = null;
        if (futhark.futhark_entry_master_weights_to_f16_2d(self.ctx.ctx, &new_forward, new_master) != 0 or new_forward == null) {
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_master);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_momentum);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_fisher);
            if (new_forward) |array| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, array);
            return AccelError.FutharkSFDUpdateFailed;
        }

        const zeroed_grad = FutharkArray2DF32.newZeros(self.ctx, self.vocab_size, self.dim, self.allocator) catch |err| {
            _ = futhark.futhark_free_f16_2d(self.ctx.ctx, new_forward);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_master);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_momentum);
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_fisher);
            return err;
        };
        self.weight.free(self.ctx);
        self.master_weight.free(self.ctx);
        ms.free(self.ctx);
        fs.free(self.ctx);
        self.weight.arr = new_forward;
        self.weight.rows = self.vocab_size;
        self.weight.cols = self.dim;
        self.master_weight.arr = new_master;
        self.master_weight.rows = self.vocab_size;
        self.master_weight.cols = self.dim;
        ms.* = .{ .arr = new_momentum, .rows = self.vocab_size, .cols = self.dim };
        fs.* = .{ .arr = new_fisher, .rows = self.vocab_size, .cols = self.dim };
        self.optimizer_step +|= 1;

        self.grad_weight.free(self.ctx);
        self.grad_weight = zeroed_grad;
    }

    pub fn readOptimizerState(self: *Self, allocator: std.mem.Allocator) AccelError!EmbeddingOptimizerState {
        try self.ensureFisherState();
        const master = try self.master_weight.valuesFlat(self.ctx, allocator);
        errdefer allocator.free(master);
        const momentum = try self.momentum_state.?.valuesFlat(self.ctx, allocator);
        errdefer allocator.free(momentum);
        const fisher = try self.fisher_state.?.valuesFlat(self.ctx, allocator);
        return .{ .master_weights = master, .momentum = momentum, .fisher = fisher, .step = self.optimizer_step, .allocator = allocator };
    }

    pub fn setOptimizerState(self: *Self, master_weights: []const f32, momentum: []const f32, fisher: []const f32, step: u64) AccelError!void {
        const total = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        if (master_weights.len != total or momentum.len != total or fisher.len != total) return AccelError.InvalidDimensions;
        for (master_weights) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (momentum) |value| if (!std.math.isFinite(value)) return AccelError.InvalidHyperparameter;
        for (fisher) |value| if (!std.math.isFinite(value) or value < 0.0) return AccelError.InvalidHyperparameter;
        var new_master = try FutharkArray2DF32.newFromFlat(self.ctx, master_weights, self.vocab_size, self.dim);
        errdefer new_master.free(self.ctx);
        var new_m = try FutharkArray2DF32.newFromFlat(self.ctx, momentum, self.vocab_size, self.dim);
        errdefer new_m.free(self.ctx);
        var new_f = try FutharkArray2DF32.newFromFlat(self.ctx, fisher, self.vocab_size, self.dim);
        errdefer new_f.free(self.ctx);
        var new_forward_array: ?*futhark.struct_futhark_f16_2d = null;
        if (futhark.futhark_entry_master_weights_to_f16_2d(self.ctx.ctx, &new_forward_array, new_master.arr) != 0 or new_forward_array == null) return AccelError.FutharkScaleWeightsFailed;
        var new_forward = FutharkArray2DF16{ .arr = new_forward_array, .rows = self.vocab_size, .cols = self.dim };
        errdefer new_forward.free(self.ctx);
        self.weight.free(self.ctx);
        self.master_weight.free(self.ctx);
        if (self.momentum_state) |*old| old.free(self.ctx);
        if (self.fisher_state) |*old| old.free(self.ctx);
        self.weight = new_forward;
        self.master_weight = new_master;
        self.momentum_state = new_m;
        self.fisher_state = new_f;
        self.optimizer_step = step;
    }

    pub fn sourceSumSquares(self: *Self) AccelError!f32 {
        if (!self.initialized) return AccelError.NullPointer;
        if (self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.weight.arr == null) return AccelError.NullPointer;
        var total: f32 = 0.0;
        const rc = futhark.futhark_entry_embedding_sum_squares(
            self.ctx.ctx,
            &total,
            self.weight.arr,
        );
        if (rc != 0) return AccelError.FutharkComputeLossFailed;
        if (futhark.futhark_context_sync(self.ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        if (!std.math.isFinite(total)) return 0.0;
        return total;
    }

    pub fn sourceRootMeanSquare(self: *Self) AccelError!f32 {
        const total = try self.sourceSumSquares();
        const count = std.math.mul(usize, self.vocab_size, self.dim) catch return AccelError.InvalidDimensions;
        if (count == 0) return 0.0;
        const mean = total / @as(f32, @floatFromInt(count));
        if (!std.math.isFinite(mean) or mean <= 0.0) return 0.0;
        return @sqrt(mean);
    }

    pub fn spectralNormalize(
        self: *Self,
        u: *FutharkArray1DF32,
        v: *FutharkArray1DF32,
        power_iters: usize,
        target: f32,
    ) AccelError!void {
        if (!self.initialized or self.ctx.ctx == null) return AccelError.NullPointer;
        if (self.master_weight.arr == null or u.arr == null or v.arr == null) return AccelError.NullPointer;
        if (power_iters == 0 or !std.math.isFinite(target) or target <= 0.0) return AccelError.InvalidHyperparameter;
        var tuple: ?*futhark.struct_futhark_opaque_tup5_embedding_spectral = null;
        const rc = futhark.futhark_entry_embedding_spectral_normalize(
            self.ctx.ctx,
            &tuple,
            self.master_weight.arr,
            u.arr,
            v.arr,
            try checkedDimensionI64(power_iters),
            target,
        );
        if (rc != 0 or tuple == null) return AccelError.FutharkForwardFailed;
        var new_master: ?*futhark.struct_futhark_f32_2d = null;
        var new_u: ?*futhark.struct_futhark_f32_1d = null;
        var new_v: ?*futhark.struct_futhark_f32_1d = null;
        var sigma_before: f32 = 0.0;
        var sigma_after: f32 = 0.0;
        const p0 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_0(self.ctx.ctx, &new_master, tuple);
        const p1 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_1(self.ctx.ctx, &new_u, tuple);
        const p2 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_2(self.ctx.ctx, &new_v, tuple);
        const p3 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_3(self.ctx.ctx, &sigma_before, tuple);
        const p4 = futhark.futhark_project_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32_4(self.ctx.ctx, &sigma_after, tuple);
        _ = futhark.futhark_free_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32(self.ctx.ctx, tuple);
        if (p0 != 0 or p1 != 0 or p2 != 0 or p3 != 0 or p4 != 0 or new_master == null or new_u == null or new_v == null or !std.math.isFinite(sigma_before) or !std.math.isFinite(sigma_after)) {
            if (new_master) |array| _ = futhark.futhark_free_f32_2d(self.ctx.ctx, array);
            if (new_u) |array| _ = futhark.futhark_free_f32_1d(self.ctx.ctx, array);
            if (new_v) |array| _ = futhark.futhark_free_f32_1d(self.ctx.ctx, array);
            return AccelError.FutharkForwardFailed;
        }
        var new_shadow: ?*futhark.struct_futhark_f16_2d = null;
        if (futhark.futhark_entry_master_weights_to_f16_2d(self.ctx.ctx, &new_shadow, new_master) != 0 or new_shadow == null) {
            _ = futhark.futhark_free_f32_2d(self.ctx.ctx, new_master);
            _ = futhark.futhark_free_f32_1d(self.ctx.ctx, new_u);
            _ = futhark.futhark_free_f32_1d(self.ctx.ctx, new_v);
            if (new_shadow) |array| _ = futhark.futhark_free_f16_2d(self.ctx.ctx, array);
            return AccelError.FutharkForwardFailed;
        }
        self.master_weight.free(self.ctx);
        self.weight.free(self.ctx);
        u.free(self.ctx);
        v.free(self.ctx);
        self.master_weight = .{ .arr = new_master, .rows = self.vocab_size, .cols = self.dim };
        self.weight = .{ .arr = new_shadow, .rows = self.vocab_size, .cols = self.dim };
        u.* = .{ .arr = new_u, .len = self.vocab_size };
        v.* = .{ .arr = new_v, .len = self.dim };
        self.last_spectral_before = sigma_before;
        self.last_spectral_after = sigma_after;
    }
};

pub const GraphBatchEncodeResult = struct {
    hashes: []u64,
    re_a: []f32,
    im_a: []f32,
    re_b: []f32,
    im_b: []f32,
    edge_srcs: []i64,
    edge_tgts: []i64,
    node_count: usize,
    edge_count: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GraphBatchEncodeResult) void {
        self.allocator.free(self.hashes);
        self.allocator.free(self.re_a);
        self.allocator.free(self.im_a);
        self.allocator.free(self.re_b);
        self.allocator.free(self.im_b);
        self.allocator.free(self.edge_srcs);
        self.allocator.free(self.edge_tgts);
    }
};

pub fn batchEncodeGraph(
    ctx: *FutharkContext,
    hashes: []const u64,
    seed: u64,
    allocator: std.mem.Allocator,
) AccelError!GraphBatchEncodeResult {
    if (ctx.ctx == null) return AccelError.NullPointer;
    ctx.mutex.lock();
    defer ctx.mutex.unlock();
    if (hashes.len == 0) return AccelError.InvalidDimensions;

    var acc_hashes = std.ArrayList(u64).init(allocator);
    errdefer acc_hashes.deinit();
    var acc_re_a = std.ArrayList(f32).init(allocator);
    errdefer acc_re_a.deinit();
    var acc_im_a = std.ArrayList(f32).init(allocator);
    errdefer acc_im_a.deinit();
    var acc_re_b = std.ArrayList(f32).init(allocator);
    errdefer acc_re_b.deinit();
    var acc_im_b = std.ArrayList(f32).init(allocator);
    errdefer acc_im_b.deinit();
    var acc_edge_srcs = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_srcs.deinit();
    var acc_edge_tgts = std.ArrayList(i64).init(allocator);
    errdefer acc_edge_tgts.deinit();

    acc_hashes.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_a.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_re_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    acc_im_b.ensureTotalCapacity(hashes.len) catch return AccelError.AllocationFailed;
    const edge_capacity = std.math.mul(usize, hashes.len, 3) catch return AccelError.InvalidDimensions;
    acc_edge_srcs.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;
    acc_edge_tgts.ensureTotalCapacity(edge_capacity) catch return AccelError.AllocationFailed;

    var offset: usize = 0;
    while (offset < hashes.len) {
        const chunk_end = hashes.len;
        const chunk = hashes[offset..chunk_end];
        const chunk_n = chunk.len;
        const chunk_ne = std.math.mul(usize, chunk_n, 3) catch return AccelError.InvalidDimensions;

        var in_chunk = FutharkArray1DU64.newFromSlice(ctx, chunk) catch |err| {
            std.debug.print("[batchEncodeGraph] chunk offset={d} upload failed: {}\n", .{ offset, err });
            return err;
        };
        defer in_chunk.free(ctx);

        var out_tup: ?*futhark.struct_futhark_opaque_tup7_graph_encode = null;
        defer {
            if (out_tup) |p| {
                _ = futhark.futhark_free_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64(ctx.ctx, p);
            }
        }

        var out_ids: ?*futhark.struct_futhark_u64_1d = null;
        var out_re_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_a: ?*futhark.struct_futhark_f32_1d = null;
        var out_re_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_im_b: ?*futhark.struct_futhark_f32_1d = null;
        var out_edge_srcs: ?*futhark.struct_futhark_i64_1d = null;
        var out_edge_tgts: ?*futhark.struct_futhark_i64_1d = null;

        defer {
            if (out_ids) |p| _ = futhark.futhark_free_u64_1d(ctx.ctx, p);
            if (out_re_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_a) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_re_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_im_b) |p| _ = futhark.futhark_free_f32_1d(ctx.ctx, p);
            if (out_edge_srcs) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
            if (out_edge_tgts) |p| _ = futhark.futhark_free_i64_1d(ctx.ctx, p);
        }

        const rc = futhark.futhark_entry_graph_batch_encode(
            ctx.ctx,
            &out_tup,
            in_chunk.arr,
            seed,
        );

        if (rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            defer freeFutharkError(ctx_err);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark entry error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] Futhark entry failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, rc });
            }
            return AccelError.FutharkForwardFailed;
        }

        const sync_rc = futhark.futhark_context_sync(ctx.ctx);
        if (sync_rc != 0) {
            const ctx_err = futhark.futhark_context_get_error(ctx.ctx);
            defer freeFutharkError(ctx_err);
            if (ctx_err) |msg| {
                std.debug.print("[batchEncodeGraph] Futhark sync error at offset={d} n={d}: {s}\n", .{ offset, chunk_n, std.mem.span(msg) });
            } else {
                std.debug.print("[batchEncodeGraph] futhark_context_sync failed at offset={d} n={d}: rc={d}\n", .{ offset, chunk_n, sync_rc });
            }
            return AccelError.FutharkSyncFailed;
        }

        const tup = out_tup orelse {
            std.debug.print("[batchEncodeGraph] out_tup null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        };
        const proj0 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_0(ctx.ctx, &out_ids, tup);
        const proj1 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_1(ctx.ctx, &out_re_a, tup);
        const proj2 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_2(ctx.ctx, &out_im_a, tup);
        const proj3 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_3(ctx.ctx, &out_re_b, tup);
        const proj4 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_4(ctx.ctx, &out_im_b, tup);
        const proj5 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_5(ctx.ctx, &out_edge_srcs, tup);
        const proj6 = futhark.futhark_project_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64_6(ctx.ctx, &out_edge_tgts, tup);
        if (proj0 != 0 or proj1 != 0 or proj2 != 0 or proj3 != 0 or proj4 != 0 or proj5 != 0 or proj6 != 0) {
            std.debug.print("[batchEncodeGraph] projection failed at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.FutharkForwardFailed;
        }

        if (out_ids == null) {
            std.debug.print("[batchEncodeGraph] out_ids null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_a == null) {
            std.debug.print("[batchEncodeGraph] out_re_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_a == null) {
            std.debug.print("[batchEncodeGraph] out_im_a null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_re_b == null) {
            std.debug.print("[batchEncodeGraph] out_re_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_im_b == null) {
            std.debug.print("[batchEncodeGraph] out_im_b null at offset={d} n={d}\n", .{ offset, chunk_n });
            return AccelError.NullPointer;
        }
        if (out_edge_srcs == null) {
            std.debug.print("[batchEncodeGraph] out_edge_srcs null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }
        if (out_edge_tgts == null) {
            std.debug.print("[batchEncodeGraph] out_edge_tgts null at offset={d} ne={d}\n", .{ offset, chunk_ne });
            return AccelError.NullPointer;
        }

        const ids_buf = allocator.alloc(u64, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(ids_buf);
        const re_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_a_buf);
        const im_a_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_a_buf);
        const re_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(re_b_buf);
        const im_b_buf = allocator.alloc(f32, chunk_n) catch return AccelError.AllocationFailed;
        defer allocator.free(im_b_buf);
        const edge_src_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_src_buf);
        const edge_tgt_buf = allocator.alloc(i64, chunk_ne) catch return AccelError.AllocationFailed;
        defer allocator.free(edge_tgt_buf);

        if (futhark.futhark_values_u64_1d(ctx.ctx, out_ids, ids_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_a, re_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_a, im_a_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_re_b, re_b_buf.ptr) != 0 or
            futhark.futhark_values_f32_1d(ctx.ctx, out_im_b, im_b_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_srcs, edge_src_buf.ptr) != 0 or
            futhark.futhark_values_i64_1d(ctx.ctx, out_edge_tgts, edge_tgt_buf.ptr) != 0) return AccelError.FutharkValuesFailed;
        if (futhark.futhark_context_sync(ctx.ctx) != 0) return AccelError.FutharkSyncFailed;
        acc_hashes.appendSlice(ids_buf) catch return AccelError.AllocationFailed;
        acc_re_a.appendSlice(re_a_buf) catch return AccelError.AllocationFailed;
        acc_im_a.appendSlice(im_a_buf) catch return AccelError.AllocationFailed;
        acc_re_b.appendSlice(re_b_buf) catch return AccelError.AllocationFailed;
        acc_im_b.appendSlice(im_b_buf) catch return AccelError.AllocationFailed;
        for (edge_src_buf) |value| acc_edge_srcs.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;
        for (edge_tgt_buf) |value| acc_edge_tgts.append(if (value >= 0) value + @as(i64, @intCast(offset)) else value) catch return AccelError.AllocationFailed;

        offset = chunk_end;
    }

    const total_n = acc_hashes.items.len;
    const total_ne = acc_edge_srcs.items.len;

    const owned_hashes = acc_hashes.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_hashes);
    const owned_re_a = acc_re_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_a);
    const owned_im_a = acc_im_a.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_a);
    const owned_re_b = acc_re_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_re_b);
    const owned_im_b = acc_im_b.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_im_b);
    const owned_edge_srcs = acc_edge_srcs.toOwnedSlice() catch return AccelError.AllocationFailed;
    errdefer allocator.free(owned_edge_srcs);
    const owned_edge_tgts = acc_edge_tgts.toOwnedSlice() catch return AccelError.AllocationFailed;

    return GraphBatchEncodeResult{
        .hashes = owned_hashes,
        .re_a = owned_re_a,
        .im_a = owned_im_a,
        .re_b = owned_re_b,
        .im_b = owned_im_b,
        .edge_srcs = owned_edge_srcs,
        .edge_tgts = owned_edge_tgts,
        .node_count = total_n,
        .edge_count = total_ne,
        .allocator = allocator,
    };
}
