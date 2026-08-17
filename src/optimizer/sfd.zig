const std = @import("std");
const Allocator = std.mem.Allocator;
const core_types = @import("../core/types.zig");
const core_tensor = @import("../core/tensor.zig");
const core_memory = @import("../core/memory.zig");
const core_io = @import("../core/io.zig");
const PRNG = core_types.PRNG;

var global_prng_counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);

fn nextSeed() u64 {
    var prng = PRNG.init(global_prng_counter.fetchAdd(1, .monotonic));
    return prng.uint64();
}

fn shapesEqual(a: Shape, b: Shape) bool {
    if (a.dims.len != b.dims.len) return false;
    var i: usize = 0;
    while (i < a.dims.len) : (i += 1) {
        if (a.dims[i] != b.dims[i]) return false;
    }
    return true;
}

fn quantizeValue(value: f32, precision: Precision) f32 {
    if (!std.math.isFinite(value)) return value;
    return switch (precision) {
        .fp16 => blk: {
            const clamped = std.math.clamp(value, -65504.0, 65504.0);
            const rounded: f16 = @floatCast(clamped);
            break :blk @floatCast(rounded);
        },
        .fp32, .fp64 => value,
    };
}

fn tensorFlagsToBits(flags: TensorFlags) u8 {
    var bits: u8 = 0;
    if (flags.in_tensor_memory) bits |= 0b001;
    if (flags.requires_grad) bits |= 0b010;
    return bits;
}

fn tensorFlagsFromBits(bits: u8) TensorFlags {
    return TensorFlags{
        .in_tensor_memory = (bits & 0b001) != 0,
        .requires_grad = (bits & 0b010) != 0,
    };
}

fn erfApprox(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;
    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = if (x < 0) -x else x;
    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x);
    return sign * y;
}

pub const Precision = enum {
    fp16,
    fp32,
    fp64,
};

const Shape = struct {
    dims: []const usize,

    pub fn totalSize(self: Shape) usize {
        var size: usize = 1;
        for (self.dims) |dim| {
            if (dim != 0 and size > std.math.maxInt(usize) / dim) @panic("Shape.totalSize overflow");
            size *= dim;
        }
        return size;
    }
};

pub const TensorFlags = struct {
    in_tensor_memory: bool = false,
    requires_grad: bool = true,
};

pub const Tensor = struct {
    data: []f32,
    shape: Shape,
    dtype: Precision = .fp32,
    flags: TensorFlags = .{},
    allocator: Allocator,

    pub fn init(allocator: Allocator, dims: []const usize) !Tensor {
        for (dims) |d| {
            if (d == 0) return error.InvalidShape;
        }
        const owned_dims = try allocator.dupe(usize, dims);
        errdefer allocator.free(owned_dims);

        const shape = Shape{ .dims = owned_dims };
        const size = shape.totalSize();

        const data = try allocator.alloc(f32, size);
        errdefer allocator.free(data);

        return Tensor{
            .data = data,
            .shape = shape,
            .allocator = allocator,
        };
    }

    pub fn zeros(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        tensor.fill(0.0);
        return tensor;
    }

    pub fn ones(allocator: Allocator, dims: []const usize) !Tensor {
        var tensor = try init(allocator, dims);
        tensor.fill(1.0);
        return tensor;
    }

    pub fn eye(allocator: Allocator, dims: []const usize) !Tensor {
        if (dims.len != 2 or dims[0] != dims[1]) return error.InvalidShape;
        var tensor = try init(allocator, dims);
        tensor.fill(0.0);
        const n = dims[0];
        var i: usize = 0;
        while (i < n) : (i += 1) {
            tensor.data[i * n + i] = 1.0;
        }
        return tensor;
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape.dims);
    }

    pub fn fill(self: *Tensor, value: f32) void {
        for (self.data) |*v| {
            v.* = value;
        }
    }

    pub fn fillRandomNormal(self: *Tensor, mean: f32, std_dev: f32) void {
        var prng = std.Random.DefaultPrng.init(nextSeed());
        const random = prng.random();

        var i: usize = 0;
        while (i + 1 < self.data.len) : (i += 2) {
            const rand_u = @max(random.float(f32), 1e-7);
            const rand_v = random.float(f32);
            const r = @sqrt(-2.0 * @log(rand_u));
            const theta = 2.0 * std.math.pi * rand_v;
            self.data[i] = mean + std_dev * r * @cos(theta);
            self.data[i + 1] = mean + std_dev * r * @sin(theta);
        }
        if (i < self.data.len) {
            const rand_u = @max(random.float(f32), 1e-7);
            const rand_v = random.float(f32);
            const z0 = @sqrt(-2.0 * @log(rand_u)) * @cos(2.0 * std.math.pi * rand_v);
            self.data[i] = mean + std_dev * z0;
        }
    }

    pub fn fillRademacher(self: *Tensor) void {
        var prng = std.Random.DefaultPrng.init(nextSeed());
        const random = prng.random();

        for (self.data) |*v| {
            v.* = if (random.float(f32) < 0.5) -1.0 else 1.0;
        }
    }

    pub fn clone(self: *const Tensor, allocator: Allocator) !Tensor {
        var new_tensor = try Tensor.init(allocator, self.shape.dims);
        @memcpy(new_tensor.data, self.data);
        new_tensor.dtype = self.dtype;
        new_tensor.flags = self.flags;
        new_tensor.flags.in_tensor_memory = false;
        return new_tensor;
    }

    pub fn copyFrom(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        @memcpy(self.data, other.data);
        self.flags.requires_grad = other.flags.requires_grad;
        self.dtype = other.dtype;
    }

    pub fn copyFromWithCast(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] = quantizeValue(other.data[i], self.dtype);
        }
        self.flags.requires_grad = other.flags.requires_grad;
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) void {
        for (self.data) |*v| {
            v.* *= scalar;
        }
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        if (!shapesEqual(self.shape, other.shape)) return error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum: f64 = 0.0;
        for (self.data) |v| {
            if (std.math.isNan(v)) return std.math.nan(f32);
            if (!std.math.isFinite(v)) return std.math.inf(f32);
            sum += @as(f64, v) * @as(f64, v);
        }
        return @floatCast(@sqrt(sum));
    }

    pub fn spectralNorm(self: *const Tensor, allocator: Allocator, max_iter: usize, eps: f32) !f32 {
        if (self.shape.dims.len != 2) return error.InvalidShape;

        const m = self.shape.dims[0];
        const n = self.shape.dims[1];

        var v = try Tensor.init(allocator, &[_]usize{n});
        defer v.deinit();
        v.fillRandomNormal(0.0, 1.0);

        var u = try Tensor.init(allocator, &[_]usize{m});
        defer u.deinit();
        u.fill(0.0);

        const effective_iter = if (max_iter == 0) @as(usize, 1) else max_iter;
        var iter: usize = 0;
        while (iter < effective_iter) : (iter += 1) {
            u.fill(0.0);
            var i: usize = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    u.data[i] += self.data[i * n + j] * v.data[j];
                }
            }

            const u_norm = u.normL2();
            if (std.math.isFinite(u_norm) and u_norm > eps) {
                u.mulScalar(1.0 / u_norm);
            }

            v.fill(0.0);
            i = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    v.data[j] += self.data[i * n + j] * u.data[i];
                }
            }

            const v_norm = v.normL2();
            if (std.math.isFinite(v_norm) and v_norm > eps) {
                v.mulScalar(1.0 / v_norm);
            }
        }

        var sigma: f64 = 0.0;
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                sigma += @as(f64, u.data[i]) * @as(f64, self.data[i * n + j]) * @as(f64, v.data[j]);
            }
        }

        const sigma_f32: f32 = @floatCast(sigma);
        return if (sigma_f32 >= 0) sigma_f32 else -sigma_f32;
    }

    pub fn matmul(self: *Tensor, A: *const Tensor, B: *const Tensor) !void {
        if (A.shape.dims.len != 2 or B.shape.dims.len != 2 or self.shape.dims.len != 2) return error.InvalidShape;

        const m = A.shape.dims[0];
        const k = A.shape.dims[1];
        const n = B.shape.dims[1];

        if (B.shape.dims[0] != k) return error.ShapeMismatch;
        if (self.shape.dims[0] != m or self.shape.dims[1] != n) return error.ShapeMismatch;
        if (self.data.len != m * n) return error.ShapeMismatch;

        if (self.data.ptr == A.data.ptr or self.data.ptr == B.data.ptr) {
            var temp = try Tensor.init(self.allocator, &[_]usize{ m, n });
            defer temp.deinit();
            try temp.matmul(A, B);
            try self.copyFrom(&temp);
            return;
        }

        self.fill(0.0);

        const b_transposed = try self.allocator.alignedAlloc(f32, @as(?u29, 32), n * k);
        defer self.allocator.free(b_transposed);
        var pi: usize = 0;
        while (pi < k) : (pi += 1) {
            var ji: usize = 0;
            while (ji < n) : (ji += 1) {
                b_transposed[ji * k + pi] = B.data[pi * n + ji];
            }
        }

        const SfdMatmulVecWidth = 8;
        const SfdMatmulVec8 = @Vector(SfdMatmulVecWidth, f32);
        const SfdMatmulBlock: usize = 32;

        const Worker = struct {
            fn run(a_ptr: []const f32, bt_ptr: []const f32, out_ptr: []f32, start: usize, end: usize, k_dim: usize, n_dim: usize) void {
                var ii: usize = start;
                while (ii < end) : (ii += SfdMatmulBlock) {
                    const i_end = @min(ii + SfdMatmulBlock, end);
                    var jj: usize = 0;
                    while (jj < n_dim) : (jj += SfdMatmulBlock) {
                        const j_end = @min(jj + SfdMatmulBlock, n_dim);
                        var i: usize = ii;
                        while (i < i_end) : (i += 1) {
                            var j: usize = jj;
                            while (j < j_end) : (j += 1) {
                                var accumulator: SfdMatmulVec8 = @splat(0.0);
                                var kk: usize = 0;
                                const limit = k_dim - k_dim % SfdMatmulVecWidth;
                                while (kk < limit) : (kk += SfdMatmulVecWidth) {
                                    const av: SfdMatmulVec8 = a_ptr[i * k_dim + kk ..][0..SfdMatmulVecWidth].*;
                                    const bv: SfdMatmulVec8 = bt_ptr[j * k_dim + kk ..][0..SfdMatmulVecWidth].*;
                                    accumulator += av * bv;
                                }
                                var sum_value: f32 = @reduce(.Add, accumulator);
                                while (kk < k_dim) : (kk += 1) {
                                    sum_value += a_ptr[i * k_dim + kk] * bt_ptr[j * k_dim + kk];
                                }
                                out_ptr[i * n_dim + j] = sum_value;
                            }
                        }
                    }
                }
            }
        };

        const thread_count = @min(core_tensor.effectiveCpuCount(), @min(m, 8));
        if (thread_count <= 1) {
            Worker.run(A.data, b_transposed, self.data, 0, m, k, n);
        } else {
            var threads: [8]std.Thread = undefined;
            var active: usize = 0;
            const chunk = (m + thread_count - 1) / thread_count;
            var start: usize = 0;
            while (start < m and active < thread_count) : (start += chunk) {
                const end = @min(start + chunk, m);
                threads[active] = std.Thread.spawn(.{}, Worker.run, .{ A.data, b_transposed, self.data, start, end, k, n }) catch |err| {
                    for (threads[0..active]) |thread| thread.join();
                    return err;
                };
                active += 1;
            }
            for (threads[0..active]) |thread| thread.join();
        }
    }

    pub fn cholesky(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.shape.dims.len != 2 or self.shape.dims[0] != self.shape.dims[1]) return error.InvalidShape;
        const n = self.shape.dims[0];
        var result = try Tensor.zeros(allocator, &.{ n, n });
        errdefer result.deinit();

        var diagonal_scale: f64 = 1.0;
        var diagonal_index: usize = 0;
        while (diagonal_index < n) : (diagonal_index += 1) {
            const diagonal_value = self.data[diagonal_index * n + diagonal_index];
            if (!std.math.isFinite(diagonal_value)) return error.MatrixNotPositiveDefinite;
            diagonal_scale = @max(diagonal_scale, @abs(@as(f64, diagonal_value)));
        }
        const positivity_threshold = diagonal_scale * 1e-12;
        const symmetry_threshold = diagonal_scale * 1e-5;

        var row: usize = 0;
        while (row < n) : (row += 1) {
            var column: usize = 0;
            while (column <= row) : (column += 1) {
                const lower_value = self.data[row * n + column];
                const upper_value = self.data[column * n + row];
                if (!std.math.isFinite(lower_value) or !std.math.isFinite(upper_value)) return error.MatrixNotPositiveDefinite;
                if (@abs(@as(f64, lower_value) - @as(f64, upper_value)) > symmetry_threshold) return error.MatrixNotPositiveDefinite;
                var sum = (@as(f64, lower_value) + @as(f64, upper_value)) * 0.5;
                var inner: usize = 0;
                while (inner < column) : (inner += 1) {
                    sum -= @as(f64, result.data[row * n + inner]) * @as(f64, result.data[column * n + inner]);
                }
                if (row == column) {
                    if (!std.math.isFinite(sum) or sum <= positivity_threshold) return error.MatrixNotPositiveDefinite;
                    result.data[row * n + column] = @floatCast(@sqrt(sum));
                } else {
                    const pivot = result.data[column * n + column];
                    if (!std.math.isFinite(pivot) or pivot <= 0.0) return error.MatrixNotPositiveDefinite;
                    const value = sum / @as(f64, pivot);
                    if (!std.math.isFinite(value)) return error.MatrixNotPositiveDefinite;
                    result.data[row * n + column] = @floatCast(value);
                }
            }
        }
        return result;
    }

    pub fn choleskyInverse(self: *const Tensor, allocator: Allocator) !Tensor {
        var lower = try self.cholesky(allocator);
        defer lower.deinit();
        const n = lower.shape.dims[0];
        var inverse_lower = try Tensor.zeros(allocator, &.{ n, n });
        defer inverse_lower.deinit();

        var column: usize = 0;
        while (column < n) : (column += 1) {
            var row: usize = 0;
            while (row < n) : (row += 1) {
                var sum: f64 = if (row == column) 1.0 else 0.0;
                var inner: usize = 0;
                while (inner < row) : (inner += 1) {
                    sum -= @as(f64, lower.data[row * n + inner]) * @as(f64, inverse_lower.data[inner * n + column]);
                }
                const pivot = lower.data[row * n + row];
                if (!std.math.isFinite(pivot) or pivot <= 0.0) return error.MatrixNotPositiveDefinite;
                const value = sum / @as(f64, pivot);
                if (!std.math.isFinite(value)) return error.MatrixNotPositiveDefinite;
                inverse_lower.data[row * n + column] = @floatCast(value);
            }
        }

        var result = try Tensor.zeros(allocator, &.{ n, n });
        errdefer result.deinit();
        var row: usize = 0;
        while (row < n) : (row += 1) {
            column = row;
            while (column < n) : (column += 1) {
                var sum: f64 = 0.0;
                var inner: usize = @max(row, column);
                while (inner < n) : (inner += 1) {
                    sum += @as(f64, inverse_lower.data[inner * n + row]) * @as(f64, inverse_lower.data[inner * n + column]);
                }
                if (!std.math.isFinite(sum)) return error.MatrixNotPositiveDefinite;
                const value: f32 = @floatCast(sum);
                result.data[row * n + column] = value;
                result.data[column * n + row] = value;
            }
        }
        return result;
    }

    pub fn outerProduct(self: *const Tensor, allocator: Allocator, other: *const Tensor) !Tensor {
        const m = self.data.len;
        const n = other.data.len;

        var result = try Tensor.init(allocator, &[_]usize{ m, n });

        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                result.data[i * n + j] = self.data[i] * other.data[j];
            }
        }

        return result;
    }

    pub fn sizeBytes(self: *const Tensor) usize {
        return self.data.len * @sizeOf(f32);
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(u32, 0x54464453, .little);
        try writer.writeInt(u8, @intFromEnum(self.dtype), .little);
        try writer.writeInt(u8, tensorFlagsToBits(self.flags), .little);
        try writer.writeInt(u64, @intCast(self.shape.dims.len), .little);
        for (self.shape.dims) |dim| {
            try writer.writeInt(u64, @intCast(dim), .little);
        }
        for (self.data) |val| {
            try writer.writeInt(u32, @as(u32, @bitCast(val)), .little);
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x54464453) return error.InvalidTensorFormat;

        const dtype_raw = try reader.readInt(u8, .little);
        const flags_raw = try reader.readInt(u8, .little);
        const ndims_u64 = try reader.readInt(u64, .little);
        if (ndims_u64 > @as(u64, std.math.maxInt(usize))) return error.InvalidShape;
        const ndims: usize = @intCast(ndims_u64);
        const dims = try allocator.alloc(usize, ndims);
        defer allocator.free(dims);

        var i: usize = 0;
        while (i < ndims) : (i += 1) {
            const dim_u64 = try reader.readInt(u64, .little);
            if (dim_u64 > @as(u64, std.math.maxInt(usize))) return error.InvalidShape;
            dims[i] = @intCast(dim_u64);
        }

        var tensor = try Tensor.init(allocator, dims);
        errdefer tensor.deinit();

        tensor.dtype = try std.meta.intToEnum(Precision, dtype_raw);
        tensor.flags = tensorFlagsFromBits(flags_raw);

        i = 0;
        while (i < tensor.data.len) : (i += 1) {
            const bits = try reader.readInt(u32, .little);
            tensor.data[i] = @as(f32, @bitCast(bits));
        }

        return tensor;
    }

    pub fn fromCoreTensor(ct: *const core_tensor.Tensor, allocator: Allocator) !Tensor {
        const t = try Tensor.init(allocator, ct.shape.dims);
        @memcpy(t.data, ct.data);
        return t;
    }

    pub fn toCoreTensor(self: *const Tensor, allocator: Allocator) !core_tensor.Tensor {
        const ct = try core_tensor.Tensor.init(allocator, self.shape.dims);
        @memcpy(ct.data, self.data);
        return ct;
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, dims: []const usize) !Tensor {
        return init(arena.allocator(), dims);
    }

    pub fn initWithPool(pool: *core_memory.PoolAllocator, dims: []const usize) !Tensor {
        return init(pool.allocator(), dims);
    }

    pub fn initWithSlab(slab: *core_memory.SlabAllocator, dims: []const usize) !Tensor {
        return init(slab.allocator(), dims);
    }

    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, dims: []const usize) !Tensor {
        return init(buddy.allocator(), dims);
    }
};

pub const SFDConfig = struct {
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    clip_threshold: f32 = 0.1,
    weight_floor: f32 = 1e-3,
    fisher_max: f32 = 1e6,
    warmup_steps: usize = 10,
    use_external_fisher: bool = false,
};

pub const KFACBlock = struct {
    A_diag: Tensor,
    G_diag: Tensor,
    damping: f32,
    alpha: f32,
    update_freq: usize,
    last_update: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, input_dim: usize, output_dim: usize, damping: f32) !KFACBlock {
        return initWithAlpha(allocator, input_dim, output_dim, damping, 0.95);
    }

    pub fn initWithAlpha(allocator: Allocator, input_dim: usize, output_dim: usize, damping: f32, alpha: f32) !KFACBlock {
        const A_shape = [_]usize{ input_dim, input_dim };
        const G_shape = [_]usize{ output_dim, output_dim };

        var A = try Tensor.eye(allocator, &A_shape);
        errdefer A.deinit();

        var G = try Tensor.eye(allocator, &G_shape);
        errdefer G.deinit();

        return KFACBlock{
            .A_diag = A,
            .G_diag = G,
            .damping = damping,
            .alpha = alpha,
            .update_freq = 10,
            .last_update = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KFACBlock) void {
        self.A_diag.deinit();
        self.G_diag.deinit();
    }

    pub fn updateStatistics(self: *KFACBlock, activations: *const Tensor, gradients: *const Tensor) !void {
        const a_dim = self.A_diag.shape.dims[0];
        const g_dim = self.G_diag.shape.dims[0];

        var row: usize = 0;
        while (row < a_dim) : (row += 1) {
            var col: usize = 0;
            while (col < a_dim) : (col += 1) {
                const idx = row * a_dim + col;
                const a_r: f32 = if (row < activations.data.len) activations.data[row] else 0.0;
                const a_c: f32 = if (col < activations.data.len) activations.data[col] else 0.0;
                const diag_term: f32 = if (row == col) self.damping else 0.0;
                const target = a_r * a_c + diag_term;
                self.A_diag.data[idx] = self.alpha * self.A_diag.data[idx] + (1.0 - self.alpha) * target;
            }
        }

        row = 0;
        while (row < g_dim) : (row += 1) {
            var col: usize = 0;
            while (col < g_dim) : (col += 1) {
                const idx = row * g_dim + col;
                const g_r: f32 = if (row < gradients.data.len) gradients.data[row] else 0.0;
                const g_c: f32 = if (col < gradients.data.len) gradients.data[col] else 0.0;
                const diag_term: f32 = if (row == col) self.damping else 0.0;
                const target = g_r * g_c + diag_term;
                self.G_diag.data[idx] = self.alpha * self.G_diag.data[idx] + (1.0 - self.alpha) * target;
            }
        }
    }

    pub fn preconditionGradient(self: *const KFACBlock, grad: *Tensor) !void {
        const g_dim = self.G_diag.shape.dims[0];
        const a_dim = self.A_diag.shape.dims[0];
        if (grad.shape.dims.len != 2 or grad.shape.dims[0] != g_dim or grad.shape.dims[1] != a_dim) {
            return error.InvalidShape;
        }

        var A_inv_sqrt = try self.computeInverseSqrt(&self.A_diag);
        defer A_inv_sqrt.deinit();
        var G_inv_sqrt = try self.computeInverseSqrt(&self.G_diag);
        defer G_inv_sqrt.deinit();
        var original = try grad.clone(self.allocator);
        defer original.deinit();
        var left_scaled = try Tensor.init(self.allocator, &[_]usize{ g_dim, a_dim });
        defer left_scaled.deinit();
        try left_scaled.matmul(&G_inv_sqrt, &original);
        try grad.matmul(&left_scaled, &A_inv_sqrt);
    }

    fn computeInverseSqrt(self: *const KFACBlock, M: *const Tensor) !Tensor {
        if (M.shape.dims.len != 2 or M.shape.dims[0] != M.shape.dims[1]) return error.InvalidShape;
        const n = M.shape.dims[0];
        var matrix = try self.allocator.alloc(f64, n * n);
        defer self.allocator.free(matrix);
        var eigenvectors = try self.allocator.alloc(f64, n * n);
        defer self.allocator.free(eigenvectors);

        var scale: f64 = 1.0;
        var row: usize = 0;
        while (row < n) : (row += 1) {
            var col: usize = 0;
            while (col < n) : (col += 1) {
                const value = (@as(f64, M.data[row * n + col]) + @as(f64, M.data[col * n + row])) * 0.5;
                if (!std.math.isFinite(value)) return error.MatrixNotPositiveDefinite;
                matrix[row * n + col] = value;
                eigenvectors[row * n + col] = if (row == col) 1.0 else 0.0;
                scale = @max(scale, @abs(value));
            }
            matrix[row * n + row] += @as(f64, @max(self.damping, @as(f32, 1e-8)));
        }

        const tolerance = scale * 1e-12;
        const max_sweeps = @max(@as(usize, 8), n * n * 16);
        var sweep: usize = 0;
        while (sweep < max_sweeps) : (sweep += 1) {
            var p_idx: usize = 0;
            var q_idx: usize = 0;
            var largest: f64 = 0.0;
            row = 0;
            while (row < n) : (row += 1) {
                var col: usize = row + 1;
                while (col < n) : (col += 1) {
                    const magnitude = @abs(matrix[row * n + col]);
                    if (magnitude > largest) {
                        largest = magnitude;
                        p_idx = row;
                        q_idx = col;
                    }
                }
            }
            if (largest <= tolerance) break;

            const app = matrix[p_idx * n + p_idx];
            const aqq = matrix[q_idx * n + q_idx];
            const apq = matrix[p_idx * n + q_idx];
            const angle = 0.5 * std.math.atan2(2.0 * apq, aqq - app);
            const cosine = @cos(angle);
            const sine = @sin(angle);

            var k: usize = 0;
            while (k < n) : (k += 1) {
                if (k != p_idx and k != q_idx) {
                    const akp = matrix[k * n + p_idx];
                    const akq = matrix[k * n + q_idx];
                    const new_kp = cosine * akp - sine * akq;
                    const new_kq = sine * akp + cosine * akq;
                    matrix[k * n + p_idx] = new_kp;
                    matrix[p_idx * n + k] = new_kp;
                    matrix[k * n + q_idx] = new_kq;
                    matrix[q_idx * n + k] = new_kq;
                }
                const vkp = eigenvectors[k * n + p_idx];
                const vkq = eigenvectors[k * n + q_idx];
                eigenvectors[k * n + p_idx] = cosine * vkp - sine * vkq;
                eigenvectors[k * n + q_idx] = sine * vkp + cosine * vkq;
            }
            matrix[p_idx * n + p_idx] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
            matrix[q_idx * n + q_idx] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
            matrix[p_idx * n + q_idx] = 0.0;
            matrix[q_idx * n + p_idx] = 0.0;
        }

        var result = try Tensor.zeros(self.allocator, M.shape.dims);
        errdefer result.deinit();
        row = 0;
        while (row < n) : (row += 1) {
            var col: usize = 0;
            while (col < n) : (col += 1) {
                var sum: f64 = 0.0;
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const eigenvalue = matrix[k * n + k];
                    if (!std.math.isFinite(eigenvalue) or eigenvalue <= 0.0) return error.MatrixNotPositiveDefinite;
                    sum += eigenvectors[row * n + k] * (1.0 / @sqrt(eigenvalue)) * eigenvectors[col * n + k];
                }
                if (!std.math.isFinite(sum)) return error.MatrixNotPositiveDefinite;
                result.data[row * n + col] = @floatCast(sum);
            }
        }
        return result;
    }
};

pub const SpectralNormalizerConfig = struct {
    power_iterations: usize = 5,
    eps: f32 = 1e-12,
    max_singular_value: f32 = 1.0,
};

pub const SpectralNormalizer = struct {
    power_iterations: usize,
    eps: f32,
    max_singular_value: f32,

    pub fn init(power_iterations: usize) SpectralNormalizer {
        return SpectralNormalizer{
            .power_iterations = power_iterations,
            .eps = 1e-12,
            .max_singular_value = 1.0,
        };
    }

    pub fn initWithConfig(config: SpectralNormalizerConfig) SpectralNormalizer {
        return SpectralNormalizer{
            .power_iterations = config.power_iterations,
            .eps = config.eps,
            .max_singular_value = config.max_singular_value,
        };
    }

    pub fn normalizeWeights(self: *SpectralNormalizer, weights: *Tensor, allocator: Allocator) !void {
        const sigma = try weights.spectralNorm(allocator, self.power_iterations, self.eps);

        if (sigma > self.max_singular_value) {
            weights.mulScalar(self.max_singular_value / sigma);
        }
    }

    pub fn lipschitzRegularization(_: *const SpectralNormalizer, loss: f32, spectral_norms: []const f32, lambda: f32) f32 {
        var reg_term: f32 = 0.0;
        for (spectral_norms) |sigma| {
            const deviation = sigma - 1.0;
            reg_term += deviation * deviation;
        }

        return loss + lambda * reg_term;
    }
};

pub const SFD = struct {
    fisher_diag: Tensor,
    momentum_buffer: Tensor,
    beta1: f32,
    beta2: f32,
    eps: f32,
    clip_threshold: f32,
    weight_floor: f32,
    fisher_max: f32,
    warmup_steps: usize,
    step_count: usize,
    allocator: Allocator,
    param_size: usize,
    initialized: bool,
    use_external_fisher: bool,

    pub fn init(allocator: Allocator, param_size: usize) !SFD {
        return initWithConfig(allocator, param_size, .{});
    }

    pub fn initWithConfig(allocator: Allocator, param_size: usize, config: SFDConfig) !SFD {
        if (param_size == 0) return error.InvalidParamSize;
        if (!std.math.isFinite(config.beta1) or config.beta1 < 0.0 or config.beta1 >= 1.0) return error.InvalidBeta1;
        if (!std.math.isFinite(config.beta2) or config.beta2 < 0.0 or config.beta2 >= 1.0) return error.InvalidBeta2;
        if (!std.math.isFinite(config.eps) or config.eps <= 0.0) return error.InvalidEpsilon;
        if (!std.math.isFinite(config.clip_threshold) or config.clip_threshold <= 0.0 or config.clip_threshold > 1.0) return error.InvalidClipThreshold;
        if (!std.math.isFinite(config.weight_floor) or config.weight_floor <= 0.0) return error.InvalidWeightFloor;
        if (!std.math.isFinite(config.fisher_max) or config.fisher_max <= 0.0) return error.InvalidFisherMax;

        const shape = [_]usize{param_size};
        var fisher = try Tensor.zeros(allocator, &shape);
        errdefer fisher.deinit();
        var momentum = try Tensor.zeros(allocator, &shape);
        errdefer momentum.deinit();
        return .{
            .fisher_diag = fisher,
            .momentum_buffer = momentum,
            .beta1 = config.beta1,
            .beta2 = config.beta2,
            .eps = config.eps,
            .clip_threshold = config.clip_threshold,
            .weight_floor = config.weight_floor,
            .fisher_max = config.fisher_max,
            .warmup_steps = config.warmup_steps,
            .step_count = 0,
            .allocator = allocator,
            .param_size = param_size,
            .initialized = true,
            .use_external_fisher = config.use_external_fisher,
        };
    }

    pub fn initWithArena(arena: *core_memory.ArenaAllocator, param_size: usize) !SFD {
        return initWithConfig(arena.allocator(), param_size, .{});
    }
    pub fn initWithPool(pool: *core_memory.PoolAllocator, param_size: usize) !SFD {
        return initWithConfig(pool.allocator(), param_size, .{});
    }
    pub fn initWithBuddy(buddy: *core_memory.BuddyAllocator, param_size: usize) !SFD {
        return initWithConfig(buddy.allocator(), param_size, .{});
    }

    pub fn deinit(self: *SFD) void {
        if (!self.initialized) return;
        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.initialized = false;
    }

    fn warmupFactor(self: *const SFD, step: usize) f32 {
        if (self.warmup_steps == 0 or step >= self.warmup_steps) return 1.0;
        return @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(self.warmup_steps));
    }

    fn applySlices(self: *SFD, gradients: []const Tensor, params: []*Tensor, lr: f32) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(lr) or lr < 0.0) return error.InvalidLearningRate;
        if (gradients.len != params.len) return error.GradientCountMismatch;

        var total: usize = 0;
        for (params, 0..) |param, index| {
            if (gradients[index].data.len != param.data.len) return error.ShapeMismatch;
            total = std.math.add(usize, total, param.data.len) catch return error.ShapeMismatch;
        }
        if (total != self.param_size) return error.ShapeMismatch;

        const next_step = self.step_count +| 1;
        const step_f: f32 = @floatFromInt(next_step);
        const m_correction = @max(1.0 - std.math.pow(f32, self.beta1, step_f), self.eps);
        const f_correction = @max(1.0 - std.math.pow(f32, self.beta2, step_f), self.eps);
        const effective_lr = lr * self.warmupFactor(next_step);

        var offset: usize = 0;
        for (params, 0..) |param, tensor_index| {
            const grad = gradients[tensor_index];
            for (grad.data, 0..) |g, element_index| {
                const gradient_is_finite = std.math.isFinite(g);
                const safe_gradient: f32 = if (gradient_is_finite) g else 0.0;
                const state_index = offset + element_index;
                const old_momentum = self.momentum_buffer.data[state_index];
                const momentum_candidate = self.beta1 * old_momentum + (1.0 - self.beta1) * safe_gradient;
                const momentum = if (std.math.isFinite(momentum_candidate)) momentum_candidate else old_momentum;
                self.momentum_buffer.data[state_index] = momentum;

                var fisher = self.fisher_diag.data[state_index];
                if (!self.use_external_fisher) {
                    const fisher_candidate = self.beta2 * fisher + (1.0 - self.beta2) * safe_gradient * safe_gradient;
                    if (std.math.isFinite(fisher_candidate)) fisher = @min(fisher_candidate, self.fisher_max);
                    self.fisher_diag.data[state_index] = fisher;
                }
                if (!gradient_is_finite) continue;

                const m_hat = momentum / m_correction;
                const fisher_hat = if (self.use_external_fisher) fisher else fisher / f_correction;
                if (!std.math.isFinite(fisher_hat) or fisher_hat < 0.0) continue;
                var delta = effective_lr * m_hat / (@sqrt(fisher_hat) + self.eps);
                const parameter_scale = @max(@abs(param.data[element_index]), self.weight_floor);
                const max_update = self.clip_threshold * parameter_scale;
                delta = std.math.clamp(delta, -max_update, max_update);
                const updated = param.data[element_index] - delta;
                if (std.math.isFinite(updated)) param.data[element_index] = updated;
            }
            offset += param.data.len;
        }
        self.step_count = next_step;
    }

    pub fn update(self: *SFD, gradients: *const Tensor, params: *Tensor, lr: f32) !void {
        const gradient_slices = [_]Tensor{gradients.*};
        var parameter_slices = [_]*Tensor{params};
        return self.applySlices(&gradient_slices, &parameter_slices, lr);
    }

    pub fn updateFusedFisher(self: *SFD, gradients: *const Tensor, params: *Tensor, lr: f32) !void {
        const gradient_slices = [_]Tensor{gradients.*};
        var parameter_slices = [_]*Tensor{params};
        return self.applySlices(&gradient_slices, &parameter_slices, lr);
    }

    pub fn updateFusedFisherMulti(self: *SFD, gradients: []const Tensor, params: []*Tensor, lr: f32) !void {
        return self.applySlices(gradients, params, lr);
    }

    pub fn adaptiveLR(self: *const SFD, grad_norm: f32, param_norm: f32) f32 {
        if (!std.math.isFinite(grad_norm) or grad_norm < 0.0 or !std.math.isFinite(param_norm) or param_norm < 0.0) return 1.0;
        const result = 1.0 / @sqrt(grad_norm / (param_norm + self.eps) + self.eps);
        return if (std.math.isFinite(result)) result else 1.0;
    }

    pub fn spectralClip(self: *SFD, tensor: *Tensor, max_eig: f32) !void {
        return self.spectralClipWithIters(tensor, max_eig, 100);
    }
    pub fn spectralClipWithIters(self: *SFD, tensor: *Tensor, max_eig: f32, power_iter: usize) !void {
        if (!self.initialized) return error.NotInitialized;
        if (!std.math.isFinite(max_eig) or max_eig <= 0.0) return error.InvalidMaxEig;
        const current = try tensor.spectralNorm(self.allocator, power_iter, 1e-6);
        if (std.math.isFinite(current) and current > max_eig) tensor.mulScalar(max_eig / current);
    }

    pub fn accumulateFisher(self: *SFD, grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        for (grads) |grad| {
            const count = @min(self.fisher_diag.data.len, grad.data.len);
            for (grad.data[0..count], 0..) |g, index| {
                if (std.math.isFinite(g)) self.fisher_diag.data[index] = @min(self.fisher_diag.data[index] + g * g, self.fisher_max);
            }
        }
    }

    pub fn resetFisher(self: *SFD) void {
        if (self.initialized) self.fisher_diag.fill(0.0);
    }

    pub fn clipGradNorm(self: *SFD, grads: []*Tensor, max_norm: f32) f32 {
        if (!std.math.isFinite(max_norm) or max_norm <= 0.0) return 0.0;
        var sum_sq: f64 = 0.0;
        for (grads) |grad| {
            const norm = grad.normL2();
            if (std.math.isFinite(norm)) sum_sq += @as(f64, norm) * @as(f64, norm);
        }
        const norm: f32 = @floatCast(@sqrt(sum_sq));
        if (norm > max_norm) for (grads) |grad| grad.mulScalar(max_norm / (norm + self.eps));
        return norm;
    }

    pub fn ampSchedule(_: *const SFD, step: usize, warmup: usize, total: usize) f32 {
        if (warmup > 0 and step < warmup) return @as(f32, @floatFromInt(step + 1)) / @as(f32, @floatFromInt(warmup));
        if (total <= warmup) return 1.0;
        const progress = @min(@as(f32, @floatFromInt(step -| warmup)) / @as(f32, @floatFromInt(total - warmup)), 1.0);
        return 0.5 * (1.0 + @cos(std.math.pi * progress));
    }

    pub fn saveState(self: *const SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;
        var file = try core_io.createFilePath(path, .{ .mode = 0o600 });
        defer file.close();
        var buffered = std.io.bufferedWriter(file.writer());
        const writer = buffered.writer();
        try writer.writeInt(u32, 0x53464433, .little);
        try writer.writeInt(u32, @bitCast(self.beta1), .little);
        try writer.writeInt(u32, @bitCast(self.beta2), .little);
        try writer.writeInt(u32, @bitCast(self.eps), .little);
        try writer.writeInt(u32, @bitCast(self.clip_threshold), .little);
        try writer.writeInt(u32, @bitCast(self.weight_floor), .little);
        try writer.writeInt(u32, @bitCast(self.fisher_max), .little);
        try writer.writeInt(u64, @intCast(self.warmup_steps), .little);
        try writer.writeInt(u64, @intCast(self.param_size), .little);
        try writer.writeInt(u64, @intCast(self.step_count), .little);
        try self.fisher_diag.save(writer);
        try self.momentum_buffer.save(writer);
        try buffered.flush();
    }

    pub fn loadState(self: *SFD, path: []const u8) !void {
        if (!self.initialized) return error.NotInitialized;
        const file = try core_io.openFilePath(path, .{ .mode = .read_only });
        defer file.close();
        var buffered = std.io.bufferedReader(file.reader());
        const reader = buffered.reader();
        const magic = try reader.readInt(u32, .little);
        if (magic != 0x53464433) return error.InvalidStateFormat;
        const beta1: f32 = @bitCast(try reader.readInt(u32, .little));
        const beta2: f32 = @bitCast(try reader.readInt(u32, .little));
        const eps: f32 = @bitCast(try reader.readInt(u32, .little));
        const clip: f32 = @bitCast(try reader.readInt(u32, .little));
        const weight_floor: f32 = @bitCast(try reader.readInt(u32, .little));
        const fisher_max: f32 = @bitCast(try reader.readInt(u32, .little));
        const warmup_u64 = try reader.readInt(u64, .little);
        const size_u64 = try reader.readInt(u64, .little);
        const step_u64 = try reader.readInt(u64, .little);
        if (!std.math.isFinite(beta1) or beta1 < 0.0 or beta1 >= 1.0 or
            !std.math.isFinite(beta2) or beta2 < 0.0 or beta2 >= 1.0 or
            !std.math.isFinite(eps) or eps <= 0.0 or !std.math.isFinite(clip) or clip <= 0.0 or clip > 1.0 or
            !std.math.isFinite(weight_floor) or weight_floor <= 0.0 or
            !std.math.isFinite(fisher_max) or fisher_max <= 0.0) return error.InvalidStateFormat;
        if (warmup_u64 > std.math.maxInt(usize) or size_u64 > std.math.maxInt(usize) or step_u64 > std.math.maxInt(usize)) return error.InvalidStateFormat;
        if (@as(usize, @intCast(size_u64)) != self.param_size) return error.ShapeMismatch;

        var fisher = try Tensor.load(self.allocator, reader);
        errdefer fisher.deinit();
        var momentum = try Tensor.load(self.allocator, reader);
        errdefer momentum.deinit();
        if (fisher.data.len != self.param_size or momentum.data.len != self.param_size) return error.ShapeMismatch;
        const trailing = reader.readByte() catch |err| switch (err) {
            error.EndOfStream => null,
            else => return err,
        };
        if (trailing != null) return error.InvalidStateFormat;
        self.fisher_diag.deinit();
        self.momentum_buffer.deinit();
        self.fisher_diag = fisher;
        self.momentum_buffer = momentum;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.clip_threshold = clip;
        self.weight_floor = weight_floor;
        self.fisher_max = fisher_max;
        self.warmup_steps = @intCast(warmup_u64);
        self.step_count = @intCast(step_u64);
    }

    pub fn writeBackFP16(params: *const Tensor, dst: []f16) !void {
        if (dst.len != params.data.len) return error.ShapeMismatch;
        for (params.data, dst) |value, *target| target.* = @floatCast(std.math.clamp(value, @as(f32, -65504.0), @as(f32, 65504.0)));
    }
    pub fn loadFromFP16(src: []const f16, params: *Tensor) !void {
        if (src.len != params.data.len) return error.ShapeMismatch;
        for (src, params.data) |value, *target| target.* = @floatCast(value);
    }

    pub fn warmStart(self: *SFD, previous: *const Tensor) void {
        if (!self.initialized) return;
        const count = @min(previous.data.len, self.fisher_diag.data.len);
        for (previous.data[0..count], 0..) |value, index| {
            if (std.math.isFinite(value) and value >= 0.0) self.fisher_diag.data[index] = @min(value, self.fisher_max);
        }
    }

    pub fn varianceReduction(self: *SFD, noise_grads: []const Tensor) !void {
        if (!self.initialized) return error.NotInitialized;
        if (noise_grads.len == 0) return error.EmptyGrads;
        var mean = try Tensor.zeros(self.allocator, self.fisher_diag.shape.dims);
        defer mean.deinit();
        var second = try Tensor.zeros(self.allocator, self.fisher_diag.shape.dims);
        defer second.deinit();
        for (noise_grads) |grad| {
            const count = @min(grad.data.len, mean.data.len);
            for (grad.data[0..count], 0..) |g, index| if (std.math.isFinite(g)) {
                mean.data[index] += g;
                second.data[index] += g * g;
            };
        }
        const divisor: f32 = @floatFromInt(noise_grads.len);
        for (self.fisher_diag.data, 0..) |*fisher, index| {
            const avg = mean.data[index] / divisor;
            const variance = @max(0.0, second.data[index] / divisor - avg * avg);
            fisher.* = @max(0.0, fisher.* - variance);
        }
    }
};

test "SFD init and deinit" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    try std.testing.expect(sfd.initialized);
    try std.testing.expectEqual(@as(usize, 4), sfd.param_size);
}

test "SFD update" {
    const gpa = std.testing.allocator;
    var sfd = try SFD.init(gpa, 4);
    defer sfd.deinit();

    const shape = [_]usize{4};
    var grads = try Tensor.init(gpa, &shape);
    defer grads.deinit();

    grads.data[0] = 1.0;
    grads.data[1] = 2.0;
    grads.data[2] = 3.0;
    grads.data[3] = 4.0;

    var params = try Tensor.init(gpa, &shape);
    defer params.deinit();

    params.fill(0.0);

    try sfd.update(&grads, &params, 0.1);
    for (params.data) |v| {
        try std.testing.expect(v < 0);
    }
}

test "Tensor clone" {
    const gpa = std.testing.allocator;
    const shape = [_]usize{ 2, 3 };
    var t1 = try Tensor.init(gpa, &shape);
    defer t1.deinit();

    t1.fill(5.0);

    var t2 = try t1.clone(gpa);
    defer t2.deinit();

    try std.testing.expectEqual(t1.data[0], t2.data[0]);
    try std.testing.expectEqual(false, t2.flags.in_tensor_memory);
}

test "KFACBlock init" {
    const gpa = std.testing.allocator;
    var block = try KFACBlock.init(gpa, 4, 4, 0.001);
    defer block.deinit();

    try std.testing.expectEqual(@as(f32, 0.001), block.damping);
    try std.testing.expectEqual(@as(f32, 0.95), block.alpha);
}

test "KFAC precondition uses inverse square roots" {
    const allocator = std.testing.allocator;
    var block = try KFACBlock.init(allocator, 2, 2, 0.001);
    defer block.deinit();
    var gradient = try Tensor.init(allocator, &.{ 2, 2 });
    defer gradient.deinit();
    gradient.fill(1.0);
    try block.preconditionGradient(&gradient);
    for (gradient.data) |value| try std.testing.expect(std.math.isFinite(value));
}

test "SpectralNormalizer normalizeWeights" {
    const gpa = std.testing.allocator;
    var normalizer = SpectralNormalizer.init(10);
    const shape = [_]usize{ 4, 4 };
    var w = try Tensor.init(gpa, &shape);
    defer w.deinit();
    w.fillRandomNormal(0.0, 2.0);
    try normalizer.normalizeWeights(&w, gpa);
    const sigma = try w.spectralNorm(gpa, 20, 1e-6);
    try std.testing.expect(sigma <= normalizer.max_singular_value + 1e-3);
}

test "SFD fused Fisher update matches persistent momentum Fisher reference" {
    const gpa = std.testing.allocator;
    var opt = try SFD.initWithConfig(gpa, 16, .{ .warmup_steps = 0 });
    defer opt.deinit();

    var grads = try Tensor.init(gpa, &.{16});
    defer grads.deinit();
    var params = try Tensor.init(gpa, &.{16});
    defer params.deinit();
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        grads.data[i] = 0.02 * @as(f32, @floatFromInt((i * 11) % 7)) - 0.05;
        params.data[i] = 0.1 * @as(f32, @floatFromInt((i * 3) % 5)) - 0.2;
    }
    const initial_params = try gpa.dupe(f32, params.data);
    defer gpa.free(initial_params);

    const lr: f32 = 0.01;
    try opt.updateFusedFisher(&grads, &params, lr);

    i = 0;
    while (i < 16) : (i += 1) {
        const g = grads.data[i];
        const m = (1.0 - opt.beta1) * g;
        const f = (1.0 - opt.beta2) * g * g;
        const m_hat = m / (1.0 - opt.beta1);
        const f_hat = f / (1.0 - opt.beta2);
        const parameter_scale = @max(@abs(initial_params[i]), @as(f32, 1e-3));
        const unclipped = lr * m_hat / (@sqrt(f_hat) + opt.eps);
        const expected = initial_params[i] - std.math.clamp(unclipped, -opt.clip_threshold * parameter_scale, opt.clip_threshold * parameter_scale);
        try std.testing.expectApproxEqAbs(expected, params.data[i], 1e-5);
        try std.testing.expectApproxEqAbs(f, opt.fisher_diag.data[i], 1e-6);
        try std.testing.expectApproxEqAbs(m, opt.momentum_buffer.data[i], 1e-6);
    }
}

test "SFD fused multi-layer update applies per-tensor slices" {
    const gpa = std.testing.allocator;
    var opt = try SFD.initWithConfig(gpa, 12, .{ .warmup_steps = 0 });
    defer opt.deinit();

    var g1 = try Tensor.init(gpa, &.{8});
    defer g1.deinit();
    var g2 = try Tensor.init(gpa, &.{4});
    defer g2.deinit();
    var p1 = try Tensor.init(gpa, &.{8});
    defer p1.deinit();
    var p2 = try Tensor.init(gpa, &.{4});
    defer p2.deinit();
    p1.fill(0.05);
    p2.fill(-0.05);
    g1.fill(0.1);
    g2.fill(-0.1);

    const grads = [_]Tensor{ g1, g2 };
    var params_list = [_]*Tensor{ &p1, &p2 };
    try opt.updateFusedFisherMulti(&grads, &params_list, 0.01);
    try std.testing.expectEqual(@as(usize, 1), opt.step_count);

    const m = (1.0 - opt.beta1) * 0.1;
    const f = (1.0 - opt.beta2) * 0.01;
    const raw_update = 0.01 * (m / (1.0 - opt.beta1)) / (@sqrt(f / (1.0 - opt.beta2)) + opt.eps);
    const expected1 = 0.05 - std.math.clamp(raw_update, -opt.clip_threshold * 0.05, opt.clip_threshold * 0.05);
    try std.testing.expectApproxEqAbs(expected1, p1.data[0], 1e-5);
}

test "SFD substitutes zero for non-finite gradients and decays state" {
    const allocator = std.testing.allocator;
    var optimizer = try SFD.initWithConfig(allocator, 2, .{ .warmup_steps = 0 });
    defer optimizer.deinit();
    var gradients = try Tensor.init(allocator, &.{2});
    defer gradients.deinit();
    var parameters = try Tensor.init(allocator, &.{2});
    defer parameters.deinit();
    gradients.data[0] = 0.25;
    gradients.data[1] = 0.5;
    parameters.fill(0.1);
    try optimizer.updateFusedFisher(&gradients, &parameters, 0.01);

    const old_parameter = parameters.data[1];
    const old_momentum = optimizer.momentum_buffer.data[1];
    const old_fisher = optimizer.fisher_diag.data[1];
    gradients.data[0] = 0.1;
    gradients.data[1] = std.math.nan(f32);
    try optimizer.updateFusedFisher(&gradients, &parameters, 0.01);
    try std.testing.expectEqual(old_parameter, parameters.data[1]);
    try std.testing.expectApproxEqAbs(old_momentum * optimizer.beta1, optimizer.momentum_buffer.data[1], 1e-7);
    try std.testing.expectApproxEqAbs(old_fisher * optimizer.beta2, optimizer.fisher_diag.data[1], 1e-7);
}

test "SFD fused multi rejects gradient count mismatch" {
    const allocator = std.testing.allocator;
    var optimizer = try SFD.initWithConfig(allocator, 2, .{ .warmup_steps = 0 });
    defer optimizer.deinit();
    var first = try Tensor.init(allocator, &.{1});
    defer first.deinit();
    var second = try Tensor.init(allocator, &.{1});
    defer second.deinit();
    var gradient = try Tensor.init(allocator, &.{1});
    defer gradient.deinit();
    const gradients = [_]Tensor{gradient};
    var parameters = [_]*Tensor{ &first, &second };
    try std.testing.expectError(error.GradientCountMismatch, optimizer.updateFusedFisherMulti(&gradients, &parameters, 0.01));
}

test "SFD FP16 writeback clamps to representable range" {
    const gpa = std.testing.allocator;
    var params = try Tensor.init(gpa, &.{4});
    defer params.deinit();
    params.data[0] = 70000.0;
    params.data[1] = -70000.0;
    params.data[2] = 0.5;
    params.data[3] = -2.25;

    var out: [4]f16 = undefined;
    try SFD.writeBackFP16(&params, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 65504.0), @as(f32, @floatCast(out[0])), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -65504.0), @as(f32, @floatCast(out[1])), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @as(f32, @floatCast(out[2])), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -2.25), @as(f32, @floatCast(out[3])), 1e-3);

    var back = try Tensor.init(gpa, &.{4});
    defer back.deinit();
    try SFD.loadFromFP16(&out, &back);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), back.data[2], 1e-3);
}

test "SFD configurable weight floor bounds first zero-weight update" {
    const allocator = std.testing.allocator;
    var optimizer = try SFD.initWithConfig(allocator, 1, .{
        .beta1 = 0.0,
        .beta2 = 0.0,
        .clip_threshold = 0.1,
        .weight_floor = 0.25,
        .warmup_steps = 0,
    });
    defer optimizer.deinit();
    var gradient = try Tensor.init(allocator, &.{1});
    defer gradient.deinit();
    var parameter = try Tensor.zeros(allocator, &.{1});
    defer parameter.deinit();
    gradient.data[0] = 1.0;
    try optimizer.updateFusedFisher(&gradient, &parameter, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.025), parameter.data[0], 1e-7);
}

test "SFD3 persists complete optimizer state" {
    const allocator = std.testing.allocator;
    var source = try SFD.initWithConfig(allocator, 2, .{
        .beta1 = 0.0,
        .beta2 = 0.0,
        .eps = 2e-7,
        .clip_threshold = 0.07,
        .weight_floor = 0.125,
        .fisher_max = 17.0,
        .warmup_steps = 3,
    });
    defer source.deinit();
    source.step_count = 9;
    source.fisher_diag.data[0] = 0.25;
    source.fisher_diag.data[1] = 0.5;
    source.momentum_buffer.data[0] = -0.75;
    source.momentum_buffer.data[1] = 1.25;

    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const directory_path = try temporary.dir.realpathAlloc(allocator, ".");
    defer allocator.free(directory_path);
    const state_path = try std.fs.path.join(allocator, &.{ directory_path, "optimizer.sfd" });
    defer allocator.free(state_path);
    try source.saveState(state_path);

    var restored = try SFD.init(allocator, 2);
    defer restored.deinit();
    try restored.loadState(state_path);
    try std.testing.expectEqual(source.beta1, restored.beta1);
    try std.testing.expectEqual(source.beta2, restored.beta2);
    try std.testing.expectEqual(source.eps, restored.eps);
    try std.testing.expectEqual(source.clip_threshold, restored.clip_threshold);
    try std.testing.expectEqual(source.weight_floor, restored.weight_floor);
    try std.testing.expectEqual(source.fisher_max, restored.fisher_max);
    try std.testing.expectEqual(source.warmup_steps, restored.warmup_steps);
    try std.testing.expectEqual(source.step_count, restored.step_count);
    try std.testing.expectEqualSlices(f32, source.fisher_diag.data, restored.fisher_diag.data);
    try std.testing.expectEqualSlices(f32, source.momentum_buffer.data, restored.momentum_buffer.data);
}

test "weighted gradient reduction produces identical local SFD state" {
    const allocator = std.testing.allocator;
    const config = SFDConfig{
        .beta1 = 0.8,
        .beta2 = 0.95,
        .eps = 1e-7,
        .clip_threshold = 0.08,
        .weight_floor = 0.02,
        .warmup_steps = 0,
    };
    var first_optimizer = try SFD.initWithConfig(allocator, 3, config);
    defer first_optimizer.deinit();
    var second_optimizer = try SFD.initWithConfig(allocator, 3, config);
    defer second_optimizer.deinit();
    var first_parameters = try Tensor.init(allocator, &.{3});
    defer first_parameters.deinit();
    var second_parameters = try Tensor.init(allocator, &.{3});
    defer second_parameters.deinit();
    first_parameters.data[0] = 0.2;
    first_parameters.data[1] = -0.4;
    first_parameters.data[2] = 0.0;
    @memcpy(second_parameters.data, first_parameters.data);
    const first_local = [_]f32{ 0.5, -0.25, 0.125 };
    const second_local = [_]f32{ -0.1, 0.75, -0.5 };
    const first_fraction: f32 = 0.25;
    const second_fraction: f32 = 0.75;
    var reduced_gradient = try Tensor.init(allocator, &.{3});
    defer reduced_gradient.deinit();
    for (reduced_gradient.data, 0..) |*value, index| {
        value.* = first_local[index] * first_fraction + second_local[index] * second_fraction;
    }
    try first_optimizer.update(&reduced_gradient, &first_parameters, 0.003);
    try second_optimizer.update(&reduced_gradient, &second_parameters, 0.003);
    try std.testing.expectEqualSlices(f32, first_parameters.data, second_parameters.data);
    try std.testing.expectEqualSlices(f32, first_optimizer.momentum_buffer.data, second_optimizer.momentum_buffer.data);
    try std.testing.expectEqualSlices(f32, first_optimizer.fisher_diag.data, second_optimizer.fisher_diag.data);
    try std.testing.expectEqual(first_optimizer.step_count, second_optimizer.step_count);
}
