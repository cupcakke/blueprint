const std = @import("std");
const builtin = @import("builtin");
const Tensor = @import("../core/tensor.zig").Tensor;

pub const OFTB = struct {
    pub const FRACTAL_SCALE: f32 = 0.7071067811865476;
    pub const FRACTAL_SCALE_SQ: f32 = 0.5000000000000001;
    pub const LOG_DET_JACOBIAN: f32 = 0.0;

    dim: usize,

    pub fn init(d: usize) OFTB {
        std.debug.assert(d != 0);
        std.debug.assert(d <= std.math.maxInt(usize) / 2);
        return OFTB{
            .dim = d,
        };
    }

    pub fn deinit(self: *OFTB) void {
        self.* = undefined;
    }

    fn vectorLen() usize {
        if (comptime builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) {
            return 16;
        }
        return 8;
    }

    pub fn forwardInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        self.forwardSliceInPlace(x.data);
    }

    pub fn forwardSliceInPlace(self: OFTB, data: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (data.len != total) return;
        const half = self.dim;
        const x1 = data[0..half];
        const x2 = data[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = x1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = x2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            x1[i..][0..VLEN].* = (va - vb) * vscale;
            x2[i..][0..VLEN].* = (va + vb) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = x1[i];
            const b = x2[i];
            x1[i] = (a - b) * scale;
            x2[i] = (a + b) * scale;
        }
    }

    pub fn backwardInPlace(self: OFTB, grad: []f32) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (grad.len != total) return error.DimensionMismatch;
        self.backwardSliceInPlace(grad);
    }

    pub fn backwardSliceInPlace(self: OFTB, grad: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (grad.len != total) return;
        const half = self.dim;
        const g1 = grad[0..half];
        const g2 = grad[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            const vscale: @Vector(VLEN, f32) = @splat(scale);
            g1[i..][0..VLEN].* = (va + vb) * vscale;
            g2[i..][0..VLEN].* = (vb - va) * vscale;
        }
        while (i < half) : (i += 1) {
            const a = g1[i];
            const b = g2[i];
            g1[i] = (a + b) * scale;
            g2[i] = (b - a) * scale;
        }
    }

    pub fn inverseInPlace(self: OFTB, x: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (self.dim > std.math.maxInt(usize) / 2) return error.DimensionOverflow;
        const total = self.dim * 2;
        if (x.data.len != total) return error.DimensionMismatch;
        self.backwardSliceInPlace(x.data);
    }

    pub fn inverseSliceInPlace(self: OFTB, data: []f32) void {
        self.backwardSliceInPlace(data);
    }

    pub fn forwardBackwardFusedInPlace(self: OFTB, activation: []f32, grad: []f32) void {
        self.forwardSliceInPlace(activation);
        self.backwardSliceInPlace(grad);
    }

    pub fn symplecticReversalInPlace(self: OFTB, activation: []f32, grad: []f32) void {
        if (self.dim == 0) return;
        const total = self.dim * 2;
        if (activation.len != total or grad.len != total) return;
        const half = self.dim;
        const a1 = activation[0..half];
        const a2 = activation[half..][0..half];
        const g1 = grad[0..half];
        const g2 = grad[half..][0..half];
        const scale: f32 = FRACTAL_SCALE;
        const VLEN: usize = comptime vectorLen();
        var i: usize = 0;
        while (i + VLEN <= half) : (i += VLEN) {
            const wa: @Vector(VLEN, f32) = a1[i..][0..VLEN].*;
            const wb: @Vector(VLEN, f32) = a2[i..][0..VLEN].*;
            const wscale: @Vector(VLEN, f32) = @splat(scale);
            a1[i..][0..VLEN].* = (wa + wb) * wscale;
            a2[i..][0..VLEN].* = (wb - wa) * wscale;
            const va: @Vector(VLEN, f32) = g1[i..][0..VLEN].*;
            const vb: @Vector(VLEN, f32) = g2[i..][0..VLEN].*;
            g1[i..][0..VLEN].* = (va + vb) * wscale;
            g2[i..][0..VLEN].* = (vb - va) * wscale;
        }
        while (i < half) : (i += 1) {
            const a = a1[i];
            const b = a2[i];
            a1[i] = (a + b) * scale;
            a2[i] = (b - a) * scale;
            const ga = g1[i];
            const gb = g2[i];
            g1[i] = (ga + gb) * scale;
            g2[i] = (gb - ga) * scale;
        }
    }

    pub fn logDeterminantJacobian(self: OFTB) f32 {
        _ = self;
        return LOG_DET_JACOBIAN;
    }

    pub fn logDeterminantAdjointShift(_: OFTB) f32 {
        return 1.0;
    }

    pub fn isSymplectic(_: OFTB) bool {
        return true;
    }

    pub fn forwardRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            self.forwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }

    pub fn inverseRows(self: OFTB, rows: *Tensor) !void {
        if (self.dim == 0) return error.InvalidDimension;
        if (rows.shape.dims.len != 2) return error.DimensionMismatch;
        const total = self.dim * 2;
        if (rows.shape.dims[1] != total) return error.DimensionMismatch;
        if (!rows.shape.isContiguous()) return error.DimensionMismatch;
        const row_count = rows.shape.dims[0];
        var r: usize = 0;
        while (r < row_count) : (r += 1) {
            self.backwardSliceInPlace(rows.data[r * total ..][0..total]);
        }
    }
};

pub fn mixForward(oftb: OFTB, x: *Tensor) !void {
    try oftb.forwardInPlace(x);
}

pub fn mixBackward(oftb: OFTB, grad: []f32) !void {
    try oftb.backwardInPlace(grad);
}

pub fn mixInverse(oftb: OFTB, x: *Tensor) !void {
    try oftb.inverseInPlace(x);
}

comptime {
    _ = OFTB;
}

test "OFTB forward then backward returns input within 1e-5 tolerance" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();

    const dim: usize = 32;
    const total = dim * 2;

    var input = try Tensor.init(allocator, &.{ 1, total });
    defer input.deinit();
    var i: usize = 0;
    while (i < input.data.len) : (i += 1) {
        input.data[i] = (random.float(f32) - 0.5) * 2.0;
    }

    const original = try allocator.dupe(f32, input.data);
    defer allocator.free(original);

    var oftb = OFTB.init(dim);
    try oftb.forwardInPlace(&input);
    try oftb.backwardInPlace(input.data);

    i = 0;
    while (i < input.data.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(input.data[i], original[i], 1e-5);
    }
}

test "OFTB inverse equals forward transpose and restores activations" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const random = prng.random();

    const dim: usize = 64;
    const total = dim * 2;

    var input = try Tensor.init(allocator, &.{ 1, total });
    defer input.deinit();
    var i: usize = 0;
    while (i < input.data.len) : (i += 1) {
        input.data[i] = (random.float(f32) - 0.5) * 4.0;
    }

    const original = try allocator.dupe(f32, input.data);
    defer allocator.free(original);

    var oftb = OFTB.init(dim);
    try oftb.forwardInPlace(&input);
    try oftb.inverseInPlace(&input);

    i = 0;
    while (i < input.data.len) : (i += 1) {
        try std.testing.expectApproxEqAbs(input.data[i], original[i], 1e-5);
    }
    try std.testing.expectApproxEqAbs(oftb.logDeterminantJacobian(), 0.0, 1e-7);
}

test "OFTB symplectic reversal acts on activation and gradient in one pass" {
    const allocator = std.testing.allocator;
    const dim: usize = 16;
    const total = dim * 2;

    var act = try Tensor.init(allocator, &.{ 1, total });
    defer act.deinit();
    var grad_data = try allocator.alloc(f32, total);
    defer allocator.free(grad_data);

    var i: usize = 0;
    while (i < total) : (i += 1) {
        act.data[i] = @as(f32, @floatFromInt(i)) * 0.05 - 0.4;
        grad_data[i] = @as(f32, @floatFromInt(total - i)) * 0.02;
    }

    const act_copy = try allocator.dupe(f32, act.data);
    defer allocator.free(act_copy);
    const grad_copy = try allocator.dupe(f32, grad_data);
    defer allocator.free(grad_copy);

    var oftb = OFTB.init(dim);
    oftb.symplecticReversalInPlace(act.data, grad_data);
    oftb.forwardSliceInPlace(act.data);
    oftb.forwardSliceInPlace(grad_data);

    i = 0;
    while (i < total) : (i += 1) {
        try std.testing.expectApproxEqAbs(act.data[i], act_copy[i], 1e-5);
        try std.testing.expectApproxEqAbs(grad_data[i], grad_copy[i], 1e-5);
    }
}
