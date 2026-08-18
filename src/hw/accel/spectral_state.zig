const std = @import("std");

pub const SpectralStateError = error{
    InvalidDimensions,
    AllocationFailed,
    OwnershipMismatch,
};

pub const StackSpectralState = struct {
    allocator: std.mem.Allocator,
    u: []f32,
    v: []f32,
    num_layers: usize,
    half: usize,
    columns: usize,
    ownership_epoch: u64,
    iterations_applied: u64,
    last_sigma_before: f32,
    last_sigma_after: f32,

    pub fn compatible(self: *const StackSpectralState, num_layers: usize, half: usize, columns: usize, ownership_epoch: u64) bool {
        return self.num_layers == num_layers and self.half == half and self.columns == columns and self.ownership_epoch == ownership_epoch;
    }

    pub fn layerU(self: *const StackSpectralState, layer: usize) SpectralStateError![]f32 {
        if (layer >= self.num_layers) return SpectralStateError.InvalidDimensions;
        const offset = layer * self.half;
        return self.u[offset .. offset + self.half];
    }

    pub fn layerV(self: *const StackSpectralState, layer: usize) SpectralStateError![]f32 {
        if (layer >= self.num_layers) return SpectralStateError.InvalidDimensions;
        const offset = layer * self.columns;
        return self.v[offset .. offset + self.columns];
    }

    pub fn deinit(self: *StackSpectralState) void {
        self.allocator.free(self.u);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

pub fn initStackSpectralState(
    allocator: std.mem.Allocator,
    num_layers: usize,
    half: usize,
    columns: usize,
    ownership_epoch: u64,
) SpectralStateError!StackSpectralState {
    if (num_layers == 0 or half == 0 or columns == 0) return SpectralStateError.InvalidDimensions;
    const u_total = std.math.mul(usize, num_layers, half) catch return SpectralStateError.InvalidDimensions;
    const v_total = std.math.mul(usize, num_layers, columns) catch return SpectralStateError.InvalidDimensions;
    const u = allocator.alloc(f32, u_total) catch return SpectralStateError.AllocationFailed;
    errdefer allocator.free(u);
    const v = allocator.alloc(f32, v_total) catch return SpectralStateError.AllocationFailed;
    errdefer allocator.free(v);
    const initial_u = 1.0 / @sqrt(@as(f32, @floatFromInt(half)));
    const initial_v = 1.0 / @sqrt(@as(f32, @floatFromInt(columns)));
    @memset(u, initial_u);
    @memset(v, initial_v);
    return StackSpectralState{
        .allocator = allocator,
        .u = u,
        .v = v,
        .num_layers = num_layers,
        .half = half,
        .columns = columns,
        .ownership_epoch = ownership_epoch,
        .iterations_applied = 0,
        .last_sigma_before = 0.0,
        .last_sigma_after = 0.0,
    };
}

pub fn ensureStackSpectralState(
    allocator: std.mem.Allocator,
    existing: ?*StackSpectralState,
    num_layers: usize,
    half: usize,
    columns: usize,
    ownership_epoch: u64,
) SpectralStateError!StackSpectralState {
    if (existing) |state| {
        if (state.compatible(num_layers, half, columns, ownership_epoch)) {
            return state.*;
        }
    }
    return initStackSpectralState(allocator, num_layers, half, columns, ownership_epoch);
}

pub const EmbeddingSpectralState = struct {
    allocator: std.mem.Allocator,
    u: []f32,
    v: []f32,
    rows: usize,
    columns: usize,
    ownership_epoch: u64,

    pub fn compatible(self: *const EmbeddingSpectralState, rows: usize, columns: usize, ownership_epoch: u64) bool {
        return self.rows == rows and self.columns == columns and self.ownership_epoch == ownership_epoch;
    }

    pub fn deinit(self: *EmbeddingSpectralState) void {
        self.allocator.free(self.u);
        self.allocator.free(self.v);
        self.* = undefined;
    }
};

pub fn initEmbeddingSpectralState(allocator: std.mem.Allocator, rows: usize, columns: usize, ownership_epoch: u64) SpectralStateError!EmbeddingSpectralState {
    if (rows == 0 or columns == 0) return SpectralStateError.InvalidDimensions;
    const u = allocator.alloc(f32, rows) catch return SpectralStateError.AllocationFailed;
    errdefer allocator.free(u);
    const v = allocator.alloc(f32, columns) catch return SpectralStateError.AllocationFailed;
    errdefer allocator.free(v);
    @memset(u, 1.0 / @sqrt(@as(f32, @floatFromInt(rows))));
    @memset(v, 1.0 / @sqrt(@as(f32, @floatFromInt(columns))));
    return EmbeddingSpectralState{
        .allocator = allocator,
        .u = u,
        .v = v,
        .rows = rows,
        .columns = columns,
        .ownership_epoch = ownership_epoch,
    };
}

pub fn recordNormalization(state: *StackSpectralState, sigma_before: f32, sigma_after: f32, iterations: usize) void {
    state.last_sigma_before = sigma_before;
    state.last_sigma_after = sigma_after;
    state.iterations_applied += iterations;
}

test "stack spectral state reuses buffers across compatible calls" {
    const allocator = std.testing.allocator;
    var state = try initStackSpectralState(allocator, 3, 8, 9, 1);
    const u_pointer = state.u.ptr;
    const v_pointer = state.v.ptr;
    var refreshed = try ensureStackSpectralState(allocator, &state, 3, 8, 9, 1);
    try std.testing.expectEqual(u_pointer, refreshed.u.ptr);
    try std.testing.expectEqual(v_pointer, refreshed.v.ptr);
    recordNormalization(&refreshed, 1.4, 0.9, 1);
    try std.testing.expectEqual(@as(u64, 1), refreshed.iterations_applied);
    recordNormalization(&refreshed, 1.1, 0.9, 1);
    try std.testing.expectEqual(@as(u64, 2), refreshed.iterations_applied);
    try std.testing.expectEqual(@as(f32, 1.1), refreshed.last_sigma_before);
    refreshed.deinit();
}

test "stack spectral state resets only on shape or ownership change" {
    const allocator = std.testing.allocator;
    var state = try initStackSpectralState(allocator, 2, 4, 5, 1);
    const u_pointer = state.u.ptr;
    var reshaped = try ensureStackSpectralState(allocator, &state, 2, 6, 7, 1);
    state.deinit();
    try std.testing.expect(u_pointer != reshaped.u.ptr);
    try std.testing.expectEqual(@as(usize, 2 * 6), reshaped.u.len);
    reshaped.deinit();

    var state_b = try initStackSpectralState(allocator, 2, 4, 5, 1);
    var reowned = try ensureStackSpectralState(allocator, &state_b, 2, 4, 5, 2);
    state_b.deinit();
    try std.testing.expectEqual(@as(u64, 2), reowned.ownership_epoch);
    reowned.deinit();
}

test "stack spectral state per-layer views are disjoint and validated" {
    const allocator = std.testing.allocator;
    var state = try initStackSpectralState(allocator, 3, 4, 5, 1);
    defer state.deinit();
    const layer0 = try state.layerU(0);
    const layer1 = try state.layerU(1);
    try std.testing.expectEqual(@as(usize, 4), layer0.len);
    try std.testing.expect(layer1.ptr == state.u.ptr + 4);
    try std.testing.expectError(SpectralStateError.InvalidDimensions, state.layerU(3));
    const v2 = try state.layerV(2);
    try std.testing.expectEqual(@as(usize, 5), v2.len);
    try std.testing.expectError(SpectralStateError.InvalidDimensions, state.layerV(3));
}

test "embedding spectral state compatibility" {
    const allocator = std.testing.allocator;
    var state = try initEmbeddingSpectralState(allocator, 16, 8, 1);
    defer state.deinit();
    try std.testing.expect(state.compatible(16, 8, 1));
    try std.testing.expect(!state.compatible(16, 8, 2));
    try std.testing.expect(!state.compatible(32, 8, 1));
    try std.testing.expect(!state.compatible(16, 16, 1));
    try std.testing.expectError(SpectralStateError.InvalidDimensions, initEmbeddingSpectralState(allocator, 0, 8, 1));
}
