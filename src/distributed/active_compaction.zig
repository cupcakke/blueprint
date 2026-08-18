const std = @import("std");

pub const CompactionError = error{
    InvalidDimensions,
    SizeOverflow,
    AllocationFailed,
    LengthOutOfRange,
    BufferTooSmall,
};

pub const CompactionPlan = struct {
    allocator: std.mem.Allocator,
    input_tokens: []u32,
    target_tokens: []u32,
    source_positions: []usize,
    row_offsets: []usize,
    row_lengths: []usize,
    compact_lengths: [1]usize,
    active_count: usize,
    padded_count: usize,
    batch_size: usize,
    sequence_length: usize,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.row_lengths);
        self.allocator.free(self.row_offsets);
        self.allocator.free(self.source_positions);
        self.allocator.free(self.target_tokens);
        self.allocator.free(self.input_tokens);
        self.input_tokens = &[_]u32{};
        self.target_tokens = &[_]u32{};
        self.source_positions = &[_]usize{};
        self.row_offsets = &[_]usize{};
        self.row_lengths = &[_]usize{};
        self.active_count = 0;
        self.padded_count = 0;
    }

    pub fn compactLengths(self: *const Self) []const usize {
        return self.compact_lengths[0..1];
    }

    pub fn compactSequenceLength(self: *const Self) usize {
        return self.active_count;
    }

    pub fn paddedRowsSkipped(self: *const Self) usize {
        return self.padded_count - self.active_count;
    }

    pub fn savedFraction(self: *const Self) f64 {
        if (self.padded_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.paddedRowsSkipped())) / @as(f64, @floatFromInt(self.padded_count));
    }

    pub fn gradientElementDivisor(self: *const Self, model_dim: usize) CompactionError!usize {
        if (model_dim == 0) return CompactionError.InvalidDimensions;
        if (self.active_count == 0) return 1;
        return std.math.mul(usize, self.active_count, model_dim) catch return CompactionError.SizeOverflow;
    }

    pub fn gradientTokenDivisor(self: *const Self) usize {
        return @max(self.active_count, 1);
    }

    pub fn gather(
        self: *const Self,
        comptime T: type,
        padded: []const T,
        model_dim: usize,
        out_compact: []T,
    ) CompactionError!void {
        if (model_dim == 0) return CompactionError.InvalidDimensions;
        const padded_elements = std.math.mul(usize, self.padded_count, model_dim) catch return CompactionError.SizeOverflow;
        if (padded.len != padded_elements) return CompactionError.InvalidDimensions;
        const compact_elements = std.math.mul(usize, self.active_count, model_dim) catch return CompactionError.SizeOverflow;
        if (out_compact.len < compact_elements) return CompactionError.BufferTooSmall;

        for (self.source_positions, 0..) |source_row, compact_row| {
            const source_base = std.math.mul(usize, source_row, model_dim) catch return CompactionError.SizeOverflow;
            const target_base = std.math.mul(usize, compact_row, model_dim) catch return CompactionError.SizeOverflow;
            @memcpy(
                out_compact[target_base .. target_base + model_dim],
                padded[source_base .. source_base + model_dim],
            );
        }
    }

    pub fn scatter(
        self: *const Self,
        comptime T: type,
        compact: []const T,
        model_dim: usize,
        out_padded: []T,
        pad_value: T,
    ) CompactionError!void {
        if (model_dim == 0) return CompactionError.InvalidDimensions;
        const compact_elements = std.math.mul(usize, self.active_count, model_dim) catch return CompactionError.SizeOverflow;
        if (compact.len != compact_elements) return CompactionError.InvalidDimensions;
        const padded_elements = std.math.mul(usize, self.padded_count, model_dim) catch return CompactionError.SizeOverflow;
        if (out_padded.len < padded_elements) return CompactionError.BufferTooSmall;

        @memset(out_padded[0..padded_elements], pad_value);
        for (self.source_positions, 0..) |target_row, compact_row| {
            const target_base = std.math.mul(usize, target_row, model_dim) catch return CompactionError.SizeOverflow;
            const source_base = std.math.mul(usize, compact_row, model_dim) catch return CompactionError.SizeOverflow;
            @memcpy(
                out_padded[target_base .. target_base + model_dim],
                compact[source_base .. source_base + model_dim],
            );
        }
    }
};

pub fn buildPlan(
    allocator: std.mem.Allocator,
    flat_input_tokens: []const u32,
    flat_target_tokens: []const u32,
    real_sequence_lengths: []const usize,
    sequence_length: usize,
) CompactionError!CompactionPlan {
    if (sequence_length == 0) return CompactionError.InvalidDimensions;
    if (real_sequence_lengths.len == 0) return CompactionError.InvalidDimensions;

    const batch_size = real_sequence_lengths.len;
    const padded_count = std.math.mul(usize, batch_size, sequence_length) catch return CompactionError.SizeOverflow;
    if (flat_input_tokens.len != padded_count) return CompactionError.InvalidDimensions;
    if (flat_target_tokens.len != padded_count) return CompactionError.InvalidDimensions;

    var active_count: usize = 0;
    for (real_sequence_lengths) |length| {
        if (length > sequence_length) return CompactionError.LengthOutOfRange;
        active_count = std.math.add(usize, active_count, length) catch return CompactionError.SizeOverflow;
    }

    const input_tokens = allocator.alloc(u32, active_count) catch return CompactionError.AllocationFailed;
    errdefer allocator.free(input_tokens);
    const target_tokens = allocator.alloc(u32, active_count) catch return CompactionError.AllocationFailed;
    errdefer allocator.free(target_tokens);
    const source_positions = allocator.alloc(usize, active_count) catch return CompactionError.AllocationFailed;
    errdefer allocator.free(source_positions);
    const offsets_len = std.math.add(usize, batch_size, 1) catch return CompactionError.SizeOverflow;
    const row_offsets = allocator.alloc(usize, offsets_len) catch return CompactionError.AllocationFailed;
    errdefer allocator.free(row_offsets);
    const row_lengths = allocator.alloc(usize, batch_size) catch return CompactionError.AllocationFailed;
    errdefer allocator.free(row_lengths);

    var cursor: usize = 0;
    row_offsets[0] = 0;
    for (real_sequence_lengths, 0..) |length, batch_index| {
        row_lengths[batch_index] = length;
        const row_base = std.math.mul(usize, batch_index, sequence_length) catch return CompactionError.SizeOverflow;
        var sequence_index: usize = 0;
        while (sequence_index < length) : (sequence_index += 1) {
            const flat_index = std.math.add(usize, row_base, sequence_index) catch return CompactionError.SizeOverflow;
            input_tokens[cursor] = flat_input_tokens[flat_index];
            target_tokens[cursor] = flat_target_tokens[flat_index];
            source_positions[cursor] = flat_index;
            cursor += 1;
        }
        row_offsets[batch_index + 1] = cursor;
    }
    if (cursor != active_count) return CompactionError.InvalidDimensions;

    return CompactionPlan{
        .allocator = allocator,
        .input_tokens = input_tokens,
        .target_tokens = target_tokens,
        .source_positions = source_positions,
        .row_offsets = row_offsets,
        .row_lengths = row_lengths,
        .compact_lengths = [1]usize{active_count},
        .active_count = active_count,
        .padded_count = padded_count,
        .batch_size = batch_size,
        .sequence_length = sequence_length,
    };
}

pub fn futharkValidTokens(lengths: []const usize, sequence_length: usize) CompactionError!usize {
    var total: usize = 0;
    for (lengths) |length| {
        const limit = @min(length, sequence_length);
        total = std.math.add(usize, total, limit) catch return CompactionError.SizeOverflow;
    }
    return total;
}

pub fn futharkElementDivisor(valid_tokens: usize, model_dim: usize) CompactionError!usize {
    if (model_dim == 0) return CompactionError.InvalidDimensions;
    if (valid_tokens == 0) return 1;
    return std.math.mul(usize, valid_tokens, model_dim) catch return CompactionError.SizeOverflow;
}

fn buildReference(
    allocator: std.mem.Allocator,
    batch_size: usize,
    sequence_length: usize,
    lengths: []const usize,
    seed: u64,
) !struct { inputs: []u32, targets: []u32, activations: []f32 } {
    const padded = batch_size * sequence_length;
    const inputs = try allocator.alloc(u32, padded);
    const targets = try allocator.alloc(u32, padded);
    const activations = try allocator.alloc(f32, padded * 4);
    @memset(inputs, 0);
    @memset(targets, 0);
    @memset(activations, 0.0);

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (lengths, 0..) |length, batch_index| {
        var sequence_index: usize = 0;
        while (sequence_index < length) : (sequence_index += 1) {
            const flat = batch_index * sequence_length + sequence_index;
            inputs[flat] = random.intRangeAtMost(u32, 1, 999);
            targets[flat] = random.intRangeAtMost(u32, 1, 999);
            var component: usize = 0;
            while (component < 4) : (component += 1) {
                activations[flat * 4 + component] = random.float(f32) - 0.5;
            }
        }
    }
    return .{ .inputs = inputs, .targets = targets, .activations = activations };
}

test "compaction plan removes every padded row and keeps active order" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 6;
    const lengths = [_]usize{ 3, 0, 6, 1 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 99);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 10), plan.active_count);
    try std.testing.expectEqual(@as(usize, 24), plan.padded_count);
    try std.testing.expectEqual(@as(usize, 14), plan.paddedRowsSkipped());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 3, 3, 9, 10 }, plan.row_offsets);
    try std.testing.expectEqualSlices(usize, &lengths, plan.row_lengths);
    try std.testing.expectEqual(@as(usize, 10), plan.compactLengths()[0]);

    var expected_cursor: usize = 0;
    for (lengths, 0..) |length, batch_index| {
        var sequence_index: usize = 0;
        while (sequence_index < length) : (sequence_index += 1) {
            const flat = batch_index * sequence_length + sequence_index;
            try std.testing.expectEqual(flat, plan.source_positions[expected_cursor]);
            try std.testing.expectEqual(reference.inputs[flat], plan.input_tokens[expected_cursor]);
            try std.testing.expectEqual(reference.targets[flat], plan.target_tokens[expected_cursor]);
            expected_cursor += 1;
        }
    }
    try std.testing.expectEqual(plan.active_count, expected_cursor);
}

test "gather then scatter reproduces the padded activation layout exactly" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 5;
    const model_dim: usize = 4;
    const lengths = [_]usize{ 5, 2, 0, 4 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 7);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    const compact = try allocator.alloc(f32, plan.active_count * model_dim);
    defer allocator.free(compact);
    try plan.gather(f32, reference.activations, model_dim, compact);

    const restored = try allocator.alloc(f32, plan.padded_count * model_dim);
    defer allocator.free(restored);
    try plan.scatter(f32, compact, model_dim, restored, 0.0);

    try std.testing.expectEqualSlices(f32, reference.activations, restored);
}

test "padded rows contribute nothing so compact and padded normalizers agree" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 8;
    const model_dim: usize = 3;
    const lengths = [_]usize{ 8, 1, 0, 3, 7 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 4242);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    var padded_valid_tokens: usize = 0;
    for (lengths) |length| padded_valid_tokens += @min(length, sequence_length);
    try std.testing.expectEqual(padded_valid_tokens, plan.active_count);
    try std.testing.expectEqual(padded_valid_tokens * model_dim, try plan.gradientElementDivisor(model_dim));
    try std.testing.expectEqual(@max(padded_valid_tokens, 1), plan.gradientTokenDivisor());

    var padded_sum: f64 = 0.0;
    for (lengths, 0..) |length, batch_index| {
        var sequence_index: usize = 0;
        while (sequence_index < sequence_length) : (sequence_index += 1) {
            if (sequence_index >= length) continue;
            const flat = batch_index * sequence_length + sequence_index;
            var component: usize = 0;
            while (component < model_dim) : (component += 1) {
                padded_sum += reference.activations[flat * 4 + component];
            }
        }
    }

    var compact_sum: f64 = 0.0;
    for (plan.source_positions) |source_row| {
        var component: usize = 0;
        while (component < model_dim) : (component += 1) {
            compact_sum += reference.activations[source_row * 4 + component];
        }
    }

    try std.testing.expectApproxEqAbs(padded_sum, compact_sum, 1e-12);
}

test "compaction of a fully dense batch is an identity mapping" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 4;
    const lengths = [_]usize{ 4, 4, 4 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 11);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    try std.testing.expectEqual(plan.padded_count, plan.active_count);
    try std.testing.expectEqual(@as(usize, 0), plan.paddedRowsSkipped());
    try std.testing.expectEqualSlices(u32, reference.inputs, plan.input_tokens);
    try std.testing.expectEqualSlices(u32, reference.targets, plan.target_tokens);
    for (plan.source_positions, 0..) |position, index| try std.testing.expectEqual(index, position);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), plan.savedFraction(), 1e-12);
}

test "an entirely empty batch compacts to zero active rows" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 3;
    const lengths = [_]usize{ 0, 0 };
    const tokens = [_]u32{0} ** 6;

    var plan = try buildPlan(allocator, &tokens, &tokens, &lengths, sequence_length);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 0), plan.active_count);
    try std.testing.expectEqual(@as(usize, 6), plan.padded_count);
    try std.testing.expectEqual(@as(usize, 1), try plan.gradientElementDivisor(8));
    try std.testing.expectEqual(@as(usize, 1), plan.gradientTokenDivisor());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), plan.savedFraction(), 1e-12);

    const restored = try allocator.alloc(f32, plan.padded_count * 2);
    defer allocator.free(restored);
    try plan.scatter(f32, &[_]f32{}, 2, restored, 0.0);
    for (restored) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "compaction rejects malformed shapes and out of range lengths" {
    const allocator = std.testing.allocator;
    const tokens = [_]u32{ 1, 2, 3, 4 };
    const lengths = [_]usize{ 2, 2 };

    try std.testing.expectError(
        CompactionError.InvalidDimensions,
        buildPlan(allocator, &tokens, &tokens, &lengths, 0),
    );
    try std.testing.expectError(
        CompactionError.InvalidDimensions,
        buildPlan(allocator, &tokens, &tokens, &[_]usize{}, 2),
    );
    try std.testing.expectError(
        CompactionError.InvalidDimensions,
        buildPlan(allocator, &tokens, &tokens, &lengths, 3),
    );
    try std.testing.expectError(
        CompactionError.LengthOutOfRange,
        buildPlan(allocator, &tokens, &tokens, &[_]usize{ 3, 1 }, 2),
    );
}

test "compaction arithmetic rejects overflowing shapes" {
    const allocator = std.testing.allocator;
    const huge = std.math.maxInt(usize) / 2 + 1;
    const lengths = [_]usize{ 1, 1 };
    const tokens = [_]u32{ 1, 2 };

    try std.testing.expectError(
        CompactionError.SizeOverflow,
        buildPlan(allocator, &tokens, &tokens, &lengths, huge),
    );

    var plan = try buildPlan(allocator, &tokens, &tokens, &lengths, 1);
    defer plan.deinit();
    try std.testing.expectError(
        CompactionError.SizeOverflow,
        plan.gradientElementDivisor(std.math.maxInt(usize)),
    );
    try std.testing.expectError(
        CompactionError.InvalidDimensions,
        plan.gradientElementDivisor(0),
    );
}

test "compaction cleans up on every allocation failure" {
    const lengths = [_]usize{ 2, 1 };
    const tokens = [_]u32{ 5, 6, 7, 8 };

    var failure_index: usize = 0;
    while (failure_index < 5) : (failure_index += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = failure_index });
        const result = buildPlan(failing.allocator(), &tokens, &tokens, &lengths, 2);
        if (result) |value| {
            var owned = value;
            owned.deinit();
        } else |err| {
            try std.testing.expectEqual(CompactionError.AllocationFailed, err);
        }
    }
}

test "gather and scatter validate their buffer shapes" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{ 2, 1 };
    const tokens = [_]u32{ 5, 6, 7, 8 };

    var plan = try buildPlan(allocator, &tokens, &tokens, &lengths, 2);
    defer plan.deinit();

    const padded = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var compact: [6]f32 = undefined;
    try plan.gather(f32, &padded, 2, &compact);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, compact[0..6]);

    var too_small: [4]f32 = undefined;
    try std.testing.expectError(CompactionError.BufferTooSmall, plan.gather(f32, &padded, 2, &too_small));
    try std.testing.expectError(CompactionError.InvalidDimensions, plan.gather(f32, padded[0..6], 2, &compact));
    try std.testing.expectError(CompactionError.InvalidDimensions, plan.gather(f32, &padded, 0, &compact));

    var restored: [8]f32 = undefined;
    try plan.scatter(f32, compact[0..6], 2, &restored, -1.0);
    try std.testing.expectEqualSlices(
        f32,
        &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -1.0 },
        &restored,
    );
    try std.testing.expectError(CompactionError.InvalidDimensions, plan.scatter(f32, compact[0..4], 2, &restored, 0.0));
}

test "half precision payloads survive the compact round trip bit exactly" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 4;
    const model_dim: usize = 2;
    const lengths = [_]usize{ 3, 1 };
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };

    var plan = try buildPlan(allocator, &tokens, &tokens, &lengths, sequence_length);
    defer plan.deinit();

    var padded: [8 * model_dim]f16 = undefined;
    @memset(&padded, 0.0);
    for (plan.source_positions, 0..) |row, index| {
        padded[row * model_dim] = @floatCast(@as(f32, @floatFromInt(index)) * 0.5);
        padded[row * model_dim + 1] = @floatCast(@as(f32, @floatFromInt(index)) * -0.25);
    }

    var compact: [4 * model_dim]f16 = undefined;
    try plan.gather(f16, &padded, model_dim, &compact);
    var restored: [8 * model_dim]f16 = undefined;
    try plan.scatter(f16, compact[0 .. plan.active_count * model_dim], model_dim, &restored, 0.0);
    try std.testing.expectEqualSlices(f16, &padded, &restored);
}

test "compact fused-backward normalizers match the padded futhark semantics" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 7;
    const model_dim: usize = 5;
    const lengths = [_]usize{ 7, 0, 2, 6, 1, 0, 4 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 20260818);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    const padded_valid = try futharkValidTokens(&lengths, sequence_length);
    const compact_valid = try futharkValidTokens(plan.compactLengths(), plan.compactSequenceLength());
    try std.testing.expectEqual(padded_valid, compact_valid);
    try std.testing.expectEqual(
        try futharkElementDivisor(padded_valid, model_dim),
        try futharkElementDivisor(compact_valid, model_dim),
    );
    try std.testing.expectEqual(try futharkElementDivisor(padded_valid, model_dim), try plan.gradientElementDivisor(model_dim));
    try std.testing.expectEqual(@max(padded_valid, 1), plan.gradientTokenDivisor());

    const element_divisor: f64 = @floatFromInt(try futharkElementDivisor(padded_valid, model_dim));

    var padded_loss: f64 = 0.0;
    for (lengths, 0..) |length, batch_index| {
        const limit = @min(length, sequence_length);
        var sequence_index: usize = 0;
        while (sequence_index < sequence_length) : (sequence_index += 1) {
            const flat = batch_index * sequence_length + sequence_index;
            const active = sequence_index < limit;
            var component: usize = 0;
            while (component < model_dim) : (component += 1) {
                const value: f64 = if (active) reference.activations[flat * 4 + (component % 4)] else 0.0;
                padded_loss += 2.0 * value / element_divisor;
            }
        }
    }

    var compact_loss: f64 = 0.0;
    for (plan.source_positions) |source_row| {
        var component: usize = 0;
        while (component < model_dim) : (component += 1) {
            const value: f64 = reference.activations[source_row * 4 + (component % 4)];
            compact_loss += 2.0 * value / element_divisor;
        }
    }

    try std.testing.expectApproxEqAbs(padded_loss, compact_loss, 1e-12);
}

test "compact embedding gradient histogram equals the padded histogram" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 6;
    const model_dim: usize = 3;
    const vocab_size: usize = 64;
    const lengths = [_]usize{ 6, 3, 0, 5, 0, 2 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 555);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    for (reference.inputs) |*token| token.* %= @as(u32, @intCast(vocab_size));

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    const padded_histogram = try allocator.alloc(f64, vocab_size * model_dim);
    defer allocator.free(padded_histogram);
    const compact_histogram = try allocator.alloc(f64, vocab_size * model_dim);
    defer allocator.free(compact_histogram);
    @memset(padded_histogram, 0.0);
    @memset(compact_histogram, 0.0);

    for (lengths, 0..) |length, batch_index| {
        const limit = @min(length, sequence_length);
        var sequence_index: usize = 0;
        while (sequence_index < sequence_length) : (sequence_index += 1) {
            if (sequence_index >= limit) continue;
            const flat = batch_index * sequence_length + sequence_index;
            const token: usize = @intCast(reference.inputs[flat]);
            var component: usize = 0;
            while (component < model_dim) : (component += 1) {
                padded_histogram[token * model_dim + component] += reference.activations[flat * 4 + (component % 4)];
            }
        }
    }

    for (plan.source_positions, 0..) |source_row, compact_row| {
        const token: usize = @intCast(plan.input_tokens[compact_row]);
        var component: usize = 0;
        while (component < model_dim) : (component += 1) {
            compact_histogram[token * model_dim + component] += reference.activations[source_row * 4 + (component % 4)];
        }
    }

    for (padded_histogram, compact_histogram) |padded_value, compact_value| {
        try std.testing.expectApproxEqAbs(padded_value, compact_value, 1e-12);
    }
}

test "compact rows carry every real token exactly once with no padding leakage" {
    const allocator = std.testing.allocator;
    const sequence_length: usize = 9;
    const lengths = [_]usize{ 0, 9, 4, 0, 8, 1 };

    const reference = try buildReference(allocator, lengths.len, sequence_length, &lengths, 31337);
    defer allocator.free(reference.inputs);
    defer allocator.free(reference.targets);
    defer allocator.free(reference.activations);

    var plan = try buildPlan(allocator, reference.inputs, reference.targets, &lengths, sequence_length);
    defer plan.deinit();

    const visited = try allocator.alloc(bool, plan.padded_count);
    defer allocator.free(visited);
    @memset(visited, false);
    for (plan.source_positions) |position| {
        try std.testing.expect(position < plan.padded_count);
        try std.testing.expect(!visited[position]);
        visited[position] = true;
    }

    for (lengths, 0..) |length, batch_index| {
        var sequence_index: usize = 0;
        while (sequence_index < sequence_length) : (sequence_index += 1) {
            const flat = batch_index * sequence_length + sequence_index;
            const should_be_active = sequence_index < length;
            try std.testing.expectEqual(should_be_active, visited[flat]);
        }
    }

    for (0..plan.batch_size) |batch_index| {
        const start = plan.row_offsets[batch_index];
        const end = plan.row_offsets[batch_index + 1];
        try std.testing.expectEqual(plan.row_lengths[batch_index], end - start);
        for (start..end) |compact_row| {
            const position = plan.source_positions[compact_row];
            try std.testing.expectEqual(batch_index, position / sequence_length);
            try std.testing.expectEqual(compact_row - start, position % sequence_length);
        }
    }
}
