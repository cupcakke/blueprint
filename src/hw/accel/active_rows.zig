const std = @import("std");

pub const ActiveRowsError = error{
    InvalidDimensions,
    ElementCountOverflow,
    AllocationFailed,
    TokenIndexOutOfRange,
};

pub const oftb_scale: f32 = 0.7071067811865476;

pub fn checkedMul(comptime T: type, a: T, b: T) ActiveRowsError!T {
    return std.math.mul(T, a, b) catch return ActiveRowsError.ElementCountOverflow;
}

pub fn checkedAdd(comptime T: type, a: T, b: T) ActiveRowsError!T {
    return std.math.add(T, a, b) catch return ActiveRowsError.ElementCountOverflow;
}

pub const ActiveBatch = struct {
    allocator: std.mem.Allocator,
    compact_input_tokens: []u32,
    compact_target_tokens: []u32,
    row_batch_index: []u32,
    row_seq_index: []u32,
    sequence_lengths: []usize,
    batch_size: usize,
    padded_seq_len: usize,
    active_rows: usize,

    pub fn paddedRows(self: *const ActiveBatch) ActiveRowsError!usize {
        return checkedMul(usize, self.batch_size, self.padded_seq_len);
    }

    pub fn activeRatio(self: *const ActiveBatch) f64 {
        const padded = self.paddedRows() catch return 0.0;
        if (padded == 0) return 0.0;
        return @as(f64, @floatFromInt(self.active_rows)) / @as(f64, @floatFromInt(padded));
    }

    pub fn writeCompactLengths(self: *const ActiveBatch, out: []i64) ActiveRowsError!void {
        if (out.len != self.active_rows) return ActiveRowsError.InvalidDimensions;
        for (out) |*value| value.* = 1;
    }

    pub fn deinit(self: *ActiveBatch) void {
        self.allocator.free(self.compact_input_tokens);
        self.allocator.free(self.compact_target_tokens);
        self.allocator.free(self.row_batch_index);
        self.allocator.free(self.row_seq_index);
        self.allocator.free(self.sequence_lengths);
        self.* = undefined;
    }
};

pub fn buildActiveBatch(
    allocator: std.mem.Allocator,
    flat_input_tokens: []const u32,
    flat_target_tokens: []const u32,
    sequence_lengths: []const usize,
    padded_seq_len: usize,
) ActiveRowsError!ActiveBatch {
    if (sequence_lengths.len == 0) return ActiveRowsError.InvalidDimensions;
    if (padded_seq_len == 0) return ActiveRowsError.InvalidDimensions;
    const expected = try checkedMul(usize, sequence_lengths.len, padded_seq_len);
    if (flat_input_tokens.len != expected or flat_target_tokens.len != expected) return ActiveRowsError.InvalidDimensions;

    var total_active: usize = 0;
    for (sequence_lengths) |length| {
        if (length > padded_seq_len) return ActiveRowsError.InvalidDimensions;
        total_active = try checkedAdd(usize, total_active, length);
    }

    const compact_input = allocator.alloc(u32, total_active) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(compact_input);
    const compact_target = allocator.alloc(u32, total_active) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(compact_target);
    const row_batch = allocator.alloc(u32, total_active) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(row_batch);
    const row_seq = allocator.alloc(u32, total_active) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(row_seq);
    const lengths_copy = allocator.alloc(usize, sequence_lengths.len) catch return ActiveRowsError.AllocationFailed;
    @memcpy(lengths_copy, sequence_lengths);

    var write: usize = 0;
    for (sequence_lengths, 0..) |length, batch_index| {
        var seq_index: usize = 0;
        while (seq_index < length) : (seq_index += 1) {
            const padded_index = batch_index * padded_seq_len + seq_index;
            compact_input[write] = flat_input_tokens[padded_index];
            compact_target[write] = flat_target_tokens[padded_index];
            row_batch[write] = @intCast(batch_index);
            row_seq[write] = @intCast(seq_index);
            write += 1;
        }
    }

    return ActiveBatch{
        .allocator = allocator,
        .compact_input_tokens = compact_input,
        .compact_target_tokens = compact_target,
        .row_batch_index = row_batch,
        .row_seq_index = row_seq,
        .sequence_lengths = lengths_copy,
        .batch_size = sequence_lengths.len,
        .padded_seq_len = padded_seq_len,
        .active_rows = total_active,
    };
}

pub fn scatterRowsToPadded(
    allocator: std.mem.Allocator,
    compact: []const f32,
    row_batch_index: []const u32,
    row_seq_index: []const u32,
    batch_size: usize,
    padded_seq_len: usize,
    row_width: usize,
) ActiveRowsError![]f32 {
    const padded_rows = try checkedMul(usize, batch_size, padded_seq_len);
    const total = try checkedMul(usize, padded_rows, row_width);
    const out = allocator.alloc(f32, total) catch return ActiveRowsError.AllocationFailed;
    @memset(out, 0.0);
    if (compact.len != try checkedMul(usize, row_batch_index.len, row_width)) return ActiveRowsError.InvalidDimensions;
    for (row_batch_index, row_seq_index, 0..) |batch_index, seq_index, row| {
        const dst_row = @as(usize, batch_index) * padded_seq_len + seq_index;
        const src_offset = row * row_width;
        const dst_offset = dst_row * row_width;
        @memcpy(out[dst_offset .. dst_offset + row_width], compact[src_offset .. src_offset + row_width]);
    }
    return out;
}

pub const ReferenceWeights = struct {
    s: []const f32,
    t: []const f32,
    half: usize,

    pub fn sWeight(self: ReferenceWeights, d: usize, j: usize) f32 {
        return self.s[d * (self.half + 1) + j];
    }

    pub fn tWeight(self: ReferenceWeights, d: usize, j: usize) f32 {
        return self.t[d * (self.half + 1) + j];
    }
};

fn clampLinear(v: f32, lo: f32, hi: f32) f32 {
    return @max(lo, @min(hi, v));
}

fn clampF16Value(v: f32) f32 {
    if (std.math.isNan(v) or std.math.isInf(v)) return 0.0;
    return clampLinear(v, -65504.0, 65504.0);
}

fn safeAccum(v: f32) f32 {
    if (std.math.isNan(v) or std.math.isInf(v)) return 0.0;
    return v;
}

pub fn couplingRow(
    row: []const f32,
    weights: ReferenceWeights,
    clip_min: f32,
    clip_max: f32,
    out: []f32,
) ActiveRowsError!void {
    const half = weights.half;
    if (row.len != half * 2 or out.len != half * 2) return ActiveRowsError.InvalidDimensions;
    const x1 = row[0..half];
    const x2 = row[half .. half * 2];
    var y1_buffer: [64]f32 = undefined;
    const yr1: []f32 = if (half <= y1_buffer.len) y1_buffer[0..half] else return ActiveRowsError.InvalidDimensions;
    var d: usize = 0;
    while (d < half) : (d += 1) {
        var sum: f32 = weights.sWeight(d, half);
        var j: usize = 0;
        while (j < half) : (j += 1) sum += weights.sWeight(d, j) * x2[j];
        const clipped = clampLinear(sum, clip_min, clip_max);
        yr1[d] = x1[d] * @exp(clipped);
    }
    var y2_buffer: [64]f32 = undefined;
    const y2: []f32 = if (half <= y2_buffer.len) y2_buffer[0..half] else return ActiveRowsError.InvalidDimensions;
    var j2: usize = 0;
    while (j2 < half) : (j2 += 1) {
        var trans: f32 = weights.tWeight(j2, half);
        var k: usize = 0;
        while (k < half) : (k += 1) trans += weights.tWeight(j2, k) * yr1[k];
        y2[j2] = x2[j2] + safeAccum(trans);
    }
    d = 0;
    while (d < half) : (d += 1) {
        out[d] = clampF16Value((yr1[d] - y2[d]) * oftb_scale);
        out[half + d] = clampF16Value((yr1[d] + y2[d]) * oftb_scale);
    }
}

pub fn invertRow(
    row: []const f32,
    weights: ReferenceWeights,
    clip_min: f32,
    clip_max: f32,
    out: []f32,
) ActiveRowsError!void {
    const half = weights.half;
    if (row.len != half * 2 or out.len != half * 2) return ActiveRowsError.InvalidDimensions;
    const y1p = row[0..half];
    const y2p = row[half .. half * 2];
    var uv1_buffer: [64]f32 = undefined;
    var uv2_buffer: [64]f32 = undefined;
    if (half > uv1_buffer.len) return ActiveRowsError.InvalidDimensions;
    const uv1: []f32 = uv1_buffer[0..half];
    const uv2: []f32 = uv2_buffer[0..half];
    for (0..half) |i| {
        uv1[i] = (y1p[i] + y2p[i]) * oftb_scale;
        uv2[i] = (y2p[i] - y1p[i]) * oftb_scale;
    }
    var x2_buffer: [64]f32 = undefined;
    const x2: []f32 = x2_buffer[0..half];
    for (0..half) |d| {
        var trans: f32 = weights.tWeight(d, half);
        for (0..half) |k| trans += weights.tWeight(d, k) * uv1[k];
        x2[d] = uv2[d] - safeAccum(trans);
    }
    for (0..half) |d| {
        var pre: f32 = weights.sWeight(d, half);
        for (0..half) |k| pre += weights.sWeight(d, k) * x2[k];
        const clipped = clampLinear(pre, clip_min, clip_max);
        out[d] = uv1[d] / @exp(clipped);
        out[half + d] = x2[d];
    }
}

pub const ReferenceModelConfig = struct {
    half: usize,
    num_layers: usize,
    clip_min: f32 = -5.0,
    clip_max: f32 = 5.0,
    grad_mean: bool = true,
    gradient_scale: f32 = 1.0,
    reconstruction_alpha: f32 = 0.3,
    forward_scale: f32 = 1.0,
    logdet_weight: f32 = -1e-3,
};

pub const ReferenceStepResult = struct {
    loss: f32,
    reconstruction_loss: f32,
    logdet_mean: f32,
    grad_s: []f32,
    grad_t: []f32,
    input_delta: []f32,
    reconstructed: []f32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ReferenceStepResult) void {
        self.allocator.free(self.grad_s);
        self.allocator.free(self.grad_t);
        self.allocator.free(self.input_delta);
        self.allocator.free(self.reconstructed);
        self.* = undefined;
    }
};

pub const ReferenceStepInputs = struct {
    inputs: []const f32,
    targets: []const f32,
    rows: usize,
    row_width: usize,
    lengths: []const usize,
    padded_seq_len: usize,
    s_stacks: []const f32,
    t_stacks: []const f32,
};

pub fn referenceTrainingStep(
    allocator: std.mem.Allocator,
    config: ReferenceModelConfig,
    inputs: ReferenceStepInputs,
) ActiveRowsError!ReferenceStepResult {
    const half = config.half;
    const d2 = half * 2;
    const per_layer = try checkedMul(usize, half, half + 1);
    const stack_total = try checkedMul(usize, per_layer, config.num_layers);
    if (inputs.s_stacks.len != stack_total or inputs.t_stacks.len != stack_total) return ActiveRowsError.InvalidDimensions;
    if (inputs.padded_seq_len == 0) return ActiveRowsError.InvalidDimensions;

    var valid_tokens: usize = 0;
    for (inputs.lengths) |length| {
        if (length > inputs.padded_seq_len) return ActiveRowsError.InvalidDimensions;
        valid_tokens = try checkedAdd(usize, valid_tokens, length);
    }
    const count_elements: usize = if (valid_tokens > 0) try checkedMul(usize, valid_tokens, d2) else 1;
    const count_elements_f32: f32 = @floatFromInt(count_elements);
    const count_tokens_f32: f32 = @max(1.0, @as(f32, @floatFromInt(valid_tokens)));
    const gradient_element_divisor: f32 = if (config.grad_mean) count_elements_f32 else 1.0;
    const gradient_token_divisor: f32 = if (config.grad_mean) count_tokens_f32 else 1.0;

    const active_tokens = allocator.alloc(usize, valid_tokens) catch return ActiveRowsError.AllocationFailed;
    defer allocator.free(active_tokens);
    {
        var write: usize = 0;
        for (inputs.lengths, 0..) |length, batch_index| {
            var j: usize = 0;
            while (j < length) : (j += 1) {
                active_tokens[write] = try checkedAdd(usize, try checkedMul(usize, batch_index, inputs.padded_seq_len), j);
                write += 1;
            }
        }
    }

    const active_count = active_tokens.len;
    const y = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    defer allocator.free(y);
    const g = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    defer allocator.free(g);
    const orig = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    defer allocator.free(orig);
    const x_recon = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(x_recon);

    var row_buffer_in: [128]f32 = undefined;
    var row_buffer_out: [128]f32 = undefined;
    if (d2 > row_buffer_in.len) return ActiveRowsError.InvalidDimensions;

    for (active_tokens, 0..) |token_index, row| {
        const src_offset = token_index * d2;
        @memcpy(orig[row * d2 .. row * d2 + d2], inputs.inputs[src_offset .. src_offset + d2]);
        @memcpy(row_buffer_in[0..d2], inputs.inputs[src_offset .. src_offset + d2]);
        var layer: usize = 0;
        while (layer < config.num_layers) : (layer += 1) {
            const weights = ReferenceWeights{ .s = inputs.s_stacks[layer * per_layer .. (layer + 1) * per_layer], .t = inputs.t_stacks[layer * per_layer .. (layer + 1) * per_layer], .half = half };
            try couplingRow(row_buffer_in[0..d2], weights, config.clip_min, config.clip_max, row_buffer_out[0..d2]);
            @memcpy(row_buffer_in[0..d2], row_buffer_out[0..d2]);
        }
        @memcpy(y[row * d2 .. row * d2 + d2], row_buffer_in[0..d2]);
    }

    const final_out = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    defer allocator.free(final_out);
    @memcpy(final_out, y);

    for (active_tokens, 0..) |token_index, row| {
        const target_offset = token_index * d2;
        var i: usize = 0;
        while (i < d2) : (i += 1) {
            const diff = safeAccum(final_out[row * d2 + i] - inputs.targets[target_offset + i]);
            const clamped = clampLinear(diff, -100.0, 100.0);
            g[row * d2 + i] = 2.0 * clamped / gradient_element_divisor;
        }
    }

    const grad_s = allocator.alloc(f32, stack_total) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(grad_s);
    const grad_t = allocator.alloc(f32, stack_total) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(grad_t);
    const delta = allocator.alloc(f32, try checkedMul(usize, active_count, d2)) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(delta);
    @memset(grad_s, 0.0);
    @memset(grad_t, 0.0);

    var ld_total: f32 = 0.0;
    const ld_shift = config.logdet_weight / gradient_token_divisor;

    var layer_index: usize = config.num_layers;
    while (layer_index > 0) {
        layer_index -= 1;
        const layer = layer_index;
        const weights = ReferenceWeights{ .s = inputs.s_stacks[layer * per_layer .. (layer + 1) * per_layer], .t = inputs.t_stacks[layer * per_layer .. (layer + 1) * per_layer], .half = half };
        const grad_s_base = grad_s[layer * per_layer .. (layer + 1) * per_layer];
        const grad_t_base = grad_t[layer * per_layer .. (layer + 1) * per_layer];

        var row: usize = 0;
        while (row < active_count) : (row += 1) {
            const y_row = y[row * d2 .. row * d2 + d2];
            const g_row = g[row * d2 .. row * d2 + d2];
            var ur1: [32]f32 = undefined;
            var ur2: [32]f32 = undefined;
            var h1: [32]f32 = undefined;
            var h2: [32]f32 = undefined;
            var x2: [32]f32 = undefined;
            var pre_scale: [32]f32 = undefined;
            var ds: [32]f32 = undefined;
            var dx1: [32]f32 = undefined;
            var dx2: [32]f32 = undefined;
            if (half > ur1.len) return ActiveRowsError.InvalidDimensions;
            for (0..half) |i| {
                ur1[i] = (y_row[i] + y_row[half + i]) * oftb_scale;
                ur2[i] = (y_row[half + i] - y_row[i]) * oftb_scale;
                h1[i] = (g_row[i] + g_row[half + i]) * oftb_scale;
                h2[i] = (g_row[half + i] - g_row[i]) * oftb_scale;
            }
            for (0..half) |d| {
                var trans: f32 = weights.tWeight(d, half);
                for (0..half) |k| trans += weights.tWeight(d, k) * ur1[k];
                x2[d] = ur2[d] - safeAccum(trans);
            }
            for (0..half) |d| {
                var pre: f32 = weights.sWeight(d, half);
                for (0..half) |k| pre += weights.sWeight(d, k) * x2[k];
                pre_scale[d] = pre;
            }
            for (0..half) |d| {
                const clipped = clampLinear(pre_scale[d], config.clip_min, config.clip_max);
                const scale = @exp(clipped);
                const dy1_total = blk: {
                    var total: f32 = h1[d];
                    for (0..half) |k| total += h2[k] * weights.tWeight(k, d);
                    break :blk total;
                };
                dx1[d] = dy1_total * scale;
                ds[d] = if (pre_scale[d] >= config.clip_min and pre_scale[d] <= config.clip_max) dy1_total * ur1[d] + ld_shift else 0.0;
                ld_total += clipped;
            }
            for (0..half) |j| {
                var total: f32 = h2[j];
                for (0..half) |k| total += ds[k] * weights.sWeight(k, j);
                dx2[j] = total;
            }
            for (0..half) |d| {
                grad_s_base[d * (half + 1) + half] += ds[d];
                grad_t_base[d * (half + 1) + half] += h2[d];
                for (0..half) |j| {
                    grad_s_base[d * (half + 1) + j] += ds[d] * x2[j];
                    grad_t_base[d * (half + 1) + j] += h2[d] * ur1[j];
                }
                x_recon[row * d2 + d] = ur1[d] / @exp(clampLinear(pre_scale[d], config.clip_min, config.clip_max));
                x_recon[row * d2 + half + d] = x2[d];
            }
            for (0..half) |d| {
                y[row * d2 + d] = ur1[d] / @exp(clampLinear(pre_scale[d], config.clip_min, config.clip_max));
                y[row * d2 + half + d] = x2[d];
                g[row * d2 + d] = dx1[d];
                g[row * d2 + half + d] = dx2[d];
            }
        }
    }

    var loss_total: f32 = 0.0;
    var recon_total: f32 = 0.0;
    for (active_tokens, 0..) |token_index, row| {
        const target_offset = token_index * d2;
        const orig_offset = token_index * d2;
        var i: usize = 0;
        while (i < d2) : (i += 1) {
            const diff = safeAccum(final_out[row * d2 + i] - inputs.targets[target_offset + i]);
            loss_total += diff * diff;
            const rdiff = safeAccum(x_recon[row * d2 + i] - inputs.inputs[orig_offset + i]);
            recon_total += rdiff * rdiff;
        }
    }
    const loss = loss_total / count_elements_f32;
    const recon_loss = recon_total / count_elements_f32;
    const logdet_mean = ld_total / count_tokens_f32;

    for (0..active_count) |row| {
        const orig_offset = active_tokens[row] * d2;
        var i: usize = 0;
        while (i < d2) : (i += 1) {
            const base = config.forward_scale * g[row * d2 + i];
            const diff = safeAccum(x_recon[row * d2 + i] - inputs.inputs[orig_offset + i]);
            const clamped = clampLinear(diff, -100.0, 100.0);
            delta[row * d2 + i] = clampF16Value(base + config.reconstruction_alpha * 2.0 * clamped / gradient_element_divisor);
        }
    }

    const scale = config.gradient_scale;
    for (grad_s) |*value| value.* *= scale;
    for (grad_t) |*value| value.* *= scale;

    return ReferenceStepResult{
        .loss = loss,
        .reconstruction_loss = recon_loss,
        .logdet_mean = logdet_mean,
        .grad_s = grad_s,
        .grad_t = grad_t,
        .input_delta = delta,
        .reconstructed = x_recon,
        .allocator = allocator,
    };
}

pub fn referenceEmbeddingBackward(
    allocator: std.mem.Allocator,
    tokens: []const u32,
    row_grads: []const f32,
    row_width: usize,
    vocab_size: usize,
) ActiveRowsError![]f32 {
    if (row_width == 0 or vocab_size == 0) return ActiveRowsError.InvalidDimensions;
    if (tokens.len * row_width != row_grads.len) return ActiveRowsError.InvalidDimensions;
    const grad_weight = allocator.alloc(f32, try checkedMul(usize, vocab_size, row_width)) catch return ActiveRowsError.AllocationFailed;
    errdefer allocator.free(grad_weight);
    @memset(grad_weight, 0.0);
    for (tokens, 0..) |token, row| {
        if (token >= vocab_size) return ActiveRowsError.TokenIndexOutOfRange;
        for (0..row_width) |c| {
            grad_weight[@as(usize, token) * row_width + c] += row_grads[row * row_width + c];
        }
    }
    return grad_weight;
}

test "active batch construction handles irregular lengths and counts rows exactly" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{ 3, 0, 5, 1, 4 };
    const padded: usize = 5;
    var flat_in: [25]u32 = undefined;
    var flat_tg: [25]u32 = undefined;
    for (0..25) |i| {
        flat_in[i] = @intCast((i * 7 + 1) % 11);
        flat_tg[i] = @intCast((i * 5 + 2) % 11);
    }
    var batch = try buildActiveBatch(allocator, &flat_in, &flat_tg, &lengths, padded);
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 3 + 0 + 5 + 1 + 4), batch.active_rows);
    try std.testing.expectEqual(@as(usize, 25), try batch.paddedRows());
    var write: usize = 0;
    for (lengths, 0..) |length, batch_index| {
        for (0..length) |j| {
            const padded_index = batch_index * padded + j;
            try std.testing.expectEqual(flat_in[padded_index], batch.compact_input_tokens[write]);
            try std.testing.expectEqual(flat_tg[padded_index], batch.compact_target_tokens[write]);
            try std.testing.expectEqual(batch_index, batch.row_batch_index[write]);
            try std.testing.expectEqual(j, batch.row_seq_index[write]);
            write += 1;
        }
    }
}

test "active batch rejects mismatched and oversized inputs" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{2};
    var flat: [4]u32 = undefined;
    try std.testing.expectError(ActiveRowsError.InvalidDimensions, buildActiveBatch(allocator, flat[0..3], &flat, &lengths, 4));
    try std.testing.expectError(ActiveRowsError.InvalidDimensions, buildActiveBatch(allocator, &flat, &flat, &lengths, 2));
    try std.testing.expectError(ActiveRowsError.InvalidDimensions, buildActiveBatch(allocator, &flat, &flat, &lengths, 0));
}

test "scatter back reproduces padded layout with zero padding" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{ 2, 3 };
    const padded: usize = 4;
    const width: usize = 2;
    var flat_in: [8]u32 = undefined;
    var flat_tg: [8]u32 = undefined;
    for (0..8) |i| {
        flat_in[i] = @intCast(i % 5);
        flat_tg[i] = @intCast((i + 1) % 5);
    }
    var batch = try buildActiveBatch(allocator, &flat_in, &flat_tg, &lengths, padded);
    defer batch.deinit();
    const compact_values = try allocator.alloc(f32, batch.active_rows * width);
    defer allocator.free(compact_values);
    for (compact_values, 0..) |*value, index| value.* = @floatFromInt(index + 1);
    const padded_values = try scatterRowsToPadded(allocator, compact_values, batch.row_batch_index, batch.row_seq_index, batch.batch_size, padded, width);
    defer allocator.free(padded_values);
    try std.testing.expectEqual(@as(usize, 2 * 4 * width), padded_values.len);
    for (batch.row_batch_index, batch.row_seq_index, 0..) |bi, sj, row| {
        const dst_row = @as(usize, bi) * padded + sj;
        for (0..width) |c| {
            try std.testing.expectEqual(compact_values[row * width + c], padded_values[dst_row * width + c]);
        }
    }
    var row_filled: [8]bool = undefined;
    @memset(&row_filled, false);
    for (batch.row_batch_index, batch.row_seq_index) |bi, sj| row_filled[@as(usize, bi) * padded + sj] = true;
    for (row_filled, 0..) |filled, index| {
        if (!filled) {
            for (0..width) |c| try std.testing.expectEqual(@as(f32, 0.0), padded_values[index * width + c]);
        }
    }
}

test "reference coupling and invert are mutual inverses" {
    const half: usize = 8;
    const per_layer = half * (half + 1);
    var s: [per_layer]f32 = undefined;
    var t: [per_layer]f32 = undefined;
    for (&s, 0..) |*value, index| value.* = 0.01 * @as(f32, @floatFromInt(index % 7)) - 0.03;
    for (&t, 0..) |*value, index| value.* = 0.008 * @as(f32, @floatFromInt(index % 5)) - 0.02;
    const weights = ReferenceWeights{ .s = &s, .t = &t, .half = half };
    var row: [16]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    for (&row) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.3;
    var forwarded: [16]f32 = undefined;
    var inverted: [16]f32 = undefined;
    try couplingRow(&row, weights, -5.0, 5.0, &forwarded);
    try invertRow(&forwarded, weights, -5.0, 5.0, &inverted);
    for (&row, inverted[0..]) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
}

fn referenceStepFixture(allocator: std.mem.Allocator, lengths: []const usize, padded: usize) !struct { s: []f32, t: []f32, inputs: []f32, targets: []f32, config: ReferenceModelConfig, inputs_shape: ReferenceStepInputs, owned: usize } {
    const half: usize = 4;
    const layers: usize = 2;
    const per_layer = half * (half + 1);
    const stack_total = per_layer * layers;
    const s = try allocator.alloc(f32, stack_total);
    const t = try allocator.alloc(f32, stack_total);
    var rng = std.Random.DefaultPrng.init(0xA11CE);
    for (s) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.02;
    for (t) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.02;
    const batch_size = lengths.len;
    const rows = batch_size * padded;
    const inputs = try allocator.alloc(f32, rows * half * 2);
    const targets = try allocator.alloc(f32, rows * half * 2);
    for (inputs) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.4;
    for (targets) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.4;
    const config = ReferenceModelConfig{ .half = half, .num_layers = layers };
    const shape = ReferenceStepInputs{
        .inputs = inputs,
        .targets = targets,
        .rows = rows,
        .row_width = half * 2,
        .lengths = lengths,
        .padded_seq_len = padded,
        .s_stacks = s,
        .t_stacks = t,
    };
    return .{ .s = s, .t = t, .inputs = inputs, .targets = targets, .config = config, .inputs_shape = shape, .owned = 0 };
}

test "compact and padded reference steps are numerically identical" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{ 3, 0, 5, 2 };
    const padded: usize = 5;
    const fixture = try referenceStepFixture(allocator, &lengths, padded);
    defer {
        allocator.free(fixture.s);
        allocator.free(fixture.t);
        allocator.free(fixture.inputs);
        allocator.free(fixture.targets);
    }
    var padded_result = try referenceTrainingStep(allocator, fixture.config, fixture.inputs_shape);
    defer padded_result.deinit();

    var flat_tokens_in: [20]u32 = undefined;
    var flat_tokens_tg: [20]u32 = undefined;
    for (0..20) |i| {
        flat_tokens_in[i] = @intCast(i % 6);
        flat_tokens_tg[i] = @intCast((i + 3) % 6);
    }
    var batch = try buildActiveBatch(allocator, &flat_tokens_in, &flat_tokens_tg, &lengths, padded);
    defer batch.deinit();
    const half: usize = fixture.config.half;
    const d2 = half * 2;
    const compact_inputs = try allocator.alloc(f32, batch.active_rows * d2);
    defer allocator.free(compact_inputs);
    const compact_targets = try allocator.alloc(f32, batch.active_rows * d2);
    defer allocator.free(compact_targets);
    for (batch.row_batch_index, batch.row_seq_index, 0..) |bi, sj, row| {
        const src = (@as(usize, bi) * padded + sj) * d2;
        @memcpy(compact_inputs[row * d2 .. row * d2 + d2], fixture.inputs[src .. src + d2]);
        @memcpy(compact_targets[row * d2 .. row * d2 + d2], fixture.targets[src .. src + d2]);
    }
    const ones = try allocator.alloc(usize, batch.active_rows);
    defer allocator.free(ones);
    @memset(ones, 1);
    var compact_result = try referenceTrainingStep(allocator, fixture.config, .{
        .inputs = compact_inputs,
        .targets = compact_targets,
        .rows = batch.active_rows,
        .row_width = d2,
        .lengths = ones,
        .padded_seq_len = 1,
        .s_stacks = fixture.inputs_shape.s_stacks,
        .t_stacks = fixture.inputs_shape.t_stacks,
    });
    defer compact_result.deinit();

    try std.testing.expectApproxEqAbs(padded_result.loss, compact_result.loss, 1e-6);
    try std.testing.expectApproxEqAbs(padded_result.reconstruction_loss, compact_result.reconstruction_loss, 1e-6);
    try std.testing.expectApproxEqAbs(padded_result.logdet_mean, compact_result.logdet_mean, 1e-6);
    try std.testing.expectEqualSlices(f32, padded_result.grad_s, compact_result.grad_s);
    try std.testing.expectEqualSlices(f32, padded_result.grad_t, compact_result.grad_t);
    for (padded_result.input_delta, compact_result.input_delta) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    }
}

test "padding rows contribute zero embedding gradient in reference backward" {
    const allocator = std.testing.allocator;
    const vocab: usize = 7;
    const width: usize = 3;
    const tokens = [_]u32{ 1, 2, 1, 4 };
    const grads = [_]f32{ 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4 };
    const grad_weight = try referenceEmbeddingBackward(allocator, &tokens, &grads, width, vocab);
    defer allocator.free(grad_weight);
    try std.testing.expectEqual(@as(f32, 4.0), grad_weight[1 * width + 0]);
    try std.testing.expectEqual(@as(f32, 2.0), grad_weight[2 * width + 0]);
    try std.testing.expectEqual(@as(f32, 4.0), grad_weight[1 * width + 1]);
    try std.testing.expectEqual(@as(f32, 4.0), grad_weight[4 * width + 0]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_weight[0 * width + 0]);
    try std.testing.expectEqual(@as(f32, 0.0), grad_weight[5 * width + 0]);
}

test "finite difference gradients match analytic s and t gradients" {
    const allocator = std.testing.allocator;
    const half: usize = 3;
    const layers: usize = 2;
    const per_layer = half * (half + 1);
    const stack_total = per_layer * layers;
    var s: [24]f32 = undefined;
    _ = &s;
    const s_storage = try allocator.alloc(f32, stack_total);
    defer allocator.free(s_storage);
    const t_storage = try allocator.alloc(f32, stack_total);
    defer allocator.free(t_storage);
    var rng = std.Random.DefaultPrng.init(0xFD);
    for (s_storage) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.02;
    for (t_storage) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.02;

    const lengths = [_]usize{2, 3};
    const padded: usize = 3;
    const rows = 6;
    const d2 = half * 2;
    const inputs = try allocator.alloc(f32, rows * d2);
    defer allocator.free(inputs);
    const targets = try allocator.alloc(f32, rows * d2);
    defer allocator.free(targets);
    for (inputs) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.3;
    for (targets) |*value| value.* = (rng.random().float(f32) - 0.5) * 0.3;

    const config = ReferenceModelConfig{ .half = half, .num_layers = layers, .logdet_weight = 0.0 };
    const shape = ReferenceStepInputs{
        .inputs = inputs,
        .targets = targets,
        .rows = rows,
        .row_width = d2,
        .lengths = &lengths,
        .padded_seq_len = padded,
        .s_stacks = s_storage,
        .t_stacks = t_storage,
    };
    var analytic = try referenceTrainingStep(allocator, config, shape);
    defer analytic.deinit();

    const eps: f32 = 1e-3;
    const probe_indices = [_]usize{ 0, 4, half, per_layer + 2, per_layer + half, stack_total - 1 };
    for (probe_indices) |index| {
        const original = s_storage[index];
        s_storage[index] = original + eps;
        var plus = try referenceTrainingStep(allocator, config, shape);
        defer plus.deinit();
        s_storage[index] = original - eps;
        var minus = try referenceTrainingStep(allocator, config, shape);
        defer minus.deinit();
        s_storage[index] = original;
        const numerical = (plus.loss - minus.loss) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(numerical, analytic.grad_s[index], 2e-3);
    }
    const t_probe_indices = [_]usize{ 1, 5, half + 1, per_layer + 3, 2 * per_layer - 2, stack_total - 1 };
    for (t_probe_indices) |index| {
        const original = t_storage[index];
        t_storage[index] = original + eps;
        var plus = try referenceTrainingStep(allocator, config, shape);
        defer plus.deinit();
        t_storage[index] = original - eps;
        var minus = try referenceTrainingStep(allocator, config, shape);
        defer minus.deinit();
        t_storage[index] = original;
        const numerical = (plus.loss - minus.loss) / (2.0 * eps);
        try std.testing.expectApproxEqAbs(numerical, analytic.grad_t[index], 2e-3);
    }
}

test "repeated tokens and one token sequences accumulate deterministically" {
    const allocator = std.testing.allocator;
    const lengths = [_]usize{ 1, 4 };
    const padded: usize = 4;
    var flat_in: [8]u32 = undefined;
    var flat_tg: [8]u32 = undefined;
    for (0..8) |i| {
        flat_in[i] = @intCast(i % 2);
        flat_tg[i] = @intCast(1 - i % 2);
    }
    flat_in[4] = 3;
    flat_in[5] = 3;
    flat_in[6] = 3;
    flat_in[7] = 3;
    var batch = try buildActiveBatch(allocator, &flat_in, &flat_tg, &lengths, padded);
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 5), batch.active_rows);
    const grad_weight = try referenceEmbeddingBackward(allocator, batch.compact_input_tokens, &[_]f32{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, 2, 4);
    defer allocator.free(grad_weight);
    try std.testing.expectEqual(@as(f32, 4.0), grad_weight[3 * 2 + 0]);
    try std.testing.expectEqual(@as(f32, 1.0), grad_weight[0 * 2 + 0]);
}
