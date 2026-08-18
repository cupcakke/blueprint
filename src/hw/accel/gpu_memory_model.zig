const std = @import("std");

pub const EstimationError = error{
    InvalidDimensions,
    ElementCountOverflow,
    ByteCountOverflow,
};

pub const bytes_per_f16: u64 = 2;
pub const bytes_per_f32: u64 = 4;
pub const bytes_per_i64: u64 = 8;
pub const bytes_per_u64: u64 = 8;

pub fn checkedMul(comptime T: type, a: T, b: T) EstimationError!T {
    return std.math.mul(T, a, b) catch return EstimationError.ElementCountOverflow;
}

pub fn checkedAdd(comptime T: type, a: T, b: T) EstimationError!T {
    return std.math.add(T, a, b) catch return EstimationError.ByteCountOverflow;
}

pub fn checkedBytes(elements: u64, element_bytes: u64) EstimationError!u64 {
    return checkedMul(u64, elements, element_bytes);
}

pub const RsfLayout = struct {
    model_dim: usize,
    num_layers: usize,

    pub fn half(self: RsfLayout) EstimationError!usize {
        if (self.model_dim == 0 or self.model_dim % 2 != 0) return EstimationError.InvalidDimensions;
        return self.model_dim / 2;
    }

    pub fn columns(self: RsfLayout) EstimationError!usize {
        const h = try self.half();
        return checkedAdd(usize, h, 1) catch return EstimationError.InvalidDimensions;
    }

    pub fn perLayerElements(self: RsfLayout) EstimationError!usize {
        const h = try self.half();
        const c = try self.columns();
        if (h == 0 or c == 0) return EstimationError.InvalidDimensions;
        return checkedMul(usize, h, c);
    }

    pub fn stackElements(self: RsfLayout) EstimationError!u64 {
        if (self.num_layers == 0) return EstimationError.InvalidDimensions;
        const per_layer: u64 = @intCast(try self.perLayerElements());
        const layers: u64 = @intCast(self.num_layers);
        return checkedMul(u64, per_layer, layers);
    }

    pub fn stackBytesFp32(self: RsfLayout) EstimationError!u64 {
        return checkedBytes(try self.stackElements(), bytes_per_f32);
    }

    pub fn stackBytesFp16(self: RsfLayout) EstimationError!u64 {
        return checkedBytes(try self.stackElements(), bytes_per_f16);
    }
};

pub const EmbeddingLayout = struct {
    vocab_size: usize,
    model_dim: usize,

    pub fn elements(self: EmbeddingLayout) EstimationError!u64 {
        if (self.vocab_size == 0 or self.model_dim == 0) return EstimationError.InvalidDimensions;
        return checkedMul(u64, @intCast(self.vocab_size), @intCast(self.model_dim));
    }

    pub fn bytesFp32(self: EmbeddingLayout) EstimationError!u64 {
        return checkedBytes(try self.elements(), bytes_per_f32);
    }

    pub fn bytesFp16(self: EmbeddingLayout) EstimationError!u64 {
        return checkedBytes(try self.elements(), bytes_per_f16);
    }
};

pub const BatchLayout = struct {
    batch_size: usize,
    max_seq_len: usize,
    model_dim: usize,

    pub fn paddedRows(self: BatchLayout) EstimationError!u64 {
        if (self.batch_size == 0 or self.max_seq_len == 0) return EstimationError.InvalidDimensions;
        return checkedMul(u64, @intCast(self.batch_size), @intCast(self.max_seq_len));
    }

    pub fn paddedBytesFp16(self: BatchLayout) EstimationError!u64 {
        if (self.model_dim == 0) return EstimationError.InvalidDimensions;
        const rows = try self.paddedRows();
        return checkedBytes(try checkedMul(u64, rows, @intCast(self.model_dim)), bytes_per_f16);
    }
};

pub const RsfStateOptions = struct {
    layout: RsfLayout,
    legacy_layer_mirrors: bool = false,
    forward_fp16_stacks: bool = true,
    fp32_master_stacks: bool = true,
    fp32_momentum_stacks: bool = true,
    fp32_fisher_stacks: bool = true,
    transient_fp32_gradients: bool = true,
    replacement_transients: bool = true,
    spectral_startup: bool = false,
};

pub const EmbeddingStateOptions = struct {
    layout: EmbeddingLayout,
    forward_fp16: bool = true,
    fp32_master: bool = true,
    fp32_gradients: bool = true,
    fp32_momentum: bool = true,
    fp32_fisher: bool = true,
    spectral_vectors: bool = true,
    replacement_transients: bool = true,
    frozen_target_fp16: bool = false,
    frozen_target_fp32_master: bool = false,
};

pub const GraphStateOptions = struct {
    enabled: bool = true,
    chunk_hashes: usize = 65536,
    node_fields: u64 = 5,
    edges_per_hash: u64 = 3,
};

pub const FutharkOverhead = struct {
    opaque_tuple_bytes: u64 = 4 * 1024 * 1024,
    device_allocator_slack_fraction: f64 = 0.02,
};

pub const BatchActivationOptions = struct {
    layout: BatchLayout,
    fp16_padded_tensors: u64 = 5,
    fp32_padded_tensors: u64 = 1,
    index_arrays: bool = true,
    compact_rows: bool = false,
    active_rows_override: u64 = 0,
};

pub const MemoryEstimate = struct {
    rsf_forward_fp16_bytes: u64 = 0,
    rsf_master_fp32_bytes: u64 = 0,
    rsf_momentum_fp32_bytes: u64 = 0,
    rsf_fisher_fp32_bytes: u64 = 0,
    rsf_transient_gradient_fp32_bytes: u64 = 0,
    rsf_replacement_transient_bytes: u64 = 0,
    rsf_spectral_transient_bytes: u64 = 0,
    rsf_legacy_mirror_fp16_bytes: u64 = 0,
    embedding_forward_fp16_bytes: u64 = 0,
    embedding_master_fp32_bytes: u64 = 0,
    embedding_gradient_fp32_bytes: u64 = 0,
    embedding_momentum_fp32_bytes: u64 = 0,
    embedding_fisher_fp32_bytes: u64 = 0,
    embedding_spectral_u_bytes: u64 = 0,
    embedding_spectral_v_bytes: u64 = 0,
    embedding_replacement_transient_bytes: u64 = 0,
    frozen_target_fp16_bytes: u64 = 0,
    frozen_target_fp32_master_bytes: u64 = 0,
    batch_activation_bytes: u64 = 0,
    batch_index_bytes: u64 = 0,
    graph_state_bytes: u64 = 0,
    graph_chunk_bytes: u64 = 0,
    nccl_buffer_bytes: u64 = 0,
    futhark_overhead_bytes: u64 = 0,

    pub fn rsfPersistentBytes(self: MemoryEstimate) EstimationError!u64 {
        var total: u64 = 0;
        total = try checkedAdd(u64, total, self.rsf_forward_fp16_bytes);
        total = try checkedAdd(u64, total, self.rsf_master_fp32_bytes);
        total = try checkedAdd(u64, total, self.rsf_momentum_fp32_bytes);
        total = try checkedAdd(u64, total, self.rsf_fisher_fp32_bytes);
        total = try checkedAdd(u64, total, self.rsf_legacy_mirror_fp16_bytes);
        return total;
    }

    pub fn embeddingPersistentBytes(self: MemoryEstimate) EstimationError!u64 {
        var total: u64 = 0;
        total = try checkedAdd(u64, total, self.embedding_forward_fp16_bytes);
        total = try checkedAdd(u64, total, self.embedding_master_fp32_bytes);
        total = try checkedAdd(u64, total, self.embedding_momentum_fp32_bytes);
        total = try checkedAdd(u64, total, self.embedding_fisher_fp32_bytes);
        total = try checkedAdd(u64, total, self.embedding_spectral_u_bytes);
        total = try checkedAdd(u64, total, self.embedding_spectral_v_bytes);
        total = try checkedAdd(u64, total, self.frozen_target_fp16_bytes);
        total = try checkedAdd(u64, total, self.frozen_target_fp32_master_bytes);
        return total;
    }

    pub fn gradientPersistentBytes(self: MemoryEstimate) EstimationError!u64 {
        return self.embedding_gradient_fp32_bytes;
    }

    pub fn activationAndTransientBytes(self: MemoryEstimate) EstimationError!u64 {
        var total: u64 = 0;
        total = try checkedAdd(u64, total, self.rsf_transient_gradient_fp32_bytes);
        total = try checkedAdd(u64, total, self.rsf_replacement_transient_bytes);
        total = try checkedAdd(u64, total, self.rsf_spectral_transient_bytes);
        total = try checkedAdd(u64, total, self.embedding_replacement_transient_bytes);
        total = try checkedAdd(u64, total, self.batch_activation_bytes);
        total = try checkedAdd(u64, total, self.batch_index_bytes);
        total = try checkedAdd(u64, total, self.graph_chunk_bytes);
        total = try checkedAdd(u64, total, self.nccl_buffer_bytes);
        total = try checkedAdd(u64, total, self.futhark_overhead_bytes);
        return total;
    }

    pub fn graphPersistentBytes(self: MemoryEstimate) EstimationError!u64 {
        return self.graph_state_bytes;
    }

    pub fn persistentBytes(self: MemoryEstimate) EstimationError!u64 {
        var total: u64 = 0;
        total = try checkedAdd(u64, total, try self.rsfPersistentBytes());
        total = try checkedAdd(u64, total, try self.embeddingPersistentBytes());
        total = try checkedAdd(u64, total, try self.gradientPersistentBytes());
        total = try checkedAdd(u64, total, try self.graphPersistentBytes());
        return total;
    }

    pub fn peakBytes(self: MemoryEstimate) EstimationError!u64 {
        return checkedAdd(u64, try self.persistentBytes(), try self.activationAndTransientBytes());
    }

    pub fn gib(value: u64) f64 {
        return @as(f64, @floatFromInt(value)) / 1073741824.0;
    }

    pub const Contributor = struct { label: []const u8, bytes: u64 };

    pub fn topContributors(self: *const MemoryEstimate, out: []Contributor) usize {
        var scratch: [24]Contributor = undefined;
        var count: usize = 0;
        const candidates = [_]Contributor{
            .{ .label = "rsf_fp32_master_stacks", .bytes = self.rsf_master_fp32_bytes },
            .{ .label = "rsf_fp32_momentum_stacks", .bytes = self.rsf_momentum_fp32_bytes },
            .{ .label = "rsf_fp32_fisher_stacks", .bytes = self.rsf_fisher_fp32_bytes },
            .{ .label = "rsf_fp16_forward_stacks", .bytes = self.rsf_forward_fp16_bytes },
            .{ .label = "rsf_fp16_legacy_layer_mirrors", .bytes = self.rsf_legacy_mirror_fp16_bytes },
            .{ .label = "rsf_fp32_transient_gradients", .bytes = self.rsf_transient_gradient_fp32_bytes },
            .{ .label = "rsf_replacement_transients", .bytes = self.rsf_replacement_transient_bytes },
            .{ .label = "rsf_spectral_transients", .bytes = self.rsf_spectral_transient_bytes },
            .{ .label = "embedding_fp32_master", .bytes = self.embedding_master_fp32_bytes },
            .{ .label = "embedding_fp32_momentum", .bytes = self.embedding_momentum_fp32_bytes },
            .{ .label = "embedding_fp32_fisher", .bytes = self.embedding_fisher_fp32_bytes },
            .{ .label = "embedding_fp32_gradients", .bytes = self.embedding_gradient_fp32_bytes },
            .{ .label = "embedding_fp16_forward", .bytes = self.embedding_forward_fp16_bytes },
            .{ .label = "frozen_target_fp16", .bytes = self.frozen_target_fp16_bytes },
            .{ .label = "frozen_target_fp32_master", .bytes = self.frozen_target_fp32_master_bytes },
            .{ .label = "batch_activations", .bytes = self.batch_activation_bytes },
            .{ .label = "graph_state", .bytes = self.graph_state_bytes },
            .{ .label = "graph_chunk_transients", .bytes = self.graph_chunk_bytes },
            .{ .label = "nccl_buffers", .bytes = self.nccl_buffer_bytes },
            .{ .label = "futhark_runtime_overhead", .bytes = self.futhark_overhead_bytes },
        };
        for (candidates) |candidate| {
            if (candidate.bytes == 0) continue;
            if (count < scratch.len) {
                scratch[count] = candidate;
                count += 1;
            }
        }
        var i: usize = 1;
        while (i < count) : (i += 1) {
            const key = scratch[i];
            var j: usize = i;
            while (j > 0 and scratch[j - 1].bytes < key.bytes) : (j -= 1) {
                scratch[j] = scratch[j - 1];
            }
            scratch[j] = key;
        }
        const limit = @min(out.len, count);
        for (0..limit) |index| out[index] = scratch[index];
        return limit;
    }
};

pub const EstimatorConfig = struct {
    rsf: RsfStateOptions,
    embedding: EmbeddingStateOptions,
    batch: BatchActivationOptions,
    graph: GraphStateOptions = .{},
    overhead: FutharkOverhead = .{},
    nccl_world_size: usize = 1,
    nccl_buffer_bytes: u64 = 0,
};

pub fn estimate(config: EstimatorConfig) EstimationError!MemoryEstimate {
    var out = MemoryEstimate{};

    const stack_elements = try config.rsf.layout.stackElements();
    if (config.rsf.forward_fp16_stacks) {
        out.rsf_forward_fp16_bytes = try checkedBytes(stack_elements, bytes_per_f16);
        out.rsf_forward_fp16_bytes = try checkedAdd(u64, out.rsf_forward_fp16_bytes, try checkedBytes(stack_elements, bytes_per_f16));
    }
    if (config.rsf.fp32_master_stacks) {
        out.rsf_master_fp32_bytes = try checkedBytes(stack_elements, bytes_per_f32);
        out.rsf_master_fp32_bytes = try checkedAdd(u64, out.rsf_master_fp32_bytes, try checkedBytes(stack_elements, bytes_per_f32));
    }
    if (config.rsf.fp32_momentum_stacks) {
        out.rsf_momentum_fp32_bytes = try checkedBytes(stack_elements, bytes_per_f32);
        out.rsf_momentum_fp32_bytes = try checkedAdd(u64, out.rsf_momentum_fp32_bytes, try checkedBytes(stack_elements, bytes_per_f32));
    }
    if (config.rsf.fp32_fisher_stacks) {
        out.rsf_fisher_fp32_bytes = try checkedBytes(stack_elements, bytes_per_f32);
        out.rsf_fisher_fp32_bytes = try checkedAdd(u64, out.rsf_fisher_fp32_bytes, try checkedBytes(stack_elements, bytes_per_f32));
    }
    if (config.rsf.legacy_layer_mirrors) {
        out.rsf_legacy_mirror_fp16_bytes = try checkedBytes(stack_elements, bytes_per_f16);
        out.rsf_legacy_mirror_fp16_bytes = try checkedAdd(u64, out.rsf_legacy_mirror_fp16_bytes, try checkedBytes(stack_elements, bytes_per_f16));
    }
    if (config.rsf.transient_fp32_gradients) {
        out.rsf_transient_gradient_fp32_bytes = try checkedBytes(stack_elements, bytes_per_f32);
        out.rsf_transient_gradient_fp32_bytes = try checkedAdd(u64, out.rsf_transient_gradient_fp32_bytes, try checkedBytes(stack_elements, bytes_per_f32));
    }
    if (config.rsf.replacement_transients) {
        var replacement: u64 = 0;
        if (config.rsf.fp32_master_stacks) replacement = try checkedAdd(u64, replacement, try checkedBytes(stack_elements, bytes_per_f32));
        if (config.rsf.fp32_momentum_stacks) replacement = try checkedAdd(u64, replacement, try checkedBytes(stack_elements, bytes_per_f32));
        if (config.rsf.fp32_fisher_stacks) replacement = try checkedAdd(u64, replacement, try checkedBytes(stack_elements, bytes_per_f32));
        if (config.rsf.forward_fp16_stacks) replacement = try checkedAdd(u64, replacement, try checkedBytes(stack_elements, bytes_per_f16));
        out.rsf_replacement_transient_bytes = replacement;
    }
    if (config.rsf.spectral_startup) {
        out.rsf_spectral_transient_bytes = try checkedBytes(stack_elements, bytes_per_f32);
    }

    const embedding_elements = try config.embedding.layout.elements();
    if (config.embedding.forward_fp16) out.embedding_forward_fp16_bytes = try checkedBytes(embedding_elements, bytes_per_f16);
    if (config.embedding.fp32_master) out.embedding_master_fp32_bytes = try checkedBytes(embedding_elements, bytes_per_f32);
    if (config.embedding.fp32_gradients) out.embedding_gradient_fp32_bytes = try checkedBytes(embedding_elements, bytes_per_f32);
    if (config.embedding.fp32_momentum) out.embedding_momentum_fp32_bytes = try checkedBytes(embedding_elements, bytes_per_f32);
    if (config.embedding.fp32_fisher) out.embedding_fisher_fp32_bytes = try checkedBytes(embedding_elements, bytes_per_f32);
    if (config.embedding.spectral_vectors) {
        out.embedding_spectral_u_bytes = try checkedBytes(@intCast(config.embedding.layout.vocab_size), bytes_per_f32);
        out.embedding_spectral_v_bytes = try checkedBytes(@intCast(config.embedding.layout.model_dim), bytes_per_f32);
    }
    if (config.embedding.replacement_transients) {
        var replacement: u64 = 0;
        if (config.embedding.fp32_master) replacement = try checkedAdd(u64, replacement, try checkedBytes(embedding_elements, bytes_per_f32));
        if (config.embedding.fp32_momentum) replacement = try checkedAdd(u64, replacement, try checkedBytes(embedding_elements, bytes_per_f32));
        if (config.embedding.fp32_fisher) replacement = try checkedAdd(u64, replacement, try checkedBytes(embedding_elements, bytes_per_f32));
        if (config.embedding.forward_fp16) replacement = try checkedAdd(u64, replacement, try checkedBytes(embedding_elements, bytes_per_f16));
        out.embedding_replacement_transient_bytes = replacement;
    }
    if (config.embedding.frozen_target_fp16) out.frozen_target_fp16_bytes = try checkedBytes(embedding_elements, bytes_per_f16);
    if (config.embedding.frozen_target_fp32_master) out.frozen_target_fp32_master_bytes = try checkedBytes(embedding_elements, bytes_per_f32);

    const padded_rows_total = try config.batch.layout.paddedRows();
    const activation_rows = if (config.batch.compact_rows and config.batch.active_rows_override > 0 and config.batch.active_rows_override < padded_rows_total)
        config.batch.active_rows_override
    else
        padded_rows_total;
    const row_bytes_f16 = try checkedBytes(try checkedMul(u64, activation_rows, @intCast(config.batch.layout.model_dim)), bytes_per_f16);
    out.batch_activation_bytes = try checkedMul(u64, row_bytes_f16, config.batch.fp16_padded_tensors);
    const fp32_batch = try checkedBytes(try checkedMul(u64, row_bytes_f16, 2), config.batch.fp32_padded_tensors);
    out.batch_activation_bytes = try checkedAdd(u64, out.batch_activation_bytes, fp32_batch);
    if (config.batch.index_arrays) {
        out.batch_index_bytes = try checkedBytes(try checkedMul(u64, activation_rows, 2), bytes_per_i64);
        out.batch_index_bytes = try checkedAdd(u64, out.batch_index_bytes, try checkedBytes(@intCast(config.batch.layout.batch_size), bytes_per_i64));
    }

    if (config.graph.enabled) {
        out.graph_state_bytes = try checkedBytes(try checkedMul(u64, @intCast(config.graph.chunk_hashes), config.graph.node_fields), bytes_per_f32);
        out.graph_state_bytes = try checkedAdd(u64, out.graph_state_bytes, try checkedBytes(try checkedMul(u64, @intCast(config.graph.chunk_hashes), config.graph.edges_per_hash), bytes_per_i64));
        out.graph_chunk_bytes = out.graph_state_bytes;
    }

    out.nccl_buffer_bytes = if (config.nccl_world_size > 1) config.nccl_buffer_bytes else 0;

    const peak_before_overhead = try out.peakBytes();
    const slack: u64 = @intFromFloat(@as(f64, @floatFromInt(peak_before_overhead)) * config.overhead.device_allocator_slack_fraction);
    out.futhark_overhead_bytes = try checkedAdd(u64, config.overhead.opaque_tuple_bytes, slack);
    return out;
}

pub const MemoryReserve = struct {
    absolute_mib: u64 = 4096,
    fraction: f64 = 0.05,

    pub fn reserveBytes(self: MemoryReserve, total_bytes: u64) u64 {
        const fractional: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) * self.fraction);
        return @max(self.absolute_mib * 1024 * 1024, fractional);
    }
};

pub const AdmissionError = error{
    EstimatedPeakExceedsDeviceMemory,
    OverflowInAdmission,
};

pub const AdmissionDecision = struct {
    admitted: bool,
    requested_peak_bytes: u64,
    free_bytes: u64,
    total_bytes: u64,
    reserve_bytes: u64,
    headroom_bytes: i64,
};

pub fn checkAdmission(peak_bytes: u64, free_bytes: u64, total_bytes: u64, reserve: MemoryReserve) AdmissionError!AdmissionDecision {
    const reserve_bytes = reserve.reserveBytes(total_bytes);
    const available = std.math.sub(u64, free_bytes, reserve_bytes) catch 0;
    const headroom: i64 = @as(i64, @intCast(available)) - @as(i64, @intCast(peak_bytes));
    return AdmissionDecision{
        .admitted = headroom >= 0,
        .requested_peak_bytes = peak_bytes,
        .free_bytes = free_bytes,
        .total_bytes = total_bytes,
        .reserve_bytes = reserve_bytes,
        .headroom_bytes = headroom,
    };
}

pub fn formatRejection(
    allocator: std.mem.Allocator,
    decision: AdmissionDecision,
    estimate_value: *const MemoryEstimate,
) ![]u8 {
    var contributors: [6]MemoryEstimate.Contributor = undefined;
    const count = estimate_value.topContributors(contributors[0..]);
    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();
    const writer = buffer.writer();
    try writer.print("memory preflight rejected: estimated peak {d} bytes ({d:.3} GiB) exceeds free {d} bytes ({d:.3} GiB) minus reserve {d} bytes ({d:.3} GiB); total device memory {d} bytes; headroom {d} bytes\n", .{
        decision.requested_peak_bytes,
        MemoryEstimate.gib(decision.requested_peak_bytes),
        decision.free_bytes,
        MemoryEstimate.gib(decision.free_bytes),
        decision.reserve_bytes,
        MemoryEstimate.gib(decision.reserve_bytes),
        decision.total_bytes,
        decision.headroom_bytes,
    });
    try writer.print("largest contributors:\n", .{});
    for (contributors[0..count]) |contributor| {
        try writer.print("  {s}: {d} bytes ({d:.3} GiB)\n", .{ contributor.label, contributor.bytes, MemoryEstimate.gib(contributor.bytes) });
    }
    try writer.print("remedies: reduce optimizer state (disable fisher or momentum stacks), reduce vocab_size, reduce num_layers, reduce batch_size or max_seq_len, disable legacy layer mirrors, skip knowledge graph construction, or increase device memory reserve\n", .{});
    return buffer.toOwnedSlice();
}

pub fn formatEstimateReport(allocator: std.mem.Allocator, estimate_value: *const MemoryEstimate, free_bytes: u64, total_bytes: u64) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    errdefer buffer.deinit();
    const writer = buffer.writer();
    try writer.print("persistent parameter and optimizer memory: {d} bytes ({d:.3} GiB)\n", .{ try estimate_value.persistentBytes(), MemoryEstimate.gib(try estimate_value.persistentBytes()) });
    try writer.print("  rsf stacks persistent: {d} bytes ({d:.3} GiB)\n", .{ try estimate_value.rsfPersistentBytes(), MemoryEstimate.gib(try estimate_value.rsfPersistentBytes()) });
    try writer.print("  embedding persistent: {d} bytes ({d:.3} GiB)\n", .{ try estimate_value.embeddingPersistentBytes(), MemoryEstimate.gib(try estimate_value.embeddingPersistentBytes()) });
    try writer.print("activation and transient peak component: {d} bytes ({d:.3} GiB)\n", .{ try estimate_value.activationAndTransientBytes(), MemoryEstimate.gib(try estimate_value.activationAndTransientBytes()) });
    try writer.print("estimated peak: {d} bytes ({d:.3} GiB)\n", .{ try estimate_value.peakBytes(), MemoryEstimate.gib(try estimate_value.peakBytes()) });
    try writer.print("device free: {d} bytes ({d:.3} GiB); device total: {d} bytes ({d:.3} GiB)\n", .{ free_bytes, MemoryEstimate.gib(free_bytes), total_bytes, MemoryEstimate.gib(total_bytes) });
    return buffer.toOwnedSlice();
}

pub const baseline_model_dim: usize = 16384;
pub const baseline_num_layers: usize = 11;
pub const baseline_vocab_size: usize = 32000;
pub const baseline_batch_size: usize = 32;
pub const baseline_max_seq_len: usize = 256;

pub fn baselineEstimate() EstimationError!MemoryEstimate {
    return estimate(.{
        .rsf = .{ .layout = .{ .model_dim = baseline_model_dim, .num_layers = baseline_num_layers } },
        .embedding = .{ .layout = .{ .vocab_size = baseline_vocab_size, .model_dim = baseline_model_dim }, .frozen_target_fp16 = true },
        .batch = .{ .layout = .{ .batch_size = baseline_batch_size, .max_seq_len = baseline_max_seq_len, .model_dim = baseline_model_dim } },
    });
}

pub fn legacyMirrorEstimate() EstimationError!MemoryEstimate {
    return estimate(.{
        .rsf = .{ .layout = .{ .model_dim = baseline_model_dim, .num_layers = baseline_num_layers }, .legacy_layer_mirrors = true },
        .embedding = .{ .layout = .{ .vocab_size = baseline_vocab_size, .model_dim = baseline_model_dim }, .frozen_target_fp16 = true, .frozen_target_fp32_master = true },
        .batch = .{ .layout = .{ .batch_size = baseline_batch_size, .max_seq_len = baseline_max_seq_len, .model_dim = baseline_model_dim } },
    });
}

test "checked byte arithmetic rejects overflow" {
    try std.testing.expectError(EstimationError.ElementCountOverflow, checkedMul(u64, std.math.maxInt(u64), 2));
    try std.testing.expectError(EstimationError.ByteCountOverflow, checkedAdd(u64, std.math.maxInt(u64), 1));
    try std.testing.expectError(EstimationError.ElementCountOverflow, checkedBytes(std.math.maxInt(u64) - 1, 4));
    try std.testing.expectEqual(@as(u64, 8), try checkedBytes(2, 4));
}

test "layout rejects invalid dimensions" {
    try std.testing.expectError(EstimationError.InvalidDimensions, (RsfLayout{ .model_dim = 0, .num_layers = 1 }).stackElements());
    try std.testing.expectError(EstimationError.InvalidDimensions, (RsfLayout{ .model_dim = 3, .num_layers = 1 }).stackElements());
    try std.testing.expectError(EstimationError.InvalidDimensions, (RsfLayout{ .model_dim = 8, .num_layers = 0 }).stackElements());
    try std.testing.expectError(EstimationError.InvalidDimensions, (EmbeddingLayout{ .vocab_size = 0, .model_dim = 4 }).elements());
}

test "layout overflow is classified not wrapped" {
    const huge = RsfLayout{ .model_dim = std.math.maxInt(usize) - 1, .num_layers = std.math.maxInt(usize) };
    try std.testing.expectError(EstimationError.ElementCountOverflow, huge.stackElements());
}

test "baseline 16384 by 11 element and byte counts are exact" {
    const layout = RsfLayout{ .model_dim = 16384, .num_layers = 11 };
    try std.testing.expectEqual(@as(usize, 8192), try layout.half());
    try std.testing.expectEqual(@as(usize, 8193), try layout.columns());
    try std.testing.expectEqual(@as(usize, 8192 * 8193), try layout.perLayerElements());
    try std.testing.expectEqual(@as(u64, 738287616), try layout.stackElements());
    try std.testing.expectEqual(@as(u64, 2953150464), try layout.stackBytesFp32());
    try std.testing.expectEqual(@as(u64, 1476575232), try layout.stackBytesFp16());
}

test "baseline stack estimates match documented gib values" {
    const est = try baselineEstimate();
    try std.testing.expectApproxEqAbs(@as(f64, 2.750), MemoryEstimate.gib(est.rsf_forward_fp16_bytes), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.501), MemoryEstimate.gib(est.rsf_master_fp32_bytes), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.501), MemoryEstimate.gib(est.rsf_momentum_fp32_bytes), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.501), MemoryEstimate.gib(est.rsf_fisher_fp32_bytes), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.501), MemoryEstimate.gib(est.rsf_transient_gradient_fp32_bytes), 0.001);
    try std.testing.expectEqual(@as(u64, 0), est.rsf_legacy_mirror_fp16_bytes);
}

test "legacy mirror estimate includes redundant fp16 mirrors" {
    const est = try legacyMirrorEstimate();
    try std.testing.expectApproxEqAbs(@as(f64, 2.750), MemoryEstimate.gib(est.rsf_legacy_mirror_fp16_bytes), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.750), MemoryEstimate.gib(est.rsf_forward_fp16_bytes), 0.001);
    const stack_only = try baselineEstimate();
    const mirror_persistent = try (try legacyMirrorEstimate()).persistentBytes();
    const stack_persistent = try stack_only.persistentBytes();
    try std.testing.expect(mirror_persistent > stack_persistent);
}

test "baseline embedding and frozen target sizes use vocab times dim" {
    const est = try baselineEstimate();
    const elements: u64 = 32000 * 16384;
    try std.testing.expectEqual(elements * 2, est.embedding_forward_fp16_bytes);
    try std.testing.expectEqual(elements * 4, est.embedding_master_fp32_bytes);
    try std.testing.expectEqual(elements * 4, est.embedding_gradient_fp32_bytes);
    try std.testing.expectEqual(elements * 2, est.frozen_target_fp16_bytes);
    try std.testing.expectEqual(@as(u64, 32000 * 4), est.embedding_spectral_u_bytes);
    try std.testing.expectEqual(@as(u64, 16384 * 4), est.embedding_spectral_v_bytes);
}

test "admission accepts baseline on 180 gib device" {
    const est = try baselineEstimate();
    const total: u64 = 180 * 1073741824;
    const free: u64 = 175 * 1073741824;
    const decision = try checkAdmission(try est.peakBytes(), free, total, .{});
    try std.testing.expect(decision.admitted);
}

test "admission rejects when reserve exceeds headroom" {
    const est = try baselineEstimate();
    const total: u64 = 32 * 1073741824;
    const free: u64 = 30 * 1073741824;
    const decision = try checkAdmission(try est.peakBytes(), free, total, .{});
    try std.testing.expect(!decision.admitted);
    try std.testing.expect(decision.headroom_bytes < 0);
}

test "rejection report names largest contributors" {
    const allocator = std.testing.allocator;
    const est = try legacyMirrorEstimate();
    const decision = try checkAdmission(try est.peakBytes(), 8 * 1073741824, 12 * 1073741824, .{});
    const report = try formatRejection(allocator, decision, &est);
    defer allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "rsf_fp32_master_stacks") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "memory preflight rejected") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "remedies") != null);
}

test "compact active-row accounting reduces activation estimate only" {
    const padded_estimate = try estimate(.{
        .rsf = .{ .layout = .{ .model_dim = 1024, .num_layers = 2 } },
        .embedding = .{ .layout = .{ .vocab_size = 1000, .model_dim = 1024 } },
        .batch = .{ .layout = .{ .batch_size = 8, .max_seq_len = 64, .model_dim = 1024 } },
    });
    const compact_estimate = try estimate(.{
        .rsf = .{ .layout = .{ .model_dim = 1024, .num_layers = 2 } },
        .embedding = .{ .layout = .{ .vocab_size = 1000, .model_dim = 1024 } },
        .batch = .{ .layout = .{ .batch_size = 8, .max_seq_len = 64, .model_dim = 1024 }, .compact_rows = true, .active_rows_override = 8 * 16 },
    });
    try std.testing.expect(compact_estimate.batch_activation_bytes < padded_estimate.batch_activation_bytes);
    try std.testing.expect(compact_estimate.batch_index_bytes < padded_estimate.batch_index_bytes);
    try std.testing.expectEqual(padded_estimate.rsfPersistentBytes(), compact_estimate.rsfPersistentBytes());
    try std.testing.expectEqual(padded_estimate.embeddingPersistentBytes(), compact_estimate.embeddingPersistentBytes());
}

test "persistent memory is not claimed to be o of dim" {
    const small = try estimate(.{
        .rsf = .{ .layout = .{ .model_dim = 1024, .num_layers = 2 } },
        .embedding = .{ .layout = .{ .vocab_size = 1000, .model_dim = 1024 } },
        .batch = .{ .layout = .{ .batch_size = 4, .max_seq_len = 16, .model_dim = 1024 } },
    });
    const large_layers = try estimate(.{
        .rsf = .{ .layout = .{ .model_dim = 1024, .num_layers = 16 } },
        .embedding = .{ .layout = .{ .vocab_size = 1000, .model_dim = 1024 } },
        .batch = .{ .layout = .{ .batch_size = 4, .max_seq_len = 16, .model_dim = 1024 } },
    });
    const activation_small = try small.activationAndTransientBytes();
    const activation_large = try large_layers.activationAndTransientBytes();
    try std.testing.expect(activation_large >= activation_small);
    try std.testing.expect((try large_layers.rsfPersistentBytes()) > 4 * (try small.rsfPersistentBytes()) / 2);
    try std.testing.expect((try large_layers.persistentBytes()) > (try small.persistentBytes()));
}
