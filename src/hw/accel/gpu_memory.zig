const std = @import("std");
const builtin = @import("builtin");

pub const MemoryEstimateError = error{
    InvalidDimensions,
    SizeOverflow,
};

pub const PreflightError = error{
    InvalidDimensions,
    SizeOverflow,
    InsufficientDeviceMemory,
    DeviceQueryFailed,
};

pub const bytes_per_f16: usize = 2;
pub const bytes_per_f32: usize = 4;
pub const bytes_per_i64: usize = 8;
pub const bytes_per_u64: usize = 8;

pub const bytes_per_kib: usize = 1024;
pub const bytes_per_mib: usize = 1024 * 1024;
pub const bytes_per_gib: usize = 1024 * 1024 * 1024;

fn mul(a: usize, b: usize) MemoryEstimateError!usize {
    return std.math.mul(usize, a, b) catch return MemoryEstimateError.SizeOverflow;
}

fn add(a: usize, b: usize) MemoryEstimateError!usize {
    return std.math.add(usize, a, b) catch return MemoryEstimateError.SizeOverflow;
}

fn mul3(a: usize, b: usize, c: usize) MemoryEstimateError!usize {
    return try mul(try mul(a, b), c);
}

fn mul4(a: usize, b: usize, c: usize, d: usize) MemoryEstimateError!usize {
    return try mul(try mul3(a, b, c), d);
}

pub const Contribution = struct {
    name: []const u8,
    bytes: usize,
    persistent: bool,

    pub fn gib(self: Contribution) f64 {
        return @as(f64, @floatFromInt(self.bytes)) / @as(f64, @floatFromInt(bytes_per_gib));
    }
};

pub const max_contributions: usize = 24;

pub const ModelShape = struct {
    model_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    batch_size: usize,
    max_seq_len: usize,
    graph_chunk_size: usize = 0,
    graph_nodes: usize = 0,
    stack_only: bool = false,
    minimal_frozen_target: bool = false,
    momentum_enabled: bool = true,
    fisher_enabled: bool = true,
    skip_knowledge_graph: bool = false,

    pub fn validate(self: ModelShape) MemoryEstimateError!void {
        if (self.model_dim == 0) return MemoryEstimateError.InvalidDimensions;
        if (self.model_dim % 2 != 0) return MemoryEstimateError.InvalidDimensions;
        if (self.num_layers == 0) return MemoryEstimateError.InvalidDimensions;
        if (self.vocab_size == 0) return MemoryEstimateError.InvalidDimensions;
        if (self.batch_size == 0) return MemoryEstimateError.InvalidDimensions;
        if (self.max_seq_len == 0) return MemoryEstimateError.InvalidDimensions;
    }

    pub fn half(self: ModelShape) usize {
        return self.model_dim / 2;
    }

    pub fn columns(self: ModelShape) MemoryEstimateError!usize {
        return try add(self.half(), 1);
    }

    pub fn perLayerElements(self: ModelShape) MemoryEstimateError!usize {
        return try mul(self.half(), try self.columns());
    }

    pub fn stackElements(self: ModelShape) MemoryEstimateError!usize {
        return try mul(self.num_layers, try self.perLayerElements());
    }

    pub fn embeddingElements(self: ModelShape) MemoryEstimateError!usize {
        return try mul(self.vocab_size, self.model_dim);
    }

    pub fn tokenCount(self: ModelShape) MemoryEstimateError!usize {
        return try mul(self.batch_size, self.max_seq_len);
    }
};

pub const Estimate = struct {
    shape: ModelShape,
    contributions: [max_contributions]Contribution = undefined,
    contribution_count: usize = 0,
    persistent_bytes: usize = 0,
    transient_bytes: usize = 0,

    pub fn totalBytes(self: Estimate) usize {
        return self.persistent_bytes + self.transient_bytes;
    }

    pub fn items(self: *const Estimate) []const Contribution {
        return self.contributions[0..self.contribution_count];
    }

    fn push(self: *Estimate, name: []const u8, bytes: usize, persistent: bool) MemoryEstimateError!void {
        if (bytes == 0) return;
        if (self.contribution_count >= max_contributions) return MemoryEstimateError.SizeOverflow;
        self.contributions[self.contribution_count] = .{
            .name = name,
            .bytes = bytes,
            .persistent = persistent,
        };
        self.contribution_count += 1;
        if (persistent) {
            self.persistent_bytes = try add(self.persistent_bytes, bytes);
        } else {
            self.transient_bytes = try add(self.transient_bytes, bytes);
        }
    }

    pub fn largest(self: *const Estimate, out: []Contribution) []Contribution {
        const n = @min(out.len, self.contribution_count);
        if (n == 0) return out[0..0];
        var copy: [max_contributions]Contribution = undefined;
        @memcpy(copy[0..self.contribution_count], self.contributions[0..self.contribution_count]);
        const slice = copy[0..self.contribution_count];
        std.mem.sort(Contribution, slice, {}, struct {
            fn lessThan(_: void, a: Contribution, b: Contribution) bool {
                return a.bytes > b.bytes;
            }
        }.lessThan);
        @memcpy(out[0..n], slice[0..n]);
        return out[0..n];
    }
};

pub fn estimate(shape: ModelShape) MemoryEstimateError!Estimate {
    try shape.validate();

    var result = Estimate{ .shape = shape };

    const stack_elems = try shape.stackElements();
    const stack_f16 = try mul(stack_elems, bytes_per_f16);
    const stack_f32 = try mul(stack_elems, bytes_per_f32);

    if (!shape.stack_only) {
        const per_layer_elems = try shape.perLayerElements();
        const mirrors = try mul4(shape.num_layers, per_layer_elems, bytes_per_f16, 2);
        try result.push("rsf per-layer fp16 mirrors (s+t)", mirrors, true);
    }

    try result.push("rsf stacked fp16 shadows (s+t)", try mul(stack_f16, 2), true);
    try result.push("rsf fp32 master stacks (s+t)", try mul(stack_f32, 2), true);
    if (shape.momentum_enabled) {
        try result.push("rsf fp32 momentum stacks (s+t)", try mul(stack_f32, 2), true);
    }
    if (shape.fisher_enabled) {
        try result.push("rsf fp32 fisher stacks (s+t)", try mul(stack_f32, 2), true);
    }
    try result.push("rsf fp32 gradient stacks (s+t, transient)", try mul(stack_f32, 2), false);

    const embed_elems = try shape.embeddingElements();
    try result.push("embedding fp16 weight", try mul(embed_elems, bytes_per_f16), true);
    try result.push("embedding fp32 master", try mul(embed_elems, bytes_per_f32), true);
    try result.push("embedding fp32 gradient", try mul(embed_elems, bytes_per_f32), true);
    if (shape.momentum_enabled) {
        try result.push("embedding fp32 momentum", try mul(embed_elems, bytes_per_f32), true);
    }
    if (shape.fisher_enabled) {
        try result.push("embedding fp32 fisher", try mul(embed_elems, bytes_per_f32), true);
    }

    if (shape.minimal_frozen_target) {
        try result.push("frozen target fp16 weight", try mul(embed_elems, bytes_per_f16), true);
    } else {
        const clone_f16 = try mul(embed_elems, bytes_per_f16);
        const clone_master = try mul(embed_elems, bytes_per_f32);
        const clone_grad = try mul(embed_elems, bytes_per_f32);
        try result.push(
            "frozen target clone (fp16 + fp32 master + fp32 grad)",
            try add(clone_f16, try add(clone_master, clone_grad)),
            true,
        );
    }

    const tokens = try shape.tokenCount();
    try result.push("batch token ids (input+target)", try mul3(tokens, bytes_per_i64, 2), false);
    try result.push("batch embedding activations fp16", try mul3(tokens, shape.model_dim, bytes_per_f16), false);
    try result.push("rsf activation buffers fp16 (fwd+inv)", try mul4(tokens, shape.model_dim, bytes_per_f16, 2), false);
    try result.push("embedding backward accumulation fp32", try mul3(tokens, shape.model_dim, bytes_per_f32), false);

    const spectral_vec = try mul(shape.model_dim, bytes_per_f32);
    try result.push("spectral power-iteration vectors (u+v)", try mul(spectral_vec, 2), true);

    if (!shape.skip_knowledge_graph) {
        const chunk = if (shape.graph_chunk_size == 0) shape.graph_nodes else @min(shape.graph_chunk_size, shape.graph_nodes);
        if (chunk > 0) {
            const hashes = try mul(chunk, bytes_per_u64);
            const feats = try mul3(chunk, bytes_per_f32, 4);
            const edges = try mul4(chunk, 3, bytes_per_i64, 2);
            try result.push("graph chunk staging (hashes+features+edges)", try add(hashes, try add(feats, edges)), false);
        }
    }

    return result;
}

pub const StackFootprint = struct {
    shadow_bytes: usize,
    master_bytes: usize,
    mirror_bytes: usize,

    pub fn total(self: StackFootprint) usize {
        return self.shadow_bytes + self.master_bytes + self.mirror_bytes;
    }
};

pub fn stackFootprint(
    model_dim: usize,
    num_layers: usize,
    mirror_layers: usize,
) MemoryEstimateError!StackFootprint {
    if (model_dim == 0 or model_dim % 2 != 0) return MemoryEstimateError.InvalidDimensions;
    if (num_layers == 0) return MemoryEstimateError.InvalidDimensions;
    if (mirror_layers > num_layers) return MemoryEstimateError.InvalidDimensions;

    const half = model_dim / 2;
    const cols = try add(half, 1);
    const per_layer = try mul(half, cols);
    const stack_elems = try mul(num_layers, per_layer);
    const mirror_elems = try mul(mirror_layers, per_layer);

    return .{
        .shadow_bytes = try mul(stack_elems, 2 * bytes_per_f16),
        .master_bytes = try mul(stack_elems, 6 * bytes_per_f32),
        .mirror_bytes = try mul(mirror_elems, 2 * bytes_per_f16),
    };
}

pub const PreflightConfig = struct {
    reserve_bytes: usize = 512 * bytes_per_mib,
    reserve_fraction: f64 = 0.05,
    require_transient_headroom: bool = true,

    pub fn fromEnv() PreflightConfig {
        var cfg = PreflightConfig{};
        if (std.posix.getenv("JAIDE_GPU_RESERVE_MIB")) |raw| {
            if (std.fmt.parseInt(usize, std.mem.trim(u8, raw, " \t\r\n"), 10)) |mib| {
                cfg.reserve_bytes = mib *| bytes_per_mib;
            } else |_| {}
        }
        if (std.posix.getenv("JAIDE_GPU_RESERVE_FRACTION")) |raw| {
            if (std.fmt.parseFloat(f64, std.mem.trim(u8, raw, " \t\r\n"))) |frac| {
                if (frac >= 0.0 and frac < 1.0) cfg.reserve_fraction = frac;
            } else |_| {}
        }
        return cfg;
    }
};

pub const PreflightReport = struct {
    estimate: Estimate,
    total_device_bytes: usize,
    free_device_bytes: usize,
    reserve_bytes: usize,
    required_bytes: usize,
    admitted: bool,

    pub fn budgetBytes(self: PreflightReport) usize {
        return self.free_device_bytes -| self.reserve_bytes;
    }
};

pub fn reserveFor(config: PreflightConfig, free_bytes: usize) usize {
    const fractional_f = @as(f64, @floatFromInt(free_bytes)) * config.reserve_fraction;
    const fractional: usize = if (fractional_f <= 0.0)
        0
    else if (fractional_f >= @as(f64, @floatFromInt(std.math.maxInt(usize))))
        std.math.maxInt(usize)
    else
        @intFromFloat(fractional_f);
    return @max(config.reserve_bytes, fractional);
}

pub fn admit(
    shape: ModelShape,
    config: PreflightConfig,
    total_device_bytes: usize,
    free_device_bytes: usize,
) PreflightError!PreflightReport {
    const est = estimate(shape) catch |err| switch (err) {
        MemoryEstimateError.InvalidDimensions => return PreflightError.InvalidDimensions,
        MemoryEstimateError.SizeOverflow => return PreflightError.SizeOverflow,
    };

    const required = if (config.require_transient_headroom)
        est.totalBytes()
    else
        est.persistent_bytes;

    const reserve = reserveFor(config, free_device_bytes);
    const budget = free_device_bytes -| reserve;

    return .{
        .estimate = est,
        .total_device_bytes = total_device_bytes,
        .free_device_bytes = free_device_bytes,
        .reserve_bytes = reserve,
        .required_bytes = required,
        .admitted = required <= budget,
    };
}

pub fn formatGiB(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(bytes_per_gib));
}

pub fn logReport(report: PreflightReport) void {
    std.debug.print(
        "[gpu-preflight] device total={d:.3} GiB free={d:.3} GiB reserve={d:.3} GiB budget={d:.3} GiB\n",
        .{
            formatGiB(report.total_device_bytes),
            formatGiB(report.free_device_bytes),
            formatGiB(report.reserve_bytes),
            formatGiB(report.budgetBytes()),
        },
    );
    std.debug.print(
        "[gpu-preflight] estimate persistent={d:.3} GiB transient={d:.3} GiB required={d:.3} GiB verdict={s}\n",
        .{
            formatGiB(report.estimate.persistent_bytes),
            formatGiB(report.estimate.transient_bytes),
            formatGiB(report.required_bytes),
            if (report.admitted) "admit" else "reject",
        },
    );
    var top: [5]Contribution = undefined;
    const largest = report.estimate.largest(top[0..]);
    for (largest, 0..) |item, index| {
        std.debug.print(
            "[gpu-preflight]   #{d} {s}: {d:.3} GiB ({s})\n",
            .{ index + 1, item.name, item.gib(), if (item.persistent) "persistent" else "transient" },
        );
    }
}

test "estimate rejects invalid dimensions" {
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, estimate(.{
        .model_dim = 0,
        .num_layers = 4,
        .vocab_size = 128,
        .batch_size = 2,
        .max_seq_len = 8,
    }));
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, estimate(.{
        .model_dim = 15,
        .num_layers = 4,
        .vocab_size = 128,
        .batch_size = 2,
        .max_seq_len = 8,
    }));
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, estimate(.{
        .model_dim = 16,
        .num_layers = 0,
        .vocab_size = 128,
        .batch_size = 2,
        .max_seq_len = 8,
    }));
}

test "estimate detects overflow instead of wrapping" {
    const huge: usize = @as(usize, 1) << 40;
    try std.testing.expectError(MemoryEstimateError.SizeOverflow, estimate(.{
        .model_dim = huge,
        .num_layers = huge,
        .vocab_size = huge,
        .batch_size = huge,
        .max_seq_len = huge,
    }));

    const even_max: usize = std.math.maxInt(usize) - 1;
    try std.testing.expectError(MemoryEstimateError.SizeOverflow, estimate(.{
        .model_dim = even_max,
        .num_layers = 2,
        .vocab_size = 2,
        .batch_size = 2,
        .max_seq_len = 2,
    }));

    var shape = ModelShape{
        .model_dim = 65536,
        .num_layers = 64,
        .vocab_size = 32000,
        .batch_size = 8,
        .max_seq_len = 128,
    };
    const ok = try estimate(shape);
    try std.testing.expect(ok.totalBytes() > 0);

    shape.vocab_size = std.math.maxInt(usize) / 1024;
    try std.testing.expectError(MemoryEstimateError.SizeOverflow, estimate(shape));
}

test "baseline b200 shape reproduces documented redundant footprint" {
    const shape = ModelShape{
        .model_dim = 16384,
        .num_layers = 11,
        .vocab_size = 32000,
        .batch_size = 32,
        .max_seq_len = 256,
    };

    try std.testing.expectEqual(@as(usize, 8192), shape.half());
    try std.testing.expectEqual(@as(usize, 8193), try shape.columns());
    try std.testing.expectEqual(@as(usize, 67_117_056), try shape.perLayerElements());
    try std.testing.expectEqual(@as(usize, 738_287_616), try shape.stackElements());

    const est = try estimate(shape);

    const stack_f32_pair: usize = 738_287_616 * 4 * 2;
    try std.testing.expectEqual(@as(usize, 5_906_300_928), stack_f32_pair);

    var found_master = false;
    var found_mirrors = false;
    for (est.items()) |item| {
        if (std.mem.eql(u8, item.name, "rsf fp32 master stacks (s+t)")) {
            try std.testing.expectEqual(stack_f32_pair, item.bytes);
            found_master = true;
        }
        if (std.mem.eql(u8, item.name, "rsf per-layer fp16 mirrors (s+t)")) {
            try std.testing.expectEqual(@as(usize, 2_953_150_464), item.bytes);
            found_mirrors = true;
        }
    }
    try std.testing.expect(found_master);
    try std.testing.expect(found_mirrors);
}

test "stack only mode removes per layer mirrors" {
    const base = ModelShape{
        .model_dim = 16384,
        .num_layers = 11,
        .vocab_size = 32000,
        .batch_size = 32,
        .max_seq_len = 256,
    };
    var lean = base;
    lean.stack_only = true;
    lean.minimal_frozen_target = true;

    const wide_est = try estimate(base);
    const lean_est = try estimate(lean);

    try std.testing.expect(lean_est.persistent_bytes < wide_est.persistent_bytes);

    for (lean_est.items()) |item| {
        try std.testing.expect(!std.mem.eql(u8, item.name, "rsf per-layer fp16 mirrors (s+t)"));
    }

    const mirrors: usize = 2_953_150_464;
    const embed_elems: usize = 32000 * 16384;
    const clone_saving = embed_elems * bytes_per_f32 * 2;
    try std.testing.expectEqual(
        wide_est.persistent_bytes - mirrors - clone_saving,
        lean_est.persistent_bytes,
    );
}

test "largest reports contributors in descending order" {
    const est = try estimate(.{
        .model_dim = 512,
        .num_layers = 4,
        .vocab_size = 1024,
        .batch_size = 2,
        .max_seq_len = 16,
    });
    var top: [4]Contribution = undefined;
    const largest = est.largest(top[0..]);
    try std.testing.expect(largest.len == 4);
    var index: usize = 1;
    while (index < largest.len) : (index += 1) {
        try std.testing.expect(largest[index - 1].bytes >= largest[index].bytes);
    }
}

test "admission accepts when budget is sufficient" {
    const shape = ModelShape{
        .model_dim = 512,
        .num_layers = 4,
        .vocab_size = 1024,
        .batch_size = 2,
        .max_seq_len = 16,
    };
    const est = try estimate(shape);
    const total = est.totalBytes() * 8;
    const report = try admit(shape, .{ .reserve_bytes = 0, .reserve_fraction = 0.0 }, total, total);
    try std.testing.expect(report.admitted);
    try std.testing.expectEqual(est.totalBytes(), report.required_bytes);
}

test "admission rejects when reserve eats the budget" {
    const shape = ModelShape{
        .model_dim = 512,
        .num_layers = 4,
        .vocab_size = 1024,
        .batch_size = 2,
        .max_seq_len = 16,
    };
    const est = try estimate(shape);
    const free = est.totalBytes() + bytes_per_mib;
    const report = try admit(
        shape,
        .{ .reserve_bytes = 64 * bytes_per_mib, .reserve_fraction = 0.0 },
        free,
        free,
    );
    try std.testing.expect(!report.admitted);
    try std.testing.expect(report.budgetBytes() < report.required_bytes);
}

test "fractional reserve dominates when larger than absolute reserve" {
    const free: usize = 100 * bytes_per_gib;
    const cfg = PreflightConfig{ .reserve_bytes = 1 * bytes_per_gib, .reserve_fraction = 0.10 };
    try std.testing.expectEqual(@as(usize, 10 * bytes_per_gib), reserveFor(cfg, free));

    const cfg_small = PreflightConfig{ .reserve_bytes = 20 * bytes_per_gib, .reserve_fraction = 0.10 };
    try std.testing.expectEqual(@as(usize, 20 * bytes_per_gib), reserveFor(cfg_small, free));
}

test "b200 target config is admitted once redundancy is removed" {
    const b200_free: usize = 180 * bytes_per_gib;
    const baseline = ModelShape{
        .model_dim = 16384,
        .num_layers = 11,
        .vocab_size = 32000,
        .batch_size = 32,
        .max_seq_len = 256,
    };
    var lean = baseline;
    lean.stack_only = true;
    lean.minimal_frozen_target = true;

    const lean_report = try admit(lean, .{}, b200_free, b200_free);
    try std.testing.expect(lean_report.admitted);

    const baseline_report = try admit(baseline, .{}, b200_free, b200_free);
    try std.testing.expect(baseline_report.required_bytes > lean_report.required_bytes);
}

test "graph staging is excluded when knowledge graph is skipped" {
    var shape = ModelShape{
        .model_dim = 256,
        .num_layers = 2,
        .vocab_size = 512,
        .batch_size = 2,
        .max_seq_len = 8,
        .graph_nodes = 500_000,
        .graph_chunk_size = 8192,
    };
    const with_graph = try estimate(shape);
    shape.skip_knowledge_graph = true;
    const without_graph = try estimate(shape);
    try std.testing.expect(with_graph.transient_bytes > without_graph.transient_bytes);
}

test "chunked graph staging never scales with total node count" {
    const chunked = try estimate(.{
        .model_dim = 256,
        .num_layers = 2,
        .vocab_size = 512,
        .batch_size = 2,
        .max_seq_len = 8,
        .graph_nodes = 500_000,
        .graph_chunk_size = 8192,
    });
    const unchunked = try estimate(.{
        .model_dim = 256,
        .num_layers = 2,
        .vocab_size = 512,
        .batch_size = 2,
        .max_seq_len = 8,
        .graph_nodes = 500_000,
        .graph_chunk_size = 0,
    });
    try std.testing.expect(chunked.transient_bytes < unchunked.transient_bytes);
}

test "stack footprint drops the mirror block in stack only mode" {
    const wide = try stackFootprint(16384, 11, 11);
    const lean = try stackFootprint(16384, 11, 0);

    try std.testing.expectEqual(@as(usize, 2_953_150_464), wide.mirror_bytes);
    try std.testing.expectEqual(@as(usize, 0), lean.mirror_bytes);
    try std.testing.expectEqual(wide.shadow_bytes, lean.shadow_bytes);
    try std.testing.expectEqual(wide.master_bytes, lean.master_bytes);
    try std.testing.expectEqual(wide.total() - 2_953_150_464, lean.total());
}

test "stack footprint matches the documented per component sizes" {
    const f = try stackFootprint(16384, 11, 0);
    try std.testing.expectEqual(@as(usize, 738_287_616 * 2 * 2), f.shadow_bytes);
    try std.testing.expectEqual(@as(usize, 738_287_616 * 6 * 4), f.master_bytes);
}

test "stack footprint validates dimensions and mirror count" {
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, stackFootprint(0, 2, 0));
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, stackFootprint(15, 2, 0));
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, stackFootprint(16, 0, 0));
    try std.testing.expectError(MemoryEstimateError.InvalidDimensions, stackFootprint(16, 2, 3));
}

test "stack footprint reports overflow rather than wrapping" {
    try std.testing.expectError(
        MemoryEstimateError.SizeOverflow,
        stackFootprint(std.math.maxInt(usize) - 1, 2, 0),
    );
}
