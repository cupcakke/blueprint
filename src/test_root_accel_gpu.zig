const std = @import("std");
const accel = @import("hw/accel/accel_interface.zig");
const gpu_memory_model = @import("hw/accel/gpu_memory_model.zig");

test "stack only initialization allocates zero per-layer mirrors" {
    const allocator = std.heap.page_allocator;
    var accelerator = try accel.RSFAccelerator.initStackOnly(64, 3, allocator, true);
    defer accelerator.deinit();
    try std.testing.expectEqual(accel.RSFOwnershipMode.stack_only, accelerator.ownership_mode);
    try std.testing.expectEqual(@as(usize, 0), accelerator.mirror_device_allocations);
    try std.testing.expectEqual(@as(usize, 0), accelerator.layers.len);
    try std.testing.expect(accelerator.stack_arrays_valid);
    try std.testing.expect(accelerator.stack_weights_s != null);
    try std.testing.expect(accelerator.stack_weights_t != null);
    try std.testing.expect(accelerator.stack_master_weights_s != null);
    try std.testing.expect(accelerator.stack_master_weights_t != null);
    try std.testing.expect(accelerator.stack_momentum_s != null);
    try std.testing.expect(accelerator.stack_momentum_t != null);
    try std.testing.expect(accelerator.stack_fisher_s != null);
    try std.testing.expect(accelerator.stack_fisher_t != null);
}

test "stack only mode rejects mirror-only entry points" {
    const allocator = std.heap.page_allocator;
    var accelerator = try accel.RSFAccelerator.initStackOnly(32, 2, allocator, true);
    defer accelerator.deinit();
    try std.testing.expectError(accel.AccelError.StackOnlyModeForbidden, accelerator.syncLayersFromStack());
    try std.testing.expectError(accel.AccelError.MirrorStateUnavailable, accelerator.layerPtr(0));
    var input = try accel.FutharkArray2DF16.newZeros(&accelerator.ctx, 4, 32, allocator);
    defer input.free(&accelerator.ctx);
    try std.testing.expectError(accel.AccelError.StackOnlyModeForbidden, accelerator.forward(&input));
    try std.testing.expectError(accel.AccelError.StackOnlyModeForbidden, accelerator.setLayerWeightsS(0, &[_]f16{0}, 1, 1));
    try std.testing.expectError(accel.AccelError.StackOnlyModeForbidden, accelerator.setLayerWeightsT(0, &[_]f16{0}, 1, 1));
}

test "stack forward and stack inverse reconstruct the input" {
    const allocator = std.heap.page_allocator;
    var accelerator = try accel.RSFAccelerator.initStackOnly(64, 3, allocator, true);
    defer accelerator.deinit();

    const batch: usize = 2;
    const seq: usize = 5;
    const dim: usize = 64;
    const total = batch * seq * dim;
    const host_input = try allocator.alloc(f16, total);
    defer allocator.free(host_input);
    var rng = std.Random.DefaultPrng.init(0x51525354);
    for (host_input) |*value| {
        value.* = @floatCast((rng.random().float(f32) - 0.5) * 0.4);
    }

    var inputs = try accel.FutharkArray3DF16.newFromFlat(&accelerator.ctx, host_input, batch, seq, dim);
    defer inputs.free(&accelerator.ctx);
    var outputs = try accelerator.stackForward(&inputs);
    defer outputs.free(&accelerator.ctx);
    var reconstructed = try accelerator.stackInverse(&outputs);
    defer reconstructed.free(&accelerator.ctx);
    try accelerator.sync();

    const flat_out = try outputs.valuesFlat(&accelerator.ctx, allocator);
    defer allocator.free(flat_out);
    const flat_recon = try reconstructed.valuesFlat(&accelerator.ctx, allocator);
    defer allocator.free(flat_recon);
    try std.testing.expectEqual(host_input.len, flat_out.len);
    try std.testing.expectEqual(host_input.len, flat_recon.len);
    for (host_input, flat_recon) |expected, actual| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), @as(f32, @floatCast(actual)), 0.05);
    }
}

test "optimizer state round trip in stack only mode" {
    const allocator = std.heap.page_allocator;
    var accelerator = try accel.RSFAccelerator.initStackOnly(32, 2, allocator, true);
    defer accelerator.deinit();

    var state = try accelerator.readOptimizerState(allocator);
    defer state.deinit();
    const per_stack = 2 * (32 / 2) * (32 / 2 + 1);
    try std.testing.expectEqual(@as(usize, per_stack), state.master_weights_s.len);
    try std.testing.expectEqual(@as(usize, per_stack), state.master_weights_t.len);
    try std.testing.expectEqual(@as(usize, per_stack), state.momentum_s.len);
    try std.testing.expectEqual(@as(usize, per_stack), state.fisher_s.len);

    const master_s = try allocator.alloc(f32, per_stack);
    defer allocator.free(master_s);
    const master_t = try allocator.alloc(f32, per_stack);
    defer allocator.free(master_t);
    const momentum = try allocator.alloc(f32, per_stack);
    defer allocator.free(momentum);
    const fisher = try allocator.alloc(f32, per_stack);
    defer allocator.free(fisher);
    for (master_s, 0..) |*value, index| value.* = 0.01 * @as(f32, @floatFromInt(index % 17));
    for (master_t, 0..) |*value, index| value.* = -0.01 * @as(f32, @floatFromInt(index % 13));
    @memset(momentum, 0.0);
    @memset(fisher, 0.0);
    try accelerator.setOptimizerState(master_s, master_t, momentum, momentum, fisher, fisher, 41);

    var reloaded = try accelerator.readOptimizerState(allocator);
    defer reloaded.deinit();
    try std.testing.expectEqual(@as(u64, 41), reloaded.step);
    for (master_s, reloaded.master_weights_s) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }
    try std.testing.expectEqual(accel.RSFOwnershipMode.stack_only, accelerator.ownership_mode);
    try std.testing.expectEqual(@as(usize, 0), accelerator.mirror_device_allocations);
}

test "frozen embedding allocates forward-only state and matches source lookup" {
    const allocator = std.heap.page_allocator;
    var context = try accel.FutharkContext.init();
    defer context.deinit();

    const vocab: usize = 97;
    const dim: usize = 32;
    const total = vocab * dim;
    const host_weights = try allocator.alloc(f16, total);
    defer allocator.free(host_weights);
    var rng = std.Random.DefaultPrng.init(0x600D5EED);
    for (host_weights) |*value| {
        value.* = @floatCast((rng.random().float(f32) - 0.5) * 0.02);
    }

    var trainable = try accel.EmbeddingAccelerator.initWithWeights(&context, allocator, vocab, dim, host_weights);
    defer trainable.deinit();
    var frozen = try accel.FrozenEmbeddingAccelerator.initFromWeightF16(&context, vocab, dim, host_weights);
    defer frozen.deinit();

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 96 };
    const lengths = [_]usize{ 3, 5 };
    var from_trainable = try trainable.forwardPadded(&tokens, &lengths, 4);
    defer from_trainable.free(&context);
    var from_frozen = try frozen.forwardPadded(&tokens, &lengths, 4, allocator);
    defer from_frozen.free(&context);
    try context.sync();
    const flat_trainable = try from_trainable.valuesFlat(&context, allocator);
    defer allocator.free(flat_trainable);
    const flat_frozen = try from_frozen.valuesFlat(&context, allocator);
    defer allocator.free(flat_frozen);
    try std.testing.expectEqual(flat_trainable.len, flat_frozen.len);
    for (flat_trainable, flat_frozen) |expected, actual| {
        try std.testing.expectEqual(expected, actual);
    }

    const exported = try frozen.exportMasterWeightsTemporary(allocator);
    defer allocator.free(exported);
    try std.testing.expectEqual(total, exported.len);
    for (host_weights, exported) |f16_value, f32_value| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatCast(f16_value)), f32_value, 1e-6);
    }
}

test "frozen embedding restores from fp32 masters deterministically" {
    const allocator = std.heap.page_allocator;
    var context = try accel.FutharkContext.init();
    defer context.deinit();

    const vocab: usize = 31;
    const dim: usize = 16;
    const total = vocab * dim;
    const masters = try allocator.alloc(f32, total);
    defer allocator.free(masters);
    for (masters, 0..) |*value, index| value.* = @floatCast(@mod(@as(f32, @floatFromInt(index)) * 0.031, 0.5) - 0.25);

    var frozen = try accel.FrozenEmbeddingAccelerator.initFromMasterWeightsF32(&context, vocab, dim, masters);
    defer frozen.deinit();
    try context.sync();
    const restored = try frozen.exportMasterWeightsTemporary(allocator);
    defer allocator.free(restored);
    for (masters, restored) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.01);
    }
}

test "chunked graph encoding is equivalent to single-shot encoding" {
    const allocator = std.heap.page_allocator;
    var context = try accel.FutharkContext.init();
    defer context.deinit();

    var rng = std.Random.DefaultPrng.init(0xABCD1234);
    const hashes = try allocator.alloc(u64, 17);
    defer allocator.free(hashes);
    for (hashes) |*value| {
        value.* = rng.random().int(u64);
    }

    var single = try accel.batchEncodeGraph(&context, hashes, 42, allocator, 17);
    defer single.deinit();
    var chunked = try accel.batchEncodeGraph(&context, hashes, 42, allocator, 4);
    defer chunked.deinit();

    try std.testing.expectEqual(single.node_count, chunked.node_count);
    try std.testing.expectEqual(single.edge_count, chunked.edge_count);
    try std.testing.expectEqualSlices(u64, single.hashes, chunked.hashes);
    try std.testing.expectEqualSlices(f32, single.re_a, chunked.re_a);
    try std.testing.expectEqualSlices(f32, single.im_a, chunked.im_a);
    try std.testing.expectEqualSlices(f32, single.re_b, chunked.re_b);
    try std.testing.expectEqualSlices(f32, single.im_b, chunked.im_b);
    try std.testing.expectEqualSlices(i64, single.edge_srcs, chunked.edge_srcs);
    try std.testing.expectEqualSlices(i64, single.edge_tgts, chunked.edge_tgts);
}

test "graph chunk size validation rejects zero and oversize chunks" {
    const allocator = std.heap.page_allocator;
    var context = try accel.FutharkContext.init();
    defer context.deinit();
    const hashes = [_]u64{ 1, 2, 3 };
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.batchEncodeGraph(&context, &hashes, 7, allocator, 0));
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.validateGraphChunkSize(0));
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.validateGraphChunkSize(16 * 1024 * 1024 + 1));
    try std.testing.expectEqual(@as(usize, 65536), try accel.validateGraphChunkSize(65536));
}

test "invalid dimensions fail before any device allocation" {
    const allocator = std.heap.page_allocator;
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.RSFAccelerator.initStackOnly(0, 2, allocator, true));
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.RSFAccelerator.initStackOnly(31, 2, allocator, true));
    try std.testing.expectError(accel.AccelError.InvalidDimensions, accel.RSFAccelerator.initStackOnly(32, 0, allocator, true));
}

test "estimator matches stack-only trainer configuration" {
    const estimate = try gpu_memory_model.estimate(.{
        .rsf = .{ .layout = .{ .model_dim = 1024, .num_layers = 4 } },
        .embedding = .{ .layout = .{ .vocab_size = 512, .model_dim = 1024 }, .frozen_target_fp16 = true },
        .batch = .{ .layout = .{ .batch_size = 8, .max_seq_len = 64, .model_dim = 1024 } },
    });
    try std.testing.expectEqual(@as(u64, 0), estimate.rsf_legacy_mirror_fp16_bytes);
    try std.testing.expect(estimate.rsf_master_fp32_bytes > 0);
    try std.testing.expect(estimate.frozen_target_fp16_bytes > 0);
    try std.testing.expectEqual(@as(u64, 0), estimate.frozen_target_fp32_master_bytes);
}
