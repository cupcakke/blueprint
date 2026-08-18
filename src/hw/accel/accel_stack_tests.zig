const std = @import("std");
const accel = @import("accel_interface.zig");
const gpu_memory = @import("gpu_memory.zig");

const test_model_dim: usize = 8;
const test_num_layers: usize = 3;

fn expectStackOnlyError(result: anytype) !void {
    try std.testing.expectError(accel.AccelError.StackOnlyModeActive, result);
}

test "stack only accelerator allocates no per layer mirrors" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    try std.testing.expect(acc.isStackOnly());
    try std.testing.expectEqual(@as(usize, 0), acc.mirrorArrayCount());
    try std.testing.expectEqual(@as(usize, 0), acc.layers.len);
    try std.testing.expectEqual(test_num_layers, acc.numLayers());
    try std.testing.expect(acc.stack_weights_s != null);
    try std.testing.expect(acc.stack_weights_t != null);
    try std.testing.expect(acc.stack_master_weights_s != null);
    try std.testing.expect(acc.stack_master_weights_t != null);
    try std.testing.expect(acc.stack_momentum_s != null);
    try std.testing.expect(acc.stack_momentum_t != null);
    try std.testing.expect(acc.stack_fisher_s != null);
    try std.testing.expect(acc.stack_fisher_t != null);
}

test "mirrored accelerator still allocates two arrays per layer" {
    var acc = try accel.RSFAccelerator.initMultiLayerWithDepthScale(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    try std.testing.expect(!acc.isStackOnly());
    try std.testing.expectEqual(test_num_layers * 2, acc.mirrorArrayCount());
    try std.testing.expectEqual(test_num_layers, acc.layers.len);
}

test "stack only device bytes match the shared estimator and exclude mirrors" {
    var lean = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer lean.deinit();

    var mirrored = try accel.RSFAccelerator.initMultiLayerWithDepthScale(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer mirrored.deinit();

    const lean_expected = try gpu_memory.stackFootprint(test_model_dim, test_num_layers, 0);
    const mirrored_expected = try gpu_memory.stackFootprint(test_model_dim, test_num_layers, test_num_layers);

    try std.testing.expectEqual(lean_expected.total(), try lean.deviceStackBytes());
    try std.testing.expectEqual(mirrored_expected.total(), try mirrored.deviceStackBytes());
    try std.testing.expectEqual(@as(usize, 0), lean_expected.mirror_bytes);
    try std.testing.expect(mirrored_expected.mirror_bytes > 0);
    try std.testing.expect((try lean.deviceStackBytes()) < (try mirrored.deviceStackBytes()));
}

test "stack only accelerator rejects every mirror facing entry point" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    const half = test_model_dim / 2;
    const cols = half + 1;
    const weights = try std.testing.allocator.alloc(f16, half * cols);
    defer std.testing.allocator.free(weights);
    @memset(weights, 0.0);

    try expectStackOnlyError(acc.layerPtr(0));
    try expectStackOnlyError(acc.setLayerWeightsS(0, weights, half, cols));
    try expectStackOnlyError(acc.setLayerWeightsT(0, weights, half, cols));

    const inputs = try std.testing.allocator.alloc(f16, 2 * test_model_dim);
    defer std.testing.allocator.free(inputs);
    @memset(inputs, 0.25);
    var input_array = try accel.FutharkArray2DF16.newFromFlat(&acc.ctx, inputs, 2, test_model_dim);
    defer input_array.free(&acc.ctx);

    try expectStackOnlyError(acc.forward(&input_array));
}

test "mirror facing entry points stay available in mirrored mode" {
    var acc = try accel.RSFAccelerator.initMultiLayerWithDepthScale(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    const half = test_model_dim / 2;
    const cols = half + 1;
    const weights = try std.testing.allocator.alloc(f16, half * cols);
    defer std.testing.allocator.free(weights);
    @memset(weights, 0.125);

    const layer = try acc.layerPtr(0);
    try std.testing.expectEqual(half, layer.weights_s.rows);
    try std.testing.expectEqual(cols, layer.weights_s.cols);

    try acc.setLayerWeightsS(0, weights, half, cols);
    try acc.setLayerWeightsT(0, weights, half, cols);

    const readback = try acc.layers[0].weights_s.valuesFlat(&acc.ctx, std.testing.allocator);
    defer std.testing.allocator.free(readback);
    for (readback) |value| try std.testing.expectEqual(@as(f16, 0.125), value);
}

test "sync layers from stack is a no op under stack only mode" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    acc.layers_mirror_valid = false;
    try acc.syncLayersFromStack();
    try std.testing.expect(!acc.layers_mirror_valid);
    try std.testing.expectEqual(@as(usize, 0), acc.mirrorArrayCount());
}

test "stack only optimizer state round trips through the stacks" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    const half = test_model_dim / 2;
    const cols = half + 1;
    const total = test_num_layers * half * cols;

    var state = try acc.readOptimizerState(std.testing.allocator);
    defer state.deinit();

    try std.testing.expectEqual(total, state.master_weights_s.len);
    try std.testing.expectEqual(total, state.master_weights_t.len);
    try std.testing.expectEqual(total, state.momentum_s.len);
    try std.testing.expectEqual(total, state.momentum_t.len);
    try std.testing.expectEqual(total, state.fisher_s.len);
    try std.testing.expectEqual(total, state.fisher_t.len);
    try std.testing.expectEqual(@as(u64, 0), state.step);

    for (state.momentum_s) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    for (state.fisher_t) |value| try std.testing.expectEqual(@as(f32, 0.0), value);

    var nonzero_masters: usize = 0;
    for (state.master_weights_s) |value| {
        try std.testing.expect(std.math.isFinite(value));
        if (value != 0.0) nonzero_masters += 1;
    }
    try std.testing.expect(nonzero_masters > 0);

    var row: usize = 0;
    while (row < half) : (row += 1) {
        try std.testing.expectEqual(@as(f32, 0.0), state.master_weights_s[row * cols + half]);
        try std.testing.expectEqual(@as(f32, 0.0), state.master_weights_t[row * cols + half]);
    }
}

test "stack only initialization matches mirrored initialization numerically" {
    var lean = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer lean.deinit();

    var mirrored = try accel.RSFAccelerator.initMultiLayerWithDepthScale(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer mirrored.deinit();

    var lean_state = try lean.readOptimizerState(std.testing.allocator);
    defer lean_state.deinit();
    var mirrored_state = try mirrored.readOptimizerState(std.testing.allocator);
    defer mirrored_state.deinit();

    try std.testing.expectEqualSlices(f32, mirrored_state.master_weights_s, lean_state.master_weights_s);
    try std.testing.expectEqualSlices(f32, mirrored_state.master_weights_t, lean_state.master_weights_t);
}

test "stack only shadow stacks agree with the master stacks" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    const shadow_s = try acc.stack_weights_s.?.valuesFlat(&acc.ctx, std.testing.allocator);
    defer std.testing.allocator.free(shadow_s);
    const master_s = try acc.stack_master_weights_s.?.valuesFlat(&acc.ctx, std.testing.allocator);
    defer std.testing.allocator.free(master_s);

    try std.testing.expectEqual(master_s.len, shadow_s.len);
    for (shadow_s, master_s) |shadow, master| {
        const widened: f32 = @floatCast(shadow);
        try std.testing.expectApproxEqAbs(master, widened, 1.0e-3);
    }
}

test "stack only initialization cleans up after host allocation failure" {
    var fail_index: usize = 0;
    while (fail_index < 8) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const allocator = failing.allocator();
        if (accel.RSFAccelerator.initStackOnly(test_model_dim, test_num_layers, allocator, true)) |created| {
            var acc = created;
            acc.deinit();
        } else |err| {
            try std.testing.expectEqual(accel.AccelError.AllocationFailed, err);
        }
        try std.testing.expectEqual(failing.allocations, failing.deallocations);
    }
}

test "stack only clip range configuration is preserved" {
    var acc = try accel.RSFAccelerator.initStackOnly(
        test_model_dim,
        test_num_layers,
        std.testing.allocator,
        true,
    );
    defer acc.deinit();

    try acc.setClipRange(-3.0, 3.0);
    try std.testing.expectEqual(@as(f16, -3.0), acc.clip_min);
    try std.testing.expectEqual(@as(f16, 3.0), acc.clip_max);
    try std.testing.expectError(accel.AccelError.InvalidClipRange, acc.setClipRange(3.0, -3.0));
}

test "stack only initialization rejects invalid shapes" {
    try std.testing.expectError(
        accel.AccelError.InvalidDimensions,
        accel.RSFAccelerator.initStackOnly(0, 2, std.testing.allocator, true),
    );
    try std.testing.expectError(
        accel.AccelError.InvalidDimensions,
        accel.RSFAccelerator.initStackOnly(7, 2, std.testing.allocator, true),
    );
    try std.testing.expectError(
        accel.AccelError.InvalidDimensions,
        accel.RSFAccelerator.initStackOnly(8, 0, std.testing.allocator, true),
    );
}
