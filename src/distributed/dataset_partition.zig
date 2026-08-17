const std = @import("std");

pub const Bounds = struct {
    start: usize,
    count: usize,
};

pub const Error = error{
    EmptyDataset,
    InvalidWorldSize,
    InvalidRank,
    Overflow,
};

pub fn bounds(total: usize, world_size: usize, rank: usize) Error!Bounds {
    if (total == 0) return Error.EmptyDataset;
    if (world_size == 0) return Error.InvalidWorldSize;
    if (rank >= world_size) return Error.InvalidRank;
    const base = total / world_size;
    const remainder = total % world_size;
    const count = base + @intFromBool(rank < remainder);
    const start = if (rank < remainder)
        std.math.mul(usize, rank, base + 1) catch return Error.Overflow
    else
        std.math.add(
            usize,
            std.math.mul(usize, remainder, base + 1) catch return Error.Overflow,
            std.math.mul(usize, rank - remainder, base) catch return Error.Overflow,
        ) catch return Error.Overflow;
    return .{ .start = start, .count = count };
}

test "dataset partitions cover every sample exactly once" {
    const total: usize = 23;
    const world_size: usize = 5;
    var covered = [_]bool{false} ** total;
    var rank: usize = 0;
    while (rank < world_size) : (rank += 1) {
        const partition = try bounds(total, world_size, rank);
        var index = partition.start;
        while (index < partition.start + partition.count) : (index += 1) {
            try std.testing.expect(!covered[index]);
            covered[index] = true;
        }
    }
    for (covered) |present| try std.testing.expect(present);
}

test "dataset partitions permit idle ranks without duplication" {
    const total: usize = 2;
    const world_size: usize = 4;
    try std.testing.expectEqual(Bounds{ .start = 0, .count = 1 }, try bounds(total, world_size, 0));
    try std.testing.expectEqual(Bounds{ .start = 1, .count = 1 }, try bounds(total, world_size, 1));
    try std.testing.expectEqual(Bounds{ .start = 2, .count = 0 }, try bounds(total, world_size, 2));
    try std.testing.expectEqual(Bounds{ .start = 2, .count = 0 }, try bounds(total, world_size, 3));
}

test "dataset partition validates topology" {
    try std.testing.expectError(Error.EmptyDataset, bounds(0, 1, 0));
    try std.testing.expectError(Error.InvalidWorldSize, bounds(1, 0, 0));
    try std.testing.expectError(Error.InvalidRank, bounds(1, 1, 1));
}
