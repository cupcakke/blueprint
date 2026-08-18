const std = @import("std");

pub const HeartbeatConfig = struct {
    interval_sec: u64 = 30,
    rank: usize = 0,
    enabled: bool = true,
};

pub const PhaseHeartbeat = struct {
    mutex: std.Thread.Mutex = .{},
    phase_name: []const u8 = "",
    phase_started_ns: i128 = 0,
    heartbeat_started_ns: i128 = 0,
    total_heartbeats: u64 = 0,
    stop_flag: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,
    config: HeartbeatConfig = .{},
    allocator: std.mem.Allocator,

    const Self = @This();

    fn elapsedSecondsSince(ns: i128) u64 {
        const now = std.time.nanoTimestamp();
        if (now <= ns) return 0;
        return @intCast(@divTrunc(now - ns, std.time.ns_per_s));
    }

    fn loopFn(self: *Self) void {
        while (!self.stop_flag.load(.acquire)) {
            var waited: u64 = 0;
            while (waited < self.config.interval_sec) {
                if (self.stop_flag.load(.acquire)) return;
                std.time.sleep(std.time.ns_per_s);
                waited += 1;
            }
            if (self.stop_flag.load(.acquire)) return;
            self.mutex.lock();
            const phase_name = self.phase_name;
            const phase_elapsed = elapsedSecondsSince(self.phase_started_ns);
            const total_elapsed = elapsedSecondsSince(self.heartbeat_started_ns);
            self.total_heartbeats += 1;
            self.mutex.unlock();
            if (self.config.enabled) {
                std.debug.print(
                    "[heartbeat] rank={d} phase={s} phase_elapsed_s={d} total_elapsed_s={d} beat={d}\n",
                    .{ self.config.rank, phase_name, phase_elapsed, total_elapsed, self.total_heartbeats },
                );
            }
        }
    }

    pub fn start(allocator: std.mem.Allocator, config: HeartbeatConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .phase_started_ns = std.time.nanoTimestamp(),
            .heartbeat_started_ns = std.time.nanoTimestamp(),
        };
        if (config.interval_sec == 0) {
            self.config.enabled = false;
            return self;
        }
        self.thread = std.Thread.spawn(.{}, loopFn, .{self}) catch {
            self.config.enabled = false;
            return self;
        };
        return self;
    }

    pub fn setPhase(self: *Self, phase_name: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.phase_name = phase_name;
        self.phase_started_ns = std.time.nanoTimestamp();
    }

    pub fn totalHeartbeats(self: *Self) u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.total_heartbeats;
    }

    pub fn stop(self: *Self) void {
        self.stop_flag.store(true, .release);
        if (self.thread) |thread| {
            thread.join();
            self.thread = null;
        }
        self.allocator.destroy(self);
    }
};

pub fn formatHeartbeatLine(buffer: []u8, rank: usize, phase_name: []const u8, phase_elapsed_s: u64, total_elapsed_s: u64, beat: u64) []const u8 {
    return std.fmt.bufPrint(buffer, "[heartbeat] rank={d} phase={s} phase_elapsed_s={d} total_elapsed_s={d} beat={d}", .{ rank, phase_name, phase_elapsed_s, total_elapsed_s, beat }) catch buffer[0..0];
}

test "heartbeat line format contains phase and elapsed values" {
    var buffer: [256]u8 = undefined;
    const line = formatHeartbeatLine(&buffer, 0, "stack_rsf_allocation", 12, 45, 3);
    try std.testing.expect(std.mem.indexOf(u8, line, "phase=stack_rsf_allocation") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "phase_elapsed_s=12") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "total_elapsed_s=45") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "beat=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "rank=0") != null);
}

test "heartbeat start and stop with zero interval disables thread" {
    const allocator = std.testing.allocator;
    var heartbeat = try PhaseHeartbeat.start(allocator, .{ .interval_sec = 0, .rank = 0 });
    heartbeat.setPhase("dataset_load");
    try std.testing.expect(!heartbeat.config.enabled);
    try std.testing.expectEqualStrings("dataset_load", heartbeat.phase_name);
    heartbeat.stop();
}

test "heartbeat thread runs and reports beats" {
    const allocator = std.testing.allocator;
    var heartbeat = try PhaseHeartbeat.start(allocator, .{ .interval_sec = 1, .rank = 0 });
    heartbeat.setPhase("graph_construction");
    std.time.sleep(3 * std.time.ns_per_s);
    try std.testing.expect(heartbeat.totalHeartbeats() >= 1);
    heartbeat.stop();
}
