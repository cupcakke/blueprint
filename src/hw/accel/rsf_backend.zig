const std = @import("std");

pub const RsfBackendKind = enum {
    futhark_kernels,
    host_reference,
    cublas_gemm,
};

pub const BackendError = error{
    UnknownBackendName,
    BackendRequiresRegeneratedKernels,
};

pub const backend_futhark_kernels_name = "futhark";
pub const backend_host_reference_name = "reference";
pub const backend_cublas_gemm_name = "cublas";

pub fn parseBackendKind(name: []const u8) BackendError!RsfBackendKind {
    if (std.mem.eql(u8, name, backend_futhark_kernels_name) or std.mem.eql(u8, name, "futhark_kernels")) return .futhark_kernels;
    if (std.mem.eql(u8, name, backend_host_reference_name) or std.mem.eql(u8, name, "host_reference")) return .host_reference;
    if (std.mem.eql(u8, name, backend_cublas_gemm_name) or std.mem.eql(u8, name, "cublas_gemm")) return .cublas_gemm;
    return BackendError.UnknownBackendName;
}

pub fn backendKindName(kind: RsfBackendKind) []const u8 {
    return switch (kind) {
        .futhark_kernels => backend_futhark_kernels_name,
        .host_reference => backend_host_reference_name,
        .cublas_gemm => backend_cublas_gemm_name,
    };
}

pub const BackendSelection = struct {
    kind: RsfBackendKind,
    available: bool,
    requires_regenerated_kernels: bool,
    detail: []const u8,
};

pub fn resolveBackendKind(kind: RsfBackendKind, compact_rows_available: bool) BackendSelection {
    return switch (kind) {
        .futhark_kernels => .{
            .kind = .futhark_kernels,
            .available = true,
            .requires_regenerated_kernels = false,
            .detail = "futhark fused stack kernels execute forward, inverse, fused backward, and optimizer updates",
        },
        .host_reference => .{
            .kind = .host_reference,
            .available = compact_rows_available,
            .requires_regenerated_kernels = false,
            .detail = "host reference backend executes the identical coupling math on active rows for correctness testing only",
        },
        .cublas_gemm => .{
            .kind = .cublas_gemm,
            .available = false,
            .requires_regenerated_kernels = true,
            .detail = "cublas_gemm backend requires regenerated Futhark kernels exporting per-layer elementwise stages and persistent spectral vectors; select an available backend instead",
        },
    };
}

pub fn validateProductionSelection(selection: BackendSelection) BackendError!void {
    if (selection.kind == .cublas_gemm) return BackendError.BackendRequiresRegeneratedKernels;
}

pub fn formatSelectionLine(buffer: []u8, selection: BackendSelection) []const u8 {
    return std.fmt.bufPrint(
        buffer,
        "[rsf-backend] selected={s} available={any} requires_regenerated_kernels={any} detail={s}",
        .{ backendKindName(selection.kind), selection.available, selection.requires_regenerated_kernels, selection.detail },
    ) catch buffer[0..0];
}

pub fn resolveFromEnvironmentName(environment_name: ?[]const u8) BackendError!RsfBackendKind {
    const name = environment_name orelse return .futhark_kernels;
    return parseBackendKind(name);
}

test "backend names parse and reject unknown values" {
    try std.testing.expectEqual(RsfBackendKind.futhark_kernels, try parseBackendKind("futhark"));
    try std.testing.expectEqual(RsfBackendKind.futhark_kernels, try parseBackendKind("futhark_kernels"));
    try std.testing.expectEqual(RsfBackendKind.host_reference, try parseBackendKind("reference"));
    try std.testing.expectEqual(RsfBackendKind.cublas_gemm, try parseBackendKind("cublas"));
    try std.testing.expectError(BackendError.UnknownBackendName, parseBackendKind("tensorcore"));
    try std.testing.expectError(BackendError.UnknownBackendName, parseBackendKind(""));
}

test "default backend is futhark kernels when environment is unset" {
    try std.testing.expectEqual(RsfBackendKind.futhark_kernels, try resolveFromEnvironmentName(null));
}

test "cublas selection is classified as requiring regenerated kernels" {
    const selection = resolveBackendKind(.cublas_gemm, true);
    try std.testing.expect(!selection.available);
    try std.testing.expect(selection.requires_regenerated_kernels);
    try std.testing.expectError(BackendError.BackendRequiresRegeneratedKernels, validateProductionSelection(selection));
}

test "futhark selection is available and validated" {
    const selection = resolveBackendKind(.futhark_kernels, true);
    try std.testing.expect(selection.available);
    try std.testing.expect(!selection.requires_regenerated_kernels);
    try validateProductionSelection(selection);
}

test "host reference availability depends on compact rows" {
    const with_compact = resolveBackendKind(.host_reference, true);
    const without_compact = resolveBackendKind(.host_reference, false);
    try std.testing.expect(with_compact.available);
    try std.testing.expect(!without_compact.available);
}

test "selection telemetry line names the backend" {
    var buffer: [512]u8 = undefined;
    const selection = resolveBackendKind(.futhark_kernels, true);
    const line = formatSelectionLine(&buffer, selection);
    try std.testing.expect(std.mem.indexOf(u8, line, "selected=futhark") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "available=true") != null);
}
