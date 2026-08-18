const std = @import("std");

const _build_gpu_enabled: bool = blk: {
    const opts = @import("build_options");
    if (@hasDecl(opts, "gpu_acceleration")) break :blk opts.gpu_acceleration;
    break :blk false;
};

pub const abi = @import("futhark_abi.zig");

pub usingnamespace abi;

pub const struct_futhark_opaque_tup6_fused_stack_gradients = if (abi.tuple_outputs_flattened)
    opaque {}
else
    abi.struct_futhark_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32;

pub const struct_futhark_opaque_tup3_stack_sfd = if (abi.tuple_outputs_flattened)
    opaque {}
else
    abi.struct_futhark_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32;

pub const struct_futhark_opaque_tup3_stack_spectral = if (abi.tuple_outputs_flattened)
    opaque {}
else
    abi.struct_futhark_opaque_tup3_arr3d_f32_f32_f32;

pub const struct_futhark_opaque_tup5_embedding_spectral = if (abi.tuple_outputs_flattened)
    opaque {}
else
    abi.struct_futhark_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32;

pub const struct_futhark_opaque_tup7_graph_encode = if (abi.tuple_outputs_flattened)
    opaque {}
else
    abi.struct_futhark_opaque_tup7_arr1d_u64_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_f32_arr1d_i64_arr1d_i64;

pub const GpuConfigurationError = error{GpuAccelerationDisabled};

pub const gpu_default_group_size: c_int = 256;
pub const gpu_default_num_groups: c_int = 4096;
pub const gpu_default_tile_size: c_int = 128;
pub const gpu_arch_sm: c_int = 100;

pub extern "c" fn futhark_context_config_set_device(cfg: ?*abi.struct_futhark_context_config, device: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*abi.struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*abi.struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*abi.struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_cache_file(cfg: ?*abi.struct_futhark_context_config, path: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_unified_memory(cfg: ?*abi.struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_logging(cfg: ?*abi.struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_debugging(cfg: ?*abi.struct_futhark_context_config, flag: c_int) void;
pub extern "c" fn futhark_context_config_set_profiling(cfg: ?*abi.struct_futhark_context_config, flag: c_int) void;

pub const unified_memory_disabled: c_int = 0;

pub const GpuContextOptions = struct {
    cache_file: ?[*:0]const u8 = null,
    unified_memory: c_int = unified_memory_disabled,
    logging: bool = false,
    debugging: bool = false,
    profiling: bool = false,
    group_size: c_int = gpu_default_group_size,
    num_groups: c_int = gpu_default_num_groups,
    tile_size: c_int = gpu_default_tile_size,
};

pub fn envFlag(name: []const u8) bool {
    var buf: [64]u8 = undefined;
    const raw = std.posix.getenv(name) orelse return false;
    if (raw.len == 0 or raw.len >= buf.len) return false;
    const lowered = std.ascii.lowerString(buf[0..raw.len], raw);
    return std.mem.eql(u8, lowered, "1") or
        std.mem.eql(u8, lowered, "true") or
        std.mem.eql(u8, lowered, "yes") or
        std.mem.eql(u8, lowered, "on");
}

pub fn gpuContextOptionsFromEnv(cache_file: ?[*:0]const u8) GpuContextOptions {
    return .{
        .cache_file = cache_file,
        .unified_memory = if (envFlag("JAIDE_FUTHARK_UNIFIED_MEMORY")) 1 else unified_memory_disabled,
        .logging = envFlag("JAIDE_FUTHARK_LOG"),
        .debugging = envFlag("JAIDE_FUTHARK_DEBUG"),
        .profiling = envFlag("JAIDE_FUTHARK_PROFILE"),
    };
}

pub fn configureGpuContextOpts(
    cfg: ?*abi.struct_futhark_context_config,
    options: GpuContextOptions,
) GpuConfigurationError!void {
    if (comptime !_build_gpu_enabled) return error.GpuAccelerationDisabled;
    futhark_context_config_set_device(cfg, "");
    futhark_context_config_set_unified_memory(cfg, options.unified_memory);
    futhark_context_config_set_default_group_size(cfg, options.group_size);
    futhark_context_config_set_default_num_groups(cfg, options.num_groups);
    futhark_context_config_set_default_tile_size(cfg, options.tile_size);
    if (options.logging) futhark_context_config_set_logging(cfg, 1);
    if (options.debugging) futhark_context_config_set_debugging(cfg, 1);
    if (options.profiling) futhark_context_config_set_profiling(cfg, 1);
    if (options.cache_file) |path| futhark_context_config_set_cache_file(cfg, path);
}

pub fn configureGpuContext(
    cfg: ?*abi.struct_futhark_context_config,
    cache_file: ?[*:0]const u8,
) GpuConfigurationError!void {
    return configureGpuContextOpts(cfg, gpuContextOptionsFromEnv(cache_file));
}
