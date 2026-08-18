#include "main_gpu.h"
#include <stdbool.h>
#include <stdint.h>

typedef int (*backward_gradients_type)(
    struct futhark_context *,
    struct futhark_opaque_tup6_arr3d_f32_arr3d_f32_arr3d_f16_f32_f32_f32 **,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    const struct futhark_i64_1d *,
    const struct futhark_f16_3d *,
    const struct futhark_f16_3d *,
    bool,
    float,
    float,
    float,
    float,
    float,
    float
);

typedef int (*stack_sfd_type)(
    struct futhark_context *,
    struct futhark_opaque_tup3_arr3d_f32_arr3d_f32_arr3d_f32 **,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    const struct futhark_f32_3d *,
    float,
    float,
    float,
    int64_t,
    float,
    float,
    float
);

typedef int (*embedding_backward_padded_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_i64_1d *,
    const struct futhark_i64_1d *,
    const struct futhark_f16_3d *,
    const struct futhark_f32_2d *
);

typedef int (*clip_matrix_type)(
    struct futhark_context *,
    struct futhark_f32_2d **,
    const struct futhark_f32_2d *,
    float
);

typedef int (*stack_spectral_type)(
    struct futhark_context *,
    struct futhark_opaque_tup3_arr3d_f32_f32_f32 **,
    const struct futhark_f32_3d *,
    float,
    int64_t
);

typedef int (*embedding_spectral_type)(
    struct futhark_context *,
    struct futhark_opaque_tup5_arr2d_f32_arr1d_f32_arr1d_f32_f32_f32 **,
    const struct futhark_f32_2d *,
    const struct futhark_f32_1d *,
    const struct futhark_f32_1d *,
    int64_t,
    float
);

_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_rsf_stack_backward_gradients_fused), backward_gradients_type), "rsf_stack_backward_gradients_fused ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_stack_update_sfd_master), stack_sfd_type), "stack_update_sfd_master ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_backward_padded), embedding_backward_padded_type), "embedding_backward_padded ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_clip_matrix_global_norm_f32), clip_matrix_type), "clip_matrix_global_norm_f32 ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_stack_spectral_normalize), stack_spectral_type), "stack_spectral_normalize ABI mismatch");
_Static_assert(__builtin_types_compatible_p(__typeof__(&futhark_entry_embedding_spectral_normalize), embedding_spectral_type), "embedding_spectral_normalize ABI mismatch");