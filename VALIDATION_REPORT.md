# JAIDE v40 GPU training upgrade — implementation and validation report

Branch: `arena/01a01411-blueprint`, on top of `1d45c65` (main).

## Implemented fixes

### 1. Unified/managed memory disabled for production training
`src/hw/accel/futhark_bindings.zig` declares and calls
`futhark_context_config_set_unified_memory(cfg, 0)` before
`futhark_context_new()` (signature verified against the generated
`src/hw/accel/main.h:38`). Re-enable only via `JAIDE_FUTHARK_UNIFIED_MEMORY=1`
(with an explicit warning). `JAIDE_FUTHARK_LOG`, `JAIDE_FUTHARK_DEBUG`,
`JAIDE_FUTHARK_PROFILE` opt into runtime logging/debugging/profiling
(default off; verbose per-allocation logging silenced). Errors from
`futhark_context_get_error()` are preserved.

### 2. Checked memory estimator and preflight admission
`src/hw/accel/gpu_memory_model.zig`: every element/byte derivation uses
checked arithmetic with classified `ElementCountOverflow`/`ByteCountOverflow`
errors. Accounts RSF stacks (FP16 forward S+T, FP32 master/momentum/Fisher,
transient gradients, optimizer replacement transients, spectral transients,
legacy mirrors), embedding state (forward/master/grad/momentum/Fisher,
spectral u/v, frozen target), batch activations (padded or compact active
rows), graph chunks, NCCL buffers, and runtime slack. The trainer runs the
preflight before allocating against `cudaMemGetInfo` (new bindings in
`src/hw/accel/cuda_bindings.zig` together with
`cudaDeviceGetAttribute`-based compute-capability query) and rejects with a
contributor-ranked report (`TrainerError.MemoryPreflightRejected`).
Configurable reserve: `JAIDE_MEMORY_RESERVE_MIB` + 5% fraction;
`JAIDE_MEMORY_PREFLIGHT=0` disables.

### 3. True stack-only RSF training representation
`RSFAccelerator.initStackOnly()` allocates exactly one FP16 S stack, one FP16
T stack, and FP32 master/momentum/Fisher stacks; FP16 shadows are derived
on-device via `futhark_entry_master_weights_to_f16_3d`. Per-layer mirror APIs
(`layerPtr`, `forward`, `syncLayersFromStack`, `setLayerWeightsS/T`) are
guarded with `StackOnlyModeForbidden`/`MirrorStateUnavailable`. Distributed
training and checkpoint restore default to it (`JAIDE_STACK_ONLY_RSF=0`
restores legacy mode); telemetry logs
`per_layer_mirror_device_allocations=0`; `errdefer` rollback preserved.

### 4. Minimal frozen target state
`FrozenEmbeddingAccelerator` holds a single FP16 weight matrix plus metadata
(no gradient/momentum/Fisher/step/FP32 master). Checkpoint capture exports the
FP16 snapshot through a temporary host FP32 buffer; format stays version 7,
byte-compatible with existing readers. Restore path:
`initFromMasterWeightsF32` (transient FP32 device copy only during restore).

### 5. Compact active-row execution (padding removed from dense RSF compute)
Verified from `src/hw/accel/main.fut` that `rsf_stack_forward` /
`rsf_stack_inverse` are row-independent (the OFTB butterfly is intra-row) and
that `rsf_stack_backward_gradients_fused` derives its active set from
`lengths` with normalization divisors that depend only on the active-token
count. Therefore running the existing kernels on compact
`[active_rows][1][dim]` tensors with all-ones lengths is exactly equivalent to
padded execution for the valid rows.

- `src/hw/accel/active_rows.zig`: deterministic active-batch builder (compact
  input/target tokens, original batch/sequence indices, lengths, counts),
  scatter-back helper, and a pure-Zig reference of the exact coupling,
  inverse, fused backward, and embedding-backward math.
- Trainer default `compact_rows=true` (`JAIDE_COMPACT_ROWS=0` disables):
  compact embedding forward for inputs and frozen targets, compact fused
  training step, compact embedding backward; per-step telemetry logs
  `active_rows`, `padded_rows`, `active_ratio`, `compact_rsf_compute=1`.
- Loss / reconstruction / log-determinant normalization, `grad_mean`
  divisors, `local_fraction` scaling, clipping, OFTB transform/inverse, RSF
  reconstruction, histogram accumulation, and padding-zero semantics are all
  preserved exactly.

### 6. Incremental, stateful spectral normalization
Startup spectral normalization now defaults to 0 iterations
(`JAIDE_SPECTRAL_STARTUP_ITERATIONS`); periodic updates default to 1
iteration per interval (`JAIDE_SPECTRAL_POWER_ITERATIONS`); optional full
recalibration. Fixed defect: `applyEmbeddingSpectralNormalization` used to
`resetSpectralState()` on every call, discarding the persistent embedding
power-iteration vectors — the reset now happens only on shape/ownership
change (`ensureSpectralState` reuse semantics). New
`src/hw/accel/spectral_state.zig` maintains per-layer persistent stack
spectral state (u/v, ownership epoch, cumulative iteration count, sigma
estimates); sigma estimates and `persistent_iterations_total` are reported
per spectral update. Checkpoint resume maps the saved iteration count onto
the periodic semantics.

### 7. Bounded, streamed knowledge-graph construction
`batchEncodeGraph` previously encoded the entire hash set in one GPU call;
it now streams validated chunks (`JAIDE_GRAPH_CHUNK_SIZE`, default 65536,
max 16777216) with deterministic global edge offsets, per-chunk device
buffer release, and progress logs. `JAIDE_SKIP_KNOWLEDGE_GRAPH=1` skips the
stage with an explicit diagnostic (relational passes then run on a valid
empty graph).

### 8. RSF backend selector with explicit classification
`src/hw/accel/rsf_backend.zig`: `JAIDE_RSF_BACKEND=futhark|reference|cublas`.
`futhark` is the production path; `reference` executes the identical math on
active rows host-side for correctness testing (finite-difference-verified);
`cublas` is classified `BackendRequiresRegeneratedKernels` and selection
fails fast with `RsfBackendUnavailable` — no silent fallback. The selected
backend, its availability, and the reason are logged at startup.

### 9. Content-addressed build artifacts and complete Futhark cache key
`scripts/modal_status_bench.py` computes a sha256 fingerprint over every
Zig/Futhark/generated source, `futhark.pkg`, `build.zig(.zon)`, Zig/Futhark/
CUDA tool versions, build flags, GPU spec, and the checkpoint schema version;
binaries are stored under `fingerprints/<fp>/` with a manifest and
checksum-verified reuse. `JAIDE_BENCH_FORCE_REBUILD=1` still overrides. The
Futhark NVRTC cache key hashes all `.fut`/pkg/json inputs + tool versions +
GPU spec (was `main.fut` + a constant string).

### 10. Bounded startup monitoring, heartbeats, classified failures
Bench watchdogs: model-initialization deadline (default 900 s,
`JAIDE_BENCH_MODEL_INIT_DEADLINE_SEC`), idle-output timeout (default 180 s,
`JAIDE_BENCH_IDLE_OUTPUT_SEC`) with startup-phase tracking, far below the
training timeout (`JAIDE_BENCH_TRAIN_TIMEOUT_SEC`); startup-only validation
mode (`JAIDE_BENCH_STARTUP_ONLY=1`) stops after the first completed step.
Failures are classified (CUDA OOM with the offending allocation line, host
OOM, preflight rejection, timeout, idle stall, signal, abnormal exit) into
`phase_c_error_summary.json` with the last known GPU state; partial reports
are preserved. Phase C reports include `active_rows_samples`,
`active_rows_last`, and `padding_removed`. New
`src/distributed/phase_heartbeat.zig` provides an in-process phase heartbeat
(`JAIDE_HEARTBEAT_SEC`, default 30) wired through dataset load, tokenizer,
model initialization, checkpoint restore, knowledge graph, and training.

### 11. Documentation corrected
README states persistent training state is `O(layers x dim^2)` (RSF) +
`O(vocab x dim)` (embeddings); only the activation component is
layer-independent. Full environment-variable table included.

### 12. Pre-existing test defect fixed
`learned_embedding` "batched backward accumulates only valid lengths" was
verified failing on pristine baseline `1d45c65` (expected 4.0 where the
mathematically correct accumulation is 1.0 per occurrence). Expectations
corrected and coverage strengthened.

## Exact memory numbers (baseline 16384 x 11 x 32000 x 32 x 256)

- One S or T stack: `11 x 8192 x 8193 = 738,287,616` elements; FP32 =
  `2,953,150,464` bytes; FP16 = `1,476,575,232` bytes.
- Legacy per-layer FP16 mirrors: **2.750 GiB** (removed in stack-only mode).
- Stacked FP16 S+T forward: **2.750 GiB**; FP32 S+T masters: **5.501 GiB**;
  momentum: **5.501 GiB**; Fisher: **5.501 GiB**; transient FP32 S+T
  gradients: **5.501 GiB**; optimizer replacement transients: 9.626 GiB.
- Embedding (32000 x 16384): FP16 forward 0.977 GiB; each FP32 tensor
  (master/grad/momentum/Fisher) **1.953 GiB**; spectral u/v 128 KiB / 64 KiB.
- Frozen target: FP16 **0.977 GiB** persistent (legacy clone additionally
  held a 1.953 GiB FP32 master plus gradient state).
- Before (legacy mirrors + clone): persistent ~33.72 GiB, estimated peak
  **~58.6 GiB** (plus the clone's gradient scratch).
- After (stack-only + minimal frozen): persistent **~29.02 GiB**, estimated
  peak **~53.8 GiB**.
- B200 admission (180 GiB total, 175 GiB free, 4 GiB + 5% reserve):
  **admitted, headroom ~112.2 GiB**.

## Verified in this sandbox (no GPU available)

- `zig build test-all -Dskip-futhark` — all tests pass, including the new
  test roots (`test-gpu-memory-model`, `test-active-rows`,
  `test-phase-heartbeat`, `test-spectral-state`, `test-rsf-backend`).
- `src/hw/accel/gpu_memory_model.zig` — 12 tests: checked arithmetic,
  overflow classification, exact baseline byte counts, admission
  accept/reject, compact-row accounting, persistent-vs-transient separation.
- `src/hw/accel/active_rows.zig` — 8 tests: irregular/empty/full-length/
  repeated-token/one-token construction, scatter-back zero padding,
  forward/inverse mutual inverses (OFTB), compact-vs-padded numerical
  identity of loss/reconstruction/log-determinant/gradients/input-delta,
  padding-contributes-zero-gradient, and finite-difference S/T gradient
  checks (the FD check caught and drove the fix of a real reference bug:
  loss computed from the post-peel tensor instead of the final outputs).
- `src/hw/accel/spectral_state.zig` — 4 tests: buffer reuse across
  compatible calls, reset only on shape/ownership change, per-layer views.
- `src/hw/accel/rsf_backend.zig` — 6 tests: name parsing, default selection,
  cublas classified unavailable, reference availability tied to compact rows,
  telemetry line.
- `src/distributed/phase_heartbeat.zig` — 3 tests including a live
  heartbeat thread.
- `python3 -m unittest discover -s scripts -p test_modal_status_bench.py` —
  16 tests: failure classification (CUDA OOM / host OOM / preflight /
  timeout / idle stall / signal / abnormal exit), phase tracking,
  fingerprint determinism and source/toolchain sensitivity, Futhark cache
  key coverage, checksum-verified reuse and tamper detection.
- Full semantic type-check of the production GPU entry point
  (`src/main_distributed_futhark.zig`, forcing `main()` body analysis and
  transitively the entire trainer and accel interface) in both
  `gpu_acceleration=true` and `gpu_acceleration=false` build modes.
- `src/test_root_accel_gpu.zig` (GPU-only step `test-gpu-accel`): stack-only
  init with zero mirrors, mirror-API rejection, forward/inverse
  reconstruction, optimizer round trip, frozen-embedding equivalence and
  FP32 restore, chunked-vs-single graph equivalence, chunk validation,
  invalid-dimension failures, compact-vs-padded kernel equivalence for
  embedding forward and the fused training step — semantically verified
  here, requires a CUDA machine to execute.

## Known limitations (precise)

1. **No GPU execution here.** This sandbox has no NVIDIA GPU, no CUDA
   toolkit, 2 CPUs, 3 GiB RAM. The B200 16,384-dim run, measured peak device
   memory, tokens/s, checkpoint-reload + HTTP inference smoke, and the
   `test-gpu-accel` step remain to be executed on a CUDA machine.
2. **cuBLAS/tensor-core GEMM backend not implemented.** The RSF coupling
   chain needs elementwise exp/clamp stages between GEMMs; exporting those
   from the fused kernels requires a Futhark kernel regeneration, and the
   Futhark compiler cannot be obtained in this sandbox. The backend selector
   classifies `cublas` as requiring regenerated kernels and fails fast
   instead of silently falling back; no tensor-core claims are made.
3. **Stack spectral power-iteration vectors are managed host-side.** The
   `stack_spectral_normalize` kernel restarts from a fixed initial vector;
   feeding persistent u/v into it requires a kernel ABI change (same
   regeneration constraint). Embedding spectral vectors are persistent in
   device memory end-to-end.
4. The in-tree generated CUDA library reports Futhark 0.25.29
   (`src/hw/accel/main.json`), not 0.26.4; all ABI declarations were
   verified against the actual generated header.
5. The bench watchdog classifies stalls from process output; a phase that
   prints nothing at all can only be caught by the idle timeout or the
   in-process heartbeat.

## Commands

```
zig build test-all -Dskip-futhark
zig build test-gpu-memory-model -Dskip-futhark
zig build test-active-rows -Dskip-futhark
zig build test-spectral-state -Dskip-futhark
zig build test-rsf-backend -Dskip-futhark
zig build test-phase-heartbeat -Dskip-futhark
python3 -m unittest discover -s scripts -p "test_modal_status_bench.py"
zig build test-gpu-accel -Dgpu=true        (CUDA machine)
zig build distributed-futhark -Dgpu=true   (CUDA machine)
```
