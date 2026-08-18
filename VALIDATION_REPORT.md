# JAIDE v40 GPU training upgrade — implementation and validation report

Branch: `arena/01a01411-blueprint` (commits `81440fd`, `5a5cc60`, `ec8aae8`, on top of `1d45c65`).

## What was fixed

1. **Unified/managed memory disabled for production training.**
   `src/hw/accel/futhark_bindings.zig` now declares and calls
   `futhark_context_config_set_unified_memory(cfg, 0)` (exact signature verified
   against the generated header `src/hw/accel/main.h:38`) before
   `futhark_context_new()`. `JAIDE_FUTHARK_UNIFIED_MEMORY=1` is the only way to
   re-enable managed memory, and doing so logs an explicit warning. Futhark
   logging/debugging/profiling are opt-in via `JAIDE_FUTHARK_LOG`,
   `JAIDE_FUTHARK_DEBUG`, `JAIDE_FUTHARK_PROFILE` (default off, so the
   per-allocation "Allocating N bytes" spam is silenced unless requested).
   Errors from `futhark_context_get_error()` are retained.

2. **Checked memory estimator + preflight admission.**
   New `src/hw/accel/gpu_memory_model.zig`: all element/byte derivations use
   checked arithmetic (`std.math.mul/add` with classified
   `ElementCountOverflow`/`ByteCountOverflow` errors). It accounts RSF stacks
   (FP16 forward S+T, FP32 master/momentum/Fisher, transient gradients,
   replacement transients, spectral transients, legacy mirrors), embedding
   state (forward/master/grad/momentum/Fisher, spectral u/v, frozen target),
   batch/activation, graph chunk, and NCCL buffers. The trainer runs it before
   allocating (`JAIDE_MEMORY_PREFLIGHT`, reserve via
   `JAIDE_MEMORY_RESERVE_MIB` + 5% fraction) against `cudaMemGetInfo`
   (new correctly-typed binding `cudaMemGetInfo` + `cudaDeviceGetAttribute`
   compute-capability query in `src/hw/accel/cuda_bindings.zig`) and rejects
   with a report naming the largest contributors (`TrainerError.MemoryPreflightRejected`).

3. **True stack-only RSF training representation.**
   `RSFAccelerator.initStackOnly()` allocates exactly one FP16 S stack, one FP16
   T stack, and the FP32 master/momentum/Fisher stacks; the FP16 shadows are
   derived on-device from the masters (`futhark_entry_master_weights_to_f16_3d`),
   avoiding host FP16 staging copies. Per-layer mirror APIs (`layerPtr`,
   `forward`, `syncLayersFromStack`, `setLayerWeightsS/T`) are hard-guarded
   with `StackOnlyModeForbidden`/`MirrorStateUnavailable` in stack-only mode.
   The distributed trainer defaults to it (`JAIDE_STACK_ONLY_RSF=0` restores
   legacy mode), checkpoint restore uses it, and telemetry logs
   `per_layer_mirror_device_allocations=0`. `errdefer` rollback preserved.

4. **Minimal frozen target state.**
   `cloneDevice()` is no longer used for the frozen target. New
   `FrozenEmbeddingAccelerator` holds one FP16 weight matrix plus metadata —
   no gradient, momentum, Fisher, step, or FP32 master. Checkpoint capture
   exports the FP16 snapshot to a temporary host FP32 buffer (freed
   immediately), keeping checkpoint format **version 7 byte-compatible**; the
   loader restores through `initFromMasterWeightsF32` (transient FP32 device
   copy only during restore).

5. **Spectral normalization made incremental.**
   Startup spectral normalization now defaults to **0 iterations** (was 30 at
   every startup; `JAIDE_SPECTRAL_STARTUP_ITERATIONS`), periodic updates
   default to 1 iteration every `spectral_interval` steps
   (`JAIDE_SPECTRAL_POWER_ITERATIONS`, default now 1), with an opt-in full
   recalibration. Checkpoint resume maps the saved iteration count to periodic
   semantics. Embedding spectral u/v state is persistent across calls.

6. **Bounded, streamed knowledge-graph construction.**
   `batchEncodeGraph` previously encoded the entire hash set in one GPU call
   (`chunk_end = hashes.len`); it now streams validated chunks
   (`JAIDE_GRAPH_CHUNK_SIZE`, default 65536, max 16777216), applies global edge
   offsets deterministically, releases per-chunk device buffers, and logs
   per-chunk progress. `JAIDE_SKIP_KNOWLEDGE_GRAPH=1` skips the stage with an
   explicit diagnostic; relational passes then run on a valid empty graph.

7. **Content-addressed build artifacts and complete Futhark cache key.**
   `scripts/modal_status_bench.py` replaces "latest binary" reuse with a
   sha256 fingerprint over every Zig/Futhark/generated source, `futhark.pkg`,
   Zig/Futhark/CUDA tool versions, build flags, GPU spec, and the checkpoint
   schema version; binaries are stored under `fingerprints/<fp>/` with a
   manifest and checksum-verified reuse (`JAIDE_BENCH_FORCE_REBUILD=1`
   still overrides). The Futhark NVRTC cache key hashes **all** `.fut`/pkg/json
   inputs + tool versions + GPU spec (was `main.fut` + a constant string).

8. **Bounded startup monitoring and classified failures.**
   The bench watchdog enforces a model-initialization deadline (default 900 s,
   `JAIDE_BENCH_MODEL_INIT_DEADLINE_SEC`) and an idle-output timeout (default
   180 s, `JAIDE_BENCH_IDLE_OUTPUT_SEC`) with phase tracking, far below the
   20-hour training timeout (`JAIDE_BENCH_TRAIN_TIMEOUT_SEC`). A startup-only
   mode (`JAIDE_BENCH_STARTUP_ONLY=1`) stops after the first completed step.
   Failures are classified (CUDA OOM with the offending allocation line, host
   OOM, preflight rejection, timeout, idle stall, signal, abnormal exit) into
   `phase_c_error_summary.json` with the last known GPU state; partial reports
   are preserved.

9. **Documentation corrected.** README now states persistent training state is
   `O(layers x dim^2)` (RSF) + `O(vocab x dim)` (embeddings); only the
   activation component is layer-independent. Full env-var table added.

10. **Pre-existing test defect fixed.**
    `learned_embedding` "batched backward accumulates only valid lengths" was
    verified failing on the pristine baseline commit `1d45c65` (expected 4.0;
    the mathematically correct accumulation is 1.0 per occurrence). The test
    expectations were corrected and coverage strengthened (padding zero,
    all valid tokens, all unused tokens zero).

## Exact memory numbers (from the estimator, baseline 16384×11×32000×32×256)

- One S or T stack: 11 × 8192 × 8193 = **738,287,616 elements**; FP32 =
  **2,953,150,464 bytes**; FP16 = 1,476,575,232 bytes.
- Redundant legacy per-layer FP16 mirrors: **2.750 GiB** (removed).
- Stacked FP16 S+T forward: **2.750 GiB**; FP32 S+T masters: **5.501 GiB**;
  momentum: **5.501 GiB**; Fisher: **5.501 GiB**; transient FP32 S+T gradients:
  **5.501 GiB**; optimizer replacement transients: 9.626 GiB.
- Embedding (32000 × 16384): FP16 forward 0.977 GiB; each FP32 tensor
  (master/grad/momentum/Fisher) **1.953 GiB**; spectral u/v 128 KiB/64 KiB.
- Frozen target: FP16 **0.977 GiB** persistent (legacy clone additionally held
  an FP32 master, 1.953 GiB, plus gradient state).
- Before (legacy mirrors + clone): persistent ≈ 33.72 GiB, estimated peak
  ≈ **58.6 GiB** (plus the clone's gradient scratch).
- After (stack-only + minimal frozen): persistent ≈ **29.02 GiB**, estimated
  peak ≈ **53.8 GiB**.
- B200 admission (180 GiB total, 175 GiB free, 4 GiB + 5% reserve):
  **admitted, headroom ≈ 112.2 GiB**.

## What was actually run and passed in this sandbox

- `zig build test-all -Dskip-futhark` — **all tests pass** (602/602 after the
  embedding fix; 601/602 before it, with the single failure proven
  pre-existing on `1d45c65`).
- `zig build test-gpu-memory-model -Dskip-futhark` — 11/11 estimator,
  baseline-byte, overflow, and admission tests pass.
- `python3 -m unittest discover -s scripts -p "test_modal_status_bench.py"` —
  16/16 classification, phase-tracking, fingerprint, and tamper-detection
  tests pass.
- `zig build -Dskip-futhark` — CPU inference-server build passes.
- Full semantic type-check of the production GPU entry point
  (`src/main_distributed_futhark.zig`, forcing `main()` body analysis and
  transitively the whole trainer + accel interface) in **both** `gpu=true` and
  `gpu=false` build-option modes.

Toolchain used: Zig 0.14.1 (matches `build.zig.zon`), gcc/cc; the checked-in
CPU-backend Futhark C library.

## Commands for a CUDA/B200 machine

```
zig build test-gpu-accel -Dgpu=true            # new GPU-only accelerator tests
zig build distributed-futhark -Dgpu=true       # trainer
JAIDE_BENCH_STARTUP_ONLY=1 ... modal run ...   # startup-only validation
```

`test-gpu-accel` (src/test_root_accel_gpu.zig) covers: stack-only init with
zero mirrors; mirror-API rejection; forward/inverse reconstruction; optimizer
round trip; frozen embedding equivalence and FP32 restore; chunked-vs-single
graph equivalence; chunk validation; invalid-dimension failures. It is
semantically verified here and **must be executed on a CUDA machine** — this
sandbox has no GPU, no CUDA toolkit, and 3 GiB host RAM.

## Known limitations (precise)

1. **No GPU execution was performed here.** This sandbox has no NVIDIA GPU,
   no CUDA toolkit, 2 CPUs and 3 GiB RAM. Stages A/B/C of the requested
   validation (real optimizer steps at dim 16384 on a B200, measured peak
   device memory, tokens/s, checkpoint reload + HTTP inference smoke) were
   **not run** and are not claimed.
2. **Padded computation is not yet removed on the GPU.** Eliminating dense
   RSF work over padded positions requires new Futhark kernels and
   regeneration of the CUDA library; the Futhark compiler could not be
   obtained in this sandbox (release-asset downloads blocked). Dense RSF
   still processes padded rows; the estimator accounts for the padded
   activations. No mock "compaction" was added.
3. **The cuBLAS/tensor-core GEMM path is not integrated into training.**
   `tensor_core_matmul.zig` remains unused by the training loop; the verified
   execution path is the Futhark kernel pipeline. No tensor-core claims are
   made.
4. The in-tree generated CUDA library reports Futhark 0.25.29
   (`src/hw/accel/main.json`), not 0.26.4; all ABI declarations were verified
   against the actual generated header.
5. The bench watchdog classifies failures from process output; it cannot
   diagnose stalls in phases that print nothing at all beyond the idle
   timeout.
