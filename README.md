# JAIDE v40

JAIDE v40 is a dependency-free, fifth-root machine learning architecture that
replaces standard neural network primitives and attention with an algebraically
invertible cross-affine coupling flow (RSF) and parameter-free fractal mixing
(OFTB), with multi-signal reconstruction learning and a dual quantum-symbolic
relational engine.

## Memory model (accurate accounting)

The reversible RSF/OFTB execution makes **activation** memory independent of
layer count, because intermediate activations are reconstructed during the
reversible backward pass. It does **not** make the full training state
`O(dim)`:

- Persistent RSF parameter and optimizer state is `O(num_layers x dim^2)`:
  one S or T stack contains `num_layers x (dim/2) x (dim/2 + 1)` elements,
  held as one FP16 forward stack each for S and T plus FP32 master, momentum,
  and Fisher stacks for the SFD optimizer.
- Persistent embedding state is `O(vocab_size x dim)` for the trainable
  source embedding plus `O(vocab_size x dim)` FP16-only state for the frozen
  target embedding when `target_source_frozen=true`.
- Transient peak additionally includes current-step FP32 S/T gradients,
  replacement stacks during optimizer updates, batch activations, and graph
  chunk buffers.

For the reference `model_dim=16384, num_layers=11, vocab_size=32000,
batch_size=32, max_seq_len=256` configuration, one S or T stack contains
`11 x 8192 x 8193 = 738,287,616` elements (2,953,150,464 bytes in FP32).
Run `zig build test-gpu-memory-model -Dskip-futhark` for the exact
persistent and peak estimates (`src/hw/accel/gpu_memory_model.zig`).

## GPU training configuration

Production training runs with explicit device memory: Futhark unified memory
is disabled before `futhark_context_new()` via
`futhark_context_config_set_unified_memory(cfg, 0)`. Allocation failures are
not silently absorbed by managed-memory oversubscription; the trainer runs a
memory preflight against `cudaMemGetInfo` before allocating large state and
rejects configurations whose estimated peak exceeds free memory minus a
configurable reserve, reporting the largest contributors.

### Environment variables

- `JAIDE_FUTHARK_UNIFIED_MEMORY=1` - opt in to managed memory (not
  recommended for production training; default is disabled).
- `JAIDE_FUTHARK_LOG=1`, `JAIDE_FUTHARK_DEBUG=1`, `JAIDE_FUTHARK_PROFILE=1` -
  opt-in Futhark runtime logging, debugging, and profiling.
- `JAIDE_FUTHARK_CACHE` - Futhark NVRTC kernel cache path.
- `JAIDE_MEMORY_PREFLIGHT=0` - disable the startup admission check.
- `JAIDE_MEMORY_RESERVE_MIB` - absolute device memory reserve (default 4096).
- `JAIDE_STACK_ONLY_RSF=0` - fall back to legacy per-layer mirrors (adds
  roughly one extra FP16 S/T copy in device memory).
- `JAIDE_SKIP_KNOWLEDGE_GRAPH=1` - skip knowledge graph construction;
  relational passes then operate on a valid empty graph.
- `JAIDE_GRAPH_CHUNK_SIZE` - hashes per GPU graph-encoding chunk (default
  65536, maximum 16777216).
- `JAIDE_SPECTRAL_STARTUP_ITERATIONS` - startup spectral normalization
  iterations (default 0, previously 30 at every startup).
- `JAIDE_SPECTRAL_POWER_ITERATIONS` - periodic spectral normalization
  iterations (default 1).
- `JAIDE_COMPACT_ROWS=0` - disable compact active-row execution; dense RSF
  compute then runs over padded positions again (default enabled). Per step
  the trainer logs `active_rows`, `padded_rows`, and `active_ratio`; loss,
  reconstruction, log-determinant normalization and `grad_mean` divisors are
  identical between compact and padded execution by construction.
- `JAIDE_HEARTBEAT_SEC` - startup/phase heartbeat interval in seconds for the
  training binary (default 30; 0 disables).
- `JAIDE_RSF_BACKEND` - RSF execution backend selector: `futhark` (default,
  production), `reference` (host reference math on active rows, correctness
  testing only), `cublas` (classified as unavailable until kernels are
  regenerated to export per-layer elementwise stages and persistent spectral
  vectors; selection fails fast with `RsfBackendUnavailable` instead of a
  silent fallback).

Persistent spectral state: the trainable-embedding power-iteration vectors
survive across periodic normalizations and are reset only when the matrix
shape or weight ownership changes; the stack keeps a per-layer persistent
spectral state bookkeeper whose iteration counter and sigma estimates are
reported per spectral update (`persistent_iterations_total`).
- `JAIDE_BENCH_MODEL_INIT_DEADLINE_SEC` - bench watchdog: model
  initialization deadline (default 900).
- `JAIDE_BENCH_IDLE_OUTPUT_SEC` - bench watchdog: no-output timeout during
  initialization (default 180).
- `JAIDE_BENCH_STARTUP_ONLY=1` - bench startup-only validation mode: stop
  after the first completed training step.
- `JAIDE_BENCH_TRAIN_TIMEOUT_SEC` - overall training timeout (default 72000).
- `JAIDE_BENCH_FORCE_REBUILD=1` - force rebuild; content-addressed build
  artifacts are otherwise reused only when the full source/toolchain
  fingerprint and executable checksums match.

## Checkpoints

Checkpoint format is version 7 (see `src/distributed/checkpoint_envelope.zig`).
The frozen target embedding is stored as FP32 master values to keep version 7
readers byte-compatible; it is exported from the FP16 device snapshot through
a temporary host buffer at capture time and not kept as persistent FP32 device
state.

## Build and test

- CPU tests: `zig build test-all -Dskip-futhark`
- GPU trainer: `zig build distributed-futhark -Dgpu=true` (requires CUDA
  toolkit, cuBLAS, NCCL, and a Futhark CUDA build).
- Bench watchdog unit tests: `python3 -m unittest discover -s scripts
  -p "test_modal_status_bench.py"`
