import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


class _StubFunction:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("stubbed modal function must not be called in unit tests")


class _StubApp:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def function(self, *args: Any, **kwargs: Any) -> Any:
        def decorator(target: Any) -> Any:
            return target

        return decorator

    def local_entrypoint(self, *args: Any, **kwargs: Any) -> Any:
        def decorator(target: Any) -> Any:
            return target

        return decorator


class _StubImage:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        def method(*args: Any, **kwargs: Any) -> "_StubImage":
            return self

        return method

    @staticmethod
    def from_registry(*args: Any, **kwargs: Any) -> "_StubImage":
        return _StubImage()

    @staticmethod
    def debian_slim(*args: Any, **kwargs: Any) -> "_StubImage":
        return _StubImage()


class _StubVolume:
    @staticmethod
    def from_name(*args: Any, **kwargs: Any) -> "_StubVolume":
        return _StubVolume()


stub_modal = type(sys)("modal")
stub_modal.App = _StubApp
stub_modal.Image = _StubImage
stub_modal.Volume = _StubVolume
stub_modal.Function = _StubFunction
stub_modal.Secret = lambda *a, **k: None
sys.modules["modal"] = stub_modal

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal_status_bench as bench


class FailureClassificationTests(unittest.TestCase):
    def test_cuda_oom_is_classified_with_allocating_line(self) -> None:
        output = "[Rank 0] Allocating 2953150464 bytes for arr->mem in space 'device'\nCUDA error: out of memory"
        result = bench._classify_failure(output, 1, False, None)
        self.assertEqual(result["category"], "cuda_oom")
        self.assertIn("2953150464", result["detail"])

    def test_host_oom_is_classified(self) -> None:
        result = bench._classify_failure("std::bad_alloc\nalloc failed", 1, False, None)
        self.assertEqual(result["category"], "host_oom")

    def test_memory_preflight_rejection_is_classified(self) -> None:
        result = bench._classify_failure("memory preflight rejected: estimated peak exceeds free memory", 1, False, None)
        self.assertEqual(result["category"], "memory_preflight_rejected")

    def test_signal_exit_is_classified(self) -> None:
        result = bench._classify_failure("whatever", -9, False, None)
        self.assertEqual(result["category"], "signal")
        self.assertIn("9", result["detail"])

    def test_abnormal_exit_is_classified(self) -> None:
        result = bench._classify_failure("no marker", 3, False, None)
        self.assertEqual(result["category"], "abnormal_exit")

    def test_timeout_with_stall_reason(self) -> None:
        stall = bench.TrainingStallState()
        stall.reason = "model initialization deadline (900s) exceeded in phase 'stack_rsf_allocation'"
        stall.phase = "stack_rsf_allocation"
        result = bench._classify_failure("partial output", 1, True, stall)
        self.assertEqual(result["category"], "timeout")
        self.assertEqual(result["phase"], "stack_rsf_allocation")
        self.assertFalse(result["model_initialization_completed"])

    def test_idle_stall_category(self) -> None:
        stall = bench.TrainingStallState()
        stall.reason = "no output for 180s during initialization phase 'futhark_context'; last line: Allocating 2953150464 bytes"
        result = bench._classify_failure("partial", 1, False, stall)
        self.assertEqual(result["category"], "idle_stall")
        self.assertIn("2953150464", result["detail"])

    def test_clean_run_has_no_category(self) -> None:
        result = bench._classify_failure("[Rank 0] Step 1 completed loss=0.5", 0, False, None)
        self.assertEqual(result["category"], "none")


class PhaseTrackingTests(unittest.TestCase):
    def test_phase_advances_through_startup_markers(self) -> None:
        state = bench.TrainingStallState()
        bench._advance_phase(state, "[FutharkContext] device=0 unified_memory=disabled")
        self.assertEqual(state.phase, "futhark_context")
        bench._advance_phase(state, "[Rank 0] memory preflight phase_ms=3")
        self.assertEqual(state.phase, "memory_preflight")
        bench._advance_phase(state, "[RSFAccelerator] stack-only mode initialized dim=16384 layers=11")
        self.assertEqual(state.phase, "stack_rsf_allocation")
        bench._advance_phase(state, "[Rank 0] Knowledge graph construction: encoding 500000 samples")
        self.assertEqual(state.phase, "graph_construction")
        bench._advance_phase(state, "[Rank 0] Starting Futhark-accelerated training")
        self.assertEqual(state.phase, "training_start")
        self.assertFalse(state.init_completed)
        bench._advance_phase(state, "[Rank 0] Step 1 completed loss=0.42")
        self.assertTrue(state.init_completed)
        self.assertTrue(state.first_step_completed)


class BuildFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name)
        accel = self.project / "src" / "hw" / "accel"
        accel.mkdir(parents=True)
        (accel / "main.fut").write_text("-- main fut v1")
        (accel / "futhark_kernels.fut").write_text("-- kernels fut v1")
        (accel / "futhark.pkg").write_text("package")
        (self.project / "build.zig").write_text("// build")
        (self.project / "src" / "main_distributed_futhark.zig").write_text("// main zig")
        (self.project / "src" / "distributed").mkdir(parents=True, exist_ok=True)
        (self.project / "src" / "distributed" / "checkpoint_envelope.zig").write_text("pub const VERSION: u32 = 7;")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fingerprint_is_deterministic(self) -> None:
        tools = {"zig": "0.14.1", "futhark": "0.26.4", "cuda": "12.8"}
        first = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        second = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        self.assertEqual(first, second)

    def test_fingerprint_changes_when_source_changes(self) -> None:
        tools = {"zig": "0.14.1", "futhark": "0.26.4", "cuda": "12.8"}
        before = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        (self.project / "src" / "main_distributed_futhark.zig").write_text("// main zig changed")
        after = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        self.assertNotEqual(before, after)

    def test_fingerprint_changes_when_futhark_source_changes(self) -> None:
        tools = {"zig": "0.14.1", "futhark": "0.26.4", "cuda": "12.8"}
        before = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        (self.project / "src" / "hw" / "accel" / "main.fut").write_text("-- main fut v2")
        after = bench._compute_build_fingerprint(str(self.project), tools, "flags", True)
        self.assertNotEqual(before, after)

    def test_fingerprint_changes_when_toolchain_changes(self) -> None:
        tools_a = {"zig": "0.14.1", "futhark": "0.26.4", "cuda": "12.8"}
        tools_b = {"zig": "0.14.1", "futhark": "0.26.5", "cuda": "12.8"}
        first = bench._compute_build_fingerprint(str(self.project), tools_a, "flags", True)
        second = bench._compute_build_fingerprint(str(self.project), tools_b, "flags", True)
        self.assertNotEqual(first, second)

    def test_checkpoint_schema_version_is_parsed(self) -> None:
        self.assertEqual(bench._checkpoint_schema_version(str(self.project)), "pub const VERSION: u32 = 7")

    def test_futhark_cache_key_covers_all_fut_files(self) -> None:
        tools = {"futhark": "0.26.4"}
        before = bench._futhark_cache_key(str(self.project), tools)
        (self.project / "src" / "hw" / "accel" / "futhark_kernels.fut").write_text("-- kernels fut v2")
        after = bench._futhark_cache_key(str(self.project), tools)
        self.assertNotEqual(before, after)

    def test_fingerprinted_binary_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as store_dir:
            with mock.patch.object(bench, "BUILD_MOUNT_PATH", Path(store_dir)):
                binary_dir = Path(store_dir) / "bin"
                binary_dir.mkdir(parents=True)
                binary = binary_dir / "jaide-distributed-futhark"
                binary.write_bytes(b"fake-elf-payload")
                inputs = bench._build_fingerprint_inputs(str(self.project))
                fingerprint = bench._compute_build_fingerprint(str(self.project), {"zig": "0.14.1"}, "flags", True)
                manifest = bench._store_fingerprinted_binaries(fingerprint, inputs, {"zig": "0.14.1"}, "flags", [binary])
                self.assertIn("jaide-distributed-futhark", manifest["binaries"])
                found = bench._find_fingerprinted_binary("jaide-distributed-futhark", fingerprint)
                self.assertIsNotNone(found)
                tampered = bench._fingerprint_dir(fingerprint) / "jaide-distributed-futhark"
                tampered.write_bytes(b"stale-different-payload")
                self.assertIsNone(bench._find_fingerprinted_binary("jaide-distributed-futhark", fingerprint))
                self.assertIsNone(bench._find_fingerprinted_binary("jaide-distributed-futhark", "0" * 64))


if __name__ == "__main__":
    unittest.main()
