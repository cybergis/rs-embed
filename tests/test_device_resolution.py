"""Tests for resolve_device_auto_torch in tools.runtime.

Covers the priority order: cuda > mps > cpu, pass-through of explicit
device strings, and the torch-not-installed fallback.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from rs_embed.tools.runtime import resolve_device_auto_torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_torch_mock(*, cuda: bool, mps: bool) -> MagicMock:
    """Return a minimal sys.modules["torch"] mock with controlled backend flags."""
    m = MagicMock()
    m.cuda.is_available.return_value = cuda
    m.backends.mps.is_available.return_value = mps
    return m


@contextmanager
def _patch_torch(mock):
    """Inject *mock* as sys.modules['torch'] for the duration of the block."""
    original = sys.modules.get("torch")
    sys.modules["torch"] = mock
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original


# ---------------------------------------------------------------------------
# Explicit device strings — must pass through unchanged
# ---------------------------------------------------------------------------


class TestExplicitDevice:
    def test_explicit_cpu(self):
        assert resolve_device_auto_torch("cpu") == "cpu"

    def test_explicit_cuda(self):
        assert resolve_device_auto_torch("cuda") == "cuda"

    def test_explicit_cuda_index(self):
        assert resolve_device_auto_torch("cuda:0") == "cuda:0"

    def test_explicit_mps(self):
        assert resolve_device_auto_torch("mps") == "mps"

    def test_explicit_never_calls_torch(self):
        """Explicit device must return immediately without touching torch."""
        with _patch_torch(None):  # would raise if imported
            assert resolve_device_auto_torch("cpu") == "cpu"


# ---------------------------------------------------------------------------
# Auto resolution — priority: cuda > mps > cpu
# ---------------------------------------------------------------------------


class TestAutoResolution:
    def test_cuda_wins_when_available(self):
        """cuda is returned when torch.cuda.is_available() is True."""
        with _patch_torch(_make_torch_mock(cuda=True, mps=False)):
            assert resolve_device_auto_torch("auto") == "cuda"

    def test_cuda_wins_over_mps(self):
        """cuda takes priority even when mps is also available."""
        with _patch_torch(_make_torch_mock(cuda=True, mps=True)):
            assert resolve_device_auto_torch("auto") == "cuda"

    def test_mps_selected_when_no_cuda(self):
        """mps is returned on Apple Silicon when cuda is absent."""
        with _patch_torch(_make_torch_mock(cuda=False, mps=True)):
            assert resolve_device_auto_torch("auto") == "mps"

    def test_cpu_fallback_when_neither(self):
        """cpu is returned when neither cuda nor mps is available."""
        with _patch_torch(_make_torch_mock(cuda=False, mps=False)):
            assert resolve_device_auto_torch("auto") == "cpu"

    def test_cpu_fallback_when_torch_missing(self):
        """cpu is returned safely (with a warning) when torch is not installed."""
        with _patch_torch(None):  # simulates missing package
            with pytest.warns(UserWarning, match="falling back to 'cpu'"):
                assert resolve_device_auto_torch("auto") == "cpu"

    def test_cpu_fallback_on_unexpected_exception(self):
        """cpu is returned with a diagnostic warning when torch.cuda.is_available()
        raises — a broken CUDA install must not look identical to a CPU machine."""
        broken = _make_torch_mock(cuda=False, mps=False)
        broken.cuda.is_available.side_effect = RuntimeError("driver error")
        with _patch_torch(broken):
            with pytest.warns(UserWarning, match="driver error"):
                assert resolve_device_auto_torch("auto") == "cpu"


# ---------------------------------------------------------------------------
# Live check — on this machine MPS should be detected
# ---------------------------------------------------------------------------


class TestLiveDevice:
    def test_auto_does_not_return_cpu_on_accelerated_hardware(self):
        """On a machine with CUDA or MPS, 'auto' should not fall back to cpu."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        has_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
        if not has_accelerator:
            pytest.skip("No GPU/MPS available on this machine")

        result = resolve_device_auto_torch("auto")
        assert result != "cpu", f"Expected cuda or mps on accelerated hardware, got {result!r}"

    def test_auto_returns_mps_on_apple_silicon(self):
        """On Apple Silicon with MPS available, 'auto' must return 'mps' (not 'cpu')."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available on this machine")
        if torch.cuda.is_available():
            pytest.skip("CUDA also available — cuda takes priority over mps")

        assert resolve_device_auto_torch("auto") == "mps"
