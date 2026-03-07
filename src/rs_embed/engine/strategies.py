"""Export strategy: polymorphic write dispatch (combined vs per-item).

Replaces ``if layout == "combined": ... else: ...`` conditionals with
a proper strategy pattern.  The exporter selects a strategy once and
then calls ``strategy.write_point()`` / ``strategy.finalize()`` without
caring about the layout.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

from ..writers import write_arrays
from .runner import run_with_retry


class ExportStrategy(ABC):
    """Abstract base for export-write strategies."""

    @abstractmethod
    def prepare(self) -> None:
        """One-time setup (create directories, etc.)."""

    @abstractmethod
    def write_point(
        self,
        *,
        out_path: str,
        arrays: Dict[str, np.ndarray],
        manifest: Dict[str, Any],
        save_manifest: bool,
        fmt: str,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> Dict[str, Any]:
        """Persist one point's arrays + manifest."""

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Flush pending writes and return a summary manifest."""


class CombinedExportStrategy(ExportStrategy):
    """Single-file output: all points in one NPZ/NetCDF."""

    def __init__(self, out_file: str) -> None:
        self.out_file = out_file

    def prepare(self) -> None:
        os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)

    def write_point(
        self,
        *,
        out_path: str,
        arrays: Dict[str, np.ndarray],
        manifest: Dict[str, Any],
        save_manifest: bool,
        fmt: str,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> Dict[str, Any]:
        return run_with_retry(
            lambda: write_arrays(
                fmt=fmt,
                out_path=out_path,
                arrays=arrays,
                manifest=manifest,
                save_manifest=save_manifest,
            ),
            retries=max_retries,
            backoff_s=retry_backoff_s,
        )

    def finalize(self) -> Dict[str, Any]:
        return {}


class PerItemExportStrategy(ExportStrategy):
    """Directory output: one file per spatial location."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir

    def prepare(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)

    def write_point(
        self,
        *,
        out_path: str,
        arrays: Dict[str, np.ndarray],
        manifest: Dict[str, Any],
        save_manifest: bool,
        fmt: str,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> Dict[str, Any]:
        return run_with_retry(
            lambda: write_arrays(
                fmt=fmt,
                out_path=out_path,
                arrays=arrays,
                manifest=manifest,
                save_manifest=save_manifest,
            ),
            retries=max_retries,
            backoff_s=retry_backoff_s,
        )

    def finalize(self) -> Dict[str, Any]:
        return {}
