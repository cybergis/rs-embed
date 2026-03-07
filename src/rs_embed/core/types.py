"""Domain types for rs-embed engine layer.

Provides enums, result objects, and configuration dataclasses that replace
the raw dicts and string constants scattered through the old functional code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .specs import InputPrepSpec, OutputSpec, SensorSpec


# ── Enums ──────────────────────────────────────────────────────────


class Status(enum.Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"


class ExportLayout(enum.Enum):
    COMBINED = "combined"
    PER_ITEM = "per_item"


class InferenceStrategy(enum.Enum):
    AUTO = "auto"
    BATCH = "batch"
    SINGLE = "single"


# ── Typed results ──────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskResult:
    """Result of a single embedding inference task."""

    status: Status
    embedding: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, embedding: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> TaskResult:
        return cls(status=Status.OK, embedding=embedding, meta=meta)

    @classmethod
    def failed(cls, error: Exception | str) -> TaskResult:
        return cls(status=Status.FAILED, error=repr(error) if isinstance(error, Exception) else str(error))


# ── Model configuration ───────────────────────────────────────────


@dataclass(frozen=True)
class ModelConfig:
    """Resolved per-model configuration (backend, sensor, type)."""

    name: str
    backend: str
    sensor: Optional[SensorSpec] = None
    model_type: str = ""

    @property
    def is_precomputed(self) -> bool:
        return "precomputed" in self.model_type.lower()


# ── Export target ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ExportTarget:
    """Resolved output target for a batch export."""

    layout: ExportLayout
    out_file: Optional[str] = None
    out_dir: Optional[str] = None
    names: Optional[List[str]] = None


# ── Export configuration ───────────────────────────────────────────


@dataclass(frozen=True)
class ExportConfig:
    """Groups the behavioral flags for a batch export.

    Replaces the ~20 keyword arguments formerly passed through every function.
    """

    format: str = "npz"
    save_inputs: bool = True
    save_embeddings: bool = True
    save_manifest: bool = True
    fail_on_bad_input: bool = False
    chunk_size: int = 16
    infer_batch_size: Optional[int] = None
    num_workers: int = 8
    continue_on_error: bool = False
    max_retries: int = 0
    retry_backoff_s: float = 0.0
    async_write: bool = True
    writer_workers: int = 2
    resume: bool = False
    show_progress: bool = True
    input_prep: Optional[InputPrepSpec | str] = "resize"

    @property
    def effective_infer_batch_size(self) -> int:
        return max(1, int(self.infer_batch_size or self.chunk_size))

    @property
    def effective_chunk_size(self) -> int:
        return max(1, int(self.chunk_size))
