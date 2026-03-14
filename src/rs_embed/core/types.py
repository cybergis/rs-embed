"""Domain types for rs-embed engine layer.

Provides enums, result objects, and configuration dataclasses that replace
the raw dicts and string constants scattered through the old functional code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from .specs import InputPrepSpec, OutputSpec, SensorSpec

# ── Enums ──────────────────────────────────────────────────────────

class Status(enum.Enum):
    """Execution status for a single task."""

    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"

class ExportLayout(enum.Enum):
    """Batch export layout policy."""

    COMBINED = "combined"
    PER_ITEM = "per_item"

class InferenceStrategy(enum.Enum):
    """Inference dispatch policy for single vs. batch execution."""

    AUTO = "auto"
    BATCH = "batch"
    SINGLE = "single"

# ── Typed results ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskResult:
    """Result of a single embedding inference task.

    Attributes
    ----------
    status : Status
        Outcome status for the task.
    embedding : np.ndarray or None
        Embedding payload when status is ``Status.OK``.
    meta : dict[str, Any] or None
        Associated metadata for the produced embedding.
    error : str or None
        Error message when task failed.
    """

    status: Status
    embedding: np.ndarray | None = None
    meta: dict[str, Any] | None = None
    error: str | None = None

    @classmethod
    def ok(cls, embedding: np.ndarray, meta: dict[str, Any] | None = None) -> TaskResult:
        """Create a successful task result.

        Parameters
        ----------
        embedding : np.ndarray
            Produced embedding payload.
        meta : dict[str, Any] or None
            Optional metadata attached to the embedding.

        Returns
        -------
        TaskResult
            Result object with ``status=Status.OK``.
        """
        return cls(status=Status.OK, embedding=embedding, meta=meta)

    @classmethod
    def failed(cls, error: Exception | str) -> TaskResult:
        """Create a failed task result.

        Parameters
        ----------
        error : Exception or str
            Error object or message to store.

        Returns
        -------
        TaskResult
            Result object with ``status=Status.FAILED``.
        """
        return cls(
            status=Status.FAILED,
            error=repr(error) if isinstance(error, Exception) else str(error),
        )

# ── Model configuration ───────────────────────────────────────────

@dataclass(frozen=True)
class ModelConfig:
    """Resolved per-model configuration used by export/inference pipelines.

    Attributes
    ----------
    name : str
        User-facing model identifier.
    backend : str
        Effective backend selected for this model.
    sensor : SensorSpec or None
        Sensor override for provider-backed models.
    model_type : str
        Lower-level model family/type hint.
    """

    name: str
    backend: str
    sensor: SensorSpec | None = None
    model_type: str = ""

    @property
    def is_precomputed(self) -> bool:
        """Whether this model is a precomputed provider model.

        Returns
        -------
        bool
            ``True`` when ``model_type`` indicates a precomputed model.
        """
        return "precomputed" in self.model_type.lower()

# ── Public export request objects ──────────────────────────────────

@dataclass(frozen=True)
class ExportModelRequest:
    """Public per-model export request.

    This is the user-facing counterpart to ``ModelConfig``.

    Attributes
    ----------
    name : str
        Model identifier or alias.
    sensor : SensorSpec or None
        Optional per-model sensor override for provider-backed models.
    modality : str or None
        Optional per-model modality selector.
    """

    name: str
    sensor: SensorSpec | None = None
    modality: str | None = None

# ── Export target ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ExportTarget:
    """Resolved output target for a batch export.

    Attributes
    ----------
    layout : ExportLayout
        Combined file or per-item output layout.
    out_file : str or None
        Output file path for combined exports.
    out_dir : str or None
        Output directory for per-item exports.
    names : list[str] or None
        Optional per-item names for output mapping.
    """

    layout: ExportLayout
    out_file: str | None = None
    out_dir: str | None = None
    names: list[str] | None = None

    @classmethod
    def combined(cls, out_file: str) -> "ExportTarget":
        """Build a combined-file export target."""
        return cls(layout=ExportLayout.COMBINED, out_file=out_file)

    @classmethod
    def per_item(cls, out_dir: str, *, names: list[str] | None = None) -> "ExportTarget":
        """Build a per-item export target."""
        return cls(layout=ExportLayout.PER_ITEM, out_dir=out_dir, names=names)

# ── Export configuration ───────────────────────────────────────────

@dataclass(frozen=True)
class ExportConfig:
    """Groups the behavioral flags for a batch export.

    Replaces the ~20 keyword arguments formerly passed through every function.

    Attributes
    ----------
    format : str
        Output serialization format.
    save_inputs : bool
        Whether to persist input arrays.
    save_embeddings : bool
        Whether to persist embeddings.
    save_manifest : bool
        Whether to write manifest metadata.
    fail_on_bad_input : bool
        If ``True``, fail on invalid input items.
    chunk_size : int
        Spatial chunk size for processing.
    infer_batch_size : int or None
        Optional explicit inference batch size.
    num_workers : int
        Worker count for preprocessing/export tasks.
    continue_on_error : bool
        If ``True``, continue after per-item errors.
    max_retries : int
        Retry count for retryable operations.
    retry_backoff_s : float
        Backoff delay in seconds between retries.
    async_write : bool
        If ``True``, write outputs asynchronously.
    writer_workers : int
        Writer worker count for async output.
    resume : bool
        Whether to resume from prior manifest/output state.
    show_progress : bool
        Whether to display progress indicators.
    input_prep : InputPrepSpec or str or None
        API-side input preprocessing policy.
    """

    format: str = "npz"
    save_inputs: bool = True
    save_embeddings: bool = True
    save_manifest: bool = True
    fail_on_bad_input: bool = False
    chunk_size: int = 16
    infer_batch_size: int | None = None
    num_workers: int = 8
    continue_on_error: bool = False
    max_retries: int = 0
    retry_backoff_s: float = 0.0
    async_write: bool = True
    writer_workers: int = 2
    resume: bool = False
    show_progress: bool = True
    input_prep: InputPrepSpec | str | None = "resize"

    @property
    def effective_infer_batch_size(self) -> int:
        """Return normalized inference batch size.

        Returns
        -------
        int
            Positive batch size, falling back to ``chunk_size`` when unset.
        """
        return max(1, int(self.infer_batch_size or self.chunk_size))

    @property
    def effective_chunk_size(self) -> int:
        """Return normalized chunk size.

        Returns
        -------
        int
            Positive chunk size.
        """
        return max(1, int(self.chunk_size))
