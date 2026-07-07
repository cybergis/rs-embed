"""Domain types for rs-embed engine layer.

Provides enums, result objects, and configuration dataclasses that replace
the raw dicts and string constants scattered through the old functional code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np

from .specs import FetchSpec, InputPrepSpec, SensorSpec

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


# ── Embedder capabilities ─────────────────────────────────────────


@dataclass(frozen=True)
class EmbedderCapabilities:
    """Explicit pipeline-routing capabilities declared on an embedder class.

    Each field answers "does this embedder's method accept this optional
    parameter?" — the questions the runtime needs to route requests
    (prefetched inputs, fetch metadata, model-specific settings).

    Declared as a class attribute (``capabilities = EmbedderCapabilities(...)``)
    this is the single source of truth for routing.  Embedder classes that do
    not declare it (``capabilities = None``) fall back to signature
    introspection for backward compatibility; all in-tree embedders must
    declare it, and a contract test asserts the declaration matches the
    actual method signatures.

    Attributes
    ----------
    input_chw : bool
        ``get_embedding()`` accepts a prefetched ``input_chw`` array.
    fetch_meta : bool
        ``get_embedding()`` accepts ``fetch_meta`` (e.g. the fetch-square
        ``roi_window_geo`` crop window).
    fetch_temporal_mode : bool
        ``fetch_input()`` accepts a ``temporal_mode`` override.
    batch_fetch_metas : bool
        ``get_embeddings_batch_from_inputs()`` accepts per-item
        ``fetch_metas``.
    model_config_single : bool
        ``get_embedding()`` accepts ``model_config``.
    model_config_batch : bool
        ``get_embeddings_batch()`` accepts ``model_config``.
    model_config_batch_inputs : bool
        ``get_embeddings_batch_from_inputs()`` accepts ``model_config``.
    """

    input_chw: bool = False
    fetch_meta: bool = False
    fetch_temporal_mode: bool = False
    batch_fetch_metas: bool = False
    model_config_single: bool = False
    model_config_batch: bool = False
    model_config_batch_inputs: bool = False


# Single source of the (method, parameter) -> capability-field mapping used by
# both the runtime introspection helpers and the embedder base class.
CAPABILITY_FIELD_BY_METHOD_PARAM: dict[tuple[str, str], str] = {
    ("get_embedding", "input_chw"): "input_chw",
    ("get_embedding", "fetch_meta"): "fetch_meta",
    ("get_embedding", "model_config"): "model_config_single",
    ("get_embeddings_batch", "model_config"): "model_config_batch",
    ("get_embeddings_batch_from_inputs", "model_config"): "model_config_batch_inputs",
    ("get_embeddings_batch_from_inputs", "fetch_metas"): "batch_fetch_metas",
    ("fetch_input", "temporal_mode"): "fetch_temporal_mode",
}


def declared_capability(
    embedder_cls: type,
    method_name: str,
    param_name: str,
) -> bool | None:
    """Return the declared capability for ``(method_name, param_name)``.

    Returns ``None`` when the class declares no ``capabilities`` object or the
    pair is not a known capability — callers then fall back to signature
    introspection.
    """
    caps = getattr(embedder_cls, "capabilities", None)
    if not isinstance(caps, EmbedderCapabilities):
        return None
    field = CAPABILITY_FIELD_BY_METHOD_PARAM.get((method_name, param_name))
    if field is None:
        return None
    return bool(getattr(caps, field))


# ── Fetch results ─────────────────────────────────────────────────


@dataclass
class FetchResult:
    """Result of a model-specific input fetch.

    Returned by ``EmbedderBase.fetch_input()`` to carry both the pixel
    array and any fetch-time metadata (e.g. S1 IW-mode decisions,
    SatVision-TOA fallback provenance).

    Attributes
    ----------
    data : np.ndarray
        CHW float32 pixel array.
    meta : dict[str, Any]
        Fetch-time metadata.  Empty dict for generic models.
    """

    data: np.ndarray
    meta: dict[str, Any]


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
    model_config : dict[str, Any] or None
        Optional user-facing model-specific runtime settings such as
        ``{"variant": "large"}``.
    model_type : str
        Lower-level model family/type hint.
    """

    name: str
    backend: str
    sensor: SensorSpec | None = None
    model_config: dict[str, Any] | None = None
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

    Prefer :meth:`configure` to pass model-specific settings as keyword
    arguments rather than constructing ``model_config`` dicts manually.

    Attributes
    ----------
    name : str
        Model identifier or alias.
    sensor : SensorSpec or None
        Optional per-model sensor override for provider-backed models.
    fetch : FetchSpec or None
        Optional per-model fetch-policy override applied to the model default
        sensor. Cannot be combined with ``sensor``.
    modality : str or None
        Optional per-model modality selector.
    model_config : dict[str, Any] or None
        Model-specific runtime settings.  Use :meth:`configure` to build
        this from keyword arguments instead of constructing the dict manually.
    """

    name: str
    sensor: SensorSpec | None = None
    fetch: FetchSpec | None = None
    modality: str | None = None
    model_config: dict[str, Any] | None = None

    @classmethod
    def configure(
        cls,
        name: str,
        *,
        sensor: SensorSpec | None = None,
        fetch: FetchSpec | None = None,
        modality: str | None = None,
        **model_kwargs: Any,
    ) -> ExportModelRequest:
        """Create a request with model settings as direct keyword arguments.

        Parameters
        ----------
        name : str
            Model identifier or alias.
        sensor : SensorSpec or None
            Optional per-model sensor override.
        fetch : FetchSpec or None
            Optional per-model fetch-policy override.
        modality : str or None
            Optional per-model modality selector.
        **model_kwargs
            Model-specific settings (e.g. ``variant="large"``).

        Returns
        -------
        ExportModelRequest

        Examples
        --------
        >>> req = ExportModelRequest.configure("dofa", variant="large")
        >>> export_batch(spatials=[...], models=[req], ...)
        """
        return cls(
            name=name,
            sensor=sensor,
            fetch=fetch,
            modality=modality,
            model_config=model_kwargs or None,
        )


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
    def combined(cls, out_file: str) -> ExportTarget:
        """Build a combined-file export target."""
        return cls(layout=ExportLayout.COMBINED, out_file=out_file)

    @classmethod
    def per_item(cls, out_dir: str, *, names: list[str] | None = None) -> ExportTarget:
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
        API-side input preprocessing policy, resolved per model exactly as
        :func:`get_embedding` does. ``None`` (the default) uses the package
        default ``"tile"`` (large inputs are tiled + stitched to preserve native
        resolution) for every model. Image-level ViT grid models (satmae,
        scalemae, ...) also tile by default but warn that tiled grids can show
        stitching seams; pass ``"resize"`` to downsample to the model image size
        for a seamless grid, or ``"auto"`` to tile only when beneficial.
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
    input_prep: InputPrepSpec | str | None = None

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
