"""Public Facade and Entry Point.

This module is the single boundary between callers and the internal pipeline
stack.  It should contain **no** heavy execution logic — only configuration
and delegation.

Responsibilities
----------------
1. **Validation** — ensure all input Specs (Spatial, Temporal, Output) are
   valid before any processing begins.
2. **Normalisation** — convert user-friendly strings (e.g. ``"sentinel-2"``,
   ``"cuda"``) into strict internal objects.
3. **Context Resolution** — route each request to the correct backend
   (Provider vs. Precomputed) and device.
4. **Delegation** — instantiate the appropriate Pipeline or Embedder and
   hand off execution.

Flow summary
------------
1. Validate and normalise user inputs / specs.
2. Resolve request context (model / backend / device / sensor / input prep).
3. Execute single / batch embedding, or delegate batch export to
   :class:`BatchExporter`.

"""

from __future__ import annotations

from typing import Any

from .core.embedding import Embedding
from .core.errors import ModelError
from .core.specs import (
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .core.types import (
    ExportConfig,
    ExportModelRequest,
    ExportTarget,
)
from .core.validation import (
    assert_supported as _assert_supported,
)
from .core.validation import (
    validate_spatial_list as _validate_spatial_list,
)
from .core.validation import (
    validate_specs as _validate_specs,
)
from .embedders.catalog import MODEL_ALIASES, MODEL_SPECS
from .tools.export_requests import (
    maybe_return_completed_combined_resume as _maybe_return_completed_combined_resume,
)
from .tools.export_requests import (
    normalize_export_config as _normalize_export_config,
)
from .tools.export_requests import (
    normalize_export_format as _normalize_export_format,
)
from .tools.export_requests import (
    normalize_export_target as _normalize_export_target,
)
from .tools.export_requests import (
    resolve_export_model_configs as _resolve_export_model_configs,
)
from .tools.model_defaults import (
    resolve_sensor_for_model as _resolve_sensor_for_model,
)
from .tools.normalization import (
    # Re-exported so `from rs_embed.api import ...` in tests/downstream still works.
    _default_provider_backend_for_api,  # noqa: F401
    _probe_model_describe,  # noqa: F401
    _resolve_embedding_api_backend,  # noqa: F401
)
from .tools.normalization import (
    normalize_backend_name as _normalize_backend_name,
)
from .tools.normalization import (
    normalize_device_name as _normalize_device_name,
)
from .tools.normalization import (
    normalize_model_name as _normalize_model_name,
)
from .tools.progress import create_progress as _create_progress
from .tools.runtime import (
    _prepare_embedding_request_context,
    provider_factory_for_backend,
)
from .tools.runtime import (
    run_embedding_request as _run_embedding_request_shared,
)

# -----------------------------------------------------------------------------
# Public: embeddings
# -----------------------------------------------------------------------------

def list_models(*, include_aliases: bool = False) -> list[str]:
    """Return the stable model catalog, independent of runtime lazy-load state.

    Parameters
    ----------
    include_aliases : bool
        If ``True``, include alias names in addition to canonical ids.

    Returns
    -------
    list[str]
        Sorted model names available in the catalog.
    """
    model_ids = set(MODEL_SPECS.keys())
    if include_aliases:
        model_ids.update(MODEL_ALIASES.keys())
    return sorted(model_ids)

def get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> Embedding:
    """Compute a single embedding.

    Parameters
    ----------
    model : str
        Model identifier or alias.
    spatial : SpatialSpec
        Spatial location/extent to embed.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec or None
        Optional sensor override.
    modality : str or None
        Optional modality selector for models that expose multiple input
        branches.
    output : OutputSpec
        Output representation policy.
    backend : str
        Backend/provider selector (for example ``"auto"`` or ``"gee"``).
    device : str
        Target inference device.
    input_prep : InputPrepSpec or str or None
        Optional API-side input preprocessing policy.

    Returns
    -------
    Embedding
        Normalized embedding output for the requested location.

    Raises
    ------
    ModelError
        If inputs/specs are invalid or requested model/backend configuration is
        unsupported.
    SpecError
        If spatial or temporal specifications fail validation.

    Notes
    -----
    This function reuses a cached embedder instance when possible to avoid
    repeatedly loading model weights / initializing providers.
    """
    _validate_specs(spatial=spatial, temporal=temporal, output=output)
    sensor_eff = _resolve_sensor_for_model(
        _normalize_model_name(model),
        sensor=sensor,
        modality=modality,
        default_when_missing=False,
    )
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor_eff,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request_shared(
        spatials=[spatial],
        temporal=temporal,
        sensor=sensor_eff,
        output=output,
        ctx=ctx,
    )[0]

def get_embeddings_batch(
    model: str,
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> list[Embedding]:
    """Compute embeddings for multiple spatials using a shared embedder instance.

    Parameters
    ----------
    model : str
        Model identifier or alias.
    spatials : list[SpatialSpec]
        Spatial requests to embed.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec or None
        Optional sensor override.
    modality : str or None
        Optional modality selector for models that expose multiple input
        branches.
    output : OutputSpec
        Output representation policy.
    backend : str
        Backend/provider selector.
    device : str
        Target inference device.
    input_prep : InputPrepSpec or str or None
        Optional API-side input preprocessing policy.

    Returns
    -------
    list[Embedding]
        Embeddings in the same order as ``spatials``.

    Raises
    ------
    ModelError
        If inputs/specs are invalid or requested model/backend configuration is
        unsupported.
    SpecError
        If spatial or temporal specifications fail validation.
    """
    _validate_spatial_list(spatials=spatials, temporal=temporal, output=output)
    sensor_eff = _resolve_sensor_for_model(
        _normalize_model_name(model),
        sensor=sensor,
        modality=modality,
        default_when_missing=False,
    )
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor_eff,
        output=output,
        backend=backend,
        device=device,
        input_prep=input_prep,
    )
    return _run_embedding_request_shared(
        spatials=spatials,
        temporal=temporal,
        sensor=sensor_eff,
        output=output,
        ctx=ctx,
    )

# -----------------------------------------------------------------------------
# Public: batch export (core)
# -----------------------------------------------------------------------------

def export_batch(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    models: list[str | ExportModelRequest],
    target: ExportTarget | None = None,
    config: ExportConfig | None = None,
    out: str | None = None,
    layout: str | None = None,
    out_dir: str | None = None,
    out_path: str | None = None,
    names: list[str] | None = None,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: SensorSpec | None = None,
    modality: str | None = None,
    per_model_sensors: dict[str, SensorSpec] | None = None,
    per_model_modalities: dict[str, str] | None = None,
    format: str = "npz",
    save_inputs: bool = True,
    save_embeddings: bool = True,
    save_manifest: bool = True,
    fail_on_bad_input: bool = False,
    chunk_size: int = 16,
    infer_batch_size: int | None = None,
    num_workers: int = 8,
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    async_write: bool = True,
    writer_workers: int = 2,
    resume: bool = False,
    show_progress: bool = True,
    input_prep: InputPrepSpec | str | None = "resize",
) -> Any:
    """Export inputs + embeddings for many spatials and many models.

    This is the recommended high-level entrypoint for batch export.
    Delegates to :class:`~rs_embed.pipelines.exporter.BatchExporter`.

    Parameters
    ----------
    spatials : list[SpatialSpec]
        Spatial requests to export.
    temporal : TemporalSpec or None
        Optional temporal filter applied to all spatial requests.
    models : list[str | ExportModelRequest]
        Model identifiers or per-model request objects.
    target : ExportTarget or None
        Preferred output target object for new code.
    config : ExportConfig or None
        Preferred export runtime configuration object for new code.
    out : str or None
        Legacy output path hint. Combined layout when file-like, per-item
        layout when directory-like.
    layout : str or None
        Explicit layout override (``"combined"`` or ``"per_item"``).
    out_dir : str or None
        Directory path for per-item exports.
    out_path : str or None
        File path for combined exports.
    names : list[str] or None
        Legacy names aligned with ``spatials`` for per-item outputs.
    backend : str
        Backend/provider selector.
    device : str
        Target inference device.
    output : OutputSpec
        Embedding output representation policy.
    sensor : SensorSpec or None
        Default sensor for all models unless overridden.
    modality : str or None
        Optional global modality selector applied to models that expose
        public modality switching.
    per_model_sensors : dict[str, SensorSpec] or None
        Per-model sensor overrides keyed by model name.
    per_model_modalities : dict[str, str] or None
        Optional per-model modality overrides keyed by model name.
    format / save_* / fail_on_bad_input / chunk_size / infer_batch_size /
    num_workers / continue_on_error / max_retries / retry_backoff_s /
    async_write / writer_workers / resume / show_progress / input_prep
        Legacy config-style keyword arguments. New code should prefer
        ``config=ExportConfig(...)``.

    Returns
    -------
    Any
        Export result object returned by :class:`BatchExporter`.

    Raises
    ------
    ModelError
        If arguments are invalid or unsupported (for example empty inputs,
        unsupported format, or incompatible model/backend settings).
    SpecError
        If spatial or temporal specifications fail validation.
    """
    from .pipelines.exporter import BatchExporter

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty list[SpatialSpec].")

    backend_n = _normalize_backend_name(backend)
    device_n = _normalize_device_name(device)

    export_config = _normalize_export_config(
        config=config,
        format=format,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        save_manifest=save_manifest,
        fail_on_bad_input=fail_on_bad_input,
        chunk_size=chunk_size,
        infer_batch_size=infer_batch_size,
        num_workers=num_workers,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        async_write=async_write,
        writer_workers=writer_workers,
        resume=resume,
        show_progress=show_progress,
        input_prep=input_prep,
    )
    _fmt, ext = _normalize_export_format(export_config.format)

    export_target = _normalize_export_target(
        n_spatials=len(spatials),
        ext=ext,
        target=target,
        out=out,
        layout=layout,
        out_dir=out_dir,
        out_path=out_path,
        names=names,
    )

    _validate_spatial_list(spatials=spatials, temporal=temporal, output=output)

    resume_manifest = _maybe_return_completed_combined_resume(
        target=export_target,
        config=export_config,
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend_n,
        device=device_n,
    )
    if resume_manifest is not None:
        return resume_manifest

    model_configs, resolved_backend = _resolve_export_model_configs(
        models=models,
        backend_n=backend_n,
        temporal=temporal,
        output=output,
        sensor=sensor,
        modality=modality,
        per_model_sensors=per_model_sensors,
        per_model_modalities=per_model_modalities,
    )

    exporter = BatchExporter(
        spatials=spatials,
        temporal=temporal,
        models=model_configs,
        target=export_target,
        output=output,
        config=export_config,
        backend=backend_n,
        resolved_backend=resolved_backend,
        device=device_n,
        provider_factory=provider_factory_for_backend(backend_n),
        progress_factory=_create_progress,
    )
    return exporter.run()
