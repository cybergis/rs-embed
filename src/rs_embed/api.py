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

from dataclasses import replace
from typing import Any

from .core.embedding import Embedding
from .core.errors import ModelError
from .core.registry import get_embedder_cls as _get_embedder_cls
from .core.specs import (
    FetchSpec,
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
    describe_model_cached as _describe_model_cached,
)
from .tools.runtime import (
    reset_runtime as _reset_runtime_shared,
)
from .tools.runtime import (
    run_embedding_request as _run_embedding_request_shared,
)
from .tools.tiling import _resolve_input_prep_spec as _resolve_input_prep_spec

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _warn_gse_input_prep(model_n: str, input_prep: Any) -> None:
    import warnings

    if model_n != "gse":
        return
    if _resolve_input_prep_spec(input_prep).mode != "resize":
        warnings.warn(
            "model='gse' manages spatial tiling automatically based on request size; "
            "input_prep is ignored.",
            UserWarning,
            stacklevel=3,
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


def describe_model(model: str) -> dict[str, Any]:
    """Return metadata for a model without loading its weights.

    Instantiates the embedder class (a lightweight operation — no checkpoint
    download, no torch import) and calls its :meth:`~EmbedderBase.describe`
    method, which returns a plain dict of static configuration.

    Parameters
    ----------
    model : str
        Canonical model id or any registered alias (e.g. ``"prithvi"``,
        ``"satmae"``, ``"galileo"``).  Call :func:`list_models` to see all
        available ids.

    Returns
    -------
    dict[str, Any]
        Model metadata including input bands, supported output modes,
        default parameters, and architecture notes.  The exact keys vary
        per model but always include ``"type"`` and ``"output"``.

    Raises
    ------
    ModelError
        If *model* is not a known id or alias.

    Examples
    --------
    >>> from rs_embed import describe_model
    >>> info = describe_model("galileo")
    >>> info["output"]
    ['pooled', 'grid']
    """
    model_n = _normalize_model_name(model)
    return dict(_describe_model_cached(model_n))


def reset_runtime() -> dict[str, int]:
    """Clear lazy-import/runtime caches in the current Python process.

    This is mainly useful in notebooks after a failed model import or when you
    want to force fresh embedder instances without restarting the kernel.

    Returns
    -------
    dict[str, int]
        Summary counts describing how many runtime/import caches were cleared.
    """
    return _reset_runtime_shared()


def get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec | None = None,
    fetch: FetchSpec | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
    **model_kwargs: Any,
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
    fetch : FetchSpec or None
        Lightweight fetch-policy override applied to the model default sensor.
        Cannot be combined with ``sensor``.
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
    **model_kwargs
        Model-specific settings passed directly as keyword arguments.
        For example, ``variant="large"`` selects the large DOFA variant.
        The accepted keys depend on the model; call :func:`describe_model`
        to see the ``"model_config"`` schema for a given model.

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

    Examples
    --------
    >>> emb = get_embedding("dofa", spatial=point, temporal=t, variant="large")
    """
    model_config = model_kwargs or None
    _warn_gse_input_prep(_normalize_model_name(model), input_prep)
    _validate_specs(spatial=spatial, temporal=temporal, output=output)
    sensor_eff = _resolve_sensor_for_model(
        _normalize_model_name(model),
        sensor=sensor,
        fetch=fetch,
        modality=modality,
        default_when_missing=False,
    )
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor_eff,
        model_config=model_config,
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
    fetch: FetchSpec | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
    **model_kwargs: Any,
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
    fetch : FetchSpec or None
        Lightweight fetch-policy override applied to the model default sensor.
        Cannot be combined with ``sensor``.
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
    **model_kwargs
        Model-specific settings passed directly as keyword arguments.
        For example, ``variant="large"`` selects the large DOFA variant.
        The accepted keys depend on the model; call :func:`describe_model`
        to see the ``"model_config"`` schema for a given model.

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

    Examples
    --------
    >>> embs = get_embeddings_batch("dofa", spatials=points, temporal=t, variant="large")
    """
    model_config = model_kwargs or None
    _warn_gse_input_prep(_normalize_model_name(model), input_prep)
    _validate_spatial_list(spatials=spatials, temporal=temporal, output=output)
    sensor_eff = _resolve_sensor_for_model(
        _normalize_model_name(model),
        sensor=sensor,
        fetch=fetch,
        modality=modality,
        default_when_missing=False,
    )
    ctx = _prepare_embedding_request_context(
        model=model,
        temporal=temporal,
        sensor=sensor_eff,
        model_config=model_config,
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
    target: ExportTarget,
    config: ExportConfig = ExportConfig(),
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: SensorSpec | None = None,
    fetch: FetchSpec | None = None,
    modality: str | None = None,
    per_model_sensors: dict[str, SensorSpec] | None = None,
    per_model_fetches: dict[str, FetchSpec] | None = None,
    per_model_modalities: dict[str, str] | None = None,
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
        Model identifiers or per-model request objects.  To pass model-specific
        settings (e.g. variant selection), use
        :meth:`ExportModelRequest.configure` instead of raw strings::

            models=[ExportModelRequest.configure("dofa", variant="large")]
    target : ExportTarget
        Output destination: use :meth:`ExportTarget.per_item` for per-item
        directory exports or :meth:`ExportTarget.combined` for a single file.
    config : ExportConfig
        Runtime configuration (format, workers, resume, etc.).
        Defaults to :class:`ExportConfig` with all defaults applied.
    backend : str
        Backend/provider selector.
    device : str
        Target inference device.
    output : OutputSpec
        Embedding output representation policy.
    sensor : SensorSpec or None
        Default sensor for all models unless overridden.
    fetch : FetchSpec or None
        Default fetch-policy override for all models unless overridden.
        Cannot be combined with ``sensor``.
    modality : str or None
        Optional global modality selector applied to models that expose
        public modality switching.
    per_model_sensors : dict[str, SensorSpec] or None
        Per-model sensor overrides keyed by model name.
    per_model_fetches : dict[str, FetchSpec] or None
        Per-model fetch-policy overrides keyed by model name. Cannot be
        combined with sensor overrides for the same model.
    per_model_modalities : dict[str, str] or None
        Optional per-model modality overrides keyed by model name.

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

    if _resolve_input_prep_spec(config.input_prep).mode != "resize":
        for _m in models:
            if _normalize_model_name(_m if isinstance(_m, str) else _m.name) == "gse":
                _warn_gse_input_prep("gse", config.input_prep)
                break

    backend_n = _normalize_backend_name(backend)
    device_n = _normalize_device_name(device)

    fmt, ext = _normalize_export_format(config.format)
    export_config = replace(config, format=fmt)

    export_target = _normalize_export_target(
        n_spatials=len(spatials),
        ext=ext,
        target=target,
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
        fetch=fetch,
        modality=modality,
        per_model_sensors=per_model_sensors,
        per_model_fetches=per_model_fetches,
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
