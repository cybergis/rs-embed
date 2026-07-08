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

from dataclasses import asdict, replace
from typing import Any

from .core.embedding import Embedding
from .core.errors import ModelError, ProviderError
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
from .providers.fetch import fetch_sensor_patch_chw as _fetch_sensor_patch_chw
from .providers.resolution import create_provider_for_backend
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
from .tools.inspection import checks_save_dir, inspect_chw, save_quicklook_rgb
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
    model_manages_own_input_prep as _model_manages_own_input_prep,
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


def _warn_managed_input_prep(model_n: str, input_prep: Any) -> None:
    import warnings

    if not _model_manages_own_input_prep(model_n):
        return
    # ``None`` is the unset/package-default sentinel; don't warn when the user
    # made no explicit choice (the model manages its own tiling regardless).
    if input_prep is None:
        return
    if _resolve_input_prep_spec(input_prep).mode != "resize":
        warnings.warn(
            f"model='{model_n}' manages spatial tiling automatically based on "
            "request size; input_prep is ignored.",
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
    input_prep: InputPrepSpec | str | None = None,
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
        API-side input preprocessing policy. ``None`` (the default) uses the
        package default ``"tile"`` (large inputs are tiled + stitched to preserve
        native resolution). Pass ``"resize"`` to downsample to the model image
        size, or ``"auto"`` to tile only when beneficial.
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
    _warn_managed_input_prep(_normalize_model_name(model), input_prep)
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
    input_prep: InputPrepSpec | str | None = None,
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
        API-side input preprocessing policy. ``None`` (the default) uses the
        package default ``"tile"`` (large inputs are tiled + stitched to preserve
        native resolution). Pass ``"resize"`` to downsample to the model image
        size, or ``"auto"`` to tile only when beneficial.
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
    _warn_managed_input_prep(_normalize_model_name(model), input_prep)
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
    input_prep: InputPrepSpec | str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
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
        :class:`ExportModelRequest` is the single per-model channel: use it
        to give one model its own ``sensor``, ``fetch``, or ``modality``, and
        use :meth:`ExportModelRequest.configure` to pass model-specific
        settings (e.g. variant selection)::

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
        Default sensor for all models unless overridden per model via
        :class:`ExportModelRequest`.
    fetch : FetchSpec or None
        Default fetch-policy override for all models unless overridden per
        model via :class:`ExportModelRequest`. Cannot be combined with
        ``sensor``.
    modality : str or None
        Optional global modality selector applied to models that expose
        public modality switching; override per model via
        :class:`ExportModelRequest`.
    input_prep : InputPrepSpec or str or None
        API-side input preprocessing policy, resolved per model exactly as
        :func:`get_embedding` does. ``None`` (the default) uses the package
        default ``"tile"`` (large inputs are tiled + stitched to preserve
        native resolution). Pass ``"resize"`` to downsample to the model
        image size, or ``"auto"`` to tile only when beneficial. This is the
        recommended way to set the policy; ``config.input_prep`` is
        equivalent, but passing both raises :class:`ModelError`.

    Returns
    -------
    list[dict[str, Any]] or dict[str, Any]
        For :meth:`ExportTarget.per_item` targets: a list with one manifest
        dict per spatial point, ordered by ``point_index``. Stable keys per
        manifest: ``point_index`` (int), ``status`` (``"ok"``, ``"partial"``,
        or ``"failed"``), ``models`` (one entry per model with ``model``,
        ``status``, and ``error`` when failed), ``summary``
        (``total_models`` / ``ok_models`` / ``failed_models``), and the
        written file's path under the format-specific key (``npz_path`` or
        ``nc_path``). Points skipped by ``config.resume`` instead carry
        ``resume_skipped=True`` and ``resume_output_path``.

        For :meth:`ExportTarget.combined` targets: a single manifest dict
        for the whole run with ``status``, ``n_items``, ``models``,
        ``summary``, and the output path under the format-specific key
        (``npz_path`` or ``nc_path``). When ``config.resume`` finds the
        export already complete, the returned manifest carries
        ``resume_skipped=True`` and ``resume_output_path``.

    Raises
    ------
    ModelError
        If arguments are invalid or unsupported (for example empty inputs,
        unsupported format, incompatible model/backend settings, or
        ``input_prep`` passed both top-level and via ``config``).
    SpecError
        If spatial or temporal specifications fail validation.
    """
    from .pipelines.exporter import BatchExporter

    if not isinstance(spatials, list) or len(spatials) == 0:
        raise ModelError("spatials must be a non-empty list[SpatialSpec].")

    if input_prep is not None and config.input_prep is not None:
        raise ModelError(
            "input_prep was passed both as export_batch(input_prep=...) and as "
            "ExportConfig(input_prep=...); pass it once (the top-level "
            "parameter is recommended)."
        )
    if input_prep is not None:
        config = replace(config, input_prep=input_prep)

    if config.input_prep is not None:
        model_names_n = {
            _normalize_model_name(_m if isinstance(_m, str) else _m.name) for _m in models
        }
        for _model_n in sorted(model_names_n):
            _warn_managed_input_prep(_model_n, config.input_prep)

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

    # Resolve models before the resume short-circuit: the resolved configs are
    # the canonical input to the request fingerprint, and an invalid model
    # list should fail rather than be reported "already complete".
    model_configs, resolved_backend = _resolve_export_model_configs(
        models=models,
        backend_n=backend_n,
        temporal=temporal,
        output=output,
        sensor=sensor,
        fetch=fetch,
        modality=modality,
    )

    resume_manifest = _maybe_return_completed_combined_resume(
        target=export_target,
        config=export_config,
        model_configs=model_configs,
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend_n,
        device=device_n,
    )
    if resume_manifest is not None:
        return resume_manifest

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


# -----------------------------------------------------------------------------
# Public: provider patch inspection
# -----------------------------------------------------------------------------


def inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: tuple[float, float] | None = None,
    return_array: bool = False,
) -> dict[str, Any]:
    """Download a patch from a provider and return an input inspection report.

    Does **not** run any embedding model. Useful for verifying that a spatial
    location and sensor configuration produce valid imagery before committing
    to a full export.

    Parameters
    ----------
    spatial : SpatialSpec
        Spatial location to inspect.
    temporal : TemporalSpec or None
        Optional temporal filter.
    sensor : SensorSpec
        Sensor/collection configuration used for the download.
    backend : str
        Provider backend name (default ``"gee"``).
    name : str
        Label used in the report and quicklook filename (default
        ``"gee_patch"``).
    value_range : tuple[float, float] or None
        Optional ``(min, max)`` range for value-range checks in the report.
    return_array : bool
        If ``True``, attach the raw ``np.ndarray`` as ``array_chw`` in the
        returned dict (not JSON-serializable).

    Returns
    -------
    dict[str, Any]
        JSON-serializable inspection report with keys ``ok``, ``report``,
        ``sensor``, ``temporal``, ``backend``, and ``artifacts``.
        When ``return_array=True``, also includes ``array_chw``.

    Raises
    ------
    ProviderError
        If the backend name is empty or the provider fails to initialize.
    """
    backend_name = str(backend).strip().lower()
    if not backend_name:
        raise ProviderError("backend must be a non-empty provider name.")
    try:
        provider = create_provider_for_backend(backend_name, allow_auto=False)
    except ModelError as exc:
        raise ProviderError(str(exc)) from exc
    x_chw = _fetch_sensor_patch_chw(
        provider,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
    )

    report = inspect_chw(
        x_chw,
        name=name,
        expected_channels=len(sensor.bands),
        value_range=value_range,
        fill_value=sensor.fill_value,
    )

    artifacts: dict[str, Any] = {}
    save_dir = checks_save_dir(sensor)
    if save_dir and x_chw.ndim == 3 and x_chw.shape[0] >= 3:
        try:
            import datetime as _dt
            import os

            ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(save_dir, f"{name}_{ts}.png")
            save_quicklook_rgb(x_chw, path=path, bands=(0, 1, 2))
            artifacts["quicklook_rgb"] = path
        except Exception as e:
            artifacts["quicklook_rgb_error"] = repr(e)

    out: dict[str, Any] = {
        "ok": bool(report.get("ok", False)),
        "report": report,
        "sensor": asdict(sensor),
        "temporal": asdict(temporal) if temporal is not None else None,
        "backend": backend_name,
        "artifacts": artifacts or None,
    }
    if return_array:
        out["array_chw"] = x_chw
    return out


def inspect_gee_patch(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: tuple[float, float] | None = None,
    return_array: bool = False,
) -> dict[str, Any]:
    """Backwards-compatible wrapper around inspect_provider_patch."""
    return inspect_provider_patch(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        backend=backend,
        name=name,
        value_range=value_range,
        return_array=return_array,
    )
