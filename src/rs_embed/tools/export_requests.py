from __future__ import annotations

import os

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import FetchSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    ModelConfig,
)
from ..core.validation import assert_supported
from .checkpoint_utils import is_incomplete_combined_manifest
from .manifest import (
    combined_resume_manifest,
    export_request_fingerprint,
    load_json_dict,
)
from .model_defaults import resolve_sensor_for_model
from .normalization import _resolve_embedding_api_backend, normalize_model_name
from .runtime import require_model_config_support


def normalize_export_format(format_name: str) -> tuple[str, str]:
    """Validate and normalize an export format name.

    Parameters
    ----------
    format_name : str
        Raw format string (e.g. ``"npz"``, ``"netcdf"``).

    Returns
    -------
    tuple[str, str]
        ``(fmt, ext)`` where *fmt* is the canonical lowercase format name
        and *ext* is the corresponding file extension including the dot
        (e.g. ``(".npz", ".nc")``).

    Raises
    ------
    ModelError
        If *format_name* is not a supported export format.
    """
    fmt = str(format_name).strip().lower()
    from ..writers import SUPPORTED_FORMATS, get_extension

    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(
            f"Unsupported export format: {format_name!r}. Supported: {SUPPORTED_FORMATS}."
        )
    return fmt, get_extension(fmt)


def normalize_export_target(
    *,
    n_spatials: int,
    ext: str,
    target: ExportTarget,
) -> ExportTarget:
    """Validate and normalize an :class:`ExportTarget`.

    Appends the correct file extension to combined targets and generates
    default per-item point names when none are provided.

    Parameters
    ----------
    n_spatials : int
        Number of spatial points in the export request.
    ext : str
        File extension including the dot (e.g. ``".npz"``), used to
        suffix combined export paths.
    target : ExportTarget
        Raw export target to validate and normalize.

    Returns
    -------
    ExportTarget
        Validated and normalized export target.

    Raises
    ------
    ModelError
        If the target is not an :class:`ExportTarget` instance, required
        fields are missing, or ``target.names`` length mismatches
        ``n_spatials``.
    """
    if not isinstance(target, ExportTarget):
        raise ModelError("target must be an ExportTarget instance.")
    if target.layout == ExportLayout.COMBINED:
        if not target.out_file:
            raise ModelError("ExportTarget.COMBINED requires out_file.")
        out_file = target.out_file if target.out_file.endswith(ext) else (target.out_file + ext)
        return ExportTarget.combined(out_file)
    if target.layout == ExportLayout.PER_ITEM:
        if not target.out_dir:
            raise ModelError("ExportTarget.PER_ITEM requires out_dir.")
        point_names = (
            target.names if target.names is not None else [f"p{i:05d}" for i in range(n_spatials)]
        )
        if len(point_names) != n_spatials:
            raise ModelError("target.names must have the same length as spatials.")
        return ExportTarget.per_item(target.out_dir, names=point_names)
    raise ModelError(f"Unsupported ExportTarget layout: {target.layout!r}.")


def resolve_export_model_configs(
    *,
    models: list[str | ExportModelRequest],
    backend_n: str,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    sensor: SensorSpec | None,
    fetch: FetchSpec | None,
    modality: str | None,
    per_model_sensors: dict[str, SensorSpec] | None,
    per_model_fetches: dict[str, FetchSpec] | None,
    per_model_modalities: dict[str, str] | None,
) -> tuple[list[ModelConfig], dict[str, str]]:
    """Resolve per-model configurations for a batch export.

    Validates each requested model, resolves its effective backend and
    sensor, and returns ready-to-use :class:`ModelConfig` instances.

    Parameters
    ----------
    models : list[str | ExportModelRequest]
        Model identifiers or pre-configured request objects.
    backend_n : str
        Normalized global backend name.
    temporal : TemporalSpec or None
        Temporal filter applied to all models.
    output : OutputSpec
        Output representation policy applied to all models.
    sensor : SensorSpec or None
        Global sensor override (applied when no per-model override exists).
    fetch : FetchSpec or None
        Global fetch-policy override.
    modality : str or None
        Global modality selector.
    per_model_sensors : dict[str, SensorSpec] or None
        Per-model sensor overrides keyed by model name.
    per_model_fetches : dict[str, FetchSpec] or None
        Per-model fetch-policy overrides keyed by model name.
    per_model_modalities : dict[str, str] or None
        Per-model modality overrides keyed by model name.

    Returns
    -------
    tuple[list[ModelConfig], dict[str, str]]
        ``(model_configs, resolved_backend)`` where *model_configs* is an
        ordered list of resolved :class:`ModelConfig` instances and
        *resolved_backend* maps each model name to its effective backend.

    Raises
    ------
    ModelError
        If *models* is empty, contains invalid types, or any model fails
        validation (unsupported backend, output mode, or model config).
    """
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty list[str] or list[ExportModelRequest].")

    per_model_sensors = per_model_sensors or {}
    per_model_fetches = per_model_fetches or {}
    per_model_modalities = per_model_modalities or {}

    requests: list[ExportModelRequest] = []
    for item in models:
        if isinstance(item, ExportModelRequest):
            requests.append(item)
        elif isinstance(item, str):
            requests.append(ExportModelRequest(name=item))
        else:
            raise ModelError("models entries must be strings or ExportModelRequest instances.")

    model_configs: list[ModelConfig] = []
    resolved_backend: dict[str, str] = {}
    for req in requests:
        model_name = req.name
        model_n = normalize_model_name(model_name)
        eff_backend = _resolve_embedding_api_backend(model_n, backend_n)
        resolved_backend[model_name] = eff_backend
        cls = get_embedder_cls(model_n)
        try:
            emb_check = cls()
            assert_supported(emb_check, backend=eff_backend, output=output, temporal=temporal)
            require_model_config_support(
                embedder=emb_check,
                model_config=req.model_config,
                method_name="get_embedding",
            )
            desc = emb_check.describe() or {}
        except ModelError:
            raise
        except Exception as _e:
            desc = {}

        modality_eff = req.modality
        if modality_eff is None:
            modality_eff = per_model_modalities.get(model_name, modality)

        sensor_eff = req.sensor
        if sensor_eff is None:
            sensor_eff = per_model_sensors.get(model_name, sensor)

        fetch_eff = req.fetch
        if fetch_eff is None:
            fetch_eff = per_model_fetches.get(model_name, fetch)

        model_configs.append(
            ModelConfig(
                name=model_name,
                backend=eff_backend,
                sensor=resolve_sensor_for_model(
                    model_n,
                    sensor=sensor_eff,
                    fetch=fetch_eff,
                    modality=modality_eff,
                    default_when_missing=True,
                ),
                model_config=req.model_config,
                model_type=str(desc.get("type", "")).lower(),
            )
        )

    return model_configs, resolved_backend


def maybe_return_completed_combined_resume(
    *,
    target: ExportTarget,
    config: ExportConfig,
    model_configs: list[ModelConfig],
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
) -> dict[str, object] | None:
    """Return a completed resume manifest if the export is already finished.

    Short-circuits the export pipeline only when a combined export file
    exists, its sidecar manifest is readable and complete, and the manifest's
    ``request_fingerprint`` matches the current request. A missing or
    unreadable sidecar, or a fingerprint from a different request (including
    pre-fingerprint exports), means the file cannot be trusted as this
    request's result — the export proceeds and rewrites it.

    Parameters
    ----------
    target : ExportTarget
        Export target specifying the output file location.
    config : ExportConfig
        Export configuration; only checked when ``config.resume`` is ``True``.
    model_configs : list[ModelConfig]
        Resolved per-model configurations of the current request.
    spatials : list[SpatialSpec]
        Spatial points from the current export request.
    temporal : TemporalSpec or None
        Temporal filter from the current export request.
    output : OutputSpec
        Output representation policy from the current export request.
    backend : str
        Normalized backend name.
    device : str
        Normalized device name.

    Returns
    -------
    dict[str, object] or None
        Completed resume manifest dict if the export is already done,
        or ``None`` if the export should proceed normally.
    """
    if target.layout != ExportLayout.COMBINED or not config.resume or not target.out_file:
        return None
    if not os.path.exists(target.out_file):
        return None
    json_path = os.path.splitext(target.out_file)[0] + ".json"
    resume_manifest = load_json_dict(json_path)
    if resume_manifest is None or is_incomplete_combined_manifest(resume_manifest):
        return None
    fingerprint = export_request_fingerprint(
        models=model_configs,
        temporal=temporal,
        output=output,
        config=config,
        spatials=spatials,
    )
    if resume_manifest.get("request_fingerprint") != fingerprint:
        return None
    return combined_resume_manifest(
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend,
        device=device,
        out_file=target.out_file,
    )
