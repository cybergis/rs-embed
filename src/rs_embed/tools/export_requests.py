from __future__ import annotations

import os
from dataclasses import replace

from ..core.errors import ModelError
from ..core.registry import get_embedder_cls
from ..core.specs import InputPrepSpec, OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from ..core.types import (
    ExportConfig,
    ExportLayout,
    ExportModelRequest,
    ExportTarget,
    ModelConfig,
)
from ..core.validation import assert_supported
from .checkpoint_utils import is_incomplete_combined_manifest
from .manifest import combined_resume_manifest, load_json_dict
from .model_defaults import resolve_sensor_for_model
from .normalization import _resolve_embedding_api_backend, normalize_model_name


def normalize_export_layout(layout: str) -> ExportLayout:
    layout_n = str(layout).strip().lower().replace("-", "_")
    if layout_n in {"combined", "single_file", "file"}:
        return ExportLayout.COMBINED
    if layout_n in {"per_item", "dir", "directory"}:
        return ExportLayout.PER_ITEM
    raise ModelError(f"Unsupported export layout: {layout!r}. Supported: 'combined', 'per_item'.")

def normalize_export_format(format_name: str) -> tuple[str, str]:
    fmt = str(format_name).strip().lower()
    from ..writers import SUPPORTED_FORMATS, get_extension

    if fmt not in SUPPORTED_FORMATS:
        raise ModelError(
            f"Unsupported export format: {format_name!r}. Supported: {SUPPORTED_FORMATS}."
        )
    return fmt, get_extension(fmt)

def _resolve_export_batch_target(
    *,
    n_spatials: int,
    ext: str,
    out: str | None,
    layout: str | None,
    out_dir: str | None,
    out_path: str | None,
    names: list[str] | None,
) -> ExportTarget:
    if (out is not None) or (layout is not None):
        if out is None or layout is None:
            raise ModelError("Provide both out and layout when using the decoupled output API.")
        if out_dir is not None or out_path is not None:
            raise ModelError("Use either out+layout or out_dir/out_path, not both.")
        layout_enum = normalize_export_layout(layout)
        if layout_enum == ExportLayout.COMBINED:
            out_path = out
        else:
            out_dir = out

    if out_dir is None and out_path is None:
        raise ModelError("export_batch requires out_dir or out_path.")
    if out_dir is not None and out_path is not None:
        raise ModelError("Provide only one of out_dir or out_path.")

    if out_path is not None:
        out_file = out_path if out_path.endswith(ext) else (out_path + ext)
        return ExportTarget(layout=ExportLayout.COMBINED, out_file=out_file)

    assert out_dir is not None
    point_names = names if names is not None else [f"p{i:05d}" for i in range(n_spatials)]
    if len(point_names) != n_spatials:
        raise ModelError("names must have the same length as spatials.")
    return ExportTarget(layout=ExportLayout.PER_ITEM, out_dir=out_dir, names=point_names)

def normalize_export_target(
    *,
    n_spatials: int,
    ext: str,
    target: ExportTarget | None,
    out: str | None,
    layout: str | None,
    out_dir: str | None,
    out_path: str | None,
    names: list[str] | None,
) -> ExportTarget:
    if target is not None:
        if not isinstance(target, ExportTarget):
            raise ModelError("target must be an ExportTarget instance.")
        if any(v is not None for v in (out, layout, out_dir, out_path, names)):
            raise ModelError(
                "Use either target=ExportTarget(...) or legacy out/layout/out_dir/out_path/names arguments, not both."
            )
        if target.layout == ExportLayout.COMBINED:
            if not target.out_file:
                raise ModelError("ExportTarget.COMBINED requires out_file.")
            out_file = target.out_file if target.out_file.endswith(ext) else (target.out_file + ext)
            return ExportTarget.combined(out_file)
        if target.layout == ExportLayout.PER_ITEM:
            if not target.out_dir:
                raise ModelError("ExportTarget.PER_ITEM requires out_dir.")
            point_names = (
                target.names
                if target.names is not None
                else [f"p{i:05d}" for i in range(n_spatials)]
            )
            if len(point_names) != n_spatials:
                raise ModelError("target.names must have the same length as spatials.")
            return ExportTarget.per_item(target.out_dir, names=point_names)
        raise ModelError(f"Unsupported ExportTarget layout: {target.layout!r}.")

    return _resolve_export_batch_target(
        n_spatials=n_spatials,
        ext=ext,
        out=out,
        layout=layout,
        out_dir=out_dir,
        out_path=out_path,
        names=names,
    )

def normalize_export_config(
    *,
    config: ExportConfig | None,
    format: str,
    save_inputs: bool,
    save_embeddings: bool,
    save_manifest: bool,
    fail_on_bad_input: bool,
    chunk_size: int,
    infer_batch_size: int | None,
    num_workers: int,
    continue_on_error: bool,
    max_retries: int,
    retry_backoff_s: float,
    async_write: bool,
    writer_workers: int,
    resume: bool,
    show_progress: bool,
    input_prep: InputPrepSpec | str | None,
) -> ExportConfig:
    default_cfg = ExportConfig()
    legacy_config_used = any(
        [
            format != default_cfg.format,
            save_inputs != default_cfg.save_inputs,
            save_embeddings != default_cfg.save_embeddings,
            save_manifest != default_cfg.save_manifest,
            fail_on_bad_input != default_cfg.fail_on_bad_input,
            chunk_size != default_cfg.chunk_size,
            infer_batch_size != default_cfg.infer_batch_size,
            num_workers != default_cfg.num_workers,
            continue_on_error != default_cfg.continue_on_error,
            max_retries != default_cfg.max_retries,
            retry_backoff_s != default_cfg.retry_backoff_s,
            async_write != default_cfg.async_write,
            writer_workers != default_cfg.writer_workers,
            resume != default_cfg.resume,
            show_progress != default_cfg.show_progress,
            input_prep != default_cfg.input_prep,
        ]
    )
    if config is not None and legacy_config_used:
        raise ModelError(
            "Use either config=ExportConfig(...) or legacy export config keyword arguments, not both."
        )

    if config is not None:
        fmt, _ext = normalize_export_format(config.format)
        return replace(config, format=fmt)

    fmt, _ext = normalize_export_format(format)
    return ExportConfig(
        format=fmt,
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

def resolve_export_model_configs(
    *,
    models: list[str | ExportModelRequest],
    backend_n: str,
    temporal: TemporalSpec | None,
    output: OutputSpec,
    sensor: SensorSpec | None,
    modality: str | None,
    per_model_sensors: dict[str, SensorSpec] | None,
    per_model_modalities: dict[str, str] | None,
) -> tuple[list[ModelConfig], dict[str, str]]:
    if not isinstance(models, list) or len(models) == 0:
        raise ModelError("models must be a non-empty list[str] or list[ExportModelRequest].")

    per_model_sensors = per_model_sensors or {}
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

        model_configs.append(
            ModelConfig(
                name=model_name,
                backend=eff_backend,
                sensor=resolve_sensor_for_model(
                    model_n,
                    sensor=sensor_eff,
                    modality=modality_eff,
                    default_when_missing=True,
                ),
                model_type=str(desc.get("type", "")).lower(),
            )
        )

    return model_configs, resolved_backend

def maybe_return_completed_combined_resume(
    *,
    target: ExportTarget,
    config: ExportConfig,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
) -> dict[str, object] | None:
    if target.layout != ExportLayout.COMBINED or not config.resume or not target.out_file:
        return None
    if not os.path.exists(target.out_file):
        return None
    json_path = os.path.splitext(target.out_file)[0] + ".json"
    resume_manifest = load_json_dict(json_path)
    if is_incomplete_combined_manifest(resume_manifest):
        return None
    return combined_resume_manifest(
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend,
        device=device,
        out_file=target.out_file,
    )
