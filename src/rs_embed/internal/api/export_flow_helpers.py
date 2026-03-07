from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ...core.export_helpers import (
    embedding_to_numpy,
    jsonable,
    sanitize_key,
    sensor_cache_key,
    sha1,
    utc_ts,
)
from ...core.specs import OutputSpec, SensorSpec, SpatialSpec, TemporalSpec
from .api_helpers import fetch_gee_patch_raw, inspect_input_raw, normalize_input_chw, normalize_model_name
from .checkpoint_helpers import (
    drop_model_arrays,
    drop_prefetch_checkpoint_arrays,
    is_incomplete_combined_manifest,
    load_saved_arrays,
    restore_prefetch_checkpoint_cache,
    store_prefetch_checkpoint_arrays,
)
from .combined_flow_helpers import (
    CombinedModelDeps,
    CombinedPrefetchDeps,
    get_or_fetch_input,
    run_combined_prefetch_tasks,
    run_pending_models,
)
from .combined_helpers import collect_input_refs_by_sensor, init_combined_export_state, summarize_combined_models
from .combined_orchestration_helpers import (
    build_combined_prefetch_tasks,
    init_combined_provider,
    restore_prefetch_cache_from_manifest,
    write_combined_checkpoint,
)
from .manifest_helpers import load_json_dict
from .point_payload_helpers import PointPayloadDeps, build_one_point_payload as _build_one_point_payload_impl
from .prefetch_helpers import build_gee_prefetch_plan, select_prefetched_channels
from .progress_helpers import create_progress
from .runtime_helpers import (
    call_embedder_get_embedding,
    get_embedder_bundle_cached,
    run_with_retry,
    sensor_key,
    supports_batch_api,
    supports_prefetched_batch_api,
)


def build_one_point_payload(
    *,
    point_index: int,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    models: List[str],
    backend: str,
    resolved_backend: Optional[Dict[str, str]] = None,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
    input_reports: Dict[Tuple[int, str], Dict[str, Any]],
    prefetch_errors: Dict[Tuple[int, str], str],
    pass_input_into_embedder: bool,
    save_inputs: bool,
    save_embeddings: bool,
    fail_on_bad_input: bool,
    continue_on_error: bool,
    max_retries: int,
    retry_backoff_s: float,
    model_progress_cb: Optional[Callable[[str], None]] = None,
    normalize_model_name_fn: Callable[[str], str] = normalize_model_name,
    sensor_key_fn: Callable[[Optional[SensorSpec]], Tuple] = sensor_key,
    get_embedder_bundle_cached_fn: Callable[[str, str, str, Tuple], Tuple[Any, Any]] = get_embedder_bundle_cached,
    run_with_retry_fn: Callable[..., Any] = run_with_retry,
    fetch_gee_patch_raw_fn: Callable[..., np.ndarray] = fetch_gee_patch_raw,
    inspect_input_raw_fn: Callable[..., Dict[str, Any]] = inspect_input_raw,
    call_embedder_get_embedding_fn: Callable[..., Any] = call_embedder_get_embedding,
    provider_factory: Optional[Callable[[], Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    deps = PointPayloadDeps(
        utc_ts=utc_ts,
        jsonable=jsonable,
        sanitize_key=sanitize_key,
        sha1=sha1,
        embedding_to_numpy=embedding_to_numpy,
        sensor_cache_key=sensor_cache_key,
        sensor_key=sensor_key_fn,
        normalize_model_name=normalize_model_name_fn,
        get_embedder_bundle_cached=get_embedder_bundle_cached_fn,
        run_with_retry=run_with_retry_fn,
        fetch_gee_patch_raw=fetch_gee_patch_raw_fn,
        inspect_input_raw=inspect_input_raw_fn,
        call_embedder_get_embedding=call_embedder_get_embedding_fn,
        provider_factory=provider_factory,
    )
    return _build_one_point_payload_impl(
        point_index=point_index,
        spatial=spatial,
        temporal=temporal,
        models=models,
        backend=backend,
        resolved_backend=resolved_backend or {},
        device=device,
        output=output,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
        inputs_cache=inputs_cache,
        input_reports=input_reports,
        prefetch_errors=prefetch_errors,
        pass_input_into_embedder=pass_input_into_embedder,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        fail_on_bad_input=fail_on_bad_input,
        continue_on_error=continue_on_error,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        deps=deps,
        model_progress_cb=model_progress_cb,
    )


def write_one_payload(
    *,
    out_path: str,
    arrays: Dict[str, np.ndarray],
    manifest: Dict[str, Any],
    save_manifest: bool,
    fmt: str,
    max_retries: int,
    retry_backoff_s: float,
    run_with_retry_fn: Callable[..., Any] = run_with_retry,
    jsonable_fn: Callable[[Any], Any] = jsonable,
) -> Dict[str, Any]:
    from ...writers import write_arrays

    return run_with_retry_fn(
        lambda: write_arrays(
            fmt=fmt,
            out_path=out_path,
            arrays=arrays,
            manifest=jsonable_fn(manifest),
            save_manifest=save_manifest,
        ),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )


def export_combined(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str],
    out_path: str,
    backend: str,
    resolved_backend: Optional[Dict[str, str]] = None,
    device: str,
    output: OutputSpec,
    resolved_sensor: Dict[str, Optional[SensorSpec]],
    model_type: Dict[str, str],
    save_inputs: bool,
    save_embeddings: bool,
    save_manifest: bool,
    fail_on_bad_input: bool,
    chunk_size: int,
    num_workers: int,
    fmt: str = "npz",
    continue_on_error: bool = False,
    max_retries: int = 0,
    retry_backoff_s: float = 0.0,
    inference_strategy: str = "auto",
    infer_batch_size: Optional[int] = None,
    resume: bool = False,
    show_progress: bool = False,
    provider_factory: Optional[Callable[[], Any]] = None,
    run_with_retry_fn: Callable[..., Any] = run_with_retry,
    fetch_gee_patch_raw_fn: Callable[..., np.ndarray] = fetch_gee_patch_raw,
    inspect_input_raw_fn: Callable[..., Dict[str, Any]] = inspect_input_raw,
    normalize_input_chw_fn: Callable[..., np.ndarray] = normalize_input_chw,
    select_prefetched_channels_fn: Callable[[np.ndarray, Tuple[int, ...]], np.ndarray] = select_prefetched_channels,
    create_progress_fn: Callable[..., Any] = create_progress,
    get_embedder_bundle_cached_fn: Callable[[str, str, str, Tuple], Tuple[Any, Any]] = get_embedder_bundle_cached,
    sensor_key_fn: Callable[[Optional[SensorSpec]], Tuple] = sensor_key,
    normalize_model_name_fn: Callable[[str], str] = normalize_model_name,
    call_embedder_get_embedding_fn: Callable[..., Any] = call_embedder_get_embedding,
    supports_prefetched_batch_api_fn: Callable[[Any], bool] = supports_prefetched_batch_api,
    supports_batch_api_fn: Callable[[Any], bool] = supports_batch_api,
    embedding_to_numpy_fn: Callable[[Any], np.ndarray] = embedding_to_numpy,
    sensor_cache_key_fn: Callable[[SensorSpec], str] = sensor_cache_key,
    sanitize_key_fn: Callable[[str], str] = sanitize_key,
    jsonable_fn: Callable[[Any], Any] = jsonable,
    utc_ts_fn: Callable[[], str] = utc_ts,
    load_json_dict_fn: Callable[[str], Optional[Dict[str, Any]]] = load_json_dict,
    is_incomplete_combined_manifest_fn: Callable[[Optional[Dict[str, Any]]], bool] = is_incomplete_combined_manifest,
    load_saved_arrays_fn: Callable[..., Dict[str, np.ndarray]] = load_saved_arrays,
    restore_prefetch_checkpoint_cache_fn: Callable[..., Dict[Tuple[int, str], np.ndarray]] = restore_prefetch_checkpoint_cache,
    store_prefetch_checkpoint_arrays_fn: Callable[..., None] = store_prefetch_checkpoint_arrays,
    drop_prefetch_checkpoint_arrays_fn: Callable[[Dict[str, np.ndarray]], None] = drop_prefetch_checkpoint_arrays,
    write_one_payload_fn: Callable[..., Dict[str, Any]] = write_one_payload,
) -> Dict[str, Any]:
    arrays, manifest, pending_models, json_path = init_combined_export_state(
        spatials=spatials,
        temporal=temporal,
        output=output,
        backend=backend,
        device=device,
        models=models,
        out_path=out_path,
        fmt=fmt,
        resume=resume,
        load_json_dict=load_json_dict_fn,
        is_incomplete_combined_manifest=is_incomplete_combined_manifest_fn,
        load_saved_arrays=load_saved_arrays_fn,
        jsonable=jsonable_fn,
        utc_ts=utc_ts_fn,
    )

    provider = init_combined_provider(
        backend=backend,
        save_inputs=save_inputs,
        save_embeddings=save_embeddings,
        provider_factory=provider_factory,
        run_with_retry=run_with_retry_fn,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )

    input_reports: Dict[Tuple[int, str], Dict[str, Any]] = {}
    prefetch_errors: Dict[Tuple[int, str], str] = {}
    (
        sensor_by_key,
        fetch_sensor_by_key,
        sensor_to_fetch,
        sensor_models,
        fetch_members,
    ) = build_gee_prefetch_plan(
        models=models,
        resolved_sensor=resolved_sensor,
        model_type=model_type,
        resolve_bands_fn=(getattr(provider, "normalize_bands", None) if provider is not None else None),
    )

    inputs_cache = restore_prefetch_cache_from_manifest(
        manifest=manifest,
        arrays=arrays,
        restore_prefetch_checkpoint_cache=restore_prefetch_checkpoint_cache_fn,
    )
    tasks = build_combined_prefetch_tasks(
        provider=provider,
        spatials=spatials,
        fetch_sensor_by_key=fetch_sensor_by_key,
        fetch_members=fetch_members,
        inputs_cache=inputs_cache,
    )

    progress = create_progress_fn(
        enabled=bool(show_progress),
        total=(len(tasks) + len(pending_models)),
        desc="export_batch[combined]",
        unit="step",
    )

    def _write_checkpoint(*, stage: str, final: bool = False) -> Dict[str, Any]:
        return write_combined_checkpoint(
            manifest=manifest,
            arrays=arrays,
            stage=stage,
            final=final,
            out_path=out_path,
            fmt=fmt,
            save_manifest=save_manifest,
            json_path=json_path,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            write_one_payload=write_one_payload_fn,
        )

    prefetch_deps = CombinedPrefetchDeps(
        run_with_retry=run_with_retry_fn,
        fetch_gee_patch_raw=fetch_gee_patch_raw_fn,
        normalize_input_chw=normalize_input_chw_fn,
        select_prefetched_channels=select_prefetched_channels_fn,
        inspect_input_raw=inspect_input_raw_fn,
    )

    def _drop_model_arrays(arrays_: Dict[str, np.ndarray], model_name: str) -> None:
        drop_model_arrays(arrays_, model_name, sanitize_key=sanitize_key_fn)

    try:
        if provider is not None and tasks:
            run_combined_prefetch_tasks(
                provider=provider,
                tasks=tasks,
                spatials=spatials,
                temporal=temporal,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                num_workers=num_workers,
                continue_on_error=continue_on_error,
                fail_on_bad_input=fail_on_bad_input,
                fetch_members=fetch_members,
                sensor_to_fetch=sensor_to_fetch,
                sensor_by_key=sensor_by_key,
                sensor_models=sensor_models,
                inputs_cache=inputs_cache,
                input_reports=input_reports,
                prefetch_errors=prefetch_errors,
                progress=progress,
                deps=prefetch_deps,
            )

        if provider is not None:
            store_prefetch_checkpoint_arrays_fn(
                arrays=arrays,
                manifest=manifest,
                sensor_by_key=sensor_by_key,
                inputs_cache=inputs_cache,
                n_items=len(spatials),
            )
            manifest = _write_checkpoint(stage="prefetched", final=False)

        def _get_or_fetch_input(i: int, skey: str, sspec: SensorSpec) -> np.ndarray:
            return get_or_fetch_input(
                i=i,
                skey=skey,
                sspec=sspec,
                provider=provider,
                spatials=spatials,
                temporal=temporal,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                fail_on_bad_input=fail_on_bad_input,
                inputs_cache=inputs_cache,
                input_reports=input_reports,
                prefetch_errors=prefetch_errors,
                deps=prefetch_deps,
            )

        input_refs_by_sensor = collect_input_refs_by_sensor(
            manifest=manifest,
            resolved_sensor=resolved_sensor,
            sensor_cache_key=sensor_cache_key_fn,
        )

        model_deps = CombinedModelDeps(
            create_progress=create_progress_fn,
            drop_model_arrays=_drop_model_arrays,
            jsonable=jsonable_fn,
            sensor_key=sensor_key_fn,
            normalize_model_name=normalize_model_name_fn,
            get_embedder_bundle_cached=get_embedder_bundle_cached_fn,
            sensor_cache_key=sensor_cache_key_fn,
            sanitize_key=sanitize_key_fn,
            run_with_retry=run_with_retry_fn,
            call_embedder_get_embedding=call_embedder_get_embedding_fn,
            supports_prefetched_batch_api=supports_prefetched_batch_api_fn,
            supports_batch_api=supports_batch_api_fn,
            embedding_to_numpy=embedding_to_numpy_fn,
        )
        manifest = run_pending_models(
            pending_models=pending_models,
            arrays=arrays,
            manifest=manifest,
            spatials=spatials,
            temporal=temporal,
            output=output,
            resolved_sensor=resolved_sensor,
            model_type=model_type,
            backend=backend,
            resolved_backend=resolved_backend or {},
            provider_enabled=(provider is not None),
            device=device,
            save_inputs=save_inputs,
            save_embeddings=save_embeddings,
            continue_on_error=continue_on_error,
            chunk_size=chunk_size,
            inference_strategy=inference_strategy,
            infer_batch_size=(chunk_size if infer_batch_size is None else infer_batch_size),
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            show_progress=show_progress,
            input_refs_by_sensor=input_refs_by_sensor,
            get_or_fetch_input_fn=_get_or_fetch_input,
            write_checkpoint_fn=_write_checkpoint,
            progress=progress,
            deps=model_deps,
        )

        manifest["status"], manifest["summary"] = summarize_combined_models(manifest["models"])

        drop_prefetch_checkpoint_arrays_fn(arrays)
        manifest.pop("prefetch", None)
        manifest = _write_checkpoint(stage="done", final=True)
        return manifest
    finally:
        progress.close()
