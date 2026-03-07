from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.specs import SensorSpec, SpatialSpec


def init_combined_provider(
    *,
    backend: str,
    save_inputs: bool,
    save_embeddings: bool,
    provider_factory: Optional[Callable[[], Any]],
    run_with_retry: Callable[..., Any],
    max_retries: int,
    retry_backoff_s: float,
) -> Optional[Any]:
    if provider_factory is None or (not (save_inputs or save_embeddings)):
        return None
    provider = provider_factory()
    run_with_retry(
        lambda: provider.ensure_ready(),
        retries=max_retries,
        backoff_s=retry_backoff_s,
    )
    return provider


def restore_prefetch_cache_from_manifest(
    *,
    manifest: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    restore_prefetch_checkpoint_cache: Callable[..., Dict[Tuple[int, str], np.ndarray]],
) -> Dict[Tuple[int, str], np.ndarray]:
    inputs_cache: Dict[Tuple[int, str], np.ndarray] = {}
    prefetch_meta = manifest.get("prefetch")
    if isinstance(prefetch_meta, dict):
        inputs_cache.update(
            restore_prefetch_checkpoint_cache(
                arrays=arrays,
                prefetch_meta=prefetch_meta,
            )
        )
    return inputs_cache


def build_combined_prefetch_tasks(
    *,
    provider: Optional[Any],
    spatials: List[SpatialSpec],
    fetch_sensor_by_key: Dict[str, SensorSpec],
    fetch_members: Dict[str, List[str]],
    inputs_cache: Dict[Tuple[int, str], np.ndarray],
) -> List[Tuple[int, str, SensorSpec]]:
    tasks: List[Tuple[int, str, SensorSpec]] = []
    if provider is None:
        return tasks
    for i, _sp in enumerate(spatials):
        for fetch_key, fetch_sensor in fetch_sensor_by_key.items():
            member_keys = fetch_members.get(fetch_key, [])
            if member_keys and all((i, mk) in inputs_cache for mk in member_keys):
                continue
            tasks.append((i, fetch_key, fetch_sensor))
    return tasks


def write_combined_checkpoint(
    *,
    manifest: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    stage: str,
    final: bool,
    out_path: str,
    fmt: str,
    save_manifest: bool,
    json_path: str,
    max_retries: int,
    retry_backoff_s: float,
    write_one_payload: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    manifest["stage"] = stage
    manifest["resume_incomplete"] = not final
    if not final:
        manifest["status"] = "running"
    if final and (not save_manifest):
        manifest.pop("manifest_path", None)
    written = write_one_payload(
        out_path=out_path,
        arrays=arrays,
        manifest=manifest,
        save_manifest=(save_manifest if final else True),
        fmt=fmt,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    if final and (not save_manifest):
        try:
            if os.path.exists(json_path):
                os.remove(json_path)
        except Exception:
            pass
    return written
