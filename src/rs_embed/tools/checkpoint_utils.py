from __future__ import annotations

from typing import Any

import numpy as np

from .serialization import jsonable as _jsonable
from ..core.specs import SensorSpec

_CHECKPOINT_PREFETCH_BCHW_PREFIX = "__prefetch_bchw__"
_CHECKPOINT_PREFETCH_CHW_PREFIX = "__prefetch_chw__"

def is_incomplete_combined_manifest(manifest: dict[str, Any] | None) -> bool:
    return bool(isinstance(manifest, dict) and manifest.get("resume_incomplete"))

def load_saved_arrays(*, fmt: str, out_path: str) -> dict[str, np.ndarray]:
    if fmt == "npz":
        with np.load(out_path, allow_pickle=False) as payload:
            return {str(k): np.asarray(payload[k]) for k in payload.files}
    if fmt == "netcdf":
        import xarray as xr

        ds = xr.open_dataset(out_path)
        try:
            return {str(k): np.asarray(ds[k].values) for k in ds.data_vars}
        finally:
            ds.close()
    raise ValueError(f"Unknown format {fmt!r}. Supported: ('npz', 'netcdf')")

def drop_prefetch_checkpoint_arrays(arrays: dict[str, np.ndarray]) -> None:
    to_drop = [
        k
        for k in list(arrays.keys())
        if k.startswith(_CHECKPOINT_PREFETCH_BCHW_PREFIX)
        or k.startswith(_CHECKPOINT_PREFETCH_CHW_PREFIX)
    ]
    for k in to_drop:
        arrays.pop(k, None)

def store_prefetch_checkpoint_arrays(
    *,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    sensor_by_key: dict[str, SensorSpec],
    inputs_cache: dict[tuple[int, str], np.ndarray],
    n_items: int,
) -> None:
    drop_prefetch_checkpoint_arrays(arrays)
    prefetch_meta: dict[str, Any] = {}
    for skey in sorted(sensor_by_key.keys()):
        hit_items = [
            (i, inputs_cache[(i, skey)]) for i in range(n_items) if (i, skey) in inputs_cache
        ]
        if not hit_items:
            continue
        entry: dict[str, Any] = {"sensor": _jsonable(sensor_by_key[skey])}

        def _store_per_item(items: list[tuple[int, np.ndarray]]) -> None:
            keys: list[str] = []
            indices: list[int] = []
            for i, x in items:
                key = f"{_CHECKPOINT_PREFETCH_CHW_PREFIX}{skey}__{i:05d}"
                arrays[key] = np.asarray(x, dtype=np.float32)
                keys.append(key)
                indices.append(int(i))
            entry["npz_keys"] = keys
            entry["indices"] = indices

        if len(hit_items) == n_items:
            key = f"{_CHECKPOINT_PREFETCH_BCHW_PREFIX}{skey}"
            try:
                arr = np.stack([np.asarray(x, dtype=np.float32) for _, x in hit_items], axis=0)
                arrays[key] = arr
                entry["npz_key"] = key
                entry["shape"] = list(arr.shape)
            except Exception as _e:
                # Some providers can return variable H/W across points.
                # Keep checkpointing by storing per-item CHW arrays instead.
                _store_per_item(hit_items)
        else:
            _store_per_item(hit_items)
        prefetch_meta[skey] = entry
    manifest["prefetch"] = prefetch_meta

def restore_prefetch_checkpoint_cache(
    *,
    arrays: dict[str, np.ndarray],
    prefetch_meta: dict[str, Any],
) -> dict[tuple[int, str], np.ndarray]:
    cache: dict[tuple[int, str], np.ndarray] = {}
    for skey, entry in prefetch_meta.items():
        if not isinstance(entry, dict):
            continue
        one_key = entry.get("npz_key")
        if isinstance(one_key, str) and one_key in arrays:
            arr = np.asarray(arrays[one_key])
            if arr.ndim >= 4:
                for i in range(arr.shape[0]):
                    cache[(int(i), str(skey))] = np.asarray(arr[i], dtype=np.float32)
            continue

        keys = entry.get("npz_keys")
        indices = entry.get("indices")
        if isinstance(keys, list) and isinstance(indices, list):
            for j, key in enumerate(keys):
                if not isinstance(key, str) or key not in arrays:
                    continue
                try:
                    idx = int(indices[j]) if j < len(indices) else int(j)
                except Exception as _e:
                    idx = int(j)
                cache[(idx, str(skey))] = np.asarray(arrays[key], dtype=np.float32)
    return cache

def drop_model_arrays(arrays: dict[str, np.ndarray], model_name: str, *, sanitize_key) -> None:
    mkey = sanitize_key(model_name)
    prefixes = (
        f"embeddings__{mkey}",
        f"embedding__{mkey}",
        f"inputs_bchw__{mkey}",
        f"input_chw__{mkey}",
    )
    for key in list(arrays.keys()):
        if key.startswith(prefixes):
            arrays.pop(key, None)
