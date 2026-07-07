from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.specs import SensorSpec


def utc_ts() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat() + "Z"


def sanitize_key(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"


def sha1(arr: np.ndarray, max_bytes: int = 2_000_000) -> str:
    h = hashlib.sha1()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    b = arr.tobytes(order="C")
    if len(b) > max_bytes:
        b = b[:max_bytes]
    h.update(b)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "min": float(np.nanmin(obj)) if obj.size else None,
            "max": float(np.nanmax(obj)) if obj.size else None,
        }
    if is_dataclass(obj):
        return jsonable(asdict(obj))
    try:
        import xarray as xr

        if isinstance(obj, xr.DataArray):
            return {
                "__xarray__": True,
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "dims": list(obj.dims),
            }
    except Exception as _e:
        pass
    return repr(obj)


def embedding_to_numpy(emb: Embedding) -> np.ndarray:
    if isinstance(emb.data, np.ndarray):
        return emb.data.astype(np.float32, copy=False)
    try:
        import xarray as xr

        if isinstance(emb.data, xr.DataArray):
            return np.asarray(emb.data.values, dtype=np.float32)
    except Exception as _e:
        pass
    return np.asarray(emb.data, dtype=np.float32)


def sensor_identity_fields(sensor: SensorSpec, *, include_bands: bool = True) -> dict[str, Any]:
    """Fetch-affecting identity fields of a :class:`SensorSpec`.

    Single source of truth for every sensor-identity key: adding a
    fetch-affecting field to ``SensorSpec`` needs exactly one edit here to
    reach both :func:`sensor_cache_key` (per-sensor cache/manifest identity)
    and ``providers.prefetch_plan.sensor_fetch_group_key`` (band-superset
    grouping, which excludes ``bands``).  ``check_*`` fields are deliberately
    excluded — they cannot affect fetched pixels.
    """
    obj: dict[str, Any] = {
        "collection": sensor.collection,
        "bands": list(sensor.bands),
        "scale_m": int(sensor.scale_m),
        "cloudy_pct": int(sensor.cloudy_pct) if sensor.cloudy_pct is not None else None,
        "fill_value": float(sensor.fill_value),
        "composite": str(sensor.composite),
        "modality": getattr(sensor, "modality", None),
        "orbit": getattr(sensor, "orbit", None),
        "use_float_linear": bool(getattr(sensor, "use_float_linear", True)),
        "s1_require_iw": bool(getattr(sensor, "s1_require_iw", True)),
        "s1_relax_iw_on_empty": bool(getattr(sensor, "s1_relax_iw_on_empty", True)),
    }
    if not include_bands:
        del obj["bands"]
    return obj


def sensor_cache_key(sensor: SensorSpec) -> str:
    # NOTE: the hash is persisted (export manifests key input refs by it), so
    # the identity-field names/types above must stay stable across versions.
    obj = sensor_identity_fields(sensor)
    data = json.dumps(obj, sort_keys=True).encode("utf-8")
    return sanitize_key(hashlib.sha1(data).hexdigest()[:12])
