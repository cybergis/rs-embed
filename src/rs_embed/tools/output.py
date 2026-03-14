from __future__ import annotations

from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.specs import OutputSpec


def _infer_native_y_axis_direction(meta: dict[str, Any]) -> tuple[str, str]:
    """Infer native y-axis direction from metadata.

    Returns:
      (direction, reason)
      direction in {"north_to_south", "south_to_north", "unknown"}.
    """
    if not isinstance(meta, dict):
        return "unknown", "meta is not a dict"

    # Geospatial affine convention: north-up rasters usually have e < 0.
    transform = meta.get("global_transform") or meta.get("transform")
    if transform is not None and hasattr(transform, "e"):
        try:
            e = float(transform.e)
            if e < 0:
                return "north_to_south", f"transform.e={e:.6g} (<0)"
            if e > 0:
                return "south_to_north", f"transform.e={e:.6g} (>0)"
        except Exception as _e:
            pass

    y_axis = str(meta.get("y_axis_direction", "")).strip().lower()
    if y_axis in {"north_to_south", "top_to_bottom"}:
        return "north_to_south", f"y_axis_direction={y_axis}"
    if y_axis in {"south_to_north", "bottom_to_top"}:
        return "south_to_north", f"y_axis_direction={y_axis}"

    return "unknown", "no orientation metadata"

def _flip_data_y(data: Any) -> tuple[Any, bool, str]:
    # xarray.DataArray path (no hard dependency import; duck typing only).
    if hasattr(data, "dims") and hasattr(data, "isel"):
        dims = tuple(str(d) for d in getattr(data, "dims", ()))
        if "y" in dims:
            return data.isel(y=slice(None, None, -1)), True, "xarray isel(y=reverse)"
        if len(dims) >= 2:
            dim = dims[-2]
            try:
                return (
                    data.isel({dim: slice(None, None, -1)}),
                    True,
                    f"xarray isel({dim}=reverse)",
                )
            except Exception as _e:
                pass

    arr = np.asarray(getattr(data, "values", data))
    if arr.ndim < 2:
        return data, False, f"ndim={arr.ndim} (<2)"

    axis = arr.ndim - 2
    return np.flip(arr, axis=axis), True, f"numpy flip axis={axis}"

def normalize_embedding_output(*, emb: Embedding, output: OutputSpec) -> Embedding:
    """Normalize embedding outputs according to OutputSpec-level conventions."""
    if output.mode != "grid":
        return emb

    policy = str(getattr(output, "grid_orientation", "north_up")).strip().lower()
    meta = dict(getattr(emb, "meta", {}) or {})
    native_dir, reason = _infer_native_y_axis_direction(meta)

    meta["grid_orientation_policy"] = policy
    meta["grid_native_y_axis_direction"] = native_dir
    meta["grid_native_orientation_reason"] = reason

    if policy == "native":
        if native_dir != "unknown":
            meta["y_axis_direction"] = native_dir
        meta["grid_orientation_applied"] = False
        return Embedding(data=emb.data, meta=meta)

    if policy != "north_up":
        # Keep data unchanged for unknown future policies.
        meta["grid_orientation_applied"] = False
        return Embedding(data=emb.data, meta=meta)

    # north_up policy: ensure y increases southward in array row order.
    if native_dir == "south_to_north":
        new_data, flipped, how = _flip_data_y(emb.data)
        meta["grid_orientation_applied"] = bool(flipped)
        meta["grid_orientation_transform"] = "flip_y" if flipped else "none"
        meta["grid_orientation_transform_detail"] = how
        if flipped:
            meta["y_axis_direction"] = "north_to_south"
            return Embedding(data=new_data, meta=meta)
        return Embedding(data=emb.data, meta=meta)

    meta["grid_orientation_applied"] = False
    if native_dir == "north_to_south":
        meta["y_axis_direction"] = "north_to_south"
    else:
        meta.setdefault("y_axis_direction", "unknown")
    return Embedding(data=emb.data, meta=meta)
