"""Format-specific writers for embedding export.

Each writer persists:
  - arrays   : dict[str, np.ndarray]  — named arrays (inputs / embeddings)
  - manifest : dict[str, Any]         — JSON-serializable metadata

Supported formats
-----------------
- **npz**    – ``numpy.savez_compressed`` + sidecar ``.json`` manifest.
- **netcdf** – CF-flavored NetCDF file with named dimensions + global attrs.
               Requires one of: ``netCDF4``, ``h5netcdf``, or ``scipy``.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# ── format → file extension mapping ────────────────────────────────

_FORMAT_EXT: dict[str, str] = {
    "npz": ".npz",
    "netcdf": ".nc",
}

SUPPORTED_FORMATS = tuple(_FORMAT_EXT.keys())

def get_extension(fmt: str) -> str:
    """Return the canonical file extension (including dot) for *fmt*."""
    try:
        return _FORMAT_EXT[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Supported: {SUPPORTED_FORMATS}")

# ── public dispatcher ──────────────────────────────────────────────

def write_arrays(
    *,
    fmt: str,
    out_path: str,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    save_manifest: bool,
) -> dict[str, Any]:
    """Persist *arrays* + *manifest* in the requested format.

    Returns an updated copy of *manifest* with format-specific path keys.
    """
    if fmt == "npz":
        return _write_npz(out_path, arrays, manifest, save_manifest)
    if fmt == "netcdf":
        return _write_netcdf(out_path, arrays, manifest, save_manifest)
    raise ValueError(f"Unknown format {fmt!r}. Supported: {SUPPORTED_FORMATS}")

# ── NPZ writer ────────────────────────────────────────────────────

def _write_npz(
    out_path: str,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    save_manifest: bool,
) -> dict[str, Any]:
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(out_path, **arrays)

    if save_manifest:
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = json_path

    manifest["npz_path"] = out_path
    manifest["npz_keys"] = sorted(arrays.keys())
    return manifest

# ── NetCDF writer ──────────────────────────────────────────────────

def _write_netcdf(
    out_path: str,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    save_manifest: bool,
) -> dict[str, Any]:
    import xarray as xr

    if not out_path.endswith(".nc"):
        out_path += ".nc"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build xarray Dataset with semantically-named dimensions.
    data_vars: dict[str, xr.DataArray] = {}
    dim_sizes: dict[str, int] = {}
    for key, arr in arrays.items():
        dims = _resolve_conflicting_dims(
            key=key,
            dims=_infer_dims(key, arr),
            shape=arr.shape,
            dim_sizes=dim_sizes,
        )
        data_vars[key] = xr.DataArray(data=arr, dims=dims)

    ds = xr.Dataset(data_vars)

    # Embed useful global attributes (CF-like).
    ds.attrs["Conventions"] = "CF-1.8"
    ds.attrs["history"] = "Created by rs-embed export_batch (format=netcdf)"
    for attr in ("created_at", "backend", "device"):
        val = manifest.get(attr)
        if val is not None:
            ds.attrs[attr] = str(val)
    if "n_items" in manifest:
        ds.attrs["n_items"] = int(manifest["n_items"])

    engine = _pick_engine()
    ds.to_netcdf(out_path, engine=engine)

    if save_manifest:
        json_path = os.path.splitext(out_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        manifest["manifest_path"] = json_path

    manifest["nc_path"] = out_path
    manifest["nc_variables"] = sorted(arrays.keys())
    return manifest

def _safe_dim_suffix(key: str) -> str:
    out = "".join((c if c.isalnum() else "_") for c in str(key))
    out = out.strip("_")
    return out or "var"

def _resolve_conflicting_dims(
    *,
    key: str,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    dim_sizes: dict[str, int],
) -> tuple[str, ...]:
    """Rename dims only when an existing dim name has a different size."""
    if len(dims) != len(shape):
        return dims

    resolved = []
    for name, size_raw in zip(dims, shape):
        size = int(size_raw)
        existing = dim_sizes.get(name)
        resolved_name = name
        if existing is not None and existing != size:
            suffix = _safe_dim_suffix(key)
            resolved_name = f"{name}__{suffix}"
            idx = 2
            while resolved_name in dim_sizes and dim_sizes[resolved_name] != size:
                resolved_name = f"{name}__{suffix}_{idx}"
                idx += 1
        dim_sizes.setdefault(resolved_name, size)
        resolved.append(resolved_name)
    return tuple(resolved)

# ── dimension inference ────────────────────────────────────────────

def _infer_dims(key: str, arr: np.ndarray) -> tuple[str, ...]:
    """Map array key + shape to semantically named NetCDF dimensions.

    Convention used by the NPZ export:
        input_chw__<model>            → (band, y, x)
        inputs_bchw__<model>          → (point, band, y, x)
        embedding__<model>            → (dim,)          for pooled
        embedding__<model>            → (band, y, x)    for grid
        embeddings__<model>           → (point, dim)    for pooled batch
        embeddings__<model>           → (point, band, y, x) for grid batch
    """
    ndim = arr.ndim

    if "bchw" in key:
        if ndim == 4:
            return ("point", "band", "y", "x")

    if "chw" in key:
        if ndim == 3:
            return ("band", "y", "x")

    if "embeddings" in key:
        if ndim == 2:
            return ("point", "dim")
        if ndim == 4:
            return ("point", "band", "y", "x")

    if "embedding" in key:
        if ndim == 1:
            return ("dim",)
        if ndim == 3:
            return ("band", "y", "x")

    # Fallback: generic numbered dimensions.
    return tuple(f"d{i}" for i in range(ndim))

# ── engine selection ───────────────────────────────────────────────

def _pick_engine() -> str:
    """Return the best available xarray NetCDF engine."""
    for engine, pkg in [
        ("netcdf4", "netCDF4"),
        ("h5netcdf", "h5netcdf"),
        ("scipy", "scipy"),
    ]:
        try:
            __import__(pkg)
            return engine
        except ImportError:
            continue
    raise ImportError(
        "No NetCDF engine available. Install one of: netCDF4, h5netcdf, or scipy.\n"
        "  pip install netCDF4     (recommended)\n"
        "  pip install h5netcdf\n"
        "  pip install scipy"
    )
