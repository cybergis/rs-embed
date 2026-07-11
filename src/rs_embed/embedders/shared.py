"""Shared numeric/output helpers for embedder implementations.

Verbatim-extracted from the on-the-fly embedders (M10-a): ViT token pooling
and grid reshaping, loaded-weight sanity stats, Hugging Face cache-dir
resolution, Sentinel-2 reflectance normalization, and xarray grid output
construction. Behavior is intentionally identical to the previous per-file
copies; model-specific wording is parameterized, never rewritten.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..core.errors import ModelError


def pool_from_tokens(tokens, pooling):
    """Pool ViT patch tokens [N,D] -> (vec [D], cls_removed). Excludes CLS if present."""
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    if len(patch) == 0:
        return tokens[0].astype("float32"), has_cls
    if pooling == "mean":
        return patch.mean(axis=0).astype("float32"), has_cls
    if pooling == "max":
        return patch.max(axis=0).astype("float32"), has_cls
    raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")


def tokens_to_grid_dhw(tokens):
    """Reshape ViT patch tokens [N,D] -> (grid [D,h,w], (h,w), cls_removed)."""
    n = len(tokens)
    h2 = int((n - 1) ** 0.5)
    has_cls = n > 1 and h2 * h2 == n - 1
    patch = tokens[1:] if has_cls else tokens
    p, d = patch.shape
    hw = int(p**0.5)
    if hw * hw != p:
        raise ModelError(f"Patch token count {p} is not a perfect square.")
    return patch.reshape(hw, hw, d).transpose(2, 0, 1).astype("float32"), (hw, hw), has_cls


def verify_loaded_params(
    model: Any,
    *,
    model_name: str,
    no_params_msg: str | None = None,
    nonfinite_msg: str | None = None,
    check_near_zero: bool = False,
) -> dict[str, float]:
    """Sanity stats over the first non-empty parameter of a freshly loaded model.

    Returns ``{"param_mean", "param_std", "param_absmax"}``. Raises ModelError when
    the model has no parameters, the parameter contains NaN/Inf, or (with
    ``check_near_zero``) the stats look uninitialized. Message overrides exist so
    each caller keeps its historical wording exactly.
    """
    import torch

    p0 = None
    for _, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            p0 = p.detach()
            break
    if p0 is None:
        raise ModelError(
            no_params_msg or f"{model_name} model has no parameters; cannot verify weights."
        )
    if not torch.isfinite(p0).all():
        raise ModelError(
            nonfinite_msg or f"{model_name} parameters contain NaN/Inf; load likely failed."
        )

    p0f = p0.float()
    stats = {
        "param_mean": float(p0f.mean().cpu()),
        "param_std": float(p0f.std().cpu()),
        "param_absmax": float(p0f.abs().max().cpu()),
    }
    if check_near_zero and stats["param_std"] < 1e-6 and stats["param_absmax"] < 1e-5:
        raise ModelError(f"{model_name} parameters look uninitialized (near-zero stats).")
    return stats


def resolve_hf_cache_dir() -> str | None:
    """Hugging Face cache dir from the env chain HUGGINGFACE_HUB_CACHE > HF_HOME > HUGGINGFACE_HOME."""
    return (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HOME")
    )


def normalize_s2(
    raw: np.ndarray,
    *,
    mode: str,
    model_name: str,
    modes_hint: str,
    allow_tchw: bool = False,
) -> np.ndarray:
    """Clip S2 SR values to [0, 10000] and apply unit_scale / per_tile_minmax / none.

    ``allow_tchw`` enables the TCHW input guard and per-frame minmax axes; without
    it the input is treated as CHW (minmax over the trailing two axes of a 3D array).
    """
    x = np.asarray(raw, dtype=np.float32)
    if allow_tchw and x.ndim not in {3, 4}:
        raise ModelError(f"{model_name} normalization expects CHW or TCHW, got {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 10000.0)

    m = str(mode).lower().strip()
    if m in {"unit", "unit_scale", "reflectance"}:
        x = x / 10000.0
    elif m in {"per_tile_minmax", "minmax", "tile_minmax"}:
        x = x / 10000.0
        if allow_tchw and x.ndim == 4:
            lo = np.min(x, axis=(2, 3), keepdims=True)
            hi = np.max(x, axis=(2, 3), keepdims=True)
        else:
            lo = np.min(x, axis=(1, 2), keepdims=True)
            hi = np.max(x, axis=(1, 2), keepdims=True)
        den = np.maximum(hi - lo, 1e-6)
        x = (x - lo) / den
    elif m in {"none", "raw"}:
        pass
    else:
        raise ModelError(
            f"Unknown {model_name} normalization mode '{mode}'. Use one of: {modes_hint}."
        )
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def import_xarray():
    """Import xarray lazily; grid output is the only path that needs it."""
    try:
        import xarray as xr
    except Exception as e:
        raise ModelError("grid output requires xarray. Install: pip install xarray") from e
    return xr


def grid_to_dataarray(grid: np.ndarray, *, meta: dict[str, Any], coords_d=None):
    """Wrap a [D,y,x] grid as the standard embedding DataArray (arange coords)."""
    xr = import_xarray()
    if coords_d is None:
        coords_d = np.arange(grid.shape[0])
    return xr.DataArray(
        grid,
        dims=("d", "y", "x"),
        coords={
            "d": coords_d,
            "y": np.arange(grid.shape[1]),
            "x": np.arange(grid.shape[2]),
        },
        name="embedding",
        attrs=meta,
    )
