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


def import_hf_hub():
    """Import huggingface_hub or raise ModelError with an install hint."""
    try:
        import huggingface_hub
    except Exception as e:
        raise ModelError(
            "Downloading model assets from Hugging Face requires huggingface_hub. "
            "Install: pip install huggingface_hub"
        ) from e
    return huggingface_hub


def hf_hub_download_cache_first(**kwargs: Any) -> str:
    """``hf_hub_download`` that tries the local cache before touching the network.

    A plain ``hf_hub_download`` issues a HEAD request to huggingface.co even on a
    warm cache (to resolve a branch revision to a commit), so every fresh process
    blocks on Hub availability, rate limits, and cache file locks. Trying
    ``local_files_only=True`` first makes warm-cache loads fully offline; the
    network path only runs on a cache miss, where any real error surfaces as
    before. Cached files are never re-checked against the Hub — delete the cached
    file (or pass ``force_download=True`` at a call site) to force a re-download.
    """
    hub = import_hf_hub()
    try:
        return str(hub.hf_hub_download(local_files_only=True, **kwargs))
    except Exception:
        return str(hub.hf_hub_download(local_files_only=False, **kwargs))


def snapshot_download_cache_first(
    *, validate: Any = None, local_files_only: bool = False, **kwargs: Any
) -> str:
    """``snapshot_download`` with the same cache-first behavior as ``hf_hub_download_cache_first``.

    With ``local_files_only=True`` the Hub returns a cached snapshot dir without
    verifying it is complete, so callers that need specific files pass
    ``validate`` (snap_dir -> bool); a snapshot failing validation falls back to
    the network download. ``local_files_only=True`` keeps the offline-only
    contract of the underlying call: cache miss raises instead of downloading.
    """
    hub = import_hf_hub()
    try:
        snap = str(hub.snapshot_download(local_files_only=True, **kwargs))
    except Exception:
        if local_files_only:
            raise
        snap = None
    if snap is not None and (validate is None or validate(snap)):
        return snap
    if local_files_only:
        raise ModelError(
            f"Cached Hugging Face snapshot is incomplete for {kwargs.get('repo_id')!r} "
            "and auto-download is disabled."
        )
    return str(hub.snapshot_download(local_files_only=False, **kwargs))


def resolve_pretrained_source_cache_first(model_id: str, *, weight_names: tuple = ()) -> str:
    """Map a HF repo id to its cached snapshot dir so ``from_pretrained`` stays offline.

    ``Mixin.from_pretrained(repo_id)`` re-resolves config and weights against the
    Hub on every fresh process even when cached. If a cached snapshot exists and
    holds ``config.json`` plus one of ``weight_names`` (default: the standard
    safetensors/bin names), return its path — ``from_pretrained`` accepts a local
    dir and skips the network entirely. On any miss return ``model_id`` unchanged
    so the caller keeps the exact previous online behavior.
    """
    if os.path.exists(model_id):
        return model_id
    try:
        hub = import_hf_hub()
        snap = str(hub.snapshot_download(repo_id=model_id, local_files_only=True))
    except Exception:
        return model_id
    names = weight_names or ("model.safetensors", "pytorch_model.bin")
    if os.path.isfile(os.path.join(snap, "config.json")) and any(
        os.path.isfile(os.path.join(snap, n)) for n in names
    ):
        return snap
    return model_id


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
