"""Unified spatial-shape preparation for square-input models.

Many encoders (ViT-style foundation models) require a **square** token grid:
their 2-D positional encodings are generated from a single ``grid_size`` scalar,
so a non-square ``H != W`` input either crashes or is silently stretched. The
historical behaviour across the on-the-fly embedders was to call
``F.interpolate(size=(image_size, image_size))`` directly, which **stretches** a
rectangular ROI to a square — distorting geometry (e.g. a 1.8:1 field becomes a
mess of horizontal stripes in the embedding) and pushing tiny ROIs far out of
the model's training distribution.

This module centralises a single, reusable strategy:

1.  **pad / crop to square first** (no aspect-ratio distortion), then
2.  resize the square to the model's required ``size``.

When the input is *extremely* rectangular (aspect ratio beyond ``aspect_tol``),
padding injects a lot of synthetic border; ``prepare_square`` warns (a square
fetch of real imagery or tiling serves such ROIs better) but still pads, so the
output can always be cropped back to the real ROI — it never silently stretches.

All helpers operate on the **last two axes** (``..., H, W``) so the same code
serves ``CHW`` and ``TCHW`` (and any leading-dim) layouts identically.

Mirrors the proven crop/pad-to-square logic in the THOR embedder; THOR keeps its
own multi-GSD ``native_snap`` path and is intentionally left untouched.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..core.errors import ModelError
from ..core.specs import SpatialSpec
from .spatial import FULL_WINDOW, square_spatial

__all__ = [
    "center_crop_to_square",
    "center_pad_to_square",
    "resize_square",
    "prepare_square",
    "roi_is_full",
    "roi_token_box",
    "crop_grid_to_roi",
    "geo_roi_from_meta",
    "roi_fetch_meta",
    "crop_grid_and_pool",
    "square_fetch_batch",
]

# Default cap on how rectangular an input may be before we stop trying to
# pad/crop it square and fall back to a plain stretch. 2.0 means "the long side
# is at most twice the short side"; beyond that, square-ing distorts the result
# more than the stretch it is meant to avoid. Also keeps reflect-padding valid
# (reflect requires pad < short-side, which holds while aspect < 2).
_DEFAULT_ASPECT_TOL = 2.0


def _hw(x: np.ndarray) -> tuple[int, int]:
    if x.ndim < 2:
        raise ModelError(f"shape prep expects an array with H,W axes, got ndim={x.ndim}.")
    return int(x.shape[-2]), int(x.shape[-1])


def center_crop_to_square(x: np.ndarray) -> np.ndarray:
    """Center-crop the last two axes (``..., H, W``) to ``side = min(H, W)``."""
    h, w = _hw(x)
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return np.ascontiguousarray(x[..., y0 : y0 + side, x0 : x0 + side])


def center_pad_to_square(
    x: np.ndarray,
    *,
    fill_value: float = 0.0,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Center-pad the last two axes to ``side = max(H, W)``.

    ``pad_mode='reflect'`` mirrors real edge content (no fake nodata border) and
    is preferred; it requires the pad amount to be smaller than the short side,
    so when that does not hold we fall back to ``'edge'`` (replicate) padding.
    ``pad_mode='constant'`` pads with ``fill_value``.
    """
    h, w = _hw(x)
    side = max(h, w)
    if side == min(h, w):
        return np.ascontiguousarray(x)
    pad_h, pad_w = side - h, side - w
    top, left = pad_h // 2, pad_w // 2
    pad_width = [(0, 0)] * (x.ndim - 2) + [(top, pad_h - top), (left, pad_w - left)]

    mode = str(pad_mode).lower()
    if mode == "reflect" and (pad_h >= h or pad_w >= w):
        # reflect needs pad < dim along each axis; replicate the edge instead.
        mode = "edge"
    if mode == "constant":
        return np.ascontiguousarray(
            np.pad(x, pad_width, mode="constant", constant_values=float(fill_value))
        )
    return np.ascontiguousarray(np.pad(x, pad_width, mode=mode))


def resize_square(x: np.ndarray, *, size: int, interp: str = "bilinear") -> np.ndarray:
    """Resize the last two axes to ``(size, size)``, preserving all leading dims.

    Each 2-D plane is interpolated independently, so this is layout-agnostic
    (CHW, TCHW, …). ``interp`` is a ``torch.nn.functional.interpolate`` mode.
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - torch is a hard model dep
        raise ModelError("shape resize requires torch. Install: pip install torch") from exc

    if size <= 0:
        raise ModelError(f"resize_square size must be > 0, got {size}.")
    h, w = _hw(x)
    if h == size and w == size:
        return np.ascontiguousarray(x.astype(np.float32, copy=False))
    lead = x.shape[:-2]
    flat = x.reshape(-1, 1, h, w).astype(np.float32, copy=False)
    t = torch.from_numpy(flat)
    align = None if interp in ("nearest", "area") else False
    kwargs: dict[str, Any] = {"size": (int(size), int(size)), "mode": interp}
    if align is not None:
        kwargs["align_corners"] = align
    out = F.interpolate(t, **kwargs)
    return out.numpy().astype(np.float32).reshape(*lead, size, size)


def prepare_square(
    x: np.ndarray,
    *,
    size: int,
    shape_adjust: str = "pad",
    aspect_tol: float = _DEFAULT_ASPECT_TOL,
    fill_value: float = 0.0,
    pad_mode: str = "reflect",
    interp: str = "bilinear",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Make ``x`` a square ``(..., size, size)`` array without aspect distortion.

    Strategy (the unified rule shared by all square-input embedders):

    - already square → just resize to ``size``;
    - otherwise → **pad** (default, keeps the whole ROI) or **crop** to square,
      *then* resize to ``size`` — never a plain stretch, so the output grid /
      ROI-pooled vector can always be cropped back to the real ROI via
      ``meta['shape_prep']['roi_window']``. Aspect ratios ≥ ``aspect_tol`` warn
      that the padded square is dominated by synthetic border (a square *fetch*
      of real imagery, or tiling, serves such ROIs better) but still pad —
      distorting silently is worse than padding loudly.

    Returns ``(out, meta)`` where ``meta['shape_prep']`` records what happened so
    callers can surface it in the embedding metadata.
    """
    adj = str(shape_adjust).lower()
    if adj not in ("pad", "crop"):
        raise ModelError(f"shape_adjust must be 'pad' or 'crop', got {shape_adjust!r}.")

    h, w = _hw(x)
    long_side, short_side = max(h, w), min(h, w)
    aspect = float(long_side) / float(short_side) if short_side > 0 else float("inf")

    meta: dict[str, Any] = {
        "orig_hw": (h, w),
        "aspect": round(aspect, 4),
        "shape_adjust": adj,
        "target_hw": (int(size), int(size)),
    }

    if h == w:
        applied = "none"
        sq = x
    elif adj == "crop":
        applied = "crop_to_square"
        sq = center_crop_to_square(x)
    else:
        if aspect >= float(aspect_tol):
            warnings.warn(
                f"prepare_square: input aspect ratio {aspect:.2f} ≥ {float(aspect_tol):.2f} — "
                "padding to square leaves most of the model input synthetic border. "
                "Prefer a square fetch of real imagery (rectangular ROIs are enlarged "
                "automatically on provider-backed paths) or input_prep='tile'.",
                stacklevel=2,
            )
        applied = "pad_to_square"
        sq = center_pad_to_square(x, fill_value=fill_value, pad_mode=pad_mode)

    meta["applied"] = applied
    meta["square_hw"] = tuple(int(v) for v in _hw(sq))
    # ROI window within the square, normalized to [0, 1] as (y0, y1, x0, x1).
    # Only padding inserts non-ROI border; crop/none cover the whole
    # square, so their window is the full frame. Downstream code uses this to crop
    # the output token grid (and pooled tokens) back to the real ROI.
    if applied == "pad_to_square":
        s = max(h, w)
        top, left = (s - h) // 2, (s - w) // 2
        roi = (top / s, (top + h) / s, left / s, (left + w) / s)
    else:
        roi = (0.0, 1.0, 0.0, 1.0)
    meta["roi_window"] = tuple(round(float(v), 6) for v in roi)
    out = resize_square(sq, size=int(size), interp=interp)
    return out, {"shape_prep": meta}


def roi_is_full(roi_window: tuple[float, float, float, float], *, eps: float = 1e-6) -> bool:
    """True if the ROI window covers the whole frame (no cropping needed)."""
    y0, y1, x0, x1 = roi_window
    return y0 <= eps and y1 >= 1.0 - eps and x0 <= eps and x1 >= 1.0 - eps


def roi_token_box(
    roi_window: tuple[float, float, float, float], *, grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    """Map a normalized ROI window to integer ``(y0, y1, x0, x1)`` token indices.

    Rounds outward (floor low edge, ceil high edge) so the ROI is never
    under-covered, and always returns at least a 1×1 box clamped to the grid.
    """
    y0f, y1f, x0f, x1f = roi_window
    y0 = max(0, min(int(np.floor(y0f * grid_h)), grid_h - 1))
    y1 = max(y0 + 1, min(int(np.ceil(y1f * grid_h)), grid_h))
    x0 = max(0, min(int(np.floor(x0f * grid_w)), grid_w - 1))
    x1 = max(x0 + 1, min(int(np.ceil(x1f * grid_w)), grid_w))
    return y0, y1, x0, x1


def crop_grid_to_roi(
    grid_dhw: np.ndarray, roi_window: tuple[float, float, float, float]
) -> np.ndarray:
    """Crop a ``(D, H', W')`` token grid back to the ROI window (no-op if full)."""
    if grid_dhw.ndim != 3:
        raise ModelError(f"crop_grid_to_roi expects (D, H, W), got {grid_dhw.shape}.")
    if roi_is_full(roi_window):
        return grid_dhw
    _, h, w = grid_dhw.shape
    y0, y1, x0, x1 = roi_token_box(roi_window, grid_h=h, grid_w=w)
    return np.ascontiguousarray(grid_dhw[:, y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# Fetch-square orchestration helpers (shared by square-input embedders)
# ---------------------------------------------------------------------------


def geo_roi_from_meta(
    fetch_meta: dict[str, Any] | None,
) -> tuple[float, float, float, float]:
    """The fetch-square ROI window carried in ``fetch_meta``, else the full frame.

    The input_chw / API-prefetch path hands the fetched square back as input_chw
    with its ROI window in ``fetch_meta['roi_window_geo']``; this reads it back so
    the output can be cropped to the ROI. Returns :data:`FULL_WINDOW` when absent.
    """
    rw = (fetch_meta or {}).get("roi_window_geo")
    if rw is None or len(tuple(rw)) != 4:
        return FULL_WINDOW
    return tuple(float(v) for v in rw)  # type: ignore[return-value]


def roi_fetch_meta(
    geo_roi: tuple[float, float, float, float],
) -> dict[str, Any] | None:
    """Wrap a geo ROI as ``fetch_meta`` for re-feeding ``get_embedding`` (the batch
    path), or ``None`` when the ROI is the full frame (nothing to crop)."""
    return None if roi_is_full(geo_roi) else {"roi_window_geo": geo_roi}


def crop_grid_and_pool(
    grid: np.ndarray,
    geo_roi: tuple[float, float, float, float],
    *,
    pooling: str = "mean",
    pooled_fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Crop a ``(D, H', W')`` token grid to the ROI and pool it.

    Returns ``(grid_out, pooled_vec)``:

    - ROI is a sub-window (a rectangle was enlarged to a square fetch) → crop the
      grid and pool **only the ROI tokens** (mean/max over ``H', W'``);
    - ROI is full → return the grid unchanged and ``pooled_fallback`` (the model's
      own pooled vector, e.g. CLS or global token pool), preserving its semantics.

    Collapses the per-model "if cropped: crop + pool ROI else fallback" blocks
    into one call. ``pooled_vec`` is ``None`` only when the ROI is full and no
    fallback was given (the caller is in grid mode and ignores it).
    """
    if roi_is_full(geo_roi):
        return grid, pooled_fallback
    g = crop_grid_to_roi(grid, geo_roi)
    vec = (g.max(axis=(1, 2)) if pooling == "max" else g.mean(axis=(1, 2))).astype(np.float32)
    return g, vec


def square_fetch_batch(
    spatials: list[SpatialSpec],
    fetch_fn: Callable[[SpatialSpec], np.ndarray],
    *,
    max_workers: int = 1,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    """Fetch each spatial as a square: ``square_spatial`` then ``fetch_fn(square)``.

    Returns ``(raws, geo_rois)`` index-aligned with ``spatials``. ``fetch_fn`` is a
    caller-supplied closure that captures the provider and fetch parameters and
    returns the raw array for a (squared) spatial — this helper never touches the
    provider, it only orchestrates squaring, parallelism, and ROI collection.

    Collapses the per-model ``_fetch_one`` square + geo_roi collection + thread
    pool boilerplate into one call. Use :func:`roi_fetch_meta` to turn each
    ``geo_rois[i]`` into the ``fetch_meta`` passed back into ``get_embedding``.
    """
    n = len(spatials)
    raws: list[np.ndarray | None] = [None] * n
    geo_rois: list[tuple[float, float, float, float]] = [FULL_WINDOW] * n

    def _one(i: int, sp: SpatialSpec) -> tuple[int, np.ndarray, tuple[float, float, float, float]]:
        sq, geo_roi = square_spatial(sp)
        return i, fetch_fn(sq), geo_roi

    if max_workers <= 1:
        for i, sp in enumerate(spatials):
            _, raws[i], geo_rois[i] = _one(i, sp)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for fut in as_completed([ex.submit(_one, i, sp) for i, sp in enumerate(spatials)]):
                i, raw, geo_roi = fut.result()
                raws[i], geo_rois[i] = raw, geo_roi

    missing = [i for i, r in enumerate(raws) if r is None]
    if missing:
        raise ModelError(f"square_fetch_batch: fetch_fn returned None at indices {missing}.")
    return [r for r in raws if r is not None], geo_rois
