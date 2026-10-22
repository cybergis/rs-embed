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

Every centered crop and pad here places the input with :func:`center_offset`,
so callers that need to know where the input landed (e.g. THOR's ``native_snap``
via :func:`center_fit`) can compute it with the same convention.
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
    "center_offset",
    "center_fit",
    "center_crop_to_square",
    "center_pad_to_square",
    "resize_square",
    "prepare_square",
    "roi_is_full",
    "roi_in_frame",
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


def center_offset(inner: int, outer: int) -> int:
    """Start of a centered length-*inner* span within length *outer*.

    Negative when *inner* is longer (the span is cropped). The odd pixel of a
    pad or crop always goes after the span.
    """
    return (outer - inner) // 2 if outer >= inner else -((inner - outer) // 2)


def center_fit(
    x: np.ndarray,
    *,
    out_h: int,
    out_w: int,
    pad_mode: str = "edge",
    fill_value: float = 0.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Center-crop or center-pad the last two axes to ``(out_h, out_w)``.

    Each axis is cropped where it is longer and padded (``np.pad`` *pad_mode*;
    ``'constant'`` uses *fill_value*) where it is shorter, placed with
    :func:`center_offset`. Returns ``(array, window)``: *window* is where the
    input lies in the result, as fractions ``(y0, y1, x0, x1)`` that extend
    past ``[0, 1]`` on a cropped axis (see :func:`roi_in_frame`). The array is
    always C-contiguous (``x`` itself when it already matches and is).
    """
    h, w = _hw(x)
    out_h, out_w = int(out_h), int(out_w)
    if out_h < 0 or out_w < 0:
        raise ModelError(f"center_fit target must be non-negative, got {(out_h, out_w)}.")
    top, left = center_offset(h, out_h), center_offset(w, out_w)
    window = (
        top / max(out_h, 1),
        (top + h) / max(out_h, 1),
        left / max(out_w, 1),
        (left + w) / max(out_w, 1),
    )
    if (h, w) == (out_h, out_w):
        return np.ascontiguousarray(x), window
    kept_h, kept_w = min(h, out_h), min(w, out_w)
    if (kept_h != out_h or kept_w != out_w) and (h == 0 or w == 0) and pad_mode != "constant":
        raise ModelError(f"center_fit cannot {pad_mode!r}-pad an empty input of shape {x.shape}.")
    y0, x0 = max(0, -top), max(0, -left)
    x = x[..., y0 : y0 + kept_h, x0 : x0 + kept_w]
    if kept_h != out_h or kept_w != out_w:
        pad_top, pad_left = max(0, top), max(0, left)
        pad_width = [(0, 0)] * (x.ndim - 2) + [
            (pad_top, out_h - kept_h - pad_top),
            (pad_left, out_w - kept_w - pad_left),
        ]
        if pad_mode == "constant":
            x = np.pad(x, pad_width, mode="constant", constant_values=float(fill_value))
        else:
            x = np.pad(x, pad_width, mode=pad_mode)
    return np.ascontiguousarray(x), window


def center_crop_to_square(x: np.ndarray) -> np.ndarray:
    """Center-crop the last two axes (``..., H, W``) to ``side = min(H, W)``."""
    side = min(_hw(x))
    return center_fit(x, out_h=side, out_w=side)[0]


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
    mode = str(pad_mode).lower()
    if mode == "reflect" and (side - h >= h or side - w >= w):
        # reflect needs pad < dim along each axis; replicate the edge instead.
        mode = "edge"
    return center_fit(x, out_h=side, out_w=side, pad_mode=mode, fill_value=fill_value)[0]


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
        top, left = center_offset(h, s), center_offset(w, s)
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


def roi_in_frame(
    frame: tuple[float, float, float, float],
    roi_window: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Map *roi_window*, given in an input's frame, into the frame of the array fed to the model.

    *frame* is where the input lies in that array, as fractions ``(y0, y1, x0,
    x1)`` of the array; it extends past ``[0, 1]`` on axes where the input was
    cropped, and lies inside on axes where it was padded. The result is clipped
    to the array, so it is the window to crop the model's token grid to.
    """
    fy0, fy1, fx0, fx1 = (float(v) for v in frame)
    y0, y1, x0, x1 = (float(v) for v in roi_window)

    def _clip(v: float) -> float:
        return min(1.0, max(0.0, v))

    return (
        _clip(fy0 + y0 * (fy1 - fy0)),
        _clip(fy0 + y1 * (fy1 - fy0)),
        _clip(fx0 + x0 * (fx1 - fx0)),
        _clip(fx0 + x1 * (fx1 - fx0)),
    )


def roi_token_box(
    roi_window: tuple[float, float, float, float], *, grid_h: int, grid_w: int
) -> tuple[int, int, int, int]:
    """Map a normalized ROI window to integer ``(y0, y1, x0, x1)`` token indices.

    Rounds outward (floor low edge, ceil high edge) so the ROI is never
    under-covered, and always returns at least a 1×1 box clamped to the grid.
    An edge within float round-off (1e-6 of a token) of a token boundary counts
    as on it, so a window computed from pixel fractions does not pull in a whole
    extra token.
    """
    eps = 1e-6
    y0f, y1f, x0f, x1f = roi_window
    y0 = max(0, min(int(np.floor(y0f * grid_h + eps)), grid_h - 1))
    y1 = max(y0 + 1, min(int(np.ceil(y1f * grid_h - eps)), grid_h))
    x0 = max(0, min(int(np.floor(x0f * grid_w + eps)), grid_w - 1))
    x1 = max(x0 + 1, min(int(np.ceil(x1f * grid_w - eps)), grid_w))
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


def square_fetch_request(spatial: SpatialSpec) -> tuple[SpatialSpec, dict[str, Any]]:
    """The generic fetch-square request for *spatial*.

    Returns the ROI enlarged to a square of real imagery plus the crop-back
    window in ``fetch_meta`` form (empty when the ROI is already square).
    Every generic-fetch fallback must use this — the model-specific
    ``fetch_input`` path squares by default, and the same spatial must fetch
    the same geometry regardless of which path serves it.
    """
    sq, geo_roi = square_spatial(spatial)
    return sq, (roi_fetch_meta(geo_roi) or {})


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


def parallel_indexed_fetch(
    n: int,
    fetch_one: Callable[[int], Any],
    *,
    max_workers: int = 1,
) -> list[Any]:
    """Call ``fetch_one(i)`` for ``i in range(n)`` and return results ordered by i.

    Sequential when ``max_workers <= 1``; otherwise a ``ThreadPoolExecutor``
    slots each result back into its submitted index. Exceptions raised by
    ``fetch_one`` propagate (via ``fut.result()`` on the threaded path).

    Collapses the per-model "sequential-or-threaded fan-out with index slotting"
    boilerplate that several embedders' batch fetch loops repeat.
    """
    results: list[Any] = [None] * n
    if max_workers <= 1:
        for i in range(n):
            results[i] = fetch_one(i)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_one, i): i for i in range(n)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
    return results


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

    def _one(i: int) -> tuple[np.ndarray | None, tuple[float, float, float, float]]:
        sq, geo_roi = square_spatial(spatials[i])
        return fetch_fn(sq), geo_roi

    results = parallel_indexed_fetch(len(spatials), _one, max_workers=max_workers)
    raws = [r for r, _ in results]
    geo_rois = [g for _, g in results]

    missing = [i for i, r in enumerate(raws) if r is None]
    if missing:
        raise ModelError(f"square_fetch_batch: fetch_fn returned None at indices {missing}.")
    return raws, geo_rois
