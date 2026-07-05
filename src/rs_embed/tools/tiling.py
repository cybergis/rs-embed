"""Input-prep resolution, tiling, stitching, and tile-aware embedding dispatch.

Extracted from api.py to keep the public-API module focused on orchestration.
All symbols are private (underscore-prefixed) and re-exported by api.py where
backward-compatibility with test imports is needed.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.specs import (
    BBox,
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from .output import normalize_embedding_output as _normalize_embedding_output
from .runtime import (
    call_embedder_get_embedding as _call_embedder_get_embedding,
)
from .runtime import (
    embedder_accepts_input_chw as _embedder_accepts_input_chw,
)
from .runtime import (
    embedder_accepts_model_config as _embedder_accepts_model_config,
)
from .runtime import (
    supports_prefetched_batch_api as _supports_prefetched_batch_api,
)
from .serialization import embedding_to_numpy as _embedding_to_numpy
from .shape import geo_roi_from_meta, roi_is_full, roi_token_box

# ---------------------------------------------------------------------------
# Preprocessing contract version
# ---------------------------------------------------------------------------

# Bump whenever tiling/stitching/aggregation behavior changes in a way that
# alters produced embeddings. Stamped into every embedding's meta["input_prep"]
# so downstream consumers can detect a contract change and pin reproducibility.
INPUT_PREP_VERSION = 1


# ---------------------------------------------------------------------------
# Resolved input-prep spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedInputPrepSpec:
    mode: str
    tile_size: int | None
    tile_stride: int | None
    max_tiles: int
    max_tiles_hard: int
    pad_edges: bool
    tile_snap_frac: float


def _resolve_input_prep_spec(
    input_prep: InputPrepSpec | str | None,
) -> _ResolvedInputPrepSpec:
    if input_prep is None:
        # ``None`` is the "unset / use package default" sentinel. The package
        # default is tiling so large on-the-fly inputs preserve native
        # resolution instead of being downsampled by a resize.
        spec: InputPrepSpec = InputPrepSpec.tile()
    elif isinstance(input_prep, str):
        mode_s = str(input_prep).strip().lower()
        if mode_s == "auto":
            spec = InputPrepSpec.auto()
        elif mode_s == "resize":
            spec = InputPrepSpec.resize()
        elif mode_s == "tile":
            spec = InputPrepSpec.tile()
        else:
            raise ModelError(
                f"input_prep string must be 'auto'/'resize'/'tile', got {input_prep!r}"
            )
    else:
        spec = input_prep
    mode = str(getattr(spec, "mode", "auto")).strip().lower()
    if mode not in {"auto", "resize", "tile"}:
        raise ModelError(f"input_prep.mode must be one of auto/resize/tile, got {mode!r}")
    tile_size = getattr(spec, "tile_size", None)
    tile_stride = getattr(spec, "tile_stride", None)
    max_tiles = int(getattr(spec, "max_tiles", 64))
    max_tiles_hard = int(getattr(spec, "max_tiles_hard", 1024))
    pad_edges = bool(getattr(spec, "pad_edges", True))
    tile_snap_frac = float(getattr(spec, "tile_snap_frac", 0.1))
    env_snap = str(os.environ.get("RS_EMBED_TILE_SNAP_FRAC") or "").strip()
    if env_snap:
        try:
            tile_snap_frac = float(env_snap)
        except ValueError as e:
            raise ModelError(
                f"RS_EMBED_TILE_SNAP_FRAC must be a float in [0, 0.5], got {env_snap!r}."
            ) from e
    # Snapping more than half a tile would downscale an axis by >33%, which
    # distorts more than the overlap it removes; clamp to a safe band.
    tile_snap_frac = float(min(0.5, max(0.0, tile_snap_frac)))
    if tile_size is not None:
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ModelError(f"input_prep.tile_size must be > 0, got {tile_size}")
    if tile_stride is not None:
        tile_stride = int(tile_stride)
        if tile_stride <= 0:
            raise ModelError(f"input_prep.tile_stride must be > 0, got {tile_stride}")
    max_tiles = max(1, max_tiles)
    # The hard ceiling can never be below the soft threshold; clamp so an
    # explicit max_tiles override above max_tiles_hard stays consistent.
    max_tiles_hard = max(max_tiles, max_tiles_hard)
    return _ResolvedInputPrepSpec(
        mode=mode,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles=max_tiles,
        max_tiles_hard=max_tiles_hard,
        pad_edges=pad_edges,
        tile_snap_frac=tile_snap_frac,
    )


# ---------------------------------------------------------------------------
# Tile geometry helpers
# ---------------------------------------------------------------------------


def _embedder_default_image_size(embedder: Any) -> int | None:
    try:
        desc = embedder.describe()
    except Exception as _e:
        return None
    if not isinstance(desc, dict):
        return None
    defaults = desc.get("defaults")
    if not isinstance(defaults, dict):
        return None
    v = defaults.get("image_size")
    try:
        n = int(v)
        return n if n > 0 else None
    except Exception as _e:
        return None


def _estimate_tile_count(*, h: int, w: int, tile_size: int, stride: int) -> int:
    ny = 1 if h <= tile_size else int(math.ceil((float(h) - float(tile_size)) / float(stride))) + 1
    nx = 1 if w <= tile_size else int(math.ceil((float(w) - float(tile_size)) / float(stride))) + 1
    return max(1, ny) * max(1, nx)


def _input_hw(x: np.ndarray) -> tuple[int, int]:
    if x.ndim not in (3, 4):
        raise ModelError(
            f"Tiling currently supports CHW or TCHW inputs only, got shape={getattr(x, 'shape', None)}"
        )
    return int(x.shape[-2]), int(x.shape[-1])


def _snap_axis_to_tile(dim: int, *, tile_size: int, snap_frac: float) -> int:
    """Return the target length for one axis so it tiles without a degenerate overlap.

    With ``stride == tile_size`` the tiler keeps tiles non-overlapping except for
    the trailing tile, which is shifted back to cover the edge. That shift overlaps
    the previous tile by ``tile_size - (dim % tile_size)`` pixels, so a *small*
    overhang past a tile multiple produces a *large* overlap — two near-identical
    tiles whose stitched grid carries duplicated cells. When the overhang is within
    ``snap_frac * tile_size`` we drop it by downscaling the axis to the multiple
    (one fewer, fully-covered tile). Only downscaling is used: a large overhang
    already yields a small overlap, so it needs no fix and no fabricated pixels.
    """
    if tile_size <= 0 or snap_frac <= 0.0 or dim <= tile_size:
        return int(dim)
    rem = int(dim) % int(tile_size)
    if 0 < rem <= int(round(snap_frac * tile_size)):
        return int(dim) - rem
    return int(dim)


def _resize_spatial_hw(x: np.ndarray, *, out_h: int, out_w: int) -> np.ndarray:
    """Bilinearly resize the last two (spatial) dims of a CHW/TCHW array.

    Used to apply :func:`_snap_axis_to_tile` before tiling. Returns ``x`` unchanged
    if torch is unavailable so snapping degrades gracefully to native tiling.
    """
    h, w = _input_hw(x)
    if int(out_h) == h and int(out_w) == w:
        return x
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return x
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    squeeze = t.ndim == 3
    if squeeze:
        t = t.unsqueeze(0)  # CHW -> 1CHW; TCHW is already NCHW-shaped for interpolate
    y = F.interpolate(t, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
    if squeeze:
        y = y[0]
    return y.detach().cpu().numpy().astype(np.float32)


@dataclass(frozen=True)
class _TileParams:
    """Per-(embedder, spec) tiling parameters shared by single and batch paths."""

    tile_size: int
    stride: int
    model_fixed_size: bool
    effective_pad_edges: bool


def _resolve_tile_params(
    embedder: Any,
    input_prep: _ResolvedInputPrepSpec,
) -> _TileParams:
    """Resolve tile size/stride and the padding policy for an embedder + spec.

    ``tile_size`` is the explicit ``input_prep.tile_size`` or, failing that, the
    model's advertised ``describe().defaults.image_size``; ``0`` signals that no
    tile size could be determined and the caller should fall back to a plain
    call.

    Edge tiles are padded to square whenever ``pad_edges`` is on (the default),
    for every model. The tiling layer must never hand a model a rectangular
    tile: square-input embedders center-pad rectangles internally
    (``shape.prepare_square``), which the stitcher cannot see — its
    input→feature mapping would keep pad cells and misplace the valid region.
    Feeding only ``tile_size``-square tiles keeps the tile→grid geometry
    deterministic (fixed-size models plainly resize a square; the stitcher's
    ownership crop removes the pad cells). ``pad_edges=False`` is an expert
    escape hatch for models that natively handle rectangular inputs with a
    proportional output grid.
    """
    model_img = _embedder_default_image_size(embedder)
    tile_size = int(input_prep.tile_size or model_img or 0)
    stride = int(input_prep.tile_stride or tile_size)
    return _TileParams(
        tile_size=tile_size,
        stride=stride,
        model_fixed_size=model_img is not None and int(model_img) > 0,
        effective_pad_edges=bool(input_prep.pad_edges),
    )


def _maybe_snap_input(
    x: np.ndarray,
    *,
    tile_size: int,
    stride: int,
    snap_frac: float,
) -> tuple[np.ndarray, tuple[int, int] | None]:
    """Apply snap-to-tile downscaling to an input before tiling.

    Returns ``(x_maybe_resized, snapped_from_hw)``: ``snapped_from_hw`` is the
    original ``(h, w)`` when a snap happened, else ``None`` (no snap requested,
    axes already on a tile multiple, or torch unavailable so the resize was a
    no-op). Only applies under the ``stride == tile_size`` overlap-shift model;
    see :func:`_snap_axis_to_tile`.
    """
    h, w = _input_hw(x)
    if not (stride == tile_size and snap_frac > 0.0):
        return x, None
    snap_h = _snap_axis_to_tile(h, tile_size=tile_size, snap_frac=snap_frac)
    snap_w = _snap_axis_to_tile(w, tile_size=tile_size, snap_frac=snap_frac)
    if (snap_h, snap_w) == (h, w):
        return x, None
    x_snapped = _resize_spatial_hw(x, out_h=snap_h, out_w=snap_w)
    # Guard against a no-op when torch is unavailable (resize returns x).
    if _input_hw(x_snapped) == (h, w):
        return x, None
    return x_snapped, (int(h), int(w))


def _augment_model_config_for_tiled_dispatch(
    embedder: Any,
    model_config: dict[str, Any] | None,
    *,
    tile_size: int,
) -> dict[str, Any] | None:
    model_name = str(getattr(embedder, "model_name", "")).strip().lower()
    if model_name != "thor":
        return model_config
    cfg = dict(model_config or {})
    cfg["_input_prep_mode"] = "tile"
    cfg["_input_prep_tile_size"] = int(tile_size)
    return cfg


def _tile_yx_starts(*, h: int, w: int, tile_size: int, stride: int) -> tuple[list[int], list[int]]:
    def _starts_1d(dim: int) -> list[int]:
        if dim <= tile_size:
            return [0]
        starts: list[int] = [0]
        pos = 0
        while True:
            nxt = int(pos + stride)
            if (nxt + tile_size) >= dim:
                last = max(0, int(dim - tile_size))
                if last != starts[-1]:
                    starts.append(last)
                break
            starts.append(nxt)
            pos = nxt
        return starts

    return _starts_1d(int(h)), _starts_1d(int(w))


def _tile_subspatial(
    spatial: SpatialSpec,
    *,
    full_h: int,
    full_w: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> SpatialSpec:
    if isinstance(spatial, BBox) and full_h > 0 and full_w > 0:
        lon0 = float(spatial.minlon) + (float(x0) / float(full_w)) * (
            float(spatial.maxlon) - float(spatial.minlon)
        )
        lon1 = float(spatial.minlon) + (float(x1) / float(full_w)) * (
            float(spatial.maxlon) - float(spatial.minlon)
        )
        lat_top = float(spatial.maxlat) - (float(y0) / float(full_h)) * (
            float(spatial.maxlat) - float(spatial.minlat)
        )
        lat_bot = float(spatial.maxlat) - (float(y1) / float(full_h)) * (
            float(spatial.maxlat) - float(spatial.minlat)
        )
        return BBox(minlon=lon0, minlat=lat_bot, maxlon=lon1, maxlat=lat_top, crs=spatial.crs)
    return spatial


# ---------------------------------------------------------------------------
# Tile slicing / padding
# ---------------------------------------------------------------------------


def _slice_and_pad_tile(
    x: np.ndarray,
    *,
    y0: int,
    x0: int,
    tile_size: int,
    pad_edges: bool,
    fill_value: float,
) -> tuple[np.ndarray, dict[str, int]]:
    h, w = _input_hw(x)
    y1 = min(h, y0 + tile_size)
    x1 = min(w, x0 + tile_size)
    tile = np.asarray(x[..., y0:y1, x0:x1], dtype=np.float32)
    valid_h = int(y1 - y0)
    valid_w = int(x1 - x0)
    if pad_edges and (valid_h != tile_size or valid_w != tile_size):
        pad_spec = [(0, 0)] * tile.ndim
        pad_spec[-2] = (0, max(0, tile_size - valid_h))
        pad_spec[-1] = (0, max(0, tile_size - valid_w))
        # Replicate the edge rather than fill with a constant. A patch/ViT model
        # tokenizes the padded tile in fixed-size patches; when the valid region
        # does not end on a patch boundary, the straddling patch is computed over
        # valid + padded pixels. Constant (e.g. zero) padding drags that boundary
        # token toward an out-of-distribution "all fill" point, which surfaces as a
        # flat colored "dead band" along the short edge of the stitched grid.
        # Edge replication keeps the boundary patch on real surface values, so it
        # blends in; fully-padded patches beyond the valid fraction are cropped by
        # the stitcher regardless of pad content. ``fill_value`` is retained for the
        # degenerate case of a zero-extent tile, where ``mode="edge"`` is undefined.
        pad_mode = "edge" if (valid_h > 0 and valid_w > 0) else "constant"
        if pad_mode == "edge":
            tile = np.pad(tile, pad_spec, mode="edge")
        else:
            tile = np.pad(tile, pad_spec, mode="constant", constant_values=float(fill_value))
    return tile, {
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
        "valid_h": valid_h,
        "valid_w": valid_w,
    }


def _tile_one_image(
    x: np.ndarray,
    *,
    spatial: SpatialSpec,
    tile_size: int,
    stride: int,
    pad_edges: bool,
    fill_value: float,
) -> tuple[list[np.ndarray], list[dict[str, int]], list[SpatialSpec]]:
    """Slice one CHW/TCHW image into a row-major tile grid.

    Returns parallel lists ``(tiles, tile_meta, tile_spatials)``; each tile_meta
    carries ``row``/``col`` plus the slice bounds from :func:`_slice_and_pad_tile`,
    and each spatial is the sub-extent from :func:`_tile_subspatial`. Shared by
    the single-point and batch tiled dispatch paths so both slice identically.
    """
    h, w = _input_hw(x)
    ys, xs = _tile_yx_starts(h=h, w=w, tile_size=tile_size, stride=stride)
    tiles: list[np.ndarray] = []
    tile_meta: list[dict[str, int]] = []
    tile_spatials: list[SpatialSpec] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tile, meta = _slice_and_pad_tile(
                x,
                y0=int(y0),
                x0=int(x0),
                tile_size=tile_size,
                pad_edges=pad_edges,
                fill_value=fill_value,
            )
            meta["row"] = int(r)
            meta["col"] = int(c)
            tiles.append(tile)
            tile_meta.append(meta)
            tile_spatials.append(
                _tile_subspatial(
                    spatial,
                    full_h=h,
                    full_w=w,
                    y0=meta["y0"],
                    y1=meta["y1"],
                    x0=meta["x0"],
                    x1=meta["x1"],
                )
            )
    return tiles, tile_meta, tile_spatials


# ---------------------------------------------------------------------------
# Tile aggregation (stitching)
# ---------------------------------------------------------------------------


def _midpoint_owned_ranges(
    items: list[tuple[int, int, int]],
) -> dict[int, tuple[int, int]]:
    """Compute non-overlapping ownership intervals via midpoint cuts.

    items: [(id, start, end)] in input-pixel coordinates. Intervals must be ordered
    by `start` and cover the domain without gaps. Overlaps are allowed.
    """
    if not items:
        return {}
    items_s = sorted(items, key=lambda t: (int(t[1]), int(t[2]), int(t[0])))
    owned: dict[int, tuple[int, int]] = {}
    prev_cut: int | None = None
    for i, (idx, start, end) in enumerate(items_s):
        s = int(start)
        e = int(end)
        if e <= s:
            raise ModelError(f"Invalid tile interval [{s},{e}) for id={idx}.")
        if i == 0:
            own_s = s
        else:
            _, _pstart, pend = items_s[i - 1]
            if s > int(pend):
                raise ModelError("Tiled stitch found a gap between tiles; unsupported tile layout.")
            own_s = int((int(pend) + s) // 2)
            if prev_cut is not None:
                own_s = max(own_s, int(prev_cut))
        if i == len(items_s) - 1:
            own_e = e
        else:
            _, nstart, _ = items_s[i + 1]
            if int(nstart) > e:
                raise ModelError("Tiled stitch found a gap between tiles; unsupported tile layout.")
            own_e = int((e + int(nstart)) // 2)
        own_s = max(s, min(own_s, e))
        own_e = max(own_s, min(own_e, e))
        if own_e == own_s:
            own_e = min(e, own_s + 1)
        owned[int(idx)] = (int(own_s), int(own_e))
        prev_cut = own_e
    return owned


def _tile_axis_ownership(
    tile_meta: list[dict[str, int]],
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Midpoint-cut ownership intervals per row and per column of a tile grid.

    The single source of tile-ownership geometry: the grid stitcher crops each
    tile to its owned extent, and the pooled merge weights each tile by its
    owned area — so overlap regions (cover-shift edge tiles) are counted
    exactly once by both output modes.
    """
    row_items: list[tuple[int, int, int]] = []
    col_items: list[tuple[int, int, int]] = []
    seen_r: set[int] = set()
    seen_c: set[int] = set()
    for m in tile_meta:
        r, c = int(m["row"]), int(m["col"])
        if r not in seen_r:
            seen_r.add(r)
            row_items.append((r, int(m["y0"]), int(m["y1"])))
        if c not in seen_c:
            seen_c.add(c)
            col_items.append((c, int(m["x0"]), int(m["x1"])))
    return _midpoint_owned_ranges(row_items), _midpoint_owned_ranges(col_items)


def _axis_feature_crops(
    owned: dict[int, tuple[int, int]],
    ref: dict[int, dict[str, int]],
    *,
    span_of: Any,
    out_len: int,
) -> dict[int, tuple[int, int]]:
    """Map per-tile owned pixel ranges to feature-cell crops along one axis.

    Cuts are snapped to feature-cell edges *sequentially*: each tile's kept
    range starts where the previous tile's snapped crop actually ended (in
    input pixels), not at the raw midpoint. Rounding each tile independently
    (floor start / ceil end) made adjacent tiles both keep the cell straddling
    a midpoint cut — one duplicated grid row/column per unaligned seam, an
    inflated output grid, and a skewed px-per-cell mapping.
    """
    order = sorted(owned.keys(), key=lambda i: int(ref[i]["start"]))
    crops: dict[int, tuple[int, int]] = {}
    cut_px: float | None = None
    for pos, idx in enumerate(order):
        start = float(ref[idx]["start"])
        span = float(span_of(idx))
        n = int(out_len)
        if span <= 0 or n <= 0:
            crops[idx] = (0, 0)
            continue
        own_s, own_e = owned[idx]
        s_px = float(own_s) if cut_px is None else float(cut_px)
        fs = int(math.floor((s_px - start) / span * n))
        fs = max(0, min(n - 1, fs))
        f_end = (float(own_e) - start) / span * n
        if pos == len(order) - 1:
            fe = int(math.ceil(f_end))  # cover the trailing edge fully
        else:
            fe = int(math.floor(f_end + 0.5))  # snap the cut to the nearest edge
        fe = max(fs + 1, min(n, fe))
        cut_px = start + fe * span / n
        crops[idx] = (fs, fe)
    return crops


def _roi_owned_overlap_weights(
    owned_rects: list[tuple[int, int, int, int]],
    *,
    roi_window: tuple[float, float, float, float],
    input_hw: tuple[int, int],
) -> np.ndarray:
    """Per-tile area (in px²) of the tile's owned region inside the ROI.

    Owned (midpoint-cut) regions partition the input, so every ROI pixel is
    weighted exactly once; tiles fully outside the fetch-square ROI window get
    weight 0 so a pooled merge averages only the requested region.
    """
    ih, iw = int(input_hw[0]), int(input_hw[1])
    ry0, ry1 = roi_window[0] * ih, roi_window[1] * ih
    rx0, rx1 = roi_window[2] * iw, roi_window[3] * iw
    out = []
    for oy0, oy1, ox0, ox1 in owned_rects:
        oh = max(0.0, min(oy1, ry1) - max(oy0, ry0))
        ow = max(0.0, min(ox1, rx1) - max(ox0, rx0))
        out.append(oh * ow)
    return np.asarray(out, dtype=np.float32)


def _aggregate_tiled_embeddings(
    *,
    embs: list[Embedding],
    tile_meta: list[dict[str, int]],
    output: OutputSpec,
    tile_size: int,
    stride: int,
    prep_meta: dict[str, Any],
    roi_window: tuple[float, float, float, float] | None = None,
) -> Embedding:
    if not embs:
        raise ModelError("No tile embeddings produced.")
    base_meta = dict(getattr(embs[0], "meta", {}) or {})
    base_meta["input_prep"] = dict(prep_meta)
    # Fetch-square ROI: the tiled input covers an enlarged square; restrict the
    # merged output to the ROI window so only the requested region is returned.
    roi_crop = roi_window is not None and not roi_is_full(roi_window)

    # Shared ownership geometry: midpoint cuts partition the input among tiles,
    # so both output modes count every pixel exactly once (cover-shift edge
    # tiles overlap their neighbor; raw valid-area weights counted the overlap
    # twice and biased pooled vectors toward seam regions).
    row_owned, col_owned = _tile_axis_ownership(tile_meta)
    owned_rects = [
        (*row_owned[int(m["row"])], *col_owned[int(m["col"])]) for m in tile_meta
    ]

    if output.mode == "pooled":
        vecs = [np.asarray(_embedding_to_numpy(e), dtype=np.float32).reshape(-1) for e in embs]
        dims = {tuple(v.shape) for v in vecs}
        if len(dims) != 1:
            raise ModelError(
                f"Tiled pooled merge requires consistent vector shapes, got {sorted(dims)}"
            )
        mat = np.stack(vecs, axis=0)
        roi_w = (
            _roi_owned_overlap_weights(
                owned_rects, roi_window=roi_window, input_hw=prep_meta["input_hw"]
            )
            if roi_crop
            else None
        )
        if roi_w is not None and float(roi_w.sum()) <= 0.0:
            roi_w = None  # ROI smaller than a tile gap — fall back to all tiles
        if str(output.pooling).lower() == "max":
            sel = mat if roi_w is None else mat[roi_w > 0.0]
            out_vec = np.max(sel if len(sel) else mat, axis=0).astype(np.float32, copy=False)
        else:
            ws = np.asarray(
                [max(1, (oy1 - oy0)) * max(1, (ox1 - ox0)) for oy0, oy1, ox0, ox1 in owned_rects],
                dtype=np.float32,
            )
            if roi_w is not None:
                ws = roi_w  # weight by ROI-overlap area only
            out_vec = (mat * ws[:, None]).sum(axis=0) / max(1.0, float(ws.sum()))
            out_vec = out_vec.astype(np.float32, copy=False)
        base_meta["input_prep"]["merged_output"] = "pooled_reduce"
        if roi_crop:
            base_meta["input_prep"]["roi_cropped"] = True
        return Embedding(data=out_vec, meta=base_meta)

    arrays = [np.asarray(_embedding_to_numpy(e), dtype=np.float32) for e in embs]
    for a in arrays:
        if a.ndim < 2:
            raise ModelError(f"Tiled grid merge expects arrays with ndim>=2, got shape={a.shape}")
    gh = int(arrays[0].shape[-2])
    gw = int(arrays[0].shape[-1])
    lead_shape = tuple(int(v) for v in arrays[0].shape[:-2])
    for a in arrays[1:]:
        if (
            tuple(int(v) for v in a.shape[:-2]) != lead_shape
            or int(a.shape[-2]) != gh
            or int(a.shape[-1]) != gw
        ):
            raise ModelError("Tiled grid merge requires consistent per-tile output shapes.")

    nrows = max(int(m["row"]) for m in tile_meta) + 1
    ncols = max(int(m["col"]) for m in tile_meta) + 1

    row_ref: dict[int, dict[str, int]] = {}
    col_ref: dict[int, dict[str, int]] = {}
    for m in tile_meta:
        r = int(m["row"])
        c = int(m["col"])
        row_ref.setdefault(r, {"start": int(m["y0"]), "valid": int(m["valid_h"])})
        col_ref.setdefault(c, {"start": int(m["x0"]), "valid": int(m["valid_w"])})
    if len(row_ref) != nrows or len(col_ref) != ncols:
        raise ModelError("Missing row/col metadata for tiled stitch.")

    # When ``pad_edges`` is on, an edge tile shorter than ``tile_size`` is
    # bottom/right-padded to ``tile_size`` *before* inference, so the per-tile
    # output grid spans the full padded ``tile_size`` — only the leading
    # ``valid / tile_size`` fraction is real data. The input→feature mapping
    # must therefore measure feature cells against the padded extent
    # (``tile_size``), not against ``valid``; otherwise the padded region's grid
    # cells (constant "dead band" embeddings) are kept instead of cropped.
    pad_edges = bool(prep_meta.get("pad_edges", True))

    row_crop = _axis_feature_crops(
        row_owned,
        row_ref,
        span_of=lambda r: tile_size if pad_edges else row_ref[r]["valid"],
        out_len=gh,
    )
    col_crop = _axis_feature_crops(
        col_owned,
        col_ref,
        span_of=lambda c: tile_size if pad_edges else col_ref[c]["valid"],
        out_len=gw,
    )
    row_heights = [int(row_crop[r][1] - row_crop[r][0]) for r in range(nrows)]
    col_widths = [int(col_crop[c][1] - col_crop[c][0]) for c in range(ncols)]

    out_h = int(sum(row_heights))
    out_w = int(sum(col_widths))
    out_arr = np.zeros(lead_shape + (out_h, out_w), dtype=np.float32)
    row_offsets = [0] * nrows
    col_offsets = [0] * ncols
    for r in range(1, nrows):
        row_offsets[r] = row_offsets[r - 1] + row_heights[r - 1]
    for c in range(1, ncols):
        col_offsets[c] = col_offsets[c - 1] + col_widths[c - 1]

    for arr, m in zip(arrays, tile_meta, strict=False):
        r = int(m["row"])
        c = int(m["col"])
        fy0, fy1 = row_crop[r]
        fx0, fx1 = col_crop[c]
        crop_h = int(fy1 - fy0)
        crop_w = int(fx1 - fx0)
        y0 = row_offsets[r]
        x0 = col_offsets[c]
        out_arr[..., y0 : y0 + crop_h, x0 : x0 + crop_w] = arr[..., fy0:fy1, fx0:fx1]

    base_meta["input_prep"]["merged_output"] = "grid_stitch"
    base_meta["input_prep"]["stitched_grid_shape"] = (int(out_h), int(out_w))
    base_meta["input_prep"]["stitch_policy"] = "midpoint_cut"

    # Fetch-square crop-back: the stitched grid covers the enlarged square; slice
    # it to the ROI window so the output is exactly the requested (rectangular)
    # region at native tiled resolution.
    if roi_crop:
        y0c, y1c, x0c, x1c = roi_token_box(roi_window, grid_h=out_h, grid_w=out_w)
        out_arr = np.ascontiguousarray(out_arr[..., y0c:y1c, x0c:x1c])
        out_h, out_w = int(out_arr.shape[-2]), int(out_arr.shape[-1])
        base_meta["input_prep"]["roi_cropped"] = True
        base_meta["input_prep"]["roi_grid_shape"] = (out_h, out_w)

    # Keep model-reported grid metadata consistent with stitched output.
    if "grid_hw" in base_meta:
        base_meta["grid_hw"] = (int(out_h), int(out_w))
    return Embedding(data=out_arr, meta=base_meta)


# ---------------------------------------------------------------------------
# Tile-aware embedding dispatch
# ---------------------------------------------------------------------------


def _build_tile_prep_meta(
    *,
    requested_mode: str,
    tile_size: int,
    stride: int,
    tile_count: int,
    effective_pad_edges: bool,
    max_tiles: int,
    max_tiles_hard: int,
    input_hw: tuple[int, int],
    tile_snap_frac: float,
    snapped_from_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the ``input_prep`` meta block stamped onto a stitched embedding.

    Shared by the single-point and batch tiled dispatch paths so both stamp an
    identical, complete reproducibility block (same keys, same ``pad_policy``
    derivation, same snap bookkeeping).
    """
    prep_meta: dict[str, Any] = {
        "prep_version": INPUT_PREP_VERSION,
        "requested_mode": str(requested_mode),
        "resolved_mode": "tile",
        "tile_layout": "cover_shift",
        "tile_size": int(tile_size),
        "tile_stride": int(stride),
        "tile_count": int(tile_count),
        "pad_edges": bool(effective_pad_edges),
        "pad_policy": "edge_replicate" if effective_pad_edges else "none",
        "max_tiles": int(max_tiles),
        "max_tiles_hard": int(max_tiles_hard),
        "input_hw": (int(input_hw[0]), int(input_hw[1])),
        "tile_snap_frac": float(tile_snap_frac),
    }
    if snapped_from_hw is not None:
        prep_meta["snapped_from_hw"] = (int(snapped_from_hw[0]), int(snapped_from_hw[1]))
        prep_meta["snapped_to_hw"] = (int(input_hw[0]), int(input_hw[1]))
    return prep_meta


def _call_embedder_get_embedding_tiled(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: np.ndarray,
    input_prep: _ResolvedInputPrepSpec,
    model_config: dict[str, Any] | None = None,
    fetch_meta: dict[str, Any] | None = None,
) -> Embedding:
    x = np.asarray(input_chw, dtype=np.float32)
    params = _resolve_tile_params(embedder, input_prep)
    tile_size = params.tile_size
    model_fixed_size = params.model_fixed_size
    effective_pad_edges = params.effective_pad_edges
    if tile_size <= 0:
        # No tile size could be determined (no explicit tile_size and the model
        # exposes no describe().defaults.image_size). Tiling is the package
        # default, so degrade gracefully to a plain (resize) call rather than
        # hard-failing; the dispatch backfills resolved_mode="resize" in meta.
        return _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
            input_chw=x,
            fetch_meta=fetch_meta,
        )
    stride = params.stride
    if stride <= 0:
        raise ModelError(f"Invalid tile_stride={stride}")
    # Snap-to-tile: when an axis only slightly exceeds a tile multiple, downscale
    # that overhang away so the axis tiles cleanly instead of spawning a
    # near-fully-overlapping shifted edge tile (which duplicates cells in the
    # stitched grid). Only applies under the stride==tile_size overlap-shift model.
    x, snapped_input_hw = _maybe_snap_input(
        x, tile_size=tile_size, stride=stride, snap_frac=input_prep.tile_snap_frac
    )
    h, w = _input_hw(x)
    tiled_model_config = _augment_model_config_for_tiled_dispatch(
        embedder,
        model_config,
        tile_size=tile_size,
    )
    num_tiles = _estimate_tile_count(h=h, w=w, tile_size=tile_size, stride=stride)
    if input_prep.mode == "auto":
        # A dimension smaller than a tile only forces a plain resize for models
        # without a fixed input size. Fixed-size models tile the long axis
        # (keeping its native resolution) while the short axis is padded to a
        # square tile and the pad cells cropped from the stitched grid, so a
        # wide/flat or tall/thin ROI keeps detail on its long side instead of
        # being squashed by a whole-ROI resize.
        small_dim_forces_resize = (h <= tile_size or w <= tile_size) and not model_fixed_size
        if (
            output.mode != "grid"
            or num_tiles <= 1
            or num_tiles > input_prep.max_tiles
            or small_dim_forces_resize
        ):
            return _call_embedder_get_embedding(
                embedder=embedder,
                spatial=spatial,
                temporal=temporal,
                sensor=sensor,
                model_config=model_config,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
                fetch_meta=fetch_meta,
            )
    elif input_prep.mode == "tile":
        if num_tiles > input_prep.max_tiles_hard:
            raise ModelError(
                f"input_prep tile would create {num_tiles} tiles "
                f"(> max_tiles_hard={input_prep.max_tiles_hard}); reduce the requested area "
                "or raise max_tiles_hard."
            )
        if num_tiles > input_prep.max_tiles:
            warnings.warn(
                f"input_prep tile is producing {num_tiles} tiles "
                f"(> max_tiles={input_prep.max_tiles}); this request may be slow or "
                "memory-heavy. Tiling proceeds; raise max_tiles to silence this warning.",
                stacklevel=2,
            )
    if stride != tile_size:
        raise ModelError(
            "Current tiled input preprocessing supports tile_stride == tile_size only; "
            "boundary tiles may still be shifted to avoid padding."
        )

    fill_value = float(sensor.fill_value) if sensor is not None else 0.0
    tiles, tile_meta, tile_spatials = _tile_one_image(
        x,
        spatial=spatial,
        tile_size=tile_size,
        stride=stride,
        pad_edges=effective_pad_edges,
        fill_value=fill_value,
    )

    if len(tiles) <= 1:
        # Single tile = the whole ROI is one input; the embedder squares + crops it
        # back to the ROI via fetch_meta['roi_window_geo'] (set by the prefetch).
        return _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            model_config=tiled_model_config if input_prep.mode == "tile" else model_config,
            output=output,
            backend=backend,
            device=device,
            input_chw=x,
            fetch_meta=fetch_meta,
        )

    can_batch_tiles = _supports_prefetched_batch_api(embedder) and (
        tiled_model_config is None
        or _embedder_accepts_model_config(type(embedder), "get_embeddings_batch_from_inputs")
    )
    if can_batch_tiles:
        try:
            batch_kwargs: dict[str, Any] = {
                "spatials": tile_spatials,
                "input_chws": tiles,
                "temporal": temporal,
                "sensor": sensor,
                "output": output,
                "backend": backend,
                "device": device,
            }
            if tiled_model_config is not None:
                batch_kwargs["model_config"] = tiled_model_config
            tile_embs = embedder.get_embeddings_batch_from_inputs(
                **batch_kwargs,
            )
            if len(tile_embs) != len(tiles):
                raise ModelError(
                    f"Tiled batch inference returned {len(tile_embs)} outputs for {len(tiles)} tiles."
                )
            tile_embs = [_normalize_embedding_output(emb=e, output=output) for e in tile_embs]
        except Exception as _e:
            tile_embs = [
                _call_embedder_get_embedding(
                    embedder=embedder,
                    spatial=tile_spatials[i],
                    temporal=temporal,
                    sensor=sensor,
                    model_config=tiled_model_config,
                    output=output,
                    backend=backend,
                    device=device,
                    input_chw=tiles[i],
                )
                for i in range(len(tiles))
            ]
    else:
        tile_embs = [
            _call_embedder_get_embedding(
                embedder=embedder,
                spatial=tile_spatials[i],
                temporal=temporal,
                sensor=sensor,
                model_config=tiled_model_config,
                output=output,
                backend=backend,
                device=device,
                input_chw=tiles[i],
            )
            for i in range(len(tiles))
        ]

    prep_meta = _build_tile_prep_meta(
        requested_mode=input_prep.mode,
        tile_size=tile_size,
        stride=stride,
        tile_count=len(tiles),
        effective_pad_edges=effective_pad_edges,
        max_tiles=input_prep.max_tiles,
        max_tiles_hard=input_prep.max_tiles_hard,
        input_hw=(h, w),
        tile_snap_frac=input_prep.tile_snap_frac,
        snapped_from_hw=snapped_input_hw,
    )
    return _aggregate_tiled_embeddings(
        embs=tile_embs,
        tile_meta=tile_meta,
        output=output,
        tile_size=tile_size,
        stride=stride,
        prep_meta=prep_meta,
        # Fetch-square ROI window (set by the prefetch for square-input models);
        # the stitched square is cropped back to it so only the ROI is returned.
        roi_window=geo_roi_from_meta(fetch_meta),
    )


def _stamp_input_prep_meta(
    emb: Embedding,
    *,
    requested_mode: str,
    resolved_mode: str,
) -> Embedding:
    """Ensure an embedding carries a self-describing ``input_prep`` meta block.

    Tiled embeddings already get a rich block from ``_aggregate_tiled_embeddings``;
    this only fills a minimal versioned block when none is present (resize /
    single-tile / auto-fell-back-to-plain paths) so every embedding is
    reproducibility-tagged regardless of which path produced it.
    """
    meta = getattr(emb, "meta", None)
    if isinstance(meta, dict):
        meta.setdefault(
            "input_prep",
            {
                "prep_version": INPUT_PREP_VERSION,
                "requested_mode": str(requested_mode),
                "resolved_mode": str(resolved_mode),
            },
        )
    return emb


def _call_embedder_get_embedding_with_input_prep(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None,
    sensor: SensorSpec | None,
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: np.ndarray | None,
    input_prep: InputPrepSpec | str | None,
    model_config: dict[str, Any] | None = None,
    fetch_meta: dict[str, Any] | None = None,
) -> Embedding:
    """Dispatch to resize (pass-through) or tiled embedding based on input_prep.

    Accepts raw ``input_prep`` (string or ``InputPrepSpec``) and resolves it
    internally.  When mode is ``resize`` or no ``input_chw`` is provided the
    call falls through to the plain embedder call; otherwise
    ``_call_embedder_get_embedding_tiled`` handles slicing, inference, and
    stitching.
    """
    spec = _resolve_input_prep_spec(input_prep)
    if spec.mode == "resize" or input_chw is None:
        emb = _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
            input_chw=input_chw,
            fetch_meta=fetch_meta,
        )
        return _stamp_input_prep_meta(emb, requested_mode=spec.mode, resolved_mode="resize")
    if not _embedder_accepts_input_chw(type(embedder)):
        if spec.mode == "tile":
            raise ModelError(
                f"Model {getattr(embedder, 'model_name', type(embedder).__name__)} does not accept input_chw; "
                "cannot apply input_prep.mode='tile'."
            )
        emb = _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            model_config=model_config,
            output=output,
            backend=backend,
            device=device,
            input_chw=input_chw,
            fetch_meta=fetch_meta,
        )
        return _stamp_input_prep_meta(emb, requested_mode=spec.mode, resolved_mode="resize")
    emb = _call_embedder_get_embedding_tiled(
        embedder=embedder,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        model_config=model_config,
        output=output,
        backend=backend,
        device=device,
        input_chw=np.asarray(input_chw, dtype=np.float32),
        input_prep=spec,
        fetch_meta=fetch_meta,
    )
    # Tiled path stamps a rich block when it actually tiles; this only backfills
    # the single-tile / sub-threshold fall-through where it returned plain.
    return _stamp_input_prep_meta(emb, requested_mode=spec.mode, resolved_mode="resize")
