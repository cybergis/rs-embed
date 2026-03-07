"""Input-prep resolution, tiling, stitching, and tile-aware embedding dispatch.

Extracted from api.py to keep the public-API module focused on orchestration.
All symbols are private (underscore-prefixed) and re-exported by api.py where
backward-compatibility with test imports is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.embedding import Embedding
from ...core.errors import ModelError
from ...core.specs import (
    BBox,
    InputPrepSpec,
    OutputSpec,
    SensorSpec,
    SpatialSpec,
)
from ...core.export_helpers import embedding_to_numpy as _embedding_to_numpy
from .output_helpers import normalize_embedding_output as _normalize_embedding_output
from .runtime_helpers import (
    call_embedder_get_embedding as _call_embedder_get_embedding,
    embedder_accepts_input_chw as _embedder_accepts_input_chw,
    supports_prefetched_batch_api as _supports_prefetched_batch_api,
)


# ---------------------------------------------------------------------------
# Resolved input-prep spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ResolvedInputPrepSpec:
    mode: str
    tile_size: Optional[int]
    tile_stride: Optional[int]
    max_tiles: int
    pad_edges: bool


def _resolve_input_prep_spec(
    input_prep: Optional[InputPrepSpec | str],
) -> _ResolvedInputPrepSpec:
    if input_prep is None:
        spec: InputPrepSpec = InputPrepSpec.resize()
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
        raise ModelError(
            f"input_prep.mode must be one of auto/resize/tile, got {mode!r}"
        )
    tile_size = getattr(spec, "tile_size", None)
    tile_stride = getattr(spec, "tile_stride", None)
    max_tiles = int(getattr(spec, "max_tiles", 9))
    pad_edges = bool(getattr(spec, "pad_edges", True))
    if tile_size is not None:
        tile_size = int(tile_size)
        if tile_size <= 0:
            raise ModelError(f"input_prep.tile_size must be > 0, got {tile_size}")
    if tile_stride is not None:
        tile_stride = int(tile_stride)
        if tile_stride <= 0:
            raise ModelError(f"input_prep.tile_stride must be > 0, got {tile_stride}")
    return _ResolvedInputPrepSpec(
        mode=mode,
        tile_size=tile_size,
        tile_stride=tile_stride,
        max_tiles=max(1, max_tiles),
        pad_edges=pad_edges,
    )


# ---------------------------------------------------------------------------
# Tile geometry helpers
# ---------------------------------------------------------------------------


def _embedder_default_image_size(embedder: Any) -> Optional[int]:
    try:
        desc = embedder.describe()
    except Exception:
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
    except Exception:
        return None


def _estimate_tile_count(*, h: int, w: int, tile_size: int, stride: int) -> int:
    ny = (
        1
        if h <= tile_size
        else int(math.ceil((float(h) - float(tile_size)) / float(stride))) + 1
    )
    nx = (
        1
        if w <= tile_size
        else int(math.ceil((float(w) - float(tile_size)) / float(stride))) + 1
    )
    return max(1, ny) * max(1, nx)


def _input_hw(x: np.ndarray) -> Tuple[int, int]:
    if x.ndim not in (3, 4):
        raise ModelError(
            f"Tiling currently supports CHW or TCHW inputs only, got shape={getattr(x, 'shape', None)}"
        )
    return int(x.shape[-2]), int(x.shape[-1])


def _tile_yx_starts(
    *, h: int, w: int, tile_size: int, stride: int
) -> Tuple[List[int], List[int]]:
    def _starts_1d(dim: int) -> List[int]:
        if dim <= tile_size:
            return [0]
        starts: List[int] = [0]
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
        return BBox(
            minlon=lon0, minlat=lat_bot, maxlon=lon1, maxlat=lat_top, crs=spatial.crs
        )
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
) -> Tuple[np.ndarray, Dict[str, int]]:
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
        tile = np.pad(
            tile, pad_spec, mode="constant", constant_values=float(fill_value)
        )
    return tile, {
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
        "valid_h": valid_h,
        "valid_w": valid_w,
    }


# ---------------------------------------------------------------------------
# Tile aggregation (stitching)
# ---------------------------------------------------------------------------


def _crop_len_for_valid(valid: int, *, nominal: int, out_len: int) -> int:
    if nominal <= 0:
        return int(out_len)
    ratio = float(valid) / float(nominal)
    n = int(round(ratio * float(out_len)))
    return max(1, min(int(out_len), n))


def _midpoint_owned_ranges(
    items: List[Tuple[int, int, int]],
) -> Dict[int, Tuple[int, int]]:
    """Compute non-overlapping ownership intervals via midpoint cuts.

    items: [(id, start, end)] in input-pixel coordinates. Intervals must be ordered
    by `start` and cover the domain without gaps. Overlaps are allowed.
    """
    if not items:
        return {}
    items_s = sorted(items, key=lambda t: (int(t[1]), int(t[2]), int(t[0])))
    owned: Dict[int, Tuple[int, int]] = {}
    prev_cut: Optional[int] = None
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
                raise ModelError(
                    "Tiled stitch found a gap between tiles; unsupported tile layout."
                )
            own_s = int((int(pend) + s) // 2)
            if prev_cut is not None:
                own_s = max(own_s, int(prev_cut))
        if i == len(items_s) - 1:
            own_e = e
        else:
            _, nstart, _ = items_s[i + 1]
            if int(nstart) > e:
                raise ModelError(
                    "Tiled stitch found a gap between tiles; unsupported tile layout."
                )
            own_e = int((e + int(nstart)) // 2)
        own_s = max(s, min(own_s, e))
        own_e = max(own_s, min(own_e, e))
        if own_e == own_s:
            own_e = min(e, own_s + 1)
        owned[int(idx)] = (int(own_s), int(own_e))
        prev_cut = own_e
    return owned


def _map_input_subrange_to_feature(
    *,
    rel_start: int,
    rel_end: int,
    valid_len: int,
    out_len: int,
) -> Tuple[int, int]:
    if valid_len <= 0 or out_len <= 0:
        return (0, 0)
    rs = max(0, min(int(valid_len), int(rel_start)))
    re = max(rs, min(int(valid_len), int(rel_end)))
    if re <= rs:
        re = min(int(valid_len), rs + 1)
    fs = int(math.floor((float(rs) / float(valid_len)) * float(out_len)))
    fe = int(math.ceil((float(re) / float(valid_len)) * float(out_len)))
    fs = max(0, min(int(out_len) - 1, fs))
    fe = max(fs + 1, min(int(out_len), fe))
    return (fs, fe)


def _aggregate_tiled_embeddings(
    *,
    embs: List[Embedding],
    tile_meta: List[Dict[str, int]],
    output: OutputSpec,
    tile_size: int,
    stride: int,
    prep_meta: Dict[str, Any],
) -> Embedding:
    if not embs:
        raise ModelError("No tile embeddings produced.")
    base_meta = dict(getattr(embs[0], "meta", {}) or {})
    base_meta["input_prep"] = dict(prep_meta)

    if output.mode == "pooled":
        vecs = [
            np.asarray(_embedding_to_numpy(e), dtype=np.float32).reshape(-1)
            for e in embs
        ]
        dims = {tuple(v.shape) for v in vecs}
        if len(dims) != 1:
            raise ModelError(
                f"Tiled pooled merge requires consistent vector shapes, got {sorted(dims)}"
            )
        mat = np.stack(vecs, axis=0)
        if str(output.pooling).lower() == "max":
            out_vec = np.max(mat, axis=0).astype(np.float32, copy=False)
        else:
            ws = np.asarray(
                [
                    max(1, int(m["valid_h"])) * max(1, int(m["valid_w"]))
                    for m in tile_meta
                ],
                dtype=np.float32,
            )
            out_vec = (mat * ws[:, None]).sum(axis=0) / max(1.0, float(ws.sum()))
            out_vec = out_vec.astype(np.float32, copy=False)
        base_meta["input_prep"]["merged_output"] = "pooled_reduce"
        return Embedding(data=out_vec, meta=base_meta)

    arrays = [np.asarray(_embedding_to_numpy(e), dtype=np.float32) for e in embs]
    for a in arrays:
        if a.ndim < 2:
            raise ModelError(
                f"Tiled grid merge expects arrays with ndim>=2, got shape={a.shape}"
            )
    gh = int(arrays[0].shape[-2])
    gw = int(arrays[0].shape[-1])
    lead_shape = tuple(int(v) for v in arrays[0].shape[:-2])
    for a in arrays[1:]:
        if (
            tuple(int(v) for v in a.shape[:-2]) != lead_shape
            or int(a.shape[-2]) != gh
            or int(a.shape[-1]) != gw
        ):
            raise ModelError(
                "Tiled grid merge requires consistent per-tile output shapes."
            )

    nrows = max(int(m["row"]) for m in tile_meta) + 1
    ncols = max(int(m["col"]) for m in tile_meta) + 1

    row_items: List[Tuple[int, int, int]] = []
    col_items: List[Tuple[int, int, int]] = []
    row_ref: Dict[int, Dict[str, int]] = {}
    col_ref: Dict[int, Dict[str, int]] = {}
    for m in tile_meta:
        r = int(m["row"])
        c = int(m["col"])
        if r not in row_ref:
            row_ref[r] = {
                "start": int(m["y0"]),
                "end": int(m["y1"]),
                "valid": int(m["valid_h"]),
                "out_len": int(gh),
            }
            row_items.append((r, int(m["y0"]), int(m["y1"])))
        if c not in col_ref:
            col_ref[c] = {
                "start": int(m["x0"]),
                "end": int(m["x1"]),
                "valid": int(m["valid_w"]),
                "out_len": int(gw),
            }
            col_items.append((c, int(m["x0"]), int(m["x1"])))

    row_owned = _midpoint_owned_ranges(row_items)
    col_owned = _midpoint_owned_ranges(col_items)

    row_heights = [0] * nrows
    col_widths = [0] * ncols
    row_crop: Dict[int, Tuple[int, int]] = {}
    col_crop: Dict[int, Tuple[int, int]] = {}
    for r in range(nrows):
        rr = row_ref.get(r)
        if rr is None:
            raise ModelError(f"Missing row metadata for tiled stitch row={r}.")
        own_s, own_e = row_owned[r]
        rel_s = int(own_s - rr["start"])
        rel_e = int(own_e - rr["start"])
        fy0, fy1 = _map_input_subrange_to_feature(
            rel_start=rel_s,
            rel_end=rel_e,
            valid_len=int(rr["valid"]),
            out_len=int(rr["out_len"]),
        )
        row_crop[r] = (fy0, fy1)
        row_heights[r] = int(fy1 - fy0)
    for c in range(ncols):
        cc = col_ref.get(c)
        if cc is None:
            raise ModelError(f"Missing col metadata for tiled stitch col={c}.")
        own_s, own_e = col_owned[c]
        rel_s = int(own_s - cc["start"])
        rel_e = int(own_e - cc["start"])
        fx0, fx1 = _map_input_subrange_to_feature(
            rel_start=rel_s,
            rel_end=rel_e,
            valid_len=int(cc["valid"]),
            out_len=int(cc["out_len"]),
        )
        col_crop[c] = (fx0, fx1)
        col_widths[c] = int(fx1 - fx0)

    out_h = int(sum(row_heights))
    out_w = int(sum(col_widths))
    out_arr = np.zeros(lead_shape + (out_h, out_w), dtype=np.float32)
    row_offsets = [0] * nrows
    col_offsets = [0] * ncols
    for r in range(1, nrows):
        row_offsets[r] = row_offsets[r - 1] + row_heights[r - 1]
    for c in range(1, ncols):
        col_offsets[c] = col_offsets[c - 1] + col_widths[c - 1]

    for arr, m in zip(arrays, tile_meta):
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
    # Keep model-reported grid metadata consistent with stitched output.
    if "grid_hw" in base_meta:
        base_meta["grid_hw"] = (int(out_h), int(out_w))
    return Embedding(data=out_arr, meta=base_meta)


# ---------------------------------------------------------------------------
# Tile-aware embedding dispatch
# ---------------------------------------------------------------------------


def _call_embedder_get_embedding_tiled(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: np.ndarray,
    input_prep: _ResolvedInputPrepSpec,
) -> Embedding:
    x = np.asarray(input_chw, dtype=np.float32)
    h, w = _input_hw(x)
    model_img = _embedder_default_image_size(embedder)
    tile_size = int(input_prep.tile_size or model_img or 0)
    if tile_size <= 0:
        raise ModelError(
            "Tiled input preprocessing requires tile_size or a model describe().defaults.image_size."
        )
    stride = int(input_prep.tile_stride or tile_size)
    if stride <= 0:
        raise ModelError(f"Invalid tile_stride={stride}")
    num_tiles = _estimate_tile_count(h=h, w=w, tile_size=tile_size, stride=stride)
    if input_prep.mode == "auto":
        if (
            output.mode != "grid"
            or h <= tile_size
            or w <= tile_size
            or num_tiles <= 1
            or num_tiles > input_prep.max_tiles
        ):
            return _call_embedder_get_embedding(
                embedder=embedder,
                spatial=spatial,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                input_chw=x,
            )
    elif input_prep.mode == "tile":
        if num_tiles > input_prep.max_tiles:
            raise ModelError(
                f"input_prep tile would create {num_tiles} tiles (> max_tiles={input_prep.max_tiles}); "
                "increase max_tiles or use resize/auto."
            )
    if stride != tile_size:
        raise ModelError(
            "Current tiled input preprocessing supports tile_stride == tile_size only; "
            "boundary tiles may still be shifted to avoid padding."
        )

    ys, xs = _tile_yx_starts(h=h, w=w, tile_size=tile_size, stride=stride)
    fill_value = float(sensor.fill_value) if sensor is not None else 0.0
    tiles: List[np.ndarray] = []
    tile_meta: List[Dict[str, int]] = []
    tile_spatials: List[SpatialSpec] = []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tile, meta = _slice_and_pad_tile(
                x,
                y0=int(y0),
                x0=int(x0),
                tile_size=tile_size,
                pad_edges=bool(input_prep.pad_edges),
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

    if len(tiles) <= 1:
        return _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend,
            device=device,
            input_chw=x,
        )

    if _supports_prefetched_batch_api(embedder):
        try:
            tile_embs = embedder.get_embeddings_batch_from_inputs(
                spatials=tile_spatials,
                input_chws=tiles,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
            )
            if len(tile_embs) != len(tiles):
                raise ModelError(
                    f"Tiled batch inference returned {len(tile_embs)} outputs for {len(tiles)} tiles."
                )
            tile_embs = [
                _normalize_embedding_output(emb=e, output=output) for e in tile_embs
            ]
        except Exception:
            tile_embs = [
                _call_embedder_get_embedding(
                    embedder=embedder,
                    spatial=tile_spatials[i],
                    temporal=temporal,
                    sensor=sensor,
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
                output=output,
                backend=backend,
                device=device,
                input_chw=tiles[i],
            )
            for i in range(len(tiles))
        ]

    prep_meta: Dict[str, Any] = {
        "requested_mode": input_prep.mode,
        "resolved_mode": "tile",
        "tile_layout": "cover_shift",
        "tile_size": int(tile_size),
        "tile_stride": int(stride),
        "tile_count": int(len(tiles)),
        "pad_edges": bool(input_prep.pad_edges),
        "max_tiles": int(input_prep.max_tiles),
        "input_hw": (int(h), int(w)),
    }
    return _aggregate_tiled_embeddings(
        embs=tile_embs,
        tile_meta=tile_meta,
        output=output,
        tile_size=tile_size,
        stride=stride,
        prep_meta=prep_meta,
    )


def _call_embedder_get_embedding_with_input_prep(
    *,
    embedder: Any,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec],
    sensor: Optional[SensorSpec],
    output: OutputSpec,
    backend: str,
    device: str,
    input_chw: Optional[np.ndarray],
    input_prep: Optional[InputPrepSpec | str],
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
        return _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend,
            device=device,
            input_chw=input_chw,
        )
    if not _embedder_accepts_input_chw(type(embedder)):
        if spec.mode == "tile":
            raise ModelError(
                f"Model {getattr(embedder, 'model_name', type(embedder).__name__)} does not accept input_chw; "
                "cannot apply input_prep.mode='tile'."
            )
        return _call_embedder_get_embedding(
            embedder=embedder,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor,
            output=output,
            backend=backend,
            device=device,
            input_chw=input_chw,
        )
    return _call_embedder_get_embedding_tiled(
        embedder=embedder,
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        output=output,
        backend=backend,
        device=device,
        input_chw=np.asarray(input_chw, dtype=np.float32),
        input_prep=spec,
    )
