"""Fetch-square ROI crop-back on the API tiling path (multi-tile case).

A square-input model whose ROI is large enough to tile must still return only the
requested (rectangular) ROI: the prefetch squares the ROI, the tiler stitches the
square at native resolution, and the stitched grid is cropped back to the ROI
window. This guards the multi-tile branch (the single-tile branch is covered by
``test_olmoearth.test_top_level_api_fetch_square_crops_grid``).
"""

from __future__ import annotations

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec
from rs_embed.tools.tiling import (
    _call_embedder_get_embedding_tiled,
    _ResolvedInputPrepSpec,
)


class _FakeSquareEmbedder:
    """Minimal embedder: each tile yields a [D,h,w] grid equal to its input size."""

    model_name = "fake_square"

    def describe(self):
        return {"defaults": {"image_size": 24}}

    def get_embedding(self, *, input_chw, output, **kw):
        x = np.asarray(input_chw, dtype=np.float32)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        d = 4
        grid = np.zeros((d, h, w), dtype=np.float32)
        import xarray as xr

        da = xr.DataArray(
            grid,
            dims=("d", "y", "x"),
            coords={"d": np.arange(d), "y": np.arange(h), "x": np.arange(w)},
            name="embedding",
            attrs={"grid_hw": (h, w)},
        )
        return Embedding(data=da, meta={"grid_hw": (h, w)})


def _tile_spec() -> _ResolvedInputPrepSpec:
    return _ResolvedInputPrepSpec(
        mode="tile",
        max_tiles=64,
        max_tiles_hard=1024,
        tile_size=0,
        tile_stride=0,
        tile_snap_frac=0.0,  # disable snapping so dims stay exact
        pad_edges=False,
    )


def test_tiled_grid_cropped_back_to_rectangular_roi():
    pytest.importorskip("xarray")
    emb_in = np.zeros((4, 48, 48), dtype=np.float32)  # square fetch -> 2x2 tiles of 24
    out = _call_embedder_get_embedding_tiled(
        embedder=_FakeSquareEmbedder(),
        spatial=None,
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="auto",
        device="cpu",
        input_chw=emb_in,
        input_prep=_tile_spec(),
        # ROI occupies the middle 50% in height, full width (a wide rectangle).
        fetch_meta={"roi_window_geo": (0.25, 0.75, 0.0, 1.0)},
    )
    h, w = out.meta["grid_hw"]
    assert (h, w) == (24, 48)  # height cropped to the ROI, width untouched
    assert h < w
    assert out.meta["input_prep"].get("roi_cropped") is True


def test_tiled_grid_full_window_not_cropped():
    pytest.importorskip("xarray")
    emb_in = np.zeros((4, 48, 48), dtype=np.float32)
    out = _call_embedder_get_embedding_tiled(
        embedder=_FakeSquareEmbedder(),
        spatial=None,
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="auto",
        device="cpu",
        input_chw=emb_in,
        input_prep=_tile_spec(),
        fetch_meta=None,  # no ROI window -> full square, no crop
    )
    assert tuple(out.meta["grid_hw"]) == (48, 48)
    assert out.meta["input_prep"].get("roi_cropped") is not True
