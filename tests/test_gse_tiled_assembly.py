"""Assembly of the GSE tiled fetch.

These pin the fix for a duplication bug: `_tile_yx_starts` pulls the LAST tile start back to
(dim - tile_size) so the final tile is full-sized rather than a short remainder, which means it
OVERLAPS its neighbour whenever the dimension is not a whole multiple of the tile size. The
previous assembly concatenated the tiles whole, so the overlapped ground was emitted twice and
the mosaic came out larger than the grid that was asked for.
"""

import importlib
import math

import numpy as np

from rs_embed.core.specs import BBox

N_BANDS = 3


def _ground(y, x):
    """A distinct value per ground pixel, so duplicated ground is detectable by value."""
    return (np.asarray(y) * 10_000 + np.asarray(x)).astype(np.float32)


def _overlapping_dims(gse_mod):
    """A grid whose last tile start is pulled back, i.e. the tiles overlap."""
    tile = int(math.isqrt(gse_mod._gse_pixel_threshold()))
    h_px, w_px = 2 * tile + tile // 3, 2 * tile + tile // 5
    ys, xs = gse_mod._tile_yx_starts(h=h_px, w=w_px, tile_size=tile, stride=tile)
    assert len(ys) > 2 and len(xs) > 2, "fixture must tile in both directions"
    assert ys[-1] < ys[-2] + tile, "fixture must produce OVERLAPPING row tiles"
    assert xs[-1] < xs[-2] + tile, "fixture must produce OVERLAPPING column tiles"
    return tile, h_px, w_px


def _patch_tiler(monkeypatch, gse_mod, *, trim=0):
    """Make each tile carry the ground it actually covers.

    `_tile_subspatial` is replaced by the pixel window itself so the fake provider knows
    precisely which ground to return -- the real one converts that window to lon/lat.
    """
    monkeypatch.setattr(
        gse_mod,
        "_tile_subspatial",
        lambda spatial, *, full_h, full_w, y0, y1, x0, x1: (y0, y1, x0, x1),
    )

    def _fake_fetch(provider, *, spatial, **kw):
        y0, y1, x0, x1 = spatial
        yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
        patch = _ground(yy, xx)
        if trim:
            patch = patch[: max(1, patch.shape[0] - trim), : max(1, patch.shape[1] - trim)]
        return np.stack([patch] * N_BANDS, axis=0), [f"A{i:02d}" for i in range(N_BANDS)]

    monkeypatch.setattr(gse_mod, "_fetch_collection_patch_all_bands_chw", _fake_fetch)


def _run(gse_mod, *, h_px, w_px):
    return gse_mod.GSEAnnualEmbedder()._fetch_tiled(
        object(),
        spatial=BBox(minlon=-88.3, minlat=40.0, maxlon=-88.1, maxlat=40.2),
        temporal=None,
        scale_m=10,
        h_px=h_px,
        w_px=w_px,
    )


def test_overlapping_tiles_assemble_to_the_requested_grid(monkeypatch):
    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    _, h_px, w_px = _overlapping_dims(gse_mod)
    _patch_tiler(monkeypatch, gse_mod)

    out, bands = _run(gse_mod, h_px=h_px, w_px=w_px)

    # Concatenating whole tiles produced len(xs)*tile_size columns by len(ys)*tile_size rows,
    # which is strictly larger than the grid that was requested.
    assert out.shape == (N_BANDS, h_px, w_px)
    assert len(bands) == N_BANDS


def test_no_ground_is_duplicated_across_the_tile_seam(monkeypatch):
    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    _, h_px, w_px = _overlapping_dims(gse_mod)
    _patch_tiler(monkeypatch, gse_mod)

    out, _ = _run(gse_mod, h_px=h_px, w_px=w_px)

    yy, xx = np.meshgrid(np.arange(h_px), np.arange(w_px), indexing="ij")
    expected = _ground(yy, xx)
    for band in range(N_BANDS):
        # Every pixel holds the ground belonging to its own position. Writing the overlap
        # twice is fine -- it is the same ground both times -- but shifting it is not.
        np.testing.assert_array_equal(out[band], expected)


def test_every_pixel_is_written(monkeypatch):
    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    _, h_px, w_px = _overlapping_dims(gse_mod)
    _patch_tiler(monkeypatch, gse_mod)

    out, _ = _run(gse_mod, h_px=h_px, w_px=w_px)

    # The canvas starts as the -9999 nodata fill; full tile coverage must leave none of it.
    # (get_embedding turns any remaining fill into NaN and reports it as nodata_fraction.)
    assert not np.any(out == -9999.0)


def test_a_provider_tile_that_comes_back_short_does_not_kill_the_grid(monkeypatch):
    """Providers occasionally return a window a pixel off; take what fits."""
    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    _, h_px, w_px = _overlapping_dims(gse_mod)
    _patch_tiler(monkeypatch, gse_mod, trim=1)

    out, _ = _run(gse_mod, h_px=h_px, w_px=w_px)

    assert out.shape == (N_BANDS, h_px, w_px)
    # What did arrive is still placed at its own ground position.
    assert out[0, 0, 0] == _ground(0, 0)
