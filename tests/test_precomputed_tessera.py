import numpy as np
from affine import Affine

from rs_embed.core.specs import BBox, OutputSpec, TemporalSpec
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder


class _FakeRegistry:
    def __init__(self, n_tiles: int):
        self._n_tiles = n_tiles

    def load_blocks_for_region(self, bounds, year):
        return list(range(self._n_tiles))


class _FakeGeoTessera:
    def __init__(self, rows):
        self._rows = rows
        self.registry = _FakeRegistry(len(rows))

    def fetch_embeddings(self, tiles):
        for i in tiles:
            yield self._rows[i]


def _fake_rows():
    d = 64
    h = w = 4

    # 2x2 tiles on a shared north-up grid:
    # x in [0,8], y in [0,8], each tile is 4x4 pixels.
    rows = [
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 1.0, dtype=np.float32),
            "EPSG:4326",
            Affine(1, 0, 0, 0, -1, 8),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 2.0, dtype=np.float32),
            "EPSG:4326",
            Affine(1, 0, 4, 0, -1, 8),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 3.0, dtype=np.float32),
            "EPSG:4326",
            Affine(1, 0, 0, 0, -1, 4),
        ),
        (
            2021,
            0.0,
            0.0,
            np.full((h, w, d), 4.0, dtype=np.float32),
            "EPSG:4326",
            Affine(1, 0, 4, 0, -1, 4),
        ),
    ]
    return rows


def test_tessera_pooled_uses_crop_canvas_not_full_mosaic(monkeypatch):
    import rs_embed.embedders.precomputed_tessera as tessera_mod

    embedder = TesseraEmbedder()
    embedder.model_name = "tessera"
    monkeypatch.setattr(embedder, "_get_gt", lambda _cache: _FakeGeoTessera(_fake_rows()))

    zeros_calls = []
    real_zeros = tessera_mod.np.zeros

    def _zeros_probe(shape, *args, **kwargs):
        zeros_calls.append(tuple(int(v) for v in shape))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(tessera_mod.np, "zeros", _zeros_probe)

    emb = embedder.get_embedding(
        spatial=BBox(minlon=4.2, minlat=3.2, maxlon=4.8, maxlat=3.8, crs="EPSG:4326"),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
    )

    # Mosaic covers 8x8; strict ROI covers 1x1.
    assert emb.meta["mosaic_hw"] == (8, 8)
    assert emb.meta["crop_hw"] == (1, 1)
    assert (8, 8, 64) not in zeros_calls
    assert (1, 1, 64) in zeros_calls
    np.testing.assert_allclose(emb.data, np.full((64,), 4.0, dtype=np.float32))
