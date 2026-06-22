import sys

import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders._vendor.copernicus_embed import _bbox_to_window, _infer_axis_order
from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder
from rs_embed.embedders.precomputed_gse_annual import GSEAnnualEmbedder
from rs_embed.embedders.precomputed_tessera import TesseraEmbedder


class _FakeRegistry:
    def __init__(self, n_tiles: int):
        self._n_tiles = n_tiles

    def load_blocks_for_region(self, bounds, year):
        return list(range(self._n_tiles))


class _FakeGeoTessera:
    def __init__(self):
        self.registry = _FakeRegistry(1)

    def fetch_embeddings(self, tiles):
        for i in tiles:
            yield {"tile": i}


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeCopernicusDataset:
    def __getitem__(self, key):
        return {
            "image": _FakeTorchTensor(
                np.array(
                    [
                        [[1.0, 3.0], [5.0, 7.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ],
                    dtype=np.float32,
                ),
            )
        }


def test_precomputed_custom_init_preserves_base_state():
    tessera = TesseraEmbedder()
    copernicus = CopernicusEmbedder()

    assert tessera._providers == {}
    assert copernicus._providers == {}


def test_gse_get_embedding_ignores_input_chw(monkeypatch):
    import rs_embed.embedders.precomputed_gse_annual as gse_mod

    embedder = GSEAnnualEmbedder()
    embedder.model_name = "gse"
    monkeypatch.setattr(embedder, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        gse_mod,
        "_fetch_collection_patch_all_bands_chw",
        lambda provider, **kw: (
            np.array(
                [
                    [[1.0, 3.0], [5.0, 7.0]],
                    [[2.0, 4.0], [6.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            ["b0", "b1"],
        ),
    )

    emb = embedder.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2020),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.array([4.0, 5.0], dtype=np.float32))


def test_gse_get_embedding_uses_sensor_scale_m(monkeypatch):
    import rs_embed.embedders.precomputed_gse_annual as gse_mod

    embedder = GSEAnnualEmbedder()
    embedder.model_name = "gse"
    embedder._get_provider = lambda _backend: object()
    seen = {}

    def _fake_fetch(provider, **kw):
        seen["scale_m"] = kw["scale_m"]
        return np.ones((2, 2, 2), dtype=np.float32), ["b0", "b1"]

    monkeypatch.setattr(gse_mod, "_fetch_collection_patch_all_bands_chw", _fake_fetch)

    emb = embedder.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2020),
        sensor=SensorSpec(collection="GSE", bands=tuple(), scale_m=60),
        output=OutputSpec.pooled(),
        backend="auto",
    )

    assert seen["scale_m"] == 60
    np.testing.assert_allclose(emb.data, np.array([1.0, 1.0], dtype=np.float32))


def test_tessera_get_embedding_ignores_input_chw(monkeypatch):
    import rs_embed.embedders.precomputed_tessera as tessera_mod

    embedder = TesseraEmbedder()
    embedder.model_name = "tessera"
    monkeypatch.setattr(tessera_mod, "_TESSERA_PROJECTION_WARNED", True)
    monkeypatch.setattr(embedder, "_get_gt", lambda _cache: _FakeGeoTessera())
    monkeypatch.setattr(
        tessera_mod,
        "_mosaic_and_crop_strict_roi",
        lambda tiles_fn, bbox_4326: (
            np.full((64, 1, 1), 1.0, dtype=np.float32),
            {"mosaic_hw": (1, 1), "crop_hw": (1, 1)},
        ),
    )

    emb = embedder.get_embedding(
        spatial=BBox(minlon=0.2, minlat=0.2, maxlon=0.8, maxlat=0.8, crs="EPSG:4326"),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.full((64,), 1.0, dtype=np.float32))


def test_copernicus_get_embedding_ignores_input_chw(monkeypatch):
    import rs_embed.embedders.precomputed_copernicus_embed as cop_mod

    embedder = CopernicusEmbedder()
    embedder.model_name = "copernicus"
    monkeypatch.setattr(cop_mod, "_COPERNICUS_PROJECTION_WARNED", True)
    monkeypatch.setattr(
        embedder,
        "_get_dataset",
        lambda *, data_dir, download: _FakeCopernicusDataset(),
    )

    emb = embedder.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.year(2021),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.ones((3, 8, 8), dtype=np.float32),
    )

    np.testing.assert_allclose(emb.data, np.array([4.0, 5.0], dtype=np.float32))


def test_copernicus_requires_tifffile(monkeypatch):
    import rs_embed.embedders._vendor.copernicus_embed as cop_mod

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "tifffile":
            raise ImportError("No module named 'tifffile'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "tifffile", raising=False)
    monkeypatch.setattr("builtins.__import__", _fake_import)
    monkeypatch.setattr(cop_mod, "_validate_large_file", lambda path, min_bytes=0: path)

    with pytest.raises(ModelError, match="pip install tifffile"):
        cop_mod.load_geotiff_meta("/tmp/fake.tif")


def test_copernicus_vendor_window_math():
    class _Meta:
        left = -180.0
        right = 180.0
        bottom = -90.125
        top = 90.125
        xres = 0.25
        yres = 0.25
        width = 1440
        height = 721

    row0, row1, col0, col1 = _bbox_to_window(
        meta=_Meta(),
        minlon=-180.0,
        minlat=89.625,
        maxlon=-179.5,
        maxlat=90.125,
    )
    assert (row0, row1, col0, col1) == (0, 2, 0, 2)


def test_copernicus_vendor_axis_order_detection():
    assert _infer_axis_order((768, 721, 1440)) == "chw"
    assert _infer_axis_order((721, 1440, 768)) == "hwc"


# ---------------------------------------------------------------------------
# Contract: on-the-fly embedders that accept input_chw must implement
# get_embeddings_batch_from_inputs as a true GPU-batch override.
# New embedders must NOT be added to this allowlist without also adding the
# batch implementation.
# ---------------------------------------------------------------------------

_KNOWN_MISSING_BATCH_FROM_INPUTS: frozenset[str] = frozenset(
    {
        "agrifm",
        "anysat",
        "galileo",
        "thor",
    }
)


def test_onthefly_embedders_that_accept_input_chw_implement_batch_from_inputs():
    """Every on-the-fly embedder accepting input_chw must override
    get_embeddings_batch_from_inputs, unless listed in the allowlist above.
    """
    from importlib import import_module

    from rs_embed.embedders.base import EmbedderBase
    from rs_embed.embedders.catalog import MODEL_SPECS as _CATALOG
    from rs_embed.tools.runtime import embedder_accepts_input_chw

    missing_override: list[str] = []
    for model_id, (module_suffix, cls_name) in _CATALOG.items():
        if not module_suffix.startswith("onthefly_"):
            continue  # precomputed embedder — not subject to this contract

        try:
            mod = import_module(f"rs_embed.embedders.{module_suffix}")
            cls = getattr(mod, cls_name)
        except Exception:
            continue  # import failure is a separate concern

        if not embedder_accepts_input_chw(cls):
            continue

        batch_fn = getattr(cls, "get_embeddings_batch_from_inputs", None)
        base_fn = getattr(EmbedderBase, "get_embeddings_batch_from_inputs", None)
        if batch_fn is base_fn:
            if model_id not in _KNOWN_MISSING_BATCH_FROM_INPUTS:
                missing_override.append(model_id)

    assert not missing_override, (
        "The following on-the-fly embedders accept input_chw but do NOT override "
        "get_embeddings_batch_from_inputs (add a real batch implementation, then "
        "remove from _KNOWN_MISSING_BATCH_FROM_INPUTS if present):\n  "
        + "\n  ".join(sorted(missing_override))
    )
