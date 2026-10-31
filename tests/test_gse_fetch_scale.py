import importlib
import sys
import types

import numpy as np
import pytest

from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, SensorSpec, TemporalSpec


def test_gse_get_embedding_uses_sensor_scale_m(monkeypatch):
    fake_xarray = types.ModuleType("xarray")
    fake_xarray.DataArray = object
    monkeypatch.setitem(sys.modules, "xarray", fake_xarray)

    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    embedder = gse_mod.GSEAnnualEmbedder()
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


def test_gse_legacy_max_pixels_env_warns_and_is_ignored(monkeypatch):
    gse_mod = importlib.import_module("rs_embed.embedders.precomputed_gse_annual")
    embedder = gse_mod.GSEAnnualEmbedder()
    embedder.model_name = "gse"
    embedder._get_provider = lambda _backend: object()
    requested = []

    def _fake_fetch(provider, **kw):
        requested.append(kw["spatial"])
        return np.ones((2, 2, 2), dtype=np.float32), ["b0", "b1"]

    monkeypatch.setattr(gse_mod, "_fetch_collection_patch_all_bands_chw", _fake_fetch)
    monkeypatch.setenv("RS_EMBED_GSE_MAX_PIXELS", "1")
    spatial = BBox(minlon=0.0, minlat=0.0, maxlon=0.1, maxlat=0.1)

    with pytest.warns(FutureWarning, match="RS_EMBED_GSE_MAX_PIXELS"):
        embedder.get_embedding(
            spatial=spatial,
            temporal=TemporalSpec.year(2020),
            sensor=SensorSpec(collection="GSE", bands=tuple(), scale_m=10),
            output=OutputSpec.pooled(),
            backend="auto",
        )

    # One request for the whole ROI: the variable no longer triggers tiling.
    assert requested == [spatial]
