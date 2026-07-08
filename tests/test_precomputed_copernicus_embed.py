import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder


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
    path = "/tmp/copernicus/embed_map_310k.tif"

    def __getitem__(self, key):
        return {
            "image": _FakeTorchTensor(
                np.array(
                    [
                        [[1.0, 3.0], [5.0, 7.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ],
                    dtype=np.float32,
                )
            )
        }


def test_copernicus_embedder_pooled_output_uses_vendored_meta(monkeypatch):
    import rs_embed.embedders.precomputed_copernicus_embed as cop_mod

    embedder = CopernicusEmbedder()
    embedder.model_name = "copernicus"
    monkeypatch.setattr(cop_mod, "_COPERNICUS_PROJECTION_WARNED", False)
    monkeypatch.setattr(
        embedder,
        "_get_dataset",
        lambda *, data_dir, download: _FakeCopernicusDataset(),
    )

    with pytest.warns(UserWarning, match="fixed product grid in EPSG:4326"):
        emb = embedder.get_embedding(
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
            temporal=TemporalSpec.year(2021),
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
        )

    np.testing.assert_allclose(emb.data, np.array([4.0, 5.0], dtype=np.float32))
    # Regression (review M9): the served year must be recorded, not {"mode": None}.
    assert emb.meta["temporal"]["mode"] == "year"
    assert emb.meta["temporal"]["year"] == 2021
    assert emb.meta["backend"] == "vendored_geotiff"
    assert emb.meta["source"] == "hf://torchgeo/copernicus_embed/embed_map_310k.tif"
    assert emb.meta["dataset_path"] == "/tmp/copernicus/embed_map_310k.tif"
    assert emb.meta["input_crs"] == "EPSG:4326"
    assert emb.meta["output_crs"] == "EPSG:4326"
    assert emb.meta["projection_mode"] == "product_native_fixed"
    assert emb.meta["product_resolution_deg"] == (0.25, 0.25)


def test_copernicus_embedder_rejects_subpixel_roi(monkeypatch):
    embedder = CopernicusEmbedder()
    embedder.model_name = "copernicus"

    class _RejectingDataset:
        def __getitem__(self, key):
            raise ModelError("Requested Copernicus bbox is smaller than one dataset pixel")

    monkeypatch.setattr(
        embedder,
        "_get_dataset",
        lambda *, data_dir, download: _RejectingDataset(),
    )

    with pytest.raises(ModelError, match="smaller than one dataset pixel"):
        embedder.get_embedding(
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=1.0),
            temporal=TemporalSpec.year(2021),
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
        )
