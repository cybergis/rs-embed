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


def test_gse_get_embedding_has_no_input_chw_param(monkeypatch):
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


def test_tessera_get_embedding_has_no_input_chw_param(monkeypatch):
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
    )

    np.testing.assert_allclose(emb.data, np.full((64,), 1.0, dtype=np.float32))


def test_copernicus_get_embedding_has_no_input_chw_param(monkeypatch):
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


def test_base_fetch_input_multi_defaults_temporal_like_get_embedding(monkeypatch):
    """fetch_input(temporal=None) must use the temporal_to_range default window.

    Regression: multi-frame spec models without a fetch_input override
    (anysat, agrifm) raised on temporal=None from the API prefetch path,
    while the same call with input_prep='resize' succeeded via the
    embedder's own temporal_to_range default.
    """
    import numpy as np

    from rs_embed.core.specs import ModelInputSpec, PointBuffer
    from rs_embed.embedders.base import EmbedderBase
    from rs_embed.embedders.meta import temporal_to_range

    seen: dict[str, object] = {}

    def fake_multi(provider, *, spatial, temporal, **kwargs):
        seen["temporal"] = temporal
        return np.zeros((2, 1, 4, 4), dtype=np.float32)

    monkeypatch.setattr("rs_embed.providers.fetch.fetch_s2_multiframe_raw_tchw", fake_multi)

    class _MultiModel(EmbedderBase):
        model_name = "multi_dummy"
        input_spec = ModelInputSpec(
            collection="C",
            bands=("B1",),
            temporal_mode="multi",
            n_frames=2,
        )

    fr = _MultiModel().fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        temporal=None,
        sensor=None,
    )
    assert fr is not None
    default = temporal_to_range(None)
    assert seen["temporal"] == default


def test_base_fetch_input_single_defaults_temporal_like_get_embedding(monkeypatch):
    """Single-composite fetch_input(temporal=None) must not fetch unfiltered.

    The direct get_embedding path resolves None to the package default
    window; the prefetch path previously passed None through to the
    provider (whole-collection composite, different data).
    """
    import numpy as np

    from rs_embed.core.specs import ModelInputSpec, PointBuffer
    from rs_embed.embedders.base import EmbedderBase
    from rs_embed.embedders.meta import temporal_to_range

    seen: dict[str, object] = {}

    def fake_single(provider, *, spatial, temporal, **kwargs):
        seen["temporal"] = temporal
        return np.zeros((1, 4, 4), dtype=np.float32)

    monkeypatch.setattr("rs_embed.providers.fetch.fetch_collection_patch_chw", fake_single)

    class _SingleModel(EmbedderBase):
        model_name = "single_dummy"
        input_spec = ModelInputSpec(collection="C", bands=("B1",))

    fr = _SingleModel().fetch_input(
        object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        temporal=None,
        sensor=None,
    )
    assert fr is not None
    assert seen["temporal"] == temporal_to_range(None)


def test_precomputed_embedders_declare_honest_capabilities():
    """Signatures are the capability contract; precomputed models must not
    advertise prefetched-input support they ignore, and must carry the
    _is_precomputed flag so the API-side prefetch skips them.

    Regression: tessera/copernicus declared (and silently ignored) input_chw
    and lacked the flag - tessera's documented cache override
    (collection="cache:<dir>") was prefetched as a GEE collection and failed.
    """
    from rs_embed.embedders.precomputed_copernicus_embed import CopernicusEmbedder as _Cop
    from rs_embed.embedders.precomputed_gse_annual import GSEAnnualEmbedder as _Gse
    from rs_embed.embedders.precomputed_tessera import TesseraEmbedder as _Tes
    from rs_embed.tools.runtime import embedder_accepts_input_chw

    embedder_accepts_input_chw.cache_clear()
    for cls in (_Tes, _Cop, _Gse):
        assert getattr(cls, "_is_precomputed", False) is True, cls.__name__
        assert embedder_accepts_input_chw(cls) is False, cls.__name__


def test_base_batch_raises_on_unsupported_model_config():
    """The batch paths must reject an unsupported model_config like the
    single path does, instead of silently dropping it.

    Regression: export_batch with model kwargs for a model whose
    get_embedding has no model_config ran the prefetched-batch path with the
    config silently ignored, while the same request on the single path raised.
    """
    import pytest

    from rs_embed.core.embedding import Embedding
    from rs_embed.core.errors import ModelError
    from rs_embed.core.specs import OutputSpec, PointBuffer
    from rs_embed.embedders.base import EmbedderBase

    class _NoConfig(EmbedderBase):
        model_name = "noconfig"

        def get_embedding(self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None):
            return Embedding(data=np.zeros(2, dtype=np.float32), meta={})

    emb = _NoConfig()
    sp = [PointBuffer(lon=0, lat=0, buffer_m=10)]
    with pytest.raises(ModelError, match="model-specific"):
        emb.get_embeddings_batch(
            spatials=sp, temporal=None, sensor=None, model_config={"variant": "x"}
        )
    with pytest.raises(ModelError, match="model-specific"):
        emb.get_embeddings_batch_from_inputs(
            spatials=sp,
            input_chws=[np.zeros((1, 2, 2), dtype=np.float32)],
            model_config={"variant": "x"},
        )
    # And without model_config both still work.
    assert len(emb.get_embeddings_batch(spatials=sp, temporal=None, sensor=None)) == 1
