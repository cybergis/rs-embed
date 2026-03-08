import json
import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import PointBuffer, TemporalSpec, SensorSpec, OutputSpec
from rs_embed.tools.runtime import get_embedder_bundle_cached


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()


def test_export_batch_prefetch_dedup_across_models(tmp_path, monkeypatch):
    # Register two on-the-fly models that require input_chw to be provided.
    class DummyA:
        calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyA.calls += 1
            assert backend == "gee"
            assert input_chw is not None
            return Embedding(
                data=np.array([float(np.sum(input_chw))], dtype=np.float32), meta={}
            )

    class DummyB:
        calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyB.calls += 1
            assert backend == "gee"
            assert input_chw is not None
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_a")(DummyA)
    registry.register("dummy_b")(DummyB)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        return np.ones((3, 4, 4), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=-122.4, lat=37.8, buffer_m=50),
        PointBuffer(lon=-122.3, lat=37.7, buffer_m=50),
    ]
    temporal = TemporalSpec.range("2020-01-01", "2020-02-01")
    sensor = SensorSpec(
        collection="C",
        bands=("B1", "B2", "B3"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
    )

    out_dir = tmp_path / "out"
    res = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_a", "dummy_b"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=True,
        save_embeddings=True,
        chunk_size=10,
        num_workers=4,
    )

    # One fetch per spatial (dedup across models sharing identical sensor)
    assert fetch_calls["n"] == len(spatials)

    # One embedding per spatial per model
    assert DummyA.calls == len(spatials)
    assert DummyB.calls == len(spatials)

    assert len(res) == len(spatials)
    for i in range(len(spatials)):
        assert (out_dir / f"p{i:05d}.npz").exists()
        assert (out_dir / f"p{i:05d}.json").exists()


def test_export_batch_prefetch_reuses_superset_and_slices_subset(tmp_path, monkeypatch):
    class DummyRGB:
        seen = []

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B4", "B3", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            assert tuple(input_chw.shape) == (3, 2, 2)
            DummyRGB.seen.append(tuple(float(input_chw[i, 0, 0]) for i in range(3)))
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyS2Superset:
        seen = []

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B2", "B3", "B4", "B8"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            assert tuple(input_chw.shape) == (4, 2, 2)
            DummyS2Superset.seen.append(
                tuple(float(input_chw[i, 0, 0]) for i in range(4))
            )
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_rgb_subset")(DummyRGB)
    registry.register("dummy_s2_superset")(DummyS2Superset)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0, "bands": []}
    band_value = {"B2": 2.0, "B3": 3.0, "B4": 4.0, "B8": 8.0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        fetch_calls["bands"].append(tuple(sensor.bands))
        x = np.zeros((len(sensor.bands), 2, 2), dtype=np.float32)
        for j, b in enumerate(sensor.bands):
            x[j] = band_value[b]
        return x

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=0.1, lat=0.1, buffer_m=10),
    ]
    out_dir = tmp_path / "superset_dedup"
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_rgb_subset", "dummy_s2_superset"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )

    assert fetch_calls["n"] == len(spatials)
    assert all(set(bands) == {"B2", "B3", "B4", "B8"} for bands in fetch_calls["bands"])
    assert DummyRGB.seen == [(4.0, 3.0, 2.0), (4.0, 3.0, 2.0)]
    assert DummyS2Superset.seen == [(2.0, 3.0, 4.0, 8.0), (2.0, 3.0, 4.0, 8.0)]


def test_export_batch_combined_npz_dedup(tmp_path, monkeypatch):
    class DummyC:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_c")(DummyC)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        return np.zeros((3, 2, 2), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=0.1, lat=0.1, buffer_m=10),
    ]
    temporal = TemporalSpec.range("2020-01-01", "2020-02-01")
    sensor = SensorSpec(
        collection="C",
        bands=("B1", "B2", "B3"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
    )

    out_path = tmp_path / "combined.npz"
    api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_c"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=True,
        save_embeddings=True,
        num_workers=4,
    )

    assert out_path.exists()
    assert fetch_calls["n"] == len(spatials)


def test_export_batch_combined_prefetch_checkpoint_handles_variable_input_shapes(
    tmp_path, monkeypatch
):
    class DummyVarShape:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            return Embedding(
                data=np.array([float(input_chw.shape[1])], dtype=np.float32), meta={}
            )

    registry.register("dummy_var_shape")(DummyVarShape)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    def fake_fetch(provider, *, spatial, temporal, sensor):
        side = 2 if float(spatial.lon) < 0.5 else 3
        return np.zeros((3, side, side), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        PointBuffer(lon=1.0, lat=1.0, buffer_m=10),
    ]
    out_path = tmp_path / "combined_var_shape.npz"
    manifest = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_var_shape"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        save_inputs=True,
        save_embeddings=True,
        show_progress=False,
    )

    assert out_path.exists()
    assert isinstance(manifest, dict)
    assert manifest.get("status") == "ok"


def test_export_batch_combined_falls_back_to_single_when_batch_api_fails(
    tmp_path, monkeypatch
):
    class DummyBatchFail:
        single_calls = 0
        batch_calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyBatchFail.single_calls += 1
            assert input_chw is not None
            return Embedding(
                data=np.array([float(spatial.lon)], dtype=np.float32), meta={}
            )

        def get_embeddings_batch_from_inputs(
            self,
            *,
            spatials,
            input_chws,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
            device="auto",
        ):
            DummyBatchFail.batch_calls += 1
            raise ValueError("synthetic batch failure")

    registry.register("dummy_batch_fail")(DummyBatchFail)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    def fake_fetch(provider, *, spatial, temporal, sensor):
        return np.zeros((3, 2, 2), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "combined_batch_fallback.npz"
    manifest = api.export_batch(
        spatials=[
            PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
            PointBuffer(lon=1.0, lat=1.0, buffer_m=10),
        ],
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_batch_fail"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        save_inputs=True,
        save_embeddings=True,
        show_progress=False,
    )

    assert out_path.exists()
    assert isinstance(manifest, dict)
    assert manifest.get("status") == "ok"
    assert DummyBatchFail.batch_calls >= 1
    assert DummyBatchFail.single_calls == 2


def test_export_batch_combined_prefetch_reuses_superset_and_slices_subset(
    tmp_path, monkeypatch
):
    class DummyRGB:
        seen = []

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B4", "B3", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            DummyRGB.seen.append(tuple(float(input_chw[i, 0, 0]) for i in range(3)))
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyS2Superset:
        seen = []

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B2", "B3", "B4", "B8"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            DummyS2Superset.seen.append(
                tuple(float(input_chw[i, 0, 0]) for i in range(4))
            )
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_rgb_subset_combined")(DummyRGB)
    registry.register("dummy_s2_superset_combined")(DummyS2Superset)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}
    band_value = {"B2": 2.0, "B3": 3.0, "B4": 4.0, "B8": 8.0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        x = np.zeros((len(sensor.bands), 2, 2), dtype=np.float32)
        for j, b in enumerate(sensor.bands):
            x[j] = band_value[b]
        return x

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x_chw, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    out_path = tmp_path / "superset_combined.npz"
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_rgb_subset_combined", "dummy_s2_superset_combined"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )

    assert out_path.exists()
    assert fetch_calls["n"] == len(spatials)
    assert DummyRGB.seen == [(4.0, 3.0, 2.0), (4.0, 3.0, 2.0)]
    assert DummyS2Superset.seen == [(2.0, 3.0, 4.0, 8.0), (2.0, 3.0, 4.0, 8.0)]


def test_export_batch_per_item_prefers_batch_inference_on_gpu(tmp_path, monkeypatch):
    class DummyBatchGPU:
        single_calls = 0
        batch_calls = 0

        def describe(self):
            return {"type": "mock", "dim": 1}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyBatchGPU.single_calls += 1
            return Embedding(
                data=np.array([float(spatial.lon)], dtype=np.float32), meta={}
            )

        def get_embeddings_batch(
            self,
            *,
            spatials,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
            device="auto",
        ):
            DummyBatchGPU.batch_calls += 1
            return [
                Embedding(data=np.array([float(s.lon)], dtype=np.float32), meta={})
                for s in spatials
            ]

    registry.register("dummy_batch_gpu_dir")(DummyBatchGPU)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()
    monkeypatch.setattr(api, "provider_factory_for_backend", lambda _b: None)

    out_dir = tmp_path / "gpu_dir_batch"
    api.export_batch(
        spatials=[
            PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
            PointBuffer(lon=1.0, lat=1.0, buffer_m=10),
        ],
        temporal=TemporalSpec.year(2022),
        models=["dummy_batch_gpu_dir"],
        out_dir=str(out_dir),
        backend="gee",
        device="cuda",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )

    assert (out_dir / "p00000.npz").exists()
    assert (out_dir / "p00001.npz").exists()
    assert DummyBatchGPU.batch_calls >= 1
    assert DummyBatchGPU.single_calls == 0


def test_export_batch_per_item_cpu_defaults_to_single_inference(tmp_path, monkeypatch):
    class DummyBatchGPUOff:
        single_calls = 0
        batch_calls = 0

        def describe(self):
            return {"type": "mock", "dim": 1}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyBatchGPUOff.single_calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

        def get_embeddings_batch(
            self,
            *,
            spatials,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
            device="auto",
        ):
            DummyBatchGPUOff.batch_calls += 1
            return [
                Embedding(data=np.array([1.0], dtype=np.float32), meta={})
                for _ in spatials
            ]

    registry.register("dummy_batch_gpu_dir_single")(DummyBatchGPUOff)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()
    monkeypatch.setattr(api, "provider_factory_for_backend", lambda _b: None)

    out_dir = tmp_path / "gpu_dir_single"
    api.export_batch(
        spatials=[
            PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
            PointBuffer(lon=1.0, lat=1.0, buffer_m=10),
        ],
        temporal=TemporalSpec.year(2022),
        models=["dummy_batch_gpu_dir_single"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )

    assert DummyBatchGPUOff.batch_calls == 0
    assert DummyBatchGPUOff.single_calls == 2


def test_export_batch_netcdf_per_item(tmp_path, monkeypatch):
    """export_batch with format='netcdf' writes .nc files with correct variables."""

    class DummyNC:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.arange(4, dtype=np.float32), meta={})

    registry.register("dummy_nc")(DummyNC)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.ones((2, 4, 4), dtype=np.float32),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    temporal = TemporalSpec.range("2021-01-01", "2021-06-01")
    sensor = SensorSpec(collection="C", bands=("B1", "B2"))

    out_dir = tmp_path / "nc_out"
    res = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_nc"],
        out_dir=str(out_dir),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        format="netcdf",
        save_inputs=True,
        save_embeddings=True,
        save_manifest=True,
    )

    assert len(res) == len(spatials)
    for i in range(len(spatials)):
        nc = out_dir / f"p{i:05d}.nc"
        assert nc.exists(), f"Missing {nc}"
        json_f = out_dir / f"p{i:05d}.json"
        assert json_f.exists(), f"Missing {json_f}"

    # Verify NetCDF contents
    import xarray as xr

    ds = xr.open_dataset(str(out_dir / "p00000.nc"))
    assert "embedding__dummy_nc" in ds.data_vars
    assert "input_chw__dummy_nc" in ds.data_vars
    assert tuple(ds["embedding__dummy_nc"].dims) == ("dim",)
    assert tuple(ds["input_chw__dummy_nc"].dims) == ("band", "y", "x")
    ds.close()


def test_export_batch_netcdf_combined(tmp_path, monkeypatch):
    """export_batch with format='netcdf' and out_path produces a combined .nc."""

    class DummyComb:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([1.0, 2.0], dtype=np.float32), meta={})

    registry.register("dummy_comb")(DummyComb)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    temporal = TemporalSpec.year(2022)
    sensor = SensorSpec(collection="C", bands=("B1",))

    out_path = tmp_path / "combined.nc"
    result = api.export_batch(
        spatials=spatials,
        temporal=temporal,
        models=["dummy_comb"],
        out_path=str(out_path),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        sensor=sensor,
        format="netcdf",
        save_inputs=True,
        save_embeddings=True,
    )

    assert out_path.exists()
    assert "nc_path" in result

    import xarray as xr

    ds = xr.open_dataset(str(out_path))
    assert "embeddings__dummy_comb" in ds.data_vars
    assert ds["embeddings__dummy_comb"].shape == (2, 2)  # (point, dim)
    ds.close()


def test_export_batch_combined_fail_on_bad_input(tmp_path, monkeypatch):
    class DummyBad:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_bad")(DummyBad)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.zeros(
            (1, 2, 2), dtype=np.float32
        ),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw",
        lambda x, *, sensor, name: {"ok": False, "report": {"issues": ["all fill"]}},
    )
    get_embedder_bundle_cached.cache_clear()

    with pytest.raises(RuntimeError, match="Input inspection failed"):
        api.export_batch(
            spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
            temporal=TemporalSpec.year(2022),
            models=["dummy_bad"],
            out_path=str(tmp_path / "bad.npz"),
            backend="gee",
            output=OutputSpec.pooled(),
            sensor=SensorSpec(collection="C", bands=("B1",)),
            save_inputs=True,
            save_embeddings=False,
            fail_on_bad_input=True,
        )


def test_export_batch_combined_partial_inputs_include_indices(tmp_path, monkeypatch):
    class DummyInputOnly:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            raise AssertionError("save_embeddings=False should skip model inference")

    registry.register("dummy_input_only")(DummyInputOnly)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    def fake_fetch(provider, *, spatial, temporal, sensor):
        if float(getattr(spatial, "lon", 0.0)) > 0.5:
            raise RuntimeError("prefetch fail")
        return np.ones((1, 2, 2), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "partial_inputs_indices.npz"
    out = api.export_batch(
        spatials=[
            PointBuffer(lon=0, lat=0, buffer_m=10),
            PointBuffer(lon=1, lat=1, buffer_m=10),
        ],
        temporal=TemporalSpec.year(2022),
        models=["dummy_input_only"],
        out_path=str(out_path),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1",)),
        save_inputs=True,
        save_embeddings=False,
        continue_on_error=True,
        show_progress=False,
    )

    assert out_path.exists()
    m = out["models"][0]
    assert m["model"] == "dummy_input_only"
    assert "inputs" in m and isinstance(m["inputs"], dict)
    assert m["inputs"]["indices"] == [0]

    with np.load(out_path) as npz:
        arr = npz[m["inputs"]["npz_key"]]
        assert arr.shape[0] == 1


def test_export_batch_prefetch_used_even_without_saving_inputs(tmp_path, monkeypatch):
    class DummyNeedInput:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            assert input_chw is not None
            return Embedding(
                data=np.array([float(np.sum(input_chw))], dtype=np.float32), meta={}
            )

    registry.register("dummy_need_input")(DummyNeedInput)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    calls = {"fetch": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        calls["fetch"] += 1
        return np.ones((2, 3, 3), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    sensor = SensorSpec(collection="C", bands=("B1", "B2"))
    out_dir = tmp_path / "out_no_inputs"
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_need_input"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=sensor,
        save_inputs=False,
        save_embeddings=True,
    )

    assert calls["fetch"] == len(spatials)
    assert (out_dir / "p00000.npz").exists()


def test_export_batch_continue_on_error_partial_manifest(tmp_path, monkeypatch):
    class DummyGood:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={"ok": True})

    class DummyBad:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            raise RuntimeError("boom")

    registry.register("dummy_good")(DummyGood)
    registry.register("dummy_bad2")(DummyBad)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_dir = tmp_path / "partial"
    res = api.export_batch(
        spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
        temporal=TemporalSpec.year(2022),
        models=["dummy_good", "dummy_bad2"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1",)),
        save_inputs=True,
        save_embeddings=True,
        continue_on_error=True,
    )

    assert len(res) == 1
    assert res[0]["status"] == "partial"
    assert any(
        m["model"] == "dummy_bad2" and m["status"] == "failed" for m in res[0]["models"]
    )
    assert (out_dir / "p00000.npz").exists()


def test_export_batch_combined_prefers_model_batch_api(tmp_path):
    class DummyBatch:
        batch_calls = 0
        single_calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyBatch.single_calls += 1
            raise RuntimeError("single path should not be used")

        def get_embeddings_batch(
            self,
            *,
            spatials,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="local",
            device="auto",
        ):
            DummyBatch.batch_calls += 1
            return [
                Embedding(data=np.array([float(i)], dtype=np.float32), meta={})
                for i in range(len(spatials))
            ]

    registry.register("dummy_batch")(DummyBatch)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "combined_batch.npz"
    mani = api.export_batch(
        spatials=[
            PointBuffer(lon=0, lat=0, buffer_m=10),
            PointBuffer(lon=1, lat=1, buffer_m=10),
        ],
        temporal=TemporalSpec.year(2022),
        models=["dummy_batch"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
    )

    assert out_path.exists()
    assert mani["status"] == "ok"
    assert DummyBatch.batch_calls == 1
    assert DummyBatch.single_calls == 0


def test_export_batch_dedup_inputs_across_models_in_file(tmp_path, monkeypatch):
    class DummyA:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyB:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_dedup_a")(DummyA)
    registry.register("dummy_dedup_b")(DummyB)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.ones((1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_dir = tmp_path / "dedup_inputs"
    api.export_batch(
        spatials=[PointBuffer(lon=0, lat=0, buffer_m=10)],
        temporal=TemporalSpec.year(2022),
        models=["dummy_dedup_a", "dummy_dedup_b"],
        out_dir=str(out_dir),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1",)),
        save_inputs=True,
        save_embeddings=True,
    )

    npz = np.load(out_dir / "p00000.npz")
    input_keys = [k for k in npz.keys() if k.startswith("input_chw__")]
    assert len(input_keys) == 1

    with open(out_dir / "p00000.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_entries = {m["model"]: m for m in manifest["models"]}
    assert model_entries["dummy_dedup_b"]["input"].get("dedup_reused") is True


def test_export_batch_resume_out_dir_skips_existing(tmp_path):
    class DummyResumeDir:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyResumeDir.calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_resume_dir")(DummyResumeDir)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    out_dir = tmp_path / "resume_dir"

    first = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_dir"],
        out_dir=str(out_dir),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )
    assert len(first) == len(spatials)
    assert DummyResumeDir.calls == len(spatials)

    second = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_dir"],
        out_dir=str(out_dir),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=False,
    )
    assert len(second) == len(spatials)
    assert DummyResumeDir.calls == len(spatials)
    assert all(bool(m.get("resume_skipped")) for m in second)


def test_export_batch_resume_out_path_skips_existing(tmp_path):
    class DummyResumeCombined:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyResumeCombined.calls += 1
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_resume_combined")(DummyResumeCombined)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "combined_resume.npz"
    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_combined"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )
    assert DummyResumeCombined.calls == len(spatials)

    skipped = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_combined"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=False,
    )
    assert bool(skipped.get("resume_skipped"))
    assert DummyResumeCombined.calls == len(spatials)


def test_export_batch_combined_saves_prefetch_checkpoint_before_inference(
    tmp_path, monkeypatch
):
    class DummyCrashAfterFetch:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            raise RuntimeError("boom-after-prefetch")

    registry.register("dummy_crash_after_fetch")(DummyCrashAfterFetch)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw",
        lambda prov, *, spatial, temporal, sensor: np.ones((2, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "prefetch_ckpt.npz"
    with pytest.raises(RuntimeError, match="boom-after-prefetch"):
        api.export_batch(
            spatials=[
                PointBuffer(lon=0, lat=0, buffer_m=10),
                PointBuffer(lon=1, lat=1, buffer_m=10),
            ],
            temporal=TemporalSpec.year(2022),
            models=["dummy_crash_after_fetch"],
            out_path=str(out_path),
            backend="gee",
            output=OutputSpec.pooled(),
            sensor=SensorSpec(collection="C", bands=("B1", "B2")),
            save_inputs=False,
            save_embeddings=True,
            show_progress=False,
        )

    assert out_path.exists()
    with open(tmp_path / "prefetch_ckpt.json", "r", encoding="utf-8") as f:
        mani = json.load(f)
    assert bool(mani.get("resume_incomplete"))
    assert mani.get("stage") == "prefetched"

    npz = np.load(out_path)
    try:
        assert any(k.startswith("__prefetch_") for k in npz.files)
    finally:
        npz.close()


def test_export_batch_combined_resume_continues_remaining_models(tmp_path, monkeypatch):
    class DummyResumeGood:
        calls = 0

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyResumeGood.calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyResumeFlaky:
        calls = 0
        fail = True

        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            DummyResumeFlaky.calls += 1
            if DummyResumeFlaky.fail:
                raise RuntimeError("flaky")
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_resume_good_ckpt")(DummyResumeGood)
    registry.register("dummy_resume_flaky_ckpt")(DummyResumeFlaky)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            return None

    fetch_calls = {"n": 0}

    def fake_fetch(provider, *, spatial, temporal, sensor):
        fetch_calls["n"] += 1
        return np.ones((2, 2, 2), dtype=np.float32)

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider())
    monkeypatch.setattr("rs_embed.providers.gee_utils.fetch_gee_patch_raw", fake_fetch)
    monkeypatch.setattr("rs_embed.providers.gee_utils.inspect_input_raw", lambda x, *, sensor, name: {"ok": True}
    )
    get_embedder_bundle_cached.cache_clear()

    out_path = tmp_path / "resume_partial.npz"
    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]

    with pytest.raises(RuntimeError, match="flaky"):
        api.export_batch(
            spatials=spatials,
            temporal=TemporalSpec.year(2022),
            models=["dummy_resume_good_ckpt", "dummy_resume_flaky_ckpt"],
            out_path=str(out_path),
            backend="gee",
            output=OutputSpec.pooled(),
            sensor=SensorSpec(collection="C", bands=("B1", "B2")),
            save_inputs=False,
            save_embeddings=True,
            show_progress=False,
        )

    first_good_calls = DummyResumeGood.calls
    first_fetch_calls = fetch_calls["n"]
    assert first_good_calls == len(spatials)
    assert first_fetch_calls == len(spatials)

    DummyResumeFlaky.fail = False
    result = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_resume_good_ckpt", "dummy_resume_flaky_ckpt"],
        out_path=str(out_path),
        backend="gee",
        output=OutputSpec.pooled(),
        sensor=SensorSpec(collection="C", bands=("B1", "B2")),
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=False,
    )

    assert result["status"] == "ok"
    assert bool(result.get("resume_incomplete")) is False
    assert DummyResumeGood.calls == first_good_calls
    assert fetch_calls["n"] == first_fetch_calls
    assert any(m.get("model") == "dummy_resume_good_ckpt" for m in result["models"])
    assert any(m.get("model") == "dummy_resume_flaky_ckpt" for m in result["models"])

    npz = np.load(out_path)
    try:
        assert "embeddings__dummy_resume_good_ckpt" in npz.files
        assert "embeddings__dummy_resume_flaky_ckpt" in npz.files
        assert not any(k.startswith("__prefetch_") for k in npz.files)
    finally:
        npz.close()


def test_export_batch_progress_updates_point_and_model_bars(tmp_path, monkeypatch):
    class DummyProgressModel:
        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([3.0], dtype=np.float32), meta={})

    registry.register("dummy_progress")(DummyProgressModel)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    state = {}

    class _FakeProgress:
        def __init__(self, *, total: int, desc: str):
            state[desc] = {
                "total": int(total),
                "updates": 0,
                "calls": [],
                "closed": False,
            }
            self._desc = desc

        def update(self, n: int = 1):
            n_i = int(n)
            state[self._desc]["updates"] += n_i
            state[self._desc]["calls"].append(n_i)

        def close(self):
            state[self._desc]["closed"] = True

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _FakeProgress(
            total=total, desc=desc
        ),
    )

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_progress"],
        out_dir=str(tmp_path / "progress"),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        chunk_size=1,
        show_progress=True,
    )

    assert state["export_batch"]["total"] == len(spatials)
    assert state["export_batch"]["updates"] == len(spatials)
    assert state["export_batch"]["closed"] is True
    assert state["infer[dummy_progress]"]["total"] == len(spatials)
    assert state["infer[dummy_progress]"]["updates"] == len(spatials)
    assert state["infer[dummy_progress]"]["closed"] is True


def test_export_batch_combined_progress_updates_model_inference(tmp_path, monkeypatch):
    class DummyBatchProgressModel:
        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

        def get_embeddings_batch(
            self, *, spatials, temporal, sensor, output, backend, device="auto"
        ):
            return [
                Embedding(data=np.array([1.0], dtype=np.float32), meta={})
                for _ in spatials
            ]

    class DummySingleProgressModel:
        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
        ):
            return Embedding(data=np.array([2.0], dtype=np.float32), meta={})

    registry.register("dummy_batch_progress")(DummyBatchProgressModel)
    registry.register("dummy_single_progress")(DummySingleProgressModel)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    state = {}

    class _FakeProgress:
        def __init__(self, *, total: int, desc: str):
            state[desc] = {
                "total": int(total),
                "updates": 0,
                "calls": [],
                "closed": False,
            }
            self._desc = desc

        def update(self, n: int = 1):
            n_i = int(n)
            state[self._desc]["updates"] += n_i
            state[self._desc]["calls"].append(n_i)

        def close(self):
            state[self._desc]["closed"] = True

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _FakeProgress(
            total=total, desc=desc
        ),
    )

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_batch_progress", "dummy_single_progress"],
        out_path=str(tmp_path / "combined_progress.npz"),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        chunk_size=1,
        show_progress=True,
    )

    assert state["export_batch[combined]"]["total"] == 2
    assert state["export_batch[combined]"]["updates"] == 2
    assert state["export_batch[combined]"]["closed"] is True
    assert state["infer[dummy_batch_progress]"]["total"] == len(spatials)
    assert state["infer[dummy_batch_progress]"]["updates"] == len(spatials)
    assert state["infer[dummy_batch_progress]"]["calls"] == [1, 1]
    assert state["infer[dummy_batch_progress]"]["closed"] is True
    assert state["infer[dummy_single_progress]"]["total"] == len(spatials)
    assert state["infer[dummy_single_progress]"]["updates"] == len(spatials)
    assert state["infer[dummy_single_progress]"]["calls"] == [1, 1]
    assert state["infer[dummy_single_progress]"]["closed"] is True


def test_export_batch_combined_progress_fills_on_model_init_failure(
    tmp_path, monkeypatch
):
    class DummyInitFailModel:
        def __init__(self):
            raise RuntimeError("init failed")

    registry.register("dummy_init_fail")(DummyInitFailModel)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    state = {}

    class _FakeProgress:
        def __init__(self, *, total: int, desc: str):
            state[desc] = {
                "total": int(total),
                "updates": 0,
                "calls": [],
                "closed": False,
            }
            self._desc = desc

        def update(self, n: int = 1):
            n_i = int(n)
            state[self._desc]["updates"] += n_i
            state[self._desc]["calls"].append(n_i)

        def close(self):
            state[self._desc]["closed"] = True

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _FakeProgress(
            total=total, desc=desc
        ),
    )

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    result = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_init_fail"],
        out_path=str(tmp_path / "combined_progress_init_fail.npz"),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        continue_on_error=True,
        show_progress=True,
    )

    assert result["status"] == "failed"
    assert state["infer[dummy_init_fail]"]["total"] == len(spatials)
    assert state["infer[dummy_init_fail]"]["updates"] == len(spatials)
    assert state["infer[dummy_init_fail]"]["closed"] is True


def test_export_batch_combined_embedder_without_input_chw_kwarg(tmp_path):
    class DummyNoInputKwarg:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self, *, spatial, temporal, sensor, output, backend, device="auto"
        ):
            DummyNoInputKwarg.calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_no_input_kwarg")(DummyNoInputKwarg)

    import rs_embed.api as api

    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]
    out_path = tmp_path / "combined_no_input_kwarg.npz"

    result = api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_no_input_kwarg"],
        out_path=str(out_path),
        backend="local",
        output=OutputSpec.pooled(),
        save_inputs=False,
        save_embeddings=True,
        show_progress=False,
    )

    assert out_path.exists()
    assert result["status"] == "ok"
    assert DummyNoInputKwarg.calls == len(spatials)
