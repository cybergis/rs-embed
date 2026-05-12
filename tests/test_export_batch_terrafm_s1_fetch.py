import threading

import numpy as np

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.core.types import ExportConfig, ExportTarget, FetchResult
from rs_embed.embedders.base import EmbedderBase
from rs_embed.pipelines.point_payload import build_one_point_payload
from rs_embed.tools.export_requests import ExportModelRequest
from rs_embed.tools.runtime import get_embedder_bundle_cached


def test_export_batch_prefetch_preserves_terrafm_s1_sensor_fields(tmp_path, monkeypatch):
    class DummyTerraFMS1(EmbedderBase):
        model_name = "dummy_terrafm_s1"
        seen_sensor = None
        seen_input = None

        def describe(self):
            return {
                "type": "on_the_fly",
                "inputs": {
                    "s1": {
                        "collection": "COPERNICUS/S1_GRD_FLOAT",
                        "bands": ["VV", "VH"],
                    }
                },
                "modalities": {
                    "s1": {
                        "collection": "COPERNICUS/S1_GRD_FLOAT",
                        "bands": ["VV", "VH"],
                        "defaults": {
                            "use_float_linear": True,
                            "s1_require_iw": True,
                            "s1_relax_iw_on_empty": True,
                        },
                    }
                },
            }

        def fetch_input(self, provider, *, spatial, temporal, sensor):
            _ = provider, spatial, temporal
            DummyTerraFMS1.seen_sensor = sensor
            assert sensor.modality == "s1"
            assert sensor.collection == "COPERNICUS/S1_GRD_FLOAT"
            assert sensor.bands == ("VV", "VH")
            assert sensor.orbit == "ASCENDING"
            assert sensor.use_float_linear is True
            assert sensor.s1_require_iw is True
            assert sensor.s1_relax_iw_on_empty is False
            data = np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                ],
                dtype=np.float32,
            )
            return FetchResult(data=data, meta={"path": "custom_s1"})

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
            fetch_meta=None,
        ):
            _ = spatial, temporal, sensor, output, backend, device
            DummyTerraFMS1.seen_input = np.asarray(input_chw, dtype=np.float32)
            assert fetch_meta == {"path": "custom_s1"}
            return Embedding(data=np.array([float(np.sum(input_chw))], dtype=np.float32), meta={})

    registry.register("dummy_terrafm_s1")(DummyTerraFMS1)

    import rs_embed.api as api

    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass

        def ensure_ready(self):
            return None

    def fake_fetch(provider, *, spatial, temporal, sensor):
        _ = provider, spatial, temporal, sensor
        raise AssertionError("generic fetch path should not be used for TerraFM S1 export")

    monkeypatch.setattr(
        "rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: DummyProvider()
    )
    monkeypatch.setattr("rs_embed.providers.fetch.fetch_sensor_patch_chw", fake_fetch)
    monkeypatch.setattr(
        "rs_embed.providers.fetch.inspect_fetch_result",
        lambda x_chw, *, sensor, name: {"ok": True, "sensor": {"bands": list(sensor.bands)}},
    )
    get_embedder_bundle_cached.cache_clear()

    sensor = SensorSpec(
        collection="COPERNICUS/S1_GRD_FLOAT",
        bands=("VV", "VH"),
        modality="s1",
        orbit="ASCENDING",
        use_float_linear=True,
        s1_require_iw=True,
        s1_relax_iw_on_empty=False,
    )
    out_dir = tmp_path / "terrafm_s1_export"
    manifests = api.export_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=10)],
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=[ExportModelRequest(name="dummy_terrafm_s1", sensor=sensor)],
        target=ExportTarget.per_item(str(out_dir)),
        config=ExportConfig(save_inputs=True, save_embeddings=True, show_progress=False),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
    )

    manifest = manifests[0] if isinstance(manifests, list) else manifests
    npz_path = out_dir / "p00000.npz"
    with np.load(npz_path, allow_pickle=False) as bundle:
        arr = bundle["input_chw__dummy_terrafm_s1"]
        np.testing.assert_allclose(arr, DummyTerraFMS1.seen_input)
    assert manifest["models"][0]["fetch_meta"] == {"path": "custom_s1"}
    assert DummyTerraFMS1.seen_sensor is not None


def test_build_one_point_payload_fallback_uses_embedder_fetch_input(monkeypatch):
    class DummyFallbackS1(EmbedderBase):
        model_name = "dummy_fallback_s1"

        def describe(self):
            return {"type": "on_the_fly"}

        def fetch_input(self, provider, *, spatial, temporal, sensor):
            _ = provider, spatial, temporal
            assert sensor.modality == "s1"
            data = np.full((2, 2, 2), 7.0, dtype=np.float32)
            return FetchResult(data=data, meta={"source": "embedder.fetch_input"})

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
            fetch_meta=None,
        ):
            _ = spatial, temporal, sensor, output, backend, device
            np.testing.assert_allclose(input_chw, np.full((2, 2, 2), 7.0, dtype=np.float32))
            assert fetch_meta == {"source": "embedder.fetch_input"}
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    class DummyProvider:
        def ensure_ready(self):
            return None

    embedder = DummyFallbackS1()

    monkeypatch.setattr(
        "rs_embed.pipelines.point_payload.get_embedder_bundle_cached",
        lambda *args, **kwargs: (embedder, threading.Lock()),
    )

    arrays, manifest = build_one_point_payload(
        point_index=0,
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_fallback_s1"],
        backend="gee",
        resolved_backend={"dummy_fallback_s1": "gee"},
        device="cpu",
        output=OutputSpec.pooled(),
        resolved_sensor={
            "dummy_fallback_s1": SensorSpec(
                collection="COPERNICUS/S1_GRD_FLOAT",
                bands=("VV", "VH"),
                modality="s1",
            )
        },
        resolved_model_config={"dummy_fallback_s1": None},
        model_type={"dummy_fallback_s1": "onthefly"},
        inputs_cache={},
        input_reports={},
        prefetch_errors={},
        pass_input_into_embedder=True,
        config=type(
            "Cfg",
            (),
            {
                "save_inputs": True,
                "save_embeddings": True,
                "fail_on_bad_input": False,
                "continue_on_error": False,
                "max_retries": 0,
                "retry_backoff_s": 0.0,
            },
        )(),
        provider_factory=DummyProvider,
        fetch_fn=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("generic fetch_fn should not be used when embedder.fetch_input exists")
        ),
        inspect_fn=lambda x_chw, *, sensor, name: {"ok": True, "name": name},
        fetch_meta_cache={},
    )

    np.testing.assert_allclose(arrays["input_chw__dummy_fallback_s1"], np.full((2, 2, 2), 7.0))
    assert manifest["models"][0]["fetch_meta"] == {"source": "embedder.fetch_input"}
