import numpy as np
import pytest
import xarray as xr

from rs_embed.core import registry
from rs_embed.core.specs import BBox, SensorSpec
from rs_embed.core.embedding import Embedding
from rs_embed.tools.serialization import (
    sanitize_key as _sanitize_key,
    sha1 as _sha1,
    jsonable as _jsonable,
    utc_ts as _utc_ts,
    embedding_to_numpy as _embedding_to_numpy,
)
from rs_embed.tools.model_defaults import (
    default_sensor_for_model as _default_sensor_for_model,
    resolve_sensor_for_model as _resolve_sensor_for_model,
)


# ── fixture to isolate registry ────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()


# ══════════════════════════════════════════════════════════════════════
# _sanitize_key
# ══════════════════════════════════════════════════════════════════════


def test_sanitize_key_slashes_spaces():
    assert _sanitize_key("foo/bar baz") == "foo_bar_baz"


def test_sanitize_key_all_underscores():
    assert _sanitize_key("___") == "item"


def test_sanitize_key_empty():
    assert _sanitize_key("") == "item"


def test_sanitize_key_already_clean():
    assert _sanitize_key("hello_world_123") == "hello_world_123"


def test_sanitize_key_special_chars():
    assert _sanitize_key("a@b#c$d") == "a_b_c_d"


def test_sanitize_key_consecutive_special():
    assert _sanitize_key("a---b///c") == "a_b_c"


# ══════════════════════════════════════════════════════════════════════
# _sha1
# ══════════════════════════════════════════════════════════════════════


def test_sha1_deterministic():
    arr = np.arange(10, dtype=np.int32)
    h1 = _sha1(arr)
    h2 = _sha1(arr.copy())
    assert h1 == h2
    assert len(h1) == 40


def test_sha1_different_arrays():
    a = np.zeros(10, dtype=np.float32)
    b = np.ones(10, dtype=np.float32)
    assert _sha1(a) != _sha1(b)


def test_sha1_different_dtypes():
    a = np.zeros(10, dtype=np.float32)
    b = np.zeros(10, dtype=np.float64)
    assert _sha1(a) != _sha1(b)


def test_sha1_large_array():
    arr = np.zeros(10_000_000, dtype=np.float32)
    h = _sha1(arr, max_bytes=1000)
    assert len(h) == 40


# ══════════════════════════════════════════════════════════════════════
# _jsonable
# ══════════════════════════════════════════════════════════════════════


def test_jsonable_primitives():
    assert _jsonable(None) is None
    assert _jsonable(42) == 42
    assert _jsonable(3.14) == 3.14
    assert _jsonable("hello") == "hello"
    assert _jsonable(True) is True


def test_jsonable_list_tuple():
    assert _jsonable([1, "a"]) == [1, "a"]
    assert _jsonable((1, "a")) == [1, "a"]


def test_jsonable_dict():
    assert _jsonable({"k": 1}) == {"k": 1}


def test_jsonable_numpy_scalar():
    assert _jsonable(np.int64(5)) == 5
    assert _jsonable(np.float32(2.5)) == pytest.approx(2.5)
    assert _jsonable(np.bool_(True)) is True


def test_jsonable_numpy_array():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    out = _jsonable(arr)
    assert out["__ndarray__"] is True
    assert out["shape"] == [2]
    assert out["min"] == 1.0
    assert out["max"] == 2.0


def test_jsonable_empty_array():
    arr = np.array([], dtype=np.float32)
    out = _jsonable(arr)
    assert out["shape"] == [0]
    assert out["min"] is None


def test_jsonable_dataclass():
    bbox = BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)
    out = _jsonable(bbox)
    assert out["minlon"] == 0.0
    assert out["maxlat"] == 1.0
    assert out["crs"] == "EPSG:4326"


def test_jsonable_xarray():
    da = xr.DataArray(np.zeros((2, 3)))
    out = _jsonable(da)
    assert out.get("__xarray__") is True
    assert out["shape"] == [2, 3]


def test_jsonable_nested():
    data = {"arr": np.array([1]), "specs": [BBox(0, 0, 1, 1)]}
    out = _jsonable(data)
    assert out["arr"]["__ndarray__"] is True
    assert out["specs"][0]["minlon"] == 0


def test_jsonable_fallback_repr():
    out = _jsonable(object())
    assert isinstance(out, str)


# ══════════════════════════════════════════════════════════════════════
# _utc_ts
# ══════════════════════════════════════════════════════════════════════


def test_utc_ts_format():
    ts = _utc_ts()
    assert ts.endswith("Z")
    assert "T" in ts


# ══════════════════════════════════════════════════════════════════════
# _embedding_to_numpy
# ══════════════════════════════════════════════════════════════════════


def test_embedding_to_numpy_ndarray():
    e = Embedding(data=np.array([1.0, 2.0], dtype=np.float64), meta={})
    out = _embedding_to_numpy(e)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 2.0])


def test_embedding_to_numpy_xarray():
    da = xr.DataArray(np.array([[1.0, 2.0]], dtype=np.float64))
    e = Embedding(data=da, meta={})
    out = _embedding_to_numpy(e)
    assert out.dtype == np.float32
    assert out.shape == (1, 2)


# ══════════════════════════════════════════════════════════════════════
# _default_sensor_for_model
# ══════════════════════════════════════════════════════════════════════


def test_default_sensor_for_model_precomputed():
    @registry.register("precomputed_test")
    class DummyPrecomputed:
        def describe(self):
            return {"type": "precomputed"}

    assert _default_sensor_for_model("precomputed_test") is None


def test_default_sensor_for_model_inputs_dict():
    @registry.register("onthefly_test")
    class DummyOnTheFly:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "COLL", "bands": ["B1", "B2"]},
                "defaults": {
                    "scale_m": 20,
                    "cloudy_pct": 5,
                    "composite": "mosaic",
                    "fill_value": 1.0,
                },
            }

    sensor = _default_sensor_for_model("onthefly_test")
    assert isinstance(sensor, SensorSpec)
    assert sensor.collection == "COLL"
    assert sensor.bands == ("B1", "B2")
    assert sensor.scale_m == 20
    assert sensor.cloudy_pct == 5
    assert sensor.composite == "mosaic"
    assert sensor.fill_value == 1.0


def test_default_sensor_for_model_s2_modality():
    @registry.register("s2_multi")
    class DummyS2:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {
                    "s2_sr": {
                        "collection": "COPERNICUS/S2_SR_HARMONIZED",
                        "bands": ["B4", "B3", "B2"],
                    },
                },
                "defaults": {},
            }

    sensor = _default_sensor_for_model("s2_multi")
    assert sensor is not None
    assert sensor.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert sensor.bands == ("B4", "B3", "B2")


def test_default_sensor_for_model_requested_modality():
    @registry.register("multi_modal")
    class DummyMulti:
        def describe(self):
            return {
                "type": "on_the_fly",
                "modalities": {
                    "s2": {
                        "collection": "COPERNICUS/S2_SR_HARMONIZED",
                        "bands": ["B4", "B3", "B2"],
                    },
                    "s1": {
                        "collection": "COPERNICUS/S1_GRD_FLOAT",
                        "bands": ["VV", "VH"],
                    },
                },
                "defaults": {"modality": "s2", "use_float_linear": True},
            }

    sensor = _default_sensor_for_model("multi_modal", modality="s1")
    assert sensor is not None
    assert sensor.modality == "s1"
    assert sensor.collection == "COPERNICUS/S1_GRD_FLOAT"
    assert sensor.bands == ("VV", "VH")


def test_resolve_sensor_for_model_merges_modality():
    @registry.register("multi_modal_resolve")
    class DummyMulti:
        def describe(self):
            return {
                "type": "on_the_fly",
                "modalities": {
                    "s1": {
                        "collection": "COPERNICUS/S1_GRD_FLOAT",
                        "bands": ["VV", "VH"],
                    }
                },
                "defaults": {"modality": "s1"},
            }

    sensor = SensorSpec(collection="COPERNICUS/S1_GRD", bands=("VV", "VH"))
    out = _resolve_sensor_for_model(
        "multi_modal_resolve", sensor=sensor, modality="s1"
    )
    assert out is not None
    assert out.modality == "s1"
    assert out.collection == "COPERNICUS/S1_GRD"


def test_default_sensor_for_model_provider_default_block():
    @registry.register("dofa_style")
    class DummyDofa:
        def describe(self):
            return {
                "type": "on_the_fly",
                "inputs": {
                    "provider_default": {
                        "collection": "COPERNICUS/S2_SR_HARMONIZED",
                        "bands": ["B1", "B2", "B3"],
                    }
                },
                "defaults": {"scale_m": 30, "cloudy_pct": 40, "composite": "mean"},
            }

    sensor = _default_sensor_for_model("dofa_style")
    assert sensor is not None
    assert sensor.collection == "COPERNICUS/S2_SR_HARMONIZED"
    assert sensor.bands == ("B1", "B2", "B3")
    assert sensor.scale_m == 30
    assert sensor.cloudy_pct == 40
    assert sensor.composite == "mean"


def test_default_sensor_for_model_input_bands_key():
    @registry.register("prithvi_style")
    class DummyPrithvi:
        def describe(self):
            return {
                "type": "onthefly",
                "input_bands": ["B2", "B3", "B4", "B8", "B11", "B12"],
                "defaults": {"scale_m": 10},
            }

    sensor = _default_sensor_for_model("prithvi_style")
    assert sensor is not None
    assert sensor.bands == ("B2", "B3", "B4", "B8", "B11", "B12")


def test_default_sensor_for_model_no_info():
    @registry.register("mystery")
    class DummyMystery:
        def describe(self):
            return {"type": "onthefly"}

    assert _default_sensor_for_model("mystery") is None


# ══════════════════════════════════════════════════════════════════════
# _embedding_to_numpy — generic fallback
# ══════════════════════════════════════════════════════════════════════


def test_embedding_to_numpy_generic_list():
    """Generic array-like (not ndarray or xarray) goes through np.asarray fallback."""
    e = Embedding(data=[1.0, 2.0, 3.0], meta={})
    out = _embedding_to_numpy(e)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_embedding_to_numpy_preserves_shape():
    arr = np.zeros((3, 4, 4), dtype=np.float64)
    e = Embedding(data=arr, meta={})
    out = _embedding_to_numpy(e)
    assert out.shape == (3, 4, 4)
    assert out.dtype == np.float32
