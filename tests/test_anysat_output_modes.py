import sys
import types
from contextlib import nullcontext

import numpy as np

try:
    import xarray  # noqa: F401
except ImportError:
    # Environments without xarray: a minimal stand-in is enough for these tests.
    # (A SimpleNamespace has no __spec__, so only install it when the real
    # package is genuinely unavailable — otherwise it poisons later imports
    # that probe sys.modules, e.g. torch._dynamo's find_spec calls.)

    class _FakeDataArray:
        def __init__(self, data, dims=None, coords=None, name=None, attrs=None):
            self.data = np.asarray(data, dtype=np.float32)
            self.dims = dims
            self.coords = coords
            self.name = name
            self.attrs = attrs or {}

        def __array__(self, dtype=None):
            return np.asarray(self.data, dtype=dtype)

    sys.modules["xarray"] = types.SimpleNamespace(DataArray=_FakeDataArray)

from rs_embed.core.specs import BBox, OutputSpec, TemporalSpec
from rs_embed.embedders.onthefly_anysat import AnySatEmbedder


def _bbox():
    return BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)


class _FakeAnySatModel:
    def __init__(self):
        self.outputs = []

    def __call__(self, _x, *, patch_size, output):
        self.outputs.append((int(patch_size), str(output)))
        if output == "dense":
            return _FakeTensor(np.arange(1 * 4 * 4 * 12, dtype=np.float32).reshape(1, 4, 4, 12))
        if output == "patch":
            return _FakeTensor(np.arange(1 * 2 * 2 * 6, dtype=np.float32).reshape(1, 2, 2, 6))
        if output == "tile":
            return _FakeTensor(np.arange(1 * 9, dtype=np.float32).reshape(1, 9))
        raise AssertionError(f"unexpected output mode: {output}")


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    @property
    def ndim(self):
        return self._arr.ndim

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, item):
        return _FakeTensor(self._arr[item])

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _patch_fake_torch(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(no_grad=lambda: nullcontext()),
    )


def test_anysat_grid_defaults_to_dense(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    model = _FakeAnySatModel()
    _patch_fake_torch(monkeypatch)
    monkeypatch.setenv("RS_EMBED_ANYSAT_FRAMES", "2")
    monkeypatch.setattr(
        anysat,
        "_load_anysat",
        lambda **kwargs: (model, {"loaded_from": "test"}, "cpu"),
    )
    monkeypatch.setattr(
        anysat,
        "_prepare_anysat_s2_input",
        lambda raw_tchw, **kwargs: {"shape": tuple(raw_tchw.shape)},
    )

    out = emb.get_embedding(
        spatial=_bbox(),
        temporal=TemporalSpec.range("2020-01-01", "2020-03-01"),
        sensor=None,
        output=OutputSpec.grid(),
        backend="auto",
        input_chw=np.zeros((2, 10, 8, 8), dtype=np.float32),
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert model.outputs == [(10, "dense")]
    assert arr.shape == (12, 4, 4)
    assert out.meta["grid_kind"] == "dense_subpatch"
    assert out.meta["grid_hw"] == (4, 4)
    assert out.meta["feature_mode"] == "dense"
    assert out.meta["dense_output_hw"] == (4, 4)


def test_anysat_pooled_keeps_patch_pooling(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    model = _FakeAnySatModel()
    _patch_fake_torch(monkeypatch)
    monkeypatch.setenv("RS_EMBED_ANYSAT_FRAMES", "2")
    monkeypatch.setattr(
        anysat,
        "_load_anysat",
        lambda **kwargs: (model, {"loaded_from": "test"}, "cpu"),
    )
    monkeypatch.setattr(
        anysat,
        "_prepare_anysat_s2_input",
        lambda raw_tchw, **kwargs: {"shape": tuple(raw_tchw.shape)},
    )

    out = emb.get_embedding(
        spatial=_bbox(),
        temporal=TemporalSpec.range("2020-01-01", "2020-03-01"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=np.zeros((2, 10, 8, 8), dtype=np.float32),
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert model.outputs == [(10, "patch")]
    assert arr.shape == (6,)
    assert out.meta["pooling"] == "patch_mean"
    assert out.meta["feature_mode"] == "patch"
    assert out.meta["patch_output_hw"] == (2, 2)


def test_anysat_grid_feature_mode_can_fall_back_to_patch(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    model = _FakeAnySatModel()
    _patch_fake_torch(monkeypatch)
    monkeypatch.setenv("RS_EMBED_ANYSAT_FRAMES", "2")
    monkeypatch.setattr(
        anysat,
        "_load_anysat",
        lambda **kwargs: (model, {"loaded_from": "test"}, "cpu"),
    )
    monkeypatch.setattr(
        anysat,
        "_prepare_anysat_s2_input",
        lambda raw_tchw, **kwargs: {"shape": tuple(raw_tchw.shape)},
    )

    out = emb.get_embedding(
        spatial=_bbox(),
        temporal=TemporalSpec.range("2020-01-01", "2020-03-01"),
        sensor=None,
        output=OutputSpec.grid(),
        backend="auto",
        input_chw=np.zeros((2, 10, 8, 8), dtype=np.float32),
        model_config={"grid_feature_mode": "patch"},
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert model.outputs == [(10, "patch")]
    assert arr.shape == (6, 2, 2)
    assert out.meta["grid_kind"] == "patch_tokens"
    assert out.meta["feature_mode"] == "patch"


def test_anysat_pooled_source_can_use_tile(monkeypatch):
    import rs_embed.embedders.onthefly_anysat as anysat

    emb = AnySatEmbedder()
    model = _FakeAnySatModel()
    _patch_fake_torch(monkeypatch)
    monkeypatch.setenv("RS_EMBED_ANYSAT_FRAMES", "2")
    monkeypatch.setattr(
        anysat,
        "_load_anysat",
        lambda **kwargs: (model, {"loaded_from": "test"}, "cpu"),
    )
    monkeypatch.setattr(
        anysat,
        "_prepare_anysat_s2_input",
        lambda raw_tchw, **kwargs: {"shape": tuple(raw_tchw.shape)},
    )

    out = emb.get_embedding(
        spatial=_bbox(),
        temporal=TemporalSpec.range("2020-01-01", "2020-03-01"),
        sensor=None,
        output=OutputSpec.pooled("max"),
        backend="auto",
        input_chw=np.zeros((2, 10, 8, 8), dtype=np.float32),
        model_config={"pooled_source": "tile"},
    )

    arr = np.asarray(out.data, dtype=np.float32)
    assert model.outputs == [(10, "tile")]
    assert arr.shape == (9,)
    np.testing.assert_allclose(arr, np.arange(9, dtype=np.float32))
    assert out.meta["pooling"] == "tile_native"
    assert out.meta["requested_pooling"] == "max"
    assert out.meta["pooled_source"] == "tile"
