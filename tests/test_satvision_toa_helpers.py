import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.embedders.onthefly_satvision_toa import (
    _coerce_fetch_result,
    _normalize_satvision_toa_input,
    _normalize_indices,
    SatVisionTOAEmbedder,
)


def test_normalize_indices_supports_negative():
    assert _normalize_indices((12, 13, -1, -2, 99), 14) == (12, 13)


def test_normalize_satvision_raw_reflectance_and_thermal():
    raw = np.full((14, 4, 4), 5000.0, dtype=np.float32)
    raw[12] = 275.0
    raw[13] = 275.0

    y = _normalize_satvision_toa_input(
        raw,
        mode="raw",
        reflectance_indices=(0, 1, 2, 3, 4, 6),
        emissive_indices=(5, 7, 8, 9, 10, 11, 12, 13),
        reflectance_divisor=10000.0,
        emissive_mins=(175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0, 175.0),
        emissive_maxs=(375.0, 375.0, 375.0, 375.0, 375.0, 375.0, 375.0, 375.0),
    )

    # 5000/10000 => 0.5 for reflectance
    assert np.allclose(y[0], 0.5)
    assert np.allclose(y[6], 0.5)
    # (275-175)/(375-175) => 0.5 for thermal
    assert np.allclose(y[12], 0.5)
    assert np.allclose(y[13], 0.5)


def test_satvision_runtime_requires_14_bands(monkeypatch):
    emb = SatVisionTOAEmbedder()

    class _FakeSensor:
        collection = "TEST/COLL"
        bands = tuple("B" + str(i) for i in range(10))
        scale_m = 500
        cloudy_pct = 30
        fill_value = 0.0
        composite = "median"

    with pytest.raises(ModelError, match="requires exactly"):
        emb._resolve_runtime(sensor=_FakeSensor(), device="cpu")


def test_satvision_default_sensor_has_14_bands():
    emb = SatVisionTOAEmbedder()
    ss = emb._default_sensor()
    assert len(ss.bands) == 14


def test_coerce_fetch_result_supports_array_and_tuple():
    raw = np.ones((14, 4, 4), dtype=np.float32)
    arr0, meta0 = _coerce_fetch_result(raw)
    assert arr0.shape == (14, 4, 4)
    assert meta0["fallback_used"] is False
    assert meta0["already_unit_scaled"] is False

    arr1, meta1 = _coerce_fetch_result((raw, {"fallback_used": True, "already_unit_scaled": True}))
    assert arr1.shape == (14, 4, 4)
    assert meta1["fallback_used"] is True
    assert meta1["already_unit_scaled"] is True


def test_satvision_single_forces_unit_norm_when_fetch_is_unit_scaled(monkeypatch):
    import rs_embed.embedders.onthefly_satvision_toa as sv

    emb = SatVisionTOAEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        emb,
        "_resolve_runtime",
        lambda **kw: {
            "model": object(),
            "model_meta": {},
            "device": "cpu",
            "model_id": "nasa-cisto-data-science-group/satvision-toa-giant-patch8-window8-128",
            "image_size": 8,
            "in_chans": 14,
            "norm_mode": "raw",
            "reflectance_indices": (0, 1, 2, 3, 4, 6),
            "emissive_indices": (5, 7, 8, 9, 10, 11, 12, 13),
            "reflectance_divisor": 10000.0,
            "emissive_mins": (175.0,) * 8,
            "emissive_maxs": (375.0,) * 8,
        },
    )
    monkeypatch.setattr(
        sv,
        "_fetch_toa_raw_chw_from_gee",
        lambda provider, spatial, temporal, sensor: (
            np.full((14, 8, 8), 0.5, dtype=np.float32),
            {"fallback_used": True, "already_unit_scaled": True},
        ),
    )
    seen = {}

    def _fake_prepare(raw_chw, **kw):
        seen["norm_mode"] = kw["norm_mode"]
        return np.asarray(raw_chw, dtype=np.float32)

    monkeypatch.setattr(emb, "_prepare_input", _fake_prepare)
    monkeypatch.setattr(
        sv,
        "_satvision_forward_batch",
        lambda model, x_chw_batch, **kw: (
            [np.full((4,), 1.0, dtype=np.float32)],
            {"tokens_kind": "pooled"},
        ),
    )

    sensor = SensorSpec(
        collection="TEST/COLL",
        bands=tuple(f"B{i}" for i in range(14)),
        scale_m=500,
    )
    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2020-01-01", "2020-01-31"),
        sensor=sensor,
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert seen["norm_mode"] == "unit"
    assert out.meta["fallback_used"] is True
    assert out.meta["norm_mode_effective"] == "unit"
