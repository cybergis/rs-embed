"""Aurora-specific behavior: lattice geometry, temporal snapping, channel
layout, encoder-only output semantics, and config parsing.

Generic conventions (capabilities, describe schema, temporal_to_range usage)
are covered by the contract tests. Live GEE/ARCO/weights tests are gated
behind RS_EMBED_LIVE_AURORA=1.
"""

from __future__ import annotations

import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

import rs_embed.embedders.onthefly_aurora as oa
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders.onthefly_aurora import (
    _ATMOS_LEVELS,
    _N_CHANNELS,
    AuroraEmbedder,
    _lattice_window,
    _tokens_to_level_grid,
    _window_from_global,
    _window_roi,
    resolve_time_pair,
)

_CHAMPAIGN = PointBuffer(lon=-88.2, lat=40.1, buffer_m=2048)
_T_SUMMER = TemporalSpec.range("2022-06-01", "2022-09-01")


# ---------------------------------------------------------------------------
# Temporal snapping
# ---------------------------------------------------------------------------


def test_time_pair_snaps_to_last_6h_boundary_before_range_end():
    t_prev, t_cur = resolve_time_pair(_T_SUMMER)
    assert t_cur == datetime(2022, 8, 31, 18, 0)
    assert t_prev == datetime(2022, 8, 31, 12, 0)


def test_time_pair_midnight_end_steps_back_a_full_boundary():
    # end 00:00 is exclusive → the last boundary strictly before it is 18:00.
    t_prev, t_cur = resolve_time_pair(TemporalSpec.range("2022-01-01", "2022-01-02"))
    assert t_cur == datetime(2022, 1, 1, 18, 0)
    assert t_prev == datetime(2022, 1, 1, 12, 0)


def test_time_pair_year_mode():
    _, t_cur = resolve_time_pair(TemporalSpec.year(2021))
    assert t_cur == datetime(2021, 12, 31, 18, 0)


def test_time_pair_none_uses_package_default_with_warning():
    with pytest.warns(UserWarning, match="package default window"):
        _, t_cur = resolve_time_pair(None)
    assert t_cur == datetime(2022, 8, 31, 18, 0)


# ---------------------------------------------------------------------------
# Lattice window geometry
# ---------------------------------------------------------------------------


def test_lattice_window_snaps_and_centers():
    win = _lattice_window(_CHAMPAIGN)
    assert win.n == 32 and win.grid_deg == 0.25
    # center lat 40.1 → snap 40.0 → NW point 40.0 + 16*0.25
    assert win.lat_max == pytest.approx(44.0)
    # center lon -88.2 → 271.8 → snap 271.75 → west point 271.75 - 16*0.25
    assert win.lon_min == pytest.approx(267.75)
    assert win.lat_min == pytest.approx(44.0 - 31 * 0.25)
    assert win.lon_max == pytest.approx(267.75 + 31 * 0.25)


def test_lattice_window_clamps_at_pole_and_lon_seam():
    pole = _lattice_window(PointBuffer(lon=10.0, lat=89.9, buffer_m=2048))
    assert pole.lat_max == pytest.approx(90.0)
    greenwich = _lattice_window(PointBuffer(lon=0.05, lat=51.5, buffer_m=2048))
    assert greenwich.lon_min == pytest.approx(0.0)
    west = _lattice_window(PointBuffer(lon=-0.4, lat=51.5, buffer_m=2048))
    assert west.lon_min == pytest.approx(352.0)
    assert west.lon_max < 360.0


def test_window_roi_point_is_near_center_and_tiny():
    win = _lattice_window(_CHAMPAIGN)
    y0, y1, x0, x1 = _window_roi(_CHAMPAIGN, win)
    assert 0.4 < y0 < y1 < 0.6
    assert 0.4 < x0 < x1 < 0.6
    assert (y1 - y0) < 0.05 and (x1 - x0) < 0.05


def test_window_roi_bbox_fractions():
    bbox = BBox(minlon=-89.0, minlat=39.5, maxlon=-87.5, maxlat=40.5)
    win = _lattice_window(bbox)
    y0, y1, x0, x1 = _window_roi(bbox, win)
    extent = win.n * win.grid_deg
    assert x1 - x0 == pytest.approx(1.5 / extent, abs=1e-4)
    assert y1 - y0 == pytest.approx(1.0 / extent, abs=1e-4)


def test_window_from_global_slices_exact_lattice_indices():
    win = _lattice_window(_CHAMPAIGN)
    g = np.arange(721 * 1440, dtype=np.float32).reshape(721, 1440)
    sl = _window_from_global(g, win)
    assert sl.shape == (32, 32)
    r0 = int(round((90.0 - win.lat_max) / 0.25))
    c0 = int(round(win.lon_min / 0.25))
    assert sl[0, 0] == g[r0, c0]
    assert sl[-1, -1] == g[r0 + 31, c0 + 31]


def test_window_from_global_rejects_non_global_fields():
    win = _lattice_window(_CHAMPAIGN)
    with pytest.raises(ModelError, match="global 0.25"):
        _window_from_global(np.zeros((100, 100), dtype=np.float32), win)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_variant_default_and_explicit():
    assert oa._resolve_variant(None) == "pretrained"
    assert oa._resolve_variant({"variant": "small"}) == "small"
    assert oa._resolve_variant({"variant": " Pretrained "}) == "pretrained"


def test_variant_invalid_raises_model_error():
    with pytest.raises(ModelError, match="Unknown Aurora variant"):
        oa._resolve_variant({"variant": "gigantic"})


# ---------------------------------------------------------------------------
# Output builder
# ---------------------------------------------------------------------------


def _fake_grid(d: int = 6, hw: int = 8) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(d, hw, hw)).astype(np.float32)


def test_build_embedding_pooled_full_window_is_token_mean():
    grid = _fake_grid()
    emb = AuroraEmbedder._build_embedding(
        grid, geo_roi=(0.0, 1.0, 0.0, 1.0), output=OutputSpec.pooled(), meta={"m": 1}
    )
    assert emb.data.shape == (6,)
    np.testing.assert_allclose(emb.data, grid.mean(axis=(1, 2)), rtol=1e-6)
    assert emb.meta["pooling"] == "token_mean"


def test_build_embedding_pooled_roi_crops_before_pooling():
    grid = _fake_grid()
    emb = AuroraEmbedder._build_embedding(
        grid, geo_roi=(0.45, 0.55, 0.45, 0.55), output=OutputSpec.pooled(), meta={}
    )
    # 0.45..0.55 of an 8-token axis rounds outward to tokens 3..5
    np.testing.assert_allclose(emb.data, grid[:, 3:5, 3:5].mean(axis=(1, 2)), rtol=1e-6)
    assert emb.meta["pooling"] == "roi_grid_mean"


def test_build_embedding_grid_dataarray():
    grid = _fake_grid()
    emb = AuroraEmbedder._build_embedding(
        grid, geo_roi=(0.0, 1.0, 0.0, 1.0), output=OutputSpec.grid(), meta={}
    )
    assert emb.data.dims == ("d", "y", "x")
    assert emb.data.shape == (6, 8, 8)
    assert emb.meta["grid_type"] == "aurora_column_tokens"
    assert emb.meta["grid_hw"] == (8, 8)


def test_build_embedding_unknown_mode_and_pooling_raise():
    grid = _fake_grid()
    with pytest.raises(ModelError, match="Unknown output mode"):
        AuroraEmbedder._build_embedding(
            grid, geo_roi=(0.0, 1.0, 0.0, 1.0), output=SimpleNamespace(mode="volume"), meta={}
        )
    with pytest.raises(ModelError, match="Unknown pooling"):
        AuroraEmbedder._build_embedding(
            grid,
            geo_roi=(0.0, 1.0, 0.0, 1.0),
            output=SimpleNamespace(mode="pooled", pooling="median"),
            meta={},
        )


def test_tokens_to_level_grid_shape_and_level_mean():
    tokens = np.stack(
        [np.full((16,), float(level), dtype=np.float32) for level in range(4) for _ in range(64)]
    )
    grid = _tokens_to_level_grid(tokens, latent_levels=4, hw=(8, 8))
    assert grid.shape == (16, 8, 8)
    np.testing.assert_allclose(grid, 1.5)  # mean over levels 0..3


def test_tokens_to_level_grid_count_mismatch():
    with pytest.raises(ModelError, match="tokens"):
        _tokens_to_level_grid(np.zeros((100, 8), dtype=np.float32), latent_levels=4, hw=(8, 8))


# ---------------------------------------------------------------------------
# Channel layout through Batch assembly (needs the aurora package)
# ---------------------------------------------------------------------------


def test_assemble_batch_preserves_raw_values_and_layout():
    pytest.importorskip("aurora")
    win = _lattice_window(_CHAMPAIGN)
    times = (datetime(2022, 8, 31, 12), datetime(2022, 8, 31, 18))
    arr = np.zeros((2, _N_CHANNELS, 32, 32), dtype=np.float32)
    for t in range(2):
        for c in range(_N_CHANNELS):
            arr[t, c] = c + 1000.0 * t
    batch = oa._assemble_batch(arr, win=win, times=times)

    # surf: channels 0..3, shape (1, 2, 32, 32), raw values untouched
    assert tuple(batch.surf_vars["2t"].shape) == (1, 2, 32, 32)
    assert float(batch.surf_vars["2t"][0, 0, 0, 0]) == 0.0
    assert float(batch.surf_vars["msl"][0, 1, 0, 0]) == 3.0 + 1000.0
    # atmos: var-major then level: channel(var i, level j) = 4 + i*13 + j
    assert tuple(batch.atmos_vars["t"].shape) == (1, 2, 13, 32, 32)
    assert float(batch.atmos_vars["u"][0, 0, 2, 0, 0]) == 4.0 + 1 * 13 + 2
    assert float(batch.atmos_vars["z"][0, 1, 12, 0, 0]) == 4.0 + 4 * 13 + 12 + 1000.0
    # static: last 3 channels of frame 0, shape (32, 32)
    assert tuple(batch.static_vars["lsm"].shape) == (32, 32)
    assert float(batch.static_vars["z"][0, 0]) == float(_N_CHANNELS - 1)
    # metadata: north-up lattice, current time only, pretraining levels
    lat = batch.metadata.lat.numpy()
    lon = batch.metadata.lon.numpy()
    assert lat[0] == pytest.approx(win.lat_max) and np.all(np.diff(lat) < 0)
    assert lon[0] == pytest.approx(win.lon_min) and np.all(np.diff(lon) > 0)
    assert batch.metadata.time == (times[1],)
    assert batch.metadata.atmos_levels == _ATMOS_LEVELS


def test_check_finite_reports_empty_era5_frame():
    arr = np.zeros((2, _N_CHANNELS, 4, 4), dtype=np.float32)
    arr[1, : oa._N_SURF] = np.nan
    with pytest.raises(ModelError, match="empty time frames"):
        oa._check_finite(arr, resolve_time_pair(_T_SUMMER))
    arr2 = np.zeros((2, _N_CHANNELS, 4, 4), dtype=np.float32)
    arr2[0, 10, 0, 0] = np.inf
    with pytest.raises(ModelError, match="non-finite"):
        oa._check_finite(arr2, resolve_time_pair(_T_SUMMER))


# ---------------------------------------------------------------------------
# Mocked end-to-end: fetch_input + get_embedding + batch equivalence
# ---------------------------------------------------------------------------


def _install_fetch_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_bins(provider, *, collection, bands, bins, lat_max, lon_min, n_lat, n_lon, grid_deg):
        assert len(bins) == 2
        t = np.arange(2 * len(bands) * n_lat * n_lon, dtype=np.float32)
        return t.reshape(2, len(bands), n_lat, n_lon), {
            "frames": [{"start": b[0], "end": b[1], "empty": False} for b in bins],
            "n_empty": 0,
        }

    def fake_slab(time_iso):
        rng = np.random.default_rng(abs(hash(time_iso)) % (2**32))
        return rng.normal(size=(5, 13, 721, 1440)).astype(np.float32)

    monkeypatch.setattr(oa, "_fetch_latlon_grid_bins_tchw", fake_bins)
    monkeypatch.setattr(oa, "_arco_global_slab", fake_slab)
    monkeypatch.setattr(
        oa,
        "_static_fields",
        lambda: {
            k: np.full((721, 1440), i, dtype=np.float32) for i, k in enumerate(("lsm", "slt", "z"))
        },
    )


def _install_model_mocks(monkeypatch: pytest.MonkeyPatch, dim: int = 6) -> None:
    def fake_loader(*, variant, dev):
        return object(), {"latent_levels": 4, "patch_size": 4, "dim": dim, "variant": variant}

    def fake_assemble(arr, *, win, times):
        return arr  # the fake encoder consumes the raw array directly

    def fake_tokens(model, batch):
        arr = np.asarray(batch, dtype=np.float64)
        base = float(arr.mean())
        return np.stack([np.full((dim,), base + i, dtype=np.float32) for i in range(4 * 8 * 8)])

    monkeypatch.setattr(oa, "_load_aurora_encoder_cached", fake_loader)
    monkeypatch.setattr(oa, "_assemble_batch", fake_assemble)
    monkeypatch.setattr(oa, "_encoder_tokens", fake_tokens)


def test_fetch_input_combined_stack_and_meta(monkeypatch):
    _install_fetch_mocks(monkeypatch)
    emb = AuroraEmbedder()
    fr = emb.fetch_input(
        object(), spatial=_CHAMPAIGN, temporal=_T_SUMMER, sensor=emb._default_sensor()
    )
    assert fr is not None
    assert fr.data.shape == (2, _N_CHANNELS, 32, 32)
    assert fr.data.dtype == np.float32
    assert fr.meta["time_pair"] == ["2022-08-31T12:00:00", "2022-08-31T18:00:00"]
    assert fr.meta["window"]["n"] == 32
    assert "roi_window_geo" in fr.meta  # point ROI is a window subset
    # static channels replicate across both frames
    np.testing.assert_array_equal(fr.data[0, -3:], fr.data[1, -3:])


def test_fetch_input_empty_bin_raises_with_lag_hint(monkeypatch):
    def empty_bins(provider, **kw):
        n = kw["n_lat"]
        return (
            np.full((2, 4, n, n), np.nan, dtype=np.float32),
            {"frames": [{"start": "s", "end": "e", "empty": True}], "n_empty": 1},
        )

    monkeypatch.setattr(oa, "_fetch_latlon_grid_bins_tchw", empty_bins)
    emb = AuroraEmbedder()
    with pytest.raises(ModelError, match="delay"):
        emb.fetch_input(
            object(), spatial=_CHAMPAIGN, temporal=_T_SUMMER, sensor=emb._default_sensor()
        )


def test_fetch_input_wrong_band_count_raises():
    emb = AuroraEmbedder()
    bad = emb._default_sensor().__class__(collection="ECMWF/ERA5_HOURLY", bands=("temperature_2m",))
    with pytest.raises(ModelError, match="surface bands"):
        emb.fetch_input(object(), spatial=_CHAMPAIGN, temporal=_T_SUMMER, sensor=bad)


def test_get_embedding_input_chw_shape_check(monkeypatch):
    _install_model_mocks(monkeypatch)
    emb = AuroraEmbedder()
    with pytest.raises(ModelError, match="combined TCHW stack"):
        emb.get_embedding(
            spatial=_CHAMPAIGN,
            temporal=_T_SUMMER,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="tensor",
            input_chw=np.zeros((4, 32, 32), dtype=np.float32),
        )


def test_get_embedding_tensor_backend_requires_input_chw():
    emb = AuroraEmbedder()
    with pytest.raises(ModelError, match="requires input_chw"):
        emb.get_embedding(
            spatial=_CHAMPAIGN,
            temporal=_T_SUMMER,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="tensor",
        )


def test_get_embedding_mocked_pooled_and_grid(monkeypatch):
    _install_fetch_mocks(monkeypatch)
    _install_model_mocks(monkeypatch)
    emb = AuroraEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _b: object())
    pooled = emb.get_embedding(
        spatial=_CHAMPAIGN,
        temporal=_T_SUMMER,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
    )
    assert pooled.data.shape == (6,)
    assert pooled.meta["model"] == "aurora"
    assert pooled.meta["temporal"]["start"] == "2022-06-01"
    assert pooled.meta["time_pair"] == ["2022-08-31T12:00:00", "2022-08-31T18:00:00"]
    assert pooled.meta["pooling"] == "roi_grid_mean"  # point ROI crops the window
    assert pooled.meta["image_size"] == 32

    grid = emb.get_embedding(
        spatial=_CHAMPAIGN,
        temporal=_T_SUMMER,
        sensor=None,
        output=OutputSpec.grid(),
        backend="auto",
    )
    assert grid.data.dims == ("d", "y", "x")
    assert grid.meta["grid_type"] == "aurora_column_tokens"


def test_single_and_batch_from_inputs_agree(monkeypatch):
    _install_fetch_mocks(monkeypatch)
    _install_model_mocks(monkeypatch)
    emb = AuroraEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _b: object())
    fr = emb.fetch_input(
        object(), spatial=_CHAMPAIGN, temporal=_T_SUMMER, sensor=emb._default_sensor()
    )
    assert fr is not None
    single = emb.get_embedding(
        spatial=_CHAMPAIGN,
        temporal=_T_SUMMER,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        input_chw=fr.data,
        fetch_meta=fr.meta,
    )
    batch = emb.get_embeddings_batch_from_inputs(
        spatials=[_CHAMPAIGN],
        input_chws=[fr.data],
        temporal=_T_SUMMER,
        output=OutputSpec.pooled(),
        fetch_metas=[fr.meta],
    )
    np.testing.assert_allclose(single.data, batch[0].data, rtol=1e-6)


def test_batch_from_inputs_length_mismatch():
    emb = AuroraEmbedder()
    with pytest.raises(ValueError, match="length mismatch"):
        emb.get_embeddings_batch_from_inputs(
            spatials=[_CHAMPAIGN], input_chws=[], temporal=_T_SUMMER
        )


# ---------------------------------------------------------------------------
# Live tests (network + weights) — RS_EMBED_LIVE_AURORA=1 to run
# ---------------------------------------------------------------------------

_LIVE = pytest.mark.skipif(
    not os.environ.get("RS_EMBED_LIVE_AURORA"),
    reason="live Aurora test; set RS_EMBED_LIVE_AURORA=1 to run",
)


@_LIVE
def test_live_gee_surface_matches_arco_registration():
    """GEE lat/lon-grid sampling must return the same ERA5 grid-point values
    as ARCO — this validates the crsTransform registration end to end."""
    from rs_embed.providers.gee import GEEProvider

    emb = AuroraEmbedder()
    provider = GEEProvider()
    win = _lattice_window(_CHAMPAIGN)
    t_prev, t_cur = resolve_time_pair(_T_SUMMER)
    surf, meta = oa._fetch_latlon_grid_bins_tchw(
        provider,
        collection="ECMWF/ERA5_HOURLY",
        bands=emb._default_sensor().bands,
        bins=oa._hour_bins((t_prev, t_cur)),
        lat_max=win.lat_max,
        lon_min=win.lon_min,
        n_lat=win.n,
        n_lon=win.n,
        grid_deg=win.grid_deg,
    )
    assert meta["n_empty"] == 0
    ds = oa._arco_dataset()
    arco_2t = ds["2m_temperature"].sel(time=np.datetime64(oa._iso(t_cur))).values.astype(np.float32)
    lat_desc = bool(ds["latitude"].values[0] > ds["latitude"].values[-1])
    if not lat_desc:
        arco_2t = arco_2t[::-1, :]
    window_2t = _window_from_global(arco_2t, win)
    np.testing.assert_allclose(surf[1, 0], window_2t, atol=0.5)


@_LIVE
def test_live_get_embedding_small_variant():
    emb = AuroraEmbedder()
    out = emb.get_embedding(
        spatial=_CHAMPAIGN,
        temporal=_T_SUMMER,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        model_config={"variant": "small"},
    )
    assert out.data.shape == (256,)
    assert np.isfinite(out.data).all()
    assert out.meta["encoder_only"] is True
