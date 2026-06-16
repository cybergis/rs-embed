"""Window-adaptive temporal sampling for Galileo (mirrors OlmoEarth).

Galileo encodes month-of-year (0–11) and pretrains on ~monthly composites,
capped at 12 frames. These tests pin the frame-count policy: sub-month windows
collapse to a single frame, multi-month windows use ~30-day frames (≤12), and
windows beyond the 12-month capacity are equal-divided with a warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_galileo as gal


def _plan(start: str, end: str, cfg=None):
    return gal._resolve_frame_plan(cfg, TemporalSpec.range(start, end))


# ---------------------------------------------------------------------------
# Frame-count policy
# ---------------------------------------------------------------------------


def test_auto_short_window_collapses_to_single_frame():
    n, meta = _plan("2022-06-01", "2022-06-15")  # 14 days
    assert n == 1
    assert meta["temporal_mode"] == "single"
    assert meta["temporal_spacing_stretched"] is False


def test_auto_multi_month_window_uses_monthly_frames():
    n, meta = _plan("2022-06-01", "2022-09-01")  # ~3 months
    assert 3 <= n <= 4  # fixed 30-day bins (last bin may be a partial sliver)
    assert meta["temporal_mode"] == "multi"
    assert meta["temporal_sampling"] == "fixed_stride"
    assert meta["temporal_spacing_stretched"] is False


def test_one_year_window_stays_fixed_monthly_twelve_frames():
    n, meta = _plan("2022-01-01", "2023-01-01")  # 365d, drops only 5d (<30)
    assert n == 12
    assert meta["temporal_sampling"] == "fixed_stride"
    assert meta["temporal_spacing_stretched"] is False
    assert "effective_stride_days" not in meta


def test_long_window_equal_divides_to_twelve_and_flags_stretched():
    n, meta = _plan("2022-01-01", "2025-01-01")  # 3 years
    assert n == 12
    assert meta["temporal_sampling"] == "equal_divided"
    assert meta["temporal_spacing_stretched"] is True
    assert meta["effective_stride_days"] > gal._FRAME_STRIDE_DAYS


# ---------------------------------------------------------------------------
# Mode resolution / overrides
# ---------------------------------------------------------------------------


def test_expand_auto_mode_picks_by_window():
    assert gal._expand_auto_mode("auto", TemporalSpec.range("2022-06-01", "2022-06-15")) == "single"
    assert gal._expand_auto_mode("auto", TemporalSpec.range("2022-06-01", "2022-09-01")) == "multi"
    # explicit modes pass through unchanged
    assert (
        gal._expand_auto_mode("single", TemporalSpec.range("2022-06-01", "2022-09-01")) == "single"
    )
    assert gal._expand_auto_mode("multi", TemporalSpec.range("2022-06-01", "2022-06-15")) == "multi"


def test_explicit_single_forces_one_frame_even_for_long_window():
    n, meta = _plan("2022-01-01", "2025-01-01", {"temporal_mode": "single"})
    assert n == 1
    assert meta["temporal_mode"] == "single"


def test_explicit_n_frames_override_bypasses_adaptive_policy():
    # model_config override
    n, meta = _plan("2022-01-01", "2025-01-01", {"n_frames": 5})
    assert n == 5
    assert meta["temporal_sampling"] == "manual"
    assert meta["temporal_spacing_stretched"] is False


def test_env_frames_override_is_honored(monkeypatch):
    monkeypatch.setenv("RS_EMBED_GALILEO_FRAMES", "7")
    n, meta = _plan("2022-01-01", "2023-06-01")
    assert n == 7
    assert meta["temporal_sampling"] == "manual"


def test_env_temporal_mode_single(monkeypatch):
    monkeypatch.setenv("RS_EMBED_GALILEO_TEMPORAL_MODE", "single")
    n, meta = _plan("2022-01-01", "2023-01-01")
    assert n == 1
    assert meta["temporal_mode"] == "single"


def test_normalize_temporal_mode_rejects_garbage():
    with pytest.raises(ModelError):
        gal._normalize_temporal_mode("sometimes")


# ---------------------------------------------------------------------------
# Warning behavior
# ---------------------------------------------------------------------------


def test_warn_fires_only_when_stretched():
    _n, stretched_meta = _plan("2022-01-01", "2025-01-01")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gal._warn_stretched_sampling(stretched_meta)
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "Galileo" in str(w[0].message)

    _n2, fixed_meta = _plan("2022-01-01", "2023-01-01")
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        gal._warn_stretched_sampling(fixed_meta)
    assert len(w2) == 0


# ---------------------------------------------------------------------------
# fetch_input override uses the adaptive frame count (prefetch/export path)
# ---------------------------------------------------------------------------


def test_fetch_input_override_uses_adaptive_frame_count(monkeypatch):
    emb = gal.GalileoEmbedder()
    seen = {}

    def _fake_fetch(provider, spatial, temporal, *, n_frames, **kw):
        seen["n_frames"] = int(n_frames)
        return np.full((int(n_frames), 10, 8, 8), 1000.0, dtype=np.float32)

    monkeypatch.setattr(gal, "_fetch_s2_10_raw_tchw", _fake_fetch)

    fr = emb.fetch_input(
        provider=object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),
        sensor=emb._default_sensor(),
    )
    assert seen["n_frames"] == 12
    assert fr is not None
    assert fr.data.shape[0] == 12
    assert fr.meta["temporal_sampling"] == "fixed_stride"


def test_fetch_input_override_single_for_short_window(monkeypatch):
    emb = gal.GalileoEmbedder()
    seen = {}

    def _fake_fetch(provider, spatial, temporal, *, n_frames, **kw):
        seen["n_frames"] = int(n_frames)
        return np.full((int(n_frames), 10, 8, 8), 1000.0, dtype=np.float32)

    monkeypatch.setattr(gal, "_fetch_s2_10_raw_tchw", _fake_fetch)
    emb.fetch_input(
        provider=object(),
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-06-20"),
        sensor=emb._default_sensor(),
    )
    assert seen["n_frames"] == 1


# ---------------------------------------------------------------------------
# Batch path: adaptive frame count + model_config threading
# ---------------------------------------------------------------------------


def test_batch_uses_adaptive_frame_count_and_threads_model_config(monkeypatch):
    emb = gal.GalileoEmbedder()
    monkeypatch.setenv("RS_EMBED_GALILEO_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())

    fetched = {}

    def _fake_fetch(provider, spatial, temporal, *, n_frames, **kw):
        fetched["n_frames"] = int(n_frames)
        return np.full((int(n_frames), 10, 8, 8), 2222.0, dtype=np.float32)

    monkeypatch.setattr(gal, "_fetch_s2_10_raw_tchw", _fake_fetch)

    seen_cfg = []

    def _fake_get_embedding(**kw):
        seen_cfg.append(kw.get("model_config"))
        arr = kw["input_chw"]
        return Embedding(data=np.array([arr.shape[0]], dtype=np.float32), meta={})

    monkeypatch.setattr(emb, "get_embedding", _fake_get_embedding)

    out = emb.get_embeddings_batch(
        spatials=[PointBuffer(lon=1.0, lat=2.0, buffer_m=256)],
        temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),
        model_config={"variant": "nano"},
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 1
    assert fetched["n_frames"] == 12  # adaptive, not the old fixed 8
    assert seen_cfg == [{"variant": "nano"}]  # model_config threaded to get_embedding


def test_batch_explicit_n_frames_override(monkeypatch):
    emb = gal.GalileoEmbedder()
    monkeypatch.setenv("RS_EMBED_GALILEO_FETCH_WORKERS", "1")
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())

    fetched = {}

    def _fake_fetch(provider, spatial, temporal, *, n_frames, **kw):
        fetched["n_frames"] = int(n_frames)
        return np.full((int(n_frames), 10, 8, 8), 1.0, dtype=np.float32)

    monkeypatch.setattr(gal, "_fetch_s2_10_raw_tchw", _fake_fetch)
    monkeypatch.setattr(
        emb,
        "get_embedding",
        lambda **kw: Embedding(data=np.array([0.0], dtype=np.float32), meta={}),
    )

    emb.get_embeddings_batch(
        spatials=[PointBuffer(lon=1.0, lat=2.0, buffer_m=256)],
        temporal=TemporalSpec.range("2022-01-01", "2025-01-01"),
        model_config={"n_frames": 3},
        output=OutputSpec.pooled(),
        backend="gee",
    )
    assert fetched["n_frames"] == 3
