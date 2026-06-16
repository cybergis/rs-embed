import numpy as np
import pytest

from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_prithvi as pr
from rs_embed.embedders.meta import temporal_to_range

# ---------------------------------------------------------------------------
# Window -> frame-count policy (clamp(window_days // stride, 1, max_frames))
# ---------------------------------------------------------------------------


def _nf(start, end, *, max_frames=4, stride=30):
    t = temporal_to_range(TemporalSpec.range(start, end))
    return pr._auto_num_frames(t, max_frames=max_frames, stride_days=stride)


def test_auto_num_frames_small_window_downgrades_to_one():
    assert _nf("2022-06-01", "2022-06-15") == 1  # 14 days < 30 -> T=1


def test_auto_num_frames_scales_with_window_and_caps():
    assert _nf("2022-06-01", "2022-09-01") == 3  # 92 days // 30
    assert _nf("2022-01-01", "2023-01-01") == 4  # 365 // 30 = 12, capped at max_frames=4
    assert _nf("2022-01-01", "2023-01-01", max_frames=6) == 6  # raised cap


def test_auto_num_frames_respects_custom_stride():
    # 92-day window with a 45-day minimum spacing -> 2 frames.
    assert _nf("2022-06-01", "2022-09-01", stride=45) == 2


# ---------------------------------------------------------------------------
# Duplicate-frame collapse (provider back-fills empty bins with identical frames)
# ---------------------------------------------------------------------------


def test_drop_duplicate_frames_collapses_identical():
    a = np.full((6, 8, 8), 0.1, dtype=np.float32)
    b = np.full((6, 8, 8), 0.2, dtype=np.float32)
    stack = np.stack([a, a, b], axis=0)
    out, dates = pr._drop_duplicate_frames(stack, ["d0", "d1", "d2"])
    assert out.shape[0] == 2
    assert dates == ["d0", "d2"]  # first occurrence's date is kept


def test_drop_duplicate_frames_all_identical_downgrades_to_one():
    a = np.full((6, 8, 8), 0.3, dtype=np.float32)
    stack = np.stack([a, a, a, a], axis=0)
    out, dates = pr._drop_duplicate_frames(stack, ["a", "b", "c", "d"])
    assert out.shape[0] == 1 and dates == ["a"]


# ---------------------------------------------------------------------------
# Token reshape / pooling against the REAL vendored architecture (tiny weights)
# ---------------------------------------------------------------------------


def test_multiframe_forward_token_count_and_pooling_real_runtime():
    torch = pytest.importorskip("torch")
    from rs_embed.embedders._vendor.prithvi_mae import PrithviMAE

    T, H, W, P = 3, 32, 32, 16
    model = PrithviMAE(
        img_size=H,
        patch_size=(1, P, P),
        num_frames=T,
        in_chans=6,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=4,
        coords_encoding=["time", "location"],
    ).eval()

    x_tchw = np.random.rand(T, 6, H, W).astype("float32")
    dates = ["2022-02-15", "2022-05-15", "2022-08-15"]
    tokens = pr._prithvi_forward_tokens_multiframe(
        model, x_tchw, lon=10.0, lat=20.0, date_strs=dates, device="cpu"
    )

    hw = H // P
    assert tokens.shape[0] == 1 + T * hw * hw  # CLS + T*(h*w)

    patch, has_cls, (gh, gw) = pr._split_prithvi_patch_tokens(tokens, T)
    assert has_cls and patch.shape == (T, hw, hw, 32) and (gh, gw) == (hw, hw)

    vec, cls_removed = pr.pool_from_tokens_tchw(tokens, T, "mean")
    assert vec.shape == (32,) and cls_removed
    # Pooling over time+space == mean of all (non-CLS) patch tokens.
    assert np.allclose(vec, tokens[1:].mean(axis=0), atol=1e-5)

    grid, (gh, gw), _ = pr.tokens_to_grid_dhw_tchw(tokens, T, "mean")
    assert grid.shape == (32, hw, hw)
    _ = torch  # silence unused


# ---------------------------------------------------------------------------
# get_embedding multi path: meta + frame count + load num_frames
# ---------------------------------------------------------------------------


def test_get_embedding_multi_uses_window_frame_count(monkeypatch):
    emb = pr.PrithviEOV2S2_6B_Embedder()
    spatial = PointBuffer(lon=10.0, lat=20.0, buffer_m=256)
    temporal = TemporalSpec.range("2022-01-01", "2023-01-01")  # 1yr -> 4 frames
    seen = {}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    # 4 distinct raw frames so dedup keeps all of them.
    raw = np.stack(
        [np.full((6, 16, 16), 1000.0 * (k + 1), dtype=np.float32) for k in range(4)], axis=0
    )
    monkeypatch.setattr(
        pr, "_fetch_s2_prithvi6_tchw", lambda *a, n_frames, **k: raw[:n_frames].copy()
    )
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "t"}),
    )

    def _fake_load(model_key, *, pretrained, bands, num_frames, coords_encoding, device):
        seen["num_frames"] = num_frames
        return object(), {"repo_id": "r"}, "cpu"

    monkeypatch.setattr(pr, "_load_prithvi", _fake_load)

    def _fake_fwd(model, x_tchw, *, lon, lat, date_strs, device):
        seen["fwd_T"] = x_tchw.shape[0]
        seen["fwd_dates"] = list(date_strs)
        n = x_tchw.shape[0]
        return np.zeros((1 + n * 4, 8), dtype=np.float32)  # hw=2 -> 4 patches/frame

    monkeypatch.setattr(pr, "_prithvi_forward_tokens_multiframe", _fake_fwd)

    out = emb.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        model_config={"temporal_mode": "multi"},
    )

    assert out.meta["temporal_mode"] == "multi"
    assert out.meta["num_frames"] == 4
    assert out.meta["requested_frames"] == 4
    assert seen["num_frames"] == 4
    assert seen["fwd_T"] == 4
    assert len(out.meta["frame_dates"]) == 4
    assert out.data.shape == (8,)


def test_get_embedding_multi_collapses_duplicate_frames_to_one(monkeypatch):
    emb = pr.PrithviEOV2S2_6B_Embedder()
    spatial = PointBuffer(lon=10.0, lat=20.0, buffer_m=256)
    temporal = TemporalSpec.range("2022-01-01", "2023-01-01")  # would request 4
    seen = {}

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    # Provider returns identical frames (window had no temporal diversity).
    dup = np.stack([np.full((6, 16, 16), 1234.0, dtype=np.float32)] * 4, axis=0)
    monkeypatch.setattr(
        pr, "_fetch_s2_prithvi6_tchw", lambda *a, n_frames, **k: dup[:n_frames].copy()
    )
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "t"}),
    )

    def _fake_load(model_key, *, pretrained, bands, num_frames, coords_encoding, device):
        seen["num_frames"] = num_frames
        return object(), {"repo_id": "r"}, "cpu"

    monkeypatch.setattr(pr, "_load_prithvi", _fake_load)
    monkeypatch.setattr(
        pr,
        "_prithvi_forward_tokens_multiframe",
        lambda model, x_tchw, *, lon, lat, date_strs, device: np.zeros(
            (1 + x_tchw.shape[0] * 4, 8), dtype=np.float32
        ),
    )

    out = emb.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        model_config={"temporal_mode": "multi"},
    )

    assert out.meta["requested_frames"] == 4
    assert out.meta["num_frames"] == 1  # duplicates collapsed -> auto downgrade
    assert seen["num_frames"] == 1


def test_effective_temporal_mode_auto_picks_by_window():
    def eff(start, end, cfg=None):
        return pr._effective_temporal_mode(cfg, TemporalSpec.range(start, end))

    assert pr._resolve_temporal_mode(None) == "auto"  # new default
    assert eff("2022-06-01", "2022-06-15") == "single"  # 14d -> 1 frame -> single
    assert eff("2022-06-01", "2022-09-01") == "multi"  # 92d -> >=2 frames -> multi
    assert eff("2022-01-01", "2023-01-01") == "multi"  # long -> multi (capped at 4)
    # explicit modes pass through regardless of window
    assert eff("2022-01-01", "2023-01-01", {"temporal_mode": "single"}) == "single"
    assert eff("2022-06-01", "2022-06-10", {"temporal_mode": "multi"}) == "multi"


def _single_path_only(monkeypatch, emb):
    """Wire mocks so the single path works and the multi path explodes if hit."""
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "t"}),
    )
    monkeypatch.setattr(
        pr, "_load_prithvi", lambda model_key, **k: (object(), {"repo_id": "r"}, "cpu")
    )
    monkeypatch.setattr(
        pr, "_prithvi_forward_tokens", lambda *a, **k: np.arange(8, dtype=np.float32).reshape(2, 4)
    )

    def _boom(*a, **k):
        raise AssertionError("multi path must not run for a single-frame window")

    monkeypatch.setattr(pr, "_prithvi_forward_tokens_multiframe", _boom)


def test_get_embedding_auto_short_window_uses_single(monkeypatch):
    """auto default + a sub-month window -> single path (T=1), no multi helpers."""
    emb = pr.PrithviEOV2S2_6B_Embedder()
    _single_path_only(monkeypatch, emb)
    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-06-15"),  # 14d -> single
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((6, 8, 8), 1000.0, dtype=np.float32),
    )
    assert out.meta["num_frames"] == 1
    assert out.meta.get("temporal_mode") != "multi"


def test_get_embedding_explicit_single_forces_single(monkeypatch):
    """temporal_mode='single' forces the single path even for a long window."""
    emb = pr.PrithviEOV2S2_6B_Embedder()
    _single_path_only(monkeypatch, emb)
    out = emb.get_embedding(
        spatial=PointBuffer(lon=10.0, lat=20.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),  # would be multi under auto
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((6, 8, 8), 1000.0, dtype=np.float32),
        model_config={"temporal_mode": "single"},
    )
    assert out.meta["num_frames"] == 1
    assert out.meta.get("temporal_mode") != "multi"


# ---------------------------------------------------------------------------
# Frame-spacing bounds: 28-day min stride, 184-day max-gap flag/warning
# ---------------------------------------------------------------------------


def test_default_stride_is_28_days():
    assert pr._DEFAULT_FRAME_STRIDE_DAYS == 28
    assert pr._DEFAULT_MAX_FRAME_STRIDE_DAYS == 184


def test_max_consecutive_gap_days():
    assert pr._max_consecutive_gap_days([]) == 0
    assert pr._max_consecutive_gap_days(["2022-02-15"]) == 0
    # Mar->Aug is the largest of [Feb->Mar=28, Mar->Aug=153]
    assert pr._max_consecutive_gap_days(["2022-02-15", "2022-03-15", "2022-08-15"]) == 153


def test_temporal_spacing_meta_in_range_no_warning(recwarn):
    meta = pr._temporal_spacing_meta(
        ["2022-02-15", "2022-05-15", "2022-08-15"], max_stride_days=184
    )
    assert meta["max_frame_gap_days"] == 92  # max(Feb15->May15=89, May15->Aug15=92)
    assert "temporal_spacing_out_of_range" not in meta
    assert len(recwarn) == 0


def test_temporal_spacing_meta_out_of_range_warns():
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        meta = pr._temporal_spacing_meta(
            ["2020-01-01", "2020-10-01", "2021-07-01"], max_stride_days=184
        )
    assert meta["temporal_spacing_out_of_range"] is True
    assert meta["max_frame_gap_days"] > 184
    assert len(w) == 1 and issubclass(w[0].category, UserWarning)


def test_resolve_max_frame_stride_days_env(monkeypatch):
    monkeypatch.delenv("RS_EMBED_PRITHVI_MAX_STRIDE_DAYS", raising=False)
    assert pr._resolve_max_frame_stride_days() == 184
    monkeypatch.setenv("RS_EMBED_PRITHVI_MAX_STRIDE_DAYS", "120")
    assert pr._resolve_max_frame_stride_days() == 120
