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


def test_get_embedding_default_mode_stays_single(monkeypatch):
    """No temporal_mode -> single path (no multi helpers touched)."""
    emb = pr.PrithviEOV2S2_6B_Embedder()
    spatial = PointBuffer(lon=10.0, lat=20.0, buffer_m=256)
    temporal = TemporalSpec.range("2022-01-01", "2023-01-01")

    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        pr,
        "_prepare_prithvi_chw",
        lambda x_chw, *, fill_value: (x_chw.astype(np.float32, copy=False), {"prep_mode": "t"}),
    )
    monkeypatch.setattr(
        pr,
        "_load_prithvi",
        lambda model_key, **k: (object(), {"repo_id": "r"}, "cpu"),
    )
    monkeypatch.setattr(
        pr,
        "_prithvi_forward_tokens",
        lambda *a, **k: np.arange(8, dtype=np.float32).reshape(2, 4),
    )

    def _boom(*a, **k):
        raise AssertionError("multi path must not run in default (single) mode")

    monkeypatch.setattr(pr, "_prithvi_forward_tokens_multiframe", _boom)

    out = emb.get_embedding(
        spatial=spatial,
        temporal=temporal,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((6, 8, 8), 1000.0, dtype=np.float32),
    )
    assert out.meta["num_frames"] == 1
    assert out.meta.get("temporal_mode") != "multi"
