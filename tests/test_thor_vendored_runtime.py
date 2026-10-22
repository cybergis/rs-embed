import types

import numpy as np
import pytest
import torch

from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec


class _FakeTHORModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.groups = {
            "group0": ["S2:Blue", "S2:Green", "S2:Red", "S2:NIR"],
            "group1": ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"],
        }
        self.out_channels = [8]
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        batch = x.shape[0]
        channel_params = {
            "S2:Blue": {"num_patch": 18},
            "S2:Green": {"num_patch": 18},
            "S2:Red": {"num_patch": 18},
            "S2:NIR": {"num_patch": 18},
            "S2:RE1": {"num_patch": 9},
            "S2:RE2": {"num_patch": 9},
            "S2:RE3": {"num_patch": 9},
            "S2:RE4": {"num_patch": 9},
            "S2:SWIR1": {"num_patch": 9},
            "S2:SWIR2": {"num_patch": 9},
        }
        features = [torch.arange(batch * 405 * 8, dtype=torch.float32).reshape(batch, 405, 8)]
        return features, channel_params


def test_thor_loads_vendored_runtime_without_external_thor_dependency(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    fake_module = types.SimpleNamespace(
        load_thor_model=lambda **kwargs: _FakeTHORModel(),
    )
    thor._load_thor_module.cache_clear()
    thor._load_thor_cached.cache_clear()
    monkeypatch.setattr(thor, "_load_thor_module", lambda: fake_module)

    model, meta = thor._load_thor_cached(
        model_key="thor_v1_base",
        model_bands=tuple(thor._THOR_MODEL_BANDS),
        pretrained=False,
        ckpt_path=None,
        ground_cover=2880,
        patch_size=16,
        dev="cpu",
    )

    assert meta["model_key"] == "thor_v1_base"
    assert meta["pretrained"] is False

    tokens, grid, fmeta = thor._thor_forward_single(
        model,
        np.zeros((10, 288, 288), dtype=np.float32),
        device="cpu",
        group_merge="mean",
    )

    assert tokens.shape == (405, 8)
    assert grid is not None
    assert grid.shape == (8, 18, 18)
    assert fmeta["expected_patch_tokens"] == 405


def test_thor_runtime_config_defaults_patch_size_to_8(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    monkeypatch.delenv("RS_EMBED_THOR_PATCH_SIZE", raising=False)
    monkeypatch.delenv("RS_EMBED_THOR_IMG", raising=False)
    monkeypatch.delenv("RS_EMBED_THOR_MODEL_KEY", raising=False)
    monkeypatch.delenv("RS_EMBED_THOR_RESIZE_MODE", raising=False)

    cfg = thor._resolve_thor_runtime_config(
        model_config=None,
        default_model_key=thor.THORBaseEmbedder.DEFAULT_MODEL_KEY,
        default_image_size=thor.THORBaseEmbedder.DEFAULT_IMAGE_SIZE,
    )

    assert cfg["patch_size"] == 8
    assert cfg["image_size"] == 288
    assert cfg["resize_mode"] == "native_snap"


def test_prepare_thor_raw_chw_native_snap_crops_small_projection_mismatch():
    import rs_embed.embedders.onthefly_thor as thor

    raw = np.zeros((10, 289, 288), dtype=np.float32)
    out, meta = thor._prepare_thor_raw_chw(
        raw,
        scale_m=10,
        patch_size=8,
        band_gsds=(10, 20),
        image_size=288,
        resize_mode="native_snap",
        shape_adjust="crop",
        shape_tol_px=8,
        max_native_side=384,
        max_native_tokens=3000,
        input_prep_mode="resize",
        fill_value=0.0,
    )

    assert out.shape == (10, 288, 288)
    assert meta["preprocess_strategy"] == "native_snap"
    assert meta["shape_adjust_applied"] == "crop_to_square"
    assert meta["effective_image_size"] == 288
    assert meta["estimated_patch_tokens"] == 1620


def test_prepare_thor_raw_chw_native_snap_snaps_square_side_to_valid_grid():
    import rs_embed.embedders.onthefly_thor as thor

    raw = np.zeros((10, 300, 300), dtype=np.float32)
    out, meta = thor._prepare_thor_raw_chw(
        raw,
        scale_m=10,
        patch_size=8,
        band_gsds=(10, 20),
        image_size=288,
        resize_mode="native_snap",
        shape_adjust="crop",
        shape_tol_px=8,
        max_native_side=384,
        max_native_tokens=3000,
        input_prep_mode="resize",
        fill_value=0.0,
    )

    assert out.shape == (10, 304, 304)
    assert meta["preprocess_strategy"] == "native_snap"
    assert meta["snapped_side"] == 304
    assert meta["effective_image_size"] == 304
    assert meta["estimated_patch_tokens"] == 1805


def _prepare_default(raw, **overrides):
    import rs_embed.embedders.onthefly_thor as thor

    kwargs = {
        "scale_m": 10,
        "patch_size": 8,
        "band_gsds": (10, 20),
        "image_size": 288,
        "resize_mode": "native_snap",
        "shape_adjust": "crop",
        "shape_tol_px": 8,
        "max_native_side": 384,
        "max_native_tokens": 3000,
        "input_prep_mode": "resize",
        "fill_value": 0.0,
        **overrides,
    }
    return thor._prepare_thor_raw_chw(raw, **kwargs)


def test_prepare_thor_raw_chw_tile_mode_forces_fixed_resize():
    raw = np.zeros((10, 300, 300), dtype=np.float32)
    with pytest.warns(UserWarning, match="tile_preserve_stitch_geometry"):
        out, meta = _prepare_default(raw, input_prep_mode="tile")

    assert out.shape == (10, 300, 300)
    assert meta["preprocess_strategy"] == "fixed_resize"
    assert meta["preprocess_reason"] == "tile_preserve_stitch_geometry"
    assert meta["effective_image_size"] == 288


def test_prepare_thor_raw_chw_large_square_falls_back_to_fixed_resize():
    raw = np.zeros((10, 512, 512), dtype=np.float32)
    with pytest.warns(UserWarning, match="native_side_limit"):
        out, meta = _prepare_default(raw)

    assert out.shape == (10, 512, 512)
    assert meta["preprocess_strategy"] == "fixed_resize"
    assert meta["preprocess_reason"] == "native_side_limit"
    assert meta["effective_image_size"] == 288
    # 5120 m of real data stretched onto a model told 2880 m at 10 m.
    assert meta["ground_cover_m"] == 2880
    assert meta["effective_gsd_m"] == pytest.approx((17.7778, 17.7778))
    assert meta["gsd_mismatch"] is True


def test_prepare_thor_raw_chw_fallback_drops_abandoned_native_meta():
    # Non-square but within tolerance, so native_snap crops to square before the
    # side limit sends the uncropped input to the fixed resize.
    raw = np.zeros((10, 520, 514), dtype=np.float32)
    with pytest.warns(UserWarning, match="native_side_limit"):
        _, meta = _prepare_default(raw)

    assert meta["shape_adjust_applied"] == "none"
    assert "square_side" not in meta
    assert meta["snapped_side"] == 512
    assert meta["effective_gsd_m"] == pytest.approx((18.0556, 17.8472))


def test_prepare_thor_raw_chw_token_limit_stays_native_when_resize_adds_tokens():
    # patch_size=4 at 2560 m needs 64^2 + 32^2 = 5120 tokens; resizing to 288
    # would need 6480 and misstate the cover as 2880 m, so the input stays native.
    raw = np.zeros((10, 256, 256), dtype=np.float32)
    with pytest.warns(UserWarning, match="would not reduce it; running natively"):
        out, meta = _prepare_default(raw, patch_size=4)

    assert out.shape == (10, 256, 256)
    assert meta["preprocess_strategy"] == "native_snap"
    assert meta["native_limit_exceeded"] == "native_token_limit"
    assert meta["ground_cover_m"] == 2560
    assert meta["estimated_patch_tokens"] == 5120
    assert meta["effective_gsd_m"] == (10.0, 10.0)
    assert meta["gsd_mismatch"] is False


def test_prepare_thor_raw_chw_token_limit_falls_back_when_resize_cuts_tokens():
    # 376 px at patch_size=4 needs 11045 tokens; 288 px needs 6480.
    raw = np.zeros((10, 380, 380), dtype=np.float32)
    with pytest.warns(UserWarning, match=r"THOR resampled .*native_token_limit"):
        out, meta = _prepare_default(raw, patch_size=4)

    assert out.shape == (10, 380, 380)
    assert meta["preprocess_reason"] == "native_token_limit"
    assert meta["estimated_patch_tokens"] == 6480
    assert meta["gsd_mismatch"] is True


def test_prepare_thor_raw_chw_side_limit_and_token_limit_agree_on_larger_inputs():
    # Inputs past either limit both resize to 288 (6480 tokens), so a larger ROI
    # never runs where a smaller one would not.
    raw = np.zeros((10, 400, 400), dtype=np.float32)
    with pytest.warns(UserWarning, match="native_side_limit"):
        _, meta = _prepare_default(raw, patch_size=4)

    assert meta["preprocess_strategy"] == "fixed_resize"
    assert meta["estimated_patch_tokens"] == 6480


def test_prepare_thor_raw_chw_coarse_scale_over_token_limit_does_not_raise():
    # At 20 m THOR upsamples to the 10 m grid, so 296 px needs 6845 tokens; the
    # 288 px fallback (6480) is cheaper and still runs.
    raw = np.zeros((10, 300, 300), dtype=np.float32)
    with pytest.warns(UserWarning, match="native_token_limit"):
        _, meta = _prepare_default(raw, scale_m=20)

    assert meta["preprocess_reason"] == "native_token_limit"
    assert meta["estimated_patch_tokens"] == 6480


def test_prepare_thor_raw_chw_token_limit_can_be_raised_to_stay_native():
    import warnings

    raw = np.zeros((10, 256, 256), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out, meta = _prepare_default(raw, patch_size=4, max_native_tokens=6000)

    assert out.shape == (10, 256, 256)
    assert meta["preprocess_strategy"] == "native_snap"
    assert "native_limit_exceeded" not in meta
    assert meta["estimated_patch_tokens"] == 5120


def test_prepare_thor_raw_chw_fixed_mode_reports_real_gsd():
    raw = np.zeros((10, 256, 256), dtype=np.float32)
    with pytest.warns(UserWarning, match=r"real pixel spacing is 8\.89x8\.89 m"):
        _, meta = _prepare_default(raw, resize_mode="fixed")

    assert meta["preprocess_reason"] == "fixed_mode"
    assert meta["ground_cover_m"] == 2880
    assert meta["effective_gsd_m"] == pytest.approx((8.8889, 8.8889))
    assert meta["gsd_mismatch"] is True


def test_prepare_thor_raw_chw_rectangular_fallback_reports_per_axis_gsd():
    raw = np.zeros((10, 256, 300), dtype=np.float32)
    with pytest.warns(UserWarning, match="rectangular_above_tolerance"):
        _, meta = _prepare_default(raw)

    assert meta["effective_gsd_m"] == pytest.approx((8.8889, 10.4167))
    assert meta["gsd_mismatch"] is True


def test_prepare_thor_raw_chw_fixed_resize_at_model_size_is_silent():
    import warnings

    # Tiles already at image_size are not resampled, so the geometry is exact.
    raw = np.zeros((10, 288, 288), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, meta = _prepare_default(raw, input_prep_mode="tile")

    assert meta["preprocess_strategy"] == "fixed_resize"
    assert meta["ground_cover_m"] == 2880
    assert meta["effective_gsd_m"] == (10.0, 10.0)
    assert meta["gsd_mismatch"] is False


def test_thor_describe_image_size_follows_env(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    monkeypatch.delenv("RS_EMBED_THOR_IMG", raising=False)
    assert thor.THORBaseEmbedder().describe()["defaults"]["image_size"] == 288
    # The tiler's default tile size comes from here; it must match the size
    # THOR consumes, or every tile would be resampled.
    monkeypatch.setenv("RS_EMBED_THOR_IMG", "256")
    assert thor.THORBaseEmbedder().describe()["defaults"]["image_size"] == 256


def test_thor_malformed_image_size_env_fails_at_embed_time_not_in_describe(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor
    from rs_embed.core.errors import ModelError

    monkeypatch.setenv("RS_EMBED_THOR_IMG", "288px")
    assert thor.THORBaseEmbedder().describe()["defaults"]["image_size"] == 288
    with pytest.raises(ModelError, match="RS_EMBED_THOR_IMG must be an integer"):
        thor._resolve_thor_runtime_config(
            model_config=None,
            default_model_key=thor.THORBaseEmbedder.DEFAULT_MODEL_KEY,
            default_image_size=thor.THORBaseEmbedder.DEFAULT_IMAGE_SIZE,
        )


def test_thor_sub_tile_roi_under_default_tile_prep_keeps_native_geometry(monkeypatch):
    import warnings

    import rs_embed.embedders.onthefly_thor as thor
    from rs_embed.tools.tiling import _call_embedder_get_embedding_with_input_prep

    for var in (
        "RS_EMBED_THOR_IMG",
        "RS_EMBED_THOR_PATCH_SIZE",
        "RS_EMBED_THOR_RESIZE_MODE",
        "RS_EMBED_THOR_MAX_NATIVE_TOKENS",
        "RS_EMBED_THOR_MAX_NATIVE_SIDE",
    ):
        monkeypatch.delenv(var, raising=False)
    seen: dict[str, object] = {}

    def _fake_load_thor(**kwargs):
        seen["ground_cover"] = kwargs["ground_cover"]
        return object(), {"embed_dim": 8}, "cpu"

    def _fake_forward(model, x_chw, *, device, group_merge):
        seen["x_shape"] = tuple(x_chw.shape)
        return np.ones((16, 8), dtype=np.float32), None, {}

    monkeypatch.setattr(thor, "_load_thor", _fake_load_thor)
    monkeypatch.setattr(thor, "_thor_forward_single", _fake_forward)

    # A 2 km ROI (200 px) is smaller than one 288 px tile. The package-default
    # tile prep used to hand it to THOR with the tile config, which stretched it
    # to 288 px (6.94 m pixels told as 10 m).
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="THOR")
        out = _call_embedder_get_embedding_with_input_prep(
            embedder=thor.THORBaseEmbedder(),
            spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=1000),
            temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
            sensor=None,
            output=OutputSpec.pooled(),
            backend="gee",
            device="cpu",
            input_chw=np.zeros((10, 200, 200), dtype=np.float32),
            input_prep=None,
        )

    assert seen["x_shape"] == (10, 192, 192)
    assert seen["ground_cover"] == 1920
    assert out.meta["preprocess_strategy"] == "native_snap"
    assert out.meta["gsd_mismatch"] is False
    assert out.meta["input_prep"]["resolved_mode"] == "resize"


def test_thor_describe_exposes_s1_modality():
    import rs_embed.embedders.onthefly_thor as thor

    desc = thor.THORBaseEmbedder().describe()

    assert desc["defaults"]["modality"] == "s2"
    assert desc["modalities"]["s1"]["collection"] == "COPERNICUS/S1_GRD_FLOAT"
    assert tuple(desc["modalities"]["s1"]["bands"]) == ("VV", "VH")


def test_thor_get_embedding_s1_uses_vvvh_model_bands(monkeypatch):
    import rs_embed.embedders.onthefly_thor as thor

    class _FakeTHORS1Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.groups = {
                "group0": ["S1:IW-VH", "S1:IW-VV"],
            }
            self.out_channels = [8]
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, x):
            batch = x.shape[0]
            channel_params = {
                "S1:IW-VV": {"num_patch": 6},
                "S1:IW-VH": {"num_patch": 6},
            }
            features = [torch.arange(batch * 36 * 8, dtype=torch.float32).reshape(batch, 36, 8)]
            return features, channel_params

    emb = thor.THORBaseEmbedder()
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        thor,
        "_fetch_s1_vvvh_raw_chw_with_meta",
        lambda *args, **kwargs: (
            np.full((2, 288, 288), 2.0, dtype=np.float32),
            {"s1_iw_applied": True},
        ),
    )

    seen: dict[str, object] = {}

    def _fake_load_thor(**kwargs):
        seen["model_bands"] = kwargs["model_bands"]
        return _FakeTHORS1Model(), {"embed_dim": 8}, "cpu"

    monkeypatch.setattr(thor, "_load_thor", _fake_load_thor)

    out = emb.get_embedding(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=256),
        temporal=TemporalSpec.range("2020-06-01", "2020-08-31"),
        sensor=SensorSpec(
            collection="COPERNICUS/S1_GRD_FLOAT",
            bands=("VV", "VH"),
            modality="s1",
        ),
        output=OutputSpec.pooled(),
        backend="auto",
    )

    assert seen["model_bands"] == ("VV", "VH")
    assert out.data.shape == (8,)
    assert out.meta["modality"] == "s1"
    assert out.meta["sensor"]["bands"] == ("VV", "VH")
    assert out.meta["sensor"]["bands_thor"] == ("VV", "VH")


def test_enable_alibi_for_timm_patches_block_and_attention():
    from timm.models.vision_transformer import Attention, Block

    from rs_embed.embedders._vendor.thor.models.patch_timm import enable_alibi_for_timm

    enable_alibi_for_timm._done = False
    enable_alibi_for_timm()

    attn = Attention(dim=8, num_heads=2, qkv_bias=True)
    x = torch.randn(1, 4, 8)
    alibi = torch.zeros(1, 2, 4, 4)
    y = attn(x, alibi)
    assert y.shape == x.shape

    blk = Block(dim=8, num_heads=2, qkv_bias=True)
    z = blk(x, alibi)
    assert z.shape == x.shape
