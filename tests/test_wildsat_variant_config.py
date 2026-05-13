import os

import numpy as np
import pytest

from rs_embed.core.errors import ModelError
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_wildsat as ws

_ENV_KEYS = [
    "RS_EMBED_WILDSAT_CKPT",
    "RS_EMBED_WILDSAT_AUTO_DOWNLOAD",
    "RS_EMBED_WILDSAT_CKPT_MIN_BYTES",
    "RS_EMBED_WILDSAT_CACHE_DIR",
    "RS_EMBED_WILDSAT_HF_REPO",
    "RS_EMBED_WILDSAT_HF_FILE",
    "RS_EMBED_WILDSAT_GDRIVE_ID",
    "RS_EMBED_WILDSAT_CKPT_FILE",
    "RS_EMBED_WILDSAT_ARCH",
]


def _clear_wildsat_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# _normalize_wildsat_variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("vitb16", "vitb16"),
        ("VITB16", "vitb16"),
        ("vit-b-16", "vitb16"),
        ("vit_base_16", "vitb16"),
        ("vit", "vitb16"),
        ("resnet50", "resnet50"),
        ("ResNet50", "resnet50"),
        ("resnet", "resnet50"),
        ("swint", "swint"),
        ("swin-t", "swint"),
        ("swin_tiny", "swint"),
        ("swin", "swint"),
    ],
)
def test_normalize_wildsat_variant_valid(raw, expected):
    assert ws._normalize_wildsat_variant(raw) == expected


def test_normalize_wildsat_variant_raises_on_unknown():
    with pytest.raises(ModelError, match="Unknown WildSAT variant"):
        ws._normalize_wildsat_variant("mobilenet")


# ---------------------------------------------------------------------------
# _resolve_wildsat_variant
# ---------------------------------------------------------------------------


def test_resolve_variant_from_model_config():
    assert ws._resolve_wildsat_variant(model_config={"variant": "swint"}) == "swint"
    assert ws._resolve_wildsat_variant(model_config={"variant": "resnet50"}) == "resnet50"
    assert ws._resolve_wildsat_variant(model_config={"variant": "vit"}) == "vitb16"


def test_resolve_variant_defaults_to_vitb16():
    assert ws._resolve_wildsat_variant(model_config=None) == "vitb16"
    assert ws._resolve_wildsat_variant(model_config={}) == "vitb16"


def test_resolve_variant_env_fallback(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_ARCH", "resnet50")
    assert ws._resolve_wildsat_variant(model_config=None) == "resnet50"


def test_resolve_variant_model_config_overrides_env(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_ARCH", "resnet50")
    assert ws._resolve_wildsat_variant(model_config={"variant": "swint"}) == "swint"


def test_resolve_variant_env_auto_gives_default(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_ARCH", "auto")
    assert ws._resolve_wildsat_variant(model_config=None) == "vitb16"


# ---------------------------------------------------------------------------
# _resolve_wildsat_ckpt_path_for_variant
# ---------------------------------------------------------------------------


def test_ckpt_path_local_env_overrides_variant(monkeypatch, tmp_path):
    _clear_wildsat_env(monkeypatch)
    p = tmp_path / "custom.pth"
    p.write_bytes(b"123")
    monkeypatch.setenv("RS_EMBED_WILDSAT_CKPT", str(p))

    def _should_not_be_called(**_kw):
        raise AssertionError("auto-download should not be called")

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_hf", _should_not_be_called)
    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_gdrive", _should_not_be_called)

    assert ws._resolve_wildsat_ckpt_path_for_variant("swint") == os.path.expanduser(str(p))


def test_ckpt_path_auto_download_disabled_raises(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_AUTO_DOWNLOAD", "0")

    with pytest.raises(ModelError, match="checkpoint is required"):
        ws._resolve_wildsat_ckpt_path_for_variant("vitb16")


def test_ckpt_path_uses_variant_gdrive_id(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    seen = {}

    def _fake_gdrive(*, file_id, cache_dir, filename, min_bytes):
        seen["file_id"] = file_id
        seen["filename"] = filename
        return "/tmp/from_gdrive/model.pth"

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_gdrive", _fake_gdrive)

    for variant, spec in ws._WILDSAT_VARIANT_SPECS.items():
        ws._resolve_wildsat_ckpt_path_for_variant(variant)
        assert seen["file_id"] == spec["gdrive_id"], f"variant={variant}"
        assert seen["filename"] == spec["filename"], f"variant={variant}"


def test_ckpt_path_default_variant_uses_vitb16_gdrive(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    seen = {}

    def _fake_gdrive(*, file_id, cache_dir, filename, min_bytes):
        seen["file_id"] = file_id
        seen["filename"] = filename
        return "/tmp/from_gdrive/model.pth"

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_gdrive", _fake_gdrive)

    ws._resolve_wildsat_ckpt_path_for_variant("vitb16")
    assert seen["file_id"] == ws._WILDSAT_VARIANT_SPECS["vitb16"]["gdrive_id"]
    assert seen["filename"] == ws._WILDSAT_VARIANT_SPECS["vitb16"]["filename"]


def test_ckpt_path_hf_overrides_variant_gdrive(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_HF_REPO", "repo/name")
    monkeypatch.setenv("RS_EMBED_WILDSAT_HF_FILE", "model.pth")
    seen = {}

    def _fake_hf(*, repo_id, filename, cache_dir, min_bytes):
        seen["repo_id"] = repo_id
        return "/tmp/from_hf/model.pth"

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_hf", _fake_hf)
    monkeypatch.setattr(
        ws,
        "_download_wildsat_ckpt_from_gdrive",
        lambda **_kw: (_ for _ in ()).throw(AssertionError("gdrive should not be called")),
    )

    out = ws._resolve_wildsat_ckpt_path_for_variant("resnet50")
    assert out == "/tmp/from_hf/model.pth"
    assert seen["repo_id"] == "repo/name"


# ---------------------------------------------------------------------------
# _WILDSAT_VARIANT_SPECS consistency
# ---------------------------------------------------------------------------


def test_variant_specs_keys_match_supported_arches():
    for variant, spec in ws._WILDSAT_VARIANT_SPECS.items():
        assert spec["arch"] in ws._SUPPORTED_ARCHES, (
            f"Variant '{variant}' has arch '{spec['arch']}' not in _SUPPORTED_ARCHES"
        )


def test_variant_specs_have_required_fields():
    for variant, spec in ws._WILDSAT_VARIANT_SPECS.items():
        assert "arch" in spec, f"variant={variant} missing 'arch'"
        assert "gdrive_id" in spec, f"variant={variant} missing 'gdrive_id'"
        assert "filename" in spec, f"variant={variant} missing 'filename'"


def test_vitl16_not_in_supported_arches():
    assert "vitl16" not in ws._SUPPORTED_ARCHES


# ---------------------------------------------------------------------------
# get_embeddings_batch forwards model_config
# ---------------------------------------------------------------------------


def test_wildsat_get_embeddings_batch_forwards_model_config(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    emb = ws.WildSATEmbedder()
    seen = {}

    original_get_embedding = emb.get_embedding

    def _capture_get_embedding(**kwargs):
        seen["model_config"] = kwargs.get("model_config")
        return original_get_embedding(**kwargs)

    monkeypatch.setattr(emb, "get_embedding", lambda **kw: _capture_get_embedding(**kw))
    monkeypatch.setattr(emb, "_get_provider", lambda _backend: object())
    monkeypatch.setattr(
        ws,
        "_fetch_s2_rgb_chw",
        lambda *args, **kwargs: np.full((3, 8, 8), 500.0, dtype=np.float32),
    )

    # Mock the heavy model loading and forward pass
    def _fake_load_wildsat(*, ckpt_path, arch_hint, prefer_branch, device):
        import types

        fake_backbone = types.SimpleNamespace()
        return fake_backbone, None, {"arch": arch_hint}, "cpu"

    monkeypatch.setattr(ws, "_load_wildsat", _fake_load_wildsat)
    monkeypatch.setattr(
        ws,
        "_wildsat_forward",
        lambda *args, **kwargs: (
            np.zeros(128, dtype=np.float32),
            None,
            {"feature_source": "backbone", "feature_dim": 128},
        ),
    )

    out = emb.get_embeddings_batch(
        spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=256)],
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        model_config={"variant": "swint"},
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(out) == 1
    assert seen["model_config"] == {"variant": "swint"}


# ---------------------------------------------------------------------------
# get_embedding uses variant to select arch and checkpoint
# ---------------------------------------------------------------------------


def test_wildsat_get_embedding_uses_variant_for_arch(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    emb = ws.WildSATEmbedder()
    seen = {}

    def _fake_load_wildsat(*, ckpt_path, arch_hint, prefer_branch, device):
        import types

        seen["arch_hint"] = arch_hint
        seen["ckpt_path"] = ckpt_path
        fake_backbone = types.SimpleNamespace()
        return fake_backbone, None, {"arch": arch_hint}, "cpu"

    monkeypatch.setattr(ws, "_load_wildsat", _fake_load_wildsat)
    monkeypatch.setattr(
        ws,
        "_wildsat_forward",
        lambda *args, **kwargs: (
            np.zeros(128, dtype=np.float32),
            None,
            {"feature_source": "backbone", "feature_dim": 128},
        ),
    )
    monkeypatch.setattr(
        ws,
        "_resolve_wildsat_ckpt_path_for_variant",
        lambda variant: f"/fake/{variant}.pth",
    )

    emb.get_embedding(
        spatial=PointBuffer(lon=1.0, lat=2.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((3, 8, 8), 5000.0, dtype=np.float32),
        model_config={"variant": "resnet50"},
    )

    assert seen["arch_hint"] == "resnet50"
    assert seen["ckpt_path"] == "/fake/resnet50.pth"


def test_wildsat_get_embedding_default_variant(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    emb = ws.WildSATEmbedder()
    seen = {}

    def _fake_load_wildsat(*, ckpt_path, arch_hint, prefer_branch, device):
        import types

        seen["arch_hint"] = arch_hint
        fake_backbone = types.SimpleNamespace()
        return fake_backbone, None, {"arch": arch_hint}, "cpu"

    monkeypatch.setattr(ws, "_load_wildsat", _fake_load_wildsat)
    monkeypatch.setattr(
        ws,
        "_wildsat_forward",
        lambda *args, **kwargs: (
            np.zeros(128, dtype=np.float32),
            None,
            {"feature_source": "backbone", "feature_dim": 128},
        ),
    )
    monkeypatch.setattr(
        ws,
        "_resolve_wildsat_ckpt_path_for_variant",
        lambda variant: f"/fake/{variant}.pth",
    )

    emb.get_embedding(
        spatial=PointBuffer(lon=1.0, lat=2.0, buffer_m=256),
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=np.full((3, 8, 8), 5000.0, dtype=np.float32),
    )

    assert seen["arch_hint"] == "vitb16"
