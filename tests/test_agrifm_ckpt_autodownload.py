import os

import pytest

from rs_embed.core.errors import ModelError
import rs_embed.embedders.onthefly_agrifm as ag


_ENV_KEYS = [
    "RS_EMBED_AGRIFM_CKPT",
    "RS_EMBED_AGRIFM_AUTO_DOWNLOAD",
    "RS_EMBED_AGRIFM_CKPT_URL",
    "RS_EMBED_AGRIFM_CACHE_DIR",
    "RS_EMBED_AGRIFM_CKPT_FILE",
    "RS_EMBED_AGRIFM_CKPT_MIN_BYTES",
]


def _clear_agrifm_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_resolve_agrifm_ckpt_uses_local_env_path(monkeypatch, tmp_path):
    _clear_agrifm_env(monkeypatch)
    p = tmp_path / "local_agrifm.pth"
    p.write_bytes(b"123")
    monkeypatch.setenv("RS_EMBED_AGRIFM_CKPT", str(p))

    def _should_not_be_called(**_kw):
        raise AssertionError("auto-download should not be called when RS_EMBED_AGRIFM_CKPT is set")

    monkeypatch.setattr(ag, "_download_agrifm_ckpt", _should_not_be_called)
    assert ag._resolve_ckpt_path() == os.path.expanduser(str(p))


def test_resolve_agrifm_ckpt_local_missing_raises(monkeypatch):
    _clear_agrifm_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_AGRIFM_CKPT", "/tmp/not_exist_agrifm_xxx.pth")

    with pytest.raises(ModelError, match="does not exist"):
        ag._resolve_ckpt_path()


def test_resolve_agrifm_ckpt_auto_download_disabled_raises(monkeypatch):
    _clear_agrifm_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_AGRIFM_AUTO_DOWNLOAD", "0")

    with pytest.raises(ModelError, match="checkpoint is required"):
        ag._resolve_ckpt_path()


def test_resolve_agrifm_ckpt_uses_default_auto_download(monkeypatch):
    _clear_agrifm_env(monkeypatch)
    seen = {}

    def _fake_download(*, url, cache_dir, filename, min_bytes):
        seen["url"] = url
        seen["cache_dir"] = cache_dir
        seen["filename"] = filename
        seen["min_bytes"] = min_bytes
        return "/tmp/agrifm_from_auto/AgriFM.pth"

    monkeypatch.setattr(ag, "_download_agrifm_ckpt", _fake_download)
    out = ag._resolve_ckpt_path()
    assert out == "/tmp/agrifm_from_auto/AgriFM.pth"
    assert seen["url"] == ag._AGRIFM_DEFAULT_CKPT_URL
    assert seen["filename"] == ag._AGRIFM_DEFAULT_CKPT_FILENAME
    assert seen["min_bytes"] == ag._AGRIFM_DEFAULT_MIN_BYTES
