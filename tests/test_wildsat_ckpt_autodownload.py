import os

import pytest

from rs_embed.core.errors import ModelError
import rs_embed.embedders.onthefly_wildsat as ws


_ENV_KEYS = [
    "RS_EMBED_WILDSAT_CKPT",
    "RS_EMBED_WILDSAT_AUTO_DOWNLOAD",
    "RS_EMBED_WILDSAT_CKPT_MIN_BYTES",
    "RS_EMBED_WILDSAT_CACHE_DIR",
    "RS_EMBED_WILDSAT_HF_REPO",
    "RS_EMBED_WILDSAT_HF_FILE",
    "RS_EMBED_WILDSAT_GDRIVE_ID",
    "RS_EMBED_WILDSAT_CKPT_FILE",
]


def _clear_wildsat_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_resolve_wildsat_ckpt_uses_local_env_path(monkeypatch, tmp_path):
    _clear_wildsat_env(monkeypatch)
    p = tmp_path / "local_wildsat.pth"
    p.write_bytes(b"123")
    monkeypatch.setenv("RS_EMBED_WILDSAT_CKPT", str(p))

    def _should_not_be_called(**_kw):
        raise AssertionError("auto-download should not be called when RS_EMBED_WILDSAT_CKPT is set")

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_hf", _should_not_be_called)
    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_gdrive", _should_not_be_called)

    assert ws._resolve_wildsat_ckpt_path() == os.path.expanduser(str(p))


def test_resolve_wildsat_ckpt_auto_download_disabled_raises(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_AUTO_DOWNLOAD", "0")

    with pytest.raises(ModelError, match="checkpoint is required"):
        ws._resolve_wildsat_ckpt_path()


def test_resolve_wildsat_ckpt_hf_vars_must_be_paired(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_HF_REPO", "foo/bar")

    with pytest.raises(
        ModelError,
        match="Set both RS_EMBED_WILDSAT_HF_REPO and RS_EMBED_WILDSAT_HF_FILE",
    ):
        ws._resolve_wildsat_ckpt_path()


def test_resolve_wildsat_ckpt_prefers_hf_when_configured(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    monkeypatch.setenv("RS_EMBED_WILDSAT_HF_REPO", "repo/name")
    monkeypatch.setenv("RS_EMBED_WILDSAT_HF_FILE", "model.pth")
    monkeypatch.setenv("RS_EMBED_WILDSAT_CACHE_DIR", "/tmp/wildsat-cache")
    monkeypatch.setenv("RS_EMBED_WILDSAT_CKPT_MIN_BYTES", "123")

    seen = {}

    def _fake_hf(*, repo_id, filename, cache_dir, min_bytes):
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["cache_dir"] = cache_dir
        seen["min_bytes"] = min_bytes
        return "/tmp/from_hf/model.pth"

    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_hf", _fake_hf)
    monkeypatch.setattr(
        ws,
        "_download_wildsat_ckpt_from_gdrive",
        lambda **_kw: "/tmp/should_not_happen.pth",
    )

    out = ws._resolve_wildsat_ckpt_path()
    assert out == "/tmp/from_hf/model.pth"
    assert seen == {
        "repo_id": "repo/name",
        "filename": "model.pth",
        "cache_dir": "/tmp/wildsat-cache",
        "min_bytes": 123,
    }


def test_resolve_wildsat_ckpt_uses_default_gdrive_source(monkeypatch):
    _clear_wildsat_env(monkeypatch)
    seen = {}

    def _fake_gdrive(*, file_id, cache_dir, filename, min_bytes):
        seen["file_id"] = file_id
        seen["cache_dir"] = cache_dir
        seen["filename"] = filename
        seen["min_bytes"] = min_bytes
        return "/tmp/from_gdrive/model.pth"

    monkeypatch.setattr(
        ws, "_download_wildsat_ckpt_from_hf", lambda **_kw: "/tmp/should_not_happen.pth"
    )
    monkeypatch.setattr(ws, "_download_wildsat_ckpt_from_gdrive", _fake_gdrive)

    out = ws._resolve_wildsat_ckpt_path()
    assert out == "/tmp/from_gdrive/model.pth"
    assert seen["file_id"] == ws._WILDSAT_DEFAULT_GDRIVE_FILE_ID
    assert seen["filename"] == ws._WILDSAT_DEFAULT_CKPT_FILENAME
    assert seen["min_bytes"] == ws._WILDSAT_DEFAULT_MIN_BYTES
