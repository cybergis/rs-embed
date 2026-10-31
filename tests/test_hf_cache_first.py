import sys

import pytest

from rs_embed.core.errors import ModelError
from rs_embed.embedders.shared import (
    hf_hub_download_cache_first,
    resolve_pretrained_source_cache_first,
    snapshot_download_cache_first,
)


class _RecordingHub:
    """Fake huggingface_hub recording the local_files_only flag of each call."""

    def __init__(self, *, local_result=None, network_result=None, local_error=None):
        self.calls = []
        self._local_result = local_result
        self._network_result = network_result
        self._local_error = local_error

    def _dispatch(self, kwargs):
        self.calls.append(bool(kwargs.get("local_files_only")))
        if kwargs.get("local_files_only"):
            if self._local_error is not None:
                raise self._local_error
            return self._local_result
        return self._network_result

    def hf_hub_download(self, **kwargs):
        return self._dispatch(kwargs)

    def snapshot_download(self, **kwargs):
        return self._dispatch(kwargs)


@pytest.fixture
def install_hub(monkeypatch):
    def _install(hub):
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        return hub

    return _install


def test_hf_hub_download_cache_hit_never_touches_network(install_hub):
    hub = install_hub(_RecordingHub(local_result="/cache/weights.pt"))
    assert hf_hub_download_cache_first(repo_id="org/model", filename="weights.pt") == (
        "/cache/weights.pt"
    )
    assert hub.calls == [True]


def test_hf_hub_download_cache_miss_falls_back_to_network(install_hub):
    hub = install_hub(
        _RecordingHub(local_error=FileNotFoundError("not cached"), network_result="/dl/weights.pt")
    )
    assert hf_hub_download_cache_first(repo_id="org/model", filename="weights.pt") == (
        "/dl/weights.pt"
    )
    assert hub.calls == [True, False]


def test_hf_hub_download_missing_hub_raises_model_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ModelError, match="huggingface_hub"):
        hf_hub_download_cache_first(repo_id="org/model", filename="weights.pt")


def test_snapshot_cache_hit_with_passing_validate(install_hub, tmp_path):
    (tmp_path / "encoder.pt").write_text("x", encoding="utf-8")
    hub = install_hub(_RecordingHub(local_result=str(tmp_path)))
    snap = snapshot_download_cache_first(
        repo_id="org/model",
        validate=lambda d: (tmp_path / "encoder.pt").is_file(),
    )
    assert snap == str(tmp_path)
    assert hub.calls == [True]


def test_snapshot_incomplete_cache_falls_back_to_network(install_hub, tmp_path):
    hub = install_hub(_RecordingHub(local_result=str(tmp_path), network_result="/dl/snap"))
    snap = snapshot_download_cache_first(
        repo_id="org/model",
        validate=lambda d: False,
    )
    assert snap == "/dl/snap"
    assert hub.calls == [True, False]


def test_snapshot_local_files_only_reraises_cache_miss(install_hub):
    install_hub(_RecordingHub(local_error=FileNotFoundError("not cached")))
    with pytest.raises(FileNotFoundError):
        snapshot_download_cache_first(repo_id="org/model", local_files_only=True)


def test_snapshot_local_files_only_incomplete_cache_raises_model_error(install_hub, tmp_path):
    install_hub(_RecordingHub(local_result=str(tmp_path)))
    with pytest.raises(ModelError, match="incomplete"):
        snapshot_download_cache_first(
            repo_id="org/model", local_files_only=True, validate=lambda d: False
        )


def test_resolve_pretrained_source_returns_cached_dir(install_hub, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text("x", encoding="utf-8")
    hub = install_hub(_RecordingHub(local_result=str(tmp_path)))
    assert resolve_pretrained_source_cache_first("org/model") == str(tmp_path)
    assert hub.calls == [True]


def test_resolve_pretrained_source_keeps_model_id_on_incomplete_cache(install_hub, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    install_hub(_RecordingHub(local_result=str(tmp_path)))
    assert resolve_pretrained_source_cache_first("org/model") == "org/model"


def test_resolve_pretrained_source_keeps_model_id_on_cache_miss(install_hub):
    install_hub(_RecordingHub(local_error=FileNotFoundError("not cached")))
    assert resolve_pretrained_source_cache_first("org/model") == "org/model"


def test_resolve_pretrained_source_passes_local_paths_through(install_hub, tmp_path):
    hub = install_hub(_RecordingHub())
    assert resolve_pretrained_source_cache_first(str(tmp_path)) == str(tmp_path)
    assert hub.calls == []


def test_resolve_pretrained_source_survives_missing_hub(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert resolve_pretrained_source_cache_first("org/model") == "org/model"
