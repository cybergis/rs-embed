import logging
import sys
import types


def _install_fake_rshf_remoteclip(monkeypatch, messages):
    import rs_embed.embedders.onthefly_remoteclip as rc

    rshf_mod = types.ModuleType("rshf")
    remoteclip_mod = types.ModuleType("rshf.remoteclip")

    class FakeRemoteCLIP:
        @classmethod
        def from_pretrained(cls, _model_id):
            for msg in messages:
                logging.warning(msg)
            return object()

    remoteclip_mod.RemoteCLIP = FakeRemoteCLIP
    rshf_mod.remoteclip = remoteclip_mod
    monkeypatch.setitem(sys.modules, "rshf", rshf_mod)
    monkeypatch.setitem(sys.modules, "rshf.remoteclip", remoteclip_mod)

    monkeypatch.setattr(rc, "_ensure_hf_weights", lambda *a, **k: ("/nonexistent/local_dir", None))
    monkeypatch.setattr(
        rc,
        "_assert_weights_loaded",
        lambda _model: {"param_mean": 0.1, "param_std": 0.2, "param_absmax": 1.0},
    )
    return rc


def test_remoteclip_suppresses_known_misleading_weight_warning(monkeypatch, caplog):
    rc = _install_fake_rshf_remoteclip(
        monkeypatch,
        ["No pretrained weights loaded for model 'ViT-B-32'. Model initialized randomly."],
    )

    with caplog.at_level(logging.WARNING):
        _, meta = rc._load_rshf_remoteclip("fake/repo")

    msgs = [r.getMessage() for r in caplog.records]
    assert not any("No pretrained weights loaded for model" in m for m in msgs)
    assert meta["init_warning_suppressed_count"] == 1
    assert meta["weights_verified"] is True


def test_remoteclip_keeps_other_warnings_visible(monkeypatch, caplog):
    rc = _install_fake_rshf_remoteclip(
        monkeypatch,
        [
            "No pretrained weights loaded for model 'ViT-B-32'. Model initialized randomly.",
            "another warning that should remain visible",
        ],
    )

    with caplog.at_level(logging.WARNING):
        _, meta = rc._load_rshf_remoteclip("fake/repo")

    msgs = [r.getMessage() for r in caplog.records]
    assert any("another warning that should remain visible" in m for m in msgs)
    assert not any("No pretrained weights loaded for model" in m for m in msgs)
    assert meta["init_warning_suppressed_count"] == 1
