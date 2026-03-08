from rs_embed.core.specs import PointBuffer
from rs_embed.export import export_npz


def test_export_npz_delegates(monkeypatch, tmp_path):
    captured = {}

    def _fake_export_batch(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("rs_embed.api.export_batch", _fake_export_batch)

    out = export_npz(
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        temporal=None,
        models=["mock_model"],
        out_path=str(tmp_path / "one"),
        save_inputs=False,
        save_embeddings=False,
        save_manifest=False,
    )

    assert out == {"ok": True}
    assert captured["format"] == "npz"
    assert captured["spatials"]
    assert captured["out_path"].endswith(".npz")
