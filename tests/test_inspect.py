import numpy as np

from rs_embed.core.specs import BBox, SensorSpec
import rs_embed.inspect as inspect_mod


def test_inspect_gee_patch_uses_shared_fetch_helper(monkeypatch):
    calls = {"helper": 0, "direct": 0}

    class _FakeProvider:
        def ensure_ready(self):
            return None

        def fetch_sensor_patch_chw(
            self, **kwargs
        ):  # pragma: no cover - should not be called
            calls["direct"] += 1
            raise AssertionError(
                "inspect_provider_patch should use fetch_provider_patch_raw helper"
            )

    def _fake_get_provider(name, **kwargs):
        assert name == "gee"
        return _FakeProvider()

    def _fake_fetch(provider, *, spatial, temporal, sensor):
        calls["helper"] += 1
        assert isinstance(provider, _FakeProvider)
        assert isinstance(spatial, BBox)
        return np.ones((1, 2, 3), dtype=np.float32)

    monkeypatch.setattr(inspect_mod, "get_provider", _fake_get_provider)
    monkeypatch.setattr(inspect_mod, "fetch_provider_patch_raw", _fake_fetch)

    out = inspect_mod.inspect_gee_patch(
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=None,
        sensor=SensorSpec(collection="FAKE/COLL", bands=("B1",)),
        return_array=True,
    )

    assert calls["helper"] == 1
    assert calls["direct"] == 0
    assert out["array_chw"].shape == (1, 2, 3)
    assert out["backend"] == "gee"
