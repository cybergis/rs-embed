import numpy as np

import rs_embed.api as inspect_mod
from rs_embed.core.specs import BBox, SensorSpec


def test_inspect_gee_patch_uses_shared_fetch_helper(monkeypatch):
    calls = {"helper": 0, "direct": 0}

    class _FakeProvider:
        def ensure_ready(self):
            return None

        def fetch_sensor_patch_chw(self, **kwargs):  # pragma: no cover - should not be called
            calls["direct"] += 1
            raise AssertionError("inspect_provider_patch should use _fetch_sensor_patch_chw helper")

    def _fake_create_provider(backend, *, allow_auto=True, auto_backend=None):
        assert backend == "gee"
        return _FakeProvider()

    def _fake_fetch(provider, *, spatial, temporal, sensor):
        calls["helper"] += 1
        assert isinstance(provider, _FakeProvider)
        assert isinstance(spatial, BBox)
        return np.ones((1, 2, 3), dtype=np.float32)

    monkeypatch.setattr(inspect_mod, "create_provider_for_backend", _fake_create_provider)
    monkeypatch.setattr(inspect_mod, "_fetch_sensor_patch_chw", _fake_fetch)

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
