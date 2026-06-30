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


def test_inspect_fetch_result_skips_all_nan_leading_frame(recwarn):
    """A TCHW stack whose first frame is an empty (all-NaN) temporal bin must be
    inspected via the first finite frame, not frame 0 — so no NaN-reduction
    RuntimeWarning leaks out and the report reflects real imagery."""
    from rs_embed.providers import fetch as fetch_mod

    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1", "B2"))
    x_tchw = np.full((3, 2, 8, 8), np.nan, dtype=np.float32)
    x_tchw[1] = 1500.0  # only the second bin had usable imagery

    out = fetch_mod.inspect_fetch_result(x_tchw, sensor=sensor, name="t")

    assert out["inspected_frame"] == 1
    assert out["n_frames"] == 3
    assert out["report"]["finite_frac"] == 1.0
    assert not any(w.category is RuntimeWarning for w in recwarn.list)


def test_inspect_fetch_result_all_empty_frames_falls_back_to_frame_zero():
    """If every frame is empty there is no finite frame to pick; fall back to
    frame 0 so inspect_chw still flags the bad input rather than indexing past."""
    from rs_embed.providers import fetch as fetch_mod

    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1", "B2"))
    x_tchw = np.full((2, 2, 4, 4), np.nan, dtype=np.float32)

    out = fetch_mod.inspect_fetch_result(x_tchw, sensor=sensor, name="t")

    assert out["inspected_frame"] == 0
    assert out["ok"] is False
