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


# ---------------------------------------------------------------------------
# inspect_model_input
# ---------------------------------------------------------------------------


class _FakeInspectProvider:
    def ensure_ready(self):
        return None


def _fake_provider_factory(backend, *, allow_auto=True, auto_backend=None):
    assert backend == "gee"
    return _FakeInspectProvider()


def test_inspect_model_input_multiframe_reports_per_bin(monkeypatch):
    """Multi-frame fetch_input results are split into per-bin frame reports,
    with empty (all-NaN sentinel) bins flagged instead of inspected."""
    from rs_embed.core.types import FetchResult

    bins = [("2022-01-01", "2022-01-31"), ("2022-01-31", "2022-03-02")]

    class _FakeEmbedder:
        def fetch_input(self, provider, *, spatial, temporal, sensor):
            x = np.full((2, 3, 4, 4), np.nan, dtype=np.float32)
            x[1] = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + 100.0
            meta = {
                "frames": [
                    {"start": bins[0][0], "end": bins[0][1], "empty": True},
                    {"start": bins[1][0], "end": bins[1][1], "empty": False},
                ],
                "n_empty": 1,
                "temporal_mode": "multi",
            }
            return FetchResult(data=x, meta=meta)

    monkeypatch.setattr(inspect_mod, "create_provider_for_backend", _fake_provider_factory)
    monkeypatch.setattr(inspect_mod, "_get_embedder_cls", lambda name: _FakeEmbedder)

    out = inspect_mod.inspect_model_input(
        "fakemodel",
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        sensor=SensorSpec(collection="FAKE/COLL", bands=("B1", "B2", "B3")),
        return_arrays=True,
    )

    assert out["model"] == "fakemodel"
    assert out["n_frames"] == 2
    assert out["n_empty"] == 1
    assert out["ok"] is True
    empty_frame, full_frame = out["frames"]
    assert empty_frame["empty"] is True
    assert "report" not in empty_frame
    assert (empty_frame["start"], empty_frame["end"]) == bins[0]
    assert full_frame["empty"] is False
    assert full_frame["report"]["ok"] is True
    assert full_frame["array_chw"].shape == (3, 4, 4)
    assert out["fetch_meta"] == {"temporal_mode": "multi"}


def test_inspect_model_input_single_frame_uses_resolved_temporal(monkeypatch):
    """A [C,H,W] fetch_input result becomes a single frame entry labeled with
    the resolved temporal range."""
    from rs_embed.core.specs import TemporalSpec
    from rs_embed.core.types import FetchResult

    class _FakeEmbedder:
        def fetch_input(self, provider, *, spatial, temporal, sensor):
            x = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4) + 100.0
            return FetchResult(data=x, meta={})

    monkeypatch.setattr(inspect_mod, "create_provider_for_backend", _fake_provider_factory)
    monkeypatch.setattr(inspect_mod, "_get_embedder_cls", lambda name: _FakeEmbedder)

    out = inspect_mod.inspect_model_input(
        "fakemodel",
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
        sensor=SensorSpec(collection="FAKE/COLL", bands=("B1", "B2", "B3")),
    )

    assert out["n_frames"] == 1
    assert out["n_empty"] == 0
    (frame,) = out["frames"]
    assert (frame["start"], frame["end"]) == ("2022-06-01", "2022-09-01")
    assert frame["report"]["ok"] is True
    assert "array_chw" not in frame


def test_inspect_model_input_rejects_precomputed(monkeypatch):
    from rs_embed.core.errors import ModelError

    class _FakePrecomputed:
        _is_precomputed = True

    monkeypatch.setattr(inspect_mod, "_get_embedder_cls", lambda name: _FakePrecomputed)

    try:
        inspect_mod.inspect_model_input(
            "fakemodel",
            spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
            sensor=SensorSpec(collection="FAKE/COLL", bands=("B1",)),
        )
    except ModelError as exc:
        assert "precomputed" in str(exc)
    else:
        raise AssertionError("expected ModelError for precomputed model")
