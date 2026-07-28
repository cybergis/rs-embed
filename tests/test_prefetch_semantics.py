"""Temporal fetch-semantics separation in the export prefetch plan.

Models may share a prefetched input (sensor dedup / band-union merging) only
when their temporal fetch semantics match: a whole-window composite is not
derivable from a binned series or vice versa. These tests pin the mechanism
(semantics-qualified cache/plan keys) and the end-to-end regression that
motivated it: exporting olmoearth alongside terrafm silently degraded
olmoearth's [T,C,H,W] series to a single composite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rs_embed.core.registry import get_embedder_cls
from rs_embed.core.specs import PointBuffer, SensorSpec, TemporalSpec
from rs_embed.providers.prefetch_plan import build_prefetch_plan
from rs_embed.tools.runtime import fetch_semantics_for_model
from rs_embed.tools.serialization import input_cache_key, sensor_cache_key

# ── fingerprint + key units ─────────────────────────────────────────


def test_fetch_semantics_classification():
    assert fetch_semantics_for_model("olmoearth").startswith("custom:")
    assert fetch_semantics_for_model("agrifm") == "multi:8"
    assert fetch_semantics_for_model("anysat") == "multi:8"
    assert fetch_semantics_for_model("clay") == "single"
    assert fetch_semantics_for_model("terramind") == "single"


def test_input_cache_key_qualifies_non_single_semantics():
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1", "B2"))
    bare = sensor_cache_key(sensor)
    assert input_cache_key(sensor, "single") == bare
    multi = input_cache_key(sensor, "multi:8")
    custom = input_cache_key(sensor, "custom:FakeEmbedder")
    assert multi != bare and custom != bare and multi != custom
    assert multi.startswith(bare)


def test_plan_splits_same_sensor_different_semantics():
    """One SensorSpec, two temporal semantics -> two fetch groups, never one."""
    sensor = SensorSpec(collection="FAKE/COLL", bands=("B1", "B2"))
    _, fetch_sensor_by_key, sensor_to_fetch, _, fetch_members = build_prefetch_plan(
        models=["m_single", "m_multi"],
        resolved_sensor={"m_single": sensor, "m_multi": sensor},
        model_type={"m_single": "on_the_fly", "m_multi": "on_the_fly"},
        fetch_semantics_by_model={"m_single": "single", "m_multi": "multi:8"},
    )
    assert len(fetch_sensor_by_key) == 2
    keys = {input_cache_key(sensor, s) for s in ("single", "multi:8")}
    assert set(sensor_to_fetch) == keys


def test_plan_merges_same_semantics_band_union():
    """Same semantics still band-union merges (dedup preserved)."""
    s_a = SensorSpec(collection="FAKE/COLL", bands=("B1", "B2"))
    s_b = SensorSpec(collection="FAKE/COLL", bands=("B2", "B3"))
    _, fetch_sensor_by_key, _, _, fetch_members = build_prefetch_plan(
        models=["a", "b"],
        resolved_sensor={"a": s_a, "b": s_b},
        model_type={"a": "on_the_fly", "b": "on_the_fly"},
        fetch_semantics_by_model={"a": "multi:8", "b": "multi:8"},
    )
    assert len(fetch_sensor_by_key) == 1
    (fetch_sensor,) = fetch_sensor_by_key.values()
    assert fetch_sensor.bands == ("B1", "B2", "B3")
    (members,) = fetch_members.values()
    assert len(members) == 2


# ── end-to-end export regression ────────────────────────────────────


class _FakeProvider:
    """Answers both fetch styles; counts calls to assert dedup behavior."""

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.n_single = 0
        self.n_multi = 0

    def ensure_ready(self):
        return None

    def fetch_sensor_patch_chw(self, *, sensor=None, **kw):
        self.n_single += 1
        return (self.rng.random((len(sensor.bands), 24, 24)) * 3000).astype(np.float32)

    def fetch_multiframe_collection_raw_tchw(self, *, bands=None, n_frames=8, **kw):
        self.n_multi += 1
        return (self.rng.random((n_frames, len(bands), 24, 24)) * 3000).astype(np.float32)


@pytest.fixture
def fake_export(monkeypatch, tmp_path):
    import rs_embed.api as api_mod
    import rs_embed.tools.runtime as runtime_mod
    from rs_embed import ExportConfig, ExportTarget, OutputSpec, export_batch

    provider = _FakeProvider()
    monkeypatch.setattr(api_mod, "provider_factory_for_backend", lambda b: lambda: provider)
    monkeypatch.setattr(runtime_mod, "provider_factory_for_backend", lambda b: lambda: provider)

    def _run(models: list[str]) -> dict[str, tuple[int, ...]]:
        out_dir = tmp_path / "_".join(models)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export_batch(
                spatials=[PointBuffer(lon=121.5, lat=31.2, buffer_m=500)],
                temporal=TemporalSpec.range("2019-06-01", "2019-08-31"),
                models=models,
                target=ExportTarget.per_item(str(out_dir), names=["p0"]),
                output=OutputSpec.pooled(),
                config=ExportConfig(save_inputs=True, save_embeddings=False, show_progress=False),
                backend="gee",
            )
        z = np.load(out_dir / "p0.npz", allow_pickle=True)
        return {k.removeprefix("input_chw__"): z[k].shape for k in z.files if k.startswith("input")}

    return _run, provider


def test_export_olmoearth_keeps_series_alongside_terrafm(fake_export):
    """The demo_export regression: co-exporting terrafm must not collapse
    olmoearth's 30-day-binned [T,C,H,W] input to a single composite."""
    run, _provider = fake_export
    shapes = run(["olmoearth", "terrafm"])
    assert len(shapes["olmoearth"]) == 4  # [T,C,H,W]
    assert shapes["olmoearth"][0] >= 2  # 2019-06..08 spans multiple 30-day bins
    assert len(shapes["terrafm"]) == 3  # single composite untouched


def test_export_same_sensor_split_by_semantics(fake_export):
    """clay and agrifm resolve to the same SensorSpec but different temporal
    semantics: each must get its own fetch with its own shape."""
    run, provider = fake_export
    shapes = run(["agrifm", "clay"])
    assert shapes["agrifm"][:2] == (8, 10)  # equal-division 8-frame series
    assert shapes["clay"] == (10, 24, 24)
    assert provider.n_multi == 1 and provider.n_single == 1


def test_export_same_semantics_still_dedups(fake_export):
    """agrifm and anysat share sensor AND semantics -> one multi fetch total."""
    run, provider = fake_export
    shapes = run(["agrifm", "anysat"])
    assert provider.n_multi == 1 and provider.n_single == 0
    # dedup: the shared input is stored once under the first model's key
    assert any(s[:2] == (8, 10) for s in shapes.values())


def test_all_registered_models_have_a_semantics_fingerprint():
    """Every provider-backed embedder classifies cleanly; unknown shapes would
    silently fall into the 'single' pool and could regress temporal fetches."""
    from rs_embed.embedders.catalog import MODEL_SPECS

    for model_id in MODEL_SPECS:
        try:
            cls = get_embedder_cls(model_id)
        except Exception:
            continue  # optional-dep import failure: not this test's concern
        if getattr(cls, "_is_precomputed", False):
            continue
        sem = fetch_semantics_for_model(model_id)
        assert sem == "single" or sem.startswith(("multi:", "custom:")), (model_id, sem)
