"""Tests for GEE fetch statistics (FetchStats + PrefetchManager integration)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from rs_embed.core.types import ExportConfig, ExportTarget
from rs_embed.tools.progress import FetchStats
from rs_embed.tools.runtime import get_embedder_bundle_cached

# ── FetchStats unit tests ──────────────────────────────────────────────────────


def test_fetch_stats_initial_state():
    stats = FetchStats()
    assert stats.total == 0
    assert stats.completed == 0
    assert stats.failed == 0
    assert stats.cache_hits == 0


def test_fetch_stats_record_planned():
    stats = FetchStats()
    stats.record_planned(5)
    assert stats.total == 5
    stats.record_planned(3)
    assert stats.total == 8


def test_fetch_stats_record_planned_ignores_negative():
    stats = FetchStats()
    stats.record_planned(-1)
    assert stats.total == 0


def test_fetch_stats_record_cache_hits():
    stats = FetchStats()
    stats.record_cache_hits(4)
    assert stats.cache_hits == 4
    stats.record_cache_hits(2)
    assert stats.cache_hits == 6


def test_fetch_stats_record_cache_hits_ignores_negative():
    stats = FetchStats()
    stats.record_cache_hits(-3)
    assert stats.cache_hits == 0


def test_fetch_stats_record_success():
    stats = FetchStats()
    stats.record_planned(3)
    stats.record_success()
    stats.record_success()
    assert stats.completed == 2
    assert stats.failed == 0


def test_fetch_stats_record_failure():
    stats = FetchStats()
    stats.record_planned(3)
    stats.record_failure()
    assert stats.failed == 1
    assert stats.completed == 0


def test_fetch_stats_mixed_outcomes():
    stats = FetchStats()
    stats.record_planned(10)
    stats.record_cache_hits(3)
    for _ in range(5):
        stats.record_success()
    stats.record_failure()
    assert stats.total == 10
    assert stats.completed == 5
    assert stats.failed == 1
    assert stats.cache_hits == 3


def test_fetch_stats_format_summary_zero():
    stats = FetchStats()
    s = stats.format_summary()
    assert "gee_fetch" in s
    assert "done=0" in s
    assert "failed=0" in s


def test_fetch_stats_format_summary_with_data():
    stats = FetchStats()
    stats.record_planned(10)
    stats.record_cache_hits(2)
    for _ in range(7):
        stats.record_success()
    stats.record_failure()
    s = stats.format_summary()
    assert "total=10" in s
    assert "done=7" in s
    assert "failed=1" in s
    assert "cached=2" in s


def test_fetch_stats_format_summary_percentage():
    stats = FetchStats()
    stats.record_planned(4)
    for _ in range(2):
        stats.record_success()
    s = stats.format_summary()
    assert "50%" in s


def test_fetch_stats_format_summary_100_percent():
    stats = FetchStats()
    stats.record_planned(5)
    for _ in range(5):
        stats.record_success()
    s = stats.format_summary()
    assert "100%" in s


def test_fetch_stats_format_summary_no_last_when_none():
    stats = FetchStats()
    stats.record_planned(1)
    stats.record_success()
    s = stats.format_summary()
    assert "last=" not in s


def test_fetch_stats_format_summary_shows_last_point_and_sensor():
    stats = FetchStats()
    stats.record_planned(2)
    stats.record_success(point=3, sensor="COPERNICUS/S2_SR_HARMONIZED")
    stats.record_success(point=7, sensor="COPERNICUS/S1_GRD")
    s = stats.format_summary()
    assert "last=point:7" in s
    assert "sensor:COPERNICUS/S1_GRD" in s


def test_fetch_stats_record_success_partial_info():
    stats = FetchStats()
    stats.record_planned(1)
    stats.record_success(point=5)  # no sensor
    s = stats.format_summary()
    assert "last=" not in s  # both must be set to show the field


def test_fetch_stats_log_writes_to_stderr(capsys):
    stats = FetchStats()
    stats.record_planned(3)
    stats.record_success()
    stats.log()
    captured = capsys.readouterr()
    assert "gee_fetch" in captured.err
    assert "total=3" in captured.err


def test_fetch_stats_thread_safety():
    stats = FetchStats()
    n = 200

    def _worker():
        stats.record_planned(1)
        stats.record_success()

    threads = [threading.Thread(target=_worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stats.total == n
    assert stats.completed == n
    assert stats.failed == 0


def test_fetch_stats_thread_safety_mixed():
    stats = FetchStats()
    n_success = 100
    n_fail = 50
    n_cache = 30

    def _success():
        stats.record_planned(1)
        stats.record_success()

    def _fail():
        stats.record_planned(1)
        stats.record_failure()

    def _cache():
        stats.record_cache_hits(1)

    threads = (
        [threading.Thread(target=_success) for _ in range(n_success)]
        + [threading.Thread(target=_fail) for _ in range(n_fail)]
        + [threading.Thread(target=_cache) for _ in range(n_cache)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert stats.total == n_success + n_fail
    assert stats.completed == n_success
    assert stats.failed == n_fail
    assert stats.cache_hits == n_cache


# ── PrefetchManager integration ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()


def _make_sensor() -> SensorSpec:
    return SensorSpec(collection="TEST", bands=["B1", "B2"], scale_m=10)


def _make_prefetch_manager(fetch_fn=None, inspect_fn=None):
    from rs_embed.core.types import ExportConfig
    from rs_embed.pipelines.prefetch import PrefetchManager

    sensor = _make_sensor()
    provider = MagicMock()
    provider.ensure_ready = MagicMock()
    provider.normalize_bands = None

    cfg = ExportConfig(
        save_inputs=True,
        save_embeddings=True,
        num_workers=1,
        continue_on_error=True,
    )

    default_fetch = fetch_fn or (lambda *a, **kw: np.ones((2, 4, 4), dtype=np.float32))
    default_inspect = inspect_fn or (lambda *a, **kw: {"ok": True})

    pm = PrefetchManager(
        provider=provider,
        models=["m1"],
        resolved_sensor={"m1": sensor},
        model_type={"m1": "onthefly"},
        config=cfg,
        fetch_fn=default_fetch,
        inspect_fn=default_inspect,
    )
    pm.plan()
    return pm, provider


def test_prefetch_manager_fetch_stats_records_planned():
    pm, _ = _make_prefetch_manager()
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    temporal = TemporalSpec.year(2022)

    stats = FetchStats()
    pm.fetch_chunk([0, 1], spatials, temporal, fetch_stats=stats)

    assert stats.total == 2
    assert stats.completed == 2
    assert stats.failed == 0
    assert stats.cache_hits == 0


def test_prefetch_manager_fetch_stats_records_cache_hits():
    pm, _ = _make_prefetch_manager()
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    temporal = TemporalSpec.year(2022)

    # First fetch populates cache
    stats = FetchStats()
    pm.fetch_chunk([0, 1], spatials, temporal, fetch_stats=stats)
    assert stats.total == 2
    assert stats.completed == 2
    assert stats.cache_hits == 0

    # Second fetch for same indices: all should be cache hits
    stats2 = FetchStats()
    pm.fetch_chunk([0, 1], spatials, temporal, fetch_stats=stats2)
    assert stats2.total == 0
    assert stats2.cache_hits == 2
    assert stats2.completed == 0


def test_prefetch_manager_fetch_stats_partial_cache_hit():
    pm, _ = _make_prefetch_manager()
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]
    temporal = TemporalSpec.year(2022)

    # Fetch only index 0 first
    stats1 = FetchStats()
    pm.fetch_chunk([0], spatials, temporal, fetch_stats=stats1)
    assert stats1.total == 1
    assert stats1.completed == 1
    assert stats1.cache_hits == 0

    # Now fetch both: index 0 is cached, index 1 is new
    stats2 = FetchStats()
    pm.fetch_chunk([0, 1], spatials, temporal, fetch_stats=stats2)
    assert stats2.total == 1  # only index 1 needs fetching
    assert stats2.completed == 1
    assert stats2.cache_hits == 1  # index 0 is cached


def test_prefetch_manager_fetch_stats_records_failures():
    def _failing_fetch(*a, **kw):
        raise RuntimeError("GEE fetch failed")

    pm, _ = _make_prefetch_manager(fetch_fn=_failing_fetch)
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]
    temporal = TemporalSpec.year(2022)

    stats = FetchStats()
    # continue_on_error=True so it won't raise
    pm.fetch_chunk([0], spatials, temporal, fetch_stats=stats)

    assert stats.total == 1
    assert stats.failed == 1
    assert stats.completed == 0


def test_prefetch_manager_fetch_stats_none_is_noop():
    """Passing fetch_stats=None should not cause any errors."""
    pm, _ = _make_prefetch_manager()
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]
    temporal = TemporalSpec.year(2022)

    # Should not raise
    pm.fetch_chunk([0], spatials, temporal, fetch_stats=None)


def test_prefetch_manager_fetch_stats_no_provider():
    """When provider is None, fetch_chunk returns early; stats stay at zero."""
    from rs_embed.core.types import ExportConfig
    from rs_embed.pipelines.prefetch import PrefetchManager

    sensor = _make_sensor()
    cfg = ExportConfig(save_inputs=True, save_embeddings=True)
    pm = PrefetchManager(
        provider=None,
        models=["m1"],
        resolved_sensor={"m1": sensor},
        model_type={"m1": "onthefly"},
        config=cfg,
    )
    pm.plan()

    stats = FetchStats()
    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]
    pm.fetch_chunk([0], spatials, TemporalSpec.year(2022), fetch_stats=stats)

    assert stats.total == 0
    assert stats.completed == 0


# ── BatchExporter / export_batch integration ──────────────────────────────────


@pytest.fixture(autouse=True)
def disable_real_progress(monkeypatch):
    import rs_embed.api as api

    class _NoOpProgress:
        def update(self, n: int = 1):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _NoOpProgress(),
    )


def _register_onthefly(name: str):
    class DummyOntheFly:
        def describe(self):
            return {
                "type": "onthefly",
                "inputs": {"collection": "C", "bands": ["B1", "B2", "B3"]},
                "defaults": {
                    "scale_m": 10,
                    "cloudy_pct": 30,
                    "composite": "median",
                    "fill_value": 0.0,
                },
            }

        def get_embedding(
            self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    DummyOntheFly.__name__ = name
    registry.register(name)(DummyOntheFly)


def _patch_gee(monkeypatch):
    """Patch GEE provider and fetch function to avoid real network calls."""

    class DummyProvider:
        def __init__(self, *a, **kw):
            pass

        def ensure_ready(self):
            pass

    monkeypatch.setattr("rs_embed.tools.runtime.get_provider", lambda _name, **_kw: DummyProvider())
    monkeypatch.setattr(
        "rs_embed.providers.fetch.fetch_sensor_patch_chw",
        lambda provider, *, spatial, temporal, sensor: np.ones((3, 8, 8), dtype=np.float32),
    )
    monkeypatch.setattr(
        "rs_embed.providers.fetch.inspect_fetch_result",
        lambda x_chw, *, sensor, name: {"ok": True},
    )


def test_export_batch_fetch_stats_logged_per_item(tmp_path, monkeypatch, capsys):
    import rs_embed.api as api

    _register_onthefly("dummy_otf_stats")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
        PointBuffer(lon=2, lat=2, buffer_m=10),
    ]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_stats"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=True,
            chunk_size=2,
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "[gee_fetch]" in captured.err


def test_export_batch_fetch_stats_not_logged_when_progress_disabled(tmp_path, monkeypatch, capsys):
    import rs_embed.api as api

    _register_onthefly("dummy_otf_nolog")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_nolog"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=False,
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "[gee_fetch]" not in captured.err


def test_export_batch_combined_fetch_stats_logged(tmp_path, monkeypatch, capsys):
    import rs_embed.api as api

    _register_onthefly("dummy_otf_combined_stats")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_combined_stats"],
        target=ExportTarget.combined(str(tmp_path / "combined.npz")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=True,
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "[gee_fetch]" in captured.err


def test_export_batch_combined_fetch_stats_not_logged_when_disabled(tmp_path, monkeypatch, capsys):
    import rs_embed.api as api

    _register_onthefly("dummy_otf_combined_nolog")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_combined_nolog"],
        target=ExportTarget.combined(str(tmp_path / "combined.npz")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=False,
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "[gee_fetch]" not in captured.err


def test_export_batch_fetch_stats_counts_in_log(tmp_path, monkeypatch, capsys):
    """The log line should reflect the correct completed count for a single chunk."""
    import rs_embed.api as api

    _register_onthefly("dummy_otf_counts")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
    ]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_counts"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=True,
            chunk_size=10,  # single chunk: both fetches happen together
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "done=2" in captured.err
    assert "total=2" in captured.err


def test_export_batch_fetch_stats_multi_chunk_cumulative(tmp_path, monkeypatch, capsys):
    """Stats accumulate across multiple chunks (chunk_size=1 forces one chunk per point)."""
    import rs_embed.api as api

    _register_onthefly("dummy_otf_multi_chunk")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [
        PointBuffer(lon=0, lat=0, buffer_m=10),
        PointBuffer(lon=1, lat=1, buffer_m=10),
        PointBuffer(lon=2, lat=2, buffer_m=10),
    ]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_multi_chunk"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=True,
            chunk_size=1,  # one point per chunk → 3 separate fetch log lines
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    lines = [ln for ln in captured.err.splitlines() if "[gee_fetch]" in ln]
    # Each chunk completion emits one log line; with chunk_size=1 and 3 points we get 3
    assert len(lines) >= 1
    # The final log line should show all 3 fetches completed (stats are cumulative)
    assert "done=3" in captured.err


def test_export_batch_fetch_stats_no_provider_no_log(tmp_path, monkeypatch, capsys):
    """Precomputed models (no GEE provider) must not emit gee_fetch log lines."""
    import rs_embed.api as api

    class DummyPrecomputed:
        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

        def get_embedding(
            self, *, spatial, temporal, sensor, output, backend, device="auto", input_chw=None
        ):
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_precomputed_stats")(DummyPrecomputed)
    get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_precomputed_stats"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            show_progress=True,
        ),
        backend="local",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "[gee_fetch]" not in captured.err


def test_export_batch_fetch_stats_last_point_and_sensor_in_log(tmp_path, monkeypatch, capsys):
    """The log line includes last=point:N,sensor:collection after a successful fetch."""
    import rs_embed.api as api

    _register_onthefly("dummy_otf_last_field")
    _patch_gee(monkeypatch)
    get_embedder_bundle_cached.cache_clear()

    spatials = [PointBuffer(lon=0, lat=0, buffer_m=10)]

    api.export_batch(
        spatials=spatials,
        temporal=TemporalSpec.year(2022),
        models=["dummy_otf_last_field"],
        target=ExportTarget.per_item(str(tmp_path / "out")),
        config=ExportConfig(
            save_inputs=True,
            save_embeddings=True,
            show_progress=True,
            num_workers=1,
        ),
        backend="gee",
        output=OutputSpec.pooled(),
    )

    captured = capsys.readouterr()
    assert "last=point:" in captured.err
    assert "sensor:" in captured.err
