"""Export-path fetch-square ROI regression tests.

A rectangular ROI is enlarged to a square fetch for square-input models; the
ROI's window travels in ``fetch_meta['roi_window_geo']`` and the output must be
cropped back to it. These tests pin the export-pipeline half of that contract:

- the prefetched batch path forwards per-item fetch metadata to
  ``get_embeddings_batch_from_inputs(fetch_metas=...)``;
- the tiled batch path strips the image-level ROI from per-tile metas (a tile
  is not the whole image) and crops the stitched grid once;
- the prefetch square-fetches only when the model's effective input_prep is a
  single-input mode (tiling cuts square tiles from any-aspect imagery itself).
"""

import threading

import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import (
    BBox,
    InputPrepSpec,
    OutputSpec,
    PointBuffer,
    TemporalSpec,
)
from rs_embed.core.types import ExportConfig, ExportTarget, FetchResult
from rs_embed.tools.runtime import get_embedder_bundle_cached


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()


@pytest.fixture(autouse=True)
def disable_real_progress(monkeypatch):
    import rs_embed.api as api

    class _NoOpProgress:
        def update(self, n: int = 1):
            _ = n

        def close(self):
            return None

    monkeypatch.setattr(
        api,
        "_create_progress",
        lambda *, enabled, total, desc, unit="item": _NoOpProgress(),
    )


class _DummyProvider:
    def __init__(self, *args, **kwargs):
        pass

    def ensure_ready(self):
        return None


def _patch_provider(monkeypatch):
    monkeypatch.setattr(
        "rs_embed.tools.runtime.get_provider", lambda _name, **_kwargs: _DummyProvider()
    )
    monkeypatch.setattr(
        "rs_embed.providers.fetch.inspect_fetch_result",
        lambda x_chw, *, sensor, name: {"ok": True},
    )
    get_embedder_bundle_cached.cache_clear()


_DESCRIBE = {
    "type": "onthefly",
    "inputs": {"collection": "C", "bands": ["B1"]},
    "defaults": {
        "scale_m": 10,
        "cloudy_pct": 30,
        "composite": "median",
        "fill_value": 0.0,
    },
}

_ROI = (0.25, 0.75, 0.0, 1.0)


def test_export_batch_prefetched_batch_receives_fetch_metas_with_roi(tmp_path, monkeypatch):
    """The tier-1 batch path must hand each item's roi_window_geo to the embedder.

    Regression: the export prefetch square-fetched rectangular ROIs and stored
    ``roi_window_geo`` in fetch meta, but the batch inference call dropped it,
    so grid outputs covered the whole square instead of the requested rectangle.
    """

    class DummyBatchRoi:
        seen_fetch_metas: list = []

        def describe(self):
            return dict(_DESCRIBE)

        def fetch_input(self, provider, *, spatial, temporal, sensor, square_input=True):
            _ = provider, spatial, temporal, sensor
            assert square_input is True
            return FetchResult(
                data=np.full((1, 4, 4), 0.5, dtype=np.float32),
                meta={"roi_window_geo": _ROI},
            )

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
            fetch_meta=None,
        ):
            raise AssertionError("batch path expected; single fallback must not run")

        def get_embeddings_batch_from_inputs(
            self,
            *,
            spatials,
            input_chws,
            temporal=None,
            sensor=None,
            output=OutputSpec.pooled(),
            backend="auto",
            device="auto",
            fetch_metas=None,
        ):
            _ = temporal, sensor, output, backend, device
            DummyBatchRoi.seen_fetch_metas.append(
                [dict(m) if m else None for m in (fetch_metas or [])]
            )
            return [Embedding(data=np.array([1.0], dtype=np.float32), meta={}) for _ in spatials]

    registry.register("dummy_batch_roi")(DummyBatchRoi)

    import rs_embed.api as api

    _patch_provider(monkeypatch)
    monkeypatch.setattr("rs_embed.pipelines.inference._device_has_gpu", lambda _d: True)

    out_path = tmp_path / "batch_roi.npz"
    manifest = api.export_batch(
        spatials=[
            PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
            PointBuffer(lon=1.0, lat=1.0, buffer_m=10),
        ],
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_batch_roi"],
        target=ExportTarget.combined(str(out_path)),
        config=ExportConfig(
            save_inputs=False, save_embeddings=True, show_progress=False, input_prep="resize"
        ),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
    )

    assert manifest.get("status") == "ok"
    assert DummyBatchRoi.seen_fetch_metas == [[{"roi_window_geo": _ROI}, {"roi_window_geo": _ROI}]]


def test_export_batch_prefetch_skips_square_fetch_under_tile_input_prep(tmp_path, monkeypatch):
    """Tile input_prep must fetch the rectangular ROI directly (square_input=False)."""

    class DummyTileFetch:
        seen_square_input: list = []

        def describe(self):
            return dict(_DESCRIBE)

        def fetch_input(self, provider, *, spatial, temporal, sensor, square_input=True):
            _ = provider, spatial, temporal, sensor
            DummyTileFetch.seen_square_input.append(bool(square_input))
            return FetchResult(data=np.full((1, 2, 2), 0.5, dtype=np.float32), meta={})

        def get_embedding(
            self,
            *,
            spatial,
            temporal,
            sensor,
            output,
            backend,
            device="auto",
            input_chw=None,
            fetch_meta=None,
        ):
            _ = spatial, temporal, sensor, output, backend, device, fetch_meta
            assert input_chw is not None
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_tile_fetch")(DummyTileFetch)

    import rs_embed.api as api

    _patch_provider(monkeypatch)

    for input_prep, expected_square in (("resize", True), ("tile", False)):
        DummyTileFetch.seen_square_input.clear()
        get_embedder_bundle_cached.cache_clear()
        out_path = tmp_path / f"square_gate_{input_prep}.npz"
        manifest = api.export_batch(
            spatials=[PointBuffer(lon=0.0, lat=0.0, buffer_m=10)],
            temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
            models=["dummy_tile_fetch"],
            target=ExportTarget.combined(str(out_path)),
            config=ExportConfig(
                save_inputs=False,
                save_embeddings=True,
                show_progress=False,
                input_prep=input_prep,
            ),
            backend="gee",
            device="cpu",
            output=OutputSpec.pooled(),
        )
        assert manifest.get("status") == "ok"
        assert DummyTileFetch.seen_square_input == [expected_square], f"input_prep={input_prep}"


class _FakeTiledRoiEmbedder:
    """Grid-mode fake: each tile's embedding grid is its own pixels (1:1 tokens)."""

    def __init__(self):
        self.seen_fetch_metas: list = []

    def describe(self):
        return {"defaults": {"image_size": 4}}

    def get_embedding(self, **kwargs):
        raise AssertionError("batch path expected")

    def get_embeddings_batch_from_inputs(
        self,
        *,
        spatials,
        input_chws,
        temporal=None,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="auto",
        device="auto",
        fetch_metas=None,
    ):
        _ = spatials, temporal, sensor, backend, device
        self.seen_fetch_metas.extend(
            [dict(m) if m else None for m in (fetch_metas or [None] * len(input_chws))]
        )
        out = []
        for x in input_chws:
            x = np.asarray(x, dtype=np.float32)
            out.append(
                Embedding(
                    data=x[:1].copy(),
                    meta={
                        "y_axis_direction": "north_to_south",
                        "grid_hw": (int(x.shape[-2]), int(x.shape[-1])),
                    },
                )
            )
        return out


def test_run_batch_tiled_strips_tile_roi_and_crops_stitched_grid():
    """Tiled batch: per-tile metas lose roi_window_geo; the stitched grid is cropped.

    An 8x8 square fetched for a 4x8 rectangular ROI (roi_window_geo rows
    0.25..0.75) is cut into four 4x4 tiles; the stitched 8x8 grid must come
    back as 4x8, and no tile may see the image-level ROI window (cropping a
    tile by it would slice the wrong region).
    """
    from rs_embed.pipelines.inference import InferenceEngine

    emb = _FakeTiledRoiEmbedder()
    cfg = ExportConfig(input_prep=InputPrepSpec.tile(tile_size=4))
    engine = InferenceEngine(device="cpu", output=OutputSpec.grid(), config=cfg)

    image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    fetch_meta = {"roi_window_geo": _ROI, "already_unit_scaled": True}

    out, succeeded = engine._run_batch_tiled(
        idxs=[0],
        spatials=[BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=0.5)],
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=threading.RLock(),
        backend="gee",
        get_input_fn=lambda i: image,
        batch_size=16,
        continue_on_error=False,
        on_done=lambda i: None,
        use_lock=False,
        model_name="fake_tiled_roi",
        get_fetch_meta_fn=lambda i: dict(fetch_meta),
    )

    assert succeeded
    assert out[0].status.value == "ok"
    grid = np.asarray(out[0].embedding)
    # 8x8 stitched grid cropped to the ROI rows (0.25..0.75 of 8 = rows 2..6).
    assert grid.shape[-2:] == (4, 8)
    np.testing.assert_allclose(grid[0], image[0, 2:6, :])
    assert out[0].meta["input_prep"]["roi_cropped"] is True
    # Every tile saw the unit-scale flag but not the image-level ROI window.
    assert len(emb.seen_fetch_metas) == 4
    for m in emb.seen_fetch_metas:
        assert m is not None
        assert m.get("already_unit_scaled") is True
        assert "roi_window_geo" not in m


def test_run_batch_tiled_single_tile_keeps_roi_meta():
    """A single tile spans the whole input, so the embedder-side crop applies."""
    from rs_embed.pipelines.inference import InferenceEngine

    emb = _FakeTiledRoiEmbedder()
    cfg = ExportConfig(input_prep=InputPrepSpec.tile(tile_size=4))
    engine = InferenceEngine(device="cpu", output=OutputSpec.grid(), config=cfg)

    image = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    fetch_meta = {"roi_window_geo": _ROI}

    out, succeeded = engine._run_batch_tiled(
        idxs=[0],
        spatials=[BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=0.5)],
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=threading.RLock(),
        backend="gee",
        get_input_fn=lambda i: image,
        batch_size=16,
        continue_on_error=False,
        on_done=lambda i: None,
        use_lock=False,
        model_name="fake_tiled_roi",
        get_fetch_meta_fn=lambda i: dict(fetch_meta),
    )

    assert succeeded
    assert out[0].status.value == "ok"
    assert emb.seen_fetch_metas == [{"roi_window_geo": _ROI}]
