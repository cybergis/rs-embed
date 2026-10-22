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
from types import SimpleNamespace

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


def test_export_batch_prefetch_always_square_fetches_model_specific(tmp_path, monkeypatch):
    """Model-specific fetches keep fetch_input's square_input=True default.

    The canonical single-embedding path always enlarges a rectangular ROI to a
    square fetch — under tile input_prep too (the stitched grid is cropped back
    to the ROI afterwards). The export prefetch must match it for every
    input_prep, or the same point embeds differently across APIs.
    """

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

    for input_prep in ("resize", "tile"):
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
        assert DummyTileFetch.seen_square_input == [True], f"input_prep={input_prep}"


def test_export_batch_merged_fetch_group_square_fetches_with_roi_meta(tmp_path, monkeypatch):
    """A merged multi-model fetch must square the ROI and hand every member the crop window.

    Regression: two models sharing one band-union fetch bypassed every
    model-specific fetch_input, so the generic fetch grabbed the raw rectangle
    with no roi_window_geo — square-input models then padded or stretched it to
    square with no way to crop back, and the same BBox embedded differently in
    a multi-model export than via get_embedding.
    """
    fetched_spatials: list = []
    seen_fetch_meta: dict[str, list] = {"a": [], "b": []}

    def _make_dummy(tag: str):
        class _Dummy:
            def describe(self):
                d = dict(_DESCRIBE)
                d["inputs"] = {"collection": "C", "bands": ["B1", "B2"]}
                return d

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
                _ = spatial, temporal, sensor, output, backend, device
                assert input_chw is not None
                seen_fetch_meta[tag].append(dict(fetch_meta) if fetch_meta else None)
                return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

        return _Dummy

    registry.register("dummy_merged_a")(_make_dummy("a"))
    registry.register("dummy_merged_b")(_make_dummy("b"))

    import rs_embed.api as api

    def fake_fetch(provider, *, spatial, temporal, sensor):
        _ = provider, temporal, sensor
        fetched_spatials.append(spatial)
        return np.full((2, 4, 4), 0.5, dtype=np.float32)

    _patch_provider(monkeypatch)
    monkeypatch.setattr("rs_embed.providers.fetch.fetch_sensor_patch_chw", fake_fetch)

    rect = BBox(minlon=0.0, minlat=0.0, maxlon=0.2, maxlat=0.05)
    out_path = tmp_path / "merged_square.npz"
    manifest = api.export_batch(
        spatials=[rect],
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_merged_a", "dummy_merged_b"],
        target=ExportTarget.combined(str(out_path)),
        config=ExportConfig(save_inputs=False, save_embeddings=True, show_progress=False),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
    )

    assert manifest.get("status") == "ok"
    # One merged fetch for both models, enlarged to a (near-)square BBox.
    assert len(fetched_spatials) == 1
    sq = fetched_spatials[0]
    w = float(sq.maxlon) - float(sq.minlon)
    h = float(sq.maxlat) - float(sq.minlat)
    assert w > 0 and h > 0
    assert abs(w - h) / max(w, h) < 0.15  # mercator square ≈ lat/lon square at low lat
    # Every member model received the crop-back window.
    for tag in ("a", "b"):
        assert len(seen_fetch_meta[tag]) == 1
        fm = seen_fetch_meta[tag][0]
        assert fm is not None and "roi_window_geo" in fm
        y0, y1, x0, x1 = fm["roi_window_geo"]
        # Wide rectangle → full width, centered vertical band.
        assert (x0, x1) == (0.0, 1.0)
        assert 0.0 < y0 < y1 < 1.0


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


class _ConfigRecordingTiledRoiEmbedder(_FakeTiledRoiEmbedder):
    """Records (input shape, model_config) per call; adapts its config for tiles."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def tiled_dispatch_model_config(self, model_config, *, tile_size):
        return {**(model_config or {}), "_tiled": int(tile_size)}

    def get_embedding(self, *, input_chw, output, model_config=None, fetch_meta=None, **kwargs):
        return self.get_embeddings_batch_from_inputs(
            spatials=[None],
            input_chws=[input_chw],
            output=output,
            model_config=model_config,
            fetch_metas=[fetch_meta],
        )[0]

    def get_embeddings_batch_from_inputs(self, *, input_chws, model_config=None, **kwargs):
        self.calls.extend((tuple(np.asarray(x).shape), model_config) for x in input_chws)
        return super().get_embeddings_batch_from_inputs(input_chws=input_chws, **kwargs)


def _run_batch_tiled_grid(emb, images, *, fetch_metas, model_config=None):
    from rs_embed.pipelines.inference import InferenceEngine

    cfg = ExportConfig(input_prep=InputPrepSpec.tile(tile_size=4))
    engine = InferenceEngine(device="cpu", output=OutputSpec.grid(), config=cfg)
    return engine._run_batch_tiled(
        idxs=list(range(len(images))),
        spatials=[BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0)] * len(images),
        temporal=None,
        sensor=None,
        embedder=emb,
        lock=threading.RLock(),
        backend="gee",
        get_input_fn=lambda i: images[i],
        batch_size=16,
        continue_on_error=False,
        on_done=lambda i: None,
        use_lock=False,
        model_name="fake_tiled_roi",
        model_config=model_config,
        get_fetch_meta_fn=lambda i: dict(fetch_metas[i]),
    )


def test_run_batch_tiled_dispatches_sub_tile_inputs_whole():
    """A sub-tile input runs unpadded with the plain config; no pad cells in the output."""
    emb = _ConfigRecordingTiledRoiEmbedder()
    small = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    large = np.arange(64, dtype=np.float32).reshape(1, 8, 8)

    out, succeeded = _run_batch_tiled_grid(
        emb,
        [small, large],
        fetch_metas=[{"roi_window_geo": _ROI}, {}],
        model_config={"k": 1},
    )

    assert succeeded
    # The sub-tile input used to be edge-padded to 4x4, run with the tile config,
    # and returned with the pad row/column still in the grid.
    assert ((1, 3, 3), {"k": 1}) in emb.calls
    assert emb.calls.count(((1, 4, 4), {"k": 1, "_tiled": 4})) == 4
    np.testing.assert_array_equal(out[0].embedding, small)
    assert out[0].meta["input_prep"]["resolved_mode"] == "resize"
    np.testing.assert_array_equal(out[1].embedding, large)
    assert out[1].meta["input_prep"]["resolved_mode"] == "tile"
    # The whole-input item keeps its ROI window for the embedder-side crop.
    assert {"roi_window_geo": _ROI} in emb.seen_fetch_metas


def test_run_batch_tiled_groups_whole_inputs_by_native_size():
    class _Flexible(_ConfigRecordingTiledRoiEmbedder):
        def resolve_input_image_size(self, model_config, *, input_hw=None):
            return 4 if input_hw is None else max(input_hw)

    emb = _Flexible()
    images = [np.ones((1, 3, 3), np.float32), np.ones((1, 2, 2), np.float32)]
    out, succeeded = _run_batch_tiled_grid(emb, images, fetch_metas=[{}, {}])

    assert succeeded
    # One dispatch per native size, smallest first, each with its own size.
    assert emb.calls == [((1, 2, 2), {"_tiled": 2}), ((1, 3, 3), {"_tiled": 3})]
    assert all(out[i].meta["input_prep"]["resolved_mode"] == "native" for i in (0, 1))


def test_run_batch_tiled_whole_input_failure_still_batches_tiled_points():
    class _FailsOnSubTile(_ConfigRecordingTiledRoiEmbedder):
        def get_embeddings_batch_from_inputs(self, *, input_chws, model_config=None, **kwargs):
            if any(np.asarray(x).shape[-1] == 3 for x in input_chws):
                raise RuntimeError("boom")
            return super().get_embeddings_batch_from_inputs(
                input_chws=input_chws, model_config=model_config, **kwargs
            )

    small = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    large = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
    with pytest.warns(UserWarning, match=r"tiled \(whole-input\) batch inference"):
        out, succeeded = _run_batch_tiled_grid(
            _FailsOnSubTile(), [small, large], fetch_metas=[{}, {}]
        )

    # The failed whole-input point is left to the per-point fallback ...
    assert succeeded is False
    assert 0 not in out
    # ... while the multi-tile point still ran batched and stitched.
    np.testing.assert_array_equal(out[1].embedding, large)


def test_run_batch_tiled_stops_whole_input_groups_after_a_failure():
    class _FlexibleFailing(_ConfigRecordingTiledRoiEmbedder):
        attempted: list = []

        def resolve_input_image_size(self, model_config, *, input_hw=None):
            return 4 if input_hw is None else max(input_hw)

        def get_embeddings_batch_from_inputs(self, *, input_chws, model_config=None, **kwargs):
            self.attempted.append(tuple(np.asarray(input_chws[0]).shape))
            raise RuntimeError("boom")

    emb = _FlexibleFailing()
    images = [np.ones((1, 3, 3), np.float32), np.ones((1, 2, 2), np.float32)]
    with pytest.warns(UserWarning, match="whole-input"):
        out, succeeded = _run_batch_tiled_grid(emb, images, fetch_metas=[{}, {}])

    # The 2x2 group fails first; the 3x3 group is left to the per-point fallback
    # instead of failing (and retrying) on its own.
    assert succeeded is False
    assert out == {}
    assert set(emb.attempted) == {(1, 2, 2)}


def test_sub_tile_input_matches_between_single_point_and_batch_tiled():
    from rs_embed.tools.tiling import _call_embedder_get_embedding_with_input_prep

    small = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    out, _ = _run_batch_tiled_grid(_ConfigRecordingTiledRoiEmbedder(), [small], fetch_metas=[{}])
    single = _call_embedder_get_embedding_with_input_prep(
        embedder=_ConfigRecordingTiledRoiEmbedder(),
        spatial=BBox(minlon=0.0, minlat=0.0, maxlon=1.0, maxlat=1.0),
        temporal=None,
        sensor=None,
        output=OutputSpec.grid(),
        backend="gee",
        device="cpu",
        input_chw=small,
        input_prep=InputPrepSpec.tile(tile_size=4),
    )

    np.testing.assert_array_equal(out[0].embedding, np.asarray(single.data))
    assert out[0].meta["input_prep"] == single.meta["input_prep"]


def test_per_item_pipelined_chunks_keep_square_fetch(tmp_path, monkeypatch):
    """Chunks beyond the first must square-fetch exactly like chunk 1.

    Regression: the pipelined per-item path clones the PrefetchManager for
    every chunk after the first, and the clone dropped ``square_fetch_keys``
    (it is injected post-construction by the exporter). Points 2+ were then
    fetched as raw rectangles with no ``roi_window_geo`` — silently diverging
    from chunk 1 and from get_embedding.
    """
    fetched_spatials: list = []
    seen_fetch_meta: list = []

    class DummyGenericFetch:
        def describe(self):
            return dict(_DESCRIBE)

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
            _ = spatial, temporal, sensor, output, backend, device
            assert input_chw is not None
            seen_fetch_meta.append(dict(fetch_meta) if fetch_meta else None)
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_generic_fetch")(DummyGenericFetch)

    import rs_embed.api as api

    def fake_fetch(provider, *, spatial, temporal, sensor):
        _ = provider, temporal, sensor
        fetched_spatials.append(spatial)
        return np.full((1, 4, 4), 0.5, dtype=np.float32)

    _patch_provider(monkeypatch)
    monkeypatch.setattr("rs_embed.providers.fetch.fetch_sensor_patch_chw", fake_fetch)

    rects = [
        BBox(minlon=0.0, minlat=0.0, maxlon=0.2, maxlat=0.05),
        BBox(minlon=1.0, minlat=1.0, maxlon=1.2, maxlat=1.05),
        BBox(minlon=2.0, minlat=2.0, maxlon=2.2, maxlat=2.05),
    ]
    out_dir = tmp_path / "per_item_sq"
    manifests = api.export_batch(
        spatials=rects,
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_generic_fetch"],
        target=ExportTarget.per_item(str(out_dir)),
        config=ExportConfig(
            save_inputs=False,
            save_embeddings=True,
            show_progress=False,
            chunk_size=1,  # 3 points -> 3 chunks -> chunks 2-3 use _clone_prefetch
        ),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
    )

    assert [m.get("status") for m in manifests] == ["ok", "ok", "ok"]
    assert len(fetched_spatials) == 3
    for i, sq in enumerate(fetched_spatials):
        w = float(sq.maxlon) - float(sq.minlon)
        h = float(sq.maxlat) - float(sq.minlat)
        assert abs(w - h) / max(w, h) < 0.15, f"fetch {i} was not squared"
    assert len(seen_fetch_meta) == 3
    for i, fm in enumerate(seen_fetch_meta):
        assert fm is not None and "roi_window_geo" in fm, f"point {i} lost roi_window_geo"


def test_api_prefetch_generic_fallback_square_fetches(monkeypatch):
    """runtime.fetch_api_side_inputs' generic fallback must fetch-square.

    Regression: when an embedder's fetch_input returned None, the API-side
    prefetch fetched the raw rectangular ROI with no roi_window_geo, while
    the export pipeline squared the same request - the same point embedded
    differently across APIs.
    """
    fetched_spatials: list = []
    seen_fetch_meta: list = []

    class DummyGenericRuntime:
        def describe(self):
            return dict(_DESCRIBE)

        def fetch_input(self, provider, *, spatial, temporal, sensor, square_input=True):
            _ = provider, spatial, temporal, sensor, square_input
            return None  # force the generic fallback

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
            _ = spatial, temporal, sensor, output, backend, device
            assert input_chw is not None
            seen_fetch_meta.append(dict(fetch_meta) if fetch_meta else None)
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_generic_runtime")(DummyGenericRuntime)

    import rs_embed.api as api

    def fake_fetch(provider, *, spatial, temporal, sensor):
        _ = provider, temporal, sensor
        fetched_spatials.append(spatial)
        return np.full((1, 4, 4), 0.5, dtype=np.float32)

    _patch_provider(monkeypatch)
    monkeypatch.setattr("rs_embed.tools.runtime._fetch_sensor_patch_chw", fake_fetch)

    rect = BBox(minlon=0.0, minlat=0.0, maxlon=0.2, maxlat=0.05)
    api.get_embedding(
        "dummy_generic_runtime",
        spatial=rect,
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        input_prep="tile",
    )

    assert len(fetched_spatials) == 1
    sq = fetched_spatials[0]
    w = float(sq.maxlon) - float(sq.minlon)
    h = float(sq.maxlat) - float(sq.minlat)
    assert abs(w - h) / max(w, h) < 0.15, "generic API-side fallback did not square-fetch"
    assert seen_fetch_meta and seen_fetch_meta[0] is not None
    assert "roi_window_geo" in seen_fetch_meta[0]


def test_point_payload_sync_fallback_squares_and_reuses_provider():
    """build_one_point_payload's sync fetch fallback must square-fetch and
    create one provider per point, not one per model.

    Regression: it was a third, subtly different fetch path - raw rectangle,
    no roi_window_geo, and a fresh provider (with ensure_ready) per model.
    """
    from rs_embed.core.types import ExportConfig
    from rs_embed.pipelines.point_payload import build_one_point_payload

    fetched_spatials: list = []
    providers_created: list = []

    class DummyPayloadModel:
        def describe(self):
            return dict(_DESCRIBE)

        def fetch_input(self, provider, *, spatial, temporal, sensor, square_input=True):
            _ = provider, spatial, temporal, sensor, square_input
            return None

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
            _ = spatial, temporal, sensor, output, backend, device, input_chw
            assert fetch_meta is not None and "roi_window_geo" in fetch_meta
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register("dummy_payload_a")(DummyPayloadModel)
    registry.register("dummy_payload_b")(DummyPayloadModel)
    get_embedder_bundle_cached.cache_clear()

    def provider_factory():
        providers_created.append(1)
        return _DummyProvider()

    def fake_fetch(provider, *, spatial, temporal, sensor):
        _ = provider, temporal, sensor
        fetched_spatials.append(spatial)
        return np.full((1, 4, 4), 0.5, dtype=np.float32)

    from rs_embed.core.specs import SensorSpec

    rect = BBox(minlon=0.0, minlat=0.0, maxlon=0.2, maxlat=0.05)
    sensors = {
        "dummy_payload_a": SensorSpec(collection="C", bands=("B1",)),
        "dummy_payload_b": SensorSpec(collection="D", bands=("B1",)),
    }
    _arrays, manifest = build_one_point_payload(
        point_index=0,
        spatial=rect,
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=["dummy_payload_a", "dummy_payload_b"],
        backend="gee",
        device="cpu",
        output=OutputSpec.pooled(),
        resolved_sensor=sensors,
        resolved_model_config={"dummy_payload_a": None, "dummy_payload_b": None},
        model_type={"dummy_payload_a": "onthefly", "dummy_payload_b": "onthefly"},
        prefetch=SimpleNamespace(cache={}, input_reports={}, errors={}, fetch_meta={}),
        pass_input_into_embedder=True,
        config=ExportConfig(save_inputs=False, save_embeddings=True, show_progress=False),
        provider_factory=provider_factory,
        fetch_fn=fake_fetch,
        inspect_fn=lambda x, *, sensor, name: {"ok": True},
    )

    assert manifest["status"] == "ok"
    # Two distinct sensors -> two fetches, both squared; one provider total.
    assert len(fetched_spatials) == 2
    for sq in fetched_spatials:
        w = float(sq.maxlon) - float(sq.minlon)
        h = float(sq.maxlat) - float(sq.minlat)
        assert abs(w - h) / max(w, h) < 0.15
    assert len(providers_created) == 1
