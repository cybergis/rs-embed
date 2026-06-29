"""Regression tests: export_batch input_prep matches get_embedding.

Image-level ViT grid models (satmae, scalemae, ...) downgrade an unset/auto
``input_prep`` to ``resize`` in :func:`get_embedding` to avoid tiled stitching
seams. The export pipeline must apply the same *model-aware* resolution per
model; otherwise the same models silently tile under ``export_batch`` and
produce edge seams ("坏带"). These tests lock that parity.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import (
    InputPrepSpec,
    OutputSpec,
    PointBuffer,
    SensorSpec,
    TemporalSpec,
)
from rs_embed.core.types import ExportConfig, FetchResult
from rs_embed.embedders.base import EmbedderBase
from rs_embed.pipelines.inference import InferenceEngine
from rs_embed.pipelines.point_payload import build_one_point_payload

_VIT_GRID_MODELS = ["satmae", "satmaepp", "satmaepp_s2_10b", "scalemae"]
_NON_VIT_MODELS = ["remoteclip", "dofa", "terramind", "galileo"]


# ---------------------------------------------------------------------------
# Engine-level: per-model resolution mirrors get_embedding's model-aware logic
# ---------------------------------------------------------------------------


def _engine(output: OutputSpec, input_prep=None) -> InferenceEngine:
    cfg = ExportConfig(input_prep=input_prep)
    return InferenceEngine(device="cpu", output=output, config=cfg)


@pytest.mark.parametrize("model_id", _VIT_GRID_MODELS)
def test_engine_default_resolves_vit_grid_to_resize(model_id):
    # Pooled output downgrades silently (the seam warning is grid-only).
    engine = _engine(OutputSpec.pooled())
    _eff, resolved, explicit_nonresize = engine._model_input_prep(model_id)
    assert resolved.mode == "resize"
    assert explicit_nonresize is False


@pytest.mark.parametrize("model_id", _NON_VIT_MODELS)
def test_engine_default_keeps_non_vit_on_tile(model_id):
    engine = _engine(OutputSpec.pooled())
    _eff, resolved, explicit_nonresize = engine._model_input_prep(model_id)
    assert resolved.mode == "tile"
    assert explicit_nonresize is True


@pytest.mark.parametrize("model_id", _VIT_GRID_MODELS)
def test_engine_explicit_tile_is_honored_for_vit_grid(model_id):
    # An explicit tile request is never downgraded (matches get_embedding).
    engine = _engine(OutputSpec.pooled(), input_prep=InputPrepSpec.tile(tile_size=224))
    _eff, resolved, explicit_nonresize = engine._model_input_prep(model_id)
    assert resolved.mode == "tile"
    assert explicit_nonresize is True


def test_engine_grid_output_warns_once_per_model():
    # Grid output emits the stitching-seam warning, exactly like get_embedding,
    # and the per-model cache means repeated points do not re-warn.
    engine = _engine(OutputSpec.grid())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine._model_input_prep("satmae")
        engine._model_input_prep("satmae")
    seam = [w for w in caught if "resolves to resize" in str(w.message)]
    assert len(seam) == 1


# ---------------------------------------------------------------------------
# End-to-end (per-item CPU path): ViT model is NOT tiled, non-ViT IS tiled
# ---------------------------------------------------------------------------


class _FakeEmbedder(EmbedderBase):
    """On-the-fly fake exposing image_size=4 and capturing per-call input H/W."""

    def __init__(self, name: str) -> None:
        self.model_name = name
        self.input_shapes: list[tuple] = []

    def describe(self):
        return {"type": "onthefly", "defaults": {"image_size": 4}, "output": ["pooled"]}

    def fetch_input(self, provider, *, spatial, temporal, sensor):
        _ = provider, spatial, temporal, sensor
        # 8x8 > image_size(4): a tile-mode model would split this into 4 tiles.
        return FetchResult(data=np.ones((3, 8, 8), dtype=np.float32), meta={"source": "fake"})

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
        self.input_shapes.append(None if input_chw is None else tuple(np.asarray(input_chw).shape))
        return Embedding(data=np.ones((1,), dtype=np.float32), meta={})


def _run_one_point(monkeypatch, model_name: str) -> tuple[_FakeEmbedder, dict]:
    embedder = _FakeEmbedder(model_name)
    monkeypatch.setattr(
        "rs_embed.pipelines.point_payload.get_embedder_bundle_cached",
        lambda *a, **k: (embedder, threading.Lock()),
    )

    class _DummyProvider:
        def ensure_ready(self):
            return None

    sensor = SensorSpec(collection="C", bands=("B1", "B2", "B3"), scale_m=10, fill_value=0.0)
    _arrays, manifest = build_one_point_payload(
        point_index=0,
        spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=10),
        temporal=TemporalSpec.range("2020-01-01", "2020-02-01"),
        models=[model_name],
        backend="gee",
        resolved_backend={model_name: "gee"},
        device="cpu",
        output=OutputSpec.pooled(),
        resolved_sensor={model_name: sensor},
        resolved_model_config={model_name: None},
        model_type={model_name: "onthefly"},
        inputs_cache={},
        input_reports={},
        prefetch_errors={},
        pass_input_into_embedder=True,
        # Real ExportConfig => input_prep defaults to None (the case that
        # exposed the bug: None used to resolve to tile for every model).
        config=ExportConfig(save_inputs=False, save_embeddings=True, max_retries=0),
        provider_factory=_DummyProvider,
        inspect_fn=lambda x_chw, *, sensor, name: {"ok": True, "name": name},
    )
    return embedder, manifest


def test_per_item_vit_grid_model_is_not_tiled(monkeypatch):
    embedder, manifest = _run_one_point(monkeypatch, "satmae")
    # Resize: embedder called once with the FULL 8x8 input (no tiling, no seams).
    assert embedder.input_shapes == [(3, 8, 8)]
    assert manifest["models"][0]["meta"]["input_prep"]["resolved_mode"] == "resize"


def test_per_item_non_vit_model_is_tiled(monkeypatch):
    embedder, manifest = _run_one_point(monkeypatch, "dofa")
    # Tile: 8x8 split into 4 tiles of 4x4; embedder called once per tile.
    assert len(embedder.input_shapes) == 4
    assert all(shp == (3, 4, 4) for shp in embedder.input_shapes)
    assert manifest["models"][0]["meta"]["input_prep"]["resolved_mode"] == "tile"
