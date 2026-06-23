"""Regression: satvision export-batch honors already_unit_scaled like get_embedding.

satvision.fetch_input returns unit-scaled [0,1] proxy data with
``already_unit_scaled=True``. The single get_embedding path (and the self-fetch
batch) force norm="unit" for such data. The export *batch* path
(get_embeddings_batch_from_inputs) used to ignore that provenance and apply the
configured norm, so under RS_EMBED_SATVISION_TOA_NORM=raw it re-scaled
already-unit data and diverged from get_embedding. This pins the two paths.
"""

from __future__ import annotations

import numpy as np

from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_satvision_toa as sv
from rs_embed.embedders.onthefly_satvision_toa import SatVisionTOAEmbedder

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=100)
_TEMPORAL = TemporalSpec.range("2020-01-01", "2020-02-01")

# Minimal runtime: 2 channels (1 reflectance, 1 emissive), norm forced to "raw"
# so a failure to honor the unit flag is observable.
_FAKE_RT = {
    "model": object(),
    "model_meta": {},
    "device": "cpu",
    "model_id": "fake",
    "image_size": 4,
    "in_chans": 2,
    "norm_mode": "raw",
    "reflectance_indices": (0,),
    "emissive_indices": (1,),
    "reflectance_divisor": 100.0,
    "emissive_mins": (0.0,),
    "emissive_maxs": (1.0,),
}


def _patch(monkeypatch, captured):
    monkeypatch.setattr(
        SatVisionTOAEmbedder,
        "_resolve_runtime",
        lambda self, *, sensor, device: dict(_FAKE_RT),
    )

    def fake_forward(model, x_batch, *, device, output_mode):
        for x in x_batch:
            captured.append(np.asarray(x).copy())
        return [np.ones((4,), dtype=np.float32) for _ in x_batch], {"tokens_kind": "fake"}

    monkeypatch.setattr(sv, "_satvision_forward_batch", fake_forward)


def test_batch_from_inputs_honors_already_unit_scaled(monkeypatch):
    captured: list[np.ndarray] = []
    _patch(monkeypatch, captured)
    emb = SatVisionTOAEmbedder()
    unit = np.full((2, 4, 4), 0.5, dtype=np.float32)  # already unit-scaled proxy

    # Export batch path, with provenance flag, under norm="raw".
    emb.get_embeddings_batch_from_inputs(
        spatials=[_SPATIAL],
        input_chws=[unit.copy()],
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        fetch_metas=[{"already_unit_scaled": True}],
    )
    # Single path with the same flag (what the per-item export path does).
    emb.get_embedding(
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=unit.copy(),
        fetch_meta={"already_unit_scaled": True},
    )

    batch_x, single_x = captured
    # already_unit_scaled forces "unit" => clip only; reflectance NOT divided.
    np.testing.assert_allclose(batch_x, 0.5)
    np.testing.assert_allclose(single_x, 0.5)
    np.testing.assert_array_equal(batch_x, single_x)


def test_batch_from_inputs_without_flag_still_applies_raw(monkeypatch):
    # Sanity: the fix only honors the flag; it does not disable "raw" mode for
    # callers that genuinely pass raw data without provenance.
    captured: list[np.ndarray] = []
    _patch(monkeypatch, captured)
    emb = SatVisionTOAEmbedder()
    unit = np.full((2, 4, 4), 0.5, dtype=np.float32)

    emb.get_embeddings_batch_from_inputs(
        spatials=[_SPATIAL],
        input_chws=[unit.copy()],
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )
    x = captured[0]
    # reflectance channel /100 -> 0.005; emissive channel min-max -> 0.5
    np.testing.assert_allclose(x[0], 0.005, rtol=1e-5)
    np.testing.assert_allclose(x[1], 0.5)
