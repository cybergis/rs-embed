"""Regression: remoteclip must normalize input_chw exactly once.

The ``input_chw`` (export / prefetched / tiled) paths used to divide raw S2 SR
values by 10000 *before* ``_s2_rgb_u8_from_chw`` divided again, double-
normalizing to a black image. The self-fetch path (``input_chw=None``) feeds
raw values and is the reference. These tests pin both paths to the same model
input so export embeddings match get_embedding.
"""

from __future__ import annotations

import numpy as np

from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.embedders import onthefly_remoteclip as rc
from rs_embed.embedders.onthefly_remoteclip import RemoteCLIPS2RGBEmbedder

_SPATIAL = PointBuffer(lon=0.0, lat=0.0, buffer_m=100)
_TEMPORAL = TemporalSpec.range("2020-01-01", "2020-02-01")
# Raw S2 surface-reflectance DN (NOT normalized to [0,1]).
_RAW = np.linspace(0.0, 9000.0, 3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16)


def _patch_common(monkeypatch, captured):
    def fake_encode_pooled(model, rgb_u8_list, *, image_size, device):
        for u8 in rgb_u8_list:
            captured.append(np.asarray(u8).copy())
        return np.zeros((len(rgb_u8_list), 4), dtype=np.float32)

    monkeypatch.setattr(rc, "_remoteclip_encode_pooled_batch", fake_encode_pooled)
    # self-fetch path returns the SAME raw array we feed as input_chw.
    monkeypatch.setattr(rc, "_fetch_s2_rgb_chw", lambda *a, **k: _RAW.copy())
    monkeypatch.setattr(RemoteCLIPS2RGBEmbedder, "_get_provider", lambda self, backend: object())
    monkeypatch.setattr(
        RemoteCLIPS2RGBEmbedder, "_get_model", lambda self, **k: (object(), {}, "cpu")
    )


def test_input_chw_path_matches_self_fetch(monkeypatch):
    captured: list[np.ndarray] = []
    _patch_common(monkeypatch, captured)
    emb = RemoteCLIPS2RGBEmbedder()

    emb.get_embedding(
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=None,
    )
    emb.get_embedding(
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=_RAW.copy(),
    )

    assert len(captured) == 2
    self_fetch_u8, input_chw_u8 = captured
    # Identical model input regardless of how the pixels arrived.
    np.testing.assert_array_equal(self_fetch_u8, input_chw_u8)
    # And a real (non-black) image — guards against the double /10000 regression.
    assert int(self_fetch_u8.max()) > 0


def test_batch_from_inputs_matches_self_fetch(monkeypatch):
    captured: list[np.ndarray] = []
    _patch_common(monkeypatch, captured)
    emb = RemoteCLIPS2RGBEmbedder()

    emb.get_embedding(
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=None,
    )
    emb.get_embeddings_batch_from_inputs(
        spatials=[_SPATIAL],
        input_chws=[_RAW.copy()],
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
    )

    assert len(captured) == 2
    self_fetch_u8, batch_u8 = captured
    np.testing.assert_array_equal(self_fetch_u8, batch_u8)
    assert int(batch_u8.max()) > 0


def test_strict_input_checks_no_false_positive_on_raw(monkeypatch):
    # The inspected ``s2_rgb_chw`` holds RAW S2 SR DN (~0..10000). The range
    # check must declare that range; declaring [0,1] flagged ~100% of valid
    # pixels as out-of-range and raised under strict checks (false positive).
    captured: list[np.ndarray] = []
    _patch_common(monkeypatch, captured)
    monkeypatch.setenv("RS_EMBED_CHECK_INPUT", "1")
    monkeypatch.setenv("RS_EMBED_CHECK_RAISE", "1")
    emb = RemoteCLIPS2RGBEmbedder()

    # Must NOT raise on valid raw imagery.
    out = emb.get_embedding(
        spatial=_SPATIAL,
        temporal=_TEMPORAL,
        sensor=None,
        output=OutputSpec.pooled(),
        backend="gee",
        input_chw=_RAW.copy(),
    )
    report = out.meta["input_checks"]["provider_s2_rgb_chw"]
    assert report["ok"] is True
    assert report.get("outside_range_frac", 0.0) == 0.0
