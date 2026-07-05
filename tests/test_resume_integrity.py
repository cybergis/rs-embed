"""Resume integrity: request fingerprinting and checkpoint state consistency.

An existing export output only counts as "already done" for the *same*
request. These tests pin the fingerprint gate on both layouts and the
combined-checkpoint consistency rules (ok-only, arrays-present, stale-model
cleanup).
"""

import json
import os

import numpy as np
import pytest

from rs_embed.core import registry
from rs_embed.core.embedding import Embedding
from rs_embed.core.specs import OutputSpec, PointBuffer, TemporalSpec
from rs_embed.core.types import ExportConfig, ExportTarget
from rs_embed.pipelines.checkpoint import CheckpointManager
from rs_embed.tools.runtime import get_embedder_bundle_cached


@pytest.fixture(autouse=True)
def clean_registry():
    registry._REGISTRY.clear()
    get_embedder_bundle_cached.cache_clear()
    yield
    registry._REGISTRY.clear()


def _register_counting_model(name: str):
    class _Dummy:
        calls = 0

        def describe(self):
            return {"type": "precomputed", "backend": ["local"], "output": ["pooled"]}

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
        ):
            _Dummy.calls += 1
            return Embedding(data=np.array([1.0], dtype=np.float32), meta={})

    registry.register(name)(_Dummy)
    return _Dummy


_SPATIALS = [PointBuffer(lon=0, lat=0, buffer_m=10), PointBuffer(lon=1, lat=1, buffer_m=10)]


def _export(out, *, model: str, temporal=None, resume=False, per_item=True):
    import rs_embed.api as api

    return api.export_batch(
        spatials=_SPATIALS,
        temporal=temporal or TemporalSpec.year(2022),
        models=[model],
        target=(ExportTarget.per_item(str(out)) if per_item else ExportTarget.combined(str(out))),
        config=ExportConfig(
            save_inputs=False, save_embeddings=True, resume=resume, show_progress=False
        ),
        backend="local",
        output=OutputSpec.pooled(),
    )


# ── per-item fingerprint gate ─────────────────────────────────────────


def test_per_item_resume_recomputes_when_request_changed(tmp_path):
    """resume=True must not reuse files written by a different request."""
    dummy = _register_counting_model("dummy_fp_item")
    out = tmp_path / "d"

    _export(out, model="dummy_fp_item")
    assert dummy.calls == 2

    second = _export(out, model="dummy_fp_item", temporal=TemporalSpec.year(2023), resume=True)
    assert dummy.calls == 4
    assert not any(m.get("resume_skipped") for m in second)


def test_per_item_resume_skips_only_matching_request(tmp_path):
    """Identical request → skip (resume still works after the fingerprint gate)."""
    dummy = _register_counting_model("dummy_fp_item_same")
    out = tmp_path / "d"

    _export(out, model="dummy_fp_item_same")
    second = _export(out, model="dummy_fp_item_same", resume=True)
    assert dummy.calls == 2
    assert all(m.get("resume_skipped") for m in second)


# ── combined fingerprint gate ─────────────────────────────────────────


def test_combined_completed_resume_requires_matching_request(tmp_path):
    """A complete combined file for a different request must be re-exported."""
    dummy = _register_counting_model("dummy_fp_comb")
    out = tmp_path / "c.npz"

    _export(out, model="dummy_fp_comb", per_item=False)
    assert dummy.calls == 2

    second = _export(
        out, model="dummy_fp_comb", temporal=TemporalSpec.year(2023), resume=True, per_item=False
    )
    assert dummy.calls == 4
    assert not second.get("resume_skipped")


def test_combined_completed_resume_ignores_missing_sidecar(tmp_path):
    """out_file exists but the sidecar is missing/corrupt → re-export, not 'skipped'."""
    dummy = _register_counting_model("dummy_fp_sidecar")
    out = tmp_path / "c.npz"

    _export(out, model="dummy_fp_sidecar", per_item=False)
    os.remove(tmp_path / "c.json")

    second = _export(out, model="dummy_fp_sidecar", resume=True, per_item=False)
    assert dummy.calls == 4
    assert not second.get("resume_skipped")
    assert second.get("status") == "ok"


def test_combined_completed_resume_skips_matching_request(tmp_path):
    dummy = _register_counting_model("dummy_fp_comb_same")
    out = tmp_path / "c.npz"

    _export(out, model="dummy_fp_comb_same", per_item=False)
    second = _export(out, model="dummy_fp_comb_same", resume=True, per_item=False)
    assert dummy.calls == 2
    assert second.get("resume_skipped")


# ── combined_init_state consistency (unit) ────────────────────────────


def _write_checkpoint(tmp_path, *, manifest: dict, arrays: dict[str, np.ndarray]):
    out = str(tmp_path / "c.npz")
    with open(out, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    with open(tmp_path / "c.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return out


def _init_state(out, *, fingerprint="fp"):
    mgr = CheckpointManager(
        ExportTarget.combined(out), ExportConfig(format="npz", resume=True, show_progress=False)
    )
    return mgr.combined_init_state(
        spatials=_SPATIALS,
        temporal=None,
        output=OutputSpec.pooled(),
        backend="local",
        device="cpu",
        models=["m1", "m2"],
        out_path=out,
        fingerprint=fingerprint,
    )


def _ckpt_manifest(*model_entries, fingerprint="fp"):
    return {
        "resume_incomplete": True,
        "request_fingerprint": fingerprint,
        "models": list(model_entries),
    }


_OK_M1 = {
    "model": "m1",
    "status": "ok",
    "embeddings": {"npz_key": "embeddings__m1"},
    "inputs": None,
}


def test_init_state_keeps_ok_model_with_arrays(tmp_path):
    out = _write_checkpoint(
        tmp_path,
        manifest=_ckpt_manifest(_OK_M1),
        arrays={"embeddings__m1": np.zeros((2, 4), dtype=np.float32)},
    )
    arrays, manifest, pending, _ = _init_state(out)
    assert pending == ["m2"]
    assert [m["model"] for m in manifest["models"]] == ["m1"]
    assert "embeddings__m1" in arrays


def test_init_state_rejects_fingerprint_mismatch(tmp_path):
    """A checkpoint from a different request must not be spliced in."""
    out = _write_checkpoint(
        tmp_path,
        manifest=_ckpt_manifest(_OK_M1, fingerprint="other"),
        arrays={"embeddings__m1": np.zeros((2, 4), dtype=np.float32)},
    )
    with pytest.warns(UserWarning, match="different export request"):
        arrays, manifest, pending, _ = _init_state(out)
    assert pending == ["m1", "m2"]
    assert manifest["models"] == []
    assert arrays == {}
    assert manifest["request_fingerprint"] == "fp"


def test_init_state_reruns_partial_model(tmp_path):
    """'partial' is not complete: its failed points must be retried."""
    entry = dict(_OK_M1, status="partial")
    out = _write_checkpoint(
        tmp_path,
        manifest=_ckpt_manifest(entry),
        arrays={"embeddings__m1": np.zeros((2, 4), dtype=np.float32)},
    )
    _, manifest, pending, _ = _init_state(out)
    assert pending == ["m1", "m2"]
    assert manifest["models"] == []


def test_init_state_reruns_ok_model_with_missing_arrays(tmp_path):
    """A manifest entry whose referenced arrays are gone is not complete."""
    out = _write_checkpoint(
        tmp_path,
        manifest=_ckpt_manifest(_OK_M1),
        arrays={"unrelated": np.zeros(2, dtype=np.float32)},
    )
    _, manifest, pending, _ = _init_state(out)
    assert pending == ["m1", "m2"]
    assert manifest["models"] == []


def test_init_state_reruns_all_when_arrays_unreadable(tmp_path):
    """Corrupt checkpoint arrays → nothing can be trusted as complete."""
    out = _write_checkpoint(tmp_path, manifest=_ckpt_manifest(_OK_M1), arrays={})
    with open(out, "wb") as f:
        f.write(b"not an npz")
    with pytest.warns(UserWarning, match="Could not load checkpoint arrays"):
        _, manifest, pending, _ = _init_state(out)
    assert pending == ["m1", "m2"]
    assert manifest["models"] == []


def test_init_state_drops_arrays_of_removed_models(tmp_path):
    """Arrays of models no longer in the request must not leak into the file."""
    removed = {
        "model": "gone",
        "status": "ok",
        "embeddings": {"npz_key": "embeddings__gone"},
        "inputs": None,
    }
    out = _write_checkpoint(
        tmp_path,
        manifest=_ckpt_manifest(_OK_M1, removed),
        arrays={
            "embeddings__m1": np.zeros((2, 4), dtype=np.float32),
            "embeddings__gone": np.ones((2, 4), dtype=np.float32),
        },
    )
    arrays, manifest, pending, _ = _init_state(out)
    assert "embeddings__gone" not in arrays
    assert "embeddings__m1" in arrays
    assert pending == ["m2"]
