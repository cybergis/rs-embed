"""Tests for rs_embed.pipelines.checkpoint — CheckpointManager.

Focuses on the combined-layout checkpoint contract: callers mutate one
manifest dict in place across writes, so ``combined_write_checkpoint`` must
return that same dict (not the jsonable copy handed to the writer).
"""

import json

import numpy as np

from rs_embed.core.types import ExportConfig, ExportTarget
from rs_embed.pipelines.checkpoint import CheckpointManager


def _manager(out_file: str) -> CheckpointManager:
    target = ExportTarget.combined(out_file)
    config = ExportConfig(format="npz", show_progress=False)
    return CheckpointManager(target, config)


def test_combined_write_checkpoint_returns_same_manifest_object(tmp_path):
    out_file = str(tmp_path / "out.npz")
    json_path = str(tmp_path / "out.json")
    mgr = _manager(out_file)
    manifest = {"models": [], "status": "running"}
    arrays = {"a": np.zeros((2, 2), dtype=np.float32)}

    returned = mgr.combined_write_checkpoint(
        manifest=manifest,
        arrays=arrays,
        stage="prefetched",
        final=False,
        out_path=out_file,
        json_path=json_path,
    )

    assert returned is manifest
    assert manifest["npz_path"] == out_file
    assert manifest["npz_keys"] == ["a"]
    assert manifest["manifest_path"] == json_path


def test_model_entries_survive_across_checkpoint_writes(tmp_path):
    """Regression: per-model checkpoint writes must not lose earlier entries.

    Mirrors the exporter setup — ``_write_ckpt`` closes over the exporter's
    manifest variable while ``run_pending_models`` rebinds its own local to
    each write's return value. When ``combined_write_checkpoint`` returned a
    fresh jsonable copy, the two drifted apart and only the first model's
    entry survived in the manifest.
    """
    out_file = str(tmp_path / "out.npz")
    json_path = str(tmp_path / "out.json")
    mgr = _manager(out_file)
    manifest = {"models": []}
    arrays = {"a": np.zeros((2, 2), dtype=np.float32)}

    def write_ckpt(*, stage, final=False):
        return mgr.combined_write_checkpoint(
            manifest=manifest,
            arrays=arrays,
            stage=stage,
            final=final,
            out_path=out_file,
            json_path=json_path,
        )

    def run_models(manifest):
        for m in ("model_a", "model_b", "model_c"):
            manifest["models"].append({"model": m, "status": "ok"})
            manifest = write_ckpt(stage=f"model:{m}")
        return manifest

    result = run_models(manifest)

    assert [e["model"] for e in result["models"]] == ["model_a", "model_b", "model_c"]

    write_ckpt(stage="done", final=True)
    with open(json_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert [e["model"] for e in on_disk["models"]] == ["model_a", "model_b", "model_c"]
    assert on_disk["resume_incomplete"] is False
