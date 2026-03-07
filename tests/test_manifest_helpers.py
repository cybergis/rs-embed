"""Tests for rs_embed.internal.api.manifest_helpers."""

import json


from rs_embed.core.specs import BBox, OutputSpec, PointBuffer, TemporalSpec
from rs_embed.tools.manifest import (
    combined_resume_manifest,
    load_json_dict,
    point_failure_manifest,
    point_resume_manifest,
)


# ══════════════════════════════════════════════════════════════════════
# load_json_dict
# ══════════════════════════════════════════════════════════════════════


def test_load_json_dict_valid_file(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"status": "ok", "count": 3}), encoding="utf-8")
    result = load_json_dict(str(p))
    assert result == {"status": "ok", "count": 3}


def test_load_json_dict_nonexistent_returns_none():
    assert load_json_dict("/no/such/file.json") is None


def test_load_json_dict_invalid_json_returns_none(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{broken json!!!", encoding="utf-8")
    assert load_json_dict(str(p)) is None


def test_load_json_dict_list_not_dict_returns_none(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_json_dict(str(p)) is None


def test_load_json_dict_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("", encoding="utf-8")
    assert load_json_dict(str(p)) is None


# ══════════════════════════════════════════════════════════════════════
# point_resume_manifest — no existing JSON
# ══════════════════════════════════════════════════════════════════════


def test_point_resume_manifest_builds_fresh(tmp_path):
    out_file = str(tmp_path / "p00000.npz")
    m = point_resume_manifest(
        point_index=0,
        spatial=PointBuffer(lon=-88.0, lat=40.0, buffer_m=500),
        temporal=TemporalSpec.range("2020-01-01", "2020-06-01"),
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        out_file=out_file,
    )
    assert m["resume_skipped"] is True
    assert m["resume_output_path"] == out_file
    assert m["point_index"] == 0
    assert m["status"] in ("ok", "skipped")
    assert m["backend"] == "gee"
    assert "spatial" in m


def test_point_resume_manifest_loads_existing_json(tmp_path):
    """When a companion JSON exists, its content is used as the base."""
    out_npz = tmp_path / "p00001.npz"
    out_npz.write_bytes(b"fake")
    out_json = tmp_path / "p00001.json"
    existing = {
        "status": "ok",
        "models": [{"model": "m1", "status": "ok"}],
        "point_index": 1,
    }
    out_json.write_text(json.dumps(existing), encoding="utf-8")

    m = point_resume_manifest(
        point_index=1,
        spatial=PointBuffer(lon=-88.0, lat=40.0, buffer_m=500),
        temporal=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        out_file=str(out_npz),
    )
    assert m["resume_skipped"] is True
    assert m["models"] == [{"model": "m1", "status": "ok"}]
    assert m["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════
# combined_resume_manifest
# ══════════════════════════════════════════════════════════════════════


def test_combined_resume_manifest_fresh(tmp_path):
    out_file = str(tmp_path / "combined.npz")
    m = combined_resume_manifest(
        spatials=[
            PointBuffer(lon=-88.0, lat=40.0, buffer_m=500),
            PointBuffer(lon=-87.0, lat=41.0, buffer_m=500),
        ],
        temporal=TemporalSpec.year(2020),
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        out_file=out_file,
    )
    assert m["resume_skipped"] is True
    assert m["n_items"] == 2
    assert isinstance(m["spatials"], list)
    assert len(m["spatials"]) == 2


# ══════════════════════════════════════════════════════════════════════
# point_failure_manifest
# ══════════════════════════════════════════════════════════════════════


def test_point_failure_manifest_captures_error():
    m = point_failure_manifest(
        point_index=5,
        spatial=PointBuffer(lon=-88.0, lat=40.0, buffer_m=500),
        temporal=None,
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        stage="prefetch",
        error=RuntimeError("GEE timeout"),
    )
    assert m["status"] == "failed"
    assert m["stage"] == "prefetch"
    assert "GEE timeout" in m["error"]
    assert m["point_index"] == 5
    assert m["models"] == []


def test_point_failure_manifest_with_temporal():
    m = point_failure_manifest(
        point_index=0,
        spatial=BBox(minlon=-89.0, minlat=39.0, maxlon=-88.0, maxlat=40.0),
        temporal=TemporalSpec.range("2020-06-01", "2020-09-01"),
        output=OutputSpec.pooled(),
        backend="gee",
        device="cpu",
        stage="inference",
        error=ValueError("bad model"),
    )
    assert m["status"] == "failed"
    assert m["stage"] == "inference"
    assert m["temporal"] is not None
