"""FastAPI backend for the rs-embed interactive web app.

Serves a Leaflet map UI (``index.html``) and three JSON endpoints that wrap the
rs-embed Python API:

* ``GET  /api/models``   — list available models (precomputed vs on-the-fly).
* ``POST /api/preview``  — fetch a Sentinel-2 RGB quicklook for the selected ROI
  (no model run) → the "S2 image first" popup-bubble step.
* ``POST /api/embed``    — run the chosen models on the ROI and return, per model,
  a PCA-RGB thumbnail of the grid embedding **and** pooled-vector stats.

Run (from the repo root, in the rsembed venv with GEE authenticated)::

    EARTHENGINE_PROJECT=ee-yfkang \
      rsembed/bin/python -m uvicorn examples.webapp.server:app --port 8000

Then open http://localhost:8000.

The ROI/temporal/model contract mirrors the demo notebook:
* point  → ``PointBuffer(lon, lat, buffer_m=1000)``  (a ~2 km box)
* bbox   → ``BBox(minlon, minlat, maxlon, maxlat)``
* a year-range [y0, y1] maps to ``TemporalSpec.year(...)`` for annual precomputed
  models and ``TemporalSpec.range("{y0}-06-01", "{y1}-09-01")`` for on-the-fly ones.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reuse the demo helpers (pca_rgb / to_dhw / pooled_vector) shipped with examples/.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # examples/
import iguide_demo_helpers as H  # noqa: E402

# --- rs-embed + Earth Engine (initialized once, lazily) ---------------------
PRECOMPUTED = {"gse", "tessera", "copernicus"}
COPERNICUS_FIXED_YEAR = 2021  # copernicus product currently covers only 2021

# Saved embedding packages (.npz) live here; served back via /api/download.
DOWNLOADS = HERE / "downloads"
DOWNLOADS.mkdir(exist_ok=True)
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)  # holds logo.png etc., served at /assets

# Pretrained downstream heads (built by build_demo_cache.py --only heads).
_HEADS: dict[str, Any] = {}  # model -> loaded regressor (lazy)


def _heads_dir() -> Path:
    import iguide_demo_helpers as _H
    return _H.DemoCache.find(HERE).root / "heads"


def _heads_meta() -> dict[str, Any] | None:
    p = _heads_dir() / "heads_meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_head(model: str):
    if model in _HEADS:
        return _HEADS[model]
    meta = _heads_meta()
    task = (meta or {}).get("task", "maize_yield")
    import joblib
    p = _heads_dir() / f"{task}__{model}.pkl"
    if not p.exists():
        return None
    reg = joblib.load(p)
    _HEADS[model] = reg
    return reg


def _predict_one(reg, vec: np.ndarray, meta: dict, info: dict, model: str) -> dict:
    """Run one head; works for both regression and classification heads."""
    v = np.nan_to_num(np.asarray(vec, dtype=np.float32)).reshape(1, -1)
    if hasattr(reg, "predict_proba"):  # classification head
        proba = reg.predict_proba(v)[0]
        classes = list(getattr(reg, "classes_", [0, 1]))
        pidx = classes.index(1) if 1 in classes else len(classes) - 1
        p = float(proba[pidx])
        names = meta.get("classes", ["negative", "positive"])
        return {"model": model, "ok": True, "kind": "classification",
                "prediction": round(p, 3), "label_pred": names[1] if p >= 0.5 else names[0],
                "score": info.get("score"), "score_name": info.get("score_name", "accuracy")}
    pred = float(reg.predict(v)[0])  # regression head
    lo, hi = info.get("y_min"), info.get("y_max")
    return {"model": model, "ok": True, "kind": "regression", "prediction": round(pred, 3),
            "in_range": bool(lo is None or (lo <= pred <= hi)),
            "score": info.get("score", info.get("r2")), "score_name": info.get("score_name", "R²")}
GRID_SAVE_MAX_CELLS = 300 * 300  # cap grids saved into the package (keeps it small)

_ee_ready = False


def _ensure_ee() -> None:
    global _ee_ready
    if _ee_ready:
        return
    import ee

    project = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    ee.Initialize(project=project) if project else ee.Initialize()
    _ee_ready = True


def _rs():
    from rs_embed import (
        BBox,
        ExportConfig,
        ExportTarget,
        OutputSpec,
        PointBuffer,
        SensorSpec,
        TemporalSpec,
        export_batch,
        get_embedding,
        inspect_provider_patch,
        list_models,
        load_export,
    )

    return locals()


# --- request models ---------------------------------------------------------
class Geometry(BaseModel):
    type: str  # "point" | "bbox"
    lon: float | None = None
    lat: float | None = None
    minlon: float | None = None
    minlat: float | None = None
    maxlon: float | None = None
    maxlat: float | None = None


class PreviewReq(BaseModel):
    geometry: Geometry
    start: str = "2022-06"   # "YYYY-MM" inclusive start month
    end: str = "2022-09"     # "YYYY-MM" inclusive end month


class EmbedReq(PreviewReq):
    models: list[str] = []
    buffer_m: int = 1000


class PredictReq(PreviewReq):
    models: list[str] = []     # embedding models to run through their heads
    buffer_m: int = 1000


# --- helpers -----------------------------------------------------------------
def _spatial(rs: dict, g: Geometry, buffer_m: int = 1000):
    if g.type == "point":
        if g.lon is None or g.lat is None:
            raise ValueError("point geometry requires lon/lat")
        return rs["PointBuffer"](lon=float(g.lon), lat=float(g.lat), buffer_m=int(buffer_m))
    if g.type == "bbox":
        for k in ("minlon", "minlat", "maxlon", "maxlat"):
            if getattr(g, k) is None:
                raise ValueError("bbox geometry requires minlon/minlat/maxlon/maxlat")
        return rs["BBox"](
            minlon=float(g.minlon), minlat=float(g.minlat),
            maxlon=float(g.maxlon), maxlat=float(g.maxlat),
        )
    raise ValueError(f"unknown geometry type: {g.type!r}")


def _ym(s: str) -> tuple[int, int]:
    """Parse a 'YYYY-MM' string into (year, month)."""
    parts = str(s).split("-")
    return int(parts[0]), int(parts[1])


def _range_dates(start: str, end: str) -> tuple[str, str, int, int]:
    """('YYYY-MM','YYYY-MM') → (start_day, end_exclusive_day, start_year, end_year).

    The end month is inclusive, so the returned end is the first day of the
    month *after* ``end`` (rs-embed ranges are half-open ``[start, end)``).
    """
    y0, m0 = _ym(start)
    y1, m1 = _ym(end)
    if (y0, m0) > (y1, m1):
        (y0, m0), (y1, m1) = (y1, m1), (y0, m0)
    start_d = f"{y0:04d}-{m0:02d}-01"
    em, ey = m1 + 1, y1
    if em > 12:
        em, ey = 1, ey + 1
    return start_d, f"{ey:04d}-{em:02d}-01", y0, y1


def _temporal(rs: dict, model: str, start: str, end: str):
    start_d, end_excl, _y0, y1 = _range_dates(start, end)
    if model in PRECOMPUTED:
        if model == "copernicus":
            return rs["TemporalSpec"].year(COPERNICUS_FIXED_YEAR)
        return rs["TemporalSpec"].year(y1)  # latest year in range for annual products
    return rs["TemporalSpec"].range(start_d, end_excl)


def _png_b64(rgb_float_hwc: np.ndarray) -> str:
    """(H,W,3) float in [0,1] → base64 PNG data URI."""
    from PIL import Image

    arr = np.clip(np.asarray(rgb_float_hwc) * 255.0, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _s2_rgb_to_float(chw: np.ndarray) -> np.ndarray:
    """Raw S2 SR (B4,B3,B2) CHW → percentile-stretched (H,W,3) float for display."""
    x = np.asarray(chw, dtype=np.float32)
    if x.ndim != 3 or x.shape[0] < 3:
        raise ValueError(f"expected 3-band CHW, got {x.shape}")
    rgb = np.transpose(x[:3], (1, 2, 0))  # H,W,3
    lo = np.nanpercentile(rgb, 2, axis=(0, 1))
    hi = np.nanpercentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


# --- app ---------------------------------------------------------------------
app = FastAPI(title="rs-embed web demo")
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")


@app.get("/")
def index() -> FileResponse:
    # no-store so iterative edits always show on a plain refresh
    return FileResponse(str(HERE / "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/api/models")
def api_models() -> Any:
    try:
        rs = _rs()
        ids = rs["list_models"]()
        models = [
            {"id": m, "type": "precomputed" if m in PRECOMPUTED else "onthefly"}
            for m in sorted(ids)
        ]
        return {"models": models, "precomputed": sorted(PRECOMPUTED)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/preview")
def api_preview(req: PreviewReq) -> Any:
    """Sentinel-2 RGB quicklook for the ROI — no model is run."""
    try:
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, getattr(req, "buffer_m", 1000))
        start_d, end_excl, _y0, _y1 = _range_dates(req.start, req.end)
        temporal = rs["TemporalSpec"].range(start_d, end_excl)
        sensor = rs["SensorSpec"](
            collection="COPERNICUS/S2_SR_HARMONIZED",
            bands=("B4", "B3", "B2"), scale_m=10, cloudy_pct=30, composite="median",
        )
        rep = rs["inspect_provider_patch"](
            spatial=spatial, temporal=temporal, sensor=sensor,
            backend="gee", name="s2_preview", return_array=True,
        )
        chw = rep.get("array_chw")
        if chw is None:
            return JSONResponse({"error": "no imagery returned for this ROI/time."},
                                status_code=502)
        rgb = _s2_rgb_to_float(chw)
        return {
            "image": _png_b64(rgb),
            "shape": [int(s) for s in np.asarray(chw).shape],
            "ok": bool(rep.get("ok", True)),
        }
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/embed")
def api_embed(req: EmbedReq) -> Any:
    """Run each selected model → PCA-RGB thumbnail + pooled stats, and save a
    downloadable ``.npz`` embedding package for use in the notebook."""
    try:
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)

        models = list(req.models)
        results: list[dict[str, Any]] = []
        pkg: dict[str, np.ndarray] = {}          # arrays written into the package
        pkg_models: list[dict[str, Any]] = []    # per-model metadata for the manifest

        def _ok(m: str, grid: np.ndarray) -> None:
            vec = H.pooled_vector(grid)
            gh, gw = int(grid.shape[1]), int(grid.shape[2])
            mtype = "precomputed" if m in PRECOMPUTED else "onthefly"
            pkg[f"pooled__{m}"] = vec.astype(np.float32)
            saved = gh * gw <= GRID_SAVE_MAX_CELLS
            if saved:
                pkg[f"grid__{m}"] = grid.astype(np.float32)
            pkg_models.append({"model": m, "type": mtype, "dim": int(vec.shape[0]),
                               "grid_hw": [gh, gw], "grid_saved": bool(saved)})
            results.append({"model": m, "type": mtype, "ok": True,
                            "image": _png_b64(H.pca_rgb(grid)), "dim": int(vec.shape[0]),
                            "grid_hw": [gh, gw], "norm": float(np.linalg.norm(vec)),
                            "vector_preview": [round(float(v), 4) for v in vec[:32]]})

        def _err(m: str, msg: Any) -> None:
            results.append({"model": m, "type": "precomputed" if m in PRECOMPUTED else "onthefly",
                            "ok": False, "error": str(msg)[:400]})

        # ── single model → get_embedding ; multiple → export_batch (shared fetch) ──
        if len(models) <= 1:
            compute = "get_embedding"
            for m in models:
                try:
                    emb = rs["get_embedding"](
                        m, spatial=spatial, temporal=_temporal(rs, m, req.start, req.end),
                        output=rs["OutputSpec"].grid(), backend="auto")
                    _ok(m, H.to_dhw(emb.data))
                except Exception as e:  # noqa: BLE001
                    _err(m, repr(e))
        else:
            compute = "export_batch"
            start_d, end_excl, _y0, _y1 = _range_dates(req.start, req.end)
            # one temporal for all models; gse/annual products convert range→year internally
            temporal = rs["TemporalSpec"].range(start_d, end_excl)
            tmp = DOWNLOADS / f"_export_{uuid.uuid4().hex[:8]}.npz"
            try:
                rs["export_batch"](
                    spatials=[spatial], temporal=temporal, models=models,
                    output=rs["OutputSpec"].grid(),
                    target=rs["ExportTarget"].combined(str(tmp)),
                    config=rs["ExportConfig"](save_inputs=False, save_embeddings=True,
                                              continue_on_error=True, show_progress=False),
                    backend="auto")
                er = rs["load_export"](str(tmp))
                # Read embeddings straight from the .npz arrays keyed by model.
                # (rs-embed's combined manifest can omit a model's entry even when
                # its `embeddings__<model>` array was written, so load_export alone
                # under-reports; the arrays are the source of truth.)
                with np.load(tmp, allow_pickle=True) as z:
                    by_model = {k[len("embeddings__"):]: k for k in z.files
                                if k.startswith("embeddings__")}
                    for m in models:
                        arr = np.asarray(z[by_model[m]]) if m in by_model else None
                        if arr is None:  # fall back to load_export's view
                            mr = er.models.get(m)
                            arr = np.asarray(mr.embeddings) if (mr and mr.embeddings is not None) else None
                        if arr is not None and arr.ndim >= 3 and arr.shape[0] >= 1 and np.isfinite(arr).any():
                            _ok(m, H.to_dhw(arr[0]))  # (1,C,H,W) → (C,H,W)
                        else:
                            mr = er.models.get(m)
                            _err(m, getattr(mr, "error", None) or "no embedding produced")
            finally:
                for p in (tmp, tmp.with_suffix(".json")):
                    try:
                        p.unlink()
                    except OSError:
                        pass

        download = None
        package = None
        if pkg:
            manifest = {
                "geometry": req.geometry.model_dump(),
                "start": req.start, "end": req.end,
                "buffer_m": req.buffer_m, "models": pkg_models, "compute": compute,
                "note": "Load with: import iguide_demo_helpers as H; H.load_embedding_package(path)",
            }
            pkg["meta"] = np.array(json.dumps(manifest))
            fname = f"rsembed_pkg_{uuid.uuid4().hex[:10]}.npz"
            np.savez_compressed(DOWNLOADS / fname, **pkg)
            download = f"/api/download/{fname}"
            package = {"filename": fname, "compute": compute,
                       "models": [mm["model"] for mm in pkg_models],
                       "grids_saved": [mm["model"] for mm in pkg_models if mm["grid_saved"]]}
        return {"results": results, "download_url": download, "package": package, "compute": compute}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/api/heads")
def api_heads() -> Any:
    """List available pretrained downstream heads (for the 'Use' tab)."""
    meta = _heads_meta()
    if not meta:
        return {"task": None, "models": [],
                "note": "No pretrained heads. Run: python build_demo_cache.py --only heads"}
    root, task = _heads_dir(), meta.get("task", "maize_yield")
    models = [{"model": m, **info} for m, info in meta.get("models", {}).items()
              if (root / f"{task}__{m}.pkl").exists()]
    return {"task": task, "kind": meta.get("kind", "regression"), "label": meta.get("label"),
            "units": meta.get("units"), "region": meta.get("region"),
            "classes": meta.get("classes"), "models": models}


@app.post("/api/predict")
def api_predict(req: PredictReq) -> Any:
    """ROI → embedding → pretrained head → predicted value, per selected model."""
    try:
        meta = _heads_meta()
        if not meta:
            return JSONResponse({"error": "no pretrained heads available"}, status_code=404)
        _ensure_ee()
        rs = _rs()
        spatial = _spatial(rs, req.geometry, req.buffer_m)
        results = []
        for m in (req.models or []):
            reg = _load_head(m)
            info = meta.get("models", {}).get(m, {})
            if reg is None:
                results.append({"model": m, "ok": False, "error": "no pretrained head for this model"})
                continue
            try:
                emb = rs["get_embedding"](
                    m, spatial=spatial, temporal=_temporal(rs, m, req.start, req.end),
                    output=rs["OutputSpec"].pooled(), backend="auto")
                results.append(_predict_one(reg, H.pooled_vector(emb.data), meta, info, m))
            except Exception as e:  # noqa: BLE001
                results.append({"model": m, "ok": False, "error": repr(e)[:300]})
        return {"task": meta.get("task"), "kind": meta.get("kind", "regression"),
                "label": meta.get("label"), "units": meta.get("units"),
                "region": meta.get("region"), "results": results}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.post("/api/predict_package")
async def api_predict_package(file: UploadFile = File(...)) -> Any:
    """Run an uploaded embedding package (.npz from /api/embed) through the heads."""
    try:
        meta = _heads_meta()
        if not meta:
            return JSONResponse({"error": "no pretrained heads available"}, status_code=404)
        z = np.load(io.BytesIO(await file.read()), allow_pickle=True)
        pooled = {k[len("pooled__"):]: np.asarray(z[k]) for k in z.files if k.startswith("pooled__")}
        if not pooled:
            return JSONResponse({"error": "no pooled vectors found in package"}, status_code=400)
        results = []
        for m, vec in pooled.items():
            reg = _load_head(m)
            info = meta.get("models", {}).get(m, {})
            if reg is None:
                results.append({"model": m, "ok": False, "error": "no pretrained head for this model"})
                continue
            try:
                results.append(_predict_one(reg, np.asarray(vec), meta, info, m))
            except Exception as e:  # noqa: BLE001
                results.append({"model": m, "ok": False, "error": repr(e)[:300]})
        return {"task": meta.get("task"), "kind": meta.get("kind", "regression"),
                "label": meta.get("label"), "units": meta.get("units"), "results": results}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": repr(e)}, status_code=500)


@app.get("/api/download/{name}")
def api_download(name: str) -> Any:
    """Serve a previously saved embedding package (path-traversal safe)."""
    if not name.startswith("rsembed_pkg_") or "/" in name or "\\" in name:
        return JSONResponse({"error": "invalid package name"}, status_code=400)
    path = DOWNLOADS / name
    if not path.exists():
        return JSONResponse({"error": "package not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/octet-stream", filename=name)
