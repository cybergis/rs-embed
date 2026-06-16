"""Reproducibly build the cache for the rs-embed I-GUIDE interactive demo.

The demo notebook (``iguide_demo.ipynb``) runs in two modes:

* ``RUN_LIVE = False`` (booth-safe default) — reads everything from
  ``examples/iguide_demo_cache/`` so it works with **no Google Earth Engine
  auth, no GPU, and no network**.
* ``RUN_LIVE = True`` — calls ``rs_embed`` live and falls back to this cache on
  any failure.

This script generates that cache. It **requires** a one-time Earth Engine
authentication (``earthengine authenticate``) and network access, so it is run
by a maintainer ahead of the demo, not by booth visitors.

Cache layout and array conventions are documented in
``iguide_demo_helpers`` (single source of truth). Everything is keyed to the
SIGSPATIAL '26 / Riverside-citrus story.

Usage
-----
    python build_demo_cache.py            # full build
    python build_demo_cache.py --quick    # tiny build for a smoke test
    python build_demo_cache.py --only twins timemachine   # subset

Outputs
-------
    examples/iguide_demo_cache/
        twins_bank.npz
        timemachine.npz
        landcover/<scene>.npz
        showdown/<bundle>.npz + .json   (rs_embed export_batch)
        showdown/labels.npz
        cache_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Demo glue (cache schema + array helpers) lives next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import iguide_demo_helpers as H  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — the Riverside-citrus / SIGSPATIAL '26 story
# ---------------------------------------------------------------------------

# Precomputed model used for the instant map scenarios (#1, #2): no checkpoint
# download, no heavy forward pass — fast and robust for a live booth.
PRECOMPUTED_MODEL = "gse"  # Google/AlphaEarth annual embeddings (2017–2024)

# Models compared on-the-fly in #3 (Instant Land-Cover) and #6 (Model Showdown).
# Kept to a fast, reliable subset; extend freely when more compute is available.
MODEL_LIST = ["satmae", "remoteclip", "dofa", "terrafm", PRECOMPUTED_MODEL]

# A Riverside, CA citrus grove (the default "click" + landcover scene center).
RIVERSIDE_GROVE = (-117.34, 33.97)  # lon, lat — near UC Riverside ag fields

# Southern California citrus belt — the Model Showdown sampling region.
# (minlon, minlat, maxlon, maxlat)
CITRUS_REGION = (-118.2, 33.4, -116.8, 34.3)

# ⏳ Time Machine hotspots — visually dramatic change, one local to UCR.
HOTSPOTS = {
    "salton_sea": (-115.85, 33.30),  # local: shrinking saline lake near Riverside
    "dubai_palm": (55.13, 25.12),  # coastal megastructure construction
    "amazon_rondonia": (-62.20, -9.40),  # deforestation "fishbone"
    "lake_powell": (-110.90, 37.05),  # reservoir drawdown
}

# 🎨 Land-Cover scenes (BBoxes) — mixed citrus / urban / desert texture.
LANDCOVER_SCENES = {
    "riverside_mix": (-117.45, 33.88, -117.20, 34.06),
}

# 🌍 Earth's Twins — curated, recognisable anchors (named) so matches read well,
# blended with a denser random land sample for breadth.
NAMED_PLACES = {
    "Riverside citrus (CA)": (-117.34, 33.97),
    "Central Valley orchards (CA)": (-119.45, 36.60),
    "Valencia citrus (Spain)": (-0.40, 39.30),
    "São Paulo citrus (Brazil)": (-48.20, -21.20),
    "Nelspruit citrus (S. Africa)": (30.97, -25.47),
    "Murcia citrus (Spain)": (-1.13, 37.99),
    "Sicily citrus (Italy)": (15.10, 37.45),
    "Punjab wheat (India)": (75.50, 30.90),
    "US Corn Belt (Iowa)": (-93.50, 42.00),
    "Sahara dunes (Algeria)": (2.00, 27.00),
    "Amazon rainforest (Brazil)": (-62.00, -4.00),
    "Boreal forest (Canada)": (-105.0, 55.0),
    "Manhattan (USA)": (-73.97, 40.78),
    "Tokyo (Japan)": (139.69, 35.69),
    "Nile Delta (Egypt)": (31.20, 30.80),
    "Mekong Delta (Vietnam)": (105.80, 9.80),
    "Pampas (Argentina)": (-61.50, -34.00),
    "Murray-Darling (Australia)": (146.0, -34.5),
    "Alps (Switzerland)": (8.00, 46.55),
    "Greenland ice margin": (-46.0, 67.0),
}

YEARS = list(range(2017, 2025))  # AlphaEarth/GSE annual coverage
TWIN_YEAR = 2023
BUFFER_M = 1280  # ~ a small grove patch at 10 m
GRID_BUFFER_M = 2560  # larger ROI for grid scenarios
N_RANDOM_GLOBAL = 1500  # extra random land points for the twins bank
CDL_YEAR = 2022
CDL_CITRUS_CLASS = 72  # USDA CDL "Citrus"
N_SHOWDOWN = 60  # ~half citrus, half other

# Illinois maize (the "all-in-one" linear demo) — mirrors examples/demo.ipynb,
# which labels points from the SPAM2020 global crop rasters.
SPAM_H_CSV = os.getenv("SPAM_H_CSV", "spam2020V2r0_global_H_TA.csv")  # harvested area
SPAM_Y_CSV = os.getenv("SPAM_Y_CSV", "spam2020V2r0_global_Y_TA.csv")  # yield
SPAM_CROP_COL = "MAIZ_A"
MAIZE_COUNTRY = "US"
MAIZE_STATE = "Illinois"
MAIZE_MIN_AREA_HA = 2500.0
N_MAIZE = 60
MAIZE_TEMPORAL = ("2019-06-01", "2019-08-31")
MAIZE_YEAR = int(MAIZE_TEMPORAL[0][:4])

CACHE = Path(__file__).resolve().parent / H.CACHE_DIRNAME


# ---------------------------------------------------------------------------
# Earth Engine + rs_embed lazy setup
# ---------------------------------------------------------------------------
def _init_ee() -> None:
    import ee

    # Newer Earth Engine accounts require a Cloud project; allow it via env var.
    project = os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    def _initialize() -> None:
        ee.Initialize(project=project) if project else ee.Initialize()

    try:
        _initialize()
    except Exception:
        ee.Authenticate()
        _initialize()


def _rs_embed():
    from rs_embed import (
        BBox,
        ExportConfig,
        ExportTarget,
        OutputSpec,
        PointBuffer,
        TemporalSpec,
        export_batch,
        get_embedding,
    )

    return dict(
        BBox=BBox,
        ExportConfig=ExportConfig,
        ExportTarget=ExportTarget,
        OutputSpec=OutputSpec,
        PointBuffer=PointBuffer,
        TemporalSpec=TemporalSpec,
        export_batch=export_batch,
        get_embedding=get_embedding,
    )


# ---------------------------------------------------------------------------
# 🌍 Earth's Twins bank
# ---------------------------------------------------------------------------
def _random_land_points(n: int, seed: int = 7) -> list[tuple[float, float]]:
    """Random global points biased to inhabited latitudes; ocean is filtered
    later by dropping embeddings that come back all-zero / non-finite."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180, 180, size=n)
    lat = rng.uniform(-55, 68, size=n)
    return list(zip(lon.tolist(), lat.tolist(), strict=True))


def build_twins_bank(quick: bool) -> None:
    api = _rs_embed()
    vectors, lons, lats, names = [], [], [], []
    pts = list(NAMED_PLACES.items())
    n_rand = 50 if quick else N_RANDOM_GLOBAL
    pts += [(f"land_{i}", ll) for i, ll in enumerate(_random_land_points(n_rand))]

    print(f"[twins] embedding {len(pts)} points with '{PRECOMPUTED_MODEL}' ({TWIN_YEAR})")
    for i, (name, (lon, lat)) in enumerate(pts):
        try:
            emb = api["get_embedding"](
                PRECOMPUTED_MODEL,
                spatial=api["PointBuffer"](lon=float(lon), lat=float(lat), buffer_m=BUFFER_M),
                temporal=api["TemporalSpec"].year(TWIN_YEAR),
                output=api["OutputSpec"].pooled(),
            )
            v = H.pooled_vector(emb.data)
            if not np.isfinite(v).all() or np.allclose(v, 0):
                continue  # ocean / no-data
            vectors.append(v.astype(np.float32))
            lons.append(float(lon))
            lats.append(float(lat))
            names.append(str(name))
        except Exception as e:  # noqa: BLE001 — best-effort sampling
            print(f"  skip {name} ({lon:.2f},{lat:.2f}): {e!r}")
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(pts)} ({len(vectors)} kept)")

    if not vectors:
        raise RuntimeError("twins bank empty — check GEE auth / model availability")
    out = CACHE / "twins_bank.npz"
    np.savez_compressed(
        out,
        vectors=np.stack(vectors).astype(np.float32),
        lon=np.array(lons, dtype=np.float32),
        lat=np.array(lats, dtype=np.float32),
        names=np.array(names, dtype=object),
        model=np.array(PRECOMPUTED_MODEL),
        year=np.array(TWIN_YEAR),
    )
    print(f"[twins] wrote {out} ({len(vectors)} points, dim={vectors[0].shape[0]})")


# ---------------------------------------------------------------------------
# ⏳ Satellite Time Machine
# ---------------------------------------------------------------------------
def build_timemachine(quick: bool) -> None:
    api = _rs_embed()
    years = YEARS[:3] if quick else YEARS
    payload: dict[str, np.ndarray] = {}
    meta_hotspots = []
    for slug, (lon, lat) in HOTSPOTS.items():
        grids = []
        ok_years = []
        print(f"[time] {slug} ({lon:.2f},{lat:.2f}) over {years[0]}–{years[-1]}")
        for yr in years:
            try:
                emb = api["get_embedding"](
                    PRECOMPUTED_MODEL,
                    spatial=api["PointBuffer"](
                        lon=float(lon), lat=float(lat), buffer_m=GRID_BUFFER_M
                    ),
                    temporal=api["TemporalSpec"].year(int(yr)),
                    output=api["OutputSpec"].grid(),
                )
                grids.append(H.to_dhw(emb.data))
                ok_years.append(int(yr))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {slug} {yr}: {e!r}")
        if not grids:
            continue
        # Pad/crop grids to a common (H,W) so they stack.
        hmin = min(g.shape[1] for g in grids)
        wmin = min(g.shape[2] for g in grids)
        grids = [g[:, :hmin, :wmin] for g in grids]
        payload[f"{slug}__grids"] = np.stack(grids).astype(np.float32)
        payload[f"{slug}__years"] = np.array(ok_years, dtype=np.int32)
        payload[f"{slug}__lon"] = np.array(float(lon), dtype=np.float32)
        payload[f"{slug}__lat"] = np.array(float(lat), dtype=np.float32)
        meta_hotspots.append(slug)
    if not payload:
        raise RuntimeError("time machine empty — check GEE auth")
    payload["hotspots"] = np.array(meta_hotspots, dtype=object)
    out = CACHE / "timemachine.npz"
    np.savez_compressed(out, **payload)
    print(f"[time] wrote {out} ({len(meta_hotspots)} hotspots)")


# ---------------------------------------------------------------------------
# 🎨 Instant Land-Cover scenes
# ---------------------------------------------------------------------------
def build_landcover(quick: bool) -> None:
    api = _rs_embed()
    (CACHE / "landcover").mkdir(parents=True, exist_ok=True)
    models = MODEL_LIST[:2] if quick else MODEL_LIST
    for scene, (minlon, minlat, maxlon, maxlat) in LANDCOVER_SCENES.items():
        bbox = api["BBox"](minlon=minlon, minlat=minlat, maxlon=maxlon, maxlat=maxlat)
        payload: dict[str, np.ndarray] = {}
        print(f"[land] scene '{scene}' across {models}")
        for m in models:
            try:
                temporal = (
                    api["TemporalSpec"].year(TWIN_YEAR)
                    if m == PRECOMPUTED_MODEL
                    else api["TemporalSpec"].range("2022-06-01", "2022-09-01")
                )
                emb = api["get_embedding"](
                    m, spatial=bbox, temporal=temporal, output=api["OutputSpec"].grid()
                )
                payload[f"grid__{m}"] = H.to_dhw(emb.data)
            except Exception as e:  # noqa: BLE001
                print(f"  skip {scene}/{m}: {e!r}")
        if payload:
            out = CACHE / "landcover" / f"{scene}.npz"
            np.savez_compressed(out, **payload)
            print(f"[land] wrote {out} ({len(payload)} model grids)")


# ---------------------------------------------------------------------------
# 🏆 Model Showdown — citrus mapping with USDA CDL labels
# ---------------------------------------------------------------------------
def _sample_cdl_citrus(n: int, quick: bool) -> tuple[list[tuple[float, float]], list[int]]:
    """Stratified sample of citrus (CDL class 72) vs. non-citrus points over the
    Southern California citrus belt. Returns ([(lon,lat)], [label])."""
    import ee

    minlon, minlat, maxlon, maxlat = CITRUS_REGION
    region = ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat])
    cdl = ee.Image(f"USDA/NASS/CDL/{CDL_YEAR}").select("cropland")
    citrus = cdl.eq(CDL_CITRUS_CLASS).rename("is_citrus")
    per_class = max(4, (n if not quick else 12) // 2)
    fc = citrus.stratifiedSample(
        numPoints=per_class,
        classBand="is_citrus",
        region=region,
        scale=30,
        geometries=True,
        seed=42,
    )
    feats = fc.getInfo()["features"]
    pts, labels = [], []
    for f in feats:
        lon, lat = f["geometry"]["coordinates"]
        pts.append((float(lon), float(lat)))
        labels.append(int(f["properties"]["is_citrus"]))
    if not pts:
        raise RuntimeError(
            f"No CDL samples in region for {CDL_YEAR}. "
            "Verify CDL citrus coverage or fall back to SPAM citrus-area labels."
        )
    print(f"[showdown] CDL sampled {sum(labels)} citrus / {len(labels) - sum(labels)} other")
    return pts, labels


def build_showdown(quick: bool) -> None:
    api = _rs_embed()
    out_dir = CACHE / "showdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    pts, labels = _sample_cdl_citrus(N_SHOWDOWN, quick)
    spatials = [
        api["PointBuffer"](lon=lon, lat=lat, buffer_m=BUFFER_M) for (lon, lat) in pts
    ]
    names = [f"pt{i:03d}" for i in range(len(pts))]
    models = MODEL_LIST[:3] if quick else MODEL_LIST

    # Save labels aligned 1:1 with the export point order.
    np.savez_compressed(
        out_dir / "labels.npz",
        y=np.array(labels, dtype=np.int16),
        lon=np.array([p[0] for p in pts], dtype=np.float32),
        lat=np.array([p[1] for p in pts], dtype=np.float32),
        names=np.array(names, dtype=object),
        cdl_year=np.array(CDL_YEAR),
        citrus_class=np.array(CDL_CITRUS_CLASS),
    )

    print(f"[showdown] export_batch: {len(spatials)} points × {models}")
    api["export_batch"](
        spatials=spatials,
        temporal=api["TemporalSpec"].range("2022-06-01", "2022-09-01"),
        models=models,
        target=api["ExportTarget"].combined(str(out_dir / "citrus_export.npz")),
        output=api["OutputSpec"].grid(),
        config=api["ExportConfig"](
            save_inputs=True, save_embeddings=True, continue_on_error=True, resume=True
        ),
        backend="auto",
    )
    print(f"[showdown] wrote bundle to {out_dir}")


# ---------------------------------------------------------------------------
# 🌽 Illinois maize — the "all-in-one" linear demo (iguide_demo_maize.ipynb)
# ---------------------------------------------------------------------------
def _load_spam_illinois(csv: Path, value_name: str):
    """Load SPAM points for Illinois with one crop value column (see demo.ipynb)."""
    import pandas as pd

    usecols = ["grid_code", "x", "y", "FIPS0", "ADM1_NAME", "year_data", SPAM_CROP_COL]
    t = pd.read_csv(csv, usecols=usecols, encoding="utf-8-sig").rename(
        columns={"x": "lon", "y": "lat", SPAM_CROP_COL: value_name}
    )
    t["FIPS0"] = t["FIPS0"].astype(str).str.upper()
    t["ADM1_NAME"] = t["ADM1_NAME"].astype(str)
    t[value_name] = pd.to_numeric(t[value_name], errors="coerce")
    t = t[
        t["FIPS0"].eq(MAIZE_COUNTRY)
        & t["ADM1_NAME"].str.contains(MAIZE_STATE, case=False, na=False)
    ]
    return t[["grid_code", "lon", "lat", value_name]].dropna()


def build_maize(quick: bool) -> None:
    api = _rs_embed()
    out_dir = CACHE / "maize"
    out_dir.mkdir(parents=True, exist_ok=True)

    h_csv, y_csv = Path(SPAM_H_CSV), Path(SPAM_Y_CSV)
    if not (h_csv.exists() and y_csv.exists()):
        raise RuntimeError(
            f"SPAM CSVs not found ({SPAM_H_CSV}, {SPAM_Y_CSV}). Download SPAM2020 "
            "H_TA + Y_TA (or set SPAM_H_CSV / SPAM_Y_CSV). See examples/demo.ipynb."
        )

    area = _load_spam_illinois(h_csv, "area")
    area = area[area["area"] >= MAIZE_MIN_AREA_HA].drop_duplicates("grid_code")
    yld = _load_spam_illinois(y_csv, "label")
    yld = yld[yld["label"] > 0].drop_duplicates("grid_code")
    df = (
        area.merge(yld[["grid_code", "label"]], on="grid_code", how="inner")
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )
    n = min(len(df), 12 if quick else N_MAIZE)
    df = df.iloc[:n]
    if df.empty:
        raise RuntimeError("No Illinois maize points after filtering.")

    spatials = [
        api["PointBuffer"](lon=float(r.lon), lat=float(r.lat), buffer_m=BUFFER_M)
        for r in df.itertuples()
    ]
    names = [f"pt{i:03d}" for i in range(len(df))]
    np.savez_compressed(
        out_dir / "labels.npz",
        y=df["label"].to_numpy(dtype="f4"),
        lon=df["lon"].to_numpy(dtype="f4"),
        lat=df["lat"].to_numpy(dtype="f4"),
        names=np.array(names, dtype=object),
        crop=np.array("maize"),
        state=np.array(MAIZE_STATE),
    )

    models = MODEL_LIST[:3] if quick else MODEL_LIST

    # Acts 1–2: grids for the single "reveal" field across models.
    scene: dict[str, np.ndarray] = {}
    for m in models:
        try:
            temporal = (
                api["TemporalSpec"].year(MAIZE_YEAR)
                if m == PRECOMPUTED_MODEL
                else api["TemporalSpec"].range(*MAIZE_TEMPORAL)
            )
            emb = api["get_embedding"](
                m, spatial=spatials[0], temporal=temporal, output=api["OutputSpec"].grid()
            )
            scene[f"grid__{m}"] = H.to_dhw(emb.data)
        except Exception as e:  # noqa: BLE001
            print(f"  skip scene/{m}: {e!r}")
    if scene:
        np.savez_compressed(out_dir / "scene.npz", **scene)

    # Act 3: shared-input export across all models (gse converts range→year).
    print(f"[maize] export_batch: {len(spatials)} points × {models}")
    api["export_batch"](
        spatials=spatials,
        temporal=api["TemporalSpec"].range(*MAIZE_TEMPORAL),
        models=models,
        target=api["ExportTarget"].combined(str(out_dir / "maize_export.npz")),
        output=api["OutputSpec"].grid(),
        config=api["ExportConfig"](
            save_inputs=True, save_embeddings=True, continue_on_error=True, resume=True
        ),
        backend="auto",
    )
    print(f"[maize] wrote bundle to {out_dir}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
BUILDERS = {
    "twins": build_twins_bank,
    "timemachine": build_timemachine,
    "landcover": build_landcover,
    "showdown": build_showdown,
    "maize": build_maize,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the rs-embed I-GUIDE demo cache.")
    ap.add_argument("--quick", action="store_true", help="tiny build for a smoke test")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=list(BUILDERS),
        help="build only these artifacts (default: all)",
    )
    args = ap.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {CACHE}")
    _init_ee()

    targets = args.only or list(BUILDERS)
    built, failed = [], {}
    for name in targets:
        try:
            print(f"\n=== build: {name} (quick={args.quick}) ===")
            BUILDERS[name](args.quick)
            built.append(name)
        except Exception as e:  # noqa: BLE001 — keep building the rest
            failed[name] = repr(e)
            traceback.print_exc()

    (CACHE / "cache_meta.json").write_text(
        json.dumps(
            {
                "story": "SIGSPATIAL'26 / Riverside citrus",
                "precomputed_model": PRECOMPUTED_MODEL,
                "models": MODEL_LIST,
                "years": YEARS,
                "twin_year": TWIN_YEAR,
                "cdl_year": CDL_YEAR,
                "hotspots": list(HOTSPOTS),
                "landcover_scenes": list(LANDCOVER_SCENES),
                "built": built,
                "failed": failed,
                "quick": args.quick,
            },
            indent=2,
        )
    )
    print(f"\nDone. built={built} failed={list(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
