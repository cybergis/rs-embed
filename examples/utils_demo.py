from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "ee_initialize",
    "iter_export_points",
    "iter_model_vectors",
    "load_one_preview_patch",
    "point_key",
    "point_row_index",
    "resolve_input_key",
    "resolve_preview_group_dirs",
    "rings_from_geom_info",
    "stretch_rgb",
    "to_feature_vector",
]


def ee_initialize(ee_module, project):
    kwargs = {"project": project} if project else {}
    ee_module.Initialize(**kwargs)


def rings_from_geom_info(geom_info: dict):
    gtype = geom_info.get("type")
    coords = geom_info.get("coordinates", [])
    if gtype == "Polygon":
        ring_seq = coords
    elif gtype == "MultiPolygon":
        ring_seq = [ring for poly in coords for ring in poly]
    else:
        raise ValueError(f"Unsupported geometry type for plotting: {gtype}")
    return [np.asarray(ring, dtype=float) for ring in ring_seq]


def stretch_rgb(rgb_hwc: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_hwc, dtype=np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(min(3, rgb.shape[-1])):
        ch = rgb[..., c]
        lo, hi = np.nanpercentile(ch, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def resolve_input_key(bundle, model: str, model_entry: dict):
    entry = model_entry if isinstance(model_entry, dict) else {}
    inp = entry.get("input")
    if not isinstance(inp, dict):
        inp = entry.get("inputs")
    inp = inp if isinstance(inp, dict) else {}

    explicit_keys = []
    if isinstance(inp.get("npz_key"), str):
        explicit_keys.append(inp["npz_key"])
    if isinstance(inp.get("npz_keys"), list):
        explicit_keys.extend(k for k in inp["npz_keys"] if isinstance(k, str))

    key = _first_existing_key(bundle, explicit_keys)
    if key is not None:
        return key

    prefixes = (f"input_chw__{model}", f"input_chw__{model}__")
    return next((key for key in bundle.keys() if key.startswith(prefixes)), None)


def load_one_preview_patch(group_dirs, df):
    for group_dir, pm, npz_path in _iter_preview_point_records(group_dirs):
        preview = _extract_preview_from_record(group_dir, pm, npz_path, df)
        if preview is not None:
            return preview
    return None


def resolve_preview_group_dirs(candidates):
    for cand in candidates:
        if not cand.exists():
            continue
        if any(cand.glob("p*.npz")):
            return [cand]
        group_dirs = sorted(p for p in cand.iterdir() if p.is_dir() and p.name.startswith("group"))
        if group_dirs:
            return group_dirs
    return []


def point_key(lon: float, lat: float, decimals: int = 4) -> str:
    return f"{round(float(lon), decimals):.{decimals}f}|{round(float(lat), decimals):.{decimals}f}"


def to_feature_vector(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        vec = a
    elif a.ndim != 3:
        vec = a.reshape(-1)
    else:
        head_dim_like = _dim_like_embedding(a.shape[0])
        tail_dim_like = _dim_like_embedding(a.shape[-1])
        if head_dim_like and not tail_dim_like:
            b = a
        elif tail_dim_like and not head_dim_like:
            b = np.moveaxis(a, -1, 0)
        else:
            b = None
        vec = np.nanmean(b.reshape(b.shape[0], -1), axis=1) if b is not None else a.reshape(-1)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def iter_export_points(export_root: Path):
    agg = export_root / "_all_points_manifest.json"
    if agg.exists():
        point_manifests = _read_json_list(agg)
        if point_manifests is None:
            print("Warning: failed to read _all_points_manifest.json:", agg)
        else:
            yield from _iter_manifest_points(point_manifests, export_root, "flat_aggregate")
            return

    sidecars = list(_iter_sidecar_points(export_root))
    if sidecars:
        yield from sidecars
        return

    yield from _iter_legacy_group_points(export_root)


def point_row_index(pm: dict, point_index_map: dict[str, int]):
    spatial = pm.get("spatial")
    if not isinstance(spatial, dict):
        return None
    return point_index_map.get(point_key(spatial.get("lon"), spatial.get("lat")))


def iter_model_vectors(bundle, pm: dict):
    models = pm.get("models", [])
    if not isinstance(models, list):
        return

    for m_entry in models:
        if not isinstance(m_entry, dict):
            continue

        model = str(m_entry.get("model", "")).strip()
        if not model:
            continue

        candidate_keys = _candidate_embedding_keys(m_entry)
        if not candidate_keys:
            continue

        arr = next((bundle[key] for key in candidate_keys if key in bundle), None)
        if arr is None:
            continue

        yield model, to_feature_vector(arr)


def _safe_nonneg_int(value):
    try:
        i = int(value)
    except Exception:
        return None
    return i if i >= 0 else None


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_json_list(path: Path):
    data = _read_json(path)
    return data if isinstance(data, list) else None


def _read_json_dict(path: Path):
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def _first_existing_key(bundle, keys):
    return next((k for k in keys if isinstance(k, str) and k in bundle), None)


def _load_preview_point_manifests(group_dir: Path):
    for name in ("_group_manifest.json", "_all_points_manifest.json"):
        manifest_path = group_dir / name
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))

    sidecars = []
    for jp in sorted(group_dir.glob("p*.json")):
        pm_side = _read_json_dict(jp)
        if pm_side is not None:
            sidecars.append(pm_side)
    return sidecars


def _iter_preview_point_records(group_dirs):
    for group_dir in group_dirs:
        point_manifests = _load_preview_point_manifests(group_dir) or []
        for pm in point_manifests:
            if not isinstance(pm, dict):
                continue
            point_index = _safe_nonneg_int(pm.get("point_index", -1))
            if point_index is None:
                continue
            npz_path = group_dir / f"p{point_index:05d}.npz"
            if npz_path.exists():
                yield group_dir, pm, npz_path


def _preview_lon_lat(pm: dict):
    spatial = pm.get("spatial")
    spatial = spatial if isinstance(spatial, dict) else {}
    lon = float(spatial.get("lon")) if spatial.get("lon") is not None else np.nan
    lat = float(spatial.get("lat")) if spatial.get("lat") is not None else np.nan
    return lon, lat


def _nearest_label_value(df, lon: float, lat: float):
    if not (np.isfinite(lon) and np.isfinite(lat)) or len(df) == 0:
        return np.nan
    d2 = (df["lon"].to_numpy() - lon) ** 2 + (df["lat"].to_numpy() - lat) ** 2
    return float(df.iloc[int(np.argmin(d2))]["label"])


def _extract_preview_from_record(group_dir: Path, pm: dict, npz_path: Path, df):
    with np.load(npz_path, allow_pickle=False) as bundle:
        for m_entry in pm.get("models", []):
            if not isinstance(m_entry, dict):
                continue

            model = str(m_entry.get("model", "")).strip()
            if not model:
                continue

            key = resolve_input_key(bundle, model, m_entry)
            if key is None:
                continue

            chw = np.asarray(bundle[key], dtype=np.float32)
            if chw.ndim == 4:
                chw = chw[0]
            if chw.ndim != 3:
                continue

            lon, lat = _preview_lon_lat(pm)
            return {
                "group": group_dir.name,
                "npz": npz_path.name,
                "model": model,
                "key": key,
                "chw": chw,
                "lon": lon,
                "lat": lat,
                "label": _nearest_label_value(df, lon, lat),
            }
    return None


def _dim_like_embedding(size: int) -> bool:
    return int(size) in {
        32,
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        4096,
    }


def _iter_manifest_points(point_manifests, base_dir: Path, mode: str):
    if not isinstance(point_manifests, list):
        return
    for pm in point_manifests:
        if not isinstance(pm, dict):
            continue
        point_index = _safe_nonneg_int(pm.get("point_index", -1))
        if point_index is None:
            continue
        yield pm, base_dir / f"p{point_index:05d}.npz", mode


def _iter_sidecar_points(export_root: Path):
    for jp in sorted(export_root.glob("p*.json")):
        pm = _read_json_dict(jp)
        if pm is not None:
            yield pm, export_root / f"{jp.stem}.npz", "flat_sidecars"


def _iter_legacy_group_points(export_root: Path):
    if not export_root.exists():
        return
    group_dirs = sorted(
        p for p in export_root.iterdir() if p.is_dir() and p.name.startswith("group")
    )
    for group_dir in group_dirs:
        point_manifests = _read_json_list(group_dir / "_group_manifest.json")
        if point_manifests is not None:
            yield from _iter_manifest_points(point_manifests, group_dir, "legacy_groups")


def _candidate_embedding_keys(m_entry: dict):
    emb_ref = m_entry.get("embedding")
    if not isinstance(emb_ref, dict):
        return []
    if isinstance(emb_ref.get("npz_key"), str):
        return [emb_ref["npz_key"]]
    if isinstance(emb_ref.get("npz_keys"), list):
        return [k for k in emb_ref["npz_keys"] if isinstance(k, str)]
    return []
