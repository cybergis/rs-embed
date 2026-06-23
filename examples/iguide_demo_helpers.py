"""Demo-only helpers for the rs-embed I-GUIDE interactive demo.

These utilities power the four interactive scenarios in ``iguide_demo.ipynb``:

* 🌍 **Earth's Twins** — :func:`cosine_topk` nearest-neighbour search over a
  pre-built global embedding bank.
* ⏳ **Satellite Time Machine** — :func:`change_curve` measures how far each
  year's embedding drifts from a baseline year.
* 🎨 **Instant Land-Cover** — :func:`kmeans_landcover` turns a grid embedding
  into an unsupervised, colourised segmentation map.
* 🏆 **Model Showdown** — :func:`classification_leaderboard` trains one tiny
  classifier per model on shared inputs and returns a fair, ranked scoreboard.

They are intentionally kept **out of the** ``rs_embed`` **library** (demo glue,
not API surface) and depend only on numpy / scikit-learn / matplotlib so the
notebook runs in cached mode with no Google Earth Engine or GPU.

This module is the single source of truth for the on-disk cache schema used by
both ``build_demo_cache.py`` (writer) and ``iguide_demo.ipynb`` (reader).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Cache schema  (shared contract between builder and notebook)
# ---------------------------------------------------------------------------
#
# examples/iguide_demo_cache/
#   twins_bank.npz            # keys: vectors (N,D) float32, lon (N,), lat (N,),
#                             #       names (N,) optional str, model (scalar str)
#   timemachine.npz           # keys: per hotspot:  <slug>__years (Y,) int,
#                             #       <slug>__grids (Y,D,H,W) float32,
#                             #       <slug>__lon, <slug>__lat (scalars);
#                             #       hotspots (list[str]) in meta json
#   landcover/<scene>.npz     # keys per model: grid__<model> (D,H,W),
#                             #       rgb (H,W,3) optional S2 quicklook
#   showdown/                 # an rs_embed export_batch bundle (combined .npz +
#                             #   .json manifest) PLUS labels.npz (y (N,) int,
#                             #   lon (N,), lat (N,))
#   cache_meta.json           # human-readable description of what was built
#
# All embedding grids are stored channel-first as (D, H, W) float32.

CACHE_DIRNAME = "iguide_demo_cache"

LANDCOVER_PALETTE = np.array(
    [
        [0.84, 0.19, 0.15],  # 0 red
        [0.20, 0.51, 0.74],  # 1 blue
        [0.30, 0.69, 0.29],  # 2 green
        [0.98, 0.75, 0.18],  # 3 amber
        [0.60, 0.31, 0.64],  # 4 purple
        [0.40, 0.76, 0.65],  # 5 teal
        [0.96, 0.50, 0.75],  # 6 pink
        [0.55, 0.63, 0.80],  # 7 slate
        [0.65, 0.34, 0.16],  # 8 brown
        [0.50, 0.50, 0.50],  # 9 grey
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Array shaping
# ---------------------------------------------------------------------------
def to_dhw(arr: Any) -> np.ndarray:
    """Coerce a 2-D/3-D embedding array (or xarray) to channel-first ``(D,H,W)``.

    Mirrors the heuristic in ``plot_utils._to_dhw`` so cached grids and live
    ``Embedding.data`` (xarray with dims ``d,y,x``) are handled identically.
    """
    if hasattr(arr, "values"):  # xarray.DataArray
        arr = arr.values
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2-D/3-D array, got shape {arr.shape}")
    common_d = (16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1408)
    # HWD -> DHW when the last axis looks like the feature dimension.
    if arr.shape[-1] in common_d and arr.shape[0] not in common_d:
        arr = np.moveaxis(arr, -1, 0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def pooled_vector(arr: Any) -> np.ndarray:
    """Return a 1-D pooled descriptor for either a pooled vector or a grid."""
    a = np.asarray(arr.values if hasattr(arr, "values") else arr, dtype=np.float32)
    if a.ndim == 1:
        return a
    dhw = to_dhw(a)
    return np.nanmean(dhw.reshape(dhw.shape[0], -1), axis=1).astype(np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation (safe for zero rows)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / (n + eps)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# ---------------------------------------------------------------------------
# 🌍 Earth's Twins — global similarity search
# ---------------------------------------------------------------------------
def cosine_topk(
    query: np.ndarray,
    bank: np.ndarray,
    k: int = 5,
    *,
    exclude_self_radius_deg: float = 0.0,
    query_lonlat: tuple[float, float] | None = None,
    bank_lonlat: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and cosine similarities of the top-``k`` bank rows.

    Parameters
    ----------
    query : np.ndarray
        Query embedding, shape ``(D,)``.
    bank : np.ndarray
        Candidate embeddings, shape ``(N, D)``.
    k : int
        Number of neighbours to return.
    exclude_self_radius_deg : float
        If > 0 and lon/lat are supplied, drop bank points within this great-circle
        (degree) radius of the query so a click doesn't just match itself.

    Returns
    -------
    (idx, sims) : tuple[np.ndarray, np.ndarray]
        ``idx`` are bank row indices (best first); ``sims`` the matching scores.
    """
    q = l2_normalize(np.asarray(query, dtype=np.float32).ravel())
    b = l2_normalize(np.asarray(bank, dtype=np.float32))
    sims = b @ q
    mask = np.ones(sims.shape[0], dtype=bool)
    if exclude_self_radius_deg > 0 and query_lonlat is not None and bank_lonlat is not None:
        d = np.hypot(bank_lonlat[:, 0] - query_lonlat[0], bank_lonlat[:, 1] - query_lonlat[1])
        mask &= d > float(exclude_self_radius_deg)
    sims_masked = np.where(mask, sims, -np.inf)
    k = int(min(k, int(mask.sum())))
    if k <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
    idx = np.argpartition(-sims_masked, k - 1)[:k]
    idx = idx[np.argsort(-sims_masked[idx])]
    return idx, sims[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# ⏳ Satellite Time Machine — temporal change
# ---------------------------------------------------------------------------
def change_curve(
    year_grids: np.ndarray | list[np.ndarray],
    *,
    baseline_index: int = 0,
    metric: str = "cosine",
) -> np.ndarray:
    """Distance of each year's pooled embedding from a baseline year.

    Parameters
    ----------
    year_grids : array or list
        Per-year grids, shape ``(Y, D, H, W)`` (or a list of ``(D,H,W)``).
    baseline_index : int
        Year index treated as "before" (distance 0 by construction).
    metric : {"cosine", "l2"}
        ``cosine`` returns ``1 - cos`` in ``[0, 2]``; ``l2`` returns Euclidean
        distance between L2-normalised pooled vectors.

    Returns
    -------
    np.ndarray
        Shape ``(Y,)`` distances; a spike marks the year a change occurred.
    """
    grids = list(year_grids)
    vecs = l2_normalize(np.stack([pooled_vector(g) for g in grids], axis=0))
    base = vecs[int(baseline_index)]
    if metric == "l2":
        return np.linalg.norm(vecs - base, axis=1).astype(np.float32)
    return (1.0 - (vecs @ base)).astype(np.float32)


# ---------------------------------------------------------------------------
# 🎨 Instant Land-Cover — unsupervised segmentation
# ---------------------------------------------------------------------------
def kmeans_landcover(
    grid_dhw: Any,
    k: int = 6,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster a grid embedding into ``k`` land-cover-like regions.

    Returns
    -------
    (labels, rgb) : tuple[np.ndarray, np.ndarray]
        ``labels`` is an ``(H, W)`` int map; ``rgb`` is an ``(H, W, 3)`` float
        image colourised with :data:`LANDCOVER_PALETTE` for direct display.
    """
    from sklearn.cluster import KMeans

    dhw = to_dhw(grid_dhw)
    d, h, w = dhw.shape
    feats = dhw.reshape(d, h * w).T  # (H*W, D)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = l2_normalize(feats)
    km = KMeans(n_clusters=int(k), n_init=10, random_state=int(seed))
    labels = km.fit_predict(feats).reshape(h, w)
    rgb = LANDCOVER_PALETTE[labels % len(LANDCOVER_PALETTE)]
    return labels.astype(np.int16), rgb.astype(np.float32)


def pca_rgb(grid_dhw: Any, *, robust_lo: float = 2.0, robust_hi: float = 98.0) -> np.ndarray:
    """Project a grid embedding to a 3-channel pseudo-colour image via PCA.

    Standalone (no plot_utils dependency) so cache rendering stays light. PCA is
    fit on the grid itself, then percentile-stretched to ``[0, 1]``.
    """
    from sklearn.decomposition import PCA

    dhw = to_dhw(grid_dhw)
    d, h, w = dhw.shape
    feats = np.nan_to_num(dhw.reshape(d, h * w).T, nan=0.0, posinf=0.0, neginf=0.0)
    comp = PCA(n_components=3, random_state=0).fit_transform(feats)
    # Deterministic sign so reruns / model swaps look stable.
    for c in range(3):
        if np.sum(comp[:, c]) < 0:
            comp[:, c] *= -1.0
    img = comp.reshape(h, w, 3)
    lo = np.percentile(img, robust_lo, axis=(0, 1))
    hi = np.percentile(img, robust_hi, axis=(0, 1))
    img = np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# 🏆 Model Showdown — fair cross-model leaderboard
# ---------------------------------------------------------------------------
@dataclass
class ModelScore:
    """One row of the Model Showdown leaderboard."""

    model: str
    accuracy: float
    f1: float
    n_train: int
    n_test: int
    dim: int


def classification_leaderboard(
    features_by_model: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    test_size: float = 0.3,
    seed: int = 42,
    n_estimators: int = 200,
) -> list[ModelScore]:
    """Train one RandomForest per model on a **shared** train/test split.

    The identical split + labels across all models is what makes the comparison
    fair — every model is judged on the same points, differing only in which
    foundation model produced the features.

    Parameters
    ----------
    features_by_model : dict[str, np.ndarray]
        Mapping ``model_name -> (N, dim)`` pooled feature matrix. ``N`` and the
        row order must match ``labels`` for every model.
    labels : np.ndarray
        Shape ``(N,)`` integer class labels (e.g. citrus=1, other=0).

    Returns
    -------
    list[ModelScore]
        Sorted by F1 (best first).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    y = np.asarray(labels).ravel()
    n = y.shape[0]
    stratify = y if len(np.unique(y)) > 1 else None
    idx_tr, idx_te = train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, stratify=stratify
    )
    scores: list[ModelScore] = []
    for name, feats in features_by_model.items():
        X = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim != 2 or X.shape[0] != n:
            # Skip models whose feature matrix doesn't line up with the labels.
            continue
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        clf.fit(X[idx_tr], y[idx_tr])
        pred = clf.predict(X[idx_te])
        scores.append(
            ModelScore(
                model=name,
                accuracy=float(accuracy_score(y[idx_te], pred)),
                f1=float(f1_score(y[idx_te], pred, average="macro")),
                n_train=int(idx_tr.size),
                n_test=int(idx_te.size),
                dim=int(X.shape[1]),
            )
        )
    scores.sort(key=lambda s: s.f1, reverse=True)
    return scores


@dataclass
class RegressionScore:
    """One row of a regression leaderboard (e.g. maize-yield prediction)."""

    model: str
    r2: float
    mae: float
    rmse: float
    n_train: int
    n_test: int
    dim: int


def regression_leaderboard(
    features_by_model: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    test_size: float = 0.3,
    seed: int = 42,
    n_estimators: int = 200,
) -> list[RegressionScore]:
    """Train one RandomForestRegressor per model on a **shared** split.

    The all-in-one maize counterpart to :func:`classification_leaderboard`:
    every model is fit and scored on the identical train/test partition of the
    same points, so the ranking reflects the model, not the data split.

    Parameters
    ----------
    features_by_model : dict[str, np.ndarray]
        ``model_name -> (N, dim)`` pooled features (row order matches targets).
    targets : np.ndarray
        Shape ``(N,)`` continuous labels (e.g. maize yield t/ha).

    Returns
    -------
    list[RegressionScore]
        Sorted by R² (best first).
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    y = np.asarray(targets, dtype=np.float32).ravel()
    n = y.shape[0]
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=test_size, random_state=seed)
    scores: list[RegressionScore] = []
    for name, feats in features_by_model.items():
        X = np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim != 2 or X.shape[0] != n:
            continue
        reg = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        reg.fit(X[idx_tr], y[idx_tr])
        pred = reg.predict(X[idx_te])
        rmse = float(np.sqrt(mean_squared_error(y[idx_te], pred)))
        scores.append(
            RegressionScore(
                model=name,
                r2=float(r2_score(y[idx_te], pred)),
                mae=float(mean_absolute_error(y[idx_te], pred)),
                rmse=rmse,
                n_train=int(idx_tr.size),
                n_test=int(idx_te.size),
                dim=int(X.shape[1]),
            )
        )
    scores.sort(key=lambda s: s.r2, reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Cache access  (graceful, never hard-fails the booth demo)
# ---------------------------------------------------------------------------
@dataclass
class DemoCache:
    """Resolves and loads the on-disk demo cache, tolerating missing pieces.

    The notebook constructs one of these and asks for each scenario's assets;
    any missing file returns ``None`` so the UI can show a friendly notice
    instead of crashing.
    """

    root: Path

    @classmethod
    def find(cls, start: str | Path | None = None) -> DemoCache:
        """Locate ``iguide_demo_cache/`` near the notebook / cwd."""
        here = Path(start) if start else Path.cwd()
        for base in [here, *here.parents]:
            cand = base / CACHE_DIRNAME
            if cand.exists():
                return cls(root=cand)
            cand = base / "examples" / CACHE_DIRNAME
            if cand.exists():
                return cls(root=cand)
        # Default to a sibling of this file even if it doesn't exist yet.
        return cls(root=Path(__file__).resolve().parent / CACHE_DIRNAME)

    @property
    def available(self) -> bool:
        return self.root.exists()

    def meta(self) -> dict[str, Any]:
        p = self.root / "cache_meta.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}

    def _load_npz(self, rel: str) -> dict[str, np.ndarray] | None:
        p = self.root / rel
        if not p.exists():
            return None
        with np.load(p, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def twins_bank(self) -> dict[str, np.ndarray] | None:
        """Global similarity bank: ``vectors``, ``lon``, ``lat``, ``names``."""
        return self._load_npz("twins_bank.npz")

    def timemachine(self) -> dict[str, np.ndarray] | None:
        """Per-hotspot per-year grids (see schema header)."""
        return self._load_npz("timemachine.npz")

    def landcover_scene(self, scene: str) -> dict[str, np.ndarray] | None:
        """Grid embeddings per model for one drawn scene + an RGB quicklook."""
        return self._load_npz(f"landcover/{scene}.npz")

    def showdown_labels(self) -> dict[str, np.ndarray] | None:
        """Citrus labels (``y``, ``lon``, ``lat``) aligned to the export bundle."""
        return self._load_npz("showdown/labels.npz")

    def showdown_dir(self) -> Path:
        """Directory holding the rs-embed ``export_batch`` bundle for #6."""
        return self.root / "showdown"

    # -- maize all-in-one (iguide_demo_maize.ipynb) ------------------------
    def maize_labels(self) -> dict[str, np.ndarray] | None:
        """Continuous yield labels (``y``, ``lon``, ``lat``) for the maize demo."""
        return self._load_npz("maize/labels.npz")

    def maize_scene(self) -> dict[str, np.ndarray] | None:
        """Per-model grid embeddings for the single 'reveal' field (Acts 1–2)."""
        return self._load_npz("maize/scene.npz")

    def maize_dir(self) -> Path:
        """Directory holding the maize ``export_batch`` bundle (Act 3)."""
        return self.root / "maize"


# ---------------------------------------------------------------------------
# Lightweight (torch-free) reader for an rs-embed combined export bundle
# ---------------------------------------------------------------------------
def load_embedding_package(path: str | Path) -> dict[str, Any]:
    """Load an ``.npz`` embedding package saved by the web app (``/api/embed``).

    Returns a dict with:
      * ``meta``   — the request manifest (geometry, years, per-model info)
      * ``pooled`` — ``{model -> (D,) pooled vector}``
      * ``grids``  — ``{model -> (D,H,W) grid}`` (only models whose grid was small
        enough to save server-side)

    Example (in the notebook)::

        import iguide_demo_helpers as H
        pkg = H.load_embedding_package("~/Downloads/rsembed_pkg_ab12cd34ef.npz")
        X = pkg["pooled"]["gse"]          # (D,) descriptor for the ROI
        H.pca_rgb(pkg["grids"]["gse"])    # PCA-RGB image of the grid embedding
    """
    z = np.load(Path(path).expanduser(), allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    pooled = {k[len("pooled__") :]: z[k] for k in z.files if k.startswith("pooled__")}
    grids = {k[len("grid__") :]: z[k] for k in z.files if k.startswith("grid__")}
    return {"meta": meta, "pooled": pooled, "grids": grids}


def load_export_features(
    npz_path: str | Path, json_path: str | Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a combined ``export_batch`` bundle into ``{model -> (N, dim)}``.

    Pools grid embeddings ``(N, D, H, W)`` to ``(N, D)`` by spatial mean so the
    Model Showdown can train on pooled descriptors. Pure numpy/json — avoids
    importing ``rs_embed`` (and torch) so cached mode stays light. Model→array
    keys are resolved from the manifest, not guessed.
    """
    npz_path = Path(npz_path)
    if json_path is None:
        json_path = npz_path.with_suffix(".json")
    manifest = json.loads(Path(json_path).read_text())
    feats: dict[str, np.ndarray] = {}
    with np.load(npz_path, allow_pickle=True) as z:
        keys = set(z.files)
        for entry in manifest.get("models", []) or []:
            name = entry.get("model")
            emb = entry.get("embeddings")
            if not name or not isinstance(emb, dict):
                continue
            arr = None
            if emb.get("npz_key") in keys:
                arr = z[emb["npz_key"]]
            elif emb.get("npz_keys"):
                parts = [z[k] for k in emb["npz_keys"] if k in keys]
                if parts:
                    arr = np.stack(parts, axis=0)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float32)
            if a.ndim == 2:  # already pooled (N, D)
                feats[str(name)] = a
            elif a.ndim >= 3:  # grid (N, D, H, W...) -> spatial mean
                a = a.reshape(a.shape[0], a.shape[1], -1)
                feats[str(name)] = np.nanmean(a, axis=2).astype(np.float32)
    return feats, manifest
