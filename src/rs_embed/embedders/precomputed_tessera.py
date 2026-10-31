from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..core.embedding import Embedding
from ..core.errors import ModelError
from ..core.registry import register
from ..core.specs import (
    OutputSpec,
    SensorSpec,
    SpatialSpec,
    TemporalSpec,
)
from ..core.types import EmbedderCapabilities
from ..tools.projection import (
    COMMON_CRS,
    CommonGrid,
    common_grid,
    crs_label,
    project_pixel_centers,
    source_pixel_indices,
    warn_if_utm_boundary,
)
from .base import EmbedderBase
from .config import model_config_value
from .meta import build_meta

_EMBED_DIMS = (64, 128, 256, 512, 768, 1024)
# GeoTessera is a 10 m product; its tiles are resampled onto the common
# EPSG:3857 grid at this pixel size.
_TESSERA_SCALE_M = 10


def _resolve_tessera_cache_dir(
    model_config: dict[str, Any] | None,
    sensor: SensorSpec | None,
) -> str | None:
    """Resolve the GeoTessera tile cache directory.

    Precedence: ``model_config['cache_dir']`` > legacy
    ``sensor.collection='cache:<dir>'`` (deprecated) > ``RS_EMBED_TESSERA_CACHE``
    env var > geotessera default cache.
    """
    v = model_config_value(model_config, "cache_dir")
    if v is not None and str(v).strip():
        return str(v).strip()
    cache_dir = os.environ.get("RS_EMBED_TESSERA_CACHE")
    collection = getattr(sensor, "collection", None) if sensor else None
    if isinstance(collection, str) and collection.startswith("cache:"):
        warnings.warn(
            "tessera: sensor.collection='cache:<dir>' is deprecated; pass "
            "model_config={'cache_dir': '<dir>'} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        cache_dir = collection.replace("cache:", "", 1).strip() or cache_dir
    return cache_dir


def _year_from_temporal(temporal: TemporalSpec | None, default_year: int = 2021) -> int:
    if temporal is None:
        warnings.warn(
            f"temporal=None: tessera defaults to year {default_year}. Pass "
            "temporal=TemporalSpec.year(...) for reproducible, self-documented results.",
            UserWarning,
            stacklevel=2,
        )
        return default_year
    temporal.validate()
    if temporal.mode == "year" and temporal.year is not None:
        return int(temporal.year)
    if temporal.mode == "range" and temporal.start:
        return int(str(temporal.start)[:4])
    return default_year


def _pool(chw: np.ndarray, pooling: str) -> np.ndarray:
    if pooling == "mean":
        return chw.mean(axis=(1, 2)).astype(np.float32)
    if pooling == "max":
        return chw.max(axis=(1, 2)).astype(np.float32)
    raise ModelError(f"Unknown pooling={pooling!r} (expected 'mean' or 'max').")


def _gather_hwc(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pick ``[N, D]`` pixel vectors from an HWC or CHW tile without copying the tile."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ModelError(f"Unexpected embedding ndim={a.ndim}, shape={a.shape}")
    if a.shape[-1] in _EMBED_DIMS:
        return a[rows, cols, :].astype(np.float32)
    if a.shape[0] in _EMBED_DIMS:
        return a[:, rows, cols].T.astype(np.float32)
    raise ModelError(f"Unexpected embedding shape: {a.shape}")


def _infer_hwc_shape(arr: np.ndarray) -> tuple[int, int, int]:
    """Return (h, w, d) without materializing a float32 copy."""
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ModelError(f"Unexpected embedding ndim={a.ndim}, shape={a.shape}")
    if a.shape[-1] in _EMBED_DIMS:
        return int(a.shape[0]), int(a.shape[1]), int(a.shape[2])
    if a.shape[0] in _EMBED_DIMS:
        return int(a.shape[1]), int(a.shape[2]), int(a.shape[0])
    raise ModelError(f"Unexpected embedding shape: {a.shape}")


def _resample_tiles_to_common_grid(
    tile_rows: Iterable[tuple[int, float, float, np.ndarray, Any, Any]],
    grid: CommonGrid,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample every tile's pixels onto *grid* (nearest neighbour) and return CHW + meta.

    ``tile_rows`` are geotessera's ``(year, tile_lon, tile_lat, embedding,
    crs, transform)`` tuples. Tiles are consumed one at a time and only the
    output canvas is allocated, so a large ROI never materializes a full
    mosaic. Each tile is placed through its own CRS, which is what lets a ROI
    straddling a UTM zone boundary be served instead of rejected.
    """
    canvas: np.ndarray | None = None
    filled = np.zeros(grid.shape, dtype=bool)
    centers_by_crs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _year, _tlon, _tlat, emb, crs, transform in tile_rows:
        h, w, d = _infer_hwc_shape(emb)
        if canvas is None:
            canvas = np.zeros(grid.shape + (d,), dtype=np.float32)
        elif d != canvas.shape[-1]:
            raise ModelError("Tiles have different embedding dimensions; cannot combine.")
        label = crs_label(crs)
        if label not in centers_by_crs:
            centers_by_crs[label] = project_pixel_centers(grid, crs)
        rows, cols, inside = source_pixel_indices(transform, *centers_by_crs[label], (h, w))
        inside &= ~filled  # first tile wins where tiles overlap
        if not inside.any():
            continue
        canvas[inside] = _gather_hwc(emb, rows[inside], cols[inside])
        filled |= inside

    if canvas is None:
        raise ModelError("No tiles fetched; cannot build the embedding grid.")
    if not filled.any():
        raise ModelError("ROI does not overlap fetched tessera tiles.")

    chw = np.moveaxis(canvas, -1, 0)
    meta = {
        "input_crs": "EPSG:4326",
        "projection_mode": "common_grid",
        "resampling": "nearest",
        "tile_crs": sorted(centers_by_crs),
        "coverage": float(filled.mean()),
        **grid.meta(),
    }
    return chw, meta


@register("tessera")
class TesseraEmbedder(EmbedderBase):
    _is_precomputed = True
    DEFAULT_BATCH_WORKERS = 4

    # Explicit pipeline-routing capabilities; the contract test asserts these
    # match the actual method signatures (tests/test_capabilities_contract.py).
    capabilities = EmbedderCapabilities(
        batch_fetch_metas=True,
        model_config_single=True,
        model_config_batch=True,
        model_config_batch_inputs=True,
    )

    def describe(self) -> dict[str, Any]:
        return {
            "type": "precomputed",
            "backend": ["auto"],
            "inputs": {"spatial": "BBox or PointBuffer (EPSG:4326)"},
            "temporal": {"mode": "year_or_range", "default_year": 2021},
            "output": ["pooled", "grid"],
            "source": "geotessera.GeoTessera",
            "defaults": {
                "cache_dir_env": "RS_EMBED_TESSERA_CACHE",
                "scale_m": _TESSERA_SCALE_M,
                "output_crs": COMMON_CRS,
            },
            "model_config": {
                "cache_dir": {
                    "type": "string",
                    "default": None,
                    "description": (
                        "GeoTessera tile cache directory. Precedence: model_config > "
                        "legacy sensor.collection='cache:<dir>' (deprecated) > "
                        "RS_EMBED_TESSERA_CACHE > geotessera default cache."
                    ),
                },
            },
            "notes": [
                "Precomputed GeoTessera tiles use a fixed source path; use backend='auto'.",
                "TemporalSpec.range uses the start year for tile lookup in v0.1.",
                f"Tiles (UTM) are resampled (nearest) onto the common {COMMON_CRS} grid at "
                f"{_TESSERA_SCALE_M} m, the same grid provider-backed models sample on.",
                "A ROI straddling a UTM zone boundary is served with a warning (seam).",
            ],
        }

    def __init__(self) -> None:
        super().__init__()
        # Cache GeoTessera instances per cache_dir to avoid repeated index scans.
        self._gt_cache: dict[str, Any] = {}

    def _get_gt(self, cache_dir: str):
        if cache_dir not in self._gt_cache:
            from geotessera import GeoTessera

            if cache_dir:
                self._gt_cache[cache_dir] = GeoTessera(cache_dir=cache_dir)
            else:
                self._gt_cache[cache_dir] = GeoTessera()
        return self._gt_cache[cache_dir]

    @staticmethod
    def _resolve_batch_workers(n_items: int) -> int:
        v = int(
            os.environ.get(
                "RS_EMBED_TESSERA_BATCH_WORKERS",
                str(TesseraEmbedder.DEFAULT_BATCH_WORKERS),
            )
        )
        return max(1, min(int(n_items), v))

    def get_embedding(
        self,
        *,
        spatial: SpatialSpec,
        temporal: TemporalSpec | None,
        sensor: SensorSpec | None,
        output: OutputSpec,
        backend: str,
        device: str = "auto",
        model_config: dict[str, Any] | None = None,
    ) -> Embedding:
        backend_n = str(backend).strip().lower()
        if backend_n == "local":
            backend_n = "auto"
        if backend_n != "auto":
            raise ModelError("tessera is precomputed; use backend='auto'.")

        grid = common_grid(spatial, scale_m=_TESSERA_SCALE_M)
        footprint = grid.bounds_4326()
        year = _year_from_temporal(temporal, default_year=2021)

        cache_dir = _resolve_tessera_cache_dir(model_config, sensor)

        # GeoTessera cache keyed by cache_dir path (empty string for default).
        cache_key = cache_dir or ""
        gt = self._get_gt(cache_key)

        bounds = (footprint.minlon, footprint.minlat, footprint.maxlon, footprint.maxlat)
        tiles = gt.registry.load_blocks_for_region(bounds=bounds, year=int(year))
        if not tiles:
            raise ModelError(f"No tessera tiles for bounds={bounds}, year={year}")

        chw, grid_meta = _resample_tiles_to_common_grid(gt.fetch_embeddings(tiles), grid)
        warn_if_utm_boundary(crss=grid_meta["tile_crs"], footprint=footprint, context="tessera")

        meta = build_meta(
            model=self.model_name,
            kind="precomputed",
            backend=backend_n,
            source="geotessera.GeoTessera",
            sensor=None,
            # Record the year actually served (temporal=None silently defaults),
            # not the raw request.
            temporal=TemporalSpec.year(int(year)),
            image_size=None,
            extra={
                "cache_dir": cache_dir,
                "bbox_4326": bounds,
                "preferred_year": year,
                "chw_shape": tuple(chw.shape),
                **grid_meta,
            },
        )

        if output.mode == "pooled":
            vec = _pool(chw, output.pooling)
            meta["pooling"] = f"{output.pooling}_hw"
            return Embedding(data=vec, meta=meta)

        if output.mode == "grid":
            try:
                import xarray as xr
            except Exception as e:
                raise ModelError("grid output requires xarray: pip install xarray") from e

            da = xr.DataArray(
                chw,
                dims=("d", "y", "x"),
                coords={
                    "d": np.arange(chw.shape[0]),
                    "y": np.arange(chw.shape[1]),
                    "x": np.arange(chw.shape[2]),
                },
                name="embedding",
                attrs=meta,
            )
            return Embedding(data=da, meta=meta)

        raise ModelError(f"Unknown output mode: {output.mode}")

    def get_embeddings_batch(
        self,
        *,
        spatials: list[SpatialSpec],
        temporal: TemporalSpec | None = None,
        sensor: SensorSpec | None = None,
        model_config: dict[str, Any] | None = None,
        output: OutputSpec = OutputSpec.pooled(),
        backend: str = "auto",
        device: str = "auto",
    ) -> list[Embedding]:
        if model_config is not None:
            # Same contract as the base batch path: an unsupported
            # model_config raises ModelError instead of TypeError/silence.
            self._require_model_config_support(model_config)
        if not spatials:
            return []

        n = len(spatials)
        out: list[Embedding | None] = [None] * n

        def _one(i: int, sp: SpatialSpec) -> tuple[int, Embedding]:
            emb = self.get_embedding(
                spatial=sp,
                temporal=temporal,
                sensor=sensor,
                output=output,
                backend=backend,
                device=device,
                model_config=model_config,
            )
            return i, emb

        mw = self._resolve_batch_workers(n)
        if mw == 1:
            for i, sp in enumerate(spatials):
                _, emb = _one(i, sp)
                out[i] = emb
        else:
            with ThreadPoolExecutor(max_workers=mw) as ex:
                futs = [ex.submit(_one, i, sp) for i, sp in enumerate(spatials)]
                for fut in as_completed(futs):
                    i, emb = fut.result()
                    out[i] = emb

        if any(e is None for e in out):
            raise ModelError("tessera batch failed to produce all outputs.")
        return [e for e in out if e is not None]
