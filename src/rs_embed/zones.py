"""Per-zone embeddings: aggregate pixel embeddings inside polygons.

``get_embedding`` answers "what does this bbox look like from space". A great deal of
analysis instead asks it per *administrative or management unit* — a census tract, county,
field, watershed, catchment — where the unit is an arbitrary polygon and the answer must
be one vector per unit, ready to join to tabular labels and hand to a downstream model.

:func:`embed_zones` does that: it sweeps the polygons' extent in tiles, embeds each tile as
a grid, works out which zone every pixel falls in, and accumulates per-zone statistics.

Two design points worth knowing before relying on the numbers.

**The pixel grid is EPSG:3857.** A grid comes back from the provider as a bare ``(D, H, W)``
array with no transform attached, and ``scale_m`` is measured in Web Mercator metres — which
run ``1/cos(latitude)`` longer than metres on the ground. Tiles are therefore requested on a
grid snapped to whole multiples of ``scale_m`` in EPSG:3857, the affine is derived from the
requested bounds and the returned shape, and the implied pixel size is checked against
``scale_m`` rather than assumed. ``pixel_ground_m`` in the result reports what a pixel
actually covers, so nobody reads "10 m" and means 10 m.

**Zones carry sums and counts, not only means.** A mean cannot be re-aggregated: averaging
the means of unequal zones is not the mean of their union. Keeping ``sum`` and ``pixels``
makes any coarser partition exact — tracts to a county, parcels to a tract — and ``pixels``
is also the honest measure of a zone's support, which is what tells you whether a model
fitted on small zones is extrapolating when applied to a large one.

Reading polygons and rasterizing them needs geopandas and rasterio, which are not part of
the base install::

    pip install "rs-embed[zones]"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from .core.errors import SpecError
from .core.specs import BBox, OutputSpec, TemporalSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd

__all__ = ["ZoneEmbedding", "ZoneEmbeddings", "embed_zones"]

# EPSG:3857 uses a sphere of this radius; the provider's grid is defined on it.
_R_MERCATOR = 6378137.0
_STATISTICS = ("mean", "sum")


@dataclass
class ZoneEmbedding:
    """One polygon's aggregated embedding.

    Attributes
    ----------
    zone_id : str
        Identifier taken from ``zone_id_field``, or the row index as a string.
    pixels : int
        Finite pixels found inside the polygon. Zero means no tile covered it, which is
        not the same as an empty embedding — check this before using ``mean``.
    area_km2 : float
        Polygon area measured in an equal-area projection, independent of the pixel grid.
    total : np.ndarray or None
        Per-band sum over the pixels inside the polygon. ``None`` when ``pixels == 0``.
    mean : np.ndarray or None
        ``total / pixels``. ``None`` when ``pixels == 0``.
    """

    zone_id: str
    pixels: int
    area_km2: float
    total: np.ndarray | None = None
    mean: np.ndarray | None = None


@dataclass
class ZoneEmbeddings:
    """Result of :func:`embed_zones`.

    Attributes
    ----------
    zones : list[ZoneEmbedding]
        One entry per input polygon, in input order.
    meta : dict[str, Any]
        Model, dimensionality, ``scale_m`` and ``pixel_ground_m``, tile accounting, and any
        per-tile failures or pixel-size warnings.
    """

    zones: list[ZoneEmbedding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def covered(self) -> list[ZoneEmbedding]:
        """Zones that actually received pixels."""
        return [z for z in self.zones if z.pixels]

    def to_frame(self):
        """Per-zone vectors as a pandas DataFrame: ``zone_id, pixels, area_km2, e000…``.

        The shape a downstream model consumes: one row per zone, one column per band.
        """
        import pandas as pd

        rows = []
        for z in self.covered:
            row: dict[str, Any] = {"zone_id": z.zone_id, "pixels": z.pixels,
                                   "area_km2": z.area_km2}
            row.update({f"e{i:03d}": float(v) for i, v in enumerate(z.mean)})
            rows.append(row)
        return pd.DataFrame(rows)

    def rollup(self, groups: dict[str, str]) -> "ZoneEmbeddings":
        """Combine zones into coarser units, exactly.

        Parameters
        ----------
        groups : dict[str, str]
            ``{zone_id: parent_id}``. Zones absent from the mapping are dropped.

        Returns
        -------
        ZoneEmbeddings
            One entry per parent, its ``total`` the sum of the children's totals and its
            ``mean`` that sum over the summed pixel count — which is why the sums are kept.
        """
        acc: dict[str, dict[str, Any]] = {}
        for z in self.covered:
            parent = groups.get(z.zone_id)
            if parent is None:
                continue
            cur = acc.setdefault(parent, {"total": np.zeros_like(z.total), "pixels": 0,
                                          "area_km2": 0.0})
            cur["total"] = cur["total"] + z.total
            cur["pixels"] += z.pixels
            cur["area_km2"] += z.area_km2
        out = [ZoneEmbedding(zone_id=k, pixels=v["pixels"], area_km2=round(v["area_km2"], 6),
                             total=v["total"], mean=v["total"] / v["pixels"])
               for k, v in acc.items()]
        return ZoneEmbeddings(zones=out, meta={**self.meta, "rolled_up_from": len(self.covered)})


def _to_lonlat(x: float, y: float) -> tuple[float, float]:
    """EPSG:3857 metres -> lon/lat degrees."""
    return (math.degrees(x / _R_MERCATOR),
            math.degrees(2 * math.atan(math.exp(y / _R_MERCATOR)) - math.pi / 2))


def _read_zones(zones: Any, zone_id_field: str | None) -> tuple["gpd.GeoDataFrame", str | None]:
    """Accept a path, a GeoDataFrame, or an iterable of ``(id, geometry)``.

    Returns the frame and the id column to use — an iterable of pairs brings its own,
    which the caller must see or the ids it supplied are silently replaced by row
    numbers.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise SpecError(
            "embed_zones needs geopandas and rasterio to read and rasterize polygons. "
            'Install them with: pip install "rs-embed[zones]"') from exc

    if isinstance(zones, gpd.GeoDataFrame):
        gdf = zones.copy()
    elif isinstance(zones, (str, bytes)) or hasattr(zones, "__fspath__"):
        gdf = gpd.read_file(zones)
    elif isinstance(zones, Iterable):
        pairs = list(zones)
        if not pairs:
            raise SpecError("zones is empty")
        gdf = gpd.GeoDataFrame({"zone_id": [str(i) for i, _g in pairs]},
                               geometry=[g for _i, g in pairs], crs="EPSG:4326")
        zone_id_field = "zone_id"
    else:
        raise SpecError(f"zones must be a path, a GeoDataFrame or an iterable of "
                        f"(id, geometry); got {type(zones).__name__}")
    if gdf.empty:
        raise SpecError("zones contains no features")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if zone_id_field is not None and zone_id_field not in gdf.columns:
        # A silent fall back to row numbers is worse than an error: the vectors key on
        # 0..n while whatever the caller joins them to expects the real identifier, and
        # the two never meet.
        raise SpecError(
            f"zone_id_field {zone_id_field!r} is not a column in the polygons; "
            f"available: {[c for c in gdf.columns if c != 'geometry'][:20]}")
    return gdf, zone_id_field


def embed_zones(
    model: str,
    *,
    zones: Any,
    temporal: TemporalSpec | None = None,
    zone_id_field: str | None = None,
    tile_px: int = 256,
    max_tiles: int | None = None,
    equal_area_crs: str = "EPSG:6933",
    backend: str = "auto",
    **kwargs: Any,
) -> ZoneEmbeddings:
    """Embed every polygon in ``zones`` by aggregating the pixels inside it.

    Parameters
    ----------
    model : str
        Model id, as accepted by :func:`~rs_embed.api.get_embedding`.
    zones : path or GeoDataFrame or iterable of (id, geometry)
        The polygons. A path is read with geopandas, so any format it supports works
        (GeoJSON, shapefile, GeoPackage, GeoParquet).
    temporal : TemporalSpec, optional
        Passed through per tile. Required by most models; precomputed annual products
        accept a year or a range.
    zone_id_field : str, optional
        Column holding each polygon's identifier. Omit to key zones by row index.
        A name that is not a column raises rather than falling back.
    tile_px : int
        Tile side in pixels. The sweep requests tiles of ``tile_px * scale_m`` metres in
        EPSG:3857; larger tiles mean fewer provider calls but a bigger array per call, and
        the provider caps how many pixels one request may return.
    max_tiles : int, optional
        Stop after this many tiles. Zones outside the tiles fetched come back with
        ``pixels == 0``, and ``meta["tiles_capped"]`` says the sweep was cut short.
    equal_area_crs : str
        Projection used for ``area_km2``. The default is global; a regional equal-area CRS
        is more accurate over a small study area.
    backend : str
        Passed through to :func:`~rs_embed.api.get_embedding`.
    **kwargs
        Forwarded verbatim to :func:`~rs_embed.api.get_embedding` (``sensor``, ``fetch``, …).

    Returns
    -------
    ZoneEmbeddings
        Per-zone sums, means, pixel counts and areas, plus provenance in ``meta``.

    Raises
    ------
    SpecError
        If ``zones`` is unusable, ``zone_id_field`` is not a column, or ``tile_px`` is not
        a positive integer.

    Examples
    --------
    >>> from rs_embed import TemporalSpec, embed_zones
    >>> zed = embed_zones(                                    # doctest: +SKIP
    ...     "gse", zones="tracts.geojson", zone_id_field="geoid",
    ...     temporal=TemporalSpec.year(2022))
    >>> zed.to_frame().head()                                 # doctest: +SKIP
    """
    from rasterio.features import rasterize  # noqa: PLC0415 - optional dependency
    from affine import Affine  # noqa: PLC0415

    from .api import get_embedding  # noqa: PLC0415 - avoids an import cycle

    if int(tile_px) <= 0:
        raise SpecError(f"tile_px must be a positive integer; got {tile_px!r}")
    tile_px = int(tile_px)

    gdf, zone_id_field = _read_zones(zones, zone_id_field)
    zone_ids = ([str(v) for v in gdf[zone_id_field]] if zone_id_field
                else [str(i) for i in range(len(gdf))])
    areas = (gdf.to_crs(equal_area_crs).area / 1e6).tolist()

    merc = gdf.to_crs("EPSG:3857")
    # Burn value 0 means "no zone", so zones are numbered from 1.
    shapes = [(g, i + 1) for i, g in enumerate(merc.geometry)
              if g is not None and not g.is_empty]
    if not shapes:
        raise SpecError("zones contains no usable geometry")

    def _tile(x0: float, y0: float, x1: float, y1: float) -> tuple[np.ndarray, dict[str, Any]]:
        lon0, lat0 = _to_lonlat(x0, y0)
        lon1, lat1 = _to_lonlat(x1, y1)
        emb = get_embedding(model, spatial=BBox(minlon=lon0, minlat=lat0,
                                                maxlon=lon1, maxlat=lat1),
                            temporal=temporal, output=OutputSpec.grid(),
                            backend=backend, **kwargs)
        data = getattr(emb.data, "values", emb.data)
        return np.asarray(data, dtype=np.float32), (emb.meta or {})

    minx, miny, maxx, maxy = merc.total_bounds
    # One probe first: scale_m and the band count come from the provider, and the sweep
    # geometry depends on scale_m.
    probe_arr, probe_meta = _tile(minx, miny,
                                 min(minx + 1000.0, maxx), min(miny + 1000.0, maxy))
    scale = float(probe_meta.get("scale_m") or 10)
    dims = int(probe_arr.shape[0])
    step = tile_px * scale

    # Snap the sweep origin to the scale grid so every tile spans an exact multiple of
    # scale_m; the returned shape then matches the request and the affine is unambiguous.
    ox = math.floor(minx / scale) * scale
    oy = math.floor(miny / scale) * scale
    nx = int(math.ceil((maxx - ox) / step))
    ny = int(math.ceil((maxy - oy) / step))

    sums = np.zeros((len(gdf) + 1, dims), dtype=np.float64)
    counts = np.zeros(len(gdf) + 1, dtype=np.int64)
    fetched = 0
    tile_errors: list[dict[str, Any]] = []
    pixel_size_warnings: list[str] = []
    sindex = merc.sindex

    from shapely.geometry import box  # noqa: PLC0415

    for ty in range(ny):
        for tx in range(nx):
            if max_tiles is not None and fetched >= int(max_tiles):
                break
            x0, y0 = ox + tx * step, oy + ty * step
            x1, y1 = x0 + step, y0 + step
            if x0 > maxx or y0 > maxy:
                continue
            # Most of a bounding box is usually empty; skip tiles no zone touches.
            if not sindex.query(box(x0, y0, x1, y1), predicate="intersects").size:
                continue
            try:
                arr, _meta = _tile(x0, y0, x1, y1)
            except Exception as exc:  # noqa: BLE001 - one bad tile must not lose the sweep
                tile_errors.append({"tile": [tx, ty], "error": f"{type(exc).__name__}: {exc}"})
                continue
            _d, h, w = arr.shape
            px, py = (x1 - x0) / w, (y1 - y0) / h
            if abs(px - scale) / scale > 0.02 or abs(py - scale) / scale > 0.02:
                pixel_size_warnings.append(
                    f"tile {tx},{ty}: implied pixel {px:.2f}x{py:.2f} m vs scale_m {scale:.0f}")
            # North-up: rows run from the tile's top edge downwards.
            zmap = rasterize(shapes, out_shape=(h, w),
                             transform=Affine(px, 0.0, x0, 0.0, -py, y1),
                             fill=0, all_touched=False, dtype="int32")
            present = np.unique(zmap)
            present = present[present > 0]
            if present.size:
                flat = arr.reshape(_d, h * w)
                zflat = zmap.reshape(h * w)
                for z in present:
                    sel = flat[:, zflat == z]
                    good = np.isfinite(sel).all(axis=0)
                    if good.any():
                        sums[z] += sel[:, good].sum(axis=1)
                        counts[z] += int(good.sum())
            fetched += 1
        if max_tiles is not None and fetched >= int(max_tiles):
            break

    out = []
    for i, zid in enumerate(zone_ids):
        n = int(counts[i + 1])
        total = sums[i + 1].copy() if n else None
        out.append(ZoneEmbedding(zone_id=zid, pixels=n, area_km2=round(float(areas[i]), 6),
                                 total=total, mean=(total / n) if n else None))

    mid_lat = _to_lonlat(0.0, (miny + maxy) / 2)[1]
    planned = nx * ny
    return ZoneEmbeddings(
        zones=out,
        meta={
            "model": model, "dims": dims, "bands": list(probe_meta.get("bands") or ())[:8],
            "scale_m": scale,
            # scale_m is Web Mercator metres; this is what a pixel covers on the ground.
            "pixel_ground_m": round(scale * math.cos(math.radians(mid_lat)), 3),
            "tile_px": tile_px, "tiles_planned": planned, "tiles_fetched": fetched,
            "tiles_capped": bool(max_tiles is not None and planned > int(max_tiles)),
            "zone_id_field": zone_id_field, "equal_area_crs": equal_area_crs,
            "zones_total": len(out), "zones_with_pixels": sum(1 for z in out if z.pixels),
            "tile_errors": tile_errors, "pixel_size_warnings": pixel_size_warnings,
        },
    )
