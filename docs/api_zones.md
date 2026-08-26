# API: Zones

One vector per polygon. `get_embedding(...)` answers "what does this bbox look like from space"; a great deal of analysis instead asks it per **administrative or management unit** — a census tract, county, field, watershed, catchment — where the unit is an arbitrary polygon and the answer must be one vector per unit, ready to join to tabular labels and hand to a downstream model.

Related pages: [API: Embedding](api_embedding.md), [API: Specs and Data Structures](api_specs.md), [Spatial ROI Handling](spatial_roi.md).

Requires the optional geo dependencies:

```bash
pip install "rs-embed[zones]"
```

---

## Quick Example

```python
from rs_embed import TemporalSpec, embed_zones

zed = embed_zones(
    "gse",
    zones="tracts.geojson",        # any format geopandas reads
    zone_id_field="geoid",
    temporal=TemporalSpec.year(2022),
)

zed.to_frame().head()              # zone_id, pixels, area_km2, e000 … e063
```

`zones` also accepts a `GeoDataFrame` or an iterable of `(id, geometry)` pairs.

---

## Data Structures

### ZoneEmbeddings

```python
ZoneEmbeddings(
    zones: list[ZoneEmbedding],
    meta:  dict[str, Any],
)
```

| Member | Meaning |
| --- | --- |
| `zones` | One entry per input polygon, in input order |
| `covered` | Only the zones that received pixels |
| `to_frame()` | Per-zone vectors as a DataFrame — the shape a model consumes |
| `rollup(groups)` | Combine zones into coarser units, exactly (see below) |

### ZoneEmbedding

| Field | Meaning |
| --- | --- |
| `zone_id` | From `zone_id_field`, or the row index as a string |
| `pixels` | Finite pixels found inside the polygon. **Zero means no tile covered it** — check before using `mean` |
| `area_km2` | Polygon area in an equal-area projection, independent of the pixel grid |
| `total` | Per-band sum over the pixels inside the polygon |
| `mean` | `total / pixels` |

### meta

Carries `model`, `dims`, `bands`, `scale_m`, `pixel_ground_m`, `tile_px`, `tiles_planned`, `tiles_fetched`, `tiles_capped`, `zones_total`, `zones_with_pixels`, `tile_errors` and `pixel_size_warnings`.

---

## Why sums and not just means

A mean cannot be re-aggregated: averaging the means of unequal zones is not the mean of their union. Keeping `total` alongside `pixels` makes any coarser partition exact.

```python
county = zed.rollup({z.zone_id: "cook" for z in zed.covered})
county.zones[0].mean          # exact, weighted by pixel count
```

`pixels` is also the honest measure of a zone's **support**. A model fitted on 3,000-pixel tracts is extrapolating when applied to a 400,000-pixel county, because pooling averages away variance in a size-dependent way — the same phenomenon as the change-of-support problem in spatial statistics. Report `pixels` rather than assuming comparability.

---

## Why the grid is EPSG:3857

A grid comes back from the provider as a bare `(D, H, W)` array with no transform attached, and `scale_m` is measured in **Web Mercator** metres, which run `1/cos(latitude)` longer than metres on the ground. A 0.02° × 0.01° bbox at 40.07°N returns a 224 × 146 grid, which matches the EPSG:3857 span divided by 10 m; reading `scale_m` as ground metres would predict 170 × 111.

`embed_zones` therefore requests tiles on a grid snapped to whole multiples of `scale_m` in EPSG:3857, derives the affine from the requested bounds and the returned shape, and **checks** the implied pixel size against `scale_m` — any disagreement over 2% is recorded in `meta["pixel_size_warnings"]` rather than silently shifting every zone boundary.

`meta["pixel_ground_m"]` reports what a pixel actually covers, so `scale_m = 10` is never mistaken for 10 m on the ground.

---

## Tiling and cost

The extent is swept in tiles of `tile_px * scale_m` metres and statistics accumulate as running sums, so no full cube is ever assembled. This matters at scale: 801 Chicago tracts span roughly 46 × 56 km, which at 10 m is 25.8 M pixels × 64 bands ≈ 6.6 GB, while the accumulator is one vector per zone.

- Larger `tile_px` means fewer provider calls but a bigger array per call, and the provider caps how many pixels one request may return.
- Tiles that no polygon touches are skipped — most of a bounding box is usually empty.
- `max_tiles` stops the sweep early; zones outside the tiles fetched come back with `pixels == 0` and `meta["tiles_capped"]` is `True`.
- A single failing tile is recorded in `meta["tile_errors"]` and the sweep continues.

---

## Functions

### embed_zones

```python
embed_zones(
    model: str,
    *,
    zones,                                  # path | GeoDataFrame | iterable of (id, geometry)
    temporal: TemporalSpec | None = None,
    zone_id_field: str | None = None,
    tile_px: int = 256,
    max_tiles: int | None = None,
    equal_area_crs: str = "EPSG:6933",
    backend: str = "auto",
    **kwargs,                               # forwarded to get_embedding
) -> ZoneEmbeddings
```

`zone_id_field` naming a column that does not exist raises `SpecError` listing the columns that do. It does **not** fall back to row numbers: the vectors would key on `0..n` while whatever they are joined to expects the real identifier, and the two would never meet.

Extra keyword arguments are forwarded verbatim to `get_embedding`, so `sensor`, `fetch` and the rest behave as documented on [API: Embedding](api_embedding.md).
