# API: Inspect

This page documents raw input inspection utilities (patch checks) used before model inference.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Export](api_export.md)

---

## inspect_provider_patch (recommended) { #inspect_provider_patch }

```python
inspect_provider_patch(
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec,
    backend: str = "gee",
    name: str = "gee_patch",
    value_range: tuple[float, float] | None = None,
    return_array: bool = False,
) -> dict[str, Any]
```

Provider-agnostic patch inspection utility (recommended entry point).
Use this when you want the same inspection flow but with a non-GEE provider backend.

This does **not** run any embedding model. It downloads a raw patch from the
provider and returns an input inspection report.

**Parameters**

- `spatial`: `BBox` or `PointBuffer` defining the area of interest
- `temporal`: `TemporalSpec` or `None` — temporal filter for the image query
- `sensor`: `SensorSpec` — sensor/collection configuration (required)
- `backend`: provider backend name (default `"gee"`)
- `name`: label used in the report (default `"gee_patch"`)
- `value_range`: optional `(lo, hi)` tuple — if set, the report flags pixels outside this range
- `return_array`: if `True`, the returned dict includes an `array_chw` entry with the raw numpy array

**Returns**

- `dict[str, Any]` — JSON-serializable report with keys:
    - `ok`: overall pass/fail boolean
    - `report`: detailed inspection (shape, band stats, quantiles, histograms, issues)
    - `sensor`: the sensor spec used (as dict)
    - `temporal`: the temporal spec used (as dict or `None`)
    - `backend`: the backend name
    - `artifacts`: optional quicklook paths (if `check_save_dir` is set on the sensor)
    - `array_chw` (only when `return_array=True`): the raw CHW numpy array

**Example**

```python
from rs_embed import PointBuffer, TemporalSpec, SensorSpec
from rs_embed.inspect import inspect_provider_patch

report = inspect_provider_patch(
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
    ),
    value_range=(0, 10000),
)
print(report["ok"])  # True if no issues detected
```

---
