# API Reference

This section is the exact reference for the public API.
If you want installation and first-run examples, start with [Quickstart](quickstart.md) instead.

---

## Core Entry Points

Most users only need these public functions:

- `get_embedding(...)`
- `get_embeddings_batch(...)`
- `export_batch(...)`
- `inspect_provider_patch(...)`

---

## Choose by Task

| I want to... | Read this page |
|---|---|
| understand spatial/temporal/output specs | [API: Specs and Data Structures](api_specs.md) |
| get one embedding or batch embeddings | [API: Embedding](api_embedding.md) |
| build export pipelines and datasets | [API: Export](api_export.md) |
| inspect raw provider patches before inference | [API: Inspect](api_inspect.md) |

---
## Useful Extras

- `export_npz(...)`: compatibility wrapper around `export_batch(...)` for single-ROI `.npz`
- `list_models()`: stable public model catalog helper

Model-specific configuration:

- `get_embedding(...)` and `get_embeddings_batch(...)` accept `model_config`
- `export_batch(...)` supports per-model `model_config` via `ExportModelRequest(...)`
- currently documented model-level `model_config` usage includes `dofa`, `anysat`, `thor`, and `satmaepp_s2_10b`
- for the currently documented variant-aware models, use a unified field: `model_config={"variant": "..."}`
- valid `variant` values still depend on the selected model and currently exposed published checkpoints, so check the corresponding model detail page
- unsupported `model_config` usage raises `ModelError` instead of being ignored silently

If you need a stable model list in code:

```python
from rs_embed import list_models

print(list_models())
```

`rs_embed.core.registry.list_models()` only reports models currently loaded into the runtime registry.

---

## Errors

rs-embed raises several explicit exception types (all in `rs_embed.core.errors`):

- `SpecError`: spec validation failure (invalid bbox, missing temporal fields, etc.)
- `ProviderError`: provider/backend errors (e.g., GEE initialization or fetch failure)
- `ModelError`: unknown model ID, unsupported parameters, unsupported export format, etc.

---

## Versioning Notes

The current version is still early stage (`0.1.x`):

- `BBox/PointBuffer` currently require `crs="EPSG:4326"`
- Precomputed models should use `backend="auto"`; on-the-fly models mainly use provider backends (typically `"gee"` or explicit provider names)
- `ExportConfig(format=...)` is the recommended way to choose export format; supported values are currently `"npz"` and `"netcdf"` and may be extended to parquet/zarr/hdf5, etc.
