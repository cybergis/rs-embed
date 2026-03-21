# Quickstart

This page is the shortest path from installation to a first successful run.
It focuses on the three core APIs most users need:

- `get_embedding(...)`
- `get_embeddings_batch(...)`
- `export_batch(...)`

Use this page top-to-bottom once. After that:

- go to [Models](models.md) to choose model IDs
- go to [API](api.md) for exact signatures and edge cases
- go to [Extending](extending.md) if you want to add a model

Canonical model IDs use short names such as `tessera`, `remoteclip`, and `prithvi`.
Legacy aliases such as `remoteclip_s2rgb` still work, but new code should use the short names.

---

## Install



For local development from the repository:

```bash
git clone https://github.com/cybergis/rs-embed.git
cd rs-embed
pip install -e ".[dev]"
```

Repository examples: `examples/playground.ipynb`, `examples/quickstart.py`



If this is your first time using Google Earth Engine, authenticate once:

```bash
earthengine authenticate
```

---

## Run the Example Script

The repository example script is the fastest way to verify your environment after cloning this repo.

```bash
python examples/quickstart.py --help
```

### Precomputed path (`backend="auto"`)

This path uses `tessera` and does not require GEE authentication.

```bash
python examples/quickstart.py --mode auto
python examples/quickstart.py --mode auto --run-export
```

### On-the-fly path (`backend="gee"`)

Authenticate Earth Engine first if this is your first time:

```bash
earthengine authenticate
```

Then run the GEE demo:

```bash
python examples/quickstart.py --mode gee --device auto
python examples/quickstart.py --mode gee --run-export --out-dir examples/_outputs/quickstart
```

### Run both

```bash
python examples/quickstart.py --mode all
python examples/quickstart.py --mode all --run-export
```

!!! tip
    If you see `ModuleNotFoundError: No module named 'rs_embed'`, run `pip install -e .` from the repository root.

---

## Minimal Mental Model

Before looking at the APIs, keep these three ideas in mind:

- `spatial`: where to extract from, usually `PointBuffer(...)` or `BBox(...)`
- `temporal`: when to extract from, either `TemporalSpec.year(...)` or `TemporalSpec.range(...)`
- `output`: what shape you want, usually `OutputSpec.pooled()` first

Two details matter a lot:

- `TemporalSpec.range(start, end)` is usually a time window for filtering and compositing, not a guarantee of one exact acquisition date.
- `OutputSpec.grid()` is often a model patch/token grid, not always a georeferenced raster grid.

If you want the full semantics, read [API: Specs and Data Structures](api_specs.md).

---

## The Three Core APIs

### 1. One ROI: `get_embedding(...)`

Use this when you want one embedding now.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "tessera",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=1024),
    temporal=TemporalSpec.year(2024),
    output=OutputSpec.pooled(pooling="mean"),
    backend="auto",
)

vec = emb.data
meta = emb.meta
```

Use `backend="auto"` unless you need to force a provider path such as `backend="gee"`.

### 2. Many ROIs, one model: `get_embeddings_batch(...)`

Use this when the model is fixed and you have multiple ROIs.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 1024),
    PointBuffer(120.5, 30.2, 1024),
]

embs = get_embeddings_batch(
    "tessera",
    spatials=spatials,
    temporal=TemporalSpec.year(2024),
    output=OutputSpec.pooled(),
    backend="auto",
)
```

### 3. Dataset export: `export_batch(...)`

Use this when you are building a dataset or benchmark and want files plus manifests.

```python
from rs_embed import (
    ExportConfig,
    ExportTarget,
    PointBuffer,
    TemporalSpec,
    export_batch,
)

spatials = [
    PointBuffer(121.5, 31.2, 1024),
    PointBuffer(120.5, 30.2, 1024),
]

export_batch(
    spatials=spatials,
    temporal=TemporalSpec.year(2024),
    models=["tessera"],
    target=ExportTarget.per_item("exports", names=["p1", "p2"]),
    backend="auto",
    config=ExportConfig(
        save_inputs=False,
        save_embeddings=True,
        resume=True,
        show_progress=True,
    ),
)
```

For new code, `target=ExportTarget(...)` plus `config=ExportConfig(...)` is the recommended style.

---

## When You Need GEE or Model Debugging

Use GEE-backed on-the-fly models when you want to fetch imagery and run model inference directly.
Typical examples include `remoteclip`, `prithvi`, `terrafm`, and `terramind`.

If the model output looks wrong, inspect the fetched patch before changing model settings:

- [`inspect_provider_patch(...)`](api_inspect.md#inspect_provider_patch): recommended inspection API

---

## What To Read Next

- [Models](models.md): choose the right model and check its input assumptions
- [API](api.md): exact signatures for specs, embedding, export, and inspect
- [Concepts](concepts.md): deeper explanation of temporal and output semantics
- [Workflows](workflows.md): extra recipes for tiling, inspection, and fair comparison
