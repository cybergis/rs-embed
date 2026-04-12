# Quickstart

This page is the shortest path from installation to a first successful run.
It focuses on the three core APIs most users need: `get_embedding(...)`, `get_embeddings_batch(...)`, and `export_batch(...)`.

Use this page top-to-bottom once. After that, go to [Models](models.md) to choose model IDs, [API](api.md) for exact signatures and edge cases, and [Extending](extending.md) only if you want to add a model.

Canonical model IDs use short names such as `tessera`, `remoteclip`, and `prithvi`.
Legacy aliases such as `remoteclip_s2rgb` still work, but new code should use the short names.

---

## Install

```bash
# base install
pip install rs-embed

# add [terratorch] only if you use terramind
pip install "rs-embed[terratorch]"
```

For local development:

```bash
git clone https://github.com/cybergis/rs-embed.git
cd rs-embed
pip install -e .  # use -e ".[terratorch]" if you need terramind
```

Repository examples are available in `examples/playground.ipynb` and `examples/quickstart.py`.

If this is your first time using Google Earth Engine, authenticate once:

```bash
earthengine authenticate
```

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

vec = emb.data   # numpy ndarray, shape (D,) — e.g. (128,) for tessera
meta = emb.meta  # dict with model-specific metadata (normalization, grid_hw, etc.)
```

`emb.data` is always a `numpy.ndarray`. For `OutputSpec.pooled()` the shape is `(D,)` where `D` depends on the model (see the `Dim` column in [Models](models.md)). For `OutputSpec.grid()` the shape is `(D, H, W)` in model token space.

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

## Repository Examples

The repository also includes optional runnable examples:

- `examples/quickstart.py` for a small end-to-end script
- `examples/playground.ipynb` for notebook-style exploration

These are examples, not the primary usage path. For normal library use, start from the API snippets above.

---

## When You Need GEE or Model Debugging

Use GEE-backed on-the-fly models when you want to fetch imagery and run model inference directly.
Typical examples include `remoteclip`, `prithvi`, `terrafm`, and `terramind`.

If the model output looks wrong, inspect the fetched patch before changing model settings:

[`inspect_provider_patch(...)`](api_inspect.md#inspect_provider_patch) is the recommended inspection API, while [`inspect_gee_patch(...)`](api_inspect.md#inspect_gee_patch) remains available as a compatibility alias.

---

## What To Read Next

From here, use [Before You Start](choosing_settings.md) before you start changing `input_prep`, model size, patch size, frame count, or fetch resolution for speed or quality tradeoffs. Then use [Models](models.md) to choose the right model and verify its input assumptions, [API](api.md) for exact signatures and edge cases, [Concepts](concepts.md) for temporal and output semantics, and [Workflows](workflows.md) for extra recipes around tiling, inspection, and fair comparison.
