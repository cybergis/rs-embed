## Quick Start

This page is for getting a first successful run quickly.

- New to `rs-embed`: follow this page top-to-bottom
- Already installed: jump to [Run `examples/quickstart.py`](#run-examplesquickstartpy)
- Need semantics first (`TemporalSpec`, `OutputSpec`): read [Core Concepts](concepts.md)
- Need task-oriented patterns after setup: read [Common Workflows](workflows.md)

Canonical model IDs now use short names (for example `remoteclip`, `prithvi`). Legacy IDs such as `remoteclip_s2rgb` still work as aliases.

This page teaches the **recommended** entry points first (`get_embedding`, `get_embeddings_batch`, `export_batch`, `inspect_provider_patch`).
Compatibility wrappers such as `export_npz(...)` and `inspect_gee_patch(...)` are documented later for older code and convenience.

---

## Install (temporary)

```bash
git clone https://github.com/Dinghye/rs-embed.git
# or: git clone git@github.com:Dinghye/rs-embed.git
cd rs-embed
conda env create -f environment.yml
conda activate rsembed
pip install -e .
```

For on-the-fly model demos (GEE + torch wrappers), install optional dependencies if needed:

```bash
pip install -e ".[gee,torch,models]"
```

Examples notebook: `examples/playground.ipynb`


## Authenticate Google Earth Engine

If you are using GEE for the first time, complete the authentication process with the following command.

```bash
earthengine authenticate
```

## Run `examples/quickstart.py`

You can run the packaged quickstart script directly:

```bash
# show CLI options
python examples/quickstart.py --help
```

### Auto mode (default, precomputed)

Runs `tessera` examples for:
- single embedding (`pooled` + `grid`)
- batch embeddings (`get_embeddings_batch`)
- optional export (`export_batch`)

```bash
python examples/quickstart.py --mode auto
python examples/quickstart.py --mode auto --run-export
```

### GEE mode (on-the-fly)

Runs `remoteclip` examples for:
- `inspect_provider_patch` (recommended; `inspect_gee_patch` remains as a backward-compatible alias)
- single embedding
- batch embeddings
- optional export

```bash
python examples/quickstart.py --mode gee --device auto
python examples/quickstart.py --mode gee --run-export --out-dir examples/_outputs/quickstart
```

### Run all demos

```bash
python examples/quickstart.py --mode all
python examples/quickstart.py --mode all --run-export
```

!!! tip
    If you see `ModuleNotFoundError: No module named 'rs_embed'`, run from repository root after installation:
    `pip install -e .`


### 1. Compute a single embedding

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(pooling="mean"),
    backend="gee",
    device="auto",
)

vec = emb.data  # shape: (D,)
meta = emb.meta
```

!!! note
    `TemporalSpec.range(start, end)` is treated as a temporal window (half-open: `[start, end)`).
    On GEE-backed on-the-fly paths, inputs are typically composites over that window (`median` by default), not an auto-selected single-day scene.

### 2. Batch compute embeddings for many points

```python
from rs_embed import PointBuffer, TemporalSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

embs = get_embeddings_batch(
    "remoteclip",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    backend="gee",
    device="auto",
)
```

### 3. Export at scale (recommended workflow)

`export_batch` is the **core** export API. It supports:

- arbitrary point / ROI lists
- multiple models per ROI
- saving inputs and embeddings
- manifests for downstream bookkeeping

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

export_batch(
    spatials=spatials,
    names=["p1", "p2"],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip", "prithvi"],
    out="exports",
    layout="per_item",
    backend="gee",
    device="auto",
    save_inputs=True,
    save_embeddings=True,
    chunk_size=32,
    num_workers=8,
    resume=True,
    show_progress=True,
)
```

`out + layout` is the recommended output-target style for new code.
Legacy `out_dir` / `out_path` arguments remain supported for backward compatibility.

## Working with Providers / Backends

rs-embed supports pluggable backends. In most setups:

- `backend="auto"` is the recommended default.
- `backend="gee"` is an explicit provider override for on-the-fly workflows.

If the behavior of a model input looks wrong, inspect the raw patch first:

- [`inspect_provider_patch`](api_inspect.md#inspect_provider_patch-recommended) (recommended, provider-agnostic)
- [`inspect_gee_patch`](api_inspect.md#inspect_gee_patch) (backward-compatible alias)

!!! tip
    For large exports, tune:
    - `chunk_size`: how many ROIs per chunk (controls memory peak)
    - `num_workers`: how many concurrent fetch workers (controls IO parallelism)
    - `resume=True`: skip files already exported in previous runs


## Export Formats

`export_batch(format=...)` is designed to be extensible.

- Current formats: `npz`, `netcdf`
- Planned: parquet / zarr / hdf5 (depending on your roadmap)

For a single ROI `.npz`, you can still use `export_batch(...)` directly:

```python
from rs_embed import export_batch, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip"],
    out="exports/one_point",
    layout="combined",  # writes exports/one_point.npz
)
```

`export_npz(...)` remains available as a convenience wrapper for single-ROI exports and shares the same performance optimizations.

!!! tip
    If you are building a repeatable dataset pipeline (many points and/or many models), prefer `export_batch(...)` and treat `export_npz(...)` as an optional convenience alias.
    See [Common Workflows](workflows.md) for the task-first export pattern.


## Performance Notes


### 1. Avoid repeated input downloads
When you use:

- `backend="gee"`
- `save_inputs=True`
- `save_embeddings=True`

`export_batch` will **prefetch each input patch once** and reuse it for both:
- saving the input patch
- computing embeddings (via `input_chw`)

### 2. IO parallelism vs inference safety
`export_batch` currently uses two-level scheduling:
- **IO level**: remote patch prefetch is parallelized (`num_workers`).
- **Inference level**:
  - model-to-model execution is serial (stability-first default),
  - but each model can use batched inference over many points when batch APIs are available (such as `get_embeddings_batch` / `get_embeddings_batch_from_inputs`): in combined mode by default, and in per-item mode when running on GPU/accelerators.

So rs-embed supports batch-level inference acceleration, while model-level scheduling remains serial by design.
