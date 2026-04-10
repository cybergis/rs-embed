# API: Export

This page covers dataset export APIs.

If you only remember one function, remember `export_batch(...)`.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Embedding](api_embedding.md)
- [API: Inspect](api_inspect.md)

---

## export_batch (primary / recommended) { #export_batch }

```python
export_batch(
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None,
    models: list[str | ExportModelRequest],
    target: ExportTarget | None = None,
    config: ExportConfig | None = None,
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: SensorSpec | None = None,
    modality: str | None = None,
) -> Any
```

Use `export_batch(...)` when you want to export:

- one or many ROIs
- one or many models
- inputs, embeddings, and manifests together

Although the public function still exposes many keyword arguments, the actual implementation first normalizes requests into:

- `ExportTarget`
- `ExportConfig`
- `ExportModelRequest` entries

That is the real shape of the API internally, and it is the shape new code should follow.

### Mental Model

Think about `export_batch(...)` as 4 decisions:

1. What to export: `spatials`, `temporal`, `models`
2. Where to write: `target=ExportTarget(...)`
3. How to run: `config=ExportConfig(...)`
4. Any shared model settings: `backend`, `sensor`, `modality`, `output`

For new code, prefer the object-style API:

- `target=ExportTarget(...)`
- `config=ExportConfig(...)`
- `models=[..., ExportModelRequest(...)]` only when one model needs special overrides

### Start Here

If you are not sure what to pass, this is the default pattern:

```python
from rs_embed import export_batch, ExportConfig, ExportTarget, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip"],
    target=ExportTarget.combined("exports/run"),
    config=ExportConfig(),
)
```

That gives you:

- one combined export artifact
- default `.npz` format
- inputs + embeddings + manifest
- default runtime behavior

---

## Parameters, Grouped by Job

### 1. Required dataset definition

These are the inputs most users always set:

- `spatials`: non-empty list of `BBox` or `PointBuffer`
- `temporal`: `TemporalSpec` or `None`
- `models`: non-empty list of model IDs or `ExportModelRequest(...)`

### 2. Output location and layout

Prefer `target=ExportTarget(...)` in new code.

```python
from rs_embed import ExportTarget

ExportTarget.per_item("exports", names=["p1", "p2"])
ExportTarget.combined("exports/run")
```

- `ExportTarget.per_item(...)`: one file per ROI
- `ExportTarget.combined(...)`: one merged file for the whole run

### 3. Shared model/runtime settings

These usually apply to all models in the call:

- `backend`: keep `backend="auto"` unless you need a specific provider such as `"gee"`
- `device`: `"auto"` is the normal choice
- `output`: usually `OutputSpec.pooled()`
- `sensor`: shared `SensorSpec` for on-the-fly models
- `modality`: shared modality override for models that expose multiple public branches

Use per-model overrides only when one model needs different settings.

### 4. ExportConfig: the knobs that matter most

`config=ExportConfig(...)` is the recommended place for runtime settings.

In the implementation, flat config keywords are folded into an `ExportConfig` object anyway, so new code should construct that object directly.

The most important ones are:

- `format`: `"npz"` or `"netcdf"`
- `save_inputs`: save model-ready input patches
- `save_embeddings`: save embedding arrays
- `save_manifest`: save JSON manifest metadata
- `resume`: skip items already exported
- `input_prep`: large-ROI policy, usually `"resize"` or `"tile"`

You can usually ignore the rest until you need performance tuning or failure recovery.

### 5. Advanced runtime controls

These matter mainly for larger runs:

- `chunk_size`: how many ROIs to process at a time
- `infer_batch_size`: batch size for models that implement batch inference
- `num_workers`: provider fetch concurrency
- `continue_on_error`: keep going if one item/model fails
- `max_retries`, `retry_backoff_s`: retry policy
- `async_write`, `writer_workers`: asynchronous writing in per-item mode
- `show_progress`: enable progress display
- `fail_on_bad_input`: fail immediately on invalid inputs

If you do not know what these mean, leave them at defaults.

Example:

```python
from rs_embed import ExportConfig

config = ExportConfig(
    format="npz",
    save_inputs=True,
    save_embeddings=True,
    save_manifest=True,
    resume=True,
    input_prep="resize",
)
```

---

## Per-Model Overrides

Most runs should pass plain model IDs:

```python
models=["remoteclip", "prithvi"]
```

Use `ExportModelRequest(...)` only when a specific model needs its own sensor or modality:

```python
from rs_embed import ExportModelRequest

models=[
    "remoteclip",
    ExportModelRequest("terrafm", modality="s1"),
]
```

`ExportModelRequest(...)` also carries per-model `model_config`, for example:

```python
from rs_embed import ExportModelRequest

models=[
    "remoteclip",
    ExportModelRequest("thor", model_config={"variant": "large"}),
]
```

Typical use cases:

- one model needs `modality="s1"`
- one model needs a different `SensorSpec`
- one model needs a different `model_config` such as `{"variant": "large"}`
- one model should override the shared export settings

This also matches the implementation path: string model IDs are first converted into `ExportModelRequest(name=...)`, then resolved.

Modality rules:

- `export_batch(...)` accepts a global `modality`
- one model can override it via `ExportModelRequest(...)`
- unsupported modality choices raise `ModelError`

`model_config` rules:

- `export_batch(...)` does not have one global `model_config` shared across all models
- pass per-model runtime settings through `ExportModelRequest(..., model_config=...)`
- unsupported `model_config` usage raises `ModelError`

---

## What Gets Returned

- `ExportTarget.per_item(...)`: returns `list[dict]`
- `ExportTarget.combined(...)`: returns `dict`

In both cases, the return value is manifest-style metadata describing what was exported.

---

## Common Patterns

### One combined export file

```python
from rs_embed import export_batch, ExportConfig, ExportTarget, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip"],
    target=ExportTarget.combined("exports/combined_run"),
    config=ExportConfig(save_inputs=True, resume=True),
)
```

### One file per ROI

```python
from rs_embed import export_batch, ExportConfig, ExportTarget, PointBuffer, TemporalSpec

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

export_batch(
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip", "prithvi"],
    target=ExportTarget.per_item("exports", names=["p1", "p2"]),
    config=ExportConfig(
        input_prep="tile",
        chunk_size=32,
        num_workers=8,
    ),
)
```

### One model needs its own modality

```python
from rs_embed import (
    export_batch,
    ExportModelRequest,
    ExportTarget,
    PointBuffer,
    TemporalSpec,
)

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=[ExportModelRequest("terrafm", modality="s1")],
    target=ExportTarget.combined("exports/terrafm_s1_run"),
    backend="gee",
)
```

### One model needs its own variant

```python
from rs_embed import (
    export_batch,
    ExportModelRequest,
    ExportTarget,
    PointBuffer,
    TemporalSpec,
)

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=[ExportModelRequest("thor", model_config={"variant": "large"})],
    target=ExportTarget.combined("exports/thor_large_run"),
    backend="gee",
)
```

---

## Runtime Behavior You Usually Need to Know

### Inference scheduling

- model scheduling is serial: one model at a time
- batch inference is used when the embedder supports it
- GPU or accelerator backends benefit the most from batch inference

### Per-item vs combined mode

- `per_item` mode writes one artifact per ROI
- `combined` mode writes one merged artifact for the run
- combined mode keeps the older behavior of preferring batch model APIs when possible

### Input reuse

If provider-backed export is used and both `save_inputs=True` and `save_embeddings=True`, rs-embed reuses the fetched input patch for both writing and embedding inference instead of downloading it twice.

!!! tip "Simple rule"
    Start with `ExportTarget.combined(...)` + `ExportConfig()`.
    Add `ExportModelRequest(...)` only for the few models that need per-model sensor, modality, or `model_config` overrides.

---
