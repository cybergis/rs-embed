# API: Export

This page covers dataset export APIs.

Related pages: [API: Specs and Data Structures](api_specs.md), [API: Embedding](api_embedding.md), [API: Load](api_load.md), and [API: Inspect](api_inspect.md).

---

## export_batch (primary / recommended) { #export_batch }

### Signature

```python
export_batch(
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec],
    models: List[str | ExportModelRequest],
    target: ExportTarget,
    config: ExportConfig = ExportConfig(),
    backend: str = "auto",
    device: str = "auto",
    output: OutputSpec = OutputSpec.pooled(),
    sensor: Optional[SensorSpec] = None,
    fetch: Optional[FetchSpec] = None,
    modality: Optional[str] = None,
    input_prep: Optional[InputPrepSpec | str] = None,
) -> List[dict] | dict
```

Use `export_batch(...)` when you want to export one or many ROIs, one or many models, and the corresponding inputs, embeddings, and manifests together.

Although the public function still exposes many keyword arguments, the implementation first normalizes requests into `ExportTarget`, `ExportConfig`, and `ExportModelRequest` entries.

That is the real shape of the API internally, and it is the shape new code should follow.

### Mental Model

Think about `export_batch(...)` as 4 decisions:

1. What to export: `spatials`, `temporal`, `models`
2. Where to write: `target=ExportTarget(...)`
3. How to run: `config=ExportConfig(...)`
4. Any shared or per-model settings: `backend`, `device`, `output`, `fetch`, `sensor`, `modality`, `input_prep`

Prefer `target=ExportTarget(...)` and `config=ExportConfig(...)`, then use `models=[..., ExportModelRequest(...)]` when one model really needs special overrides.

### Default Pattern

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

That gives you one combined export artifact in the default `.npz` format, including inputs, embeddings, and a manifest, all with the default runtime behavior.

---

## Parameters, Grouped by Job

### 1. Required dataset definition

| Parameter  | Meaning                                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------------------- |
| `spatials` | Non-empty list of `BBox` or `PointBuffer`.                                                                              |
| `temporal` | `TemporalSpec` or `None`. The parameter is optional at the API level, but some models or data sources still require it. |
| `models`   | Non-empty list of model IDs or `ExportModelRequest(...)`.                                                               |

### 2. Output location and layout

Prefer `target=ExportTarget(...)` in new code.

```python
from rs_embed import ExportTarget

ExportTarget.per_item("exports", names=["p1", "p2"])
ExportTarget.combined("exports/run")
```

`ExportTarget.per_item(...)` writes one file per ROI, while `ExportTarget.combined(...)` writes one merged file for the whole run.

For `combined`, the output extension is normalized from `config.format` if missing. For `per_item`, `names=[...]` must have the same length as `spatials`.

### 3. Shared model/runtime settings

These usually apply to all models in the call:

| Setting    | Typical use                                                                |
| ---------- | -------------------------------------------------------------------------- |
| `backend`  | Keep `backend="auto"` unless you need a specific provider such as `"gee"`. |
| `device`   | `"auto"` is the normal choice.                                             |
| `output`   | Usually `OutputSpec.pooled()`.                                             |
| `fetch`    | Shared `FetchSpec` for resolution or compositing overrides.                |
| `sensor`   | Shared `SensorSpec` for advanced on-the-fly source overrides.              |
| `modality` | Shared modality override for models that expose multiple public branches.  |
| `input_prep` | Large-ROI policy, usually `"tile"` (the default) or `"resize"`. Same semantics as `get_embedding(input_prep=...)`. |

Use per-model overrides only when one model needs different settings.

#### Rule Of Thumb

Use `fetch=FetchSpec(...)` for shared resolution or compositing overrides. Use `sensor=SensorSpec(...)` only when a job really needs custom `collection` or `bands`. `fetch` and `sensor` cannot be passed together.

### 4. Per-model overrides

The single per-model channel is `ExportModelRequest(...)` or `ExportModelRequest.configure(...)` inside `models=[...]`.

Inline values on `ExportModelRequest(...)` take precedence over the corresponding global `sensor` / `fetch` / `modality`.

### 5. ExportConfig: the knobs that matter most

`config=ExportConfig(...)` is the recommended place for runtime settings.

The most important ones are:

| Option            | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| `format`          | `"npz"` or `"netcdf"`.                            |
| `save_inputs`     | Save model-ready input patches.                   |
| `save_embeddings` | Save embedding arrays.                            |
| `save_manifest`   | Save JSON manifest metadata.                      |
| `resume`          | Skip items already exported.                      |

You can usually ignore the rest until you need performance tuning or failure recovery.

`config` is optional; the default is `ExportConfig()`. `ExportConfig(input_prep=...)` is also accepted and is equivalent to the top-level `input_prep` parameter (recommended); passing both raises `ModelError`.

### 6. Advanced runtime controls

These matter mainly for larger runs. `chunk_size` controls how many ROIs are processed at a time, `infer_batch_size` controls model batch size when batching is supported, and `num_workers` controls provider fetch concurrency. The remaining knobs such as `continue_on_error`, retry settings, asynchronous writing, progress display, and `fail_on_bad_input` mostly matter when you are tuning larger or less reliable runs.

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
)
```

---

## Per-Model Overrides

### When To Use Them

Most runs should pass plain model IDs:

```python
models=["remoteclip", "prithvi"]
```

Use `ExportModelRequest(...)` when a specific model needs its own fetch, sensor, or modality:

```python
from rs_embed import ExportModelRequest, FetchSpec

models=[
    "remoteclip",
    ExportModelRequest("prithvi", fetch=FetchSpec(scale_m=30)),
]
```

`ExportModelRequest.configure(...)` also accepts model-specific settings as keyword arguments, for example:

```python
from rs_embed import ExportModelRequest

models=[
    "remoteclip",
    ExportModelRequest.configure("thor", variant="large"),
]
```

Typical use cases are when one model needs its own `FetchSpec`, `modality="s1"`, a different `SensorSpec`, a different variant such as `variant="large"`, or some override of the shared export settings.

This also matches the implementation path: string model IDs are first converted into `ExportModelRequest(name=...)`, then resolved.

### Rules

`export_batch(...)` accepts a global `modality`, one model can override it through `ExportModelRequest(...)`, and unsupported modality choices raise `ModelError`.

`export_batch(...)` does not have one global model-settings parameter shared across all models. Pass per-model settings through `ExportModelRequest.configure("model", variant=...)`. Unsupported keyword arguments raise `ModelError`.

For `sensor` / `fetch` / `modality`, the effective precedence is: inline `ExportModelRequest(...)` value first, then the corresponding global argument.

---

## What Gets Returned

### Return Shape

`ExportTarget.per_item(...)` returns `List[dict]` — one manifest per spatial point, ordered by `point_index`. Stable keys per manifest: `point_index`, `status` (`"ok"` / `"partial"` / `"failed"`), `models` (one entry per model with `model`, `status`, and `error` when failed), `summary` (`total_models` / `ok_models` / `failed_models`), and the written file path under the format-specific key (`npz_path` or `nc_path`). Points skipped by `config.resume` instead carry `resume_skipped=True` and `resume_output_path`.

`ExportTarget.combined(...)` returns a single `dict` manifest for the whole run with `status`, `n_items`, `models`, `summary`, and the output path under the format-specific key (`npz_path` or `nc_path`). When `config.resume` finds the export already complete, the manifest carries `resume_skipped=True` and `resume_output_path`.

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
        chunk_size=32,
        num_workers=8,
    ),
    input_prep="tile",
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

### Shared fetch override across models

```python
from rs_embed import FetchSpec, export_batch, ExportTarget, PointBuffer, TemporalSpec

export_batch(
    spatials=[PointBuffer(121.5, 31.2, 2048)],
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    models=["remoteclip", "prithvi"],
    fetch=FetchSpec(scale_m=10),
    target=ExportTarget.combined("exports/shared_sampling"),
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
    models=[ExportModelRequest.configure("thor", variant="large")],
    target=ExportTarget.combined("exports/thor_large_run"),
    backend="gee",
)
```

---

## Runtime Behavior You Usually Need to Know

### Inference scheduling

Model scheduling is serial, so one model runs at a time. Batch inference is used when the embedder supports it, and GPU or accelerator backends benefit the most from that path.

### Per-item vs combined mode

`per_item` mode writes one artifact per ROI, while `combined` mode writes one merged artifact for the run. Combined mode also keeps the older behavior of preferring batch model APIs when possible.

### Input reuse

If provider-backed export is used and both `save_inputs=True` and `save_embeddings=True`, rs-embed reuses the fetched input patch for both writing and embedding inference instead of downloading it twice.

!!! tip "Simple rule"
    Start with `ExportTarget.combined(...)` + `ExportConfig()`.
    Add `ExportModelRequest.configure(...)` only for the few models that need per-model sensor, fetch, modality, or variant overrides.

---
