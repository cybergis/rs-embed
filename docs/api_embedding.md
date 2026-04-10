# API: Embedding

This page covers single-ROI and batch embedding APIs.

Model IDs in examples use canonical short names (for example `remoteclip`). Legacy IDs (for example `remoteclip_s2rgb`) remain supported as aliases.

Related pages:

- [API: Specs and Data Structures](api_specs.md)
- [API: Export](api_export.md)
- [API: Inspect](api_inspect.md)

---

## Embedding Data Structure

`get_embedding` / `get_embeddings_batch` return an `Embedding`:

```python
from rs_embed.core.embedding import Embedding

Embedding(
    data: np.ndarray | xarray.DataArray,
    meta: dict[str, Any],
)
```

- `data`: the embedding data (float32, vector or grid)
- `meta`: includes model info, input info (optional), and export/check reports, etc.

---

## Embedding Functions

### get_embedding

```python
get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec | None = None,
    model_config: dict[str, Any] | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> Embedding
```

Computes the embedding for a single ROI.

**Parameters**

- `model`: model ID (see the *Supported Models* page, or use `rs_embed.list_models()`)
- `spatial`: `BBox` or `PointBuffer`
- `temporal`: `TemporalSpec` or `None`
- `sensor`: input descriptor for on-the-fly models; for most precomputed models this can be `None`
- `model_config`: optional model-specific runtime settings; for the currently documented variant-aware models, use it mainly as `{"variant": ...}`
- `modality`: optional model-facing modality selector (for example `s1`, `s2`, `s2_l2a`) for models that expose multiple input branches
- `output`: `OutputSpec.pooled()` or `OutputSpec.grid(...)`
- `backend`: access backend. `backend="auto"` is the public default and the recommended choice. For provider-backed on-the-fly models it resolves to a compatible provider backend; for precomputed models it lets rs-embed choose the model-compatible access path.
- `device`: `"auto" / "cpu" / "cuda"` (if the model depends on torch)
- `input_prep`: `"resize"` (default), `"tile"`, `"auto"`, or `InputPrepSpec(...)`

Modality contract:

- All public embedding APIs accept `modality`.
- Only models that explicitly expose a given modality can use it.
- Unsupported modality selections raise a `ModelError`.

`model_config` contract:

- `model_config` is optional and model-specific
- for the currently documented variant-aware models, the public field is unified as `variant`
- examples currently documented in model pages include:
  - `dofa`: `{"variant": "base" | "large"}`
  - `anysat`: `{"variant": "base"}`
  - `thor`: `{"variant": "tiny" | "small" | "base" | "large"}`
  - `satmaepp_s2_10b`: `{"variant": "large"}`
- if a model does not document `model_config`, leave it unset; unsupported usage raises `ModelError`
- when available, `describe()["model_config"]` is the machine-readable schema for supported keys and values

**Returns**

- `Embedding`

**Example**

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(pooling="mean"),
    backend="auto",
    device="auto",
    input_prep="resize",  # default
)
vec = emb.data  # (D,)
```

**Example with `model_config`**

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "thor",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    model_config={"variant": "large"},
)
```

---

### get_embeddings_batch

```python
get_embeddings_batch(
    model: str,
    *,
    spatials: list[SpatialSpec],
    temporal: TemporalSpec | None = None,
    sensor: SensorSpec | None = None,
    model_config: dict[str, Any] | None = None,
    modality: str | None = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = "resize",
) -> list[Embedding]
```

Batch-computes embeddings for multiple ROIs using the same embedder instance (often more efficient than looping over `get_embedding`).

**Parameters**

- `spatials`: a non-empty `list[SpatialSpec]`
- Others are the same as `get_embedding`

**Returns**

- `list[Embedding]` (same length as `spatials`)

**Example**

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
)
```

**Batch example with `model_config`**

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

embs = get_embeddings_batch(
    "anysat",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    model_config={"variant": "base"},
)
```

For model-specific keys and caveats, use the model detail pages as the source of truth.
For export-time usage of the same settings, see [API: Export](api_export.md).

---
