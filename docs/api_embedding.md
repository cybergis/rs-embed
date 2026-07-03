# API: Embedding

This page covers single-ROI and batch embedding APIs.

Related pages: [API: Specs and Data Structures](api_specs.md), [API: Export](api_export.md), and [API: Inspect](api_inspect.md).

---

## Embedding Data Structure

`get_embedding(...)` returns one `Embedding`, and `get_embeddings_batch(...)` returns `List[Embedding]`:

```python
from rs_embed.core.embedding import Embedding

Embedding(
    data: np.ndarray | xarray.DataArray,
    meta: Dict[str, Any],
)
```

`data` holds the embedding itself as a float32 vector or grid, and `meta` carries the runtime metadata for that returned result.

For the general `Embedding.meta` schema, common fields, and its relationship to `describe_model()` / `Model.describe()`, see [API Specs & Data Structures](api_specs.md#embedding).

---

## Embedding Functions

### get_embedding

#### Signature

```python
get_embedding(
    model: str,
    *,
    spatial: SpatialSpec,
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    fetch: Optional[FetchSpec] = None,
    modality: Optional[str] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = None,
    **model_kwargs,
) -> Embedding
```

Computes the embedding for a single ROI.

#### Parameters

Core inputs:

| Parameter  | Meaning                                                                                                                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`    | Model ID. See [Supported Models](models.md) or call `rs_embed.list_models()`.                                                                                                                                    |
| `spatial`  | `BBox` or `PointBuffer`.                                                                                                                                                                                         |
| `temporal` | `TemporalSpec` or `None`. The parameter is optional at the API level, but some models or data sources still require it.                                                                                          |
| `sensor`   | Full input descriptor for on-the-fly models. Most precomputed models can leave this as `None`. When provided, it overrides source-level details such as collection, bands, scale, and compositing.               |
| `fetch`    | Lightweight sampling override for common cases such as `scale_m`, `cloudy_pct`, `composite`, and `fill_value`. It is applied on top of the model's resolved default sensor and cannot be combined with `sensor`. |
| `output`   | Usually `OutputSpec.pooled()` or `OutputSpec.grid(...)`.                                                                                                                                                         |

Runtime and branch selection:

| Parameter    | Meaning                                                                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modality`   | Optional model-facing modality selector such as `s1`, `s2`, or `s2_l2a` for models that expose multiple branches. The API normalizes common aliases such as `sentinel-1 -> s1` and `sentinel-2 -> s2`.              |
| `backend`    | Access backend. `backend="auto"` is the public default and the recommended choice. Precomputed models typically expect `auto`, while provider-backed on-the-fly paths commonly use `gee` through the auto resolver. |
| `device`     | `"auto"`, `"cpu"`, or `"cuda"` for torch-backed models.                                                                                                                                                             |
| `input_prep` | `"resize"`, `"tile"`, `"auto"`, `InputPrepSpec(...)`, or `None` (default). For image-level ViT patch-token grid models (`scalemae`, `satmae`, `satmaepp`), `None`/`"auto"` with `OutputSpec.grid()` resolves to `"resize"` and warns; explicit `"tile"` also warns. Explicit `"resize"` does not warn. |

Model-specific settings:

| Parameter        | Meaning                                                                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `variant`        | Common model-specific selector passed through `**model_kwargs`, for example `variant="large"` or `variant="base"`. Only models that declare variant support accept it. |
| `**model_kwargs` | Model-specific settings passed directly as keyword arguments, such as `variant="large"`. Accepted keys depend on the model.                                            |

#### Rules And Contracts

`fetch` vs `sensor`

Prefer `fetch=FetchSpec(...)` when you only want to change fetch policy. Use `sensor=SensorSpec(...)` only for advanced source overrides such as custom collections, band lists, or modality-specific source contracts. Passing both raises `ModelError`.

`modality`

`modality` can do two things depending on the request shape. If you do not pass `sensor`, it helps resolve the model's default sensor profile. If you do pass `sensor`, the modality is merged into that sensor selection. Unsupported modality names raise `ModelError`.

`input_prep`

String values must be one of `"resize"`, `"auto"`, or `"tile"`. `tile` is stricter than `auto`: it requires a provider-backed fetch path, a resolvable sensor, and model support for `input_chw`. If `tile` is requested and no `sensor` is passed, the API tries to fill in the model's default sensor automatically.

`**model_kwargs`

Model-specific settings are optional and vary by model. Pass them as direct keyword arguments rather than as a dict. Variant-aware models currently documented include `dofa`, `anysat`, `thor`, `satmaepp` (the `s2_10b` modality), and `prithvi`; accepted values are documented on the corresponding model pages. If a model does not accept keyword settings, passing unknown keys raises `ModelError`. `describe_model(model_id)["model_config"]` is the machine-readable schema for supported keys and values.

#### Returns

Returns one `Embedding`.

#### Example

Minimal call:

```python
from rs_embed import FetchSpec, PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(pooling="mean"),
    fetch=FetchSpec(scale_m=10),
    backend="auto",
    device="auto",
    input_prep="tile",  # default; pass "resize" to downsample instead
)
vec = emb.data  # (D,)
```

Variant-aware models use the same call shape, with an extra keyword such as `variant="large"`.

---

### get_embeddings_batch

#### Signature

```python
get_embeddings_batch(
    model: str,
    *,
    spatials: List[SpatialSpec],
    temporal: Optional[TemporalSpec] = None,
    sensor: Optional[SensorSpec] = None,
    fetch: Optional[FetchSpec] = None,
    modality: Optional[str] = None,
    output: OutputSpec = OutputSpec.pooled(),
    backend: str = "auto",
    device: str = "auto",
    input_prep: InputPrepSpec | str | None = None,
    **model_kwargs,
) -> List[Embedding]
```

Batch-computes embeddings for multiple ROIs using the same embedder instance (often more efficient than looping over `get_embedding`).

#### Parameters

Core inputs:

| Parameter  | Meaning                                                                                                                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`    | Model ID. See [Supported Models](models.md) or call `rs_embed.list_models()`.                                                                                                                                    |
| `spatials` | Non-empty `List[SpatialSpec]`. Each item is a `BBox` or `PointBuffer`. Output order matches input order.                                                                                                         |
| `temporal` | `TemporalSpec` or `None`. The parameter is optional at the API level, but some models or data sources still require it.                                                                                          |
| `sensor`   | Full input descriptor for on-the-fly models. Most precomputed models can leave this as `None`. When provided, it overrides source-level details such as collection, bands, scale, and compositing.               |
| `fetch`    | Lightweight sampling override for common cases such as `scale_m`, `cloudy_pct`, `composite`, and `fill_value`. It is applied on top of the model's resolved default sensor and cannot be combined with `sensor`. |
| `output`   | Usually `OutputSpec.pooled()` or `OutputSpec.grid(...)`.                                                                                                                                                         |

Runtime and branch selection:

| Parameter    | Meaning                                                                                                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modality`   | Optional model-facing modality selector such as `s1`, `s2`, or `s2_l2a` for models that expose multiple branches. The API normalizes common aliases such as `sentinel-1 -> s1` and `sentinel-2 -> s2`.              |
| `backend`    | Access backend. `backend="auto"` is the public default and the recommended choice. Precomputed models typically expect `auto`, while provider-backed on-the-fly paths commonly use `gee` through the auto resolver. |
| `device`     | `"auto"`, `"cpu"`, or `"cuda"` for torch-backed models.                                                                                                                                                             |
| `input_prep` | `"resize"`, `"tile"`, `"auto"`, `InputPrepSpec(...)`, or `None` (default). For image-level ViT patch-token grid models (`scalemae`, `satmae`, `satmaepp`), `None`/`"auto"` with `OutputSpec.grid()` resolves to `"resize"` and warns; explicit `"tile"` also warns. Explicit `"resize"` does not warn. |

Model-specific settings:

| Parameter        | Meaning                                                                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `variant`        | Common model-specific selector passed through `**model_kwargs`, for example `variant="large"` or `variant="base"`. Only models that declare variant support accept it. |
| `**model_kwargs` | Model-specific settings passed directly as keyword arguments, such as `variant="large"`. Accepted keys depend on the model.                                            |

#### Rules And Contracts

`fetch` vs `sensor`

Prefer `fetch=FetchSpec(...)` when you only want to change fetch policy. Use `sensor=SensorSpec(...)` only for advanced source overrides such as custom collections, band lists, or modality-specific source contracts. Passing both raises `ModelError`.

`modality`

`modality` can do two things depending on the request shape. If you do not pass `sensor`, it helps resolve the model's default sensor profile. If you do pass `sensor`, the modality is merged into that sensor selection. Unsupported modality names raise `ModelError`.

`input_prep`

String values must be one of `"resize"`, `"auto"`, or `"tile"`. `tile` is stricter than `auto`: it requires a provider-backed fetch path, a resolvable sensor, and model support for `input_chw`. If `tile` is requested and no `sensor` is passed, the API tries to fill in the model's default sensor automatically.

`**model_kwargs`

Model-specific settings are optional and vary by model. Pass them as direct keyword arguments rather than as a dict. Variant-aware models currently documented include `dofa`, `anysat`, `thor`, `satmaepp` (the `s2_10b` modality), and `prithvi`; accepted values are documented on the corresponding model pages. If a model does not accept keyword settings, passing unknown keys raises `ModelError`. `describe_model(model_id)["model_config"]` is the machine-readable schema for supported keys and values.

#### Typical Use

Use `fetch` here when you want to keep the model default collection and band contract, override only common knobs such as `scale_m` or `cloudy_pct`, or compare multiple models under one shared sampling policy.

#### Returns

Returns `List[Embedding]` with the same length and order as `spatials`.

#### Example

Minimal batch call:

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

Variant-aware batch calls follow the same pattern, for example by adding `variant="base"` to `get_embeddings_batch(...)`.

For model-specific keys and caveats, use the model detail pages as the source of truth.
For export-time usage of the same settings, see [API: Export](api_export.md).

---
