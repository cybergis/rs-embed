# Workflows

This page is a recipe collection for common tasks after you already know the basic APIs.
Use [Quickstart](quickstart.md) for the first-run path and [API](api.md) for exact signatures.

---

## One ROI Prototype

Use `get_embedding(...)` when you want one ROI embedding now and want the smallest possible call.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="auto",
    device="auto",
)
```

- you are prototyping
- you want to inspect metadata
- you are debugging model behavior on one location

---

## Many ROIs, One Model

Use `get_embeddings_batch(...)` when the model is fixed and you have multiple ROIs.

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embeddings_batch

spatials = [
    PointBuffer(121.5, 31.2, 2048),
    PointBuffer(120.5, 30.2, 2048),
]

embs = get_embeddings_batch(
    "remoteclip",
    spatials=spatials,
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="auto",
)
```

- same model, many points
- you want simpler code than manual loops
- you may benefit from embedder-level batch inference

---

## Build a Dataset Export

Use `export_batch(...)` for reproducible data pipelines and downstream experiments.
For new code, prefer `target=ExportTarget(...)` plus `config=ExportConfig(...)`.

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
    backend="auto",
    config=ExportConfig(save_inputs=True, save_embeddings=True, resume=True),
)
```

- Stable ROI names make exports/manifests easier to track.
- Apply one temporal policy consistently across all items for fair comparisons.
- Mix multiple models in one export job when building benchmark datasets.
- `per_item` keeps each ROI grouped together; useful for inspection and resume.
- Move runtime knobs into `ExportConfig(...)` instead of adding more top-level keywords.

---

## Inspect Inputs Before Modeling

Use patch inspection when outputs look suspicious (clouds, wrong band order, bad dynamic range, etc.).

### Preferred: provider-agnostic

```python
from rs_embed import inspect_provider_patch, PointBuffer, TemporalSpec, SensorSpec

report = inspect_provider_patch(
    spatial=PointBuffer(121.5, 31.2, 2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
    ),
    backend="gee",
)
```

---

## Large ROI with Tiling

If you request large ROIs for on-the-fly models, try API-side tiling:

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(121.5, 31.2, 8000),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.grid(),
    backend="auto",
    input_prep="tile",
)
```

Use `input_prep="tile"` when:

- `OutputSpec.grid()` matters
- large ROI resize would lose too much detail
- you accept extra runtime cost for better spatial structure preservation

---

## Fair Cross-Model Comparison

When benchmarking models, prefer:

- same ROI list
- same temporal window
- same compositing policy (`SensorSpec.composite`)
- `OutputSpec.pooled()` first
- default model normalization unless replicating original training setup

Then use [Supported Models](models.md) to review model-specific preprocessing and required side inputs.

---

## See Also

- [Quickstart](quickstart.md): first-run setup and the three core APIs
- [Concepts](concepts.md): semantic meaning of temporal, output, backend, and sensor
- [Models](models.md): model capability matrix and detail links
- [API](api.md): exact signatures and parameter docs
