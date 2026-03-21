# Model Detail Template

Use this page as a template when adding a new per-model documentation page (for example `docs/models/<model_id>.md` in the future).

Goal: make every model page answer the same questions quickly, so users can compare models without re-reading source code.

---

## Copy This Template

1. Duplicate this page into a new model doc file
2. Rename the title to the target model (for example `RemoteCLIP (remoteclip)`)
3. Fill every section marked `TODO`
4. Link the page from [Supported Models (Overview)](models.md)
5. Add or update any model-specific caveats in [Supported Models (Advanced Reference)](models_reference.md) if needed

---

## Template Body

### `<Model Name>` (`<model_id>`)

> One-sentence summary: what this model is good at and what rs-embed adapter path it uses.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `<model_id>` |
| Family / Backbone | `TODO` |
| Adapter type | `precomputed` / `on-the-fly` |
| Typical backend | `auto` / `gee` / provider-specific |
| Primary input | `TODO` |
| Temporal mode | `year` / `range` / model-specific |
| Output modes | `pooled`, `grid` |
| Model config keys | none / `variant` / `TODO` |
| Extra side inputs | none / `TODO` |
| Training alignment (adapter path) | Low / Medium / High + short note |

---

## When To Use This Model

### Good fit for

- `TODO`
- `TODO`

### Avoid or be careful when

- `TODO`
- `TODO`

---

## Input Contract (What the Adapter Expects)

### Spatial / temporal

- `SpatialSpec`: `TODO`
- `TemporalSpec`: `TODO` (for example `TemporalSpec.range(...)` only)
- Temporal semantics in rs-embed: `TODO` (single composite vs multi-frame)

### Sensor / channels

- Collection (default): `TODO`
- Band order (required): `TODO`
- Resolution / `scale_m`: `TODO`
- Fill / no-data behavior: `TODO`

### Extra metadata / side inputs

- Required side inputs beyond image tensor: `TODO`
- How they are derived in rs-embed: `TODO`

---

## Preprocessing Pipeline (Current rs-embed Path)

Document the **actual adapter path**, not the idealized paper pipeline.

1. Raw provider tensor format: `TODO` (for example `CHW`, `TCHW`)
2. Value range assumptions: `TODO`
3. Normalization steps: `TODO`
4. Resize / crop / pad policy: `TODO`
5. Any channel conversions or wavelength handling: `TODO`

### Environment variables that change behavior

| Env var | Default | Effect |
|---|---|---|
| `TODO` | `TODO` | `TODO` |

---

## Output Semantics

### `OutputSpec.pooled()`

- Output shape: `TODO`
- Pooling behavior: `TODO`
- Any caveats: `TODO`

### `OutputSpec.grid()`

- Output shape: `TODO`
- Is this georeferenced raster space or token grid? `TODO`
- Stitching / tiling behavior caveats (if relevant): `TODO`

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "<model_id>",
    spatial=PointBuffer(lon=0.0, lat=0.0, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",  # TODO adjust if not gee
)
```

### Example with model-specific options

```python
# TODO: show custom SensorSpec / env vars / output mode / input_prep as needed
```

---

## Common Failure Modes / Debugging

- `TODO`: missing optional dependency
- `TODO`: wrong band order / channel count
- `TODO`: temporal mode mismatch (`year` vs `range`)
- `TODO`: backend mismatch (`auto` vs provider-specific backend)

Recommended first check:

- Use `inspect_provider_patch(...)` to inspect raw inputs before blaming the model.

---

## Reproducibility Notes

For fair comparisons, document what must be held constant across models:

- ROI definition (`BBox` / `PointBuffer`)
- temporal window and compositing policy
- output mode (`pooled` vs `grid`)
- normalization mode / env overrides
- channel order / wavelength schema

---

## Source of Truth (Code Pointers)

- Catalog registration: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/<file>.py`
- Related helpers: `TODO`

---

## Doc Maintenance Checklist

- [ ] Model ID matches `MODEL_SPECS`
- [ ] Input channels / band order verified in adapter code
- [ ] Temporal semantics verified in adapter code
- [ ] Env vars verified in adapter code (not copied from stale docs)
- [ ] Example snippet runs or is syntactically valid
- [ ] Linked from [Supported Models (Overview)](models.md)
