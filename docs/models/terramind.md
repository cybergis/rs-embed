# TerraMind (`terramind`)

> TerraTorch-backed TerraMind adapter for Sentinel-2 SR 12-band inputs, supporting provider and tensor backends with TerraMind-specific z-score normalization.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `terramind` |
| Family / Backbone | TerraMind via TerraTorch `BACKBONE_REGISTRY` |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`), also supports `backend="tensor"` |
| Primary input | S2 SR 12-band (`B1..B12` subset used by adapter order) |
| Temporal mode | `range` (provider path normalized via shared helper) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none required in current adapter |
| Training alignment (adapter path) | High when default TerraMind z-score normalization is preserved |

---

## When To Use This Model

### Good fit for

- strict multispectral S2 experiments with TerraMind checkpoints
- comparisons requiring a strong S2 12-band encoder
- workflows that need both provider and direct tensor backend paths

### Be careful when

- changing normalization mode away from TerraMind stats (`zscore`)
- assuming `grid` is georeferenced raster space (it is patch-token grid)
- using tensor backend without matching expected channel order and preprocessing assumptions

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider path: `SpatialSpec` + temporal normalized to range via shared helper
- Tensor path: no provider fetch; pass `input_chw` as `CHW`

### Sensor / channels (provider path)

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: adapter fetch order `_S2_SR_12_BANDS` = `B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

TerraMind internal semantic mapping is also tracked in metadata (`bands_terramind`).

`input_chw` contract (provider override path):

- must be `CHW` with 12 bands in adapter fetch order
- raw SR values expected in `0..10000`

### Tensor backend contract

- `backend="tensor"` requires `input_chw`
- accepted shape: `CHW`
- batch tensor inputs should use `get_embeddings_batch_from_inputs(...)`
- `C` must be `12`
- adapter resizes to `224` and applies TerraMind normalization before forward

---

## Preprocessing Pipeline (Current rs-embed Path)

### Provider path

1. Fetch 12-band raw S2 SR patch (`0..10000`), composite over temporal window
2. Optional input inspection checks on raw values (`value_range=(0,10000)`)
3. Resize to fixed `224x224`
4. Apply TerraMind normalization (`RS_EMBED_TERRAMIND_NORMALIZE`, default `zscore`)
   - `zscore`: uses TerraMind v1 or v01 stats depending on model key prefix
   - `raw/none`: no z-score stats, only `nan_to_num`
5. Forward TerraMind backbone and extract token tensor
6. Pool tokens or reshape to patch-token grid

### Tensor path

- Reads `input_chw`, resizes to `224`, applies same normalization, then forwards model

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_TERRAMIND_MODEL_KEY` | `terramind_v1_small` | TerraMind backbone key |
| `RS_EMBED_TERRAMIND_MODALITY` | `S2L2A` | Modality passed to TerraMind/TerraTorch |
| `RS_EMBED_TERRAMIND_NORMALIZE` | `zscore` | Input normalization mode (`zscore` vs raw/none) |
| `RS_EMBED_TERRAMIND_LAYER_INDEX` | `-1` | Which layer output to select when sequence-like outputs are returned |
| `RS_EMBED_TERRAMIND_PRETRAINED` | `1` | Use pretrained weights |
| `RS_EMBED_TERRAMIND_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |

Fixed adapter behavior:

- image size is fixed to `224` in current implementation

---

## Output Semantics

### `OutputSpec.pooled()`

- Pools TerraMind tokens with `mean` / `max` according to `OutputSpec.pooling`
- Metadata records pooling mode and whether CLS removal happened

### `OutputSpec.grid()`

- Returns ViT patch-token grid `(D,H,W)` as `xarray.DataArray`
- Metadata includes `grid_type="vit_patch_tokens"`, `grid_hw`, and `cls_removed`
- Grid is model token layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "terramind",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Example normalization/model tuning (env-controlled)

```python
# Example (shell):
# export RS_EMBED_TERRAMIND_MODEL_KEY=terramind_v1_small
# export RS_EMBED_TERRAMIND_NORMALIZE=zscore
# export RS_EMBED_TERRAMIND_MODALITY=S2L2A
```

---

## Common Failure Modes / Debugging

- wrong channel count for `input_chw` / tensor backend (`C` must be 12)
- backend mismatch (`tensor` path requires `input_chw`; provider path requires provider backend)
- hidden normalization changes via `RS_EMBED_TERRAMIND_NORMALIZE`
- TerraTorch import/build issues for selected model key or optional deps

Recommended first checks:

- verify provider raw input channel order and range before normalization
- inspect metadata for model key, modality, normalization mode, and layer index

---

## Reproducibility Notes

Keep fixed across runs:

- model key (`v1` vs `v01` changes which z-score stats are used)
- normalization mode (`zscore` strongly recommended)
- modality (`S2L2A` default)
- output mode/pooling choice
- temporal window and compositing settings (provider path)

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_terramind.py`
- Token/grid helpers: `src/rs_embed/embedders/_vit_mae_utils.py`
