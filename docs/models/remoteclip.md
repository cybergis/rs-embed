# RemoteCLIP (`remoteclip`)

> Sentinel-2 RGB on-the-fly embedding via `rshf.remoteclip.RemoteCLIP`, with pooled vector or ViT token-grid outputs.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `remoteclip` |
| Aliases | `remoteclip_s2rgb` |
| Family / Backbone | RemoteCLIP (CLIP-style ViT via `rshf.remoteclip.RemoteCLIP`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider-backed; prefer `backend="auto"` in public API |
| Primary input | S2 RGB (`B4,B3,B2`) |
| Temporal mode | `TemporalSpec.range(...)` required |
| Output modes | `pooled`, `grid` |
| Extra side inputs | none |
| Training alignment (adapter path) | Medium (higher if wrapper `model.transform(...)` matches training pipeline; fallback is generic CLIP preprocess) |

---

## When To Use This Model

### Good fit for

- fast RGB baselines on Sentinel-2
- CLIP-style embedding experiments and retrieval setups
- simple comparisons with `pooled` vectors across multiple models

### Be careful when

- you need strict multispectral semantics (RGB-only path)
- you assume `grid` is georeferenced pixels (it is a ViT token grid)
- your wrapper/model build only exposes pooled outputs (then `grid` can fail)

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- `SpatialSpec`: `BBox` or `PointBuffer`
- `TemporalSpec`: **must** be `TemporalSpec.range(start, end)`
- Temporal semantics: window filter + composite (default `median`), not single-scene selection

### Sensor / channels

Default path (if `sensor` is omitted):

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: `("B4", "B3", "B2")` (RGB in that order)
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`

Adapter notes:

- `sensor.collection` can be used as a checkpoint override with `hf:<repo_or_path>` (for example `hf:MVRL/remote-clip-vit-base-patch32`)
- `input_chw` (prefetched/raw path) must be `CHW` with exactly 3 bands in `(B4,B3,B2)` order

---

## Preprocessing Pipeline (Current rs-embed Path)

1. Fetch S2 RGB patch from provider (or reuse `input_chw`)
2. Normalize raw SR values to `[0,1]` (for `input_chw`, divide by `10000` and clip)
3. Optional input checks / quicklook export via `SensorSpec.check_*`
4. Convert `CHW [0,1]` -> `uint8 HWC`
5. Model preprocess:
   - preferred: `model.transform(rgb_u8, image_size)` if available
   - fallback: `Resize(224) -> CenterCrop(224) -> ToTensor -> CLIP normalization`
6. Forward pass to get tokens (preferred) or pooled vector (fallback path)

Current adapter image size:

- fixed `224` in this adapter path

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_REMOTECLIP_FETCH_WORKERS` | `8` | Provider prefetch worker count for batch APIs |
| `RS_EMBED_REMOTECLIP_BATCH_SIZE` | CPU:`8`, CUDA:`64` | Inference batch size for batch APIs |
| `HUGGINGFACE_HUB_CACHE` / `HF_HOME` / `HUGGINGFACE_HOME` | unset | Controls HF cache path used for model snapshot downloads |

Checkpoint override (not env-based in this adapter):

- set `sensor.collection="hf:<repo_or_local_path>"`

---

## Output Semantics

### `OutputSpec.pooled()`

- Returns a vector `(D,)`
- If token sequence is available, adapter mean-pools tokens (records `pooling="token_mean"` in metadata)
- If wrapper only returns pooled vector, adapter returns it directly

### `OutputSpec.grid()`

- Requires token sequence output `[N,D]`
- Returns ViT token grid as `xarray.DataArray` with shape `(D, Ht, Wt)`
- This is a **patch-token grid**, not georeferenced raster pixels
- Typical note from adapter: often `7x7` for ViT-B/32 at 224px (checkpoint-dependent)

---

## Examples

### Minimal example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="auto",
)
```

### Custom checkpoint via `sensor.collection="hf:..."`

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec, SensorSpec

emb = get_embedding(
    "remoteclip",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=SensorSpec(
        collection="hf:MVRL/remote-clip-vit-base-patch32",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
    ),
    output=OutputSpec.grid(),
    backend="auto",
)
```

---

## Common Failure Modes / Debugging

- `backend` is not provider-compatible (`backend="auto"` is the recommended public default)
- `TemporalSpec` is missing or not `range`
- `input_chw` shape is not `CHW` with 3 channels
- missing optional dependencies (`rshf`, `huggingface_hub`, torch stack)
- `grid` requested but wrapper only exposes pooled outputs / no token path

Recommended first check:

- Use `inspect_provider_patch(...)` to verify raw RGB inputs and temporal composite quality.

---

## Reproducibility Notes

For fair comparisons, keep fixed:

- same ROI and temporal window
- same `SensorSpec.composite` (`median` / `mosaic`)
- same `OutputSpec` mode (prefer `pooled` first)
- same checkpoint (`sensor.collection="hf:..."` override if used)
- same preprocessing path (be aware wrapper transform vs CLIP fallback can differ)

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_remoteclip.py`
- Token/grid utilities: `src/rs_embed/embedders/onthefly_remoteclip.py` (token reshape helpers)
