# DOFA (`dofa`)

> TorchGeo DOFA adapter for multispectral inputs with explicit per-channel wavelengths, supporting provider and tensor backends.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `dofa` |
| Family / Backbone | TorchGeo DOFA (`dofa_base_patch16_224` / `dofa_large_patch16_224`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`), also supports `backend="tensor"` |
| Primary input | Multiband SR CHW + wavelengths (µm) |
| Temporal mode | provider path requires `TemporalSpec.range(...)` |
| Output modes | `pooled`, `grid` |
| Extra side inputs | **required** wavelength vector (`wavelengths_um`) |
| Training alignment (adapter path) | Medium-High (when wavelengths and band semantics are correct) |

---

## When To Use This Model

### Good fit for

- multispectral experiments where wavelength-aware modeling matters
- custom sensor/band combinations (if you provide matching wavelengths)
- comparing spectral models against S2-specific models

### Be careful when

- wavelengths are missing or mismatched with channels
- assuming arbitrary bands can be inferred automatically (only known sets like S2 are inferable)
- comparing results without logging `variant` and wavelengths used

---

## Input Contract (Current Adapter Path)

### Spatial / temporal

- Provider path requires `TemporalSpec.range(start, end)`
- Tensor path does not use provider/temporal fetch semantics

### Sensor / channels (provider path)

Default `SensorSpec` if omitted:

- Collection: `COPERNICUS/S2_SR_HARMONIZED`
- Bands: S2 SR 12-band set (`B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`)
- `scale_m=10`, `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`

Wavelengths:

- Adapter requires one wavelength (µm) per channel
- If `sensor.wavelengths` is not provided, adapter tries to infer from `sensor.bands` (works for recognized S2 band names)
- `len(wavelengths_um)` must equal channel count `C`

`input_chw` contract (provider override path):

- must be `CHW` with `C == len(bands)`
- raw SR values expected (`0..10000`), adapter converts to `[0,1]`

### Tensor backend contract

- `backend="tensor"` requires `input_chw` as `CHW`
- batch tensor inputs should use `get_embeddings_batch_from_inputs(...)`
- `sensor.wavelengths` should be provided, or `sensor.bands` must allow wavelength inference

---

## Preprocessing Pipeline (Current rs-embed Path)

### Provider path

1. Fetch multiband SR patch and scale to `[0,1]`
2. Optional input inspection (`expected_channels=len(bands)`, value range `[0,1]`)
3. Resize to fixed `224x224` (bilinear; no crop/pad)
4. Load DOFA model variant (`base` / `large`)
5. Forward with image tensor + wavelength vector
6. Return pooled embedding or reshape tokens to patch-token grid

### Tensor path

1. Read `input_chw` (`CHW`)
2. Resize to `224x224`
3. Resolve wavelengths from `sensor.wavelengths` or infer from `sensor.bands`
4. Forward DOFA with image + wavelengths

Fixed adapter behavior:

- image size fixed to `224` in current implementation

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_DOFA_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_DOFA_BATCH_SIZE` | CPU:`8`, CUDA:`64` | Inference batch size for batch APIs |

Non-env model selection knobs (passed via `sensor` fields in current adapter path):

- `sensor.variant`: `base` / `large`
- `sensor.bands`: channel semantics for provider fetch and wavelength inference
- `sensor.wavelengths`: explicit wavelength vector (µm)

---

## Output Semantics

### `OutputSpec.pooled()`

- Returns DOFA pooled vector `(D,)`
- Metadata includes wavelength vector, variant, preprocess strategy, token metadata

### `OutputSpec.grid()`

- Reshapes DOFA patch tokens to `xarray.DataArray` `(D,H,W)` (usually square token grid)
- Requires token count to be a perfect square
- Grid is ViT patch-token layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed example (S2 wavelengths inferred automatically)

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "dofa",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Custom bands / wavelengths example (conceptual)

```python
from rs_embed import SensorSpec

sensor = SensorSpec(
    collection="COPERNICUS/S2_SR_HARMONIZED",
    bands=("B2", "B3", "B4", "B8"),
    scale_m=10,
)
# If bands are non-standard, provide wavelengths explicitly via an extended sensor object/field used by your code path.
```

---

## Common Failure Modes / Debugging

- provider path called with non-`range` temporal spec
- wavelength vector missing or wrong length for channel count
- unsupported band names for automatic wavelength inference
- tensor backend called without `input_chw`
- unknown `variant` (must be `base` or `large`)

Recommended first checks:

- print/log `bands` and `wavelengths_um` used by the adapter
- verify provider input is scaled/ordered as expected before forward pass

---

## Reproducibility Notes

Keep fixed and record:

- `variant` (`base` vs `large`)
- exact `bands` and `wavelengths_um`
- temporal window and compositing (provider path)
- output mode (`pooled` vs `grid`)
- whether backend is `provider` or `tensor`

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_dofa.py`
- Wavelength inference map: `src/rs_embed/embedders/onthefly_dofa.py`
