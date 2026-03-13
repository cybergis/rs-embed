# TerraFM-B (`terrafm`)

> TerraFM-B adapter supporting both provider and tensor backends, with two input modalities (`s2` 12-band Sentinel-2 SR or `s1` VV/VH Sentinel-1) and model-native feature-map grids.

## Quick Facts

| Field | Value |
|---|---|
| Model ID | `terrafm` |
| Aliases | `terrafm_b` |
| Family / Backbone | TerraFM-B from Hugging Face (`MBZUAI/TerraFM`) |
| Adapter type | `on-the-fly` |
| Typical backend | provider backend (`gee`), also supports `backend="tensor"` |
| Primary input | S2 SR 12-band or S1 VV/VH (selected by `sensor.modality`) |
| Temporal mode | provider path requires `TemporalSpec.range(...)` (v0.1 behavior) |
| Output modes | `pooled`, `grid` |
| Extra side inputs | modality settings on `sensor` (`modality`, `orbit`, `use_float_linear`) |
| Training alignment (adapter path) | Medium-High when modality-specific preprocessing matches the intended TerraFM path |

---

## When To Use This Model

### Good fit for

- experiments comparing S2 and S1 representations under one backbone family
- workflows needing both provider fetch and direct tensor backend
- model-native feature-map outputs instead of token-only grids

### Be careful when

- mixing S1 and S2 runs without logging modality and preprocessing path
- passing tensor backend inputs with wrong channel count (`C` must be `2` or `12`)
- assuming `backend="auto"` selects a non-provider path; it resolves through the embedder's provider contract unless you use `tensor`

---

## Input Contract (Current Adapter Path)

### Backend modes

- `backend="tensor"`:
  - requires `input_chw` as `CHW`
  - batch tensor inputs should use `get_embeddings_batch_from_inputs(...)`
  - adapter resizes to `224`
- provider backend (`gee` / provider-compatible, including `auto` via provider resolution):
  - requires `TemporalSpec.range(...)` in v0.1
  - fetches S2 or S1 based on `modality` (or `sensor.modality` if passed through `SensorSpec`)

### Modality selection (`modality` or `sensor.modality`)

- `s2` (default):
  - 12-band Sentinel-2 SR input (`B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12`)
  - provider fetch returns normalized `[0,1]`
  - `input_chw` override expects raw SR `0..10000`, adapter scales to `[0,1]`

- `s1`:
  - 2-band Sentinel-1 VV/VH input (`VV`,`VH`)
  - provider path fetches raw VV/VH then normalizes via shared S1 normalization helper
  - `input_chw` override expects raw VV/VH and applies `log1p` + percentile scaling to `[0,1]`

### Sensor fields used by adapter (provider path)

- common: `scale_m`, `cloudy_pct`, `composite`
- S1-specific: `orbit`, `use_float_linear`

Channel sanity:

- TerraFM path is strict: `C` must be `12` (S2) or `2` (S1)

---

## Preprocessing Pipeline (Current rs-embed Path)

### Provider path

1. Validate `TemporalSpec.range(...)`
2. Select modality from `modality` (`s2` / `s1`)
3. Fetch provider patch:
   - S2: 12-band SR -> normalize to `[0,1]`
   - S1: VV/VH raw -> shared S1 normalization helper -> `[0,1]`
4. Optional input inspection on normalized provider input
5. Resize to fixed `224x224`
6. Load TerraFM-B from HF code + `.pth` weights
7. Forward:
   - `pooled`: TerraFM forward returns CLS embedding `(D,)`
   - `grid`: adapter calls `extract_feature(...)` and uses last-layer feature map `(D,H,W)`

### Tensor backend path

1. Read `input_chw` (`CHW`)
2. Resize to `224x224`
3. Validate channel count (`2` or `12`)
4. Load TerraFM-B and run same forward/grid extraction path

Notes:

- Tensor backend path does not apply provider-specific fetch normalization automatically; you are responsible for matching expected input scale/semantics.

---

## Environment Variables / Tuning Knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_EMBED_TERRAFM_FETCH_WORKERS` | `8` | Provider prefetch workers for batch APIs |
| `RS_EMBED_TERRAFM_BATCH_SIZE` | CPU:`8`, CUDA:`64` | Inference batch size for batch APIs |

Related cache envs (used by HF asset download path):

- `HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `HUGGINGFACE_HOME`

Adapter behavior notes:

- image size is fixed to `224` in current implementation
- weights/code are fetched from `MBZUAI/TerraFM` (`terrafm.py` + `TerraFM-B.pth`)

---

## Output Semantics

### `OutputSpec.pooled()`

- Returns TerraFM forward output (CLS embedding) `(D,)`
- This is not token pooling; it is the model’s pooled embedding path

### `OutputSpec.grid()`

- Returns last-layer TerraFM feature map via `extract_feature(...)`
- `xarray.DataArray` shape `(D,H,W)`
- Metadata includes `grid_type="feature_map"`
- Grid is model feature-map layout, not georeferenced raster pixels

---

## Examples

### Minimal provider-backed S2 example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "terrafm",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    modality="s2",
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### Minimal provider-backed S1 example

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec, SensorSpec

sensor = SensorSpec(
    collection="COPERNICUS/S1_GRD_FLOAT",
    bands=("VV", "VH"),
    scale_m=10,
    composite="median",
    use_float_linear=True,
)

emb = get_embedding(
    "terrafm",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    sensor=sensor,
    modality="s1",
    output=OutputSpec.pooled(),
    backend="gee",
)
```

Notes:

- Prefer passing `modality="s1"` / `modality="s2"` directly at the public API layer.
- Setting `modality="s1"` is what switches TerraFM onto the S1 path; changing only `collection` / `bands` is not enough.
- `use_float_linear=True` matches `COPERNICUS/S1_GRD_FLOAT`; set it to `False` for `COPERNICUS/S1_GRD`.

---

## Common Failure Modes / Debugging

- using an unsupported backend; use `backend="auto"`, an explicit provider backend, or `tensor`
- provider path with non-`range` temporal spec
- tensor backend without `input_chw`
- wrong channel count (`C` must be `2` or `12`)
- S1/S2 modality mismatch between data and `modality`
- HF asset download issues (code or `.pth` weights)

Recommended first checks:

- inspect metadata `modality`, `source`, `grid_type`, and weight file info
- verify tensor input scale/normalization if using `backend="tensor"`
- start with S2 default path before enabling S1 overrides

---

## Reproducibility Notes

Keep fixed and record:

- backend mode (`provider` vs `tensor`)
- modality (`s2` / `s1`) and S1-specific options (`orbit`, `use_float_linear`)
- temporal window + compositing settings (provider path)
- output mode (`pooled` / `grid`)
- TerraFM HF asset source/cache snapshot if benchmarking

---

## Source of Truth (Code Pointers)

- Registration/catalog: `src/rs_embed/embedders/catalog.py`
- Adapter implementation: `src/rs_embed/embedders/onthefly_terrafm.py`
