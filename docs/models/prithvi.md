# Prithvi-EO v2 (`prithvi`)



## Quick Facts

| Field                | Value                                                         |
| -------------------- | ------------------------------------------------------------- |
| Model ID             | `prithvi`                                                     |
| Aliases              | `prithvi_eo_v2_s2_6b`                                         |
| Family / Backbone    | Prithvi-EO v2 via vendored `PrithviMAE` runtime               |
| Adapter type         | `on-the-fly`                                                  |
| Model config keys    | `variant` (default: `prithvi_eo_v2_100_tl`), `temporal_mode` (`single`/`multi`, default `single`), `max_frames` (default `4`) |
| Training alignment   | Medium in `single` mode; higher in `multi` mode, which feeds a real multi-frame series like Prithvi's pretraining (also depends on resize/pad choices) |

!!! success "Prithvi In 30 Seconds"
    Prithvi-EO v2 is the IBM/NASA geospatial foundation model for a fixed Sentinel-2 6-band subset (`BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`), and its defining feature in `rs-embed` is that the vendored runtime requires *both* temporal coordinates and *location* coordinates as explicit model side inputs — the adapter derives them from the window midpoint and ROI center so you don't have to, but they are still a real part of the forward pass.

    In `rs-embed`, its most important characteristics are:

    - **required** temporal (`year, day_of_year`) and location (`lat, lon`) side inputs auto-derived by the adapter: see [Input Contract](#input-contract)
    - **multi-temporal at heart**: defaults to a single composite (`temporal_mode="single"`), but `temporal_mode="multi"` feeds a real time series matching its 4-timestep pretraining: see [Temporal mode](#temporal-mode-temporal_mode)
    - 30 m default `sensor.scale_m`, not the more common S2 10 m default — a frequent source of silent drift: see [Environment Variables / Tuning Knobs](#environment-variables-tuning-knobs)
    - `resize` vs `pad` preprocessing changes token geometry and should be treated as part of the experiment, not as a cosmetic knob: see [Environment Variables / Tuning Knobs](#environment-variables-tuning-knobs)

---

## Input Contract

| Field                 | Value                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Backend               | provider only (`gee` / `auto`)                                                                |
| `TemporalSpec`        | `range` preferred; `year(YYYY)` is normalized to `[YYYY-01-01, (YYYY+1)-01-01)`               |
| Default collection    | `COPERNICUS/S2_SR_HARMONIZED`                                                                 |
| Default bands (order) | `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2` (6-band, S2 semantic names)                    |
| Default fetch         | `scale_m=30` (note: **not** 10 m), `cloudy_pct=30`, `composite="median"`, `fill_value=0.0`    |
| `input_chw`           | `CHW`, `C=6`, raw SR `0..10000` — adapter clips and replaces non-finite values                |
| Side inputs           | **required** temporal coords `(year, day_of_year)` and location coords `(lat, lon)` — **auto-derived** by adapter from window midpoint and ROI center |

---

## Preprocessing Pipeline

!!! tip "Resize is the default — tiling is also available"
    The pipeline below shows the default `input_prep="resize"` path. For large ROIs, use `input_prep="tile"` to split the input into tiles and preserve spatial detail. See [Choosing Settings](../choosing_settings.md#input-preparation-resize-vs-tile).

```mermaid
flowchart LR
    INPUT["S2 fetch"] --> MODE{temporal_mode}
    MODE -- "single (default)" --> S1["1 median composite\n[6,H,W]"]
    MODE -- "multi" --> SN["T-frame series\n[T,6,H,W]\n(dup frames dropped)"]
    S1 --> PREP["Normalize → [0,1]\n→ per-frame resize or pad"]
    SN --> PREP
    PREP --> SIDE["Auto-derive side inputs:\ntemporal per frame (year+DOY)\nlocation (lat+lon)"]
    SIDE --> FWD["Prithvi encoder\n[B,6,T,H,W]"]
    FWD --> POOL["pooled: vector (D,)\n(pooled over time+space)"]
    FWD --> GRID["grid: patch-token grid (D,H',W')\n(pooled over time)"]
```

---

## Architecture Concept

```mermaid
flowchart LR
    S2["S2 6-band + side inputs\n(temporal + location)"] --> V{Variant}
    V -- "100_tl / 300_tl" --> E1["Encoder\npatch 16×16\n→ grid (D,14,14)"]
    V -- "600_tl" --> E2["Encoder\npatch 14×14\n→ grid (D,16,16)"]
    E1 --> OUT["pooled: vector\ngrid: patch-token grid"]
    E2 --> OUT
```

---

## Environment Variables / Tuning Knobs

| Env var                          | Default                | Effect                                                          |
| -------------------------------- | ---------------------- | --------------------------------------------------------------- |
| `RS_EMBED_PRITHVI_KEY`           | `prithvi_eo_v2_100_tl` | Prithvi variant selector                                        |
| `RS_EMBED_PRITHVI_PRETRAINED`    | `1`                    | Use pretrained weights vs random init                           |
| `RS_EMBED_PRITHVI_CACHE_DIR`     | unset                  | Optional Hugging Face cache dir for config/checkpoint downloads |
| `RS_EMBED_PRITHVI_WEIGHTS_ONLY`  | `1`                    | `torch.load(..., weights_only=...)` compatibility toggle        |
| `RS_EMBED_PRITHVI_PREP`          | `resize`               | Per-frame fit mode inside the adapter: `resize` or `pad` (independent of API `input_prep`) |
| `RS_EMBED_PRITHVI_IMG`           | `224`                  | Target square size for `resize` mode                            |
| `RS_EMBED_PRITHVI_PATCH_MULT`    | `16`                   | Pad multiple for `pad` mode                                     |
| `RS_EMBED_PRITHVI_TEMPORAL_MODE` | `single`               | Default temporal mode when `temporal_mode` is not in `model_config`: `single` or `multi` |
| `RS_EMBED_PRITHVI_MAX_FRAMES`    | `4`                    | Frame cap in `multi` mode (matches Prithvi-EO-2.0's 4-timestep pretraining) |
| `RS_EMBED_PRITHVI_FRAME_STRIDE_DAYS` | `30`               | Minimum spacing between frames in `multi` mode; `T = clamp(window_days // stride, 1, max_frames)` |
| `RS_EMBED_PRITHVI_FETCH_WORKERS` | `8`                    | Provider prefetch workers for batch APIs                        |
| `RS_EMBED_PRITHVI_BATCH_SIZE`    | CPU:`4`, CUDA:`16`     | Inference batch size for batch APIs                             |

---

## Model-specific Settings

`variant` selects the Prithvi-EO v2 backbone size. In `rs-embed`, pass it as `variant="prithvi_eo_v2_100_tl" | "prithvi_eo_v2_300_tl" | "prithvi_eo_v2_600_tl"`, or use the short aliases `"100_tl"` / `"300_tl"` / `"600_tl"` (the `100m_tl` / `300m_tl` / `600m_tl` spellings are also accepted).

| Variant   | Model key (runtime)    | HF repo                                      | Checkpoint file             | Patch size `(T,H,W)` | Embed dim | Transformer blocks | Attention heads | Notes                                                                                                        |
| --------- | ---------------------- | -------------------------------------------- | --------------------------- | -------------------- | --------- | ------------------ | --------------- | ------------------------------------------------------------------------------------------------------------ |
| `100_tl`  | `prithvi_eo_v2_100_tl` | `ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL` | `Prithvi_EO_V2_100M_TL.pt`  | `(1, 16, 16)`        | 768       | 12                 | 12              | Current default. ~100M params; ViT-B-class encoder with temporal+location side inputs.                      |
| `300_tl`  | `prithvi_eo_v2_300_tl` | `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` | `Prithvi_EO_V2_300M_TL.pt`  | `(1, 16, 16)`        | 1024      | 24                 | 16              | ~300M params; ViT-L-class encoder. Same patch geometry as `100_tl`, so token grid counts match.             |
| `600_tl`  | `prithvi_eo_v2_600_tl` | `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL` | `Prithvi_EO_V2_600M_TL.pt`  | `(1, 14, 14)`        | 1280      | 32                 | 16              | Highest capacity. ~600M params; uses a **different** spatial patch size (14, not 16), so token grid geometry differs from the smaller two variants at the same `RS_EMBED_PRITHVI_IMG`. |

!!! info "How To Read Embed Dim"
    `Embed dim` is Prithvi's encoder `embed_dim`. It becomes the pooled embedding width `(D,)` and the channel dimension of a `grid` output `(D,H,W)`.

!!! warning "Patch Size Differs For `600_tl`"
    `100_tl` and `300_tl` use a `(1,16,16)` patch, while `600_tl` uses `(1,14,14)`. At the default `RS_EMBED_PRITHVI_IMG=224`, that means `14×14` patch tokens for the smaller variants but `16×16` patch tokens for `600_tl`. If you compare grids across variants, either keep variant fixed or use `RS_EMBED_PRITHVI_PREP=pad` with an `IMG` that divides cleanly by both `14` and `16` (for example `224`, which does).

All three variants share the same fixed Sentinel-2 6-band input (`BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`) and the same required temporal+location side inputs derived by the adapter. The checkpoints ship a `num_frames=4` config, but the adapter loads the runtime with the frame count it actually feeds — `1` in `single` mode, or the window-derived `T` in `multi` mode (see [Temporal mode](#temporal-mode-temporal_mode)).

`variant` overrides `RS_EMBED_PRITHVI_KEY`. For export jobs, use `ExportModelRequest.configure("prithvi", variant="prithvi_eo_v2_300_tl")`.

### Temporal mode (`temporal_mode`)

Prithvi-EO 2.0 was pretrained on **4-timestep** HLS series with **1–6 month gaps** between consecutive frames ([arXiv:2412.02732](https://arxiv.org/abs/2412.02732)), and each frame's real `(year, day_of_year)` is fed through the temporal embedding. `rs-embed` exposes this via `temporal_mode`:

- **`single`** (default): one median composite over the whole window (`T=1`). Backward-compatible; nothing changes for existing callers.
- **`multi`**: a true `[B,6,T,H,W]` series with per-frame dates. The frame count is derived from the requested window rather than forced to a fixed `T`:

    ```
    T = clamp(window_days // 30, 1, max_frames)   # max_frames default = 4
    ```

    A ~30-day minimum spacing (the low end of the training interval) means short windows collapse to `T=1` instead of being padded with duplicate frames. Frames that the provider back-fills for empty sub-windows (identical copies of the whole-window composite) are **dropped**, so a window with no real temporal diversity also degrades to `T=1`. Frame dates align with the provider's binning (shared `split_date_range`).

!!! note "Output dimensionality is unchanged"
    Multi-frame tokens are pooled over time (and space), so `pooled` is still `(D,)` and `grid` is still `(D, H', W')` — switching modes never changes the embedding shape, only the values and the `num_frames` / `frame_dates` metadata. To stay near the training regime, give `multi` a window of a few months up to ~a year.

```python
emb = get_embedding(
    "prithvi",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-01-01", "2023-01-01"),  # ~1 yr -> 4 frames
    output=OutputSpec.pooled(),
    backend="gee",
    temporal_mode="multi",
)
```

Tuning knobs: `max_frames` (or env `RS_EMBED_PRITHVI_MAX_FRAMES`) caps the count; `RS_EMBED_PRITHVI_FRAME_STRIDE_DAYS` sets the minimum spacing; `RS_EMBED_PRITHVI_TEMPORAL_MODE` sets the default mode.

Example:

```python
from rs_embed import PointBuffer, TemporalSpec, OutputSpec, get_embedding

emb = get_embedding(
    "prithvi",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    variant="300_tl",
)
```

---

## Examples

### Minimal example (explicit temporal window)

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "prithvi",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
)
```

### With custom preprocessing mode (env-controlled)

```python
# Example (shell):
export RS_EMBED_PRITHVI_PREP=pad
export RS_EMBED_PRITHVI_PATCH_MULT=16
export RS_EMBED_PRITHVI_PRETRAINED=1
```

### With variant selection

```python
from rs_embed import get_embedding, PointBuffer, TemporalSpec, OutputSpec

emb = get_embedding(
    "prithvi",
    spatial=PointBuffer(lon=121.5, lat=31.2, buffer_m=2048),
    temporal=TemporalSpec.range("2022-06-01", "2022-09-01"),
    output=OutputSpec.pooled(),
    backend="gee",
    variant="prithvi_eo_v2_300_tl",
)
```

---

## Paper & Links

- **Prithvi-EO 2.0** (the `*_tl` variants here): [arXiv:2412.02732](https://arxiv.org/abs/2412.02732) — pretrained on 4-timestep HLS series with 1–6 month gaps; informs `temporal_mode="multi"`.
- **Prithvi-EO 1.0**: [arXiv:2310.18660](https://arxiv.org/abs/2310.18660)
- **Model**: [ibm-nasa-geospatial](https://huggingface.co/ibm-nasa-geospatial)

---

## Reference

- Default `scale_m` is `30`, not `10` — this is intentional and differs from most other S2 models.
- `temporal_mode="single"` (default) sends one composite (`T=1`); `temporal_mode="multi"` sends a window-derived `T`-frame series (capped at `max_frames=4`, ~30-day min spacing, duplicate frames dropped). Output shape is identical in both modes.
- `resize` vs `pad` preprocessing changes token geometry; treat it as part of experiment design.
- Variant `600_tl` uses patch size 14 (not 16), producing a different grid shape than `100_tl`/`300_tl`.
