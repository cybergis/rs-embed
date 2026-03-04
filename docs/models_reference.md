# Supported Models (Advanced Reference)

This page preserves the **detailed model comparison matrices** and preprocessing notes.

If you are choosing a model for the first time, start with:

- [Supported Models (Overview)](models.md)

If you are authoring a new per-model doc page, use:

- [Model Detail Template](model_detail_template.md)

---

This page is best used after you already narrowed down candidate models and want to compare:

1. preprocessing assumptions
2. temporal packaging
3. side-input requirements
4. environment-variable tuning knobs

---

## How To Use This Page

### Quick chooser by goal

| Goal | Start with | Why |
|---|---|---|
| Fast baseline / simple pipeline | `tessera`, `gse`, `copernicus` | Precomputed embeddings, fewer runtime dependencies |
| General on-the-fly RGB experiments | `remoteclip`, `satmae`, `scalemae` | Simple S2 RGB input paths |
| Time-series modeling | `agrifm`, `anysat`, `galileo` | Native multi-frame temporal packaging |
| Multispectral / strict spectral semantics | `dofa`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions |
| S1/S2 modality experiments | `terrafm` | Supports S2 or S1 paths (per call) |

### Readability tips

- Start with **Quick Comparison** if you are deciding between models
- Read **Temporal Handling** and **Multi-frame Semantics** before comparing temporal models
- Read **Modality and Extra Inputs Matrix** if you need fair cross-model benchmarking
- Read **Environment Variables...** only when tuning preprocessing or reproducing training pipelines

## Precomputed Embeddings

| **Model** | **ID** | **Output** | **Resolution** | **Dim** | **Time Coverage** | **Notes** |
|---|---|---|---|---|---|---|
| **Tessera** | `tessera` | pooled / grid | 10m | 128 | 2017–2025 | GeoTessera global tile embeddings |
| **Google Satellite Embedding (Alpha Earth)** | `gse` | pooled / grid | 10 m | 64 | 2017–2024 | Annual embeddings via GEE |
| **Copernicus Embed** | `copernicus` | pooled / grid | 0.25° | 768 | 2021 | Official Copernicus embeddings |

---

## On-the-fly Foundation Models

Source of truth:

- `src/rs_embed/embedders/catalog.py`
- `src/rs_embed/embedders/onthefly_*.py`
- `src/rs_embed/embedders/_vit_mae_utils.py`
- `src/rs_embed/embedders/runtime_utils.py`

Documented on-the-fly IDs:

- `remoteclip`
- `satmae`
- `scalemae`
- `anysat`
- `galileo`
- `wildsat`
- `prithvi`
- `terrafm`
- `terramind`
- `dofa`
- `fomo`
- `thor`
- `agrifm`
- `satvision`

### Quick Comparison

| Model ID | Architecture / Backbone | Input | Default Preprocessing | Resize / Crop / Pad | Output Structure | Training Alignment |
|---|---|---|---|---|---|---|
| `remoteclip` | `rshf.remoteclip.RemoteCLIP` (open_clip style CLIP ViT) | S2 RGB (`B4,B3,B2`) | raw SR `0..10000` -> `/10000` -> RGB `uint8`; then model transform if available, else CLIP norm | image size 224; fallback path uses `Resize + CenterCrop`; no pad | pooled vector or ViT token grid | Medium (high if wrapper transform matches training; fallback is generic CLIP pipeline) |
| `satmae` | `rshf.satmae.SatMAE` | S2 RGB (`B4,B3,B2`) | raw SR -> `/10000` -> RGB `uint8`; prefer model transform, else CLIP norm | default 224; CLIP fallback has `Resize + CenterCrop`; no pad | token sequence -> pooled or patch-token grid | Medium |
| `scalemae` | `rshf.scalemae.ScaleMAE` (ViT style) | S2 RGB (`B4,B3,B2`) + `input_res_m` | raw SR -> `/10000` -> RGB `uint8`; CLIP norm tensor; pass `input_res_m` | default 224; CLIP path has `Resize + CenterCrop`; no pad | token sequence or pooled vector depending on wrapper output | Medium |
| `anysat` | AnySat from upstream `hubconf.py` (`AnySat`) | S2 10-band TCHW (or CHW auto-expanded) | clip to `0..10000`; normalize mode default `per_tile_zscore`; builds per-frame `s2_dates` | resize TCHW to default 24; no crop, no pad | patch output `[D,H,W]`, pooled by spatial mean/max | Medium |
| `galileo` | `Encoder` from official `single_file_galileo.py` | S2 10-band TCHW (or CHW auto-expanded) | clip to `0..10000`; normalize mode default `unit_scale`; constructs Galileo tensors with configurable `T` + per-frame `months`, optional NDVI channel | default 64 with patch 8; bilinear resize; no pad | pooled token vector and S2-group token grid | Medium |
| `wildsat` | WildSAT backbone + optional image head from checkpoint | S2 RGB CHW | clip to `0..10000` then `/10000`; default normalization `minmax`; convert to `uint8` then unit tensor | default 224; resize RGB; no pad | pooled branch output and optional grid (token or feature path) | Medium-Low |
| `prithvi` | TerraTorch `BACKBONE_REGISTRY` Prithvi backbone | S2 6-band (`BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`) | raw SR -> `/10000` -> clamp `[0,1]`; prep mode from env | default mode `resize` to 224; optional `pad` to patch multiple (legacy) | token sequence -> pooled or patch-token grid | Medium |
| `terrafm` | TerraFM-B from HF code/weights | S2 12-band or S1 VV/VH | S2: `/10000` to `[0,1]`; S1: `log1p` + p99 scaling to `[0,1]` | resize to 224; no pad | pooled embedding, optional feature-map grid | Medium |
| `terramind` | TerraTorch `BACKBONE_REGISTRY` TerraMind backbone | S2 SR 12-band | raw `0..10000`; resize 224; z-score with TerraMind v1/v01 pretrained mean/std | fixed 224; no pad | token sequence -> pooled or patch-token grid | High |
| `dofa` | TorchGeo DOFA (`dofa_base_patch16_224` / `dofa_large_patch16_224`) | multi-band SR CHW + wavelengths | raw SR -> `/10000` to `[0,1]`; provide/infer wavelengths | bilinear resize to 224; explicitly no crop/pad | pooled vector or token grid (usually 14x14) | Medium-High |
| `fomo` | FoMo `MultiSpectralViT` (FoMo-Bench) | S2 SR 12-band | clip `0..10000`; default `unit_scale` (optional minmax/none) | default 64; bilinear resize; no pad | token sequence pooled; grid as spectral-mean patch-token map | Medium |
| `thor` | THOR via TerraTorch + `thor_terratorch_ext` | S2 SR 10-band | clip `0..10000`; default `thor_stats` z-score after reflectance scaling | default 288; bilinear resize; no pad | pooled tokens and grouped token grid | Medium-High |
| `agrifm` | AgriFM `PretrainingSwinTransformer3DEncoder` | S2 10-band time series `[T,C,H,W]` | clip `0..10000`; default `agrifm_stats` z-score using official config stats | default 224; TCHW resize; no pad | feature map grid `[D,H,W]`, pooled by spatial mean/max | High |
| `satvision` | `timm` `SwinTransformerV2` (SatVision-TOA checkpoints) | TOA 14 channels in strict order | channel-aware normalization to `[0,1]` (`auto/raw/unit`, reflectance + emissive calibration) | default 128; bilinear resize; no pad | model output as pooled or grid depending on tensor shape | High (if band order and calibration match checkpoint) |

### Temporal Handling 

- For most on-the-fly adapters, `TemporalSpec.range(start, end)` means: filter imagery in `[start, end)`, then build one composite patch for model input (`median` by default, or `mosaic` if configured via `SensorSpec.composite`).
- In these adapters, `meta.input_time` is typically the midpoint of the temporal window and is mainly metadata (or an auxiliary time signal for models that require it), not a guaranteed single-scene acquisition date.
- Multi-frame adapters: `agrifm`, `anysat`, and `galileo` fetch TCHW sequences by splitting the requested range into sub-windows and compositing each sub-window into one frame.
- Current single-composite adapters include: `remoteclip`, `satmae`, `scalemae`, `wildsat`, `prithvi`, `terrafm`, `terramind`, `dofa`, `fomo`, `thor`, and `satvision`.

### Multi-frame Semantics

Shared behavior for current multi-frame adapters (`agrifm`, `anysat`, `galileo`):

- Frame construction: split `TemporalSpec.range(start, end)` into `T` equal sub-windows (end-exclusive), then composite each sub-window into one frame.
- Missing-observation fallback: if a sub-window has no valid image, provider path reuses a fallback composite so frame count remains stable.
- Fixed frame count: runtime always ensures exact `T` frames for model input.
  For user-provided `input_chw`, `CHW` is repeated to `T`, and `TCHW` is padded/truncated to `T`.
- Sensor compositing policy: frame composite mode follows `SensorSpec.composite` (`median` default, `mosaic` optional).

Per-model temporal packaging:

| Model ID | Frame count env (default) | Temporal side input | Notes |
|---|---|---|---|
| `agrifm` | `RS_EMBED_AGRIFM_FRAMES` (`8`) | none (uses `TCHW` directly) | Temporal information is encoded only in the frame stack. |
| `anysat` | `RS_EMBED_ANYSAT_FRAMES` (`8`) | `s2_dates` (per-frame DOY, `0..364`) | DOY values are derived from each frame bin midpoint date. |
| `galileo` | `RS_EMBED_GALILEO_FRAMES` (`8`) | `months` (per-frame month, `1..12`) | By default from frame bin midpoints; `RS_EMBED_GALILEO_MONTH` can force a constant month for all frames. |

### Modality and Extra Inputs Matrix

Interpretation:

- "Backbone multimodal" means the upstream foundation model family supports multiple modalities.
- "Current rs-embed path" means what this implementation currently feeds in practice.
- "Requires extra metadata" means additional non-image inputs required by the forward path (hard requirement).

| Model ID | Backbone multimodal? | Current rs-embed path uses multiple modalities? | Multi-input forward (beyond image tensor)? | Requires extra metadata? |
|---|---|---|---|---|
| `remoteclip` | No | No | No | No |
| `satmae` | No | No | No | No |
| `scalemae` | No | No | Yes (`input_res_m`) | Yes: scale/resolution (`sensor.scale_m`) |
| `anysat` | Yes | Partially (S2-only imagery, plus temporal date tokens) | Yes (`s2`, `s2_dates`) | Yes: day-of-year/date signal (derived from temporal range) |
| `galileo` | Yes | Mostly S2 path in current adapter + temporal month tokens | Yes (multiple tensors + masks + `months`) | Yes: month/time signal (derived from temporal range) |
| `wildsat` | No | No | No | No |
| `prithvi` | No (this adapter path) | No | Yes (`x`, `temporal_coords`, `location_coords`) | Yes: location + time are required |
| `terrafm` | Yes (`S1`/`S2`) | Yes (select one modality per call: `s1` or `s2`) | No | No hard extra metadata (optional S1 options: orbit, linear/DB path) |
| `terramind` | Yes | Usually single selected modality (`S2L2A` default) | No (single selected modality tensor in this adapter) | No hard extra metadata |
| `dofa` | Yes (spectral generalization) | Yes (multi-band spectral input) | Yes (image + wavelength list) | Yes: per-band wavelengths (explicit or inferable from bands) |
| `fomo` | No | No | No | No |
| `thor` | No (this adapter path) | No | No | No |
| `agrifm` | No (this adapter path) | No | No extra side tensor, but temporal stack `[T,C,H,W]` required | Temporal coverage is important (no separate metadata tensor) |
| `satvision` | No (this adapter path) | No | No separate side tensor | Yes: strict 14-channel order/calibration schema (band semantics) |

Practically multi-input models:

- `prithvi`: image + temporal coords + location coords
- `anysat`: image/time-series + date tokens (`s2_dates`)
- `galileo`: image-derived tensors + masks + per-frame month tokens (`months`)
- `dofa`: image + wavelength vector
- `scalemae`: image + `input_res_m`

### Environment Variables That Directly Change Preprocessing/Temporal Packaging

| Model ID | Main preprocessing env keys |
|---|---|
| `remoteclip` | fixed `image_size=224` in code path; no per-model preprocess env switch |
| `satmae` | `RS_EMBED_SATMAE_IMG` |
| `scalemae` | `RS_EMBED_SCALEMAE_IMG` |
| `anysat` | `RS_EMBED_ANYSAT_IMG`, `RS_EMBED_ANYSAT_NORM`, `RS_EMBED_ANYSAT_FRAMES` |
| `galileo` | `RS_EMBED_GALILEO_IMG`, `RS_EMBED_GALILEO_PATCH`, `RS_EMBED_GALILEO_NORM`, `RS_EMBED_GALILEO_INCLUDE_NDVI`, `RS_EMBED_GALILEO_FRAMES`, `RS_EMBED_GALILEO_MONTH` |
| `wildsat` | `RS_EMBED_WILDSAT_IMG`, `RS_EMBED_WILDSAT_NORM` |
| `prithvi` | `RS_EMBED_PRITHVI_PREP`, `RS_EMBED_PRITHVI_IMG`, `RS_EMBED_PRITHVI_PATCH_MULT` |
| `terrafm` | modality and sensor-side options (`s2`/`s1`); image size fixed to 224 in implementation |
| `terramind` | `RS_EMBED_TERRAMIND_NORMALIZE` (default z-score stats), image size fixed 224 |
| `dofa` | image size fixed 224; provider/tensor channels and wavelengths drive preprocessing |
| `fomo` | `RS_EMBED_FOMO_IMG`, `RS_EMBED_FOMO_NORM` |
| `thor` | `RS_EMBED_THOR_IMG`, `RS_EMBED_THOR_NORMALIZE` |
| `agrifm` | `RS_EMBED_AGRIFM_IMG`, `RS_EMBED_AGRIFM_NORM`, `RS_EMBED_AGRIFM_FRAMES` |
| `satvision` | `RS_EMBED_SATVISION_TOA_IMG`, `RS_EMBED_SATVISION_TOA_NORM`, channel-index and calibration env keys |

### Practical Guidance

- For highest reproducibility, keep each model's default normalization mode unless you can match the original training pipeline exactly.
- For strict-schema models (`satvision`, `terramind`, `thor`, `agrifm`), do not change channel order unless checkpoint metadata explicitly allows it.
- If comparing embeddings across models, standardize ROI and temporal compositing first; model preprocessing differences are substantial.
