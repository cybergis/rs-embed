# Supported Models (Advanced Reference)

This page is for **cross-model comparison after you already have a shortlist**.

If you are choosing a model for the first time, start with [Supported Models (Overview)](models.md).

If you need the exact contract for one specific model, use the per-model detail pages in **Reference -> Model Details**.

If you are authoring a new per-model doc page, use [Model Detail Template](model_detail_template.md).

---

Use this page to compare preprocessing assumptions, temporal packaging, side-input requirements, and the environment variables that materially change model behavior. The main sections are [Precomputed Embeddings](#precomputed-embeddings), [Quick Comparison](#quick-comparison), [Temporal Handling](#temporal-handling), [Multi-frame Semantics](#multi-frame-semantics), [Modality and Extra Inputs Matrix](#modality-and-extra-inputs-matrix), and [Preprocessing and Temporal Env Vars](#preprocessing-and-temporal-env-vars).

---

## How To Use This Page

### Reading tips

Start with **Quick Comparison** if you are deciding between models. Read **Temporal Handling** and **Multi-frame Semantics** before comparing temporal models, and use **Modality and Extra Inputs Matrix** when you need a fair benchmark across models with different side inputs. **Preprocessing and Temporal Env Vars** matters mainly when you are tuning preprocessing or reproducing training pipelines.

Canonical model IDs in this page use the short public names from `MODEL_SPECS`, such as `remoteclip`, `prithvi`, `terrafm`, and `thor`.
Some linked detail-page filenames still retain older names for compatibility.

## Precomputed Embeddings

| **Model**                                    | **ID**       | **Output**    | **Resolution** | **Dim** | **Time Coverage** | **Notes**                         |
| -------------------------------------------- | ------------ | ------------- | -------------- | ------- | ----------------- | --------------------------------- |
| **Tessera**                                  | `tessera`    | pooled / grid | 10m            | 128     | 2017–2025         | GeoTessera global tile embeddings |
| **Google Satellite Embedding (Alpha Earth)** | `gse`        | pooled / grid | 10 m           | 64      | 2017–2024         | Annual embeddings via GEE         |
| **Copernicus Embed**                         | `copernicus` | pooled / grid | 0.25°          | 768     | 2021              | Official Copernicus embeddings    |

---

## On-the-fly Foundation Models

The source of truth for this section is the adapter code in `src/rs_embed/embedders/catalog.py`, `src/rs_embed/embedders/onthefly_*.py`, `src/rs_embed/embedders/_vit_mae_utils.py`, and `src/rs_embed/embedders/runtime_utils.py`.

### Quick Comparison

Use this table for a first-pass side-by-side comparison of input assumptions and preprocessing behavior.

| Model ID | Architecture / Backbone | Default Fetch Resolution | Input | Default Preprocessing | Resize / Crop / Pad | Output Structure | Training Alignment |
|---|---|---|---|---|---|---|---|
| `remoteclip` | `rshf.remoteclip.RemoteCLIP` (open_clip style CLIP ViT) | 10m | S2 RGB (`B4,B3,B2`) | raw SR `0..10000` -> `/10000` -> RGB `uint8`; then model transform if available, else CLIP norm | image size 224; fallback path uses `Resize + CenterCrop`; no pad | pooled vector or ViT token grid | Medium (high if wrapper transform matches training; fallback is generic CLIP pipeline) |
| `satmae` | `rshf.satmae.SatMAE` | 10m | S2 RGB (`B4,B3,B2`) | raw SR -> `/10000` -> RGB `uint8`; prefer model transform, else CLIP norm | default 224; CLIP fallback has `Resize + CenterCrop`; no pad | token sequence -> pooled or patch-token grid | Medium |
| `satmaepp` | `rshf.satmaepp.SatMAEPP` | 10m | S2 RGB (`B4,B3,B2`) | raw SR -> `/10000` -> RGB `uint8`; SatMAE++ fMoW eval preprocessing (`Normalize + Resize(short side) + CenterCrop`), default channel order `rgb` | default 224; source-aligned short-side resize + center crop; no pad | token sequence -> pooled or patch-token grid | High |
| `satmaepp_s2_10b` | SatMAE++ grouped-channel source branch (`models_mae_group_channels.py`, `base` / `large` runtime families) | 10m | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) | raw 10-band CHW; source Sentinel min/max mapping to `uint8`; `ToTensor + Resize(short side) + CenterCrop` | default 96 with patch size 8; source-style resize/crop; no pad | grouped token sequence -> pooled or group-reduced spatial token grid | High |
| `scalemae` | `rshf.scalemae.ScaleMAE` (ViT style) | 10m | S2 RGB (`B4,B3,B2`) + `input_res_m` | raw SR -> `/10000` -> RGB `uint8`; ImageNet eval normalize; derive effective `input_res_m` after preprocess | default 224; `Resize(short side) + CenterCrop`; no pad | token sequence or pooled vector depending on wrapper output | Medium-High |
| `anysat` | AnySat from upstream `hubconf.py` (`AnySat`, `tiny` / `small` / `base`) | 10m | S2 10-band TCHW (or CHW auto-expanded) | clip to `0..10000`; normalize mode default `per_tile_zscore`; builds per-frame `s2_dates` | resize TCHW to default 24; no crop, no pad | grid defaults to dense sub-patch output `[D,H,W]`; pooled defaults to patch-grid mean/max, optional native tile vector | Medium |
| `galileo` | `Encoder` from official `single_file_galileo.py` | 10m | S2 10-band TCHW (or CHW auto-expanded) | clip to `0..10000`; normalize mode default `none` with optional `official_stats`; constructs Galileo tensors with configurable `T` + per-frame `months` | default 64 with patch 8; bilinear resize; no pad | pooled token vector and official-style patch-mean token grid | Medium |
| `wildsat` | WildSAT backbone + optional image head from checkpoint | 10m | S2 RGB CHW | clip to `0..10000` then `/10000`; default normalization `minmax`; convert to `uint8` then unit tensor | default 224; resize RGB; no pad | pooled branch output and optional grid (token or feature path) | Medium-Low |
| `prithvi` | Vendored `PrithviMAE` runtime with HF checkpoints | 30m | S2 6-band (`BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`) | raw SR -> `/10000` -> clamp `[0,1]`; prep mode from env | default mode `resize` to 224; optional `pad` to patch multiple (legacy) | token sequence -> pooled or patch-token grid | Medium |
| `terrafm` | TerraFM-B from vendored runtime + HF weights | 10m | S2 12-band or S1 VV/VH | S2: `/10000` to `[0,1]`; S1: `log1p` + p99 scaling to `[0,1]` | resize to 224; no pad | pooled embedding, optional feature-map grid | Medium |
| `terramind` | TerraTorch `BACKBONE_REGISTRY` TerraMind backbone | 10m | S2 SR 12-band | raw `0..10000`; resize 224; z-score with TerraMind v1/v01 pretrained mean/std | fixed 224; no pad | token sequence -> pooled or patch-token grid | High |
| `dofa` | DOFA ViT (`base` / `large`, official checkpoints) | 10m | multi-band SR CHW + wavelengths | raw SR `0..10000` -> `0..255`-like scale -> official per-band mean/std; provide/infer wavelengths | bilinear resize to 224; explicitly no crop/pad | pooled vector or token grid (usually 14x14) | Medium-High |
| `fomo` | FoMo `MultiSpectralViT` (FoMo-Bench) | 10m | S2 SR 12-band | clip `0..10000`; default `unit_scale` (optional minmax/none) | default 64; bilinear resize; no pad | token sequence pooled; grid as spectral-mean patch-token map | Medium |
| `thor` | Fully vendored THOR runtime (`tiny` / `small` / `base` / `large`) | 10m | S2 SR 10-band | clip `0..10000`; default `thor_stats` z-score after reflectance scaling | default 288; bilinear resize; no pad | pooled tokens and grouped token grid | Medium-High |
| `agrifm` | AgriFM `PretrainingSwinTransformer3DEncoder` | 10m | S2 10-band time series `[T,C,H,W]` | clip `0..10000`; default `agrifm_stats` z-score using official config stats | default 224; TCHW resize; no pad | feature map grid `[D,H,W]`, pooled by spatial mean/max | High |
| `satvision` | `timm` `SwinTransformerV2` (SatVision-TOA checkpoints) | 1000m | TOA 14 channels in strict order | channel-aware normalization to `[0,1]` (`auto/raw/unit`, reflectance + emissive calibration) | default 128; bilinear resize; no pad | model output as pooled or grid depending on tensor shape | High (if band order and calibration match checkpoint) |

Here, "Default Fetch Resolution" refers to the default source-side resolution used when fetching raw inputs. It does not mean the final spatial size of the tensor after model-specific resize, crop, or pad.

### Temporal Handling

Read this section before comparing any model that accepts `TemporalSpec.range(...)`.

For most on-the-fly adapters, `TemporalSpec.range(start, end)` means "filter imagery in `[start, end)` and build one composite patch for model input," usually with `median` and optionally `mosaic` through `SensorSpec.composite`.

The multi-frame adapters `agrifm`, `anysat`, and `galileo` instead split the requested range into sub-windows and composite one frame per bin. Current single-composite adapters include `remoteclip`, `satmae`, `satmaepp`, `satmaepp_s2_10b`, `scalemae`, `wildsat`, `prithvi`, `terrafm`, `terramind`, `dofa`, `fomo`, `thor`, and `satvision`.

### Multi-frame Semantics

This section only matters for adapters that construct multi-frame inputs from one requested time window.

Shared behavior for current multi-frame adapters (`agrifm`, `anysat`, `galileo`):

All three split `TemporalSpec.range(start, end)` into `T` equal end-exclusive sub-windows and composite each sub-window into one frame. If a sub-window has no valid observation, the provider path reuses a fallback composite so frame count stays stable. The runtime always enforces exactly `T` frames; for user-provided inputs that means `CHW` is repeated to `T` and `TCHW` is padded or truncated to `T`. Frame compositing follows `SensorSpec.composite`, with `median` as the default and `mosaic` as the main alternative.

Per-model temporal packaging:

| Model ID  | Frame count env (default)       | Temporal side input                  | Notes                                                                                                    |
| --------- | ------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `agrifm`  | `RS_EMBED_AGRIFM_FRAMES` (`8`)  | none (uses `TCHW` directly)          | Temporal information is encoded only in the frame stack.                                                 |
| `anysat`  | `RS_EMBED_ANYSAT_FRAMES` (`8`)  | `s2_dates` (per-frame DOY, `0..364`) | DOY values are derived from each frame bin midpoint date.                                                |
| `galileo` | `RS_EMBED_GALILEO_FRAMES` (`8`) | `months` (per-frame month, `1..12`)  | By default from frame bin midpoints; `RS_EMBED_GALILEO_MONTH` can force a constant month for all frames. |

### Modality and Extra Inputs Matrix

Use this table to avoid unfair comparisons between plain image encoders and adapters that require side inputs.

Interpretation:

"Backbone multimodal" means the upstream model family supports multiple modalities. "Current rs-embed path" means what this implementation actually feeds today. "Requires extra metadata" means the forward path needs non-image inputs as a hard requirement.

| Model ID          | Backbone multimodal?          | Current rs-embed path uses multiple modalities?           | Multi-input forward (beyond image tensor)?                    | Requires extra metadata?                                            |
| ----------------- | ----------------------------- | --------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| `remoteclip`      | No                            | No                                                        | No                                                            | No                                                                  |
| `satmae`          | No                            | No                                                        | No                                                            | No                                                                  |
| `satmaepp`        | No                            | No                                                        | No                                                            | No                                                                  |
| `satmaepp_s2_10b` | No (this adapter path)        | No                                                        | No                                                            | No (but strict 10-band order is required)                           |
| `scalemae`        | No                            | No                                                        | Yes (`input_res_m`)                                           | Yes: scale/resolution (`sensor.scale_m`)                            |
| `anysat`          | Yes                           | Partially (S2-only imagery, plus temporal date tokens)    | Yes (`s2`, `s2_dates`)                                        | Yes: day-of-year/date signal (derived from temporal range)          |
| `galileo`         | Yes                           | Mostly S2 path in current adapter + temporal month tokens | Yes (multiple tensors + masks + `months`)                     | Yes: month/time signal (derived from temporal range)                |
| `wildsat`         | No                            | No                                                        | No                                                            | No                                                                  |
| `prithvi`         | No (this adapter path)        | No                                                        | Yes (`x`, `temporal_coords`, `location_coords`)               | Yes: location + time are required                                   |
| `terrafm`         | Yes (`S1`/`S2`)               | Yes (select one modality per call: `s1` or `s2`)          | No                                                            | No hard extra metadata (optional S1 options: orbit, linear/DB path) |
| `terramind`       | Yes                           | Usually single selected modality (`S2L2A` default)        | No (single selected modality tensor in this adapter)          | No hard extra metadata                                              |
| `dofa`            | Yes (spectral generalization) | Yes (multi-band spectral input)                           | Yes (image + wavelength list)                                 | Yes: per-band wavelengths (explicit or inferable from bands)        |
| `fomo`            | No                            | No                                                        | No                                                            | No                                                                  |
| `thor`            | No (this adapter path)        | No                                                        | No                                                            | No                                                                  |
| `agrifm`          | No (this adapter path)        | No                                                        | No extra side tensor, but temporal stack `[T,C,H,W]` required | Temporal coverage is important (no separate metadata tensor)        |
| `satvision`       | No (this adapter path)        | No                                                        | No separate side tensor                                       | Yes: strict 14-channel order/calibration schema (band semantics)    |

In practice, the most obviously multi-input models here are `prithvi` (image plus temporal and location coordinates), `anysat` (time series plus `s2_dates`), `galileo` (image-derived tensors plus masks and `months`), `dofa` (image plus wavelengths), and `scalemae` (image plus `input_res_m`).

### Preprocessing and Temporal Env Vars

This table only lists env vars that materially change model input construction or temporal packaging.

| Model ID          | Main preprocessing env keys                                                                                                                                                                                                            |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `remoteclip`      | fixed `image_size=224` in code path; no per-model preprocess env switch                                                                                                                                                                |
| `satmae`          | `RS_EMBED_SATMAE_IMG`                                                                                                                                                                                                                  |
| `satmaepp`        | `RS_EMBED_SATMAEPP_ID`, `RS_EMBED_SATMAEPP_IMG`, `RS_EMBED_SATMAEPP_CHANNEL_ORDER`, `RS_EMBED_SATMAEPP_BGR`                                                                                                                            |
| `satmaepp_s2_10b` | `RS_EMBED_SATMAEPP_S2_CKPT_REPO`, `RS_EMBED_SATMAEPP_S2_CKPT_FILE`, `RS_EMBED_SATMAEPP_S2_MODEL_FN`, `RS_EMBED_SATMAEPP_S2_IMG`, `RS_EMBED_SATMAEPP_S2_PATCH`, `RS_EMBED_SATMAEPP_S2_GRID_REDUCE`, `RS_EMBED_SATMAEPP_S2_WEIGHTS_ONLY` |
| `scalemae`        | `RS_EMBED_SCALEMAE_IMG`                                                                                                                                                                                                                |
| `anysat`          | `RS_EMBED_ANYSAT_IMG`, `RS_EMBED_ANYSAT_NORM`, `RS_EMBED_ANYSAT_FRAMES`, `RS_EMBED_ANYSAT_GRID_MODE`, `RS_EMBED_ANYSAT_POOLED_SOURCE`                                                                                                  |
| `galileo`         | `RS_EMBED_GALILEO_IMG`, `RS_EMBED_GALILEO_PATCH`, `RS_EMBED_GALILEO_NORM`, `RS_EMBED_GALILEO_FRAMES`, `RS_EMBED_GALILEO_MONTH`                                                                                                         |
| `wildsat`         | `RS_EMBED_WILDSAT_IMG`, `RS_EMBED_WILDSAT_NORM`                                                                                                                                                                                        |
| `prithvi`         | `RS_EMBED_PRITHVI_PREP`, `RS_EMBED_PRITHVI_IMG`, `RS_EMBED_PRITHVI_PATCH_MULT`                                                                                                                                                         |
| `terrafm`         | modality and sensor-side options (`s2`/`s1`); image size fixed to 224 in implementation                                                                                                                                                |
| `terramind`       | `RS_EMBED_TERRAMIND_NORMALIZE` (default z-score stats), image size fixed 224                                                                                                                                                           |
| `dofa`            | image size fixed 224; provider/tensor channels and wavelengths drive preprocessing                                                                                                                                                     |
| `fomo`            | `RS_EMBED_FOMO_IMG`, `RS_EMBED_FOMO_NORM`                                                                                                                                                                                              |
| `thor`            | `RS_EMBED_THOR_IMG`, `RS_EMBED_THOR_NORMALIZE`                                                                                                                                                                                         |
| `agrifm`          | `RS_EMBED_AGRIFM_IMG`, `RS_EMBED_AGRIFM_NORM`, `RS_EMBED_AGRIFM_FRAMES`                                                                                                                                                                |
| `satvision`       | `RS_EMBED_SATVISION_TOA_IMG`, `RS_EMBED_SATVISION_TOA_NORM`, channel-index and calibration env keys                                                                                                                                    |

### Practical Guidance

For highest reproducibility, keep each model's default normalization mode unless you can match the original training pipeline exactly. For strict-schema models such as `satvision`, `terramind`, `thor`, and `agrifm`, do not change channel order unless checkpoint metadata explicitly allows it. If you are comparing embeddings across models, standardize ROI and temporal compositing first, because preprocessing differences are substantial.
