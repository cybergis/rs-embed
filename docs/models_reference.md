# Supported Models (Advanced Reference)


## How To Use This Page

Read [Temporal Handling](#temporal-handling) and [Multi-frame Semantics](#multi-frame-semantics) before comparing temporal models. Use [Modality and Extra Inputs Matrix](#modality-and-extra-inputs-matrix) when you need a fair benchmark across models with different side inputs. [Preprocessing and Temporal Env Vars](#preprocessing-and-temporal-env-vars) matters mainly when tuning preprocessing or reproducing training pipelines.

## Precomputed Embeddings

| **Model**                                    | **ID**       | **Output**    | **Resolution** | **Dim** | **Time Coverage** | **Notes**                         |
| -------------------------------------------- | ------------ | ------------- | -------------- | ------- | ----------------- | --------------------------------- |
| **Tessera**                                  | `tessera`    | pooled / grid | 10m            | 128     | 2017–2025         | GeoTessera global tile embeddings |
| **Google Satellite Embedding (Alpha Earth)** | `gse`        | pooled / grid | 10 m           | 64      | 2017–2024         | Annual embeddings via GEE         |
| **Copernicus Embed**                         | `copernicus` | pooled / grid | 0.25°          | 768     | 2021              | Official Copernicus embeddings    |

---

## On-the-fly Foundation Models

For per-model input contracts, preprocessing pipelines, and environment variables, see each model's detail page (linked from [Models Overview](models.md)). The tables below focus on **cross-model comparison dimensions** that are hard to see from individual pages.

### Temporal Handling

Read this section before comparing any model that accepts `TemporalSpec.range(...)`.

For most on-the-fly adapters, `TemporalSpec.range(start, end)` means "filter imagery in `[start, end)` and build one composite patch for model input," usually with `median` and optionally `mosaic` through `SensorSpec.composite`.

The multi-frame adapters `agrifm`, `anysat`, and `galileo` instead split the requested range into sub-windows and composite one frame per bin. Current single-composite adapters include `remoteclip`, `satmae`, `satmaepp`, `satmaepp_s2_10b`, `scalemae`, `wildsat`, `prithvi`, `terrafm`, `terramind`, `dofa`, `fomo`, `thor`, `satvision`, and `olmoearth`.

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

!!! info "Interpretation"
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
| `thor`            | Yes (`S1`/`S2`)               | Yes (select one modality per call: `s1` or `s2`)          | No                                                            | No hard extra metadata (optional S1 options: orbit, linear/DB path) |
| `agrifm`          | No (this adapter path)        | No                                                        | No extra side tensor, but temporal stack `[T,C,H,W]` required | Temporal coverage is important (no separate metadata tensor)        |
| `satvision`       | No (this adapter path)        | No                                                        | No separate side tensor                                       | Yes: strict 14-channel order/calibration schema (band semantics)    |
| `olmoearth`       | Yes (multi-modal architecture) | S2 L2A only in this adapter                              | Yes (image + mask + timestamps; all derived automatically)    | No hard extra metadata (timestamps derived from temporal midpoint)  |

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
| `thor`            | `RS_EMBED_THOR_IMG`, `RS_EMBED_THOR_NORMALIZE`, plus modality and sensor-side options (`s2`/`s1`)                                                                                                                                      |
| `agrifm`          | `RS_EMBED_AGRIFM_IMG`, `RS_EMBED_AGRIFM_NORM`, `RS_EMBED_AGRIFM_FRAMES`                                                                                                                                                                |
| `satvision`       | `RS_EMBED_SATVISION_TOA_IMG`, `RS_EMBED_SATVISION_TOA_NORM`, channel-index and calibration env keys                                                                                                                                    |
| `olmoearth`       | `RS_EMBED_OLMOEARTH_VARIANT`, `RS_EMBED_OLMOEARTH_IMAGE_SIZE`, `RS_EMBED_OLMOEARTH_PATCH_SIZE`                                                                                                                                          |

### Practical Guidance

For highest reproducibility, keep each model's default normalization mode unless you can match the original training pipeline exactly. For strict-schema models such as `satvision`, `terramind`, `thor`, and `agrifm`, do not change channel order unless checkpoint metadata explicitly allows it. If you are comparing embeddings across models, standardize ROI and temporal compositing first, because preprocessing differences are substantial.
