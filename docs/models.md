# Supported Models (Overview)

This page is the model selection entry point.
Use it to answer one question quickly: which model IDs should I shortlist for this task?

Once you have a shortlist, use [Advanced Model Reference](models_reference.md) for side-by-side preprocessing and temporal details, then open the linked detail page for the exact contract, caveats, and examples.
If you are about to change `input_prep`, `variant`, fetch resolution, patch size, or image size, read [Before You Start](choosing_settings.md) first, because those knobs affect both runtime cost and embedding semantics.

---

## How To Read This Page

Start with the quick chooser, then scan the catalog table for input and temporal fit, and open the detail page before benchmarking or production use.

Canonical model IDs use the short public names shown on this page, such as `remoteclip`, `prithvi`, `terrafm`, and `thor`.
Some detail-page filenames still use older names for compatibility, but the canonical IDs above are the names users should copy into code.

---

## Quick Chooser by Goal

| Goal                                      | Good starting models                                        | Why                                                |
| ----------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
| Fast baseline / simple pipeline           | `tessera`, `gse`, `copernicus`                              | Precomputed embeddings, fewer runtime dependencies |
| Simple S2 RGB on-the-fly experiments      | `remoteclip`, `satmae`, `satmaepp`, `scalemae`              | Straightforward RGB input paths                    |
| Time-series temporal modeling             | `prithvi`, `olmoearth`, `galileo`, `anysat`, `agrifm`       | Native multi-frame temporal packaging — see [Temporal Sampling](temporal_sampling.md) |
| Multispectral / strict spectral semantics | `satmaepp` (`modality="s2_10b"`), `dofa`, `clay`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions                  |
| Mixed-modality experiments (S1/S2)        | `terrafm`, `thor`                                           | Supports S2 or S1 path (per call)                  |

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID     | Type        | Primary Input / Source              | Default Resolution | Dim  | Temporal mode            | Notes                                                                      | Detail                         |
| ------------ | ----------- | ----------------------------------- | ------------------ | ---- | ------------------------ | -------------------------------------------------------------------------- | ------------------------------ |
| `tessera`    | Precomputed | GeoTessera embedding tiles          | 10m                | 128  | yearly coverage product  | Fast baseline, source-fixed precomputed workflow; product-native fixed CRS | [detail](models/tessera.md)    |
| `gse`        | Precomputed | Google Satellite Embedding (annual) | 10m                | 64   | `TemporalSpec.year(...)` | Annual product via provider path                                           | [detail](models/gse.md)        |
| `copernicus` | Precomputed | Copernicus embeddings               | 0.25°              | 768  | limited (2021)           | Coarse resolution product on fixed EPSG:4326 grid                          | [detail](models/copernicus.md) |

### On-the-fly Foundation Models

| Model ID          | Primary Input                    | Dim  | Default Resolution | Input size (px) | Temporal style          | Notable requirements                                    | Detail                         |
| ----------------- | -------------------------------- | ---- | ------------------ | --------------- | ----------------------- | ------------------------------------------------------- | ------------------------------ |
| `prithvi`         | S2 6-band                       | 768  | 30m                | 224             | multi-frame (auto, ≤4)  | required temporal + location side inputs                 | [detail](models/prithvi.md)    |
| `olmoearth`       | S2 L2A 12-band / S1 VV/VH       | 128–1024 | 10m            | 256 (flexible)  | multi-frame (auto, ≤12) | FlexiViT; 4 sizes (nano/tiny/base/large) | [detail](models/olmoearth.md) |
| `dofa`            | Multispectral + wavelengths     | 768  | 10m                | 224             | single composite        | wavelength vector required                              | [detail](models/dofa.md)       |
| `clay`            | S2 L2A 10-band                  | 1024 | 10m                | 256             | single composite        | metadata conditioning (latlon/time/gsd/wavelengths)     | [detail](models/clay.md)       |
| `terramind`       | S2 12-band                      | 384  | 10m                | 224             | single composite        | ViT-S class; strict z-score normalization               | [detail](models/terramind.md)  |
| `terrafm`         | S2 12-band or S1 VV/VH          | 768  | 10m                | 224             | single composite        | dual-modality by channel count                          | [detail](models/terrafm.md)    |
| `thor`            | S2 10-band or S1 VV/VH          | 768  | 10m                | 288             | single composite        | dual-modality; grouped tokens; native-snap              | [detail](models/thor.md)       |
| `galileo`         | S2 10-band time series          | 128  | 10m                | 64              | multi-frame (auto, ≤12) | nano default; month tokens                              | [detail](models/galileo.md)    |
| `anysat`          | S2 10-band time series          | 768  | 10m                | 24              | multi-frame (fixed `T`) | JEPA; `s2_dates` DOY side input                         | [detail](models/anysat.md)     |
| `agrifm`          | S2 10-band time series          | 1024 | 10m                | 224             | multi-frame (fixed `T`) | Video Swin; fixed `T` frame stack                       | [detail](models/agrifm.md)     |
| `fomo`            | S2 12-band                      | 768  | 10m                | 64              | single composite        | per-channel spectral modality keys                      | [detail](models/fomo.md)       |
| `wildsat`         | S2 RGB                          | 256  | 10m                | 224             | single composite        | biodiversity training; image_head default               | [detail](models/wildsat.md)    |
| `satvision`       | TOA 14-channel (MODIS)          | 4096 | 1000m              | 128             | single composite        | SwinV2 Giant; strict channel calibration                | [detail](models/satvision.md)  |
| `remoteclip`      | S2 RGB (`B4,B3,B2`)             | 512  | 10m                | 224             | single composite        | CLIP projection; RGB preprocessing                      | [detail](models/remoteclip.md) |
| `scalemae`        | S2 RGB + scale                  | 1024 | 10m                | 224             | single composite        | `sensor.scale_m` is a model input                       | [detail](models/scalemae.md)   |
| `satmae`          | S2 RGB (`B4,B3,B2`)             | 1024 | 10m                | 224             | single composite        | ViT-L; MAE token/grid                                   | [detail](models/satmae.md)     |
| `satmaepp`        | S2 RGB (`B4,B3,B2`) or S2 10-band | 1024 | 10m              | 224 (rgb) / 96 (s2_10b) | single composite | `modality=rgb` (default) or `s2_10b`; ViT-L; fMoW eval preprocessing; 10-band uses strict band order + grouped-channel tokens | [detail](models/satmaepp.md)   |

**Input size (px)** is the fixed spatial size each model's encoder consumes: inputs are resized to it before the forward pass (in the default `input_prep="tile"` fetch mode, large ROIs are instead cut into tiles of this size at native resolution and the grids stitched; user-provided data is always resized — see [User Data API](user_data.md)). Together with Default Resolution it gives the native footprint of one forward pass, e.g. galileo 64 px × 10 m ≈ 640 m. `olmoearth` (FlexiViT) accepts any size divisible by its patch size and manages its own tiling; 256 is its training tile size. `anysat` and `prithvi` sizes are env-tunable (`RS_EMBED_ANYSAT_IMG`, `RS_EMBED_PRITHVI_IMG`).

---

## Temporal and Comparison Notes (What People Usually Miss)

`TemporalSpec.range(start, end)` is usually a compositing window rather than a single-scene selector, and `OutputSpec.grid()` may be a token or patch grid rather than a georeferenced raster, especially for ViT-like backbones. Cross-model comparisons are usually easiest with `OutputSpec.pooled()` plus fixed ROI, temporal, and compositing settings.

Precomputed products can also keep their own product-native projection instead of the common provider-backed EPSG:3857 sampling grid. Today that matters especially for `tessera` and `copernicus`, so check each detail page before comparing grid outputs directly against on-the-fly models.

On this page, "Default Resolution" means the default source-side fetch resolution, not the final resized tensor shape sent into the backbone. Multi-frame models such as `prithvi`, `olmoearth`, `galileo`, `anysat`, and `agrifm` also need extra attention to frame count and temporal side inputs — how each one turns a `TemporalSpec.range` into frames is summarized in [Temporal Sampling](temporal_sampling.md).

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## More Detail

For cross-model preprocessing, temporal packaging, and environment knobs, continue to [Advanced Model Reference](models_reference.md). For user-facing guidance on how to trade compute for quality, spatial detail, or temporal fidelity, read [Before You Start](choosing_settings.md). If you are adding a new adapter, use [Extending](extending.md) to keep the implementation and documentation consistent.
