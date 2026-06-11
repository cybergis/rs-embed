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
| Time-series temporal modeling             | `agrifm`, `anysat`, `galileo`                               | Native multi-frame temporal packaging              |
| Multispectral / strict spectral semantics | `satmaepp_s2_10b`, `dofa`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions                  |
| Mixed-modality experiments (S1/S2)        | `terrafm`, `thor`                                           | Supports S2 or S1 path (per call)                  |

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID     | Type        | Primary Input / Source              | Default Resolution | Dim  | Temporal mode            | Notes                                                                      | Detail                         |
| ------------ | ----------- | ----------------------------------- | ------------------ | ---- | ------------------------ | -------------------------------------------------------------------------- | ------------------------------ |
| `tessera`    | Precomputed | GeoTessera embedding tiles          | 10m                | 128  | yearly coverage product  | Fast baseline, source-fixed precomputed workflow; product-native fixed CRS | [detail](models/tessera.md)    |
| `gse`        | Precomputed | Google Satellite Embedding (annual) | 10m                | 64   | `TemporalSpec.year(...)` | Annual product via provider path                                           | [detail](models/gse.md)        |
| `copernicus` | Precomputed | Copernicus embeddings               | 0.25°              | 768  | limited (2021)           | Coarse resolution product on fixed EPSG:4326 grid                          | [detail](models/copernicus.md) |

### On-the-fly Foundation Models

| Model ID          | Primary Input                    | Dim  | Default Resolution | Temporal style   | Notable requirements                                    | Detail                         |
| ----------------- | -------------------------------- | ---- | ------------------ | ---------------- | ------------------------------------------------------- | ------------------------------ |
| `remoteclip`      | S2 RGB (`B4,B3,B2`)             | 512  | 10m                | single composite | CLIP projection; RGB preprocessing                      | [detail](models/remoteclip.md) |
| `satmae`          | S2 RGB (`B4,B3,B2`)             | 1024 | 10m                | single composite | ViT-L; MAE token/grid                                   | [detail](models/satmae.md)     |
| `satmaepp`        | S2 RGB (`B4,B3,B2`)             | 1024 | 10m                | single composite | ViT-L; fMoW eval preprocessing                          | [detail](models/satmaepp.md)   |
| `satmaepp_s2_10b` | S2 10-band                      | 1024 | 10m                | single composite | strict band order; grouped-channel tokens               | [detail](models/satmaepp.md)   |
| `scalemae`        | S2 RGB + scale                  | 1024 | 10m                | single composite | `sensor.scale_m` is a model input                       | [detail](models/scalemae.md)   |
| `wildsat`         | S2 RGB                          | 256  | 10m                | single composite | biodiversity training; image_head default               | [detail](models/wildsat.md)    |
| `prithvi`         | S2 6-band                       | 768  | 30m                | single composite | required temporal + location side inputs                 | [detail](models/prithvi.md)    |
| `terrafm`         | S2 12-band or S1 VV/VH          | 768  | 10m                | single composite | dual-modality by channel count                          | [detail](models/terrafm.md)    |
| `terramind`       | S2 12-band                      | 384  | 10m                | single composite | ViT-S class; strict z-score normalization               | [detail](models/terramind.md)  |
| `dofa`            | Multispectral + wavelengths     | 768  | 10m                | single composite | wavelength vector required                              | [detail](models/dofa.md)       |
| `fomo`            | S2 12-band                      | 768  | 10m                | single composite | per-channel spectral modality keys                      | [detail](models/fomo.md)       |
| `thor`            | S2 10-band or S1 VV/VH          | 768  | 10m                | single composite | dual-modality; grouped tokens; native-snap              | [detail](models/thor.md)       |
| `satvision`       | TOA 14-channel (MODIS)          | 4096 | 1000m              | single composite | SwinV2 Giant; strict channel calibration                | [detail](models/satvision.md)  |
| `anysat`          | S2 10-band time series          | 768  | 10m                | multi-frame      | JEPA; `s2_dates` DOY side input                         | [detail](models/anysat.md)     |
| `galileo`         | S2 10-band time series          | 128  | 10m                | multi-frame      | nano default; month tokens                              | [detail](models/galileo.md)    |
| `agrifm`          | S2 10-band time series          | 1024 | 10m                | multi-frame      | Video Swin; fixed `T` frame stack                       | [detail](models/agrifm.md)     |
| `olmoearth`       | S2 L2A 12-band                  | 128–1024 | 10m            | single composite | FlexiViT; 4 sizes (nano/tiny/base/large); requires `[olmoearth]` extra | [detail](models/olmoearth.md) |

---

## Temporal and Comparison Notes (What People Usually Miss)

`TemporalSpec.range(start, end)` is usually a compositing window rather than a single-scene selector, and `OutputSpec.grid()` may be a token or patch grid rather than a georeferenced raster, especially for ViT-like backbones. Cross-model comparisons are usually easiest with `OutputSpec.pooled()` plus fixed ROI, temporal, and compositing settings.

Precomputed products can also keep their own product-native projection instead of the common provider-backed EPSG:3857 sampling grid. Today that matters especially for `tessera` and `copernicus`, so check each detail page before comparing grid outputs directly against on-the-fly models.

On this page, "Default Resolution" means the default source-side fetch resolution, not the final resized tensor shape sent into the backbone. Multi-frame models such as `agrifm`, `anysat`, and `galileo` also need extra attention to frame count and temporal side inputs.

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## More Detail

For cross-model preprocessing, temporal packaging, and environment knobs, continue to [Advanced Model Reference](models_reference.md). For user-facing guidance on how to trade compute for quality, spatial detail, or temporal fidelity, read [Before You Start](choosing_settings.md). If you are adding a new adapter, use [Extending](extending.md) to keep the implementation and documentation consistent.
