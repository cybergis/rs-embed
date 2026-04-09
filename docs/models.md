# Supported Models (Overview)

This page is the model selection entry point.
Use it to answer one question quickly: which model IDs should I shortlist for this task?

Once you have a shortlist, use [Advanced Model Reference](models_reference.md) for side-by-side preprocessing and temporal details, then open the linked detail page for the exact contract, caveats, and examples.

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
| Mixed-modality experiments (S1/S2)        | `terrafm`                                                   | Supports S2 or S1 path (per call)                  |

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID     | Type        | Primary Input / Source              | Default Resolution | Outputs          | Temporal mode            | Notes                                                                      | Detail                         |
| ------------ | ----------- | ----------------------------------- | ------------------ | ---------------- | ------------------------ | -------------------------------------------------------------------------- | ------------------------------ |
| `tessera`    | Precomputed | GeoTessera embedding tiles          | 10m                | `pooled`, `grid` | yearly coverage product  | Fast baseline, source-fixed precomputed workflow; product-native fixed CRS | [detail](models/tessera.md)    |
| `gse`        | Precomputed | Google Satellite Embedding (annual) | 10m                | `pooled`, `grid` | `TemporalSpec.year(...)` | Annual product via provider path                                           | [detail](models/gse.md)        |
| `copernicus` | Precomputed | Copernicus embeddings               | 0.25°              | `pooled`, `grid` | limited (2021)           | Coarse resolution product on fixed EPSG:4326 grid                          | [detail](models/copernicus.md) |

### On-the-fly Foundation Models

| Model ID          | Primary Input                                      | Default Resolution | Temporal style          | Outputs          | Notable requirements                                          | Detail                         |
| ----------------- | -------------------------------------------------- | ------------------ | ----------------------- | ---------------- | ------------------------------------------------------------- | ------------------------------ |
| `remoteclip`      | S2 RGB (`B4,B3,B2`)                                | 10m                | single composite window | `pooled`, `grid` | provider backend; RGB preprocessing                           | [detail](models/remoteclip.md) |
| `satmae`          | S2 RGB (`B4,B3,B2`)                                | 10m                | single composite window | `pooled`, `grid` | RGB path; ViT token/grid behavior                             | [detail](models/satmae.md)     |
| `satmaepp`        | S2 RGB (`B4,B3,B2`)                                | 10m                | single composite window | `pooled`, `grid` | SatMAE++ fMoW-style eval preprocessing; channel order control | [detail](models/satmaepp.md)   |
| `satmaepp_s2_10b` | S2 SR 10-band (`B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`) | 10m                | single composite window | `pooled`, `grid` | strict 10-band order; grouped-channel token handling          | [detail](models/satmaepp.md)   |
| `scalemae`        | S2 RGB + scale                                     | 10m                | single composite window | `pooled`, `grid` | requires `sensor.scale_m` / `input_res_m`                     | [detail](models/scalemae.md)   |
| `wildsat`         | S2 RGB                                             | 10m                | single composite window | `pooled`, `grid` | normalization options                                         | [detail](models/wildsat.md)    |
| `prithvi`         | S2 6-band                                          | 30m                | single composite window | `pooled`, `grid` | required temporal + location side inputs                      | [detail](models/prithvi.md)    |
| `terrafm`         | S2 12-band or S1 VV/VH                             | 10m                | single composite window | `pooled`, `grid` | choose modality per call                                      | [detail](models/terrafm.md)    |
| `terramind`       | S2 SR 12-band                                      | 10m                | single composite window | `pooled`, `grid` | strict normalization/channel semantics                        | [detail](models/terramind.md)  |
| `dofa`            | Multispectral + wavelengths                        | 10m                | single composite window | `pooled`, `grid` | wavelength vector required/inferred                           | [detail](models/dofa.md)       |
| `fomo`            | S2 12-band                                         | 10m                | single composite window | `pooled`, `grid` | normalization mode choices                                    | [detail](models/fomo.md)       |
| `thor`            | S2 SR 10-band                                      | 10m                | single composite window | `pooled`, `grid` | THOR stats normalization; bounded native-snap for near-square inputs; tile for large ROIs | [detail](models/thor.md)       |
| `satvision`       | TOA 14-channel                                     | 1000m              | single composite window | `pooled`, `grid` | strict channel order + calibration                            | [detail](models/satvision.md)  |
| `anysat`          | S2 10-band time series                             | 10m                | multi-frame             | `pooled`, `grid` | frame dates (`s2_dates`) side input                           | [detail](models/anysat.md)     |
| `galileo`         | S2 10-band time series                             | 10m                | multi-frame             | `pooled`, `grid` | month tokens + grouped tensors                                | [detail](models/galileo.md)    |
| `agrifm`          | S2 10-band time series                             | 10m                | multi-frame             | `pooled`, `grid` | fixed `T` frame stack behavior                                | [detail](models/agrifm.md)     |

---

## Temporal and Comparison Notes (What People Usually Miss)

`TemporalSpec.range(start, end)` is usually a compositing window rather than a single-scene selector, and `OutputSpec.grid()` may be a token or patch grid rather than a georeferenced raster, especially for ViT-like backbones. Cross-model comparisons are usually easiest with `OutputSpec.pooled()` plus fixed ROI, temporal, and compositing settings.

Precomputed products can also keep their own product-native projection instead of the common provider-backed EPSG:3857 sampling grid. Today that matters especially for `tessera` and `copernicus`, so check each detail page before comparing grid outputs directly against on-the-fly models.

On this page, "Default Resolution" means the default source-side fetch resolution, not the final resized tensor shape sent into the backbone. Multi-frame models such as `agrifm`, `anysat`, and `galileo` also need extra attention to frame count and temporal side inputs.

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## More Detail

For cross-model preprocessing, temporal packaging, and environment knobs, continue to [Advanced Model Reference](models_reference.md). If you are adding a new adapter, use [Extending](extending.md) to keep the implementation and documentation consistent.
