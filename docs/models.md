# Supported Models (Overview)

This page is the **model selection entry point**.

Use it to quickly narrow down candidates, then jump to:

- [Supported Models (Advanced Reference)](models_reference.md) for detailed matrices (preprocessing / temporal semantics / env vars)
- [Model Detail Template](model_detail_template.md) to create a per-model documentation page

---

## How To Use This Page

Recommended flow:

1. Pick a shortlist from the overview tables below
2. Validate temporal + input assumptions in [Advanced Reference](models_reference.md)
3. For production/benchmark runs, confirm model-specific caveats and env vars before exporting

---

## Quick Chooser by Goal

| Goal | Good starting models | Why |
|---|---|---|
| Fast baseline / simple pipeline | `tessera`, `gse`, `copernicus` | Precomputed embeddings, fewer runtime dependencies |
| Simple S2 RGB on-the-fly experiments | `remoteclip`, `satmae`, `scalemae` | Straightforward RGB input paths |
| Time-series temporal modeling | `agrifm`, `anysat`, `galileo` | Native multi-frame temporal packaging |
| Multispectral / strict spectral semantics | `dofa`, `terramind`, `thor`, `satvision` | Strong channel/schema assumptions |
| Mixed-modality experiments (S1/S2) | `terrafm` | Supports S2 or S1 path (per call) |

---

## Available Detail Pages (Current)

Canonical model IDs are the short names shown below. Legacy IDs (for example `remoteclip_s2rgb`, `prithvi_eo_v2_s2_6b`, `thor_1_0_base`) are still accepted as aliases in the API/CLI.

- [`remoteclip`](models/remoteclip_s2rgb.md)
- [`prithvi`](models/prithvi_eo_v2_s2_6b.md)
- [`anysat`](models/anysat.md)
- [`galileo`](models/galileo.md)
- [`terramind`](models/terramind.md)
- [`dofa`](models/dofa.md)
- [`satvision`](models/satvision_toa.md)
- [`thor`](models/thor_1_0_base.md)
- [`agrifm`](models/agrifm.md)
- [`satmae`](models/satmae_rgb.md)
- [`scalemae`](models/scalemae_rgb.md)
- [`wildsat`](models/wildsat.md)
- [`fomo`](models/fomo.md)
- [`terrafm`](models/terrafm_b.md)
- [`tessera`](models/tessera.md)
- [`gse`](models/gse_annual.md)
- [`copernicus`](models/copernicus_embed.md)

More model detail pages can be added using the [Model Detail Template](model_detail_template.md).

---

## Model Catalog Snapshot

### Precomputed Embeddings

| Model ID | Type | Primary Input / Source | Outputs | Temporal mode | Notes | Detail |
|---|---|---|---|---|---|---|
| `tessera` | Precomputed | GeoTessera embedding tiles | `pooled`, `grid` | yearly coverage product | Fast baseline, source-fixed precomputed workflow | [detail](models/tessera.md) |
| `gse` | Precomputed | Google Satellite Embedding (annual) | `pooled`, `grid` | `TemporalSpec.year(...)` | Annual product via provider path | [detail](models/gse_annual.md) |
| `copernicus` | Precomputed | Copernicus embeddings | `pooled`, `grid` | limited (2021) | Coarse resolution product | [detail](models/copernicus_embed.md) |

### On-the-fly Foundation Models

| Model ID | Primary Input | Temporal style | Outputs | Notable requirements | Detail |
|---|---|---|---|---|---|
| `remoteclip` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | provider backend; RGB preprocessing | [detail](models/remoteclip_s2rgb.md) |
| `satmae` | S2 RGB (`B4,B3,B2`) | single composite window | `pooled`, `grid` | RGB path; ViT token/grid behavior | [detail](models/satmae_rgb.md) |
| `scalemae` | S2 RGB + scale | single composite window | `pooled`, `grid` | requires `sensor.scale_m` / `input_res_m` | [detail](models/scalemae_rgb.md) |
| `wildsat` | S2 RGB | single composite window | `pooled`, `grid` | normalization options | [detail](models/wildsat.md) |
| `prithvi` | S2 6-band | single composite window | `pooled`, `grid` | required temporal + location side inputs | [detail](models/prithvi_eo_v2_s2_6b.md) |
| `terrafm` | S2 12-band or S1 VV/VH | single composite window | `pooled`, `grid` | choose modality per call | [detail](models/terrafm_b.md) |
| `terramind` | S2 SR 12-band | single composite window | `pooled`, `grid` | strict normalization/channel semantics | [detail](models/terramind.md) |
| `dofa` | Multispectral + wavelengths | single composite window | `pooled`, `grid` | wavelength vector required/inferred | [detail](models/dofa.md) |
| `fomo` | S2 12-band | single composite window | `pooled`, `grid` | normalization mode choices | [detail](models/fomo.md) |
| `thor` | S2 SR 10-band | single composite window | `pooled`, `grid` | strict stats-based normalization | [detail](models/thor_1_0_base.md) |
| `satvision` | TOA 14-channel | single composite window | `pooled`, `grid` | strict channel order + calibration | [detail](models/satvision_toa.md) |
| `anysat` | S2 10-band time series | multi-frame | `pooled`, `grid` | frame dates (`s2_dates`) side input | [detail](models/anysat.md) |
| `galileo` | S2 10-band time series | multi-frame | `pooled`, `grid` | month tokens + grouped tensors | [detail](models/galileo.md) |
| `agrifm` | S2 10-band time series | multi-frame | `pooled`, `grid` | fixed `T` frame stack behavior | [detail](models/agrifm.md) |

---

## Temporal and Comparison Notes (What People Usually Miss)

- `TemporalSpec.range(start, end)` is usually a **window for compositing**, not a single-scene acquisition selector.
- `OutputSpec.grid()` may be a **token/patch grid**, not a georeferenced raster grid (especially for ViT-like backbones).
- Cross-model comparisons are easiest with `OutputSpec.pooled()` and fixed ROI/temporal/compositing settings.
- Multi-frame models (`agrifm`, `anysat`, `galileo`) need extra attention to frame count and temporal side inputs.

Read the details in [Supported Models (Advanced Reference)](models_reference.md).

---

## Per-model Detail Pages (Template-first)

This repo now includes a reusable template page for documenting each model consistently:

- [Model Detail Template](model_detail_template.md)

Recommended fields for each model page:

- what the model expects (`input` contract, band order, temporal mode)
- what rs-embed currently feeds (current adapter behavior)
- preprocessing defaults and env knobs
- output semantics (`pooled` vs `grid` details)
- caveats for reproducibility / fair benchmarking

---

## Source of Truth in Code

Model registration source of truth:

- `src/rs_embed/embedders/catalog.py` (`MODEL_SPECS`)

Implementation details for each adapter live in:

- `src/rs_embed/embedders/onthefly_*.py`
- `src/rs_embed/embedders/precomputed_*.py`

When docs and code disagree, check code first and update docs accordingly.
