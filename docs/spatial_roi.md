# Spatial ROI Handling: From Region to Embedding

You pass `rs-embed` a geographic **region** and get back an embedding **of that
region**. But most on-the-fly model encodes a **fixed-size square** input (e.g.
256×256 tokens). This page explains the contract that bridges the two — how an
arbitrary ROI (a thin rectangle, a near-square box, or a large non-square area)
becomes the square the encoder wants, and how the output is mapped back to your
ROI so the embedding still describes the region you asked for.

This is the spatial counterpart to [Temporal Sampling](temporal_sampling.md): the
same way a `TemporalSpec.range` window is turned into the frames each model wants,
a spatial ROI is turned into the square frame each encoder wants — and then mapped
back.

!!! abstract "The one idea"
    A rectangular ROI is **enlarged to a square of real imagery** (not stretched),
    encoded, and the output is **cropped back to the ROI**. So a 1.8:1 field is
    never smeared into striped embeddings, and the returned vector/grid describes
    your region, not the enlarged square.

---

## The contract, in three stages

```mermaid
flowchart LR
    ROI["Your ROI\n(BBox / PointBuffer)"] --> SQ["1. Square the request\nenlarge rectangle → centered\nsquare of real imagery\n(record roi_window)"]
    SQ --> ENC["2. Encode the square\nresize to model size,\nor tile at native res"]
    ENC --> CROP["3. Crop output back\ngrid → crop to roi_window\npooled → pool ROI tokens only"]
    CROP --> OUT["Embedding of your ROI"]
```

**1. Square the request — fetch real pixels, don't stretch.**
If the ROI is a rectangle, `rs_embed.tools.spatial.square_spatial` enlarges it to a
**centered square in EPSG:3857**, so the model sees the *real* surrounding imagery
instead of a stretched rectangle. It returns the ROI's normalized position inside
that square (`roi_window = (y0, y1, x0, x1)`). A `PointBuffer` or an already-square
`BBox` is left unchanged (`roi_window` = the full frame).

**2. Encode the square.**
The undistorted square is run through the encoder — resized to the model's fixed
input size on the resize path, or split into native-resolution tiles on the tile
path (`input_prep="tile"`, the package default for most models).

**3. Crop the output back to the ROI.**
The encoder's token grid / feature map is cropped to `roi_window`
(`crop_grid_to_roi` / `roi_token_box`); for `pooled` output only the ROI tokens are
pooled (`crop_grid_and_pool`). The stitched grid from the tile path is cropped the
same way, and tiled `pooled` embeddings are weighted by each tile's ROI-overlap
area. The result describes **your** ROI, not the enlarged square.

---

## By ROI shape

| Your ROI | What happens | Output |
| -------- | ------------ | ------ |
| **Small rectangle** (e.g. 1.8:1) | Enlarge to the bounding square, encode, crop the grid back to the rectangle | Rectangular token grid / ROI-only pooled vector |
| **Near-square** (≈ model input) | No enlargement; straight resize + encode | Full square grid |
| **Large non-square** | Tiled at native resolution; stitched grid cropped back, pooled weighted by ROI overlap | ROI-shaped stitched field / ROI-weighted vector |
| **Point** (`PointBuffer`) | Already square; encode directly | Full square grid |

---

## Scope: which models

This contract applies to **all on-the-fly models** — anything that runs a
fixed-input-size encoder (`prithvi`, `olmoearth`, `galileo`, `anysat`, `agrifm`,
`dofa`, `fomo`, `terramind`, `terrafm`, `thor`, `satmae`, `satmaepp`, `scalemae`,
`remoteclip`, `wildsat`, `satvision`).

**Precomputed products (`tessera`, `gse`, `copernicus`) are exempt.** They sample
your ROI directly from a precomputed global grid — there is no fixed model input to
square against, so no enlarge/crop step is needed (`gse` does its own large-request
tiling; see its page).

!!! note "Token granularity limits the crop"
    Cropping back is only as fine as the model's token grid. A coarse grid (e.g.
    `satvision`'s 4×4 SwinV2-Giant output) cannot isolate a sub-2:1 ROI — that is a
    token-granularity limit, not a bug. Finer-grid models crop cleanly.

---

## Fallback: when the request can't be fetch-squared

`square_spatial` cannot always enlarge — the square might run past valid lat/lon at
the poles or antimeridian, or you may pass a non-square array directly via
`input_chw` (bypassing the fetch). In those cases the array itself is squared by
`rs_embed.tools.shape.prepare_square`:

| Case | What happens | `meta["shape_prep"]["applied"]` |
| ---- | ------------ | ------------------------------- |
| Already square | Resize to `image_size` | `none` |
| Any rectangle | Make square (pad **or** crop), then resize; the ROI window is recorded so the output is cropped back | `pad_to_square` / `crop_to_square` |

`pad` (default) reflect-pads the short side (keeps the whole ROI, adds synthetic
border); `crop` center-crops the long side (no synthetic data, discards margins).
There is no silent stretch fallback: extremely rectangular inputs (`aspect ≥ 2.0`)
still pad, with a `UserWarning` that most of the model input is synthetic border —
prefer a square fetch of real imagery (automatic on provider-backed paths) or
`input_prep="tile"` for such ROIs.

### The `shape_adjust` knob

For most models this fallback is internal and fixed to `pad`. One model exposes the
choice:

| Model | How to set | Default |
| ----- | ---------- | ------- |
| [`olmoearth`](models/olmoearth.md) | `shape_adjust="pad"`/`"crop"` (keyword / `model_config`) | `pad` |

[`thor`](models/thor.md) also squares non-square inputs, but through its own bounded
`native_snap` path (`RS_EMBED_THOR_SHAPE_ADJUST`, default `crop`) — separate from
this fallback; see the THOR page.

---

## What gets recorded

| Field | Meaning |
| ----- | ------- |
| `meta["input_prep"]["roi_cropped"]` | `True` when the output was cropped back to a sub-ROI |
| `meta["shape_prep"]["applied"]` | `none` / `pad_to_square` / `crop_to_square` (fallback path) |
| `meta["shape_prep"]["orig_hw"]` / `["square_hw"]` / `["target_hw"]` | ROI shape, squared shape, final model size |
| `meta["shape_prep"]["aspect"]` | `long_side / short_side` (≥ 2.0 triggers the heavy-padding warning) |

`roi_cropped=True` means you got the ROI, not the square.

---

## Practical advice

The contract removes aspect distortion but **cannot invent spatial detail** the ROI
never had. For tiny ROIs the real fix is to request a **larger, roughly square**
BBox (e.g. ~2.56 km at 10 m → native 256×256) so no upsampling is needed at all.
Keeping `aspect < 2.0` also keeps the pad fallback (direct `input_chw` rectangles)
mostly real imagery instead of synthetic border.
