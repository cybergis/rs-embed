# Changelog

All notable changes to `rs-embed` will be documented in this file.

The project keeps this changelog as the canonical release record. GitHub Releases should summarize the same versioned changes rather than introducing a second source of truth.

The format is based on Keep a Changelog, and the project follows Semantic Versioning with extra care around model and embedding semantics.

## [Unreleased]

### Added

- **`anysat` takes a `patch_size`.** AnySat is trained across patch sizes, so its patch is now a knob like `olmoearth`'s `patch_size` and `thor`'s `RS_EMBED_THOR_PATCH_SIZE`: `model_config={"patch_size": n}` or `RS_EMBED_ANYSAT_PATCH_SIZE=n`, in pixels per patch side (AnySat's own `patch_size` argument is `10 × n` m). It sets the patch-token grid, `image_size / n` per side, which `grid_feature_mode="patch"` and the default patch-pooled output use; the `dense` grid stays at pixel resolution and the `tile` vector keeps its shape. The default, `1`, matches the previous behavior at `scale_m=10`. The resize target (`RS_EMBED_ANYSAT_IMG`, default `24`) must be divisible by `n`, and so must both sides of an input fed at its own size; otherwise the call raises a `ModelError` instead of letting AnySat drop the leftover pixels. `meta` records `patch_size` (pixels) and `patch_size_m` (the value passed to AnySat).

### Changed

- **GEE pixels download as binary arrays instead of JSON.** Every GEE fetch (sensor patches, S1 VV/VH, multi-frame time series, and all-band products such as `gse`) used `sampleRectangle(...).getInfo()`, which sends each pixel value as a JSON number: a 4 km `gse` patch (64 bands) was 233 MB of text to generate, transfer and parse. Fetches now call `ee.data.computePixels` for binary float32 on the common EPSG:3857 grid, a fifth of the bytes with no parsing; that `gse` patch drops from about 14 s to about 5 s (and to about 2 s together with the concurrent time-bin change). The fetched values are unchanged bit for bit — same pixels, same fill for masked pixels, same north-up rows — which the opt-in `tests/test_gee_compute_pixels_live.py` checks against the old call; the one exception is an ROI whose south edge lies exactly on a pixel line (see Fixed).
  - **Request limit:** GEE caps one `computePixels` request at 48 MiB (5 bytes per float32 value) and 32,768 px per side, instead of `sampleRectangle`'s 262,144 pixels. An oversized `BBox` is still split along pixel boundaries, but into far fewer, larger pieces that each carry every band, so GEE computes the composite once per piece: a 20 km S2 median (12 bands at 10 m) now takes 91 s instead of 214 s. A `PointBuffer`, which is never split, downloads its bands in as few evenly sized requests as the limit allows.
  - **`PointBuffer` reach:** a `PointBuffer` is never split, so one larger than 262,144 px (a `buffer_m` above 2,560 m at 10 m) used to fail; it now works up to the single-band limit (a `buffer_m` of about 15.8 km at 10 m).
  - **API:** `ProviderBase.fetch_array_chw` takes the ROI as `spatial=` (a `SpatialSpec`) instead of a provider-native `region=`, since the download grid is derived from it.
- **`anysat` no longer takes its patch size from `sensor.scale_m`.** The adapter passed the fetch resolution as AnySat's patch size in meters, a leftover from when `OutputSpec` carried its own `scale_m`; that is also why `scale_m` had to be a multiple of 10. A fetch at `scale_m=20` therefore ran with 2×2-pixel patches. The patch size now comes from `patch_size` alone, and `scale_m` no longer needs to be a multiple of 10.
  - **Who was affected:** `anysat` calls with `scale_m` other than 10. Where `scale_m // 10` divides `RS_EMBED_ANYSAT_IMG` (at the default 24: `scale_m` = 20, 30, 40, 60, 80, 120 or 240), pass `patch_size=scale_m // 10` to reproduce the earlier output. Other multiples of 10, such as 50, made AnySat silently drop the leftover pixels, so those grids did not line up with the ROI; that setting now raises instead. `resume=True` cannot detect the change, because the export fingerprint (sensor and `model_config`) is the same: re-run such exports without `resume`.
- **`olmoearth` `patch_size` / `image_size` reject non-integers.** A bool or a fractional value (`2.9`, `np.float32(3.7)`, `True`) used to be truncated by `int()` and run silently at a size nobody asked for; it now raises a `ModelError`. Whole-number floats such as `4.0` are still accepted. `anysat`'s `patch_size` follows the same rule.

### Fixed

- **GEE fetches no longer add a row south of an ROI whose south edge lies exactly on a pixel line.** `sampleRectangle` returned one extra pixel row beyond the ROI when the ROI's south edge fell exactly on the EPSG:3857 pixel lattice, so the array was one row taller than the ROI's common grid: the row held the fill value on clipped paths (sensor patches, all-band products such as `gse`) and pixels from outside the ROI on the S1 and multi-frame paths. Fetches now return exactly the common grid.
  - **Who was affected:** only ROIs whose south edge is a whole multiple of the pixel size in EPSG:3857 meters — in practice a `PointBuffer` centered exactly on the equator (latitude 0) with a `buffer_m` that is a multiple of `scale_m`, or a `BBox` with `minlat=0`. Other ROIs, including the common `buffer_m=2048`, are unaffected.
- **Inputs that fit in one tile are no longer padded or resampled by the tiler.** Under `input_prep="tile"` (the package default), an ROI no larger than one tile could reach the model three different ways, and none matched a plain call. Exports that use the batched tile tier (per-item exports on GPU, and every combined export) edge-padded it to the tile size, ran the model on the padded tile, and returned the grid with the pad rows and columns still in it; pooled vectors included the pad too. Single-point calls passed it unpadded but with the tile-dispatch model config, which forced `thor` into a fixed resize (see below). And an ROI up to `tile_snap_frac` (10% by default) larger than a tile was first shrunk to the tile size, changing its pixel spacing. Exports and `get_embedding` could therefore disagree for the same ROI. Such inputs are now always passed whole, unsnapped and unpadded, in both the single-point and the batch path, by the same rule the user-data path (`get_embedding_from_data`) already used. A flexible-size model (`olmoearth`) consumes the input at its native size (`resolved_mode="native"`); every other model gets the caller's `model_config`, exactly as with `input_prep="resize"` (`resolved_mode="resize"`). The plain-call fallbacks of `input_prep="auto"` likewise get the unsnapped input now. `INPUT_PREP_VERSION` is now `3` (`2` for this fix, `3` for the ROI-crop fix below). For exports with an on-the-fly model, the resume fingerprint includes it under any input prep, since the ROI-crop fix also changes `resize`-mode outputs, so `resume=True` re-runs outputs written under an earlier version when the export saved a manifest (the default); with `save_manifest=False`, delete those outputs or run without `resume`.
  - **Who was affected:** exports through the batched tile tier with the default or `tile` input prep, for every on-the-fly model with a batch-from-inputs API, on ROIs no larger than one tile (the model's tile size, e.g. 224 px or 2.24 km at 10 m for a 224 px model), and every path on ROIs at most 10% larger than a tile. Single-point `get_embedding` never padded; apart from the snap it was affected only through the tile-dispatch model config, which changes behavior for `thor` (below). Re-export grid and pooled embeddings for such ROIs. Inputs that need more than one tile are unchanged.
  - **`olmoearth` sub-tile ROIs now run at their native size.** They used to be resized to the tile size (256 px by default, or an explicit `tile_size`); a 200 px ROI now runs at 200 px, as it already did through `get_embedding_from_data`, so its grid has fewer cells. An explicit `image_size` in the model kwargs still fixes the working size.
- **`thor` no longer silently runs at the wrong pixel spacing.** THOR's fixed-resize path stretched the input to `RS_EMBED_THOR_IMG` (288) but told the model a ground cover of `scale_m × 288`, and THOR reads each band's GSD from its channel config. For any input of another size it ran at the wrong scale: a 256 px (2560 m) input was treated as 2880 m at 10 m when its real spacing was 8.89 m, so patch-size embeddings and output grid cells were off by about 11%. Nothing was logged; the only trace was `preprocess_strategy="fixed_resize"` in the metadata.
  - **Sub-tile ROIs under the default input prep were stretched.** Through the single-tile path above, a `thor` ROI smaller than 288 px (2.88 km at 10 m) was resized up to 288 px; a 200 px ROI ran at 6.94 m pixels told as 10 m. These ROIs now go through `native_snap` and keep their real spacing.
  - **The native-limit fallback no longer adds tokens.** Past `RS_EMBED_THOR_MAX_NATIVE_SIDE` or `RS_EMBED_THOR_MAX_NATIVE_TOKENS`, THOR fell back to the 288 px resize even when that had more tokens than the native input (5120 → 6480 at `patch_size=4`, 20480 → 25920 at `patch_size=2`). It now falls back only when the resize lowers the token count. Otherwise it runs natively, warns, and records `meta["native_limit_exceeded"]`.
  - **Every fixed resize inside THOR warns and records the real spacing.** `resize_mode="fixed"`, rectangular inputs beyond `RS_EMBED_THOR_SHAPE_TOL_PX`, native-limit fallbacks, and tiles whose size differs from `RS_EMBED_THOR_IMG` emit a `UserWarning`. They also record `effective_gsd_m` (per axis, `(y, x)`) and `gsd_mismatch` in the metadata; `native_snap` outputs carry the same keys. These keys only describe THOR's own resize: when the shared tiler first shrinks an input that needs several tiles (snap-to-tile, up to `tile_snap_frac`), that is recorded separately in `meta["input_prep"]["snapped_from_hw"]`.
  - **The default tile size follows `RS_EMBED_THOR_IMG`.** `describe()` reported a fixed 288, so with a custom `RS_EMBED_THOR_IMG` every default-size tile was resampled.
  - **Who was affected:** `thor` ROIs smaller than 288 px under the default input prep; `RS_EMBED_THOR_PATCH_SIZE` below 8 on inputs over the token limit (at `patch_size=4`, above about 196 px); custom `RS_EMBED_THOR_IMG` with tiling; and the fixed-resize cases above on inputs whose side was not `RS_EMBED_THOR_IMG`. Re-export affected THOR embeddings.
- **`thor` `native_snap` no longer pads with zeros or crops the ROI in the wrong frame, and its pooled output no longer depends on the ROI's shape** ([#137](https://github.com/cybergis/rs-embed/issues/137)). To reach a valid side (a multiple of 16 px at `patch_size=8` and 10 m), `native_snap` padded with the fill value (0), whose pixels fed the border tokens and the pooled mean, and a tie cropped (a 200 px input lost 4 px per side, about 8% of the ROI). The ROI window (`roi_window_geo`) was then applied as if the output grid covered the unsnapped input, so a rectangular ROI could be cropped one token off. Rectangular ROIs also pooled the group-merged grid while square ROIs pooled tokens, so under `group_merge="concat"` or `"sum"` the two gave vectors of different length or scale.
  - **What changed:**
    - `native_snap` snaps to the nearest valid side, a tie now pads up, and padding replicates edge pixels. `RS_EMBED_THOR_SHAPE_ADJUST=pad` replicates edges too.
    - When the snapped side would break `RS_EMBED_THOR_MAX_NATIVE_SIDE` / `MAX_NATIVE_TOKENS`, the valid side one unit below is used if it meets the limits and either the breach would force a resample to `RS_EMBED_THOR_IMG` or the snap was a tie that padded up.
    - The input's position is recorded in `meta["shape_prep"]["input_window"]`, and the ROI crop is mapped through it.
    - Grid output drops the ROI's context margin and any whole tokens of padding, and records the kept box in `shape_prep.grid_crop_box` (of `shape_prep.model_grid_hw`).
    - Every ROI's pooled output pools THOR's tokens inside the ROI window, per channel group. ROI sub-windows are labelled `pooling="roi_token_<mean|max>"` (previously `roi_grid_<mean|max>` over the merged grid).
    - An empty input raises a `ModelError`.
  - **Who was affected:**
    - `thor` with the default `RS_EMBED_THOR_RESIZE_MODE=native_snap`, under `input_prep="resize"` and (since the fix above) for ROIs smaller than one tile under the default prep.
    - Inputs that were padded now see edge values instead of zeros.
    - Inputs halfway between two valid sides now pad up instead of cropping (200 px gives 26×26 instead of 24×24 at `patch_size=8`).
    - Inputs from 393 to 407 px under default limits now run natively at 384 px instead of a 288 px resize.
    - Pooled vectors of rectangular ROIs change: groups are weighted by token count, and under `concat` the length matches square ROIs.
    - Re-export THOR pooled embeddings of rectangular ROIs, and grids of affected sizes.
- **Square-input models crop the ROI in their own input's frame** ([#139](https://github.com/cybergis/rs-embed/issues/139)). A square fetch records the ROI's window relative to the fetched array (`roi_window_geo`), but several models pad, crop or center-crop that array before the encoder, and cropped their output grid with the raw window anyway.
  - **What was wrong:**
    - When the fetched or user-provided array was not exactly square, the ROI crop could land one token row or column off.
    - Padding cells were never removed from grid output.
    - The window was also rounded to 6 decimals in `square_spatial` and `prepare_square`, so an edge exactly on a token boundary pulled in an extra token on large grids.
  - **Models affected:**
    - `agrifm`, `anysat`, `clay`, `dofa`, `fomo`, `galileo` and `prithvi` (both `RS_EMBED_PRITHVI_PREP` modes);
    - `remoteclip`, whose resize-then-center-crop drops the long axis' margins;
    - `olmoearth` with a sub-window ROI, where `_merge_geo_roi` replaced the pad window with the raw ROI window instead of composing the two.
    - Models that resize the whole array (`satmae`, `satmaepp`, `satmaepp_s2`, `scalemae`, `satvision_toa`, `terrafm`, `terramind`, `wildsat`) were already correct.
  - **What changed:**
    - `prepare_square` records the input's exact position in the square as `meta["shape_prep"]["input_window"]`, for pad, crop and none; `roi_window` is now that window clipped to the square, unrounded.
    - The shared `shape.model_roi_window` maps the ROI window through it, and `shape.crop_grid_for_output` applies it. Every model above uses them: grid output is cropped to the mapped window, which also drops whole cells of padding; pooled output pools the ROI's cells only when the ROI is a sub-window, and otherwise keeps the model's own pooled vector.
    - `square_spatial` returns exact windows.
    - A ROI that lies wholly in a part of the input the model never sees (cropped away) emits a `UserWarning` instead of silently returning the nearest edge.
    - `remoteclip` maps through its center crop only for its built-in preprocessing; a wrapper with its own `transform` keeps the unmapped window, since its geometry is unknown.
    - `agrifm`, `anysat`, `fomo` and `remoteclip` now record `shape_prep` in their metadata, as the other models above already did.
  - `INPUT_PREP_VERSION` is now `3`, so `resume=True` re-runs on-the-fly outputs written before this fix (see above).
  - **Who was affected:** grid and ROI-pooled output of the models above whenever the array fed to them was not square. That covers rectangular arrays passed through `get_embedding_from_data` / `input_chw` (whose grids now exclude padding cells, so their shape changes) and fetches off by a pixel. It also covers windows on token boundaries of large grids.
- **ROI-pooled vectors are pooled like the model's own pooled vector** ([#140](https://github.com/cybergis/rs-embed/issues/140)). When a rectangular `BBox` is fetched as a square, pooled output pools only the ROI's tokens. On several models that vector ignored the ROI, lay in another space than the model's pooled vector for a whole input, or used another reduction, so ROI and whole-input vectors from one export were not comparable.
  - **What was wrong:**
    - The ROI was ignored and the whole-square vector returned without a warning: `satvision_toa` (its pooled path had no token grid), `anysat` with `pooled_source="tile"`, and the no-token fallbacks of `remoteclip`, `fomo`, `wildsat` and `scalemae`. The `fomo` and `wildsat` fallbacks were still labelled `roi_grid_*`.
    - The ROI vector lay in another space: `dofa` skipped `fc_norm`; `remoteclip` pooled its 768-wide intermediate tokens, while its pooled vector is the 512-wide projected CLIP embedding; `wildsat` pooled its 768-wide encoder tokens, while its pooled vector passes the CLS token through torchvision's `heads` and, by default, the image head.
    - The reduction differed: `terrafm`, `wildsat` and `remoteclip` max-pooled the ROI for `pooling="max"`, although their pooled vector is a CLS token that ignores `pooling`. `fomo` and `satmaepp_s2` took the max over cells whose spectral or group tokens were first averaged, while their whole-input max is over all tokens.
  - **What changed:**
    - `shape.pool_for_output` builds the pooled output of every on-the-fly model except `olmoearth` and `thor`, which already pool their own tokens over the ROI window. It raises a `ModelError` if an ROI vector's shape differs from the model's pooled vector. `crop_grid_and_pool` is removed, and `crop_grid_for_output` now only crops grid output.
    - Models that pool tokens reduce the ROI's tokens with `pooling`, and several tokens per cell with the same operation. `RS_EMBED_SATMAEPP_S2_GRID_REDUCE` now only affects grid output.
    - Models whose pooled vector is a CLS token or a fixed mean pass the ROI's token mean through the same norm, head or projection: `fc_norm` for `dofa`, `ln_post` and `proj` for `remoteclip`, torchvision's `heads` and the image head for `wildsat`. `satvision_toa` pools its normed last-stage tokens, so its ROI vector is its pooled vector restricted exactly to the ROI. `anysat` with `pooled_source="tile"` pools the patch tokens of the same forward (`output="all"`). These models record `pooling="max"` as `meta["requested_pooling"]`, for a whole input too, and the tiled path merges their tiles as a mean instead of an elementwise max.
    - Without a token grid, the whole-square vector is returned with a `UserWarning` and `meta["roi_pooled"] = False`; ROI-pooled vectors carry `meta["roi_pooled"] = True` and the pooled cells' extent in `meta["roi_grid_hw"]`. `agrifm`'s pooled `meta["grid_hw"]` is now its model grid for every ROI.
    - An ROI whose margin is under one token on every side covers every token, so it keeps the model's own pooled vector. A CLS model therefore no longer switches to a token mean for an almost-square `BBox`.
    - Every ROI-pooled vector is labelled `roi_grid_<mean|max>` (`prithvi` multi-frame: `roi_grid_temporal_<mean|max>`). `agrifm` and `anysat` used their whole-input label, and `galileo` used `grid_max` for an ROI max. `dofa` and `remoteclip` now label a whole-input vector `model_pooled`, and `terrafm` labels it `model_cls`.
  - `INPUT_PREP_VERSION` is now `4`, so `resume=True` re-runs on-the-fly outputs written before this fix.
  - **Who was affected:** pooled output of the models above for a rectangular `BBox` fetched as a square, and tiled pooled output of the CLS and fixed-mean models with `pooling="max"`. Other whole-input pooled vectors and grid output are unchanged; whole-input meta changes only in the labels above.
- **ROI windows and fetch flags survive a combined-export resume.** A resumed combined export restored the prefetched input arrays from the checkpoint but not their fetch metadata. Resumed square-fetched inputs were therefore not cropped back to the ROI, and processing flags such as `satvision_toa`'s `already_unit_scaled` were lost. The prefetch checkpoint now stores and restores each input's fetch metadata; the final manifest drops it together with the prefetch arrays.
- **`clay` and `dofa` crop the ROI on the tensor-backend batch path.** `get_embeddings_batch_from_inputs(backend="tensor")` dropped the per-item `fetch_metas`, so the outputs covered the whole fetched square.

## [0.3.1] — 2026-10-14

Bug-fix release for large Google Earth Engine requests ([#134](https://github.com/cybergis/rs-embed/issues/134)). `gse` could return duplicated mosaics for ROIs above about 262,144 pixels, and every GEE-backed model could gain a fill-value row or column, or lose a real one, where an oversized request was split. If you exported `gse` embeddings, or grid outputs from on-the-fly models, for `BBox` ROIs above that size, re-export them.

### Fixed

- **`gse` no longer returns duplicated mosaics for large ROIs.** `gse` used to tile oversized requests itself: it sized them with an estimate that undercounts EPSG:3857 pixels by about 1/cos(lat), cut overlapping tiles, and joined them with a plain concatenation. Results came back duplicated at any latitude: a near-complete copy of the ROI in each quadrant at high latitude, about 2.2× at 35–38°N and 500 m. Pooled vectors were biased the same way, and some requests raised a `ValueError`. `gse` now makes a single provider request and leaves the splitting to the provider.
- **GEE requests above the `sampleRectangle` pixel limit now stitch exactly.** The provider's pixel-limit fallback cut requests at the Mercator midpoint and then trimmed or padded each seam toward an estimated length, which could insert a fill-value row/column or drop a real one. Cuts now fall on pixel edges of the common EPSG:3857 grid, so a split request reproduces the single-request grid exactly. Oversized `BBox` requests are also split up front instead of after a failed request.
  - **Who was affected:** `gse` and all on-the-fly models (single-image, Sentinel-1 and multi-frame fetches share this path), for a `BBox` ROI above 262,144 pixels at `scale_m`, roughly 5 km × 5 km at 10 m. Not every such request was hit; it depended on where the cut landed. For on-the-fly models a seam showed up as a black line in the model input, with the image beyond it shifted by one pixel. That perturbed grid embeddings near the seam and slightly biased pooled vectors.
  - **Not affected:** `PointBuffer` ROIs (an oversized one raises an error instead), `BBox` ROIs under the limit, `tessera`, `copernicus_embed`, user-provided data (`get_embedding_from_data`), and model-side `input_prep="tile"` tiling.

### Deprecated

- **`RS_EMBED_GSE_MAX_PIXELS`** no longer has any effect and emits a `FutureWarning` when set; it will be removed in a future release. It only served as a workaround for the `gse` duplication above.

## [0.3.0] — 2026-09-27

Bring-your-own-data: embed imagery you already have (arrays or GeoTIFFs) without any provider fetch, plus cache-first Hugging Face loading and restored `tessera` downloads. `tessera` grid outputs now come back on the common EPSG:3857 grid instead of native UTM, so their shape, transform and `meta` keys differ from 0.2.x; re-export tessera grids you want to compare against provider-backed models.

### Fixed

- **`tessera` downloads work again.** `pyproject.toml` now requires `geotessera>=0.10`; the old tile host (`dl2.geotessera.org`) was retired upstream and the data moved to Source Cooperative in that release.
- **Tessera embeddings now land on the common EPSG:3857 grid.** Previously tessera returned native UTM tiles, making pixel-for-pixel comparison with provider-fetched patches impossible and cross-UTM-zone ROIs an error. Tiles are now resampled (nearest-neighbour) onto the shared 10 m Web Mercator lattice; cross-zone ROIs produce a `UserWarning` instead of crashing. Grid/projection logic consolidated in the new `tools/projection.py`.
- **Warm-cache model loading no longer contacts the Hugging Face Hub.** All `hf_hub_download` / `snapshot_download` calls (plus the `rshf` loaders for satmae/scalemae/satmaepp) now resolve from the local HF cache first; a network request only fires on a cache miss. Shared helpers live in `embedders/shared.py`.
- **`resolve_pretrained_source_cache_first` no longer crashes on bin-only repos.** The local-dir shortcut now requires `model.safetensors`; repos that publish only `pytorch_model.bin` (e.g. MVRL/satmaepp) fall back to the repo-id path whose cached-bin fallback still works.
- **Clay batch path no longer forces Earth Engine auth on machines without GEE.** `ClayEmbedder.get_embeddings_batch_from_inputs` no longer initializes a provider when running from prefetched inputs, matching the lazy-provider convention already followed by the single-embedding path.

### Added

- **`UserData(crs=..., transform=...)` and `UserData.from_raster(path, ...)`** — a declaration can now carry a CRS and pixel-to-CRS affine; the footprint is derived from them (so `spatial` may be omitted), and the array is resampled to the common EPSG:3857 grid at the model's `scale_m` before embedding. `from_raster` reads pixels, CRS, and transform directly from a GeoTIFF via `rasterio`. Geo-conditioned models (clay, prithvi) accept georeferenced declarations without manual `spatial=` input. `crs` and `transform` must be supplied together.
- **Bring-your-own-data API** — `get_embedding_from_data` / `get_embeddings_batch_from_data` embed imagery you provide directly, with no provider fetch or auth required. Band matching, tiling/stitching, and temporal grouping follow the same `input_prep` policy as the fetch path. Flexible-size models (`olmoearth`) adapt to the input's native size automatically. `list_models_for_data` reports which catalog models a given declaration can serve without loading weights. See [docs/user_data.md](docs/user_data.md).
- **[docs/benchmarks.md](docs/benchmarks.md)** — end-to-end latency and throughput for every registered backend (H100 + GEE, 20-point Midwest workload), with per-request fetch-vs-compute decomposition.

## [0.2.1] — 2026-07-27

An export-correctness fix for time-series models plus a new input-inspection API. Multi-model exports that combined a temporal model (olmoearth/agrifm/anysat/galileo/prithvi) with other models on a shared collection should be re-exported (see Fixed).

### Fixed

- **Export prefetch no longer collapses temporal-model inputs to a single composite.** Multi-model exports could silently degrade time-series models: a band-union merged fetch group (e.g. olmoearth exported alongside terrafm) dropped the model's own `fetch_input`, and same-sensor models with different temporal semantics (e.g. clay vs. agrifm/anysat/galileo, which resolve to an identical SensorSpec) shared one cached single-frame input. Both the saved `input_chw__*` arrays and the computed embeddings were affected (`temporal_mode: single`, duplicated frames). Prefetch cache and plan keys now include each model's temporal fetch fingerprint (`custom:*` / `multi:n` / `single`): models share a fetch only when sensor identity **and** temporal semantics match, custom-`fetch_input` models always fetch through their own path, and spec-driven equal-division models still band-union merge among themselves. Temporal-model inputs cached by resumed exports under the old keys are refetched (their cached values were wrong). Re-export any multi-model runs that included olmoearth/agrifm/anysat/galileo/prithvi alongside other models sharing a collection.

### Added

- **`inspect_model_input`** — fetch and inspect the raw provider input a model would actually receive, without loading weights or running inference. Goes through the embedder's own `fetch_input` path (same bands, temporal binning, fetch-square geometry, and empty-bin handling as embed time); multi-frame models return one report entry per temporal bin, flagging empty bins the model would drop. `examples/playground.ipynb` §3.3 now uses it instead of hand-reconstructing the olmoearth bin loop.
- **`examples/play_with_embedding.ipynb`** — a self-contained downstream-task walkthrough over exported embeddings (PCA, clustering, and a small torch prediction head); depends only on public libraries.

## [0.2.0] — 2026-06-30

Two new models (Clay v1.5, OlmoEarth), window-adaptive temporal sampling, a new package-wide `input_prep="tile"` default, and a large batch of correctness fixes that make `get_embedding`, the batch APIs, and `export_batch` return identical embeddings for the same point. **Default embeddings differ from 0.1.x** (see Changed); pin `input_prep`/`temporal_mode` explicitly where strict reproducibility is required.

### Added

- **New model `clay`** — Clay v1.5 foundation model (ViT-L MAE, 1024-d, patch 8): 10-band Sentinel-2 L2A input with the official normalization and lat/lon + time + GSD + wavelength conditioning; `pooled` is the encoder CLS token. Configure with `model_config={"model_size": ...}`; see [docs/models/clay.md](docs/models/clay.md).
- **New model `olmoearth`** — OlmoEarth v1/v1.1/v1.2 family (Allen AI, 12-band S2 L2A, FlexiViT), variants nano→large, default `base_v1_2` (1024-d); included in the base install. See [docs/models/olmoearth.md](docs/models/olmoearth.md).
- **Prithvi multi-frame input** — `temporal_mode="auto"/"multi"` feeds a real HLS time series instead of one median composite; frame count follows the window length (≤4).
- **Fetch-square for rectangular ROIs, on every path** — rectangles are fetched as enlarged squares of real imagery and the output is cropped back to the ROI, so nothing is stretched and the same BBox yields the same geometry via any API. See `docs/spatial_roi.md`.
- **`model_config` for more models** — `terramind` (`model_key`, `modality`), `remoteclip` (`model_id`), `terrafm` (`cache_dir`), `tessera` (`cache_dir`), `copernicus` (`data_dir`). Legacy env vars / `sensor.collection` prefixes still work as fallbacks.
- **Quality of life** — `gse` auto-tiles requests larger than the GEE sample limit; `export_batch(show_progress=True)` prints per-chunk GEE fetch statistics.

### Changed

- **Default `input_prep` is now `"tile"` (was `"resize"`) for all models** — inputs keep native resolution instead of being downsampled. This changes default embeddings vs 0.1.x; pass `input_prep="resize"` to reproduce the old output.
- **`olmoearth` and `galileo` default to `temporal_mode="auto"`** — windows over ~1 month now sample multiple frames (up to 12 composites, so up to ~12× the fetch cost). Pass `temporal_mode="single"` (olmoearth) or a fixed `n_frames` (galileo) for the cheaper path.
- **`satmaepp` 10-band path is now `modality="s2_10b"`** — the standalone `satmaepp_s2*` model ids are removed (embeddings unchanged).
- **`export_batch` per-model settings go through `ExportModelRequest`** — the `per_model_sensors`/`per_model_fetches`/`per_model_modalities` dict parameters are removed; a top-level `input_prep` parameter is added.
- **Typed errors and louder failures** — export pipelines raise `ProviderError`/`ModelError` instead of bare `RuntimeError`; failed `model.to(device)` moves, CUDA fallbacks to CPU, and provider import failures now warn instead of passing silently.
- **`temporal=None` now warns** that the package default window (2022-06-01..2022-09-01) is used, and embedding meta always records the actually-resolved window.
- **Internal restructuring** (no behavior change) — normalization lives entirely in embedders (fetch returns raw values), provider/embedder layers were separated, and pipeline routing is driven by explicit `EmbedderCapabilities` declarations.

### Fixed

- **Cross-path consistency:** the same point now yields identical embeddings via `get_embedding`, the batch APIs, and `export_batch` — per-model `input_prep` resolution, ROI crop-back on tiled/batch paths, per-point metadata alignment, and `remoteclip`/`satvision` normalization mismatches that produced garbage or inconsistent export embeddings.
- **Tiling/stitching:** eliminated the flat "dead band" along short ROI edges, duplicated seam rows/columns, and double-counted tile overlaps in pooled merges; edge tiles are edge-replicate padded, never stretched.
- **Temporal models:** `prithvi` no longer degrades to a single composite on tiled/export paths; `olmoearth` multi mode covers windows beyond 12 months instead of dropping them; all five temporal adapters record frame-diversity metadata and warn when real frames are sparser than requested.
- **Provider fetch:** S1 `orbit` requests are honored; non-S2 collections are no longer clipped to the S2 value range; cloud filtering uses each collection's own property instead of emptying Landsat/S1/DEM collections.
- **Export robustness:** all writers are atomic and `resume=True` verifies a request fingerprint before reusing outputs (a changed request re-exports instead of returning stale results); combined manifests keep all models; duplicate model names raise; `load_export` aligns rows by `point_index` so failed points leave NaN rows instead of shifting survivors.
- **Misc:** precomputed models (`gse`, `copernicus`) accept `TemporalSpec.range` (warn + use the start year); `tessera` no longer downloads every tile twice; weight caches are device-keyed (no more cpu/cuda races); CLI import error fixed.

## [0.1.3] — 2026-04-12

### Added

- `wildsat` now supports `variant` selection via `model_config={"variant": "..."}` (or the `variant=` keyword). Available variants are `vitb16` (default), `resnet50`, and `swint`, each backed by its own ImageNet-pretrained checkpoint that auto-downloads from Google Drive. The previous `vitl16` arch has been removed as no upstream checkpoint exists. Users who need other pre-training initializations (CLIP, Prithvi, SatCLIP, etc.) can still point `RS_EMBED_WILDSAT_CKPT` at a local checkpoint. The `RS_EMBED_WILDSAT_ARCH` environment variable is still respected as a fallback but is now overridden by `variant` when both are set.

### Changed

- `GEEProvider` initialization now prefers an explicit Google Cloud project when one is provided via `project=...`, `EE_PROJECT`, or `GOOGLE_CLOUD_PROJECT`, but no longer hard-requires rs-embed callers to pass one explicitly. When no project is supplied, rs-embed now lets `ee.Initialize()` and `geemap.ee_initialize()` resolve Earth Engine's configured default project first, while still surfacing a clear error message when authentication is missing or no usable Cloud/quota project is configured.
- `thor` now defaults `RS_EMBED_THOR_PATCH_SIZE` to `8` instead of `16`, increasing the default token-grid density while keeping `RS_EMBED_THOR_IMG=288`. THOR also now defaults to a bounded `native_snap` preprocessing policy for ordinary resize-mode inputs: near-square inputs can keep a snapped native side when they stay within configured side/token limits, while tiled inputs still force fixed per-tile resize so stitched grids remain geometrically stable. The THOR model page now documents the patch-size/image-size coupling, native-snap limits, and concrete environment-variable examples for common tuning patterns.

### Fixed

- `anysat` and `satmaepp_s2_10b` now validate the `variant` keyword in a single place instead of accepting a wider alias set in `_normalize_*_variant` and then raising a second `ModelError` deeper in the runtime resolver. Previously, passing `variant="tiny"` / `"small"` to `anysat` or `variant="base"` to `satmaepp_s2_10b` would first be silently normalized and then rejected with a confusingly-located "currently exposes only variant=..." error. The normalize helpers now only accept the variants that actually map to a wired checkpoint (`anysat` → `base`, `satmaepp_s2_10b` → `large`), raise an immediate and descriptive `ModelError` for anything else, and the duplicate runtime guards have been removed. The `satmaepp_s2_10b` env-var path (`RS_EMBED_SATMAEPP_S2_MODEL_FN`) now also raises a clear error for unknown `model_fn` values instead of silently producing a `variant=None` runtime config. The `describe()` output for both adapters already advertises `choices: ["base"]` / `choices: ["large"]`, so the schema side was already correct; this fix just makes the validation code match it.
- `BBox.validate()` now enforces geographic bounds: longitudes must be in `[-180, 180]` and latitudes in `[-90, 90]`. Out-of-range coordinates previously passed validation and caused confusing downstream errors from the GEE provider.
- `describe_model()` now returns a cached copy of the embedder's `describe()` output instead of instantiating a new embedder class on every call. The cache is keyed by canonical model name and is cleared by `reset_runtime()`. The returned dict is always a shallow copy so callers cannot mutate the cached entry.
- `fetch_api_side_inputs()` now wraps per-spatial fetch errors in a `ModelError` that includes the spatial index and the original exception, making it easier to pinpoint which location caused a failure when running `get_embeddings_batch()` with `input_prep="tile"` or `"auto"`.
- `run_embedding_request()` now uses `strict=True` in the `zip` of spatials and prefetched inputs. A length mismatch between the two lists now raises immediately instead of silently truncating the result.
- Loading checkpoint arrays during combined-export resume now emits a `warnings.warn` instead of silently swallowing the exception. Users will see a clear message indicating that array loading failed and that all inputs will be re-fetched.
- `_write_per_item_chunk` no longer accesses the private `_shutdown` attribute of `ThreadPoolExecutor` to guard against double-shutdown. The outer `finally` block now relies on the documented idempotency of `ThreadPoolExecutor.shutdown()` instead, removing a fragile dependency on CPython internals that could break on future Python versions.
- `sensor_key()` no longer applies `int()` truncation to `scale_m` and `cloudy_pct` when building the embedder instance cache key. Previously, float values such as `10.1` and `10.9` were both mapped to `10`, allowing two sensors with different resolutions to share a cached embedder instance incorrectly. The raw field values are now used directly.
- `_run_per_item` now closes all progress bars (main and per-model) inside the `finally` block of the chunk-pipeline loop. Previously the cleanup ran after the `try/finally`, so an unhandled exception (e.g. `continue_on_error=False`) would leave progress bars open and leak display resources in notebook environments.

## [0.1.2] - 2026-04-03

This release rolls up upstream-alignment work and correctness fixes that may change default embedding behavior for some model adapters compared with `0.1.1`. Users who need strict reproducibility across versions should review the model-specific changes below and pin explicit options where needed.

### Changed

- Standardised NumPy docstrings across all public functions and classes in `export.py`, `inspect.py`, `writers.py`, and the `tools/` and `providers/` layers. No behaviour changes.

### Added

- Versioned documentation with a version selector powered by `mike` and MkDocs Material. Each release tag deploys a pinned version; pushes to `main` update a `dev` alias. The `mike`, `mkdocs-material`, and `pymdown-extensions` packages are now included in the `[dev]` optional group.

- `load_export(path)` reader API that loads any export produced by `export_batch(...)` — both combined (single file) and per-item (directory) layouts — and returns a structured `ExportResult`. Failed points are NaN-filled rather than dropped, partial model runs are surfaced via `status="partial"`, and `ExportResult.embedding(model)` provides a typed shortcut to the embedding array.

### Changed

- `anysat` now defaults `grid` output to native `dense` features while keeping pooled output on patch-grid pooling by default, with new AnySat-specific switches for `grid_feature_mode` (`dense`/`patch`) and `pooled_source` (`patch`/`tile`).
- `galileo` now aligns more closely with the upstream NASA Harvest runtime: `grid` output prefers Galileo's own patch-level token averaging path, automatic NDVI derivation has been removed, and the default normalization mode is now `none` with an `official_stats` option for upstream pretraining statistics.
- `satvision_toa` now uses the vendored official SatVision runtime as its only model path and narrows provider-side preprocessing to the default MODIS proxy route (`MOD09GA` reflectance + `MOD21A1D` thermal proxy). Custom collections are no longer treated as implicit GEE fallbacks; callers should pass calibrated `input_chw` directly for non-default inputs.

### Fixed

- `device="auto"` now correctly selects MPS on Apple Silicon instead of silently falling back to CPU. Follows the PyTorch-recommended priority (`cuda > mps > cpu`), giving an approximately 4x speedup on Apple M-series hardware for all API calls that use the default device.
- `galileo` month overrides now use the official zero-based month indexing expected by Galileo embeddings, fixing the previous one-month offset in `RS_EMBED_GALILEO_MONTH`.
- `satvision_toa` `grid` output now consistently extracts spatial features from the official-style SwinV2 path instead of misinterpreting pooled vectors as token grids, and the tightened fetch path records explicit proxy provenance in metadata.
- `scalemae` now follows the official feature-extraction path more closely: the adapter unwraps common `rshf` wrappers to call the nested ScaleMAE backbone's `forward_features(...)` instead of falling back to wrapper `forward()`, uses ImageNet eval preprocessing (`Resize(short side) + CenterCrop + Normalize`) with effective post-preprocess `input_res_m`, and fixes the declarative input metadata to reflect raw Sentinel-2 SR inputs plus adapter-managed preprocessing.
- `satmaepp` and `satmaepp_s2_10b` now align more closely with the official SatMAE++ preprocessing paths. The RGB adapter now defaults to `rgb` channel order for the published fMoW-RGB checkpoint and no longer pre-resizes provider/input overrides before the official eval transform, while the Sentinel-2 10-band adapter no longer sanitizes inputs with adapter-side `clip` / `nan_to_num` before the source-style `SentinelNormalize -> ToTensor -> Resize(short side) -> CenterCrop` pipeline.

## [0.1.1] - 2026-04-01

### Added

- Automated pull request changelog enforcement with a `skip-changelog` escape hatch for docs, tests, CI, and other internal-only changes.
- Tag-driven GitHub Release publishing that uses the matching `CHANGELOG.md` section as the release notes.
- Trusted Publishing release automation for PyPI and TestPyPI, including a manual TestPyPI dry run and install smoke test.

### Changed

- The contribution and release workflow now treats `CHANGELOG.md` as the canonical source for user-visible release notes.
- The tag-triggered release flow now validates `src/rs_embed/_version.py`, publishes to PyPI, and only then creates the GitHub Release.
- The tag-triggered release flow now validates the matching `CHANGELOG.md` entry before publishing to PyPI, so a missing release-notes section fails early instead of after package upload.
- The base package installation now includes the Copernicus GeoTIFF runtime (`tifffile` and `imagecodecs`) instead of requiring a separate extra.
- The public docs now default to `pip install rs-embed` for published releases while keeping editable installs documented for repository development.

### Deprecated

### Removed

### Fixed

- The TestPyPI smoke test now verifies package importability and the `rs-embed` CLI entry point, not just installability and version metadata.

## [0.1.0] - 2026-03-31

### Added

- Initial public alpha release of `rs-embed`.
- Unified ROI to embedding API centered on `get_embedding(...)`, `get_embeddings_batch(...)`, `export_batch(...)`, and `inspect_provider_patch(...)`.
- Support for precomputed embedding products including `tessera`, `gse`, and `copernicus`.
- Support for on-the-fly model adapters including `satmae`, `satmaepp`, `satmaepp_s2_10b`, `prithvi`, `scalemae`, `remoteclip`, `dofa`, `satvision`, `anysat`, `galileo`, `wildsat`, `fomo`, `terramind`, `terrafm`, `thor`, and `agrifm`.
- Documentation site covering quickstart, model behavior, API contracts, and extension guidance.

[Unreleased]: https://github.com/cybergis/rs-embed/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/cybergis/rs-embed/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/cybergis/rs-embed/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cybergis/rs-embed/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cybergis/rs-embed/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.3
[0.1.2]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.2
[0.1.1]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.1
[0.1.0]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.0
