# Changelog

All notable changes to `rs-embed` will be documented in this file.

The project keeps this changelog as the canonical release record. GitHub Releases should summarize the same versioned changes rather than introducing a second source of truth.

The format is based on Keep a Changelog, and the project follows Semantic Versioning with extra care around model and embedding semantics.

## [Unreleased]

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

[Unreleased]: https://github.com/cybergis/rs-embed/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cybergis/rs-embed/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.3
[0.1.2]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.2
[0.1.1]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.1
[0.1.0]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.0
