# Changelog

All notable changes to `rs-embed` will be documented in this file.

The project keeps this changelog as the canonical release record. GitHub Releases should summarize the same versioned changes rather than introducing a second source of truth.

The format is based on Keep a Changelog, and the project follows Semantic Versioning with extra care around model and embedding semantics.

## [Unreleased]

### Added

- **Multi-frame temporal mode for Prithvi (`temporal_mode="multi"`).** Prithvi-EO 2.0 is a multi-temporal foundation model — pretrained on 4-timestep HLS series with 1–6 month gaps between consecutive frames ([arXiv:2412.02732](https://arxiv.org/abs/2412.02732)) — but the `prithvi` adapter previously always collapsed the requested window into a single median composite (`T=1`). A new `temporal_mode` `model_config` key (default `"auto"`) adds a `"multi"` mode that feeds a real `[B, 6, T, H, W]` time series with per-frame `(year, day_of_year)` coordinates through the vendored runtime's temporal encoder. `auto` resolves from the window — `single` when it yields one frame (sub-month range), else `multi` — so multi-month/-year requests sample temporally by default instead of collapsing into a single composite (force the old behavior with `temporal_mode="single"`; note `multi` fetches one composite per frame). The frame count is **derived from the requested window** rather than forced to a fixed value: `T = clamp(window_days // 28, 1, max_frames)` (default `max_frames=4`, matching the pretraining regime; the ~28-day minimum spacing is the low end of the model's 1–6 month training interval). The window is then split into `T` equal bins, so the whole period is represented. Short windows therefore collapse to `T=1` instead of being padded with duplicate frames, and because the provider back-fills imagery-free sub-windows with the whole-window composite, **bit-identical frames are dropped** — so a window lacking real temporal diversity also degrades gracefully to `T=1`. Per-frame dates align with the provider's binning (shared `split_date_range`). Because `T` is capped at 4, very long windows (beyond ~2 years) yield an effective frame gap above the ~6 month (`184`-day) training maximum; rather than silently truncate, the adapter records `max_frame_gap_days` in metadata and flags `temporal_spacing_out_of_range` (with a `UserWarning`) when the gap exceeds `RS_EMBED_PRITHVI_MAX_STRIDE_DAYS`. The vendored `PrithviMAE` is loaded with `num_frames=T`; no learned weights depend on the frame count (positional embeddings are re-interpolated at runtime), so checkpoints remain weight-compatible. **Output dimensionality is unchanged**: `pooled` is still `(D,)` (pooled over time and space) and `grid` is still `(D, H', W')` (pooled over time), so switching modes never changes the embedding shape — only its values and the `num_frames` / `frame_dates` metadata. Both the single and batch paths are supported (the batch path groups items by prepared `(T, H, W)` shape, loading one runtime per distinct frame count). Configurable via `model_config={"temporal_mode": "multi", "max_frames": 4}` or the `RS_EMBED_PRITHVI_TEMPORAL_MODE` / `RS_EMBED_PRITHVI_MAX_FRAMES` / `RS_EMBED_PRITHVI_FRAME_STRIDE_DAYS` / `RS_EMBED_PRITHVI_MAX_STRIDE_DAYS` environment variables. See `docs/models/prithvi.md` and `tests/test_prithvi_multiframe.py`.

- **OlmoEarth v1/v1.1 embedder (`olmoearth`).** Adds support for the [OlmoEarth](https://huggingface.co/collections/allenai/olmoearth) foundation model family from Allen AI, trained on the Major TOM dataset. All 7 released variants are supported: `nano`, `tiny`, `base`, `large` (v1) and `nano_v1_1`, `tiny_v1_1`, `base_v1_1` (v1.1), with embedding dimensions 128/192/768/1024. The adapter fetches all 12 Sentinel-2 L2A bands from GEE in OlmoEarth's native band-set order, applies per-band mean±2σ normalization (OlmoEarth COMPUTED strategy), and encodes with the FlexiViT encoder. Both `pooled` and `grid` output modes are supported. `patch_size` (default 4) and `image_size` (default 256) are configurable via `model_config` or environment variables. Requires the `olmoearth-pretrain-minimal` package: `pip install rs-embed[olmoearth]`.

- **GEE fetch statistics reporting in `export_batch`.** When `show_progress=True`, a `[gee_fetch]` summary line is now printed to stderr after each prefetch chunk completes, reporting total planned fetches, completed, failed, cache hits, and the most recently processed point/sensor. This gives users visibility into GEE quota consumption, cache reuse, and whether runtime is dominated by fetching vs. inference. No output is emitted when `show_progress=False` or when no GEE provider is involved (e.g. precomputed models). The underlying `FetchStats` class in `tools/progress.py` is thread-safe and accumulates counts cumulatively across chunks.

### Fixed

- **`olmoearth` `temporal_mode="multi"` silently dropped everything beyond the first ~12 months.** Multi-frame binning used fixed 30-day bins capped at 12, so a request spanning more than ~1 year (e.g. a 3-year window) embedded only its first 12 monthly frames and discarded the rest with no indication. The adapter now detects when capping would drop ≥ one full 30-day bin of trailing time and instead **equal-divides the whole window into 12 frames**, so the entire requested period is represented. Because those frames are then spaced wider than OlmoEarth's monthly training cadence, the case is surfaced rather than hidden: `meta["temporal_sampling"]` becomes `"equal_divided"`, `meta["temporal_spacing_stretched"]=True`, `meta["effective_stride_days"]` records the cadence, and a `UserWarning` is emitted (embeddings from such windows are extrapolated). Windows up to ~12 months are unchanged — they keep the in-distribution fixed 30-day cadence (a ≤1-bin trailing remainder is still dropped as before, so the YEAR regime is untouched). The fixed-stride-vs-equal-division policy is centralized in the reusable `rs_embed.tools.temporal.fixed_or_equal_bins`, which composes the existing `split_date_range_fixed_days` and `split_date_range` helpers. See `docs/models/olmoearth.md`.

- **`remoteclip` `pooled` output was inconsistent between single and batch/tiled paths.** `get_embedding` returned a 768-d mean over raw ViT tokens, while `get_embeddings_batch[_from_inputs]` returned the 512-d projected CLIP embedding (`encode_image`) — so the same point yielded different-dimensional, non-comparable vectors depending on ROI size (≤224 px ran single, larger tiled) or which API was called. The single path now also uses `encode_image`, so `pooled` is the canonical 512-d CLIP embedding on every path. `grid` extraction was also rewritten to read open_clip's `forward_intermediates` patch grid (the previous code scanned for 3-D tensors that open_clip never returns, so it always fell through to a forward hook whose token axis order was open_clip-version dependent and could silently collapse to a 1×1 grid); the hook fallback now handles batch-first and sequence-first layouts. Single and batch `grid` outputs are now identical. See `docs/models/remoteclip.md`.

- **Combined `export_batch` manifest lost all but the first model entry.** `combined_write_checkpoint` returned a fresh copy of the manifest (the jsonable copy passed to `write_arrays`) instead of the dict the caller keeps mutating. In multi-model combined exports, each per-model checkpoint rebound the loop's manifest to that copy while the checkpoint writer kept serializing the stale original, so every model entry after the first was dropped from both the in-memory result and the on-disk manifest — and resume would needlessly re-run already-completed models. The method now merges the writer's path keys (`npz_path`, `nc_path`, etc.) back into the original manifest and returns it, preserving the in-place mutation contract. Per-item exports were unaffected. Regression tests added in `tests/test_checkpoint_manager.py`.

- **`tessera` no longer re-fetches every tile twice per point.** `_mosaic_and_crop_strict_roi` in `embedders/precomputed_tessera.py` was invoking its `tiles_rows_factory` callable twice — once to scan tile bounds (Pass 1) and again to paste pixels into the crop canvas (Pass 2) — which caused `geotessera.fetch_embeddings(tiles)` to re-iterate (and on a cold cache, re-download / re-parse) every tile block on the second pass. The factory is now called once and materialized into a list that both passes share. This roughly halves the I/O cost of the first chunk of an `export_batch` run that includes `tessera`, which was previously the dominant contributor to the long "stuck at 0/N" interval users observed before the first chunk completed. Memory cost: all tile blocks for the current point are now resident simultaneously instead of one at a time; for typical 2–4 km buffers this is 1–4 tile blocks per point. Reduce `RS_EMBED_TESSERA_BATCH_WORKERS` if running with very large buffers on memory-constrained hosts.

- **CLI `ModuleNotFoundError` on import.** `rs_embed.cli` was importing from `rs_embed.export` and `rs_embed.inspect`, two modules that do not exist in the current package layout. The imports now point directly to `rs_embed.api` (`export_batch`, `inspect_gee_patch`). The `export-npz` subcommand call site has been updated to match `export_batch`'s current signature: a single spatial argument is wrapped in `spatials=[...]`, the output path becomes `ExportTarget.combined(args.out)`, and the flat boolean flags (`save_inputs`, `save_manifest`, etc.) are grouped into an `ExportConfig` object. The stub injection in `tests/test_cli_parsers.py` that was masking the broken imports has been removed and the integration test updated to patch `cli.export_batch` instead of `cli.export_npz`.

- **GEE tile orientation made explicit and self-contained.** `GEEProvider.fetch_array_chw` now applies `_flip_sample_tile_y` internally before returning, giving the method a documented north-up contract. Previously the flip was the caller's responsibility inside `_fetch_provider_array_chw_with_bbox_fallback`, which created a leaky abstraction: any `ProviderBase` subclass overriding `fetch_array_chw` had to know to return south-up raw data or risk being flipped twice. The caller no longer applies a second flip. The `_sample_image_bands_raw_chw` docstring now documents that GEE's `reproject(crs=..., scale=...)` **without** `.clip()` naturally returns north-up rows, and warns explicitly that adding `.clip()` would change the row ordering to south-up and break the multiframe fetch path. `_flip_sample_tile_y` docstring updated to describe exactly which call pattern requires it and which does not.

### Changed

- **`olmoearth` `temporal_mode` now defaults to `"auto"` (was `"single"`).** `auto` picks the mode from the requested window: `single` when the range spans a single temporal bin (≤ ~1 month, where multi adds nothing), otherwise `multi`. This makes multi-month/-year ranges sample temporally by default instead of silently collapsing into one composite. **Cost note:** any range longer than ~1 month now resolves to `multi`, which fetches one composite per bin (up to 12) — roughly up to 12× the GEE fetches/time of single mode, which matters for large `export_batch` runs. Pass `temporal_mode="single"` (or env `RS_EMBED_OLMOEARTH_TEMPORAL_MODE=single`) to force the cheaper single composite. `auto`/`single`/`multi` are all accepted.

- **`galileo` now samples a window-adaptive frame count instead of a fixed 8, and gained a `temporal_mode` knob (default `"auto"`).** Galileo is a multi-temporal model that encodes month-of-year (0–11) and pretrains on ~monthly composites capped at 12 frames, but the adapter previously always split the requested window into a fixed 8 equal frames — so a 2-week window became 8 near-identical sub-monthly frames (denser than training) while a 3-year window became 8 frames ~4.5 months apart (far sparser than training). The frame count is now derived from the window to match the monthly cadence: ~30-day frames, at most 12. `temporal_mode="auto"` resolves to `single` (T=1) for sub-month windows and `multi` otherwise; `single`/`multi` force the mode. Windows beyond the 12-month capacity are **equal-divided into 12 frames** (covering the whole range) and flagged as out-of-distribution: `meta["temporal_sampling"]="equal_divided"`, `meta["temporal_spacing_stretched"]=True`, `meta["effective_stride_days"]`, plus a `UserWarning`. This reuses the same `rs_embed.tools.temporal.fixed_or_equal_bins` policy as `olmoearth`. **Cost note:** multi-month ranges fetch one composite per frame (up to 12); sub-month windows are now *cheaper* (1 frame instead of 8). `temporal_mode` and a manual frame-count override (`n_frames`) are settable via `model_config` — newly threaded through `get_embedding`/`get_embeddings_batch` — or via `RS_EMBED_GALILEO_TEMPORAL_MODE` / `RS_EMBED_GALILEO_FRAMES`. The window-adaptive count also applies to the API/export prefetch path via a new `fetch_input` override. See `docs/models/galileo.md`.

- **Image-level ViT patch-token grid models now use safer `input_prep` defaults.** For `remoteclip`, `scalemae`, `satmae`, `satmaepp`, and `satmaepp_s2_10b`, `OutputSpec.grid()` with `input_prep=None` or `"auto"` resolves to `"resize"` and emits a warning because tiled patch-token grids can show stitching seams. Explicit `"tile"` remains available but warns and records seam-risk metadata; explicit `"resize"` is the recommended no-warning path.

- **`ProviderBase.fetch_array_chw` contract tightened to north-up.** The method is now required to return north-up CHW float32. Custom provider implementations must either return north-up data directly or apply `_flip_sample_tile_y` internally. Test fake providers in `test_api_helpers.py` have been updated accordingly (north-up row generation, renamed test functions).

### Added

- **Orientation regression test for `_fetch_all_bands_impl`.** `test_gee_provider.py` now includes `test_fetch_all_bands_impl_passes_through_gee_row_order`, which injects a mock GEE `sampleRectangle` response with a known row order and asserts the function output matches exactly (no flip applied). This locks in the pass-through behaviour so that any accidental flip addition is caught immediately, and documents that the north-up guarantee for `fetch_collection_patch_all_bands_chw` comes from GEE's empirical behaviour for the `reproject(crs=..., scale=...) + clip` call pattern rather than from Python-level normalization.

### Changed

- **Normalization responsibility moved entirely to embedders.** `NormalizationSpec` has been removed from `ModelInputSpec` and from `rs_embed.core.specs`. The `apply_normalization()` helper in `rs_embed.providers.fetch` has been removed. `fetch_input()` and all fetch helpers (`fetch_collection_patch_chw`, `fetch_s2_rgb_chw`, `fetch_s2_multiframe_raw_tchw`, `fetch_s1_vvvh_raw_chw`) now consistently return raw provider values (S2 DN in [0, 10000], S1 linear float, etc.); each embedder applies its own normalization inside `get_embedding()`. This eliminates a misleading contract where `ModelInputSpec.normalization` was declared but never automatically applied, and removes a normalize→denormalize round-trip that existed in some batch paths (remoteclip, wildsat).

- **Internal refactor: provider and embedder layer separation.** The `embedders/runtime_utils.py` grab-bag module has been removed and its contents redistributed by responsibility:
  - Provider selection and lifecycle management → `providers/resolution.py` (`default_provider_backend_name`, `resolve_provider_backend_name`, `is_provider_backend`, `get_cached_provider`, `create_provider_for_backend`)
  - Provider fetch helpers and satellite normalization → `providers/fetch.py` (`fetch_sensor_patch_chw`, `fetch_collection_patch_chw`, `fetch_s2_rgb_chw`, `fetch_s1_vvvh_raw_chw`, `fetch_s2_multiframe_raw_tchw`, `normalize_s1_vvvh_chw`, `apply_normalization`, etc.)
  - Device resolution and embedder instance loading → `tools/runtime.py` (`resolve_device_auto_torch`, `load_cached_with_device`)
  - Input coercion → `tools/normalization.py` (`coerce_input_to_tchw`, `coerce_single_input_chw`)
  - This fixes an inverted dependency where `tools/` was importing from `embedders/`.
- **Internal refactor: embedder utility modules.** `embedders/_vit_mae_utils.py`, `embedders/image_utils.py`, and `embedders/token_utils.py` have been removed. Image preprocessing (`ensure_torch`, `resize_rgb_u8`, `fetch_s2_rgb_u8_from_provider`, CLIP norm, etc.) and ViT token operations (`pool_from_tokens`, `tokens_to_grid_dhw`) are now defined locally in each embedder that needs them. This makes each embedder file self-contained and eliminates hidden shared state between model implementations.
- **Internal rename:** `embedders/meta_utils.py` → `embedders/meta.py`, `embedders/config_utils.py` → `embedders/config.py`. No functional change; the `_utils` suffix was removed to match the naming convention of other modules in the package.

### Added

- `gse` now automatically tiles large spatial requests. If the estimated pixel footprint of a `BBox` at the requested `scale_m` exceeds the GEE `sampleRectangle` limit (default 512×512, overridable via `RS_EMBED_GSE_MAX_PIXELS`), the region is split into a sub-BBox grid, each tile is fetched from `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, and the embedding arrays are concatenated before pooling or grid output is applied. This removes the previous hard cap on request size and makes the result equivalent to a single large fetch.

### Fixed

- `export_batch` no longer raises `ModelError` when a `TemporalSpec.range` is passed alongside precomputed models (`gse`, `copernicus`) that only index by year. Both embedders now extract the start year from the range, emit a `UserWarning`, and continue. `copernicus` still raises if the extracted year is not in `SUPPORTED_YEARS`. The early-exit validation in `assert_supported` was also tightened from `"year" in mode_hint` to an exact `mode_hint == "year"` check, which fixes a latent bug where `tessera`'s `"year_or_range"` hint was incorrectly matched and would have blocked range inputs before the embedder's own graceful handling could run.
- `gse` batch inference now runs concurrently on CPU in `export_batch`. Previously, chunk-level batching was only enabled when a GPU was detected; precomputed models that do their own remote IO benefit from batching regardless of device.
- `get_embedding` and `get_embeddings_batch` no longer raise `ModelError` when `input_prep='tile'` is passed with `model='gse'`. `export_batch` no longer silently ignores `input_prep` for GSE. Both APIs now emit a `UserWarning` when a non-default `input_prep` is passed alongside `model='gse'`, clarifying that GSE manages its own tiling.
- `input_prep='tile'` no longer disables `get_embeddings_batch` for precomputed models in `export_batch`. The `allow_nonresize` guard in `_evaluate_batch_capability` now applies only to prefetched-input batch APIs; no-input batch APIs (which never receive an `input_chw` to tile) are unaffected.

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

[Unreleased]: https://github.com/cybergis/rs-embed/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.2
[0.1.1]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.1
[0.1.0]: https://github.com/cybergis/rs-embed/releases/tag/v0.1.0
