# Temporal Sampling

Several on-the-fly models are **multi-temporal**: they were pretrained on a *series* of images over time, not a single scene. In `rs-embed` you only ever pass a `TemporalSpec.range(start, end)` — the adapter decides how many frames `T` to sample from that window and how to space them. This page explains how that decision is made, model by model.

!!! abstract "The one idea"
    A `TemporalSpec.range` is a **window**, not a single date. For multi-temporal models the window is turned into `T` frames whose count and spacing are chosen to match how each model was trained. Wider window → more frames (up to a model-specific cap); a sub-month window collapses to a single composite (`T=1`).

---

## Why the frame count is derived from the window

Older versions packed every window into a fixed number of frames (e.g. always 8). That is wrong at both ends:

- a 2-week window split into 8 frames → 8 near-identical composites (denser than the model ever saw in training);
- a 3-year window split into 8 frames → frames months apart (far sparser than training).

Instead, the temporal models now choose `T` from the **window length**, matched to their training cadence, and fall back gracefully when the window is too short (collapse to `T=1`) or too long (cover the whole window but flag it as out-of-distribution). Monthly-cadence models (`olmoearth`, `galileo`) share a single policy helper, `rs_embed.tools.temporal.fixed_or_equal_bins`.

---

## `temporal_mode`: `auto` / `single` / `multi`

Models that support adaptive sampling expose a `temporal_mode` knob (default `auto`):

| `temporal_mode` | Meaning |
| --------------- | ------- |
| `auto` (default) | Pick from the window: `single` when the window is too short to yield ≥2 frames, else `multi`. |
| `single` | One composite over the whole range (`T=1`) — cheapest; ignores temporal structure. |
| `multi` | Force the multi-frame series even for a short window. |

Set it as a keyword argument (`get_embedding(..., temporal_mode="single")`) or via the model's environment variable.

---

## Decision table

How each model turns `TemporalSpec.range(start, end)` (length = `window_days`) into frames:

| Model | Per-frame time encoding | Training cadence | `auto` → `single` when | `multi` frame count | Max frames | Out-of-distribution (too-wide) handling |
| ----- | ----------------------- | ---------------- | ---------------------- | ------------------- | ---------- | --------------------------------------- |
| [`olmoearth`](models/olmoearth.md) | sequence position + month (0–11) | ~monthly | `window_days` ≤ ~30 (one 30-day bin) | fixed 30-day bins anchored at `start` | **12** | `window_days` > ~390 → equal-divide into 12 frames; `temporal_spacing_stretched=True` + `UserWarning` |
| [`galileo`](models/galileo.md) | month-of-year (0–11) | ~monthly | `window_days` ≤ ~30 (one 30-day frame) | ~30-day frames anchored at `start` | **12** | `window_days` > ~390 → equal-divide into 12 frames; `temporal_spacing_stretched=True` + `UserWarning` |
| [`prithvi`](models/prithvi.md) | (year, day-of-year) | 1–6 month gaps (≈28–184 days) | `window_days` < ~56 (i.e. `window_days // 28 < 2`) | `T = clamp(window_days // 28, 1, 4)`, equal bins | **4** | window beyond ~2 yr → frame gap > 184 days; `temporal_spacing_out_of_range=True` + `UserWarning` |
| [`anysat`](models/anysat.md) | day-of-year (0–364) | dense (S2 revisit) | — (no `auto`; always multi) | fixed `N` equal frames (`RS_EMBED_ANYSAT_FRAMES`, default 8) | — | none — date-aware and cadence-agnostic by design |
| [`agrifm`](models/agrifm.md) | none (frame order only) | dense intra-season | — (no `auto`; always multi) | fixed `N` equal frames (`RS_EMBED_AGRIFM_FRAMES`, default 8) | — | n/a — no calendar encoding, only frame order |

!!! note "Why `anysat` and `agrifm` keep a fixed frame count"
    `anysat` injects each frame's real day-of-year and was trained for *any* cadence, so it doesn't need a window-derived count or an out-of-distribution warning. `agrifm` is a Video Swin model that encodes only frame **order** (no calendar date) and was trained on fixed-length intra-season stacks, so changing the frame count would change its learned temporal receptive field. Both therefore stay on a fixed `N` (tune via their `_FRAMES` env var).

---

## Worked examples

Using `olmoearth` (monthly, cap 12):

| `TemporalSpec.range` | `window_days` | `auto` resolves to | Frames (`T`) | Notes |
| -------------------- | ------------- | ------------------ | ------------ | ----- |
| `2022-06-01 → 2022-06-20` | 19 | `single` | 1 | < 30 days → one composite |
| `2022-06-01 → 2022-09-01` | 92 | `multi` | 3–4 | fixed 30-day bins |
| `2022-01-01 → 2023-01-01` | 365 | `multi` | 12 | full monthly year (in-distribution) |
| `2022-01-01 → 2025-01-01` | 1096 | `multi` | 12 (equal-divided) | > ~390 days → **stretched**, ~91-day spacing, `UserWarning` |

`galileo` behaves the same way (monthly, cap 12). `prithvi` uses the same idea with a ~28-day stride and a cap of 4, so its single/multi switch is at ~56 days and it saturates at 4 frames.

---

## Out-of-distribution windows

When a window is wider than a model's frame budget can cover at its training cadence, `rs-embed` **does not silently drop the trailing time**. Instead it covers the whole window with the available frames (spaced wider than training) and records that the result is extrapolated:

- **`olmoearth` / `galileo`**: `meta["temporal_sampling"]="equal_divided"`, `meta["temporal_spacing_stretched"]=True`, `meta["effective_stride_days"]`, plus a `UserWarning`.
- **`prithvi`**: `meta["temporal_spacing_out_of_range"]=True`, `meta["max_frame_gap_days"]`, plus a `UserWarning`.

To stay in-distribution, **narrow the temporal window** so the frames land near the model's training cadence.

---

## Data availability: empty sub-windows are never silent

Choosing `T` frames assumes each frame's sub-window actually contains a clear scene. When some don't (clouds, gaps in coverage, a tiny trailing bin), `rs-embed` makes the gap explicit instead of pretending the series is full. Two mechanisms, by model family:

- **`olmoearth`** (drops empty bins): the empty bins are removed from the sequence, so `T` shrinks. Recorded as `meta["n_bins"]` (bins the window produced), `meta["n_frames"]` (frames encoded), and `meta["dropped_bins"]` (the dropped `[start, end]` ranges), with a `UserWarning`.
- **`prithvi` / `galileo` / `anysat` / `agrifm`** (back-fill): the multi-frame provider fetch keeps `T` fixed by **back-filling empty sub-windows with the whole-window composite** (and padding short series by repeating the last frame) — i.e. duplicate frames. `rs-embed` records `meta["n_frames_requested"]`, `meta["n_distinct_frames"]`, and `meta["n_backfilled_frames"]`, and emits a `UserWarning` naming how many of the `T` sub-windows actually had distinct imagery. `prithvi` additionally **collapses** the duplicate frames (so its encoded `T` shrinks — see also `meta["requested_frames"]` vs `meta["num_frames"]`); the others feed the duplicate timesteps to the encoder.

So `n_distinct_frames < n_frames_requested` (or `n_frames < n_bins` for `olmoearth`) means the window was sparser than the requested frame count — **narrow/shift the window, raise `cloudy_pct`, or lower the frame count** to recover genuine temporal diversity. The warning fires from the shared fetch helper, so it covers the single, tiled, batch, and export paths uniformly.

---

## Cost

`multi` fetches **one composite per frame** from the provider (e.g. GEE), so a multi-month/-year window costs up to `T`× the fetches/time of a single composite — up to **12×** for `olmoearth`/`galileo`, **4×** for `prithvi`. This matters most for large `export_batch` runs over many points. To control it:

- pass `temporal_mode="single"` to force one composite regardless of window length;
- or narrow the window so `auto` resolves to fewer frames.

Sub-month windows are *cheaper* than the old fixed-count behavior, since `auto` collapses them to `T=1`.

---

## See also

- [Models Overview](models.md) — the on-the-fly model table (Temporal style column).
- [Advanced Reference](models_reference.md) — per-model temporal packaging and side inputs.
- Per-model temporal sections: [olmoearth](models/olmoearth.md#temporal_mode), [galileo](models/galileo.md#temporal-sampling), [prithvi](models/prithvi.md#temporal-mode-temporal_mode).
