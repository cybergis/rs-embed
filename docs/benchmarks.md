# Latency & Throughput

How long does an embedding actually take? This page reports end-to-end
timing for every registered backend, measured through the same public calls
you use (`get_embedding` / `export_batch`) — including the data-acquisition
step, which most model papers leave out and which turns out to be the whole
story.

!!! abstract "The one idea"
    **Data acquisition dominates; the model almost never does.** On an H100,
    model compute stays within 1–2 s per point across architectures from
    22M to 2.6B parameters — while fetching the input takes 75–90% of the
    end-to-end time for single-frame models and pushes time-series models an
    order of magnitude higher. Pick a backend by its **input footprint and
    serving path**, not by its parameter count.

---

## The numbers

One H100 node (SXM 80 GB, 16 CPU cores), imagery served by Google Earth
Engine, 20 random points over the agricultural US Midwest, each a 2 048 m
point buffer embedded with the model's default configuration. Measured
2026-08.

### Single-frame: one composite per request

| Model | Params | Dim | Fetch p50 (s) | Compute p50 (s) | Total p50 (s) | Total p95 (s) | Seq. (pt/s) | Batch (pt/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SatMAE | 330M | 1024 | 3.6 | 1.11 | 4.8 | 7.2 | 0.21 | 1.60 |
| Scale-MAE | 303M | 1024 | 4.2 | 1.13 | 5.4 | 7.6 | 0.19 | 1.45 |
| SatVision-TOA | 2.6B | 4096 | 4.7 | 1.01 | 5.6 | 6.8 | 0.18 | 1.24 |
| RemoteCLIP | 151M | 512 | 5.5 | 1.10 | 6.8 | 8.6 | 0.15 | 1.18 |
| THOR | 94M | 768 | 6.4 | 0.98 | 7.4 | 9.8 | 0.13 | 0.91 |
| Clay v1.5 | 311M | 1024 | 6.5 | 1.03 | 7.5 | 10.6 | 0.13 | 0.94 |
| SatMAE++ | 330M | 1024 | 6.1 | 1.28 | 7.6 | 9.5 | 0.13 | 0.73 |
| TerraMind (small) | 22M | 384 | 5.9 | 1.81 | 7.8 | 14.4 | 0.13 | 0.42 |
| WildSAT | 87M | 256 | 6.3 | 2.13 | 8.3 | 13.1 | 0.12 | 0.62 |
| TerraFM | 113M | 768 | 7.2 | 1.82 | 9.0 | 14.8 | 0.11 | 0.56 |
| DOFA | 111M | 768 | 7.1 | 2.08 | 9.4 | 12.8 | 0.11 | 0.60 |
| FoMo | 67M | 768 | 8.6 | 0.96 | 9.7 | 16.4 | 0.10 | 0.80 |
| Prithvi-EO-2.0 | 113M | 768 | 9.9 | 1.11 | 11.0 | 24.0 | 0.09 | 0.74 |

### Multi-frame: a time series per request

| Model | Params | Dim | Fetch p50 (s) | Compute p50 (s) | Total p50 (s) | Total p95 (s) | Seq. (pt/s) | Batch (pt/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OlmoEarth | 258M | 768 | 31.8 | 3.23 | 35.1 | 60.3 | 0.03 | 0.15 |
| AgriFM | 88M | 1024 | 42.6 | 1.49 | 44.5 | 68.2 | 0.02 | 0.14 |
| AnySat | 126M | 768 | 41.6 | 4.85 | 46.8 | 60.8 | 0.02 | 0.08 |
| Galileo (nano) | 1.0M | 128 | 54.3 | 2.41 | 56.5 | 71.3 | 0.02 | 0.10 |

### Precomputed: retrieval only, no GPU

| Model | Dim | Retrieval p50 (s) | p95 (s) | Seq. (pt/s) | Batch (pt/s) |
|---|---:|---:|---:|---:|---:|
| Copernicus-Embed | 768 | <0.1 | <0.1 | 10,000* | 11.02 |
| Tessera | 128 | 1.3 | 2.6 | 0.75 | 1.42 |
| Google Satellite Embedding | 64 | 15.3 | 25.7 | 0.07 | 0.14 |

*\* in-memory lookup after a one-time ~3 s load — see the note below; the
Batch column is the realistic end-to-end rate.*

??? info "Exactly what was measured"
    **Latency** is the wall time of a single `get_embedding()` call with
    weights already loaded, split at the provider boundary: *Fetch* is GEE
    network I/O plus server-side compositing (or product retrieval for
    precomputed backends), *Compute* is everything else — preprocessing,
    inference, output assembly. Medians and p95 over the 20 points,
    single-threaded.

    **Throughput** compares two ways of processing the same 20 points:
    *Seq.* is a plain loop (the reciprocal of median latency), *Batch* is
    `export_batch()`, which overlaps fetching and inference with 8 prefetch
    workers at inference batch size 8.

    The 2 048 m buffer is a radius: each request covers ≈4.1 km × 4.1 km,
    i.e. ~410×410 pixels per band at the default 10 m scale.
    Copernicus-Embed uses a 20 km buffer instead, matching its coarse grid.
    Parameter counts are read off the loaded checkpoints. The three
    precomputed backends run entirely on CPU; only on-the-fly models touch
    the GPU, and none of them needs more than 10.5 GB of it.

---

## How to read it

The split into three tables *is* the finding. Single-frame models — whatever
their size — cluster between 5 and 11 seconds per point, and most of that
is waiting for pixels. Multi-frame models pay for their input, not their
weights: a full year of Sentinel-1/2 imagery puts them at 35–57 s per
point, 89–96% of it acquisition. The starkest illustration sits at the two
ends of the catalog: the **largest** model here (SatVision-TOA, 2.6B) is
among the *fastest* end to end because it consumes a single monthly
composite, while the **smallest** (Galileo-nano, 1.0M) is the *slowest*
because it consumes a year-long time series.

!!! tip "Anything beyond a handful of points belongs in `export_batch`"
    The batch path overlaps fetching with inference and delivers **2–8×**
    the throughput of a plain loop on identical hardware and provider —
    compare the *Seq.* and *Batch* columns. A hand-rolled
    download-to-disk pipeline fares even worse: in our side-by-side test it
    ran 12–16× slower than the export path with the same weights.

!!! note "The three precomputed backends are three different animals"
    **Copernicus-Embed** loads its coarse grid once (~3 s) and then answers
    from memory in under a millisecond — its *Batch* rate measures pipeline
    overhead, not lookup cost. **Tessera** reads locally cached tiles; the
    numbers above are warm-cache, and the *first* visit to a new tile
    downloads it, which can take tens of seconds. **Google Satellite
    Embedding** is precomputed but served through GEE, so it behaves like a
    heavy fetch — if you only need pooled vectors at points, Google's
    server-side `reduceRegions` is the faster tool for that job.

!!! warning "Fetch times move with provider load"
    Between two sweeps two weeks apart, per-model fetch medians shifted by
    up to ±30% — in both directions. Read the table for **ratios and
    regimes**, not as guaranteed absolute numbers.
