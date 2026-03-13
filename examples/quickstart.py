from __future__ import annotations

"""
rs-embed quickstart script.

Examples:
  python examples/quickstart.py --mode auto
  python examples/quickstart.py --mode gee --device auto
  python examples/quickstart.py --mode all --run-export
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "values"):
        return np.asarray(x.values)
    return np.asarray(x)


def _show_embedding(tag: str, emb) -> None:
    arr = _as_numpy(emb.data)
    meta = emb.meta or {}
    print(f"\n[{tag}]")
    print(f"shape={arr.shape}, dtype={arr.dtype}")
    print(
        "meta_preview:",
        {
            "model": meta.get("model"),
            "type": meta.get("type"),
            "backend": meta.get("backend"),
            "source": meta.get("source"),
            "input_time": meta.get("input_time"),
        },
    )


def _show_batch(tag: str, embeddings: Iterable) -> None:
    embs = list(embeddings)
    print(f"\n[{tag}] count={len(embs)}")
    for i, emb in enumerate(embs):
        arr = _as_numpy(emb.data)
        print(f"  #{i}: shape={arr.shape}, dtype={arr.dtype}")


def run_auto_demo(*, run_export: bool, out_dir: Path) -> None:
    from rs_embed import (
        PointBuffer,
        TemporalSpec,
        OutputSpec,
        export_batch,
        get_embedding,
        get_embeddings_batch,
    )

    print("\n=== Auto quickstart (precomputed: tessera) ===")
    spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=1024)
    temporal = TemporalSpec.year(2024)

    pooled = get_embedding(
        "tessera",
        spatial=spatial,
        temporal=temporal,
        output=OutputSpec.pooled(pooling="mean"),
        backend="auto",
    )
    _show_embedding("single/pooled", pooled)

    grid = get_embedding(
        "tessera",
        spatial=spatial,
        temporal=temporal,
        output=OutputSpec.grid(scale_m=10),
        backend="auto",
    )
    _show_embedding("single/grid", grid)

    spatials = [
        PointBuffer(lon=121.5, lat=31.2, buffer_m=1024),
        PointBuffer(lon=120.5, lat=30.2, buffer_m=1024),
    ]
    batch = get_embeddings_batch(
        "tessera",
        spatials=spatials,
        temporal=temporal,
        output=OutputSpec.pooled(pooling="mean"),
        backend="auto",
    )
    _show_batch("batch/pooled", batch)

    if run_export:
        auto_out = out_dir / "auto_export"
        auto_out.mkdir(parents=True, exist_ok=True)
        manifests = export_batch(
            out=str(auto_out),
            layout="per_item",
            names=["p1", "p2"],
            spatials=spatials,
            temporal=temporal,
            models=["tessera"],
            output=OutputSpec.pooled(),
            backend="auto",
            save_inputs=False,
            save_embeddings=True,
            save_manifest=True,
            resume=True,
            show_progress=True,
        )
        print(f"\n[export/auto] wrote {len(manifests)} items to: {auto_out}")


def run_gee_demo(*, device: str, run_export: bool, out_dir: Path) -> None:
    from rs_embed import (
        PointBuffer,
        SensorSpec,
        TemporalSpec,
        OutputSpec,
        export_batch,
        get_embedding,
        get_embeddings_batch,
        inspect_provider_patch,
    )

    print("\n=== GEE quickstart (on-the-fly: remoteclip) ===")
    print("Ensure GEE is authenticated first: earthengine authenticate")

    spatial = PointBuffer(lon=121.5, lat=31.2, buffer_m=512)
    temporal = TemporalSpec.range("2022-06-01", "2022-09-01")
    sensor = SensorSpec(
        collection="COPERNICUS/S2_SR_HARMONIZED",
        bands=("B4", "B3", "B2"),
        scale_m=10,
        cloudy_pct=30,
        composite="median",
    )

    check = inspect_provider_patch(
        spatial=spatial,
        temporal=temporal,
        sensor=sensor,
        backend="gee",
        name="quickstart_patch",
        return_array=False,
    )
    report = check.get("report") or {}
    print("\n[inspect_provider_patch]")
    print(
        f"ok={check.get('ok')}, shape={report.get('shape')}, dtype={report.get('dtype')}"
    )

    pooled = get_embedding(
        "remoteclip",
        spatial=spatial,
        temporal=temporal,
        output=OutputSpec.pooled(pooling="mean"),
        backend="gee",
        device=device,
    )
    _show_embedding("single/pooled", pooled)

    spatials = [
        PointBuffer(lon=121.5, lat=31.2, buffer_m=512),
        PointBuffer(lon=120.5, lat=30.2, buffer_m=512),
    ]
    batch = get_embeddings_batch(
        "remoteclip",
        spatials=spatials,
        temporal=temporal,
        output=OutputSpec.pooled(pooling="mean"),
        backend="gee",
        device=device,
    )
    _show_batch("batch/pooled", batch)

    print("\n=== Modality quickstart (on-the-fly: terrafm / s1) ===")
    terrafm_s1 = get_embedding(
        "terrafm",
        spatial=spatial,
        temporal=temporal,
        sensor=SensorSpec(
            collection="COPERNICUS/S1_GRD_FLOAT",
            bands=("VV", "VH"),
            scale_m=10,
            composite="median",
            use_float_linear=True,
        ),
        modality="s1",
        output=OutputSpec.pooled(),
        backend="gee",
        device=device,
    )
    _show_embedding("single/pooled/terrafm_s1", terrafm_s1)

    if run_export:
        gee_out = out_dir / "gee_export"
        gee_out.mkdir(parents=True, exist_ok=True)
        manifests = export_batch(
            out=str(gee_out),
            layout="per_item",
            names=["p1", "p2"],
            spatials=spatials,
            temporal=temporal,
            models=["remoteclip"],
            output=OutputSpec.pooled(),
            backend="gee",
            device=device,
            save_inputs=True,
            save_embeddings=True,
            save_manifest=True,
            chunk_size=8,
            num_workers=4,
            resume=True,
            show_progress=True,
        )
        print(f"\n[export/gee] wrote {len(manifests)} items to: {gee_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rs-embed quickstart")
    parser.add_argument(
        "--mode",
        choices=("auto", "gee", "all"),
        default="auto",
        help="Which workflow to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Model device for on-the-fly models: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--run-export",
        action="store_true",
        help="Also run export_batch examples (slower).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/_outputs/quickstart"),
        help="Base output directory for export examples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("auto", "all"):
        run_auto_demo(run_export=args.run_export, out_dir=args.out_dir)
    if args.mode in ("gee", "all"):
        run_gee_demo(
            device=args.device, run_export=args.run_export, out_dir=args.out_dir
        )

    print("\nQuickstart finished.")


if __name__ == "__main__":
    main()
