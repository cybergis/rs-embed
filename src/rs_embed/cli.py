"""Command-Line Interface.

Provides ``python -m rs_embed`` (via :mod:`rs_embed.__main__`) with the
following subcommands:

- ``inspect-gee`` — download a patch from Google Earth Engine and print an
  input-inspection report (no model run).
- ``export-npz``  — export raw inputs and/or embeddings for one or more
  models into ``.npz`` + JSON manifest files.
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.specs import BBox, OutputSpec, PointBuffer, SensorSpec, TemporalSpec
from .export import export_npz
from .inspect import inspect_gee_patch


def _parse_bands(s: str) -> tuple[str, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--bands must be a comma-separated list, e.g. 'B4,B3,B2'")
    return tuple(parts)

def _parse_models(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "--models must be a comma-separated list, e.g. 'remoteclip,prithvi'"
        )
    return parts

def _parse_value_range(s: str | None) -> tuple[float, float] | None:
    if not s:
        return None
    try:
        lo, hi = s.split(",")
        return (float(lo), float(hi))
    except Exception as e:
        raise argparse.ArgumentTypeError("--value-range must be 'lo,hi' (floats)") from e

def _add_spatial_args(p: argparse.ArgumentParser) -> None:
    sp = p.add_mutually_exclusive_group(required=True)
    sp.add_argument(
        "--bbox",
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        nargs=4,
        type=float,
        help="EPSG:4326 bbox",
    )
    sp.add_argument(
        "--pointbuffer",
        metavar=("LON", "LAT", "BUFFER_M"),
        nargs=3,
        type=float,
        help="EPSG:4326 pointbuffer (meters)",
    )

def _parse_spatial(args) -> BBox | PointBuffer:
    if args.bbox is not None:
        return BBox(*args.bbox)
    lon, lat, buf = args.pointbuffer
    return PointBuffer(lon=lon, lat=lat, buffer_m=buf)

def _add_temporal_args(p: argparse.ArgumentParser) -> None:
    tg = p.add_mutually_exclusive_group(required=False)
    tg.add_argument("--year", type=int, help="Year mode (will use [year-01-01, year+1-01-01)")
    tg.add_argument(
        "--range",
        metavar=("START", "END"),
        nargs=2,
        help="Date range, e.g. 2022-06-01 2022-09-01",
    )

def _parse_temporal(args) -> TemporalSpec | None:
    if getattr(args, "year", None) is not None:
        return TemporalSpec.year(int(args.year))
    if getattr(args, "range", None) is not None:
        return TemporalSpec.range(args.range[0], args.range[1])
    return None

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rs-embed", description="rs-embed utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # inspect-gee
    # ------------------------------------------------------------------
    ig = sub.add_parser(
        "inspect-gee",
        help="Download a patch from Google Earth Engine and output an input-inspection report (no model run).",
    )

    ig.add_argument("--collection", required=True, help="GEE ImageCollection (or Image) id")
    ig.add_argument("--bands", required=True, type=_parse_bands, help="Comma-separated band list")
    ig.add_argument("--scale-m", type=int, default=10, help="Pixel scale (meters)")
    ig.add_argument(
        "--cloudy-pct",
        type=int,
        default=30,
        help="Best-effort cloud filter (CLOUDY_PIXEL_PERCENTAGE)",
    )
    ig.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Default fill value used by sampleRectangle",
    )
    ig.add_argument(
        "--composite",
        choices=["median", "mosaic"],
        default="median",
        help="How to composite collection",
    )

    _add_spatial_args(ig)
    _add_temporal_args(ig)

    ig.add_argument("--value-range", default=None, help="Optional sanity range 'lo,hi' for values")
    ig.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to save a quicklook PNG (first 3 bands)",
    )

    # ------------------------------------------------------------------
    # export-npz
    # ------------------------------------------------------------------
    ex = sub.add_parser(
        "export-npz",
        help="Export raw GEE inputs + embeddings for one or more models into a .npz plus a JSON manifest.",
    )
    ex.add_argument("--models", required=True, type=_parse_models, help="Comma-separated model IDs")
    ex.add_argument("--out", required=True, help="Output .npz path")

    _add_spatial_args(ex)
    _add_temporal_args(ex)

    ex.add_argument("--backend", default="gee", help="Backend (default: gee)")
    ex.add_argument("--device", default="auto", help="Device for model inference (default: auto)")

    ex.add_argument(
        "--output",
        choices=["pooled", "grid"],
        default="pooled",
        help="Embedding output mode",
    )
    ex.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
        help="Pooling method for pooled output",
    )

    # Optional global sensor overrides (applied to ALL models when provided)
    ex.add_argument(
        "--collection",
        default=None,
        help="Override sensor.collection for input download and embedding",
    )
    ex.add_argument(
        "--bands",
        default=None,
        type=_parse_bands,
        help="Override sensor bands (comma-separated)",
    )
    ex.add_argument("--scale-m", type=int, default=10, help="Override pixel scale (meters)")
    ex.add_argument("--cloudy-pct", type=int, default=30, help="Override cloud filter percentage")
    ex.add_argument("--fill-value", type=float, default=0.0, help="Override fill value")
    ex.add_argument(
        "--composite",
        choices=["median", "mosaic"],
        default="median",
        help="Override composite",
    )

    ex.add_argument(
        "--value-range",
        default=None,
        help="Optional sanity range 'lo,hi' for input checks",
    )

    ex.add_argument("--no-inputs", action="store_true", help="Do not save raw inputs into the npz")
    ex.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Do not run models / save embeddings",
    )
    ex.add_argument("--no-json", action="store_true", help="Do not write a sidecar .json manifest")
    ex.add_argument(
        "--fail-on-bad-input",
        action="store_true",
        help="If input checks fail, stop and raise instead of exporting.",
    )
    ex.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep exporting remaining models/items when one fails.",
    )
    ex.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry count for network fetch / inference / write failures.",
    )
    ex.add_argument(
        "--retry-backoff-s",
        type=float,
        default=0.0,
        help="Base backoff seconds between retries (exponential).",
    )

    return p

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cmd == "inspect-gee":
        sensor = SensorSpec(
            collection=args.collection,
            bands=args.bands,
            scale_m=args.scale_m,
            cloudy_pct=args.cloudy_pct,
            fill_value=args.fill_value,
            composite=args.composite,
            check_input=True,
            check_raise=False,
            check_save_dir=args.save_dir,
        )

        spatial = _parse_spatial(args)
        temporal = _parse_temporal(args)
        value_range = _parse_value_range(args.value_range)

        out = inspect_gee_patch(
            spatial=spatial, temporal=temporal, sensor=sensor, value_range=value_range
        )
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    if args.cmd == "export-npz":
        spatial = _parse_spatial(args)
        temporal = _parse_temporal(args)
        value_range = _parse_value_range(args.value_range)
        if value_range is not None:
            raise SystemExit("--value-range is currently supported only for inspect-gee.")

        output = (
            OutputSpec.pooled(pooling=args.pooling)
            if args.output == "pooled"
            else OutputSpec.grid()
        )

        sensor_override = None
        if args.collection is not None:
            if args.bands is None:
                raise SystemExit("--bands is required when --collection is provided for export-npz")
            sensor_override = SensorSpec(
                collection=args.collection,
                bands=args.bands,
                scale_m=args.scale_m,
                cloudy_pct=args.cloudy_pct,
                fill_value=args.fill_value,
                composite=args.composite,
                check_input=True,
                check_raise=False,
            )

        manifest = export_npz(
            out_path=args.out,
            models=args.models,
            spatial=spatial,
            temporal=temporal,
            sensor=sensor_override,
            output=output,
            backend=args.backend,
            device=args.device,
            save_inputs=not args.no_inputs,
            save_embeddings=not args.no_embeddings,
            save_manifest=not args.no_json,
            fail_on_bad_input=args.fail_on_bad_input,
            continue_on_error=args.continue_on_error,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
        )

        json.dump(manifest, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    raise SystemExit(f"Unknown command: {args.cmd}")

if __name__ == "__main__":
    main()
