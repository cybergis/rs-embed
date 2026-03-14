from __future__ import annotations

"""Lightweight, on-the-fly input image inspection.

This module is intentionally dependency-light (numpy only by default).
It is used to sanity-check patches downloaded from Google Earth Engine (GEE)
right before they are fed into on-the-fly embedders.

You can enable checks in two ways:

1) Per-request via SensorSpec:
   SensorSpec(..., check_input=True)

2) Globally via environment variables:
   RS_EMBED_CHECK_INPUT=1
   RS_EMBED_CHECK_RAISE=1
   RS_EMBED_CHECK_SAVE_DIR=/tmp/rs_embed_checks

When enabled, we return a report dict that can be attached into embedding meta.
If `check_raise` is enabled and issues are detected, embedders may raise.
"""

import os
from typing import Any

import numpy as np

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in ("", "0", "false", "no", "off")

def checks_enabled(sensor: Any = None) -> bool:
    """Return True if input checks should run."""
    if _env_flag("RS_EMBED_CHECK_INPUT", "0"):
        return True
    return bool(getattr(sensor, "check_input", False))

def checks_should_raise(sensor: Any = None) -> bool:
    """Return True if embedders should raise on detected issues."""
    if "RS_EMBED_CHECK_RAISE" in os.environ:
        return _env_flag("RS_EMBED_CHECK_RAISE", "1")
    return bool(getattr(sensor, "check_raise", True))

def checks_save_dir(sensor: Any = None) -> str | None:
    """Optional directory to save quicklooks/stat dumps."""
    d = os.environ.get("RS_EMBED_CHECK_SAVE_DIR")
    if d:
        return str(d)
    return getattr(sensor, "check_save_dir", None)

def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception as _e:
        return None

def inspect_chw(
    x_chw: np.ndarray,
    *,
    name: str = "input",
    expected_channels: int | None = None,
    value_range: tuple[float, float] | None = None,
    fill_value: float | None = None,
    max_pixels_for_full_stats: int = 1_500_000,
    quantiles: tuple[float, ...] = (0.01, 0.5, 0.99),
    hist_bins: int = 32,
    hist_clip_range: tuple[float, float] | None = None,
    max_bands_for_hist: int = 16,
) -> dict[str, Any]:
    """Inspect a CHW numpy array and return a compact report.

    Parameters
    ----------
    x_chw:
        Input array with shape [C,H,W].
    expected_channels:
        If set, we flag a mismatch.
    value_range:
        If set, we compute fraction of values outside [lo, hi].
    fill_value:
        If set, we compute fraction of pixels equal to fill_value.
    max_pixels_for_full_stats:
        To keep inspection cheap, if C*H*W exceeds this value we downsample
        (strided sampling) for per-band stats.
    """
    report: dict[str, Any] = {
        "name": name,
        "ok": True,
        "issues": [],
        "shape": tuple(int(i) for i in getattr(x_chw, "shape", ())),
        "dtype": str(getattr(x_chw, "dtype", None)),
    }

    # Basic shape checks
    if not isinstance(x_chw, np.ndarray):
        report["ok"] = False
        report["issues"].append(f"{name}: not a numpy array")
        return report

    if x_chw.ndim != 3:
        report["ok"] = False
        report["issues"].append(f"{name}: expected CHW with ndim=3, got ndim={x_chw.ndim}")
        return report

    c, h, w = (int(x_chw.shape[0]), int(x_chw.shape[1]), int(x_chw.shape[2]))
    if expected_channels is not None and c != int(expected_channels):
        report["ok"] = False
        report["issues"].append(f"{name}: channel mismatch (C={c}, expected {expected_channels})")

    if h <= 0 or w <= 0:
        report["ok"] = False
        report["issues"].append(f"{name}: non-positive H/W ({h},{w})")
        return report

    # Downsample if huge (strided sampling across spatial dims)
    x = x_chw
    total = int(c) * int(h) * int(w)
    if total > int(max_pixels_for_full_stats):
        stride = int(np.ceil(np.sqrt(total / max_pixels_for_full_stats)))
        stride = max(stride, 1)
        x = x_chw[:, ::stride, ::stride]
        report["downsample_stride"] = stride

    xf = x.astype(np.float32, copy=False)

    finite = np.isfinite(xf)
    finite_frac = float(finite.mean())
    report["finite_frac"] = finite_frac
    if finite_frac < 0.999:
        report["ok"] = False
        n_bad = int((~finite).sum())
        report["issues"].append(f"{name}: contains NaN/Inf (count≈{n_bad} on sampled data)")

    # Replace non-finite for stats
    xf2 = np.where(finite, xf, np.nan)

    # Per-band stats
    bmin = np.nanmin(xf2, axis=(1, 2))
    bmax = np.nanmax(xf2, axis=(1, 2))
    bmean = np.nanmean(xf2, axis=(1, 2))
    bstd = np.nanstd(xf2, axis=(1, 2))

    report["band_min"] = [float(v) for v in bmin]
    report["band_max"] = [float(v) for v in bmax]
    report["band_mean"] = [float(v) for v in bmean]
    report["band_std"] = [float(v) for v in bstd]

    # Quantiles (computed once; export both legacy and compact fields).
    qs = tuple(float(q) for q in quantiles) if quantiles else ()
    qv = None
    if qs:
        try:
            qv = np.nanquantile(xf2, qs, axis=(1, 2))  # [Q, C]
            report["band_quantiles"] = {
                f"p{int(round(q * 100)):02d}": [float(v) for v in qv[i]] for i, q in enumerate(qs)
            }
            for qi, q in enumerate(qs):
                key = f"band_p{int(round(q * 100)):02d}"
                report[key] = [float(v) for v in qv[qi]]
            report["quantiles"] = list(qs)
        except Exception as e:
            report.setdefault("warnings", []).append(f"{name}: failed to compute quantiles: {e!r}")

    # Histograms (computed once; export both legacy and compact fields).
    if int(hist_bins) > 0 and c <= int(max_bands_for_hist):
        try:
            if hist_clip_range is not None:
                h_lo, h_hi = float(hist_clip_range[0]), float(hist_clip_range[1])
            elif value_range is not None:
                h_lo, h_hi = float(value_range[0]), float(value_range[1])
            elif isinstance(qv, np.ndarray) and qv.size > 0:
                lo_vals = qv[0]
                hi_vals = qv[-1]
                h_lo = float(np.nanmin(lo_vals))
                h_hi = float(np.nanmax(hi_vals))
            else:
                h_lo = float(np.nanmin(bmin))
                h_hi = float(np.nanmax(bmax))

            if not np.isfinite(h_lo) or not np.isfinite(h_hi) or (h_hi <= h_lo):
                raise ValueError(f"bad hist range ({h_lo},{h_hi})")

            edges = np.linspace(h_lo, h_hi, int(hist_bins) + 1, dtype=np.float32)
            counts = []
            for bi in range(c):
                v = xf2[bi].ravel()
                v = v[np.isfinite(v)]
                if v.size == 0:
                    counts.append([0] * int(hist_bins))
                    continue
                h, _ = np.histogram(v, bins=edges)
                counts.append([int(x) for x in h])

            # Current compact fields
            report["hist_bins"] = [float(x) for x in edges]
            report["band_hist"] = counts
            report["hist_range"] = [float(h_lo), float(h_hi)]
            # Backward-compatible structure
            report["hist"] = {
                "bins": report["hist_bins"],
                "counts": counts,
                "range": report["hist_range"],
            }
        except Exception as e:
            report.setdefault("warnings", []).append(f"{name}: histogram failed: {e!r}")

    # Constant / near-constant bands are often a sign of empty ROI, fill, or bad reprojection
    const = (bstd < 1e-6) | (~np.isfinite(bstd))
    if bool(const.any()):
        report["ok"] = False
        idx = np.where(const)[0].tolist()
        report["issues"].append(f"{name}: near-constant bands at indices {idx}")

    # Value range checks
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        outside = (xf2 < lo) | (xf2 > hi)
        outside_frac = float(np.nanmean(outside))
        report["outside_range_frac"] = outside_frac
        if outside_frac > 0.001:
            report["ok"] = False
            report["issues"].append(
                f"{name}: values outside range [{lo},{hi}] (frac≈{outside_frac:.4f} on sampled data)"
            )

    # Fill ratio checks
    fv = _safe_float(fill_value)
    if fv is not None:
        # equality on float is OK here because fill_value is often 0.0
        fill_mask = xf2 == fv
        fill_frac = float(np.nanmean(fill_mask))
        report["fill_value"] = fv
        report["fill_frac"] = fill_frac
        if fill_frac > 0.98:
            report["ok"] = False
            report["issues"].append(
                f"{name}: almost all pixels are fill_value={fv} (frac≈{fill_frac:.4f} on sampled data)"
            )

    return report

def maybe_inspect_chw(
    x_chw: np.ndarray,
    *,
    sensor: Any = None,
    name: str = "input",
    expected_channels: int | None = None,
    value_range: tuple[float, float] | None = None,
    fill_value: float | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run inspect_chw if enabled; optionally attach report to meta.

    Returns the report dict (or None if checks disabled).
    """
    if not checks_enabled(sensor):
        return None

    report = inspect_chw(
        x_chw,
        name=name,
        expected_channels=expected_channels,
        value_range=value_range,
        fill_value=fill_value,
    )

    if meta is not None:
        # Avoid huge meta: keep a single report object under a stable key.
        meta.setdefault("input_checks", {})
        meta["input_checks"][name] = report

        # Store the inspection config for reproducibility
        meta.setdefault("input_checks_config", {})
        meta["input_checks_config"].update(
            {
                "enabled": True,
                "raise": checks_should_raise(sensor),
                "save_dir": checks_save_dir(sensor),
            }
        )

    return report

def save_quicklook_rgb(
    x_chw: np.ndarray,
    *,
    path: str,
    bands=(0, 1, 2),
    pmin: float = 2.0,
    pmax: float = 98.0,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if x_chw.ndim != 3:
        raise ValueError(f"Expected CHW, got {x_chw.shape}")
    c, h, w = x_chw.shape
    r, g, b = bands
    if max(r, g, b) >= c:
        raise ValueError(f"bands={bands} out of range for C={c}")

    rgb = np.stack([x_chw[r], x_chw[g], x_chw[b]], axis=-1).astype(np.float32)

    # Clean non-finite values before stretching.
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.empty_like(rgb)
    if vmin is not None and vmax is not None:
        lo = float(vmin)
        hi = float(vmax)
        scale = max(hi - lo, 1e-6)
        out = (rgb - lo) / scale
    else:
        # Robust percentile stretch per channel.
        for ch in range(3):
            lo, hi = np.percentile(rgb[..., ch], (pmin, pmax))
            if hi <= lo + 1e-6:
                out[..., ch] = 0.0
            else:
                out[..., ch] = (rgb[..., ch] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(out, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
