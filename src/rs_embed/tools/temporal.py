from __future__ import annotations

from datetime import date, datetime, timedelta

from ..core.errors import SpecError
from ..core.specs import TemporalSpec


def temporal_to_start_end(temporal: TemporalSpec) -> tuple[str, str]:
    temporal.validate()
    if temporal.mode == "range":
        if temporal.start is None or temporal.end is None:
            raise SpecError("TemporalSpec.range requires start and end.")
        return str(temporal.start), str(temporal.end)
    if temporal.mode == "year":
        if temporal.year is None:
            raise SpecError("TemporalSpec.year requires year.")
        y = int(temporal.year)
        return f"{y}-01-01", f"{y + 1}-01-01"
    raise SpecError(f"Unknown TemporalSpec mode: {temporal.mode}")


def split_date_range(start: str, end: str, n_parts: int) -> tuple[tuple[str, str], ...]:
    """Split [start, end) into n non-empty date bins using end-exclusive semantics."""
    s = date.fromisoformat(str(start))
    e = date.fromisoformat(str(end))
    if e <= s:
        raise SpecError(f"Invalid date range: start={start}, end={end}")

    total_days = max(1, (e - s).days)
    n = max(1, int(n_parts))
    bounds = [s + timedelta(days=(total_days * i) // n) for i in range(n + 1)]
    bounds[-1] = e

    out = []
    for i in range(n):
        a = bounds[i]
        b = bounds[i + 1]
        if b <= a:
            b = min(e, a + timedelta(days=1))
        if b <= a:
            continue
        out.append((a.isoformat(), b.isoformat()))
    if not out:
        out.append((str(start), str(end)))
    return tuple(out)


def split_temporal_range(temporal: TemporalSpec, n_parts: int) -> tuple[tuple[str, str], ...]:
    start, end = temporal_to_start_end(temporal)
    return split_date_range(start, end, n_parts)


def split_date_range_fixed_days(
    start: str,
    end: str,
    *,
    stride_days: int = 30,
    max_bins: int | None = None,
) -> tuple[tuple[str, str], ...]:
    """Split [start, end) into consecutive fixed-length bins anchored at *start*.

    Unlike :func:`split_date_range` (equal division into *n* parts), every bin
    here is exactly *stride_days* long except the last, which is truncated at
    *end*. This mirrors temporal binning schemes that use a fixed stride from
    an arbitrary anchor date (e.g. OlmoEarth's 30-day pretraining frames).

    When *max_bins* is set and the range produces more bins, the trailing bins
    are dropped (the caller is responsible for surfacing the truncation).
    """
    s = date.fromisoformat(str(start))
    e = date.fromisoformat(str(end))
    if e <= s:
        raise SpecError(f"Invalid date range: start={start}, end={end}")
    if int(stride_days) < 1:
        raise SpecError(f"stride_days must be >= 1, got {stride_days}")

    out: list[tuple[str, str]] = []
    cur = s
    while cur < e:
        if max_bins is not None and len(out) >= int(max_bins):
            break
        nxt = min(e, cur + timedelta(days=int(stride_days)))
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return tuple(out)


def midpoint_date(start: str, end: str) -> str:
    start_dt = datetime.fromisoformat(str(start))
    end_dt = datetime.fromisoformat(str(end))
    if end_dt <= start_dt:
        raise SpecError(f"Invalid date range for midpoint: start={start}, end={end}")
    mid_dt = start_dt + (end_dt - start_dt) / 2
    return mid_dt.date().isoformat()


def temporal_frame_midpoints(temporal: TemporalSpec, n_frames: int) -> tuple[str, ...]:
    bins = split_temporal_range(temporal, n_frames)
    return tuple(midpoint_date(a, b) for a, b in bins)
