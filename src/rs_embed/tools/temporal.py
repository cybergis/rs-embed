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
