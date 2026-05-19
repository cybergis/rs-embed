from __future__ import annotations

import sys
import threading
from typing import Any


class NoOpProgress:
    def update(self, n: int = 1) -> None:
        _ = n

    def close(self) -> None:
        return None


class SimpleProgress:
    """Minimal fallback progress indicator when tqdm is unavailable."""

    def __init__(self, *, total: int, desc: str):
        self.total = max(0, int(total))
        self.desc = desc
        self.done = 0
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self.done = min(self.total, self.done + max(0, int(n)))
        pct = int((100 * self.done) / self.total)
        if pct == self._last_pct and self.done < self.total:
            return
        self._last_pct = pct

        width = 24
        fill = int((width * self.done) / self.total)
        bar = ("#" * fill) + ("." * (width - fill))
        sys.stderr.write(f"\r{self.desc} [{bar}] {self.done}/{self.total} ({pct:3d}%)")
        if self.done >= self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def close(self) -> None:
        if self.total > 0 and self.done < self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()


class FetchStats:
    """Thread-safe accumulator for GEE image fetch statistics.

    Updated by :class:`~rs_embed.pipelines.prefetch.PrefetchManager` during
    ``fetch_chunk`` and surfaced as log messages when progress reporting is on.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._completed = 0
        self._failed = 0
        self._cache_hits = 0
        self._last_point: int | None = None
        self._last_sensor: str | None = None

    @property
    def total(self) -> int:
        """Total GEE fetch operations planned across all chunks."""
        with self._lock:
            return self._total

    @property
    def completed(self) -> int:
        """Number of fetch operations that succeeded."""
        with self._lock:
            return self._completed

    @property
    def failed(self) -> int:
        """Number of fetch operations that failed."""
        with self._lock:
            return self._failed

    @property
    def cache_hits(self) -> int:
        """Number of fetch operations skipped due to cache reuse."""
        with self._lock:
            return self._cache_hits

    def record_planned(self, n: int = 1) -> None:
        """Register *n* newly planned fetch tasks."""
        with self._lock:
            self._total += max(0, int(n))

    def record_cache_hits(self, n: int = 1) -> None:
        """Register *n* fetches skipped due to a cache hit."""
        with self._lock:
            self._cache_hits += max(0, int(n))

    def record_success(self, *, point: int | None = None, sensor: str | None = None) -> None:
        """Register one successful fetch, optionally recording the point/sensor."""
        with self._lock:
            self._completed += 1
            if point is not None:
                self._last_point = point
            if sensor is not None:
                self._last_sensor = sensor

    def record_failure(self) -> None:
        """Register one failed fetch."""
        with self._lock:
            self._failed += 1

    def format_summary(self) -> str:
        """Return a compact summary line suitable for stderr logging."""
        with self._lock:
            t, c, f, h = self._total, self._completed, self._failed, self._cache_hits
            last_pt, last_s = self._last_point, self._last_sensor
        pct = int(100 * c / t) if t > 0 else 0
        msg = f"[gee_fetch] total={t} | done={c} ({pct}%) | failed={f} | cached={h}"
        if last_pt is not None and last_s is not None:
            msg += f" | last=point:{last_pt},sensor:{last_s}"
        return msg

    def log(self) -> None:
        """Write the current summary to stderr, respecting any active tqdm bar."""
        msg = self.format_summary()
        try:
            from tqdm import tqdm

            tqdm.write(msg, file=sys.stderr)
        except Exception:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()


def create_progress(*, enabled: bool, total: int, desc: str, unit: str = "item") -> Any:
    """Create a progress bar, falling back gracefully when tqdm is unavailable.

    Parameters
    ----------
    enabled : bool
        Whether to show progress at all. Returns a no-op object when ``False``.
    total : int
        Total number of steps.
    desc : str
        Label displayed next to the progress bar.
    unit : str
        Unit label per step (default ``"item"``).

    Returns
    -------
    Any
        A tqdm progress bar, a :class:`SimpleProgress` fallback, or a
        :class:`NoOpProgress` when disabled or ``total <= 0``.
    """
    if (not enabled) or int(total) <= 0:
        return NoOpProgress()

    try:
        from tqdm.auto import tqdm

        return tqdm(total=int(total), desc=desc, unit=unit, leave=False)
    except Exception as _e:
        return SimpleProgress(total=int(total), desc=desc)
