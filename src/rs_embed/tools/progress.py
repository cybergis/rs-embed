from __future__ import annotations

import sys
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


def create_progress(*, enabled: bool, total: int, desc: str, unit: str = "item") -> Any:
    if (not enabled) or int(total) <= 0:
        return NoOpProgress()

    try:
        from tqdm.auto import tqdm

        return tqdm(total=int(total), desc=desc, unit=unit, leave=False)
    except Exception as _e:
        return SimpleProgress(total=int(total), desc=desc)
