"""Retry utilities and parallel execution runner."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, TypeVar

_T = TypeVar("_T")


def run_with_retry(
    fn: Callable[[], _T],
    *,
    retries: int = 0,
    backoff_s: float = 0.0,
) -> _T:
    """Run *fn* with bounded retries and optional exponential backoff."""
    tries = max(0, int(retries))
    backoff = max(0.0, float(backoff_s))
    last_err: Optional[Exception] = None
    for attempt in range(tries + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            if attempt < tries and backoff > 0:
                time.sleep(backoff * (2**attempt))
    raise last_err  # type: ignore[misc]


class ParallelRunner:
    """Thread pool with retry logic and optional progress tracking.

    Use as a context manager::

        with ParallelRunner(num_workers=4, max_retries=2) as runner:
            results = runner.map_unordered(fn, items)
    """

    def __init__(
        self,
        num_workers: int = 8,
        max_retries: int = 0,
        retry_backoff_s: float = 0.0,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._executor: Optional[ThreadPoolExecutor] = None

    # -- context manager -------------------------------------------------

    def __enter__(self) -> ParallelRunner:
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # -- public API ------------------------------------------------------

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)
        return self._executor

    def submit(self, fn: Callable[..., _T], *args: Any) -> Future[_T]:
        return self.executor.submit(self._retry_wrap(fn), *args)

    def map_unordered(
        self,
        fn: Callable[..., _T],
        items: List[Any],
        *,
        progress: Any = None,
    ) -> Dict[int, _T]:
        """Run *fn(item)* for each item, returning ``{index: result}``."""
        wrapped = self._retry_wrap(fn)
        fut_to_idx = {
            self.executor.submit(wrapped, item): idx
            for idx, item in enumerate(items)
        }
        results: Dict[int, _T] = {}
        for fut in as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            results[idx] = fut.result()
            if progress is not None:
                progress.update(1)
        return results

    # -- internals -------------------------------------------------------

    def _retry_wrap(self, fn: Callable[..., _T]) -> Callable[..., _T]:
        if self.max_retries <= 0:
            return fn

        def _wrapped(*args: Any, **kwargs: Any) -> _T:
            return run_with_retry(
                lambda: fn(*args, **kwargs),
                retries=self.max_retries,
                backoff_s=self.retry_backoff_s,
            )

        return _wrapped
