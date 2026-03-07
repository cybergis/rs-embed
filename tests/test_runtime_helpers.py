"""Tests for rs_embed.internal.api.runtime_helpers — run_with_retry."""

import time

import pytest

from rs_embed.internal.api.runtime_helpers import run_with_retry


# ── run_with_retry: success paths ──────────────────────────────────


def test_retry_succeeds_immediately():
    assert run_with_retry(lambda: 42) == 42


def test_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return "ok"

    assert run_with_retry(flaky, retries=2) == "ok"
    assert calls["n"] == 2


def test_retry_succeeds_on_last_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("fail")
        return "done"

    assert run_with_retry(flaky, retries=3) == "done"
    assert calls["n"] == 4  # 1 initial + 3 retries


# ── run_with_retry: failure paths ──────────────────────────────────


def test_retry_zero_retries_raises_immediately():
    with pytest.raises(ValueError, match="boom"):
        run_with_retry(lambda: (_ for _ in ()).throw(ValueError("boom")), retries=0)


def test_retry_exhausted_raises_original_exception():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError(f"fail #{calls['n']}")

    with pytest.raises(RuntimeError, match="fail #3"):
        run_with_retry(always_fail, retries=2)
    assert calls["n"] == 3  # 1 initial + 2 retries


def test_retry_preserves_exception_type():
    class CustomError(Exception):
        pass

    with pytest.raises(CustomError):
        run_with_retry(lambda: (_ for _ in ()).throw(CustomError("x")), retries=1)


# ── run_with_retry: backoff timing ────────────────────────────────


def test_retry_backoff_sleeps_exponentially():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("fail")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        run_with_retry(always_fail, retries=2, backoff_s=0.05)
    elapsed = time.monotonic() - t0
    # backoff = 0.05 * 2^0 + 0.05 * 2^1 = 0.05 + 0.10 = 0.15s
    assert elapsed >= 0.12  # allow some tolerance
    assert elapsed < 1.0  # sanity upper bound


def test_retry_zero_backoff_does_not_sleep():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("fail")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError):
        run_with_retry(always_fail, retries=3, backoff_s=0.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5


# ── run_with_retry: edge cases ────────────────────────────────────


def test_retry_negative_retries_treated_as_zero():
    calls = {"n": 0}

    def always_fail():
        calls["n"] += 1
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        run_with_retry(always_fail, retries=-5)
    assert calls["n"] == 1


def test_retry_negative_backoff_treated_as_zero():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("fail")
        return "ok"

    t0 = time.monotonic()
    assert run_with_retry(flaky, retries=1, backoff_s=-10.0) == "ok"
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5
