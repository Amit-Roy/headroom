"""Cooperative process-CPU pacing for selected native inference phases."""

from __future__ import annotations

import contextlib
import math
import os
import threading
import time
from collections.abc import Iterator

CPU_TARGET_PERCENT_ENV = "HEADROOM_CPU_TARGET_PERCENT"
_phase_lock = threading.Lock()
_thread_state = threading.local()


def cumulative_pacing_wait_seconds() -> float:
    """Return this thread's cumulative time spent in pacing sleeps."""
    return getattr(_thread_state, "pacing_wait_seconds", 0.0)


def _target_fraction() -> float | None:
    raw = os.environ.get(CPU_TARGET_PERCENT_ENV)
    if raw is None:
        return None
    try:
        target = float(raw)
    except ValueError as exc:
        raise ValueError(f"{CPU_TARGET_PERCENT_ENV} must be a finite number from 1 to 100") from exc
    if not math.isfinite(target) or (target != 0 and (target < 1 or target > 100)):
        raise ValueError(f"{CPU_TARGET_PERCENT_ENV} must be 0 or a finite number from 1 to 100")
    return None if target == 0 else target / 100.0


@contextlib.contextmanager
def paced_cpu_phase() -> Iterator[None]:
    """Pace a synchronous CPU phase toward a process CPU percentage target."""
    target = _target_fraction()
    if target is None:
        yield
        return

    with _phase_lock:
        wall_start = time.monotonic()
        cpu_start = time.process_time()
        try:
            yield
        finally:
            wall_elapsed = time.monotonic() - wall_start
            cpu_elapsed = time.process_time() - cpu_start
            delay = cpu_elapsed / target - wall_elapsed
            if delay > 0:
                sleep_start = time.monotonic()
                try:
                    time.sleep(delay)
                finally:
                    slept = max(0.0, time.monotonic() - sleep_start)
                    _thread_state.pacing_wait_seconds = cumulative_pacing_wait_seconds() + slept
