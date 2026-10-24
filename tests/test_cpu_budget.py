"""Tests for cooperative CPU pacing."""

from __future__ import annotations

import contextlib
import threading

import pytest

from headroom import cpu_budget
from headroom.transforms import kompress_compressor


def test_paces_to_configured_cpu_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", "60")
    monotonic = iter((10.0, 11.0, 11.0, 11.0))
    process_time = iter((20.0, 21.0))
    sleeps: list[float] = []
    monkeypatch.setattr(cpu_budget.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(cpu_budget.time, "process_time", lambda: next(process_time))
    monkeypatch.setattr(cpu_budget.time, "sleep", sleeps.append)

    with cpu_budget.paced_cpu_phase():
        pass

    assert sleeps == pytest.approx([2 / 3])


def test_does_not_sleep_when_wall_time_already_covers_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", "60")
    monotonic = iter((10.0, 12.0))
    process_time = iter((20.0, 21.0))
    sleeps: list[float] = []
    monkeypatch.setattr(cpu_budget.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(cpu_budget.time, "process_time", lambda: next(process_time))
    monkeypatch.setattr(cpu_budget.time, "sleep", sleeps.append)

    with cpu_budget.paced_cpu_phase():
        pass

    assert sleeps == []


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0.5", "100.1", "bad"])
def test_rejects_invalid_cpu_targets(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", value)
    with pytest.raises(ValueError, match="HEADROOM_CPU_TARGET_PERCENT"):
        with cpu_budget.paced_cpu_phase():
            pass


def test_exception_still_paces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", "50")
    monotonic = iter((0.0, 1.0, 1.0, 1.0))
    process_time = iter((0.0, 1.0))
    sleeps: list[float] = []
    monkeypatch.setattr(cpu_budget.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(cpu_budget.time, "process_time", lambda: next(process_time))
    monkeypatch.setattr(cpu_budget.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="inference failed"):
        with cpu_budget.paced_cpu_phase():
            raise RuntimeError("inference failed")

    assert sleeps == pytest.approx([1.0])
    assert cpu_budget._phase_lock.acquire(blocking=False)
    cpu_budget._phase_lock.release()


def test_onnx_get_scores_wraps_run_and_preserves_output_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = object()
    calls: list[bool] = []

    class Session:
        def run(self, _outputs: list[str], _inputs: dict[str, object]) -> list[object]:
            calls.append(True)
            return [result]

    @contextlib.contextmanager
    def pacer():
        calls.append(False)
        yield

    monkeypatch.setattr(kompress_compressor, "paced_cpu_phase", pacer)
    scores = kompress_compressor._OnnxModel(Session()).get_scores([[1]], [[1]])

    assert scores is result
    assert calls == [False, True]


def test_smoke_run_wraps_session_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    class Session:
        def run(self, _outputs: list[str], _inputs: dict[str, object]) -> None:
            calls.append(True)

    @contextlib.contextmanager
    def pacer():
        calls.append(False)
        yield

    monkeypatch.setattr(kompress_compressor, "paced_cpu_phase", pacer)
    kompress_compressor._smoke_run(Session())

    assert calls == [False, True]


@pytest.mark.parametrize(
    ("target", "native_elapsed", "expected_elapsed", "expected_sleep"),
    [(10, 0.1, 0.1, 0.9), (100, 6.0, 6.0, 0.0)],
)
def test_timed_canary_excludes_pacing_but_keeps_native_time(
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    native_elapsed: float,
    expected_elapsed: float,
    expected_sleep: float,
) -> None:
    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", str(target))
    clocks = {"wall": 0.0, "cpu": 0.0}
    sleeps: list[float] = []

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clocks["wall"] += delay

    class Session:
        def run(self, _outputs: list[str], _inputs: dict[str, object]) -> list[object]:
            clocks["wall"] += native_elapsed
            clocks["cpu"] += native_elapsed
            return [[[0.8, 0.8]]]

    class Tokenizer:
        def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    monkeypatch.setattr(cpu_budget.time, "monotonic", lambda: clocks["wall"])
    monkeypatch.setattr(cpu_budget.time, "process_time", lambda: clocks["cpu"])
    monkeypatch.setattr(cpu_budget.time, "perf_counter", lambda: clocks["wall"])
    monkeypatch.setattr(cpu_budget.time, "sleep", sleep)
    monkeypatch.setattr(kompress_compressor, "_model_device_type", lambda *_args: "cpu")
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    monkeypatch.setattr(
        kompress_compressor,
        "_acquire_execution_slot",
        lambda *_args, **_kwargs: (semaphore, 0.0),
    )

    compressor = kompress_compressor.KompressCompressor()
    elapsed = compressor._timed_canary(
        kompress_compressor._OnnxModel(Session()), Tokenizer(), "onnx"
    )

    assert elapsed == pytest.approx(expected_elapsed)
    assert sum(sleeps) == pytest.approx(expected_sleep)


def test_unset_or_zero_target_disables_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.delenv("HEADROOM_CPU_TARGET_PERCENT", raising=False)
    monkeypatch.setattr(cpu_budget.time, "sleep", sleeps.append)
    with cpu_budget.paced_cpu_phase():
        pass

    monkeypatch.setenv("HEADROOM_CPU_TARGET_PERCENT", "0")
    with cpu_budget.paced_cpu_phase():
        pass
    assert sleeps == []
