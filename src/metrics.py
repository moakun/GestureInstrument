"""Rolling latency/rate instrumentation.

Plan 1.4: log p50/p95 for every pipeline stage from day one. "If you only add this in
Phase 7 you will have already made three untraceable performance mistakes."

Everything here is O(1) per sample and allocation-light, so it can run in the hot loop.
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np


class Stat:
    """A rolling window of millisecond samples with p50/p95."""

    __slots__ = ("name", "_buf")

    def __init__(self, name: str, window: int = 180) -> None:
        self.name = name
        self._buf: deque[float] = deque(maxlen=window)

    def add(self, ms: float) -> None:
        self._buf.append(ms)

    @property
    def n(self) -> int:
        return len(self._buf)

    def percentiles(self) -> tuple[float, float]:
        """(p50, p95) in ms; (0, 0) when empty."""
        if not self._buf:
            return (0.0, 0.0)
        a = np.fromiter(self._buf, dtype=np.float32, count=len(self._buf))
        return (float(np.percentile(a, 50)), float(np.percentile(a, 95)))

    def __str__(self) -> str:
        p50, p95 = self.percentiles()
        return f"{self.name} p50={p50:5.1f} p95={p95:5.1f}"


class Rate:
    """Sliding-window event rate (fps)."""

    __slots__ = ("_ts", "_window")

    def __init__(self, window_s: float = 2.0) -> None:
        self._ts: deque[float] = deque()
        self._window = window_s

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._ts.append(now)
        cutoff = now - self._window
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class Pipeline:
    """The four stages the plan asks for, plus capture and render rates."""

    def __init__(self, window: int = 180) -> None:
        self.capture_to_submit = Stat("cap->sub", window)
        self.submit_to_callback = Stat("sub->cb ", window)
        self.callback_to_render = Stat("cb->draw", window)
        self.total = Stat("TOTAL   ", window)
        self.capture_fps = Rate()
        self.render_fps = Rate()
        self.infer_fps = Rate()

    def record_result(self, capture_ts: float, submit_ts: float, done_ts: float,
                      render_ts: float) -> None:
        """Record one end-to-end pass. Times are perf_counter seconds."""
        if capture_ts <= 0.0 or submit_ts <= 0.0:
            return                                  # result with no matching submit
        self.capture_to_submit.add((submit_ts - capture_ts) * 1e3)
        self.submit_to_callback.add((done_ts - submit_ts) * 1e3)
        self.callback_to_render.add((render_ts - done_ts) * 1e3)
        self.total.add((render_ts - capture_ts) * 1e3)

    def summary(self) -> str:
        return (f"cap {self.capture_fps.fps:4.1f}fps | "
                f"infer {self.infer_fps.fps:4.1f}fps | "
                f"draw {self.render_fps.fps:4.1f}fps | "
                + " | ".join(str(s) for s in (self.capture_to_submit,
                                              self.submit_to_callback,
                                              self.callback_to_render,
                                              self.total)))
