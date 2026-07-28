"""Generate synthetic landmark streams with realistic transitions and jitter.

The phantom-note problem only appears *between* poses: going 2 -> 5 the hand physically
passes through 3 and 4, and the classifier correctly reports them. Static test poses can
never reproduce that, so this builds continuous streams by interpolating between poses at
a fixed framerate — which passes through exactly those intermediate curl values.

Landmark jitter is added because real MediaPipe output is never still, and a debounce
that only works on perfectly stable input is worthless.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import features as F                                     # noqa: E402
from synthhand import synthetic_hand, to_screen          # noqa: E402

FPS = 30.0
JITTER = 0.0015          # per-landmark gaussian noise, in palm-scale units


def pose(count: int, pinch: bool = False) -> np.ndarray:
    """Landmarks for a held finger count, optionally pinching.

    Counts 0-4 use the fingers only, 5 adds an abducted thumb — matching how
    `features.finger_count` is structured so the thumb is never load-bearing.
    """
    if not 0 <= count <= 5:
        raise ValueError(count)
    extended = set(F.FINGER_NAMES[:min(count, 4)])
    return synthetic_hand(extended, thumb_out=(count == 5), pinch=pinch)


class Stream:
    """Builds a list of (t, landmarks) frames at a fixed framerate."""

    def __init__(self, fps: float = FPS, jitter: float = JITTER, seed: int = 0) -> None:
        self.dt = 1.0 / fps
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        self.frames: list[tuple[float, np.ndarray, tuple[float, float]]] = []
        self.t = 0.0
        self._last: np.ndarray | None = None
        self.marks: list[tuple[str, float]] = []      # (name, t) for asserting timing
        self.center = (0.5, 0.5)                      # screen position of the hand

    def at(self, x: float, y: float) -> "Stream":
        """Place the hand on screen. Mode C reads pitch from this y."""
        self.center = (x, y)
        return self

    def _emit(self, lm: np.ndarray) -> None:
        noisy = lm + self.rng.normal(0.0, self.jitter, lm.shape).astype(np.float32)
        self.frames.append((self.t, noisy.astype(np.float32), self.center))
        self.t += self.dt
        self._last = lm

    def hold(self, lm: np.ndarray, seconds: float) -> "Stream":
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self._emit(lm)
        return self

    def move_to(self, lm: np.ndarray, seconds: float) -> "Stream":
        """Interpolate from the current pose — this is what creates transitional poses."""
        start = self._last if self._last is not None else lm
        n = max(1, int(round(seconds / self.dt)))
        for i in range(1, n + 1):
            a = i / n
            self._emit((1.0 - a) * start + a * lm)
        return self

    def mark(self, name: str) -> "Stream":
        self.marks.append((name, self.t))
        return self

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def duration(self) -> float:
        return self.t


def features_stream(stream: Stream) -> list[tuple[float, F.HandFeatures]]:
    """Run a stream through the feature layer exactly as the live pipeline does."""
    out: list[tuple[float, F.HandFeatures]] = []
    prev: np.ndarray | None = None
    prev_t = 0.0
    for t, lm, center in stream.frames:
        dt = (t - prev_t) if prev is not None else 0.0
        out.append((t, F.extract(lm, to_screen(lm, center=center), prev, dt)))
        prev, prev_t = lm, t
    return out


def run(stream: Stream, tracker) -> list:
    """Feed a stream through a HandTracker and collect every event."""
    events = []
    for t, f in features_stream(stream):
        events.extend(tracker.update(f, t))
    return events


def sweep_with_pinches(n_pinches: int = 10, seed: int = 0) -> Stream:
    """The plan's Phase 3 exit scenario: sweep 0->5->0 while pinching at known moments.

    Each sweep passes through every intermediate count, so a naive implementation fires a
    machine-gun arpeggio here. Pinches happen at marked times on a settled pose.
    """
    s = Stream(seed=seed)
    s.hold(pose(0), 0.4)
    for i in range(n_pinches):
        # Sweep up through 1,2,3,4 to 5, then back down: every intermediate count is
        # visited twice per iteration.
        for target in (1, 2, 3, 4, 5):
            s.move_to(pose(target), 0.12).hold(pose(target), 0.06)
        for target in (4, 3, 2, 1, 0):
            s.move_to(pose(target), 0.12).hold(pose(target), 0.06)
        # Settle, then one deliberate pinch.
        s.hold(pose(1), 0.20)
        s.mark(f"pinch{i}")
        s.move_to(pose(1, pinch=True), 0.05).hold(pose(1, pinch=True), 0.15)
        s.move_to(pose(1), 0.05).hold(pose(1), 0.25)
    return s
