#!/usr/bin/env python3
"""Phase 1 unit tests for logic that needs no camera.

Covers the handedness convention, the metrics window, and the HUD's tolerance of
degenerate landmarks. Hardware-dependent criteria (sustained fps, p95 latency, and the
visual handedness check) are verified by ``src/main.py``.

Run:  .venv\\Scripts\\python.exe tests\\test_phase1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hud                                              # noqa: E402
import landmarks                                        # noqa: E402
from metrics import Pipeline, Rate, Stat                # noqa: E402

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def test_handedness() -> None:
    print("handedness (mirror convention)")
    assert landmarks.MIRROR, "these expectations assume MIRROR=True"
    # We flip the frame before inference, so MediaPipe's label is image-space and must
    # be swapped to name the user's actual hand.
    check("image 'Left' -> user's Right", landmarks.true_handedness("Left") == "Right")
    check("image 'Right' -> user's Left", landmarks.true_handedness("Right") == "Left")
    check("swap is an involution",
          landmarks.true_handedness(landmarks.true_handedness("Left")) == "Left")
    check("unknown label passes through", landmarks.true_handedness("Neither") == "Neither")


def test_stat() -> None:
    print("metrics.Stat")
    s = Stat("t", window=100)
    check("empty -> (0,0)", s.percentiles() == (0.0, 0.0))
    for v in range(1, 101):
        s.add(float(v))
    p50, p95 = s.percentiles()
    check("p50 of 1..100 ~= 50.5", abs(p50 - 50.5) < 1.0, f"got {p50}")
    check("p95 of 1..100 ~= 95.05", abs(p95 - 95.05) < 1.0, f"got {p95}")
    check("p95 >= p50", p95 >= p50)

    small = Stat("t", window=10)          # window must evict old samples
    for v in range(100):
        small.add(float(v))
    check("window caps sample count", small.n == 10, f"got {small.n}")
    check("window keeps newest", small.percentiles()[0] > 90.0)


def test_rate() -> None:
    print("metrics.Rate")
    r = Rate(window_s=2.0)
    check("no samples -> 0 fps", r.fps == 0.0)
    t0 = 1000.0
    for i in range(61):                    # 60 fps for 1s, synthetic clock
        r.tick(t0 + i / 60.0)
    check("60 evenly spaced ticks -> ~60 fps", abs(r.fps - 60.0) < 1.0, f"got {r.fps}")
    for i in range(1, 121):                # 3s later at 30 fps; old ticks must fall out
        r.tick(t0 + 3.0 + i / 30.0)
    check("window drops stale ticks -> ~30 fps", abs(r.fps - 30.0) < 1.5, f"got {r.fps}")


def test_pipeline() -> None:
    print("metrics.Pipeline")
    p = Pipeline()
    p.record_result(capture_ts=1.000, submit_ts=1.001, done_ts=1.018, render_ts=1.020)
    check("cap->sub = 1ms", abs(p.capture_to_submit.percentiles()[0] - 1.0) < 0.05)
    check("sub->cb = 17ms", abs(p.submit_to_callback.percentiles()[0] - 17.0) < 0.05)
    check("cb->draw = 2ms", abs(p.callback_to_render.percentiles()[0] - 2.0) < 0.05)
    check("total = 20ms", abs(p.total.percentiles()[0] - 20.0) < 0.05)
    # A result whose submit was evicted must not pollute the stats with garbage.
    n_before = p.total.n
    p.record_result(0.0, 0.0, 1.018, 1.020)
    check("result with no matching submit is ignored", p.total.n == n_before)


def test_hud_robustness() -> None:
    print("hud drawing")
    img = np.zeros((480, 640, 3), np.uint8)
    lm = np.random.default_rng(0).random((21, 3)).astype(np.float32)
    hud.draw_hand(img, lm, "Left", 0.97)
    check("draws something", int(img.sum()) > 0)
    check("21 landmarks connected", len(hud.HAND_CONNECTIONS) == 21)
    check("both hands have colors", set(hud.HAND_COLORS) == {"Left", "Right"})
    # Landmarks stray outside [0,1] when a hand is partly out of frame; must not throw.
    wild = np.array([[-5.0, 12.0, 0.0]] * 21, dtype=np.float32)
    try:
        hud.draw_hand(img, wild, "Right", 0.5)
        hud.draw_stats(img, ["a", "bb"])
        hud.draw_hint(img, "hint")
        check("out-of-frame landmarks don't raise", True)
    except Exception as exc:                                   # pragma: no cover
        check("out-of-frame landmarks don't raise", False, repr(exc))


def main() -> int:
    print("=" * 58)
    print("Gesture Instrument - Phase 1 unit tests")
    print("=" * 58)
    for fn in (test_handedness, test_stat, test_rate, test_pipeline, test_hud_robustness):
        fn()
    print("-" * 58)
    if _failures:
        print(f"FAILED: {len(_failures)} -> {', '.join(_failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
