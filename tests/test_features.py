#!/usr/bin/env python3
"""Phase 2 unit tests: feature math against synthetic hands with known geometry.

Checks exact values where the geometry pins them down, and — more importantly — the
*invariance properties* the whole design rests on: curls must not change when the hand
rotates, and ratios must not change when it moves closer to the camera.

Run:  .venv\\Scripts\\python.exe tests\\test_features.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import features as F                                    # noqa: E402
from synthhand import rotation, synthetic_hand, to_screen, transform   # noqa: E402

ALL = set(F.FINGER_NAMES)
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def test_palm_scale() -> None:
    print("palm_scale")
    lm = synthetic_hand(ALL)
    check("canonical hand -> 1.0", abs(F.palm_scale(lm) - 1.0) < 1e-3,
          f"got {F.palm_scale(lm)}")
    big = transform(lm, scale=2.5)
    check("scales linearly with hand size", abs(F.palm_scale(big) - 2.5) < 1e-3,
          f"got {F.palm_scale(big)}")
    moved = transform(lm, translate=(3.0, -2.0, 1.0))
    check("translation invariant", abs(F.palm_scale(moved) - 1.0) < 1e-3)


def test_curl_extremes() -> None:
    print("curl_cos extremes")
    straight = F.finger_curls(synthetic_hand(ALL))
    folded = F.finger_curls(synthetic_hand(set()))
    check("straight finger -> cos ~ -1",
          all(c < -0.99 for c in straight.values()), f"{straight}")
    check("folded finger -> cos ~ +1",
          all(c > 0.99 for c in folded.values()), f"{folded}")
    check("all four fingers reported", set(straight) == ALL)
    check("extremes straddle both thresholds",
          max(straight.values()) < F.CURL_EXTENDED < F.CURL_RETRACTED < min(folded.values()))


def test_rotation_invariance() -> None:
    print("rotation invariance (the point of using angles)")
    lm = synthetic_hand({"index", "middle"}, thumb_out=True)
    base_curls = F.finger_curls(lm)
    base_pinch, base_thumb = F.pinch_ratio(lm), F.thumb_open(lm)
    worst_curl = worst_pinch = worst_thumb = 0.0
    for rx, ry, rz in [(0.4, 0, 0), (0, 0.5, 0), (0, 0, 1.1), (0.3, -0.6, 0.9),
                       (math.pi / 2, 0, 0), (-0.8, 0.8, -0.8)]:
        rot = transform(lm, rot=rotation(rx, ry, rz))
        curls = F.finger_curls(rot)
        worst_curl = max(worst_curl, max(abs(curls[k] - base_curls[k]) for k in ALL))
        worst_pinch = max(worst_pinch, abs(F.pinch_ratio(rot) - base_pinch))
        worst_thumb = max(worst_thumb, abs(F.thumb_open(rot) - base_thumb))
    check("curls unchanged under 3D rotation", worst_curl < 1e-3, f"max drift {worst_curl:.2e}")
    check("pinch unchanged under 3D rotation", worst_pinch < 1e-3, f"max drift {worst_pinch:.2e}")
    check("thumb unchanged under 3D rotation", worst_thumb < 1e-3, f"max drift {worst_thumb:.2e}")


def test_scale_invariance() -> None:
    print("scale invariance (hand nearer/further from camera)")
    lm = synthetic_hand({"index"}, thumb_out=True)
    base = (F.pinch_ratio(lm), F.thumb_open(lm))
    worst = 0.0
    for s in (0.3, 0.75, 1.0, 2.0, 6.0):
        big = transform(lm, scale=s, translate=(s, -s, 0.5))
        worst = max(worst, abs(F.pinch_ratio(big) - base[0]),
                    abs(F.thumb_open(big) - base[1]))
    check("pinch/thumb ratios unchanged by distance", worst < 1e-3, f"max drift {worst:.2e}")


def test_thumb_and_pinch() -> None:
    print("thumb_open / pinch_ratio")
    out = F.thumb_open(synthetic_hand(ALL, thumb_out=True))
    tucked = F.thumb_open(synthetic_hand(ALL, thumb_out=False))
    check("abducted thumb reads open", out > F.THUMB_OPEN, f"got {out:.3f}")
    check("tucked thumb reads closed", tucked < F.THUMB_CLOSED, f"got {tucked:.3f}")

    pinched = F.pinch_ratio(synthetic_hand({"index"}, pinch=True))
    apart = F.pinch_ratio(synthetic_hand({"index"}, thumb_out=True))
    check("pinched -> below close threshold", pinched < F.PINCH_CLOSE, f"got {pinched:.3f}")
    check("open -> above release threshold", apart > F.PINCH_OPEN, f"got {apart:.3f}")
    check("pinch hysteresis band is ordered", F.PINCH_CLOSE < F.PINCH_OPEN)


def test_motion_energy() -> None:
    print("motion_energy (the phantom-note killer)")
    lm = synthetic_hand(ALL)
    check("no previous frame -> 0", F.motion_energy(lm, None, 0.033) == 0.0)
    check("dt <= 0 -> 0", F.motion_energy(lm, lm, 0.0) == 0.0)
    check("identical frames -> 0", F.motion_energy(lm, lm, 0.033) == 0.0)
    # Every landmark moves 0.1 palm-widths in 0.5 s -> 0.2 palm-widths/second.
    moved = transform(lm, translate=(0.1, 0.0, 0.0))
    got = F.motion_energy(moved, lm, 0.5)
    check("uniform translation -> d/(scale*dt)", abs(got - 0.2) < 1e-4, f"got {got:.4f}")
    check("faster motion reads higher",
          F.motion_energy(moved, lm, 0.1) > F.motion_energy(moved, lm, 0.5))
    # Scale invariance: a big hand moving proportionally reads the same energy.
    big_prev, big_now = transform(lm, scale=3.0), transform(moved, scale=3.0)
    check("motion is scale-invariant",
          abs(F.motion_energy(big_now, big_prev, 0.5) - 0.2) < 1e-4)
    check("mismatched shapes -> 0", F.motion_energy(lm, lm[:10], 0.033) == 0.0)


def test_hand_position() -> None:
    print("hand_y / hand_x (Mode C pitch axis)")
    lm = synthetic_hand(ALL)
    top = to_screen(lm, center=(0.5, 0.2))
    bottom = to_screen(lm, center=(0.5, 0.8))
    check("hand high on screen -> smaller y", F.hand_y(top) < F.hand_y(bottom))
    check("y stays in 0..1", 0.0 < F.hand_y(top) < 1.0 and 0.0 < F.hand_y(bottom) < 1.0)
    check("y tracks the offset", abs((F.hand_y(bottom) - F.hand_y(top)) - 0.6) < 1e-3,
          f"got {F.hand_y(bottom) - F.hand_y(top):.4f}")
    left = to_screen(lm, center=(0.25, 0.5))
    right = to_screen(lm, center=(0.75, 0.5))
    check("hand_x separates left from right", F.hand_x(left) < F.hand_x(right))


def test_finger_count() -> None:
    print("finger_count")
    cases = [
        (set(), False, 0),
        ({"index"}, False, 1),
        ({"index", "middle"}, False, 2),
        ({"index", "middle", "ring"}, False, 3),
        (ALL, False, 4),                 # four fingers, thumb tucked
        (ALL, True, 5),                  # ...plus an open thumb
    ]
    for ext, thumb, want in cases:
        got = F.finger_count(synthetic_hand(ext, thumb_out=thumb))
        check(f"{len(ext)} fingers, thumb_out={thumb} -> {want}", got == want, f"got {got}")


def test_thumb_not_load_bearing() -> None:
    print("thumb is never load-bearing (plan 2.2)")
    # A misread thumb must not change counts 0-3; it may only confuse 4 with 5.
    for n in range(4):
        ext = set(list(F.FINGER_NAMES)[:n])
        a = F.finger_count(synthetic_hand(ext, thumb_out=False))
        b = F.finger_count(synthetic_hand(ext, thumb_out=True))
        check(f"count {n} unaffected by thumb state", a == b == n, f"got {a} vs {b}")


def test_anisotropy_correction() -> None:
    print("isotropic() undoes camera aspect distortion")
    aspect = 640.0 / 480.0
    # A partially-bent finger: fully straight/folded fingers are exactly antiparallel/
    # parallel and so are immune to skew, which would hide the bug this test is for.
    lm = synthetic_hand(bend_deg=75.0, thumb_out=True)
    truth = F.finger_curls(lm)
    check("bend_deg gives the expected angle",
          abs(truth["index"] - (-math.cos(math.radians(75.0)))) < 1e-4,
          f"got {truth['index']:.4f}")

    # Rotating the hand in-plane must not change curls. Under raw (anisotropic) image
    # coords it does — that is exactly the tilt-sensitivity the plan warns about.
    raw_err = iso_err = 0.0
    for angle in (0.0, 0.5, 1.0, math.pi / 4, math.pi / 3):
        rot = transform(lm, rot=rotation(0, 0, angle))
        screen = to_screen(rot, aspect=aspect)
        raw = F.finger_curls(screen)
        fixed = F.finger_curls(F.isotropic(screen, aspect))
        raw_err = max(raw_err, max(abs(raw[k] - truth[k]) for k in ALL))
        iso_err = max(iso_err, max(abs(fixed[k] - truth[k]) for k in ALL))
    check("raw normalized coords DO distort curls", raw_err > 0.02, f"drift only {raw_err:.4f}")
    check("isotropic() restores true curls", iso_err < 1e-3, f"drift {iso_err:.2e}")
    check("correction is a real improvement", iso_err < raw_err / 10.0,
          f"raw {raw_err:.4f} vs iso {iso_err:.2e}")
    print(f"       (worst curl drift: raw {raw_err:.3f} -> corrected {iso_err:.2e})")


def test_extract() -> None:
    print("extract() bundle")
    prev = synthetic_hand({"index"}, thumb_out=True)
    lm = transform(prev, translate=(0.05, 0.0, 0.0))
    screen = to_screen(lm, center=(0.5, 0.3))
    f = F.extract(lm, screen_lm=screen, prev_shape_lm=prev, dt=0.5)
    check("count matches finger_count", f.count == F.finger_count(lm), f"got {f.count}")
    check("n_extended = 1", f.n_extended == 1, f"got {f.n_extended}")
    check("pinch matches", abs(f.pinch - F.pinch_ratio(lm)) < 1e-6)
    check("motion computed", abs(f.motion - 0.1) < 1e-3, f"got {f.motion:.4f}")
    check("y from screen coords, not shape", abs(f.y - F.hand_y(screen)) < 1e-6)
    check("curls has 4 entries", set(f.curls) == ALL)
    check("frozen dataclass", type(f).__dataclass_params__.frozen)   # type: ignore[attr-defined]
    # Purity: extract must not mutate its inputs.
    before = lm.copy()
    F.extract(lm, screen_lm=screen, prev_shape_lm=prev, dt=0.5)
    check("does not mutate input", np.array_equal(lm, before))


def main() -> int:
    print("=" * 60)
    print("Gesture Instrument - Phase 2 feature tests")
    print("=" * 60)
    for fn in (test_palm_scale, test_curl_extremes, test_rotation_invariance,
               test_scale_invariance, test_thumb_and_pinch, test_motion_energy,
               test_hand_position, test_finger_count, test_thumb_not_load_bearing,
               test_anisotropy_correction, test_extract):
        fn()
    print("-" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} -> {', '.join(_failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
