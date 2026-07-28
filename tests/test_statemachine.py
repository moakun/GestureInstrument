#!/usr/bin/env python3
"""Phase 3 tests: hysteresis, motion-gated latching, trigger edges, and the exit scenario.

The headline test is `test_exit_criteria`: sweep 0->5->0 ten times while pinching at known
moments, and assert **exactly 10 note-ons and zero phantoms** (plan 3.x).

Run:  .venv\\Scripts\\python.exe tests\\test_statemachine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import gesture_stream as G                                          # noqa: E402
import statemachine as S                                            # noqa: E402

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def test_schmitt() -> None:
    print("Schmitt hysteresis")
    s = S.Schmitt(lo=-0.68, hi=-0.25)                # falling: True below lo
    check("starts False", s.state is False)
    check("above lo stays False", s(-0.50) is False)
    check("below lo -> True", s(-0.80) is True)
    check("inside band holds True", s(-0.50) is True, "single threshold would flip here")
    check("above hi -> False", s(-0.10) is False)
    check("inside band holds False", s(-0.50) is False)

    r = S.Schmitt(lo=0.35, hi=0.55, rising=True)     # rising: True above hi
    check("rising: below hi stays False", r(0.50) is False)
    check("rising: above hi -> True", r(0.60) is True)
    check("rising: inside band holds True", r(0.40) is True)
    check("rising: below lo -> False", r(0.30) is False)

    # Chattering right on one threshold must not produce a single flip.
    c = S.Schmitt(lo=0.25, hi=0.35)
    c(0.20)
    flips = sum(1 for x in [0.26, 0.24, 0.26, 0.24, 0.26] if c(x) != c.state or False)
    check("no oscillation inside the band", c.state is True, f"flips={flips}")
    check("lo > hi is rejected", _raises(lambda: S.Schmitt(0.5, 0.1)))

    q = S.Schmitt(lo=0.25, hi=0.35)
    q.resync(0.10)
    check("resync adopts state without an edge", q.state is True)


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def test_selection_latch() -> None:
    print("SelectionLatch (ms-based debounce + motion gate)")
    lat = S.SelectionLatch(confirm_ms=100.0, max_motion=1.2)
    check("no commit on first sighting", lat.update(3, 0.1, 0.000) is None)
    check("no commit before window", lat.update(3, 0.1, 0.033) is None)
    check("no commit at 66ms", lat.update(3, 0.1, 0.066) is None)
    check("commits at 100ms", lat.update(3, 0.1, 0.100) == 3)
    check("does not re-commit the same value", lat.update(3, 0.1, 0.200) is None)

    # Motion gate: a value held steadily but *while moving* must never commit.
    lat2 = S.SelectionLatch(confirm_ms=100.0, max_motion=1.2)
    for i in range(20):
        lat2.update(4, 3.0, i * 0.033)
    check("moving hand never commits", lat2.stable is None)
    for i in range(20, 30):
        lat2.update(4, 0.1, i * 0.033)
    check("commits once motion settles", lat2.stable == 4)

    # A flickering observation restarts the window rather than accumulating.
    lat3 = S.SelectionLatch(confirm_ms=100.0)
    for i, v in enumerate([2, 3, 2, 3, 2, 3]):
        lat3.update(v, 0.1, i * 0.033)
    check("alternating values never commit", lat3.stable is None)

    # Framerate independence: the same wall-clock window at 15 fps and 60 fps.
    for fps in (15.0, 60.0):
        lat4 = S.SelectionLatch(confirm_ms=100.0)
        committed_at = None
        for i in range(40):
            t = i / fps
            if lat4.update(1, 0.1, t) is not None:
                committed_at = t
                break
        ok = committed_at is not None and abs(committed_at - 0.100) < (1.5 / fps)
        check(f"commits at ~100ms at {fps:.0f} fps", ok, f"got {committed_at}")

    lat5 = S.SelectionLatch(confirm_ms=100.0)
    lat5.update(2, 0.1, 0.0); lat5.update(2, 0.1, 0.2)
    lat5.reset()
    check("reset clears the latch", lat5.stable is None)


def test_pinch_trigger() -> None:
    print("PinchTrigger (single-frame, refractory)")
    tr = S.PinchTrigger(close=0.25, open_=0.35, refractory_ms=80.0)
    check("open -> nothing", tr.update(0.50, 0.000) is None)
    check("crossing close fires immediately", tr.update(0.20, 0.033) == "on",
          "a pinch must cost no confirmation latency")
    check("staying pinched does not re-fire", tr.update(0.18, 0.066) is None)
    check("release fires off", tr.update(0.40, 0.100) == "off")
    check("re-pinch after refractory fires", tr.update(0.20, 0.200) == "on")

    # Jitter across the close threshold inside the refractory window must not double-fire.
    tr2 = S.PinchTrigger(close=0.25, open_=0.35, refractory_ms=80.0)
    tr2.update(0.50, 0.0)
    fires = sum(1 for i, v in enumerate([0.20, 0.40, 0.20, 0.40, 0.20])
                if tr2.update(v, 0.010 + i * 0.010) == "on")
    check("refractory suppresses jitter double-fires", fires == 1, f"got {fires}")

    # Reacquisition mid-pinch must not manufacture a note (plan 7.2).
    tr3 = S.PinchTrigger()
    tr3.resync(0.10)
    check("resync adopts pinched state", tr3.pinched is True)
    check("no note-on while still pinched", tr3.update(0.10, 1.0) is None)
    check("no spurious off on release", tr3.update(0.50, 1.1) is None)
    check("next real pinch fires normally", tr3.update(0.20, 1.2) == "on")


def test_quantizer() -> None:
    print("HysteresisQuantizer (Mode C pitch, deadband)")
    q = S.HysteresisQuantizer(n=5, deadband=0.06)
    check("primes to the raw cell", q(0.10) == 0)
    check("moves on a clear crossing", q(0.50) == 2)
    check("clamps at the top", q(1.00) == 4)
    check("clamps at the bottom", q(-1.0) == 0)

    # Hovering exactly on a boundary must not warble.
    q2 = S.HysteresisQuantizer(n=5, deadband=0.06)
    q2(0.19)
    seq = [q2(0.20 + d) for d in (0.000, -0.002, 0.001, -0.001, 0.002)]
    check("hovering on a boundary is stable", len(set(seq)) == 1, f"got {seq}")
    check("a decisive move past the deadband still commits", q2(0.25) == 1)


def test_hand_tracker_basics() -> None:
    print("HandTracker")
    tr = S.HandTracker("Right")
    s = G.Stream().hold(G.pose(0), 0.5).move_to(G.pose(2), 0.15).hold(G.pose(2), 0.5)
    events = G.run(s, tr)
    kinds = [e.kind for e in events]
    check("reports hand_found once", kinds.count("hand_found") == 1)
    selects = [e.value for e in events if e.kind == "select"]
    check("settles on the held count", tr.count == 2, f"count={tr.count}, selects={selects}")
    check("no trigger events without a pinch",
          not any(k.startswith("trigger") for k in kinds), f"{kinds}")

    # Dropout: the selection is held briefly, then released (plan 7.2).
    lost = tr.absent(10.0)
    check("absent reports hand_lost", [e.kind for e in lost] == ["hand_lost"])
    check("selection survives a brief dropout", tr.tick_absent(10.1) == [])
    check("held selection still readable", tr.count == 2)
    dropped = tr.tick_absent(10.4)
    check("selection cleared after the hold window",
          [e.kind for e in dropped] == ["select"] and tr.count is None)


def test_no_autofire_on_reacquisition() -> None:
    print("no auto-fire when a hand reappears mid-pinch")
    tr = S.HandTracker("Right")
    s1 = G.Stream().hold(G.pose(1), 0.4)
    G.run(s1, tr)
    tr.absent(5.0)
    # Comes back already pinching — must stay silent until an actual new pinch.
    s2 = G.Stream(seed=3).hold(G.pose(1, pinch=True), 0.4)
    events = G.run(s2, tr)
    check("no note-on on reacquisition",
          not any(e.kind == "trigger_on" for e in events),
          f"{[e.kind for e in events]}")
    # Release, then pinch again: that one must fire.
    s3 = G.Stream(seed=4).hold(G.pose(1), 0.3).move_to(G.pose(1, pinch=True), 0.05) \
        .hold(G.pose(1, pinch=True), 0.2)
    events3 = G.run(s3, tr)
    check("a genuine later pinch still fires",
          sum(1 for e in events3 if e.kind == "trigger_on") == 1,
          f"{[e.kind for e in events3]}")


def test_motion_gate_blocks_transitions() -> None:
    print("motion gate suppresses transitional poses")
    # Sweep 0 -> 5 quickly, with only a brief settle at the end.
    tr = S.HandTracker("Left")
    s = G.Stream().hold(G.pose(0), 0.4)
    for target in (1, 2, 3, 4, 5):
        s.move_to(G.pose(target), 0.10)
    s.hold(G.pose(5), 0.5)
    selects = [e.value for e in G.run(s, tr) if e.kind == "select"]
    check("ends on the intended count", tr.count == 5, f"selects={selects}")
    check("does not commit every intermediate count", len(selects) <= 3,
          f"committed {selects} - the gate should suppress most transitions")

    # With the gate disabled, the same sweep should commit far more - proving the gate,
    # not the geometry, is what suppresses them.
    tr2 = S.HandTracker("Left", max_motion=1e9, confirm_ms=0.0)
    ungated = [e.value for e in G.run(s, tr2) if e.kind == "select"]
    check("gate is what removes them", len(ungated) > len(selects),
          f"gated={selects} ungated={ungated}")


def test_fist_does_not_trigger() -> None:
    print("a closed fist must not fire a note")
    import features as F
    fist = G.pose(0)
    ratio = F.pinch_ratio(fist)
    print(f"       synthetic fist pinch_ratio={ratio:.3f} (threshold {F.PINCH_CLOSE}); "
          f"a real recorded fist measures 0.263-0.302")
    check("the fist really does look like a pinch by ratio alone",
          ratio < F.PINCH_OPEN, f"got {ratio:.3f} - test no longer exercises the gate")

    tr = S.HandTracker("Right")
    s = G.Stream().hold(G.pose(1), 0.4).move_to(fist, 0.15).hold(fist, 0.6)
    events = G.run(s, tr)
    check("no note-on from making a fist",
          not any(e.kind == "trigger_on" for e in events),
          f"{[e.kind for e in events]}")

    # ...but a real pinch from an open hand still fires.
    s2 = G.Stream(seed=11).hold(G.pose(1), 0.3).move_to(G.pose(1, pinch=True), 0.05) \
        .hold(G.pose(1, pinch=True), 0.2)
    tr2 = S.HandTracker("Right")
    check("a genuine pinch still fires",
          sum(1 for e in G.run(s2, tr2) if e.kind == "trigger_on") == 1)


def test_exit_criteria() -> None:
    print("EXIT CRITERIA: 10 sweeps 0->5->0 with 10 deliberate pinches")
    n = 10
    tr = S.HandTracker("Right")
    stream = G.sweep_with_pinches(n_pinches=n)
    events = G.run(stream, tr)
    ons = [e for e in events if e.kind == "trigger_on"]
    offs = [e for e in events if e.kind == "trigger_off"]
    print(f"       stream: {len(stream)} frames, {stream.duration:.1f}s, "
          f"{len(events)} events")
    check(f"exactly {n} note-ons, zero phantoms", len(ons) == n, f"got {len(ons)}")
    check(f"exactly {n} note-offs", len(offs) == n, f"got {len(offs)}")

    # Each note-on must land just after its marked pinch moment.
    marks = [t for name, t in stream.marks if name.startswith("pinch")]
    if len(ons) == len(marks):
        lags = [on.t - m for on, m in zip(ons, marks)]
        check("every note-on follows its intended pinch",
              all(0.0 <= lag < 0.20 for lag in lags),
              f"lags={[round(x, 3) for x in lags]}")
        print(f"       trigger lag: min {min(lags) * 1e3:.0f} ms, "
              f"max {max(lags) * 1e3:.0f} ms")
    check("on/off strictly alternate",
          [e.kind for e in events if e.kind.startswith("trigger")] ==
          ["trigger_on", "trigger_off"] * n)


def main() -> int:
    print("=" * 62)
    print("Gesture Instrument - Phase 3 state machine tests")
    print("=" * 62)
    for fn in (test_schmitt, test_selection_latch, test_pinch_trigger, test_quantizer,
               test_hand_tracker_basics, test_no_autofire_on_reacquisition,
               test_motion_gate_blocks_transitions, test_fist_does_not_trigger,
               test_exit_criteria):
        fn()
    print("-" * 62)
    if _failures:
        print(f"FAILED: {len(_failures)} -> {', '.join(_failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
