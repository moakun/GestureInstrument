#!/usr/bin/env python3
"""Phase 2 exit criteria: replay a recorded clip, score finger counts vs ground truth.

    "a unit test replays a recorded .npz and asserts the extracted finger-count sequence
     matches hand-labelled ground truth >=95% on *static* frames"

Skips (does not fail) when no recording exists, so a fresh clone can still run the suite
— recordings are user data and are git-ignored.

Run:  .venv\\Scripts\\python.exe tests\\test_replay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import replay                                          # noqa: E402

TARGET_ACCURACY = 0.95
DATA_DIR = ROOT / "data"


def find_sessions() -> list[Path]:
    return sorted(DATA_DIR.glob("counts_*.npz")) if DATA_DIR.is_dir() else []


def main() -> int:
    print("=" * 64)
    print("Gesture Instrument - Phase 2 replay test")
    print("=" * 64)
    paths = find_sessions()
    if not paths:
        print("SKIP: no labelled recording found.")
        print(f"      expected: {DATA_DIR / 'counts_<hand>.npz'}")
        print("      record one with:")
        print("        .venv\\Scripts\\python.exe tools\\record_session.py --hand Left")
        return 0

    failures = 0
    for path in paths:
        s = replay.load(path)
        motion = replay.motion_series(s)
        score = replay.score_counts(s)
        static_pct = 100.0 * score.n_static / max(len(s), 1)
        print(f"\n{path.name}  hand={s.hand}  frames={len(s)}  aspect={s.aspect:.3f}")
        print(f"  static frames (motion <= {replay.STATIC_MAX_MOTION}): "
              f"{score.n_static} ({static_pct:.0f}%)  "
              f"motion p50={np.percentile(motion, 50):.2f} p95={np.percentile(motion, 95):.2f}")
        print(replay.format_confusion(score.confusion))
        ok = score.accuracy >= TARGET_ACCURACY
        print(f"  [{'PASS' if ok else 'FAIL'}] accuracy {100 * score.accuracy:.1f}% "
              f">= {100 * TARGET_ACCURACY:.0f}%")
        if not ok:
            failures += 1
            worst = sorted(score.per_class.items(),
                           key=lambda kv: (kv[1] if kv[1] == kv[1] else 2.0))[:3]
            print("       worst classes: " + ", ".join(
                f"{c}={100 * a:.0f}%" for c, a in worst if a == a))
            print("       try: .venv\\Scripts\\python.exe tools\\tune_thresholds.py")

    print("-" * 64)
    if failures:
        print(f"FAILED: {failures} session(s) below {100 * TARGET_ACCURACY:.0f}%")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
