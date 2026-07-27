#!/usr/bin/env python3
"""Sweep feature thresholds against recorded ground truth and report the best.

The point of the pure-function feature layer: iterate on thresholds a hundred times
without touching a camera. Hand geometry varies more between people than you'd guess,
and hard-coded thresholds are the #1 reason these demos fail on someone else's hands
(plan 7.1) — this is the cheap, offline half of that calibration.

    .venv\\Scripts\\python.exe tools\\tune_thresholds.py
    .venv\\Scripts\\python.exe tools\\tune_thresholds.py data\\counts_left.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features as F                                   # noqa: E402
import replay                                          # noqa: E402


def report_distributions(s: replay.Session) -> None:
    """Where the feature clusters actually sit, per ground-truth class."""
    print("\nfeature distributions by ground-truth count "
          "(curl p50 per finger | thumb_open p50):")
    for label in range(6):
        idx = np.flatnonzero(s.label == label)
        if idx.size == 0:
            continue
        curls = {n: [] for n in F.FINGER_NAMES}
        thumbs = []
        for i in idx:
            lm = s.lm_world[i]
            for n, c in F.finger_curls(lm).items():
                curls[n].append(c)
            thumbs.append(F.thumb_abduction(lm))
        cs = "  ".join(f"{n[:3]}={np.median(v):+.2f}" for n, v in curls.items())
        print(f"  count {label} (n={idx.size:4d}):  {cs}   thumb_abd={np.median(thumbs):+.2f}")


def sweep(s: replay.Session) -> tuple[float, float, float]:
    """Grid-search curl and thumb thresholds. Returns (best_curl, best_thumb, acc)."""
    curls = np.round(np.arange(-0.90, 0.05, 0.05), 2)
    thumbs = np.round(np.arange(-0.60, 1.00, 0.05), 2)   # thumb_abduction is a cosine
    best = (F.CURL_EXTENDED, F.THUMB_ABDUCTED, -1.0)
    grid = np.zeros((len(curls), len(thumbs)), dtype=np.float32)
    for i, c in enumerate(curls):
        for j, t in enumerate(thumbs):
            acc = replay.score_counts(s, float(c), float(t)).accuracy
            grid[i, j] = acc
            if acc > best[2]:
                best = (float(c), float(t), acc)

    print("\ncurl threshold sweep (best accuracy over all thumb thresholds):")
    for i, c in enumerate(curls):
        row_best = grid[i].max()
        bar = "#" * int(round(row_best * 40))
        marker = "  <- current" if abs(c - F.CURL_EXTENDED) < 1e-6 else ""
        print(f"  curl {c:+.2f}: {100 * row_best:5.1f}%  {bar}{marker}")
    return best


def main() -> int:
    paths = ([Path(sys.argv[1])] if len(sys.argv) > 1
             else sorted((ROOT / "data").glob("counts_*.npz")))
    if not paths or not paths[0].is_file():
        print("No recording found. Record one first:")
        print("  .venv\\Scripts\\python.exe tools\\record_session.py --hand Left")
        return 1

    for path in paths:
        s = replay.load(path)
        print("=" * 64)
        print(f"{path.name}  hand={s.hand}  frames={len(s)}")
        print("=" * 64)

        base = replay.score_counts(s)
        print(f"current thresholds: curl={F.CURL_EXTENDED:+.2f} thumb={F.THUMB_ABDUCTED:+.2f}"
              f"  ->  {100 * base.accuracy:.1f}% on {base.n_static} static frames")
        print(replay.format_confusion(base.confusion))

        report_distributions(s)
        best_curl, best_thumb, best_acc = sweep(s)

        print(f"\nBEST: curl={best_curl:+.2f} thumb={best_thumb:.2f} "
              f"-> {100 * best_acc:.1f}%  (current {100 * base.accuracy:.1f}%)")
        if best_acc > base.accuracy + 0.005:
            print("\nTo adopt, edit src/features.py:")
            print(f"  CURL_EXTENDED = {best_curl:.2f}")
            print(f"  THUMB_ABDUCTED = {best_thumb:+.2f}")
            tuned = replay.score_counts(s, best_curl, best_thumb)
            print(replay.format_confusion(tuned.confusion))
        else:
            print("Current thresholds are already at/near the optimum for this clip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
