#!/usr/bin/env python3
"""Record labelled landmark clips to .npz for offline threshold tuning.

This is the payoff of keeping `features.py` pure: record once, then iterate on thresholds
a hundred times without ever touching a camera (plan 1.x / 2.x).

Guided mode walks you through holding each finger count 0-5, so every frame carries
ground-truth. That is what `tests/test_replay.py` scores against for the Phase 2 exit
criteria.

    .venv\\Scripts\\python.exe tools\\record_session.py                  # guided, Left hand
    .venv\\Scripts\\python.exe tools\\record_session.py --hand Right
    .venv\\Scripts\\python.exe tools\\record_session.py --free -o data/freeplay.npz

Schema (npz):
    ts        (N,)     float64  capture perf_counter seconds
    lm_norm   (N,21,3) float32  normalized image landmarks (screen position)
    lm_world  (N,21,3) float32  metric world landmarks (shape features)
    label     (N,)     int8     ground-truth finger count, -1 if unlabelled
    seg       (N,)     int16    recording segment; never compute motion across segments
    hand      ()       str      which hand was recorded
    aspect    ()       float64  frame width/height, for features.isotropic()
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hud                                              # noqa: E402
from capture import Camera                              # noqa: E402
from landmarks import Landmarker                        # noqa: E402

MODEL = ROOT / "assets" / "hand_landmarker.task"
_FONT = cv2.FONT_HERSHEY_SIMPLEX

PROMPTS = {
    0: "FIST - no fingers",
    1: "1 finger  (index)",
    2: "2 fingers (index+middle)",
    3: "3 fingers (index+middle+ring)",
    4: "4 fingers (thumb TUCKED)",
    5: "5 fingers (thumb OUT)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record labelled landmark clips")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--hand", choices=["Left", "Right"], default="Left",
                   help="which hand to record (Mode C selects instrument with Left)")
    p.add_argument("-c", "--camera", type=int, default=0)
    p.add_argument("--hold", type=float, default=2.5, help="seconds recorded per count")
    p.add_argument("--countdown", type=float, default=3.0, help="seconds to get ready")
    p.add_argument("--free", action="store_true",
                   help="free-form capture (label -1) instead of the guided 0-5 flow")
    return p.parse_args()


def banner(img: np.ndarray, title: str, sub: str, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 74), (w, h), (0, 0, 0), -1)
    cv2.putText(img, title, (12, h - 40), _FONT, 0.95, color, 2, cv2.LINE_AA)
    cv2.putText(img, sub, (12, h - 12), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    if not MODEL.is_file():
        print(f"Missing model: {MODEL}", file=sys.stderr)
        return 1
    out = args.out or (ROOT / "data" /
                       f"{'free' if args.free else 'counts'}_{args.hand.lower()}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    segments = [(-1, "FREE PLAY - move naturally")] if args.free else list(PROMPTS.items())
    ts_l, norm_l, world_l, label_l, seg_l = [], [], [], [], []
    aborted = False

    with Camera(args.camera) as cam, Landmarker(str(MODEL), num_hands=2) as lmk:
        cam.wait_for_first_frame()
        w, h = cam.actual_size
        aspect = w / float(h)
        print(f"recording {args.hand} hand -> {out}")
        print("keys: q/ESC abort, SPACE skip segment")

        last_seq = -1
        for seg_i, (label, prompt) in enumerate(segments):
            phase_end = time.perf_counter() + args.countdown
            recording = False
            captured = 0
            while True:
                frame = cam.read()
                if frame is None:
                    time.sleep(0.002)
                    continue
                if frame.seq != last_seq:
                    last_seq = frame.seq
                    view = cv2.flip(frame.image, 1)
                    lmk.submit(view, frame.ts, frame.seq)
                else:
                    time.sleep(0.001)
                    continue

                res = lmk.latest()
                present = res is not None and args.hand in res.hands
                if res is not None:
                    for lbl, lm in res.hands.items():
                        hud.draw_hand(view, lm, lbl, res.scores.get(lbl, 0.0))

                now = time.perf_counter()
                remain = phase_end - now
                if not recording:
                    banner(view, f"GET READY: {prompt}",
                           f"starts in {max(remain, 0.0):3.1f}s   "
                           f"({args.hand} hand {'DETECTED' if present else 'NOT VISIBLE'})",
                           (60, 200, 255))
                    if remain <= 0.0:
                        recording, phase_end = True, now + args.hold
                else:
                    banner(view, f"REC  {prompt}",
                           f"{max(remain, 0.0):3.1f}s left   captured {captured} frames",
                           (60, 60, 255))
                    cv2.circle(view, (view.shape[1] - 28, 28), 11, (60, 60, 255), -1)
                    if present and res.ts_ms is not None and args.hand in res.world:
                        ts_l.append(res.capture_ts)
                        norm_l.append(res.hands[args.hand])
                        world_l.append(res.world[args.hand])
                        label_l.append(label)
                        seg_l.append(seg_i)
                        captured += 1
                    if remain <= 0.0:
                        print(f"  segment {seg_i} (label {label}): {captured} frames")
                        break

                cv2.imshow("record_session", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    aborted = True
                    break
                if key == ord(" "):
                    print(f"  segment {seg_i} skipped")
                    break
            if aborted:
                break

    cv2.destroyAllWindows()
    if aborted and not ts_l:
        print("aborted, nothing recorded")
        return 1
    if not ts_l:
        print("no frames captured - was the hand visible?", file=sys.stderr)
        return 1

    np.savez_compressed(
        out,
        ts=np.asarray(ts_l, dtype=np.float64),
        lm_norm=np.asarray(norm_l, dtype=np.float32),
        lm_world=np.asarray(world_l, dtype=np.float32),
        label=np.asarray(label_l, dtype=np.int8),
        seg=np.asarray(seg_l, dtype=np.int16),
        hand=np.array(args.hand),
        aspect=np.array(aspect, dtype=np.float64),
    )
    n = len(ts_l)
    print(f"\nsaved {n} frames -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    if not args.free:
        for label in sorted(set(label_l)):
            print(f"  label {label}: {label_l.count(label):4d} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
