#!/usr/bin/env python3
"""Phase 1 vision spine: tracked hands on screen, fully instrumented. No audio yet.

    .venv\\Scripts\\python.exe src\\main.py                  # live window
    .venv\\Scripts\\python.exe src\\main.py --headless -s 20 # benchmark, prints verdict

Mirror convention (plan 1.3): the frame is flipped **once**, before inference, so
landmark coordinates already line up with what you see. MediaPipe then labels hands in
mirrored space, which ``landmarks.true_handedness`` corrects at the boundary.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hud
from capture import Camera
from landmarks import Landmarker
from metrics import Pipeline

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "hand_landmarker.task"

# Exit criteria from the plan (Phase 1).
TARGET_FPS = 30.0
TARGET_P95_MS = 40.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture Instrument - Phase 1 vision spine")
    p.add_argument("-c", "--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30, help="requested; this cam caps at 30")
    p.add_argument("--headless", action="store_true", help="no window; benchmark only")
    p.add_argument("-s", "--seconds", type=float, default=0.0, help="auto-exit after N s")
    p.add_argument("--no-gpu", action="store_true", help="skip the GPU delegate attempt")
    p.add_argument("--hands", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not MODEL.is_file():
        print(f"Missing model: {MODEL}\nSee README.md > Setup > Assets.", file=sys.stderr)
        return 1

    metrics = Pipeline()
    last_seq_submitted = -1
    last_result_ts_ms = -1
    frames_drawn = 0
    both_hands_frames = 0
    result_frames = 0
    first_result_t = 0.0       # steady state begins here; startup is not "sustained"
    last_result_t = 0.0
    t_start = time.perf_counter()
    last_log = t_start

    with Camera(args.camera, args.width, args.height, args.fps) as cam, \
            Landmarker(str(MODEL), num_hands=args.hands, try_gpu=not args.no_gpu) as lmk:
        first = cam.wait_for_first_frame()
        print(f"camera: {cam.actual_size[0]}x{cam.actual_size[1]} "
              f"codec={cam.actual_fourcc} | delegate={lmk.delegate} | "
              f"model={MODEL.name}")
        print("keys: q / ESC to quit" if not args.headless else "headless benchmark...")
        del first

        base = None          # pristine mirrored frame; never drawn on
        while True:
            frame = cam.read()
            res = lmk.latest()
            new_frame = frame is not None and frame.seq != last_seq_submitted
            new_result = res is not None and res.ts_ms != last_result_ts_ms

            # Only work when something actually changed. Without this the loop spins at
            # thousands of fps, re-drawing identical frames and starving the inference
            # thread of CPU — which *raises* the latency we're trying to measure.
            if not new_frame and not new_result:
                time.sleep(0.001)
                continue

            # --- submit only genuinely new frames -------------------------------
            if new_frame:
                last_seq_submitted = frame.seq
                metrics.capture_fps.tick(frame.ts)
                base = cv2.flip(frame.image, 1)          # mirror once, before inference
                lmk.submit(base, frame.ts, frame.seq)
            if base is None:
                continue
            view = base.copy()                           # draw on a scratch copy

            # --- consume the newest result --------------------------------------
            if new_result:
                last_result_ts_ms = res.ts_ms
                result_frames += 1
                metrics.infer_fps.tick(res.done_ts)
                last_result_t = res.done_ts
                if not first_result_t:
                    first_result_t = res.done_ts
                if len(res.hands) >= 2:
                    both_hands_frames += 1
            if res is not None:
                for label, lm in res.hands.items():
                    hud.draw_hand(view, lm, label, res.scores.get(label, 0.0))

            # --- HUD -------------------------------------------------------------
            now = time.perf_counter()
            c2s50, c2s95 = metrics.capture_to_submit.percentiles()
            s2c50, s2c95 = metrics.submit_to_callback.percentiles()
            tot50, tot95 = metrics.total.percentiles()
            hands_txt = ", ".join(sorted(res.hands)) if res and res.hands else "none"
            hud.draw_stats(view, [
                f"cap {metrics.capture_fps.fps:4.1f}  infer {metrics.infer_fps.fps:4.1f}"
                f"  draw {metrics.render_fps.fps:4.1f} fps",
                f"cap->sub  p50 {c2s50:5.1f}  p95 {c2s95:5.1f} ms",
                f"sub->cb   p50 {s2c50:5.1f}  p95 {s2c95:5.1f} ms",
                f"TOTAL     p50 {tot50:5.1f}  p95 {tot95:5.1f} ms",
                f"hands: {hands_txt}  ({lmk.delegate})",
            ])
            hud.draw_hint(view, "Phase 1 vision spine - no audio yet")

            # Record a stage breakdown once per *new* result. Recording every iteration
            # would measure how old the latest result is, not how long rendering took.
            if new_result:
                metrics.record_result(res.capture_ts, res.submit_ts, res.done_ts,
                                      time.perf_counter())
            metrics.render_fps.tick(now)
            frames_drawn += 1

            if not args.headless:
                cv2.imshow("Gesture Instrument - Phase 1", view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            if args.headless and now - last_log >= 2.0:
                last_log = now
                print("  " + metrics.summary())
            if args.seconds and now - t_start >= args.seconds:
                break

    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    _, tot95 = metrics.total.percentiles()
    # "Sustained" = steady state. Model load + XNNPACK init + camera warmup take a few
    # seconds and would otherwise drag the average below the real running rate.
    span = last_result_t - first_result_t
    infer_fps = (result_frames - 1) / span if span > 0 else 0.0
    warmup = first_result_t - t_start if first_result_t else 0.0
    print("\n" + "=" * 62)
    print(f"ran {elapsed:.1f}s (warmup {warmup:.1f}s, steady {span:.1f}s) | "
          f"drew {frames_drawn} frames | {result_frames} results")
    print(metrics.summary())
    print(f"dropped out-of-order results: {lmk.dropped}")
    print(f"frames with BOTH hands: {both_hands_frames} "
          f"({100.0 * both_hands_frames / max(result_frames, 1):.0f}% of results)")
    print("-" * 62)
    ok_fps = infer_fps >= TARGET_FPS - 1.0        # camera hard-caps at ~30.3
    ok_lat = 0.0 < tot95 < TARGET_P95_MS
    print(f"[{'PASS' if ok_fps else 'FAIL'}] sustained inference >= {TARGET_FPS:.0f} fps"
          f"  (got {infer_fps:.1f})")
    print(f"[{'PASS' if ok_lat else 'FAIL'}] pipeline p95 < {TARGET_P95_MS:.0f} ms"
          f"  (got {tot95:.1f})")
    print("[ MANUAL ] handedness correct when you wave each hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
