#!/usr/bin/env python3
"""Enumerate cameras and report what each backend actually negotiates.

Cameras lie: you ask for MJPG 640x480@60 and get YUYV 640x480@30. This prints what
you *really* got, plus a measured grab rate, so Phase 1 tuning is based on facts.

Run:  .venv\\Scripts\\python.exe tools\\camera_probe.py
"""
from __future__ import annotations

import time

import cv2

WIDTH, HEIGHT, FPS = 640, 480, 60
BACKENDS = [("CAP_DSHOW", cv2.CAP_DSHOW), ("CAP_MSMF", cv2.CAP_MSMF)]


def fourcc_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)) if n else "(none)"


def probe(index: int, name: str, api: int) -> bool:
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  cam{index} {name:10s} opened but read() failed")
        cap.release()
        return False

    # measure real sustained grab rate over ~2s
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        if cap.read()[0]:
            n += 1
    measured = n / (time.perf_counter() - t0)

    print(f"  cam{index} {name:10s} OK  "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"fourcc={fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))} "
          f"reported_fps={cap.get(cv2.CAP_PROP_FPS):.0f} "
          f"measured_fps={measured:.1f} shape={frame.shape}")
    cap.release()
    return True


def main() -> int:
    print("Probing camera indices 0..2 (MJPG 640x480@60 requested)")
    found = 0
    for index in range(3):
        for name, api in BACKENDS:
            if probe(index, name, api):
                found += 1
    print(f"\n{found} working (index, backend) combination(s).")
    if not found:
        print("No camera found. Check that no other app holds it, and that "
              "Settings > Privacy & security > Camera allows desktop apps.")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
