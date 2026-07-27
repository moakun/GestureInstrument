#!/usr/bin/env python3
"""Find whether this camera will actually give us MJPG and/or 60 fps.

The first probe showed DSHOW negotiating YUY2@30 despite an MJPG@60 request. Property
order matters in OpenCV (setting FOURCC after the frame size can be ignored, and vice
versa), so this tries several orderings/resolutions and reports what really sticks.

Run:  .venv\\Scripts\\python.exe tools\\camera_format_probe.py
"""
from __future__ import annotations

import time

import cv2


def fourcc_str(v: float) -> str:
    n = int(v)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)) if n else "(none)"


def measure(cap: cv2.VideoCapture, seconds: float = 1.5) -> float:
    cap.read()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if cap.read()[0]:
            n += 1
    return n / (time.perf_counter() - t0)


def trial(label: str, api: int, w: int, h: int, fps: int,
          fourcc: str | None, fourcc_first: bool) -> None:
    cap = cv2.VideoCapture(0, api)
    if not cap.isOpened():
        print(f"  {label:44s} could not open")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def set_codec() -> None:
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    def set_size() -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)

    if fourcc_first:
        set_codec(); set_size()
    else:
        set_size(); set_codec()

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"  {label:44s} read() failed")
        cap.release()
        return
    got = measure(cap)
    print(f"  {label:44s} -> {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
          f"codec={fourcc_str(cap.get(cv2.CAP_PROP_FOURCC)):6s} "
          f"reported={cap.get(cv2.CAP_PROP_FPS):>3.0f} measured={got:5.1f}")
    cap.release()


def main() -> int:
    print("Camera 0 format/rate trials (measured fps is what matters)\n")
    print("DSHOW:")
    trial("MJPG 640x480@60, codec first", cv2.CAP_DSHOW, 640, 480, 60, "MJPG", True)
    trial("MJPG 640x480@60, size first", cv2.CAP_DSHOW, 640, 480, 60, "MJPG", False)
    trial("MJPG 320x240@60, codec first", cv2.CAP_DSHOW, 320, 240, 60, "MJPG", True)
    trial("YUY2 640x480@30 (baseline)", cv2.CAP_DSHOW, 640, 480, 30, "YUY2", True)
    trial("no codec req 640x480@60", cv2.CAP_DSHOW, 640, 480, 60, None, True)
    print("MSMF:")
    trial("MJPG 640x480@60, codec first", cv2.CAP_MSMF, 640, 480, 60, "MJPG", True)
    trial("no codec req 640x480@60", cv2.CAP_MSMF, 640, 480, 60, None, True)
    trial("no codec req 320x240@60", cv2.CAP_MSMF, 320, 240, 60, None, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
