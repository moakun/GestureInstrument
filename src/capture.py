"""Threaded, latest-frame-wins camera capture.

Decouples grab latency from processing: a dedicated thread drains the driver as fast as
it will go, and the consumer always gets the *newest* frame, never a queued stale one.
Without this, OpenCV buffers frames and you end up reacting to a hand position from
150 ms ago (see plan 1.1).

Measured on this machine (`tools/camera_format_probe.py`): the webcam is hard-capped at
~30 fps and only offers **YUY2** — MJPG is not available at any resolution or backend, so
the plan's MJPG tip is a no-op here. We still request it: it is free, and it matters on
cameras that do offer it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2

# Windows: DSHOW generally has lower latency than MSMF and reports FOURCC honestly.
# Both measured ~30.3 fps here. On Linux this falls through to CAP_ANY (V4L2).
DEFAULT_API = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY


@dataclass(frozen=True)
class Frame:
    """A captured frame plus the instant it was grabbed.

    ``seq`` increments per successful grab so consumers can tell a genuinely new frame
    from a re-read of the same one (re-submitting a duplicate to the landmarker burns
    inference for nothing).
    """

    image: "cv2.typing.MatLike"
    ts: float          # time.perf_counter() at grab
    seq: int


class Camera:
    """Latest-frame-wins capture on a daemon thread."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, api: int | None = None, fourcc: str | None = "MJPG") -> None:
        self.cap = cv2.VideoCapture(index, DEFAULT_API if api is None else api)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. Is another app using it? "
                f"Check Settings > Privacy & security > Camera."
            )
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # do not accumulate stale frames

        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._run = True
        self._seq = 0
        self._fail_streak = 0
        self._thread = threading.Thread(target=self._loop, name="capture", daemon=True)
        self._thread.start()

    # -- properties reflecting what the driver actually negotiated --------------
    @property
    def actual_size(self) -> tuple[int, int]:
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def actual_fourcc(self) -> str:
        n = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)) if n else "(none)"

    def _loop(self) -> None:
        while self._run:
            ok, image = self.cap.read()
            if not ok or image is None:
                self._fail_streak += 1
                time.sleep(0.005)
                continue
            self._fail_streak = 0
            with self._lock:
                self._seq += 1
                self._frame = Frame(image, time.perf_counter(), self._seq)

    def read(self) -> Frame | None:
        """Return the most recent frame, or None if nothing has arrived yet."""
        with self._lock:
            return self._frame

    def wait_for_first_frame(self, timeout: float = 5.0) -> Frame:
        """Block until the first frame lands, so callers don't spin on None."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.01)
        raise RuntimeError(f"No frame from camera within {timeout:.1f}s")

    def close(self) -> None:
        self._run = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
