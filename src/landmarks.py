"""MediaPipe HandLandmarker wrapper: LIVE_STREAM mode, handedness fix, safe handoff.

Three gotchas from the plan (1.2/1.3) are handled here so nothing downstream repeats them:

1. ``detect_async`` timestamps must be **strictly increasing integer ms**, derived from a
   monotonic clock. Two frames inside the same millisecond would otherwise raise, so the
   counter below forces monotonicity.
2. The result callback runs on **MediaPipe's own thread**. It must not draw or block, so
   it only converts to numpy and drops the result into a 1-slot mailbox.
3. MediaPipe's ``handedness`` refers to the *image*. We mirror the frame for a usable
   view, which swaps left/right — corrected once, here, at the boundary.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision

# We flip the frame horizontally for a mirror view; unmirrored video is unusable for
# gestures. Set False only if you also stop flipping in main.
MIRROR = True

_SWAP = {"Left": "Right", "Right": "Left"}


def true_handedness(label: str) -> str:
    """Map MediaPipe's image-space handedness to the user's actual hand."""
    return _SWAP.get(label, label) if MIRROR else label


@dataclass
class HandsResult:
    """One inference result, already converted to plain numpy for Phase 2."""

    hands: dict[str, np.ndarray] = field(default_factory=dict)  # "Left"/"Right" -> (21,3) f32
    scores: dict[str, float] = field(default_factory=dict)
    ts_ms: int = 0
    capture_ts: float = 0.0     # perf_counter when the frame was grabbed
    submit_ts: float = 0.0      # perf_counter when handed to MediaPipe
    done_ts: float = 0.0        # perf_counter when the callback fired
    seq: int = 0


class Landmarker:
    """Async hand landmarker with a 1-slot result mailbox (latest wins)."""

    def __init__(self, model_path: str, num_hands: int = 2,
                 det_conf: float = 0.6, pres_conf: float = 0.6, track_conf: float = 0.6,
                 try_gpu: bool = True) -> None:
        self._lock = threading.Lock()
        self._latest: HandsResult | None = None
        self._pending: dict[int, tuple[float, float, int]] = {}  # ts_ms -> (capture, submit, seq)
        self._last_ts_ms = -1
        self.delegate = "CPU"
        self.dropped = 0          # results that arrived after a newer one (rare)

        opts = dict(num_hands=num_hands, min_hand_detection_confidence=det_conf,
                    min_hand_presence_confidence=pres_conf,
                    min_tracking_confidence=track_conf)
        if try_gpu:
            try:
                self.lm = self._make(model_path, mpy.BaseOptions.Delegate.GPU, opts)
                self.delegate = "GPU"
                return
            except Exception:
                pass   # GPU delegate is unavailable on most desktop builds; CPU is fine
        self.lm = self._make(model_path, mpy.BaseOptions.Delegate.CPU, opts)

    def _make(self, model_path: str, delegate, opts: dict) -> vision.HandLandmarker:
        base = mpy.BaseOptions(model_asset_path=model_path, delegate=delegate)
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.LIVE_STREAM,
                result_callback=self._on_result,
                **opts,
            )
        )

    # -- MediaPipe's thread: convert and stash only. No drawing, no blocking. ----
    def _on_result(self, result, image, timestamp_ms: int) -> None:
        done = time.perf_counter()
        hands: dict[str, np.ndarray] = {}
        scores: dict[str, float] = {}
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            cat = handed[0]
            label = true_handedness(cat.category_name)
            arr = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            # If both hands report the same label (happens when hands overlap), keep the
            # more confident one. A positional tiebreak comes in Phase 7.
            if label not in hands or cat.score > scores[label]:
                hands[label], scores[label] = arr, float(cat.score)

        with self._lock:
            capture_ts, submit_ts, seq = self._pending.pop(timestamp_ms, (0.0, 0.0, 0))
            # Stale results can arrive out of order; never let one clobber a newer result.
            if self._latest is not None and timestamp_ms < self._latest.ts_ms:
                self.dropped += 1
                return
            self._latest = HandsResult(hands, scores, timestamp_ms,
                                       capture_ts, submit_ts, done, seq)
            # Guard against unbounded growth if any submit never gets a callback.
            if len(self._pending) > 60:
                for old in sorted(self._pending)[:-30]:
                    del self._pending[old]

    def submit(self, bgr: np.ndarray, capture_ts: float, seq: int) -> None:
        """Hand a frame to MediaPipe. Non-blocking; the result arrives via callback."""
        ts_ms = int(capture_ts * 1000.0)
        with self._lock:
            if ts_ms <= self._last_ts_ms:      # enforce strictly increasing
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
            self._pending[ts_ms] = (capture_ts, time.perf_counter(), seq)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.lm.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

    def latest(self) -> HandsResult | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self.lm.close()

    def __enter__(self) -> "Landmarker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
