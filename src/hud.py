"""HUD overlay — Phase 1 subset: skeletons, handedness, and latency stats.

Runs on the **main thread** only. Phase 6 extends this with pinch/motion bars and note
state; the rule that it must never block the rest of the pipeline starts now.

Connections are hard-coded rather than imported from ``mp.solutions`` (the legacy API)
to keep this dependent only on the Tasks API.
"""
from __future__ import annotations

import cv2
import numpy as np

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (9, 10), (10, 11), (11, 12),               # middle
    (13, 14), (14, 15), (15, 16),              # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
)

# BGR. Distinct hues so left/right are unmistakable at a glance.
HAND_COLORS = {"Left": (255, 190, 60), "Right": (80, 160, 255)}
_WHITE = (255, 255, 255)
_DIM = (170, 170, 170)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hand(img: np.ndarray, lm: np.ndarray, label: str, score: float) -> None:
    """Draw one hand's skeleton. ``lm`` is (21,3) normalized coordinates."""
    h, w = img.shape[:2]
    color = HAND_COLORS.get(label, _WHITE)
    pts = np.empty((21, 2), dtype=np.int32)
    pts[:, 0] = np.clip(lm[:, 0] * w, -1e4, 1e4)
    pts[:, 1] = np.clip(lm[:, 1] * h, -1e4, 1e4)

    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        # Fingertips larger: they carry the pinch/pluck signal.
        cv2.circle(img, tuple(p), 5 if i in (4, 8, 12, 16, 20) else 3, color, -1, cv2.LINE_AA)

    x, y = int(pts[0][0]), int(pts[0][1])
    cv2.putText(img, f"{label} {score:.2f}", (x - 30, y + 24),
                _FONT, 0.55, color, 2, cv2.LINE_AA)


def draw_stats(img: np.ndarray, lines: list[str]) -> None:
    """Top-left stats panel, drawn over a translucent strip for legibility."""
    if not lines:
        return
    pad, lh = 8, 18
    height = pad * 2 + lh * len(lines)
    width = max(210, 8 * max(len(s) for s in lines) + pad * 2)
    panel = img[0:height, 0:width]
    cv2.rectangle(panel, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, img[0:height, 0:width], 0.55, 0, img[0:height, 0:width])
    for i, text in enumerate(lines):
        cv2.putText(img, text, (pad, pad + lh * (i + 1) - 5),
                    _FONT, 0.45, _WHITE if i == 0 else _DIM, 1, cv2.LINE_AA)


def draw_hint(img: np.ndarray, text: str) -> None:
    """Bottom-left hint line."""
    h = img.shape[0]
    cv2.putText(img, text, (8, h - 10), _FONT, 0.45, _DIM, 1, cv2.LINE_AA)
