"""Pure feature extraction: (21,3) landmarks -> scalars. Zero state, zero I/O.

Everything here is a pure function of its arguments, which is the plan's single best
architectural bet (1.x): record 30 seconds of landmarks once, then iterate on thresholds
a hundred times without ever touching a camera.

**Coordinate spaces.** Shape features (curls, thumb, pinch) need an *isotropic* space or
their angles are wrong. Two valid inputs:

* MediaPipe **world landmarks** — metric, isotropic. Preferred; pass them straight in.
* **Normalized image** landmarks passed through :func:`isotropic` first. Raw normalized
  coords are anisotropic (x spans frame width, y spans height), so on a 4:3 camera a
  straight finger reads a different curl angle horizontally than vertically — which
  defeats the whole point of using angles instead of the ``tip.y < pip.y`` heuristic.

:func:`hand_y` is the exception: it wants **normalized** coords, since it measures
position on screen, not shape. (Scaling x never affects y, so either array works.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (mcp, pip, tip) per finger. The thumb is deliberately absent — see `thumb_open`.
FINGERS: dict[str, tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
FINGER_NAMES = tuple(FINGERS)

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
PALM_IDX = (0, 5, 9, 13, 17)          # wrist + the four finger MCPs

# Defaults; Phase 7 calibration replaces these per user.
CURL_EXTENDED = -0.60                  # cos below this = finger straight
CURL_RETRACTED = -0.25                 # cos above this = finger folded
THUMB_OPEN = 0.55
THUMB_CLOSED = 0.40
PINCH_CLOSE = 0.25
PINCH_OPEN = 0.35
_EPS = 1e-9


def isotropic(lm: np.ndarray, aspect: float) -> np.ndarray:
    """Undistort normalized image landmarks so distances/angles are meaningful.

    ``aspect`` is width/height (1.333 for 640x480). MediaPipe documents z as using
    roughly the same scale as x, so it gets the same correction.
    """
    out = np.array(lm, dtype=np.float32, copy=True)
    out[:, 0] *= aspect
    out[:, 2] *= aspect
    return out


def palm_scale(lm: np.ndarray) -> float:
    """Wrist -> middle-finger MCP. The reference length everything else divides by.

    Never threshold on raw coordinates: this is what makes features invariant to how
    close the hand is to the camera.
    """
    return float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST])) + 1e-6


def curl_cos(lm: np.ndarray, mcp: int, pip: int, tip: int) -> float:
    """Cosine of the joint angle at PIP. ~-1.0 = straight, ~0..+1 = folded.

    Rotation-invariant by construction, unlike the ``tip.y < pip.y`` heuristic every
    tutorial uses — that one breaks the instant you tilt your hand, and you *will* tilt
    your hand while playing (plan 2.2).
    """
    a = lm[mcp] - lm[pip]
    b = lm[tip] - lm[pip]
    a = a / (np.linalg.norm(a) + _EPS)
    b = b / (np.linalg.norm(b) + _EPS)
    return float(np.dot(a, b))


def finger_curls(lm: np.ndarray) -> dict[str, float]:
    """Curl cosine for the four non-thumb fingers."""
    return {name: curl_cos(lm, *idx) for name, idx in FINGERS.items()}


def thumb_open(lm: np.ndarray, s: float | None = None) -> float:
    """Thumb abduction: |thumb tip - index MCP| / palm scale.

    The thumb needs its own rule. Its PIP angle barely changes between open and closed;
    what actually changes is abduction. Expect this to be the worst-performing digit
    regardless — never make thumb state load-bearing in a mapping (plan 2.2).
    """
    s = palm_scale(lm) if s is None else s
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP])) / s


def pinch_ratio(lm: np.ndarray, s: float | None = None) -> float:
    """|thumb tip - index tip| / palm scale. The trigger signal; < ~0.25 is pinched."""
    s = palm_scale(lm) if s is None else s
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / s


def motion_energy(lm: np.ndarray, prev_lm: np.ndarray | None, dt: float,
                  s: float | None = None) -> float:
    """Mean per-landmark speed, in palm-widths per second.

    The key insight of the whole project: **while the hand is moving fast, its pose
    classification is meaningless.** Transitional poses (the 3 and 4 you pass through
    going from 2 to 5) occur exactly when motion is high, so gating selection changes on
    low motion removes phantom notes almost for free — and unlike a longer debounce
    window it costs *zero* latency when the hand is already still (plan 2.4).
    """
    if prev_lm is None or dt <= 0.0 or lm.shape != prev_lm.shape:
        return 0.0
    s = palm_scale(lm) if s is None else s
    return float(np.linalg.norm(lm - prev_lm, axis=1).mean()) / (s * dt)


def hand_y(lm: np.ndarray) -> float:
    """Palm centroid height in **normalized** coords: 0.0 = top of frame, 1.0 = bottom.

    This is the Mode C pitch axis. Vertical hand position is high-resolution, low-latency
    and needs no classification at all, which sidesteps the hardest problem in the project.
    """
    return float(lm[list(PALM_IDX), 1].mean())


def hand_x(lm: np.ndarray) -> float:
    """Palm centroid horizontal position, normalized. Used for the Phase 7 L/R tiebreak."""
    return float(lm[list(PALM_IDX), 0].mean())


def extended_fingers(curls: dict[str, float],
                     threshold: float = CURL_EXTENDED) -> dict[str, bool]:
    """Stateless extension test. Phase 3 replaces this with a Schmitt trigger.

    A single threshold oscillates at the boundary; that is Phase 3's problem to fix with
    hysteresis. Here it stays stateless so features remain pure.
    """
    return {name: c < threshold for name, c in curls.items()}


def finger_count(lm: np.ndarray, s: float | None = None,
                 curl_threshold: float = CURL_EXTENDED,
                 thumb_threshold: float = THUMB_OPEN) -> int:
    """Count 0-5, structured so the unreliable thumb is never load-bearing.

    Counts 1-4 use the four fingers only; 5 additionally requires an open thumb. So a
    misread thumb can only ever confuse 4 with 5, never shift 1<->2 (plan 2.2).
    """
    s = palm_scale(lm) if s is None else s
    n = sum(extended_fingers(finger_curls(lm), curl_threshold).values())
    if n == 4 and thumb_open(lm, s) > thumb_threshold:
        return 5
    return n


@dataclass(frozen=True)
class HandFeatures:
    """Everything Phase 3 needs from one hand on one frame."""

    curls: dict[str, float]
    n_extended: int
    count: int
    thumb: float
    pinch: float
    motion: float
    y: float
    x: float
    scale: float


def extract(shape_lm: np.ndarray, screen_lm: np.ndarray | None = None,
            prev_shape_lm: np.ndarray | None = None, dt: float = 0.0) -> HandFeatures:
    """Full feature set for one hand.

    Args:
        shape_lm: isotropic landmarks (world, or normalized via :func:`isotropic`).
        screen_lm: normalized image landmarks for position. Defaults to ``shape_lm``.
        prev_shape_lm: previous frame's ``shape_lm``, for motion energy.
        dt: seconds since the previous frame.
    """
    screen = shape_lm if screen_lm is None else screen_lm
    s = palm_scale(shape_lm)
    curls = finger_curls(shape_lm)
    n_ext = sum(extended_fingers(curls).values())
    thumb = thumb_open(shape_lm, s)
    return HandFeatures(
        curls=curls,
        n_extended=n_ext,
        count=5 if (n_ext == 4 and thumb > THUMB_OPEN) else n_ext,
        thumb=thumb,
        pinch=pinch_ratio(shape_lm, s),
        motion=motion_energy(shape_lm, prev_shape_lm, dt, s),
        y=hand_y(screen),
        x=hand_x(screen),
        scale=s,
    )
