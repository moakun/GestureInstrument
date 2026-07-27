"""Synthetic hand geometry for testing features without a camera.

Builds anatomically-plausible (21,3) landmark arrays with *known* finger states, so the
feature math can be checked against exact expected values and invariance properties.

Layout matches MediaPipe's: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
17-20 pinky. Fingers point along -y ("up" in image convention); the palm lies in z=0.
"""
from __future__ import annotations

import math

import numpy as np

# (mcp_index, mcp_position) for each finger, in palm-scale units.
_MCP = {
    "index": (5, (-0.30, -0.95, 0.0)),
    "middle": (9, (0.00, -1.00, 0.0)),      # defines palm_scale = 1.0
    "ring": (13, (0.28, -0.93, 0.0)),
    "pinky": (17, (0.52, -0.82, 0.0)),
}
_SEG = 0.40          # phalanx length


def synthetic_hand(extended: set[str] | None = None, thumb_out: bool = False,
                   pinch: bool = False, bend_deg: float | None = None) -> np.ndarray:
    """A (21,3) float32 hand.

    Args:
        extended: finger names that should be straight; the rest are folded.
        thumb_out: thumb abducted away from the index MCP.
        pinch: place the thumb tip on the index tip (overrides ``thumb_out`` position).
        bend_deg: if given, every finger bends by this angle at PIP instead of being
            fully straight/folded, giving ``curl_cos == -cos(bend)``. Needed to test
            anything sensitive to the *actual* angle: fully straight (0 deg) and fully
            folded (180 deg) are exactly antiparallel/parallel vector pairs, and those
            survive any linear map unchanged — so they cannot reveal skew distortion.
    """
    extended = set() if extended is None else set(extended)
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.0, 0.0, 0.0)                                  # wrist

    for name, (mcp_i, mcp_pos) in _MCP.items():
        mcp = np.array(mcp_pos, dtype=np.float32)
        pip = mcp + (0.0, -_SEG, 0.0)
        if bend_deg is not None:
            # Continue straight (0,-1,0), rotated by `bend` in the palm plane.
            theta = math.radians(bend_deg)
            d = np.array([math.sin(theta), -math.cos(theta), 0.0], dtype=np.float32)
            dip, tip = pip + _SEG * 0.6 * d, pip + _SEG * d
        elif name in extended:
            dip = pip + (0.0, -_SEG * 0.6, 0.0)
            tip = pip + (0.0, -_SEG, 0.0)                    # straight -> curl cos = -1
        else:
            dip = pip + (0.0, _SEG * 0.5, 0.0)
            tip = pip + (0.0, _SEG * 0.9, 0.0)               # folded back -> cos = +1
        lm[mcp_i], lm[mcp_i + 1], lm[mcp_i + 2], lm[mcp_i + 3] = mcp, pip, dip, tip

    # Thumb chain 1..4, splayed to the -x side.
    lm[1] = (-0.30, -0.20, 0.0)
    lm[2] = (-0.55, -0.40, 0.0)
    if pinch:
        lm[3] = lm[8] + (-0.12, 0.10, 0.0)
        lm[4] = lm[8] + (0.0, 0.02, 0.0)                     # thumb tip on index tip
    elif thumb_out:
        lm[3] = (-0.80, -0.55, 0.0)
        lm[4] = (-1.05, -0.70, 0.0)                          # far from index MCP
    else:
        lm[3] = (-0.45, -0.62, 0.0)
        lm[4] = (-0.22, -0.80, 0.0)                          # tucked near index MCP
    return lm


def rotation(rx: float = 0.0, ry: float = 0.0, rz: float = 0.0) -> np.ndarray:
    """3x3 rotation from intrinsic X-Y-Z Euler angles (radians)."""
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry),
                              np.sin(ry), np.cos(rz), np.sin(rz))
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def transform(lm: np.ndarray, rot: np.ndarray | None = None, scale: float = 1.0,
              translate: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Apply rotation, uniform scale, then translation."""
    out = np.array(lm, dtype=np.float32, copy=True)
    if rot is not None:
        out = out @ rot.T
    out *= scale
    out += np.array(translate, dtype=np.float32)
    return out.astype(np.float32)


def to_screen(lm: np.ndarray, center: tuple[float, float] = (0.5, 0.5),
              scale: float = 0.15, aspect: float = 640.0 / 480.0) -> np.ndarray:
    """Project an isotropic hand into anisotropic normalized image coords.

    Inverts what `features.isotropic` corrects: x (and z) are compressed by the aspect
    ratio, exactly as a real camera's normalized coordinates are.
    """
    out = np.array(lm, dtype=np.float32, copy=True) * scale
    out[:, 0] = out[:, 0] / aspect + center[0]
    out[:, 1] = out[:, 1] + center[1]
    out[:, 2] = out[:, 2] / aspect
    return out
