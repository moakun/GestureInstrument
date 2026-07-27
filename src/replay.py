"""Replay recorded landmark sessions through the feature layer.

Lets thresholds be scored and tuned offline against ground truth, with no camera in the
loop. Used by ``tests/test_replay.py`` (Phase 2 exit criteria) and
``tools/tune_thresholds.py``.

Motion energy is computed **within a recording segment only** — consecutive frames
across a segment boundary are seconds apart and would fabricate a huge false motion
spike right where the ground-truth label changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import features as F

# Above this palm-widths/second the pose is in transit and its classification is
# meaningless; those frames are excluded from "static" scoring (plan 2.4/3.2).
STATIC_MAX_MOTION = 1.2


@dataclass(frozen=True)
class Session:
    """A recorded clip. ``lm_world`` drives shape, ``lm_norm`` drives screen position."""

    ts: np.ndarray          # (N,)     float64
    lm_norm: np.ndarray     # (N,21,3) float32
    lm_world: np.ndarray    # (N,21,3) float32
    label: np.ndarray       # (N,)     int8, -1 = unlabelled
    seg: np.ndarray         # (N,)     int16
    hand: str
    aspect: float
    path: Path

    def __len__(self) -> int:
        return int(self.ts.shape[0])


def load(path: str | Path) -> Session:
    """Load a .npz written by ``tools/record_session.py``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        return Session(
            ts=z["ts"], lm_norm=z["lm_norm"], lm_world=z["lm_world"],
            label=z["label"], seg=z["seg"],
            hand=str(z["hand"]), aspect=float(z["aspect"]), path=path,
        )


def motion_series(s: Session) -> np.ndarray:
    """Per-frame motion energy, (N,) float32. First frame of each segment is 0."""
    out = np.zeros(len(s), dtype=np.float32)
    for i in range(1, len(s)):
        if s.seg[i] != s.seg[i - 1]:
            continue                              # never measure across a gap
        dt = float(s.ts[i] - s.ts[i - 1])
        out[i] = F.motion_energy(s.lm_world[i], s.lm_world[i - 1], dt)
    return out


def count_series(s: Session, curl_threshold: float = F.CURL_EXTENDED,
                 thumb_threshold: float = F.THUMB_ABDUCTED) -> np.ndarray:
    """Predicted finger count per frame, (N,) int8."""
    return np.array(
        [F.finger_count(s.lm_world[i], curl_threshold=curl_threshold,
                        thumb_threshold=thumb_threshold) for i in range(len(s))],
        dtype=np.int8,
    )


@dataclass(frozen=True)
class Score:
    accuracy: float
    n_static: int
    n_total: int
    confusion: np.ndarray          # (6,6) int; rows = truth, cols = predicted

    @property
    def per_class(self) -> dict[int, float]:
        totals = self.confusion.sum(axis=1)
        return {c: (float(self.confusion[c, c]) / totals[c] if totals[c] else float("nan"))
                for c in range(self.confusion.shape[0])}


def score_counts(s: Session, curl_threshold: float = F.CURL_EXTENDED,
                 thumb_threshold: float = F.THUMB_ABDUCTED,
                 max_motion: float = STATIC_MAX_MOTION) -> Score:
    """Finger-count accuracy on *static*, labelled frames.

    Static-only is the honest measure: transitional frames are genuinely ambiguous, and
    the state machine (Phase 3) gates on motion precisely so they never reach the mapping.
    """
    motion = motion_series(s)
    pred = count_series(s, curl_threshold, thumb_threshold)
    keep = (s.label >= 0) & (motion <= max_motion)
    truth = s.label[keep].astype(int)
    got = pred[keep].astype(int)

    confusion = np.zeros((6, 6), dtype=int)
    for t, p in zip(truth, got):
        if 0 <= t <= 5 and 0 <= p <= 5:
            confusion[t, p] += 1
    acc = float((truth == got).mean()) if truth.size else 0.0
    return Score(acc, int(truth.size), len(s), confusion)


def format_confusion(c: np.ndarray) -> str:
    """Readable confusion matrix; rows = ground truth, columns = predicted."""
    lines = ["      pred:" + "".join(f"{p:>6d}" for p in range(6)) + "     acc"]
    for t in range(6):
        total = c[t].sum()
        acc = f"{100.0 * c[t, t] / total:5.1f}%" if total else "    -"
        lines.append(f"  true {t}:" + "".join(f"{c[t, p]:>6d}" for p in range(6)) + f"  {acc}")
    return "\n".join(lines)
