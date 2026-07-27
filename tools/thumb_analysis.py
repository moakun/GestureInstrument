#!/usr/bin/env python3
"""Find a thumb feature that actually separates count 4 (tucked) from 5 (out).

The default `thumb_open` (thumb tip -> index MCP) barely separates them on real hands:
with all four fingers extended the thumb tip stays roughly the same distance from the
index MCP whether it is tucked alongside or abducted. This scores candidate features by
Fisher discriminant d' = |mu5 - mu4| / (sigma4 + sigma5), which rewards a *wide, stable*
gap rather than a threshold that happens to fit one clip.

    .venv\\Scripts\\python.exe tools\\thumb_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import features as F                                   # noqa: E402
import replay                                          # noqa: E402

WRIST, T_CMC, T_MCP, T_IP, T_TIP = 0, 1, 2, 3, 4
INDEX_MCP, INDEX_TIP, PINKY_MCP = 5, 8, 17


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def candidates(lm: np.ndarray) -> dict[str, float]:
    """Every plausible thumb-extension signal, all scale-normalized."""
    s = F.palm_scale(lm)
    return {
        "tip->index_mcp (current)": np.linalg.norm(lm[T_TIP] - lm[INDEX_MCP]) / s,
        "tip->pinky_mcp": np.linalg.norm(lm[T_TIP] - lm[PINKY_MCP]) / s,
        "tip->wrist": np.linalg.norm(lm[T_TIP] - lm[WRIST]) / s,
        "tip->index_tip (pinch)": np.linalg.norm(lm[T_TIP] - lm[INDEX_TIP]) / s,
        "ip_curl_cos(2,3,4)": _angle(lm[T_MCP] - lm[T_IP], lm[T_TIP] - lm[T_IP]),
        "mcp_curl_cos(1,2,3)": _angle(lm[T_CMC] - lm[T_MCP], lm[T_IP] - lm[T_MCP]),
        "abduction cos(thumb,index)": _angle(lm[T_TIP] - lm[T_CMC], lm[INDEX_MCP] - lm[WRIST]),
        "abduction cos(thumb,palm)": _angle(lm[T_TIP] - lm[T_CMC], lm[PINKY_MCP] - lm[INDEX_MCP]),
        "tip perp dist to palm axis": float(
            np.linalg.norm(np.cross(lm[T_TIP] - lm[WRIST],
                                    (lm[INDEX_MCP] - lm[WRIST])
                                    / (np.linalg.norm(lm[INDEX_MCP] - lm[WRIST]) + 1e-9)))) / s,
    }


def main() -> int:
    paths = sorted((ROOT / "data").glob("counts_*.npz"))
    if not paths:
        print("No recording found.")
        return 1
    s = replay.load(paths[0])
    names = list(candidates(s.lm_world[0]))
    by_class: dict[int, dict[str, list[float]]] = {c: {n: [] for n in names} for c in range(6)}
    for i in range(len(s)):
        c = int(s.label[i])
        if 0 <= c <= 5:
            for n, v in candidates(s.lm_world[i]).items():
                by_class[c][n].append(v)

    print(f"{paths[0].name}: separating count 4 (thumb tucked) from 5 (thumb out)\n")
    rows = []
    for n in names:
        a = np.array(by_class[4][n]); b = np.array(by_class[5][n])
        d = abs(b.mean() - a.mean()) / (a.std() + b.std() + 1e-9)
        # A threshold is only usable if the two ranges genuinely clear each other.
        gap = (b.min() - a.max()) if b.mean() > a.mean() else (a.min() - b.max())
        rows.append((d, gap, n, a, b))
    rows.sort(reverse=True, key=lambda r: r[0])

    print(f"{'feature':<30}{'count4 mean+-sd':>20}{'count5 mean+-sd':>20}{'d-prime':>9}{'gap':>8}")
    for d, gap, n, a, b in rows:
        flag = "  <== separable" if gap > 0.02 else ("  (overlaps)" if gap <= 0 else "")
        print(f"{n:<30}{a.mean():>12.3f}+-{a.std():.3f}{b.mean():>12.3f}+-{b.std():.3f}"
              f"{d:>9.2f}{gap:>8.3f}{flag}")

    best_d, best_gap, best_n, a, b = rows[0]
    print(f"\nbest: {best_n}  d'={best_d:.2f}  clear gap={best_gap:.3f}")
    if best_gap > 0:
        lo, hi = (a.max(), b.min()) if b.mean() > a.mean() else (b.max(), a.min())
        print(f"      midpoint threshold = {(lo + hi) / 2:.3f}   "
              f"(count4 up to {lo:.3f}, count5 from {hi:.3f})")

    # How the full range behaves across every class, to confirm monotonic behaviour.
    print(f"\nper-class medians for '{best_n}':")
    for c in range(6):
        v = np.array(by_class[c][best_n])
        if v.size:
            print(f"  count {c}: {np.median(v):.3f}  (min {v.min():.3f}, max {v.max():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
