#!/usr/bin/env python3
"""End-to-end test of the first playable path: gesture in, correct note out.

Runs the whole chain — synthetic landmarks -> features -> state machine -> Mode C
mapping -> synth — against `NullSynth`, so it is deterministic and silent.

Run:  .venv\\Scripts\\python.exe tests\\test_playable.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import audio                                                    # noqa: E402
import gesture_stream as G                                      # noqa: E402
import statemachine as S                                        # noqa: E402
from main import resolve_pitch                                  # noqa: E402

with open(ROOT / "config" / "mappings.yaml", encoding="utf-8") as _fh:
    CFG = yaml.safe_load(_fh)
N_CELLS = len(CFG["scale"]) * CFG["continuous"]["octaves"]

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def test_resolve_pitch() -> None:
    print("Mode C pitch resolution")
    scale, root = CFG["scale"], CFG["root"]
    inst0, p0 = resolve_pitch(CFG, 0, 3)               # left count 3 -> harp, octave 0
    check("cell 0 is the root", p0 == root, f"got {p0}")
    check("left count picks the instrument", inst0 == CFG["left_hand"][3]["instrument"],
          f"got {inst0}")

    pitches = [resolve_pitch(CFG, c, 3)[1] for c in range(N_CELLS)]
    check("pitch rises monotonically with height cell",
          all(b > a for a, b in zip(pitches, pitches[1:])), f"{pitches}")
    check("one octave up after a full scale",
          pitches[len(scale)] == pitches[0] + 12, f"{pitches[:len(scale) + 1]}")
    check("only pentatonic degrees are produced",
          {(p - root) % 12 for p in pitches} <= set(scale), f"{sorted({(p - root) % 12 for p in pitches})}")

    # Left hand selects octave as well as instrument: 4 -> harp+1, 3 -> harp+0.
    _, high = resolve_pitch(CFG, 0, 4)
    check("left count 4 shifts up an octave", high == p0 + 12, f"{high} vs {p0}")
    check("unknown left count still resolves", resolve_pitch(CFG, 0, None)[1] > 0)
    check("pitch stays in MIDI range", all(0 <= p <= 127 for p in pitches))


def play(stream: G.Stream, left_count: int = 3) -> list:
    """Run a stream through tracker + mapping into a NullSynth; return its events."""
    synth = audio.NullSynth()
    tracker = S.HandTracker("Right")
    quant = S.HysteresisQuantizer(N_CELLS, CFG["continuous"]["deadband"])
    sounding = None
    for t, f in G.features_stream(stream):
        cell = quant(1.0 - f.y if CFG["continuous"].get("invert", True) else f.y)
        for e in tracker.update(f, t):
            if e.kind == "trigger_on":
                inst, pitch = resolve_pitch(CFG, cell, left_count)
                synth.note_on(inst, pitch, 100)
                sounding = (inst, pitch)
            elif e.kind == "trigger_off" and sounding:
                synth.note_off(*sounding)
                sounding = None
    return synth.events


def _pinch_at(y: float, seed: int) -> G.Stream:
    s = G.Stream(seed=seed).at(0.5, y)
    s.hold(G.pose(1), 0.4)
    s.move_to(G.pose(1, pinch=True), 0.05).hold(G.pose(1, pinch=True), 0.2)
    s.move_to(G.pose(1), 0.05).hold(G.pose(1), 0.2)
    return s


def test_height_controls_pitch() -> None:
    print("hand height controls pitch (the Mode C payoff)")
    high = [e for e in play(_pinch_at(0.15, 1)) if e[0] == "on"]
    low = [e for e in play(_pinch_at(0.85, 2)) if e[0] == "on"]
    check("a high hand plays one note", len(high) == 1, f"{high}")
    check("a low hand plays one note", len(low) == 1, f"{low}")
    if high and low:
        check("higher hand -> higher pitch", high[0][2] > low[0][2],
              f"high={high[0][2]} low={low[0][2]}")
        print(f"       y=0.15 -> MIDI {high[0][2]}   y=0.85 -> MIDI {low[0][2]}")


def test_instrument_selection() -> None:
    print("left hand selects the instrument")
    for left_count, expected in ((2, "guitar"), (3, "harp"), (5, "piano")):
        got = [e for e in play(_pinch_at(0.5, 5), left_count=left_count) if e[0] == "on"]
        check(f"left count {left_count} -> {expected}",
              bool(got) and got[0][1] == expected, f"{got}")


def test_sustain_routing() -> None:
    print("release behaviour follows the instrument")
    # Harp rings: NullSynth records the note_off call, but the real Synth drops it.
    with audio.Synth(str(ROOT / "assets" / "FluidR3_GM.sf2"),
                     CFG["instruments"], CFG["sustain"], realtime=False) as s:
        s.note_on("harp", 60)
        s.note_off("harp", 60)
        check("harp keeps ringing after release", s.sounding == 1)
        s.note_on("piano", 62)
        s.note_off("piano", 62)
        check("piano is damped on release", s.sounding == 1, "only the harp should remain")
        s.all_off()


def test_no_sound_without_pinch() -> None:
    print("no note without a deliberate pinch")
    s = G.Stream(seed=9).at(0.5, 0.3)
    for target in (1, 2, 3, 4, 5, 4, 3, 2, 1, 0):
        s.move_to(G.pose(target), 0.12).hold(G.pose(target), 0.08)
    events = play(s)
    check("sweeping fingers alone makes no sound", not events, f"{events}")


def main() -> int:
    print("=" * 62)
    print("Gesture Instrument - first playable path (Phase 4)")
    print("=" * 62)
    for fn in (test_resolve_pitch, test_height_controls_pitch, test_instrument_selection,
               test_sustain_routing, test_no_sound_without_pinch):
        fn()
    print("-" * 62)
    if _failures:
        print(f"FAILED: {len(_failures)} -> {', '.join(_failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
