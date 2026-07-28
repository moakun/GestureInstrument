#!/usr/bin/env python3
"""Phase 4 tests: instrument semantics, retrigger, clamping — no audio device needed.

Everything runs with ``realtime=False``, so this passes on a machine with no sound card
and never makes a noise.

Run:  .venv\\Scripts\\python.exe tests\\test_audio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import audio                                                # noqa: E402

SF2 = ROOT / "assets" / "FluidR3_GM.sf2"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def test_null_backend() -> None:
    print("NullSynth (pipeline testing without hardware)")
    n = audio.NullSynth()
    n.note_on("piano", 60, 100)
    n.note_off("piano", 60)
    n.all_off()
    check("records calls in order",
          [e[0] for e in n.events] == ["on", "off", "all_off"], f"{n.events}")
    check("captures pitch and velocity", n.events[0] == ("on", "piano", 60, 100))


def test_channels_and_programs(s: audio.Synth) -> None:
    print("channel allocation")
    check("one channel per instrument", len(set(s.channels.values())) == len(s.programs))
    check("all instruments present", set(s.channels) == set(s.programs))
    check("channels are distinct ints",
          all(isinstance(c, int) for c in s.channels.values()))
    check("info reports instruments", set(s.info.instruments) == set(s.programs))


def test_sustain_semantics(s: audio.Synth) -> None:
    print("per-instrument sustain (plan 4.1)")
    check("piano sustains (damper matters)", s.sustains("piano") is True)
    check("guitar rings", s.sustains("guitar") is False)
    check("harp rings", s.sustains("harp") is False)

    s.all_off()
    s.note_on("piano", 60)
    check("piano note is sounding", s.sounding == 1)
    s.note_off("piano", 60)
    check("piano note-off releases it", s.sounding == 0)

    s.note_on("harp", 62)
    check("harp note is sounding", s.sounding == 1)
    s.note_off("harp", 62)
    check("harp note-off is a no-op - let it ring", s.sounding == 1,
          "sending note-off to a plucked sample cuts it dead")
    s.note_off("harp", 62, force=True)
    check("force=True releases a ringing note", s.sounding == 0)

    s.note_on("guitar", 64)
    s.note_on("harp", 67)
    s.all_off()
    check("all_off silences ringing instruments too", s.sounding == 0)


def test_retrigger(s: audio.Synth) -> None:
    print("retrigger")
    s.all_off()
    s.note_on("harp", 60)
    s.note_on("harp", 60)
    check("retriggering the same pitch does not double-count", s.sounding == 1)
    s.note_on("harp", 62)
    check("a different pitch adds a voice", s.sounding == 2)
    s.all_off()


def test_clamping(s: audio.Synth) -> None:
    print("input clamping")
    s.all_off()
    s.note_on("piano", 500, 999)          # nonsense from a bad mapping must not crash
    s.note_on("piano", -20, -5)
    check("out-of-range pitches are clamped, not crashed", s.sounding == 2)
    s.all_off()
    try:
        s.note_on("theremin", 60)
        check("unknown instrument raises", False, "no exception")
    except KeyError:
        check("unknown instrument raises KeyError", True)
    s.note_off("theremin", 60, force=True)     # must not raise
    check("note_off on unknown instrument is safe", True)


def test_initial_silence(s: audio.Synth) -> None:
    """Must run before anything plays: a note-off starts a release, it does not mute.

    Checking silence *after* other tests would be measuring their decay tails — and the
    clamping test deliberately plays pitch 0 (~8 Hz), which rings far longer than any
    musical note.
    """
    print("offline render - initial silence")
    silence = s.render(2048)
    check("render returns (N,2) int16", silence.shape == (2048, 2)
          and silence.dtype == np.int16, f"{silence.shape} {silence.dtype}")
    check("a fresh synth renders silence", int(np.abs(silence).max()) < 10,
          f"peak {int(np.abs(silence).max())}")


def test_render(s: audio.Synth) -> None:
    print("offline render")
    s.all_off()
    s.render(48000)                      # flush earlier release tails
    s.note_on("piano", 60, 110)
    tone = s.render(24000)
    peak = int(np.abs(tone.astype(np.int32)).max())
    check("a note produces audible output", peak > 1000, f"peak {peak}")
    check("output does not clip", peak < 32767, f"peak {peak}")
    s.all_off()

    # Polyphony: a chord must be louder than a single note, i.e. voices really sum.
    s.note_on("piano", 60, 110)
    one = int(np.abs(s.render(12000).astype(np.int32)).max())
    s.all_off()
    for p in (60, 64, 67, 72):
        s.note_on("piano", p, 110)
    many = int(np.abs(s.render(12000).astype(np.int32)).max())
    s.all_off()
    check("polyphonic voices sum", many > one, f"one={one} chord={many}")


def main() -> int:
    print("=" * 60)
    print("Gesture Instrument - Phase 4 audio tests")
    print("=" * 60)
    if not SF2.is_file():
        print(f"SKIP: missing SoundFont {SF2}")
        return 0
    test_null_backend()
    # One synth for every remaining test: loading a 148 MB SoundFont per test is slow.
    with audio.Synth(str(SF2), realtime=False) as s:
        check("offline synth reports realtime=False", s.realtime is False)
        test_initial_silence(s)                # must precede anything that plays
        test_channels_and_programs(s)
        test_sustain_semantics(s)
        test_retrigger(s)
        test_clamping(s)
        test_render(s)
    print("-" * 60)
    if _failures:
        print(f"FAILED: {len(_failures)} -> {', '.join(_failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
