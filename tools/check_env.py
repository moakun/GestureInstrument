#!/usr/bin/env python3
"""Phase 0 exit-criteria check for the Gesture Instrument.

Verifies that (1) every Python dependency imports, (2) the two required assets are
present and non-trivial, and (3) FluidSynth can actually synthesize audio.

The audio check is done two ways:

* **Offline render** (definitive, headless-safe): ``get_samples`` renders a note to a
  numpy buffer and we assert it is non-silent. This proves the synthesis path works
  even when no audio device is reachable from this process.
* **Live playback** (best-effort): if an audio driver opens, a short C-major-pentatonic
  phrase is played on piano / guitar / harp so you can *hear* it.

Run:  .venv\\Scripts\\python.exe tools\\check_env.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fsbootstrap import ensure_fluidsynth_on_path  # noqa: E402

GM = {"piano": 0, "guitar": 24, "harp": 46}   # General MIDI program numbers
PENTATONIC = [0, 2, 4, 7, 9]                   # C major pentatonic (semitone offsets)
ROOT_NOTE = 60                                 # C4


def check_imports() -> object:
    print("[1/3] imports")
    import numpy
    import cv2
    import mediapipe
    import sounddevice
    import yaml

    ensure_fluidsynth_on_path()
    import fluidsynth

    print(f"      numpy       {numpy.__version__}")
    print(f"      opencv      {cv2.__version__}")
    print(f"      mediapipe   {mediapipe.__version__}")
    print(f"      sounddevice {sounddevice.__version__}")
    print(f"      pyyaml       {yaml.__version__}")
    print(f"      pyfluidsynth {fluidsynth.api_version}")
    return fluidsynth


def check_assets() -> Path:
    print("[2/3] assets")
    model = ROOT / "assets" / "hand_landmarker.task"
    sf2 = ROOT / "assets" / "FluidR3_GM.sf2"
    assert model.is_file(), f"MISSING: {model}"
    assert sf2.is_file(), f"MISSING: {sf2}"
    assert model.stat().st_size > 1_000_000, "hand_landmarker.task looks truncated"
    assert sf2.stat().st_size > 50_000_000, "FluidR3_GM.sf2 looks truncated"
    with open(sf2, "rb") as fh:
        assert fh.read(4) == b"RIFF", "FluidR3_GM.sf2 is not a valid SoundFont"
    print(f"      hand_landmarker.task  {model.stat().st_size / 1e6:5.1f} MB")
    print(f"      FluidR3_GM.sf2        {sf2.stat().st_size / 1e6:5.1f} MB  (RIFF ok)")
    return sf2


def check_synth(fluidsynth, sf2: Path) -> None:
    print("[3/3] fluidsynth")
    import numpy as np

    # --- definitive offline proof: render a note, assert non-silent ---
    fs = fluidsynth.Synth(samplerate=48000, gain=0.6)
    sfid = fs.sfload(str(sf2))
    assert sfid != -1, "sfload failed"
    for ch, (name, prog) in enumerate(GM.items()):
        fs.program_select(ch, sfid, 0, prog)
    fs.noteon(0, ROOT_NOTE, 100)
    samples = np.asarray(fs.get_samples(48000))   # 1s stereo int16
    fs.noteoff(0, ROOT_NOTE)
    peak = int(np.abs(samples.astype(np.int32)).max())
    fs.delete()
    assert peak > 0, "offline render was silent — synthesis path broken"
    print(f"      offline render peak amplitude: {peak} / 32767  -> synthesis OK")

    # --- best-effort live playback so you can hear it ---
    # Start audio *only*: bypass Synth.start(), which also spins up a MIDI-input
    # driver (winmidi) we don't want -> harmless but noisy "no MIDI device" errors.
    # This audio-only pattern is what the Phase 4 backend should use.
    fs = fluidsynth.Synth(samplerate=48000, gain=0.8)
    sfid = fs.sfload(str(sf2))
    for ch, (name, prog) in enumerate(GM.items()):
        fs.program_select(ch, sfid, 0, prog)
    audio_driver = fluidsynth.new_fluid_audio_driver(fs.settings, fs.synth)
    live_ok = bool(audio_driver)
    if live_ok:
        print("      audio driver opened (dsound); playing test phrase "
              "(piano -> guitar -> harp)...")
        for ch, name in ((0, "piano"), (1, "guitar"), (2, "harp")):
            for deg in PENTATONIC:
                fs.noteon(ch, ROOT_NOTE + deg, 110)
                time.sleep(0.16)
            time.sleep(0.15)
            for deg in PENTATONIC:
                fs.noteoff(ch, ROOT_NOTE + deg)
        time.sleep(0.4)
        fluidsynth.delete_fluid_audio_driver(audio_driver)
    else:
        print("      audio driver did not open; offline synthesis already verified")
    fs.delete()


def main() -> int:
    print("=" * 56)
    print("Gesture Instrument - Phase 0 environment check")
    print("=" * 56)
    fluidsynth = check_imports()
    sf2 = check_assets()
    check_synth(fluidsynth, sf2)
    print("-" * 56)
    print("PHASE 0 ENVIRONMENT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
