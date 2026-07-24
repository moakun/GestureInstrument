#!/usr/bin/env python3
"""Probe which FluidSynth audio driver opens a real output device on this machine.

Creates *only* an audio driver (no MIDI router/driver, so no spurious "winmidi"
device errors), plays a short arpeggio, and reports whether the driver was created.
Run one driver per invocation so its stderr is attributable:

    .venv\\Scripts\\python.exe tools\\audio_probe.py dsound
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fsbootstrap import ensure_fluidsynth_on_path  # noqa: E402

ensure_fluidsynth_on_path()
import fluidsynth as F  # noqa: E402

driver = sys.argv[1] if len(sys.argv) > 1 else None
sf2 = str(ROOT / "assets" / "FluidR3_GM.sf2")

fs = F.Synth(samplerate=48000, gain=0.9)
sfid = fs.sfload(sf2)
fs.program_select(0, sfid, 0, 0)

default_drv = fs.get_setting("audio.driver")
print(f"compiled-default audio.driver = {default_drv!r}")
if driver:
    fs.setting("audio.driver", driver)

# audio-only start (bypass pyfluidsynth.start(), which also spins up MIDI)
ad = F.new_fluid_audio_driver(fs.settings, fs.synth)
print(f"AUDIO_DRIVER_CREATED={bool(ad)} driver={driver or default_drv!r}")

for d in [0, 4, 7, 12, 16]:
    fs.noteon(0, 60 + d, 115)
    time.sleep(0.28)
time.sleep(0.6)

if ad:
    F.delete_fluid_audio_driver(ad)
fs.delete()
print(f"DONE driver={driver or default_drv!r}")
