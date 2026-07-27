# Gesture Instrument

Hand-tracked harp / guitar / piano. Two hands in front of a webcam: finger poses and
hand height select instrument + pitch; a pinch fires the note. See
[`gesture-instrument-plan.md`](gesture-instrument-plan.md) for the full design.

**Stack:** Python 3.11 · MediaPipe HandLandmarker · OpenCV · FluidSynth · numpy

## Locked decisions (Phase 0)

| Decision | Choice |
|---|---|
| Mapping mode | **C — continuous pitch** (right-hand height → pitch, pinch = note-on) |
| Trigger | **Pinch** (single-frame, lowest latency) |
| Audio backend | **FluidSynth** (`dsound` driver on Windows) |
| Scale | **C major pentatonic** `[0, 2, 4, 7, 9]`, root C4 = 60 |

Machine-readable config: [`config/mappings.yaml`](config/mappings.yaml).

## Setup (Windows)

The venv, the SoundFont (`.sf2`), the model (`.task`), and the vendored FluidSynth
binaries are **git-ignored** — reconstruct them with the steps below. (Requires
Python 3.11 on `PATH`; verified with 3.11.8.)

### 1. Python environment

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Assets (~156 MB total)

```bash
# MediaPipe hand landmarker (float16)
curl -L -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# FluidR3_GM.sf2 General MIDI SoundFont (~148 MB) — has piano/guitar/harp
curl -L -o assets/FluidR3_GM.sf2 \
  "https://github.com/urish/cinto/raw/master/media/FluidR3%20GM.sf2"
```

### 3. FluidSynth native library

FluidSynth is a **native** library; `pip install pyfluidsynth` only provides the Python
bindings. There is no winget/choco/scoop package on this setup, so we **vendor** the
official Windows release into `vendor/fluidsynth/` (git-ignored):

1. Download `fluidsynth-v2.5.6-win10-x64-cpp11.zip` from
   <https://github.com/FluidSynth/fluidsynth/releases>.
2. Extract so that `vendor/fluidsynth/bin/libfluidsynth-3.dll` exists.

[`src/fsbootstrap.py`](src/fsbootstrap.py) puts that `bin` dir on `PATH` and the DLL
search path at runtime, so `import fluidsynth` finds it — **no changes to system PATH
needed.** Call `ensure_fluidsynth_on_path()` once before importing `fluidsynth`.

> On macOS/Linux install FluidSynth system-wide instead (`brew install fluid-synth` /
> `sudo apt install fluidsynth libfluidsynth3`); the bootstrap is then a no-op.

## Verify (Phase 0 exit criteria)

```bash
.venv/Scripts/python.exe tools/check_env.py
```

Checks every import, validates the assets, proves FluidSynth synthesis (offline render),
and plays a short piano→guitar→harp phrase. Expected last line: `PHASE 0 ENVIRONMENT: OK`.

Optional — find your working audio driver (`dsound`/`wasapi`/`waveout`):

```bash
.venv/Scripts/python.exe tools/audio_probe.py dsound
```

## Run (Phase 1 — vision spine)

Tracked hands on screen with live latency instrumentation. No audio yet.

```bash
.venv/Scripts/python.exe src/main.py
```

Press `q` or `ESC` to quit. Useful flags: `--headless -s 20` (benchmark, prints a
pass/fail verdict against the Phase 1 exit criteria), `-c 1` (camera index),
`--no-gpu`, `--hands 2`.

Unit tests for the camera-independent logic:

```bash
.venv/Scripts/python.exe tests/test_phase1.py
```

### Measured on this machine

| Stage | p50 | p95 |
|---|---|---|
| capture → submit | 0.8 ms | 1.5 ms |
| submit → callback (inference) | 16.8 ms | 17.4 ms |
| callback → render | 1.5 ms | 2.3 ms |
| **total** | **19.2 ms** | **20.4 ms** |

Sustained 30.2 fps inference (camera-capped), CPU delegate, ~4.7 s warmup.

## Notes / gotchas

- **Audio driver:** this build supports `dsound` (default, verified), `wasapi`, `waveout`.
  The bundled `sdl3` audio driver is **not** compiled in.
- **MIDI noise:** pyfluidsynth's `Synth.start()` also spins up a MIDI-*input* driver
  (winmidi) that errors harmlessly when no MIDI keyboard is attached. Start audio only via
  `new_fluid_audio_driver(fs.settings, fs.synth)` to keep logs clean (Phase 4 will do this).
- **OpenCV 5 / numpy 2:** installed versions are opencv `5.0.0` and numpy `2.4.6` — newer
  than the plan assumed. Watch for API drift in Phase 1 (e.g. `cv2.VideoWriter.fourcc`).
- **System encoding is GBK (cp936):** Python's `open()` defaults to the locale codec, not
  UTF-8. **All text file I/O in code must pass `encoding="utf-8"`** (config, calibration
  JSON, recorded sessions), and config files are kept ASCII-only as a safety net.
- **The webcam is hard-capped at 30 fps and offers YUY2 only** — MJPG is unavailable at
  every resolution/backend tried (`tools/camera_format_probe.py`), so the plan's MJPG tip
  is a no-op on this hardware. **Consequence for Phase 3:** the plan's `confirm=3` costs
  100 ms here, not 50 ms. Specify confirmation windows in **milliseconds**, not frames,
  so they stay framerate-independent.
- **Never re-draw when nothing changed.** An earlier version of the main loop spun at
  ~2400 fps re-rendering identical frames; it starved the inference thread and inflated
  measured p95 from 20 ms to 52 ms. The loop now sleeps unless a new frame or new result
  arrived, and stage timings are recorded once per *result* (recording per iteration
  measures result age, not render cost).
