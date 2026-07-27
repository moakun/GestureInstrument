# Gesture Instrument — Hand-Tracked Harp / Guitar / Piano

**Status:** planning
**Stack:** Python 3.11 · OpenCV · MediaPipe Tasks (HandLandmarker) · FluidSynth (or sounddevice)
**Goal:** Two hands in front of a webcam. Finger poses select instrument + pitch. A discrete trigger fires the note.

---

## 0. Read this before you write any code

Three things about the original idea will bite you. Better to know now than after 400 lines.

### 0.1 A held pose is a *state*, not an *event*

"Show 3 fingers → play C" has no defined moment of attack. If you naively play a note whenever the pose is 3, you emit a note every frame — 30–60 notes/second. If you only play on *change*, you hit the second problem.

### 0.2 Transitional poses fire phantom notes

Going from `2` to `5`, your hand physically passes through `3` and `4`. The classifier is doing its job — it correctly reports 3 and 4 for a few frames each. You get a machine-gun arpeggio you never asked for. This is *the* defining problem of this project, and no amount of classifier accuracy fixes it, because the intermediate readings are not errors.

**Fix:** decouple **selection** from **triggering**.
- Selection (which instrument, which note) = the finger pose. Sampled continuously, never triggers anything.
- Trigger (note-on) = a separate, impulsive, unambiguous action.

Good triggers, in order of reliability:

| Trigger | Detection | Notes |
|---|---|---|
| **Pinch** (thumb tip ↔ index tip) | normalized distance crosses a threshold | Most reliable. Binary, fast, no ambiguity. |
| **Pluck** (fast downward wrist motion) | wrist velocity crosses threshold + direction | Feels the most musical. Needs velocity smoothing. |
| **Palm flip** (palm normal crosses camera plane) | sign of palm normal z-component | Novel, but slow and awkward in practice. |

Start with pinch. Add pluck in Phase 8 if you want expressiveness (pluck velocity → MIDI velocity is a genuinely nice touch).

### 0.3 The mapping is musically thin as specified

6 states per hand (0–5 fingers). If one hand picks the instrument and the other picks the note, you get **3 instruments × 5 notes**. Five notes is not an instrument, it's a doorbell.

Three mapping architectures, pick one in Phase 0:

| Mode | Left hand | Right hand | Range | Verdict |
|---|---|---|---|---|
| **A — literal (your original)** | instrument (1–3) | scale degree (1–5) | 5 notes | Fastest to build. Gets boring in ~10 minutes. Fine as a Phase 3 milestone. |
| **B — combinatorial** | instrument (1–3) + octave (via 4/5) | scale degree (1–5) + pinch triggers | ~15–20 notes | Good sweet spot. Recommended default. |
| **C — hybrid continuous** | instrument + octave (pose) | **hand height → pitch** (continuous), pinch = note-on | full scale, glissando | Most fun, most musical. Dodges finger classification for pitch entirely. |

Mode C is a real recommendation, not a stretch goal. Vertical hand position is high-resolution, low-latency, robust, and requires zero classification — you quantize the y-coordinate into scale degrees. It sidesteps your hardest technical problem. Consider building B and C, not A.

### 0.4 Latency budget — set expectations now

| Stage | Realistic cost |
|---|---|
| Camera exposure + USB transfer | 15–35 ms (frame period at 30 fps = 33 ms) |
| MediaPipe HandLandmarker, 2 hands, CPU, `lite` model | 6–15 ms |
| Gesture logic + mapping | < 1 ms |
| **Debounce / confirmation window** | **33–100 ms ← dominant, and tunable** |
| Audio buffer (128 frames @ 48 kHz, double-buffered) | ~5 ms |
| **End-to-end** | **~70–150 ms** |

For reference: a live musician notices ~20 ms, and above ~100 ms the instrument feels detached from the gesture. You will land in "playable toy," not "playable instrument." That's fine for the stated goal — but it means **every millisecond of debounce is a direct tax on feel**, and it's why the pinch trigger matters: a pinch is unambiguous in 1 frame, a finger count needs 3.

---

## 1. Architecture

```
┌──────────────┐   frame (ring size 1, drop stale)
│ Capture thd  │──────────────┐
│ OpenCV, MJPG │              ▼
└──────────────┘   ┌────────────────────────────┐
                   │ HandLandmarker LIVE_STREAM │  async callback
                   │ (own internal thread)      │
                   └────────────┬───────────────┘
                                ▼  21×3 landmarks × 2 hands
                   ┌────────────────────────────┐
                   │ Feature layer              │  palm-normalized,
                   │ curl cosines, pinch dist,  │  rotation-invariant
                   │ wrist velocity, hand y     │
                   └────────────┬───────────────┘
                                ▼
                   ┌────────────────────────────┐
                   │ State machine              │  Schmitt triggers,
                   │ selection latch + trigger  │  motion gate, debounce
                   │ edge detector              │
                   └────────────┬───────────────┘
                                ▼  NoteOn(ch,pitch,vel) / NoteOff
                   ┌────────────────────────────┐      ┌───────────┐
                   │ Mapping (config-driven)    │─────▶│ Synth     │
                   └────────────────────────────┘      │ FluidSynth│
                                                       └───────────┘
                   ┌────────────────────────────┐
                   │ HUD overlay (main thread)  │  never blocks audio
                   └────────────────────────────┘
```

**Hard rule:** audio never runs in the CV loop, and the CV loop never blocks on audio. FluidSynth owns its own realtime thread; you only call `noteon`/`noteoff`, which are non-blocking. If you build the sounddevice mixer instead, the callback touches nothing but pre-allocated numpy arrays and an atomic voice table — no allocation, no locks, no logging.

### Repo layout

```
gesture-instrument/
├─ assets/
│  ├─ hand_landmarker.task
│  └─ FluidR3_GM.sf2
├─ src/
│  ├─ capture.py       # threaded camera, latest-frame-wins
│  ├─ landmarks.py     # MediaPipe wrapper, handedness fix
│  ├─ features.py      # pure functions, numpy only, unit-testable
│  ├─ statemachine.py  # latches, Schmitt triggers, edge detection
│  ├─ mapping.py       # config → (channel, midi_note, velocity)
│  ├─ audio.py         # FluidSynth / sounddevice backend behind one interface
│  ├─ hud.py
│  └─ main.py
├─ config/mappings.yaml
├─ tests/              # replay recorded landmark sequences, assert note events
└─ tools/
   ├─ record_session.py   # dump landmarks+timestamps to .npz
   └─ measure_latency.py
```

`features.py` and `statemachine.py` being pure functions over numpy arrays is the single best architectural decision here — it means you can record 30 seconds of landmarks once and then iterate on thresholds a hundred times without ever touching a camera.

---

## Phase 0 — Decisions & environment (½ day)

**Deliverable:** working env, models downloaded, one decision written down per row of the table above.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install "mediapipe>=0.10.14" opencv-python numpy sounddevice pyfluidsynth pyyaml
# model
curl -L -o assets/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

FluidSynth is a **native** library — `pip install pyfluidsynth` only gives you the bindings.
- macOS: `brew install fluid-synth`
- Debian/Ubuntu: `sudo apt install fluidsynth libfluidsynth3`
- Windows: download the release, put the DLL on `PATH`

Grab any GM SoundFont (`FluidR3_GM.sf2`, ~140 MB) — it has all three instruments you need.

**Decisions locked (2026-07-24):**
- [x] Mapping mode: **C — continuous pitch** (right-hand height → pitch; pinch = note-on). Sidesteps finger-count classification for pitch. Mode B deferred as a stretch.
- [x] Trigger: **pinch** (unambiguous in a single frame → lowest latency). Pluck deferred to Phase 8.
- [x] Audio backend: **FluidSynth** (installed & verified; `dsound` driver on Windows).
- [x] Scale: **C major pentatonic** (`[0, 2, 4, 7, 9]`, root C4 = 60) — with pentatonic, *wrong notes still sound fine*.

**Exit criteria:** `python -c "import mediapipe, fluidsynth, cv2, sounddevice"` succeeds and a test `noteon` makes a sound.

> **Phase 0 status: DONE.** Env, assets, and audio verified via `tools/check_env.py` →
> `PHASE 0 ENVIRONMENT: OK`. On Windows, FluidSynth is a vendored native lib (no
> winget/choco package), wired up by `src/fsbootstrap.py`; audio driver is `dsound`.
> Config for the locked decisions: `config/mappings.yaml`. See `README.md` for setup.

---

## Phase 1 — Vision spine (1 day)

**Goal:** landmarks on screen, instrumented, at stable framerate. No music yet.

### 1.1 Capture — the cheap latency wins

```python
import cv2, threading, time

class Camera:
    """Latest-frame-wins capture. Decouples grab latency from processing."""
    def __init__(self, index=0, width=640, height=480, fps=60):
        api = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY  # V4L2 on Linux
        self.cap = cv2.VideoCapture(index, api)
        # MJPG matters: raw YUYV saturates USB2 bandwidth and silently caps you at 10-15 fps
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # do not accumulate stale frames
        self._frame, self._ts, self._lock = None, 0.0, threading.Lock()
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005); continue
            with self._lock:
                self._frame, self._ts = f, time.perf_counter()

    def read(self):
        with self._lock:
            return (None, 0.0) if self._frame is None else (self._frame, self._ts)

    def close(self):
        self._run = False; time.sleep(0.05); self.cap.release()
```

`CAP_PROP_BUFFERSIZE = 1` plus a dedicated grab thread is worth more latency than any model optimization you'll do. Without it, OpenCV queues frames and you end up reacting to a hand position from 150 ms ago.

**640×480 is deliberate.** MediaPipe downscales internally anyway; 1080p capture buys you nothing but USB bandwidth and memcpy.

### 1.2 Landmarker in LIVE_STREAM mode

```python
import mediapipe as mp
from mediapipe.tasks import python as mpy
from mediapipe.tasks.python import vision

class Landmarker:
    def __init__(self, model="assets/hand_landmarker.task", on_result=None):
        try:
            base = mpy.BaseOptions(model_asset_path=model,
                                   delegate=mpy.BaseOptions.Delegate.GPU)
            self.lm = self._make(base, on_result)
        except Exception:                       # GPU delegate is unavailable on many desktops
            base = mpy.BaseOptions(model_asset_path=model)
            self.lm = self._make(base, on_result)

    @staticmethod
    def _make(base, cb):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base,
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                result_callback=cb,
            ))

    def submit(self, bgr, ts_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.lm.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
```

Gotchas that will cost you an hour each if you don't know them:
- `detect_async` timestamps **must be strictly increasing integers (ms)**. Reuse of a timestamp raises. Derive from a monotonic clock, not `time.time()`.
- The result callback runs on **MediaPipe's thread**. Do not do OpenCV drawing there. Hand results to the main thread via a 1-slot slot, same pattern as the camera.
- Use the `float16` model. The `full` variant costs ~2× inference for accuracy you won't perceive.

### 1.3 Handedness is mirrored — fix it once, at the boundary

You will flip the frame for a mirror view (`cv2.flip(frame, 1)`) because unmirrored video is unusable for gestures. MediaPipe's `handedness` label refers to the *image*, so after flipping, its "Left" is your right hand.

Decide the convention **once**, in `landmarks.py`, and never think about it again:

```python
MIRROR = True
def true_handedness(label: str) -> str:
    return {"Left": "Right", "Right": "Left"}[label] if MIRROR else label
```

### 1.4 Instrumentation from day one

Log a rolling p50/p95 of: capture→submit, submit→callback, callback→render, and total. Print on a HUD. If you only add this in Phase 7 you will have already made three untraceable performance mistakes.

**Exit criteria:** two hands tracked at ≥30 fps sustained; total pipeline p95 < 40 ms excluding debounce; handedness labels correct when you wave each hand.

> **Phase 1 status: DONE.** Sustained **30.2 fps** (camera-capped), pipeline **p95 20.4 ms**
> (inference ~17 ms, CPU delegate), ~4.7 s warmup; both hands and handedness confirmed
> visually. Verify with `python src/main.py --headless -s 20`.
>
> Two hardware/design findings:
> - **The webcam is hard-capped at 30 fps and offers YUY2 only** — MJPG is unavailable at
>   every resolution and backend (`tools/camera_format_probe.py`), so 1.1's MJPG tip is a
>   no-op here. **Consequence for Phase 3:** `confirm=3` costs 100 ms at 30 fps, not the
>   50 ms this plan assumes at 60 fps — so specify confirmation windows in **milliseconds**.
> - **Never re-render when nothing changed.** A first cut of the loop spun at ~2400 fps
>   redrawing identical frames; it starved the inference thread and inflated measured p95
>   from 20 ms to 52 ms. Record stage timings once per *result*, not per iteration.

---

## Phase 2 — Feature extraction (1 day)

**Goal:** pure functions, `(21,3) float32 → feature dict`. Zero state. Fully unit-tested against recorded clips.

### 2.1 Normalize first

Never threshold on raw coordinates. Two scale-invariant references:

```python
import numpy as np

def palm_scale(lm):                     # wrist → middle MCP
    return float(np.linalg.norm(lm[9] - lm[0])) + 1e-6
```

### 2.2 Finger extension — do NOT use "tip is above PIP"

The `tip.y < pip.y` heuristic every tutorial uses breaks the instant you tilt your hand, and you *will* tilt your hand while playing. Use the joint angle instead — it's rotation-invariant by construction:

```python
FINGERS = {"index": (5, 6, 8), "middle": (9, 10, 12),
           "ring": (13, 14, 16), "pinky": (17, 18, 20)}   # (mcp, pip, tip)

def curl_cos(lm, mcp, pip, tip):
    """cos of the angle at PIP. ~-1.0 = straight, ~0..+1 = folded."""
    a = lm[mcp] - lm[pip]
    b = lm[tip] - lm[pip]
    a /= np.linalg.norm(a) + 1e-9
    b /= np.linalg.norm(b) + 1e-9
    return float(a @ b)

def finger_curls(lm):
    return {n: curl_cos(lm, *idx) for n, idx in FINGERS.items()}
```

**Thumb needs its own rule.** Its PIP angle barely changes between open and closed; what changes is abduction. Use distance from thumb tip to the index MCP, normalized:

```python
def thumb_open(lm, s=None):
    s = s or palm_scale(lm)
    return float(np.linalg.norm(lm[4] - lm[5])) / s      # ~>0.55 open, <0.40 closed
```

Expect the thumb to be your worst-performing digit regardless. Design the mapping so thumb state is never load-bearing — e.g. counts 1–4 use fingers only, 5 = all open.

### 2.3 Pinch (the trigger)

```python
def pinch_ratio(lm, s=None):
    s = s or palm_scale(lm)
    return float(np.linalg.norm(lm[4] - lm[8])) / s      # <0.25 = pinched
```

### 2.4 Motion energy (the phantom-note killer)

```python
def motion_energy(lm, prev_lm, dt, s=None):
    if prev_lm is None or dt <= 0: return 0.0
    s = s or palm_scale(lm)
    return float(np.linalg.norm(lm - prev_lm, axis=1).mean()) / (s * dt)
```

This is the key insight of the whole project: **while the hand is moving fast, its pose classification is meaningless.** Gate all selection changes on `motion_energy < threshold`. Transitional poses occur exactly when motion is high, so this filter removes them almost for free — and unlike a longer debounce window, it costs you *zero* latency when the hand is already still.

### 2.5 Hand height (Mode C)

```python
def hand_y(lm):
    return float(lm[[0, 5, 9, 13, 17]][:, 1].mean())    # palm centroid, normalized 0..1
```

**Exit criteria:** a unit test replays a recorded `.npz` and asserts the extracted finger-count sequence matches hand-labelled ground truth ≥95% on *static* frames.

> **Phase 2 status: code complete, awaiting a labelled recording.** `src/features.py` is
> pure numpy; 51 synthetic-geometry tests pass (`tests/test_features.py`), covering the
> invariances the design rests on — curls unchanged under 3D rotation, ratios unchanged
> with camera distance, thumb never load-bearing for counts 0–3.
>
> **Use metric world landmarks for shape, not normalized ones.** MediaPipe also returns
> `hand_world_landmarks` (metres, isotropic). Normalized image coords are *anisotropic* —
> x spans the frame width, y the height — so on this 4:3 camera curl cosines drift by
> **0.277** as the hand rotates in-plane. The plan's hysteresis band is only 0.35 wide
> (`-0.60`…`-0.25`), so that distortion is nearly the whole band: enough to flip a
> finger's state purely from tilting the hand, which is exactly what 2.2 uses angles to
> avoid. `features.isotropic()` corrects normalized coords (drift → 1.3e-06) for clips
> where world landmarks aren't available.
>
> Tooling for the exit criteria: `tools/record_session.py` (guided, labelled 0–5 capture),
> `tests/test_replay.py` (scores static frames), `tools/tune_thresholds.py` (offline
> threshold grid-search — the cheap half of 7.1 calibration).
>
> **Phase 2 status: DONE — 100% on 456 labelled static frames** (target ≥95%).
>
> **`thumb_open` as specified in 2.2 does not work, and had to be replaced.** On a real
> clip it separated a tucked thumb (median 0.692) from an abducted one (0.734) by just
> **0.027** — inside frame noise. Every count-4 was read as a 5; accuracy 83.3%. With the
> fingers extended, the thumb tip simply doesn't change its distance to the index MCP.
> A threshold grid-search "fixed" it at 0.70, but that value sits inside the 0.04 sliver
> and is fitted to one clip, not a real signal.
>
> The replacement, `features.thumb_abduction()`, measures **direction instead of
> distance**: the cosine between the thumb axis (CMC→tip) and the palm across-axis (index
> MCP→pinky MCP). A tucked thumb lies across the palm pointing at the pinky; an abducted
> one points away, so the sign flips. Margin **0.863** vs 0.027 — and it needs no
> `palm_scale` division, so it doesn't inherit that estimate's error. This is escalation
> step 1 of 7.3 ("better features"), and it was indeed enough.
>
> Both thresholds are now **centred in their measured 100%-accuracy plateau** rather than
> set to a fitted optimum: `curl` 100% across −0.78…−0.57 → **−0.68**; `thumb_abduction`
> 100% across −0.22…+0.90 → **+0.45**. 2.2's advice to keep the thumb non-load-bearing
> proved its worth: with the thumb threshold badly wrong, counts 0–3 still scored 100%
> and only the 4↔5 boundary broke.

---

## Phase 3 — State machine (1 day)

**Goal:** turn noisy per-frame features into clean, discrete, correctly-timed events.

### 3.1 Schmitt trigger per finger

One threshold oscillates at the boundary. Two thresholds don't.

```python
class Schmitt:
    __slots__ = ("lo", "hi", "state")
    def __init__(self, lo, hi, init=False):
        self.lo, self.hi, self.state = lo, hi, init
    def __call__(self, x):
        if self.state and x > self.hi:   self.state = False
        elif not self.state and x < self.lo: self.state = True
        return self.state
```

For curls: extended when `cos < -0.60`, retracted when `cos > -0.25`. Tune per-user in Phase 7.

### 3.2 Selection latch (debounce + motion gate)

```python
class Latch:
    """Commits a new value only after N stable frames AND while the hand is still."""
    def __init__(self, confirm=3, max_motion=1.2):
        self.confirm, self.max_motion = confirm, max_motion
        self.stable = self.cand = None
        self.n = 0
    def update(self, obs, motion):
        if motion > self.max_motion:
            self.n = 0; self.cand = None
            return None
        if obs == self.cand: self.n += 1
        else: self.cand, self.n = obs, 1
        if self.n >= self.confirm and self.cand != self.stable:
            self.stable = self.cand
            return self.stable          # transition committed
        return None
```

`confirm=3` at 60 fps is 50 ms — acceptable *because selection doesn't need to be fast*. Only the trigger does.

### 3.3 Trigger edge detector

```python
class PinchTrigger:
    def __init__(self, close=0.25, open_=0.35, refractory_ms=80):
        self.s = Schmitt(close, open_)
        self.refr = refractory_ms / 1000.0
        self.last = -1e9
    def update(self, ratio, t):
        was = self.s.state
        now = self.s(ratio)
        if now and not was and (t - self.last) > self.refr:
            self.last = t
            return "on"
        if was and not now:
            return "off"
        return None
```

`confirm=1` here — a single frame. That's the whole point of the pinch: it's unambiguous, so it needs no confirmation window, so it costs no latency. The refractory period (80 ms ≈ 750 BPM) prevents double-triggers from landmark jitter without slowing real playing.

**Exit criteria:** replay a recorded clip where you deliberately sweep 0→5→0 fingers ten times while pinching at known moments. Assert exactly 10 note-ons, zero phantoms.

---

## Phase 4 — Audio engine (1 day)

**Goal:** `NoteOn(instrument, pitch, velocity)` → sound, in under 10 ms, polyphonic, with a real release tail.

### 4.1 FluidSynth (recommended)

Real sampled instruments, free polyphony, proper envelopes, sustain pedal, pitch bend. You are not going to beat this with hand-rolled sample playback.

```python
import fluidsynth

GM = {"piano": 0, "guitar": 24, "harp": 46}   # nylon guitar; 25 = steel

class Synth:
    def __init__(self, sf2="assets/FluidR3_GM.sf2", driver=None, sr=48000):
        self.fs = fluidsynth.Synth(samplerate=sr, gain=0.6)
        self.fs.setting("audio.period-size", 128)   # must precede start()
        self.fs.setting("audio.periods", 2)
        self.fs.start(driver=driver)                # coreaudio | alsa/pulseaudio | dsound
        sfid = self.fs.sfload(sf2)
        self.ch = {}
        for i, (name, prog) in enumerate(GM.items()):
            self.fs.program_select(i, sfid, 0, prog)
            self.ch[name] = i
        self._sounding = set()

    def note_on(self, inst, pitch, vel=100):
        c = self.ch[inst]
        if (c, pitch) in self._sounding:            # retrigger cleanly
            self.fs.noteoff(c, pitch)
        self.fs.noteon(c, pitch, vel)
        self._sounding.add((c, pitch))

    def note_off(self, inst, pitch):
        c = self.ch[inst]
        self._sounding.discard((c, pitch))
        self.fs.noteoff(c, pitch)

    def all_off(self):
        for c, p in list(self._sounding): self.fs.noteoff(c, p)
        self._sounding.clear()
```

**Instrument semantics differ and you should respect them:**
- **Piano** — note-off matters (damper). Map release to pinch-release.
- **Harp / plucked guitar** — the sample decays naturally. Sending note-off cuts it dead and sounds wrong. Fire note-on and *never* send note-off; let it ring. Track this per-instrument in your config.

Wrap `period-size` in a fallback: 128 frames fails on some ALSA/Pulse configs. Try 128 → 256 → default.

### 4.2 sounddevice mixer (alternative — only if you want per-sample control)

```python
import numpy as np, sounddevice as sd

SR, BLOCK, MAXV = 48000, 128, 32

class Mixer:
    def __init__(self, samples: dict[str, np.ndarray]):   # float32 mono, pre-loaded
        self.samples = samples
        self.buf = np.zeros((MAXV, 2), np.float32)        # (pos, gain) per voice slot
        self.src = [None] * MAXV
        self.stream = sd.OutputStream(
            samplerate=SR, blocksize=BLOCK, channels=1, dtype="float32",
            latency="low", callback=self._cb)
        self.stream.start()

    def _cb(self, out, frames, t, status):
        out[:] = 0.0                                       # no allocation in the callback
        for i in range(MAXV):
            s = self.src[i]
            if s is None: continue
            p = int(self.buf[i, 0]); g = self.buf[i, 1]
            n = min(frames, len(s) - p)
            if n <= 0:
                self.src[i] = None; continue
            out[:n, 0] += s[p:p + n] * g
            self.buf[i, 0] = p + n
        np.clip(out, -1.0, 1.0, out)

    def play(self, name, gain=0.7):
        s = self.samples[name]
        for i in range(MAXV):
            if self.src[i] is None:
                self.buf[i] = (0.0, gain); self.src[i] = s; return
```

Rules for the callback, non-negotiable: no `print`, no allocation, no locks, no exceptions. Pre-load every sample as `float32` at the stream samplerate at startup — resampling in the callback will glitch.

**Exit criteria:** hammer 20 note-ons in 2 seconds. No dropouts, no clipping, measured `note_on()` call cost < 1 ms.

---

## Phase 5 — Mapping layer (½ day)

**Goal:** all musical decisions in a config file, none in code. You will retune this fifty times; recompiling your intuition is not a workflow.

```yaml
# config/mappings.yaml
scale:  [0, 2, 4, 7, 9]        # C major pentatonic, semitone offsets
root:   60                     # C4
mode:   B

left_hand:                     # selector: instrument + octave
  0: {instrument: piano,  octave: 0}
  1: {instrument: piano,  octave: 0}
  2: {instrument: guitar, octave: 0}
  3: {instrument: harp,   octave: 0}
  4: {instrument: harp,   octave: 1}
  5: {instrument: piano,  octave: 1}

right_hand:                    # selector: scale degree
  1: 0
  2: 1
  3: 2
  4: 3
  5: 4

trigger: pinch                 # pinch | pluck
trigger_hand: right

sustain:                       # per-instrument note-off behaviour
  piano:  true                 # send note-off on release
  guitar: false                # let it ring
  harp:   false
```

```python
def resolve(cfg, left_count, right_count):
    sel = cfg["left_hand"].get(left_count)
    deg = cfg["right_hand"].get(right_count)
    if sel is None or deg is None:
        return None
    pitch = cfg["root"] + 12 * sel["octave"] + cfg["scale"][deg]
    return sel["instrument"], pitch
```

Hot-reload the YAML on file change (`os.stat().st_mtime` poll, once a second). Being able to retune the scale without restarting the camera is worth the fifteen lines.

**Mode C variant:** replace `right_hand` count lookup with
`degree = int((1.0 - hand_y) * len(scale) * octaves)`, clamped. Add a small deadband at each boundary so a hand hovering between degrees doesn't warble.

**Exit criteria:** changing the scale in YAML changes what you hear, without a restart.

---

## Phase 6 — HUD & feedback (½ day)

Visual feedback isn't decoration here — it's how you debug a system whose output is a sound that already happened.

Draw on the mirrored frame:
- Skeleton per hand, colour-coded by handedness (after the mirror fix).
- Per-hand: raw count, latched count, and a confirmation progress bar (`n / confirm`).
- Pinch ratio as a bar with both Schmitt thresholds drawn as lines. You will tune thresholds by watching this.
- Motion energy bar with the gate threshold marked. When phantom notes happen, this tells you instantly whether the gate is too loose.
- Currently sounding notes, and the last event with its age in ms.
- p50/p95 latency per stage.

Keep the HUD on the main thread and skip rendering if the frame budget is tight — the instrument must never stutter because you're drawing a bar chart.

**Exit criteria:** you can diagnose a phantom note purely from the HUD without adding a print statement.

---

## Phase 7 — Tuning & robustness (1–2 days)

This is where the project actually becomes playable. Budget real time for it.

### 7.1 Calibration routine

A 30-second guided flow at startup: "open hand… close hand… pinch… release… hold each count 1–5." Record the feature distributions, set per-user Schmitt thresholds at the midpoint between cluster means, persist to `~/.gesture-instrument/calib.json`. Hand geometry varies more between people than you'd guess, and hard-coded thresholds are the #1 reason these demos fail on someone else's hands.

### 7.2 Failure modes and mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Notes fire while changing pose | transitional poses | motion gate (2.4) — tighten `max_motion` |
| Notes drop out mid-play | occlusion, hand leaves frame | on hand-lost, hold last selection 300 ms before resetting; never auto-fire on reacquisition |
| Thumb count unreliable | thumb kinematics | never make thumb load-bearing in the mapping |
| Backlit hands → no detection | camera auto-exposure | disable auto-exposure, fix it manually; sit facing the light |
| Framerate collapses after a minute | thermal throttle / other apps | check p95 log; drop to `lite` model or 30 fps cap |
| Left/right swap intermittently | hands crossing | MediaPipe handedness is unreliable when hands overlap — add a positional tiebreak (leftmost wrist x = left hand) |
| Audio crackle | buffer too small | step period-size 128 → 256 |

### 7.3 If the heuristic classifier still isn't good enough

Escalate, in this order:

1. **Better features.** Add MCP-joint angles and inter-fingertip distances. Usually enough.
2. **Tiny learned classifier.** Record ~200 labelled frames per class with `tools/record_session.py`. Normalize the 21 landmarks (translate to wrist, scale by palm, rotate so index-MCP→pinky-MCP is horizontal) → 63-dim vector → `sklearn` kNN or a 2-layer MLP. Inference cost is negligible next to the landmarker. This will beat your hand-tuned thresholds and takes an afternoon.
3. **MediaPipe Gesture Recognizer** with a custom-trained model. Heaviest option, only worth it if 1 and 2 both fail.

Don't start at 3. Heuristics that you can debug beat a black box you can't, especially when the failure mode is timing rather than accuracy.

**Exit criteria:** a friend who has never used it can play a recognizable melody within two minutes.

---

## Phase 8 — Stretch goals, roughly by value/effort

| Idea | Effort | Payoff |
|---|---|---|
| **Pluck velocity → MIDI velocity** | S | Huge. This is what makes it feel like an instrument instead of a keypad. |
| **Mode C continuous pitch** | S | Huge. Glissando on a harp is the single best demo moment available to you. |
| **Chord mode** — left hand pose = chord, one trigger fires 3–4 notes with 10 ms stagger (strum) | S | Very high. Guitar strum with staggered note-ons sounds startlingly real. |
| Sustain via off-hand fist | S | Medium |
| MIDI out (`python-rtmidi`) to a DAW instead of internal synth | S | High if you already use a DAW — gets you any instrument, any effects, for free |
| Loop recorder — quantized 4-bar loop, layer takes | M | High |
| Metronome + quantize-to-grid note timing | M | High — quantization *hides your latency*, which is the cheapest possible fix for the 0.4 problem |
| Reverb / delay send | S | Medium — a bit of reverb makes cheap samples sound much better |
| Two-player mode (4 hands, `num_hands=4`) | M | Fun, costs ~2× inference |
| Web port (MediaPipe.js + WebAudio) | L | Shareable, and WebAudio latency is honestly competitive |

**Quantize-to-grid deserves a second look.** If notes snap to a 16th-note grid at a fixed tempo, a 120 ms pipeline delay becomes musically invisible — you play slightly ahead, the grid catches you. It converts your hardest engineering constraint into a non-issue for the price of a metronome.

---

## Timeline

| Phase | Est. | Cumulative |
|---|---|---|
| 0 Decisions & env | 0.5 d | 0.5 |
| 1 Vision spine | 1 d | 1.5 |
| 2 Features | 1 d | 2.5 |
| 3 State machine | 1 d | 3.5 |
| 4 Audio | 1 d | 4.5 |
| 5 Mapping | 0.5 d | 5 |
| 6 HUD | 0.5 d | 5.5 |
| 7 Tuning | 1–2 d | 6.5–7.5 |

**First playable moment is end of Phase 4** (~4.5 days) — hard-code the mapping temporarily so you get sound out before building the config layer. Do not skip Phase 7; the difference between phases 1–6 and phase 7 is the difference between a screenshot and something you'd actually pick up again tomorrow.

---

## Definition of done

- Both hands tracked simultaneously at ≥30 fps sustained on your machine.
- Zero phantom notes across a 60-second freeform play session.
- End-to-end trigger→sound latency measured (240 fps phone video of a pinch + speaker cone, count frames) and documented.
- Scale, mapping, and instruments changeable without touching code.
- Someone else can play it after a 30-second calibration.
