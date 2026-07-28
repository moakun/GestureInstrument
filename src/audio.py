"""Audio backend: NoteOn/NoteOff -> sound, polyphonic, with real release tails.

FluidSynth owns its own realtime thread; `note_on`/`note_off` are non-blocking, so the CV
loop never waits on audio and audio never runs in the CV loop (the plan's hard rule).

Two things this does that the plan's 4.1 sketch does not, both learned the hard way:

* **Start audio only.** `pyfluidsynth.Synth.start()` also creates a MIDI *input* driver,
  which errors noisily ("not enough MIDI in devices found") when no MIDI keyboard is
  attached. Creating the audio driver directly avoids it entirely.
* **Discover the DLL first.** On Windows FluidSynth is a vendored native library;
  `fsbootstrap` must run before `import fluidsynth`.

Instrument semantics differ and are respected per-instrument (plan 4.1):
piano note-off matters (damper), while harp and plucked guitar decay naturally — sending
note-off there cuts the sample dead and sounds wrong, so those are left to ring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fsbootstrap import ensure_fluidsynth_on_path

ensure_fluidsynth_on_path()
import fluidsynth  # noqa: E402  (must follow the DLL bootstrap)

# General MIDI program numbers in FluidR3_GM.sf2.
GM_PROGRAMS = {"piano": 0, "guitar": 24, "harp": 46}   # 24 = nylon guitar, 25 = steel
# True = send note-off on release (damper). False = let the sample ring.
DEFAULT_SUSTAIN = {"piano": True, "guitar": False, "harp": False}

PERIOD_SIZE_FALLBACKS = (128, 256, None)   # None = leave FluidSynth's default


class Backend(Protocol):
    """The one interface the mapping layer talks to."""

    def note_on(self, instrument: str, pitch: int, velocity: int = 100) -> None: ...
    def note_off(self, instrument: str, pitch: int) -> None: ...
    def all_off(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class SynthInfo:
    driver: str | None
    period_size: int | None
    samplerate: int
    realtime: bool
    instruments: tuple[str, ...]


class Synth:
    """FluidSynth backend, one MIDI channel per instrument.

    Set ``realtime=False`` to skip the audio driver and render with :meth:`render`
    instead — that is what makes the output objectively measurable (and testable on a
    machine with no audio device).
    """

    def __init__(self, sf2: str, instruments: dict[str, int] | None = None,
                 sustain: dict[str, bool] | None = None, driver: str | None = "dsound",
                 samplerate: int = 48000, gain: float = 0.6,
                 period_size: int | None = 128, periods: int = 2,
                 realtime: bool = True) -> None:
        self.programs = dict(instruments or GM_PROGRAMS)
        self.sustain = dict(sustain or DEFAULT_SUSTAIN)
        self.fs = fluidsynth.Synth(samplerate=samplerate, gain=gain)
        self._driver_handle = None
        self._period_size: int | None = None
        self._driver_name: str | None = None

        sfid = self.fs.sfload(sf2)
        if sfid == -1:
            raise RuntimeError(f"Could not load SoundFont: {sf2}")
        self.sfid = sfid

        self.channels: dict[str, int] = {}
        for ch, (name, program) in enumerate(sorted(self.programs.items())):
            self.fs.program_select(ch, sfid, 0, program)
            self.channels[name] = ch

        self._sounding: set[tuple[int, int]] = set()
        if realtime:
            self._start_audio(driver, period_size, periods)
        self.realtime = realtime and self._driver_handle is not None

    # -- setup ---------------------------------------------------------------
    def _start_audio(self, driver: str | None, period_size: int | None,
                     periods: int) -> None:
        """Create the audio driver, stepping period-size up if the small one fails.

        128 frames fails on some configurations; the plan's advice is 128 -> 256 ->
        default, which is what this does.
        """
        if driver:
            self.fs.setting("audio.driver", driver)
        self.fs.setting("audio.periods", periods)
        sizes = ([period_size] + [s for s in PERIOD_SIZE_FALLBACKS if s != period_size]
                 if period_size else list(PERIOD_SIZE_FALLBACKS))
        for size in sizes:
            if size is not None:
                self.fs.setting("audio.period-size", size)   # must precede driver creation
            try:
                handle = fluidsynth.new_fluid_audio_driver(self.fs.settings, self.fs.synth)
            except Exception:
                handle = None
            if handle:
                self._driver_handle = handle
                self._period_size = size
                self._driver_name = driver or str(self.fs.get_setting("audio.driver"))
                return

    @property
    def info(self) -> SynthInfo:
        return SynthInfo(self._driver_name, self._period_size,
                         int(self.fs.get_setting("synth.sample-rate") or 48000),
                         self.realtime, tuple(sorted(self.programs)))

    def sustains(self, instrument: str) -> bool:
        """Whether note-off should be sent for this instrument."""
        return self.sustain.get(instrument, True)

    # -- playing -------------------------------------------------------------
    def note_on(self, instrument: str, pitch: int, velocity: int = 100) -> None:
        ch = self.channels.get(instrument)
        if ch is None:
            raise KeyError(f"unknown instrument {instrument!r}; have {sorted(self.channels)}")
        pitch = int(np.clip(pitch, 0, 127))
        velocity = int(np.clip(velocity, 1, 127))
        if (ch, pitch) in self._sounding:
            self.fs.noteoff(ch, pitch)          # retrigger cleanly; a replucked string stops
        self.fs.noteon(ch, pitch, velocity)
        self._sounding.add((ch, pitch))

    def note_off(self, instrument: str, pitch: int, force: bool = False) -> None:
        """Release a note. No-op for ringing instruments unless ``force``.

        Harp and plucked guitar decay on their own; cutting them with a note-off is the
        difference between a pluck and a click.
        """
        if not force and not self.sustains(instrument):
            return
        ch = self.channels.get(instrument)
        if ch is None:
            return
        pitch = int(np.clip(pitch, 0, 127))
        self._sounding.discard((ch, pitch))
        self.fs.noteoff(ch, pitch)

    def all_off(self) -> None:
        """Panic: silence everything, ringing instruments included."""
        for ch, pitch in list(self._sounding):
            self.fs.noteoff(ch, pitch)
        self._sounding.clear()

    @property
    def sounding(self) -> int:
        return len(self._sounding)

    # -- offline ---------------------------------------------------------------
    def render(self, n_frames: int) -> np.ndarray:
        """Render ``n_frames`` stereo frames offline as int16 (N, 2).

        Only meaningful when constructed with ``realtime=False`` — with a live driver the
        audio thread is already consuming the synth.
        """
        return np.asarray(self.fs.get_samples(n_frames), dtype=np.int16).reshape(-1, 2)

    # -- teardown --------------------------------------------------------------
    def close(self) -> None:
        try:
            self.all_off()
        finally:
            if self._driver_handle:
                fluidsynth.delete_fluid_audio_driver(self._driver_handle)
                self._driver_handle = None
            self.fs.delete()

    def __enter__(self) -> "Synth":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullSynth:
    """No-op backend that records calls. Lets the pipeline be tested with no audio device."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int, int]] = []

    def note_on(self, instrument: str, pitch: int, velocity: int = 100) -> None:
        self.events.append(("on", instrument, pitch, velocity))

    def note_off(self, instrument: str, pitch: int, force: bool = False) -> None:
        self.events.append(("off", instrument, pitch, 0))

    def all_off(self) -> None:
        self.events.append(("all_off", "", 0, 0))

    def close(self) -> None:
        pass

    def sustains(self, instrument: str) -> bool:
        return DEFAULT_SUSTAIN.get(instrument, True)

    def __enter__(self) -> "NullSynth":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
