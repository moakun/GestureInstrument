"""Turn noisy per-frame features into clean, discrete, correctly-timed events.

The one job here is to separate **selection** (which instrument, which note — sampled
continuously, never triggers anything) from **triggering** (an impulsive, unambiguous
note-on). Conflating them is what produces the machine-gun arpeggio when a hand passes
through 3 and 4 on its way from 2 to 5 (plan 0.1/0.2).

**Confirmation windows are in milliseconds, not frames.** The plan specifies
``confirm=3``, reasoning that 3 frames at 60 fps is 50 ms. This camera is hard-capped at
30 fps, where the same 3 frames cost 100 ms — a direct, doubled tax on feel. Time-based
windows behave identically whatever the framerate delivers, including when it dips.

Everything is state, so it is deliberately kept out of `features.py`, and every class
takes an explicit timestamp rather than calling the clock itself — that is what lets the
whole machine be replayed offline from a recording.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Selection commits only while the hand is still. Transitional poses occur exactly when
# motion is high, so this removes phantoms almost for free — and unlike a longer debounce
# window it costs *zero* latency when the hand is already still (plan 2.4).
MAX_SELECT_MOTION = 1.2
SELECT_CONFIRM_MS = 100.0        # 3 frame-intervals at 30 fps
PINCH_REFRACTORY_MS = 80.0       # ~750 BPM ceiling; stops jitter double-triggers
HAND_LOST_HOLD_MS = 300.0        # keep the last selection this long before resetting


class Schmitt:
    """Two-threshold hysteresis. One threshold oscillates at the boundary; two don't.

    Default (*falling*) polarity matches curl cosines: ``True`` once ``x`` drops below
    ``lo``, back to ``False`` only once it climbs above ``hi``. Set ``rising=True`` for
    signals where *larger* means active, such as thumb abduction.
    """

    __slots__ = ("lo", "hi", "state", "rising")

    def __init__(self, lo: float, hi: float, init: bool = False, rising: bool = False) -> None:
        if lo > hi:
            raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
        self.lo, self.hi, self.state, self.rising = lo, hi, init, rising

    def __call__(self, x: float) -> bool:
        if self.rising:
            if not self.state and x > self.hi:
                self.state = True
            elif self.state and x < self.lo:
                self.state = False
        else:
            if not self.state and x < self.lo:
                self.state = True
            elif self.state and x > self.hi:
                self.state = False
        return self.state

    def resync(self, x: float) -> bool:
        """Force state from ``x`` without hysteresis, and without reporting an edge.

        Used when a hand reappears: the band it re-enters in is unknown, and we must not
        manufacture a transition from it (plan 7.2 — never auto-fire on reacquisition).
        """
        self.state = (x > self.hi) if self.rising else (x < self.lo)
        return self.state


class SelectionLatch:
    """Commit a new value only after it has been stable for a while *and* the hand is still.

    Selection does not need to be fast — only the trigger does — so this window is the
    cheap place to buy robustness (plan 3.2).
    """

    def __init__(self, confirm_ms: float = SELECT_CONFIRM_MS,
                 max_motion: float = MAX_SELECT_MOTION, min_samples: int = 2) -> None:
        self.confirm_s = confirm_ms / 1000.0
        self.max_motion = max_motion
        self.min_samples = min_samples          # a lone stale frame must never commit
        self.stable: object | None = None
        self._cand: object | None = None
        self._since: float = 0.0
        self._n: int = 0

    def update(self, obs: object, motion: float, t: float) -> object | None:
        """Feed one frame. Returns the value if this frame *commits* a change, else None."""
        if motion > self.max_motion:
            self._cand, self._n = None, 0       # in transit: whatever we see is meaningless
            return None
        if obs != self._cand:
            self._cand, self._since, self._n = obs, t, 1
            return None
        self._n += 1
        if (self._n >= self.min_samples and (t - self._since) >= self.confirm_s
                and self._cand != self.stable):
            self.stable = self._cand
            return self.stable
        return None

    def reset(self) -> None:
        self.stable = self._cand = None
        self._n = 0

    @property
    def progress(self) -> float:
        """0..1 toward committing the current candidate — drawn as a HUD bar in Phase 6."""
        if self._cand is None or self._cand == self.stable or self.confirm_s <= 0:
            return 0.0
        return 1.0 if self._n >= self.min_samples and self._since == 0.0 else min(
            1.0, self._n / max(self.min_samples, 1))


class PinchTrigger:
    """Note-on/note-off edge detector on the pinch ratio.

    Deliberately **no confirmation window**: a pinch is unambiguous in a single frame, so
    it needs none, so it costs no latency. That is the whole reason the trigger is a pinch
    rather than a finger count (plan 0.2/3.3). The refractory period only suppresses
    landmark-jitter double-fires, and is short enough not to slow real playing.
    """

    def __init__(self, close: float = 0.25, open_: float = 0.35,
                 refractory_ms: float = PINCH_REFRACTORY_MS) -> None:
        self.schmitt = Schmitt(close, open_)     # falling: pinched when ratio < close
        self.refractory_s = refractory_ms / 1000.0
        self._last_on = float("-inf")
        self._suppressed = False

    @property
    def pinched(self) -> bool:
        return self.schmitt.state

    def update(self, ratio: float, t: float) -> str | None:
        """Returns 'on', 'off', or None."""
        was = self.schmitt.state
        now = self.schmitt(ratio)
        if now and not was:
            if self._suppressed:                 # came back mid-pinch; wait for a release
                return None
            if (t - self._last_on) > self.refractory_s:
                self._last_on = t
                return "on"
            return None
        if was and not now:
            if self._suppressed:
                self._suppressed = False         # released: edges are trustworthy again
                return None
            return "off"
        return None

    def resync(self, ratio: float) -> None:
        """Adopt the current pinch state silently (used when a hand reappears)."""
        self.schmitt.resync(ratio)
        self._suppressed = self.schmitt.state


class HysteresisQuantizer:
    """Continuous value -> discrete index, with a deadband so hovering doesn't warble.

    Mode C maps hand height to scale degree. Without a deadband, a hand resting exactly on
    a boundary flickers between two degrees many times a second (plan 5, Mode C variant).
    """

    def __init__(self, n: int, deadband: float = 0.06, lo: float = 0.0,
                 hi: float = 1.0) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.lo, self.hi = n, lo, hi
        self.deadband = deadband
        self.index = 0
        self._primed = False

    def __call__(self, x: float) -> int:
        span = (self.hi - self.lo) or 1.0
        pos = (min(max(x, self.lo), self.hi) - self.lo) / span   # 0..1
        raw = min(int(pos * self.n), self.n - 1)
        if not self._primed:
            self.index, self._primed = raw, True
            return self.index
        if raw != self.index:
            # Only move once past the boundary *plus* the deadband, in cell-width units.
            boundary = (self.index + 1) / self.n if raw > self.index else self.index / self.n
            margin = self.deadband / self.n
            if (raw > self.index and pos > boundary + margin) or \
               (raw < self.index and pos < boundary - margin):
                self.index = raw
        return self.index

    def reset(self) -> None:
        self._primed = False
        self.index = 0


@dataclass(frozen=True)
class Event:
    """Something the mapping layer should act on."""

    kind: str          # "select" | "trigger_on" | "trigger_off" | "hand_found" | "hand_lost"
    hand: str
    t: float
    value: object = None


@dataclass
class HandTracker:
    """Per-hand state: latched selection, pinch edges, and dropout handling.

    Feeding features in and getting events out is the whole interface, which keeps this
    replayable frame-by-frame from a recording with no camera or clock involved.
    """

    hand: str
    curl_lo: float = -0.68
    curl_hi: float = -0.25
    thumb_lo: float = 0.35
    thumb_hi: float = 0.55
    pinch_close: float = 0.25
    pinch_open: float = 0.35
    confirm_ms: float = SELECT_CONFIRM_MS
    max_motion: float = MAX_SELECT_MOTION
    refractory_ms: float = PINCH_REFRACTORY_MS
    lost_hold_ms: float = HAND_LOST_HOLD_MS

    fingers: dict = field(init=False, default_factory=dict)
    thumb: Schmitt = field(init=False)
    latch: SelectionLatch = field(init=False)
    trigger: PinchTrigger = field(init=False)
    present: bool = field(init=False, default=False)
    raw_count: int = field(init=False, default=0)
    _lost_at: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        from features import FINGER_NAMES
        self.fingers = {n: Schmitt(self.curl_lo, self.curl_hi) for n in FINGER_NAMES}
        self.thumb = Schmitt(self.thumb_lo, self.thumb_hi, rising=True)
        self.latch = SelectionLatch(self.confirm_ms, self.max_motion)
        self.trigger = PinchTrigger(self.pinch_close, self.pinch_open, self.refractory_ms)

    @property
    def count(self) -> int | None:
        """The committed finger count, or None if nothing has been confirmed yet."""
        return self.latch.stable            # type: ignore[return-value]

    def pose_count(self, curls: dict[str, float], thumb_abduction: float) -> int:
        """Hysteresis-filtered finger count. Thumb only ever promotes 4 -> 5 (plan 2.2)."""
        n = sum(self.fingers[name](c) for name, c in curls.items())
        return 5 if (n == 4 and self.thumb(thumb_abduction)) else n

    def update(self, f, t: float) -> list[Event]:
        """Advance one frame with this hand's `features.HandFeatures`."""
        events: list[Event] = []
        if not self.present:
            self.present = True
            # Adopt the observed pinch state silently so reappearing mid-pinch cannot
            # manufacture a note (plan 7.2).
            self.trigger.resync(f.pinch)
            for name, c in f.curls.items():
                self.fingers[name].resync(c)
            self.thumb.resync(f.thumb)
            events.append(Event("hand_found", self.hand, t))

        self.raw_count = self.pose_count(f.curls, f.thumb)
        committed = self.latch.update(self.raw_count, f.motion, t)
        if committed is not None:
            events.append(Event("select", self.hand, t, committed))

        # A pinch is the thumb meeting an *extended* index — not any two digits drifting
        # close. Without this gate a closed fist fires notes: on a real recorded hand a
        # fist measures pinch_ratio 0.263-0.302 against a 0.25 threshold, a margin of
        # 0.013. The index Schmitt supplies the hysteresis, so the partial index bend of a
        # genuine pinch does not drop the gate mid-gesture.
        gated = f.pinch if self.fingers["index"].state else max(f.pinch, self.pinch_open + 0.01)
        edge = self.trigger.update(gated, t)
        if edge == "on":
            events.append(Event("trigger_on", self.hand, t))
        elif edge == "off":
            events.append(Event("trigger_off", self.hand, t))
        return events

    def absent(self, t: float) -> list[Event]:
        """Advance one frame in which this hand was not detected."""
        if not self.present:
            return []
        self.present = False
        self._lost_at = t
        return [Event("hand_lost", self.hand, t)]

    def tick_absent(self, t: float) -> list[Event]:
        """Call while the hand stays missing; drops the selection after the hold window."""
        if self.present or self.latch.stable is None:
            return []
        if (t - self._lost_at) * 1000.0 >= self.lost_hold_ms:
            self.latch.reset()
            return [Event("select", self.hand, t, None)]
        return []
