#!/usr/bin/env python3
"""Phase 4 exit criteria: hammer 20 note-ons in 2 s. No dropouts, no clipping, <1 ms calls.

Measured two ways, because each catches something the other cannot:

* **Live** — real driver, real audio thread. Measures what `note_on()` actually costs the
  CV loop, which is the number that matters for latency.
* **Offline** — `realtime=False` and `render()`, giving the exact sample buffer. That is
  the only way to *prove* absence of clipping and dropouts rather than assume it.

    .venv\\Scripts\\python.exe tools\\audio_bench.py
    .venv\\Scripts\\python.exe tools\\audio_bench.py --quiet   # offline only, no sound
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from audio import Synth                                     # noqa: E402

SF2 = ROOT / "assets" / "FluidR3_GM.sf2"
CONFIG = ROOT / "config" / "mappings.yaml"
N_NOTES = 20
BURST_SECONDS = 2.0
PENTATONIC = [0, 2, 4, 7, 9]
ROOT_NOTE = 60


def load_config() -> tuple[dict, dict, dict]:
    """Instruments, sustain map, audio settings — all from config, none hard-coded."""
    with open(CONFIG, encoding="utf-8") as fh:          # system codec is GBK, not UTF-8
        cfg = yaml.safe_load(fh)
    return cfg["instruments"], cfg["sustain"], cfg.get("audio", {})


def pitches(n: int) -> list[int]:
    return [ROOT_NOTE + 12 * (i // len(PENTATONIC)) + PENTATONIC[i % len(PENTATONIC)]
            for i in range(n)]


def bench_live(instruments: dict, sustain: dict, audio_cfg: dict,
               play: bool) -> list[float]:
    """Fire N notes over 2 s on the real driver; return per-call costs in ms."""
    synth = Synth(str(SF2), instruments, sustain,
                  driver=audio_cfg.get("driver", "dsound"),
                  samplerate=audio_cfg.get("samplerate", 48000),
                  period_size=audio_cfg.get("period_size", 128),
                  periods=audio_cfg.get("periods", 2),
                  realtime=True)
    info = synth.info
    print(f"  driver={info.driver} period_size={info.period_size} "
          f"sr={info.samplerate} realtime={info.realtime}")
    if not info.realtime:
        print("  no audio device opened; skipping live measurement")
        synth.close()
        return []

    gap = BURST_SECONDS / N_NOTES
    costs: list[float] = []
    names = list(sorted(instruments))
    next_t = time.perf_counter()
    for i, pitch in enumerate(pitches(N_NOTES)):
        inst = names[i % len(names)]
        t0 = time.perf_counter()
        synth.note_on(inst, pitch, 100)
        costs.append((time.perf_counter() - t0) * 1e3)
        next_t += gap
        while time.perf_counter() < next_t:
            time.sleep(0.001)
    if play:
        time.sleep(1.2)                                  # let the tails ring
    t0 = time.perf_counter()
    synth.all_off()
    off_ms = (time.perf_counter() - t0) * 1e3
    print(f"  all_off() cost: {off_ms:.3f} ms")
    synth.close()
    return costs


def bench_offline(instruments: dict, sustain: dict, audio_cfg: dict) -> dict:
    """Render the same burst to a buffer and inspect it for clipping and dropouts."""
    sr = audio_cfg.get("samplerate", 48000)
    synth = Synth(str(SF2), instruments, sustain, samplerate=sr, realtime=False)
    names = list(sorted(instruments))
    gap_frames = int(sr * BURST_SECONDS / N_NOTES)

    chunks = [synth.render(int(sr * 0.05))]              # lead-in silence
    for i, pitch in enumerate(pitches(N_NOTES)):
        synth.note_on(names[i % len(names)], pitch, 100)
        chunks.append(synth.render(gap_frames))
    chunks.append(synth.render(int(sr * 1.0)))           # release tails
    synth.close()

    audio = np.concatenate(chunks, axis=0).astype(np.int32)
    mono = audio.mean(axis=1)
    peak = int(np.abs(audio).max())
    clipped = int((np.abs(audio) >= 32767).sum())

    # A dropout is a run of digital silence in the middle of a sounding passage.
    body = mono[int(sr * 0.1):-int(sr * 0.5)]
    silent = np.abs(body) < 2
    longest = 0
    run = 0
    for s in silent:
        run = run + 1 if s else 0
        longest = max(longest, run)
    return {
        "frames": audio.shape[0],
        "seconds": audio.shape[0] / sr,
        "peak": peak,
        "peak_dbfs": 20 * np.log10(max(peak, 1) / 32767.0),
        "clipped_samples": clipped,
        "longest_silence_ms": 1000.0 * longest / sr,
        "rms_dbfs": 20 * np.log10(max(float(np.sqrt((mono ** 2).mean())), 1.0) / 32767.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="offline only; makes no sound")
    args = ap.parse_args()
    if not SF2.is_file():
        print(f"Missing SoundFont: {SF2}", file=sys.stderr)
        return 1
    instruments, sustain, audio_cfg = load_config()

    print("=" * 62)
    print(f"Phase 4 audio benchmark - {N_NOTES} note-ons in {BURST_SECONDS:.0f}s")
    print("=" * 62)
    print(f"instruments: {instruments}   sustain: {sustain}")

    costs: list[float] = []
    if not args.quiet:
        print("\n[live] real driver, audible")
        costs = bench_live(instruments, sustain, audio_cfg, play=True)
        if costs:
            arr = np.array(costs)
            print(f"  note_on() cost: mean {arr.mean():.3f} ms  "
                  f"p50 {np.percentile(arr, 50):.3f}  p95 {np.percentile(arr, 95):.3f}  "
                  f"max {arr.max():.3f} ms")

    print("\n[offline] rendered buffer analysis")
    m = bench_offline(instruments, sustain, audio_cfg)
    print(f"  rendered {m['seconds']:.2f}s ({m['frames']} frames)")
    print(f"  peak {m['peak']}/32767 ({m['peak_dbfs']:+.1f} dBFS)   "
          f"rms {m['rms_dbfs']:+.1f} dBFS")
    print(f"  clipped samples: {m['clipped_samples']}")
    print(f"  longest mid-passage silence: {m['longest_silence_ms']:.1f} ms")

    print("-" * 62)
    ok_cost = (not costs) or max(costs) < 1.0
    ok_clip = m["clipped_samples"] == 0
    ok_drop = m["longest_silence_ms"] < 10.0
    ok_sound = m["peak"] > 1000
    if costs:
        print(f"[{'PASS' if ok_cost else 'FAIL'}] note_on() < 1 ms "
              f"(max {max(costs):.3f} ms)")
    else:
        print("[ SKIP ] note_on() cost - no audio device")
    print(f"[{'PASS' if ok_clip else 'FAIL'}] no clipping "
          f"({m['clipped_samples']} clipped samples)")
    print(f"[{'PASS' if ok_drop else 'FAIL'}] no dropouts "
          f"(longest silence {m['longest_silence_ms']:.1f} ms)")
    print(f"[{'PASS' if ok_sound else 'FAIL'}] produced audible output "
          f"(peak {m['peak']})")
    return 0 if (ok_cost and ok_clip and ok_drop and ok_sound) else 1


if __name__ == "__main__":
    raise SystemExit(main())
