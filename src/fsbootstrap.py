"""Make the vendored FluidSynth discoverable before ``import fluidsynth``.

pyfluidsynth locates ``libfluidsynth-3.dll`` via :func:`ctypes.util.find_library`,
which on Windows searches **PATH only** — it does *not* consult directories
registered with :func:`os.add_dll_directory`. So we add the vendored ``bin`` dir to
both:

* ``PATH``               → so ``find_library('libfluidsynth-3')`` locates the DLL, and
* the DLL search path    → so its own dependencies (``SDL3.dll``, ``sndfile.dll``) load.

Call :func:`ensure_fluidsynth_on_path` once, at process start, *before* importing
``fluidsynth``. It is a no-op on macOS/Linux (where FluidSynth is installed
system-wide via brew/apt) and when nothing is vendored.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_BIN = _PROJECT_ROOT / "vendor" / "fluidsynth" / "bin"

_done = False


def ensure_fluidsynth_on_path() -> Path | None:
    """Register the vendored FluidSynth ``bin`` dir. Returns it, or ``None``.

    Idempotent. Safe to call on any platform.
    """
    global _done
    if _done:
        return _VENDOR_BIN if _VENDOR_BIN.is_dir() else None
    _done = True

    if sys.platform != "win32" or not _VENDOR_BIN.is_dir():
        return None

    bin_str = str(_VENDOR_BIN)
    # 1) PATH — so ctypes' find_library can *locate* libfluidsynth-3.dll
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_str not in path_parts:
        os.environ["PATH"] = bin_str + os.pathsep + os.environ.get("PATH", "")
    # 2) DLL search dir — so libfluidsynth's dependent DLLs *load*
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(bin_str)
    return _VENDOR_BIN
