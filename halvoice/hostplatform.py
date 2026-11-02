"""Tells which platform hal-voice runs on: plain Linux, WSL2 or an unsupported system.

The kernel's `osrelease` decides: WSL2 kernels say `microsoft-standard-WSL2`. Native Windows
and macOS have no `/proc` and no PulseAudio socket, so the voice loop refuses to start there
with a clear error instead of failing later in the audio code. `$HALVOICE_OSRELEASE` names a
file that replaces `/proc/sys/kernel/osrelease`, which is how the tests fake a platform.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

OSRELEASE = Path("/proc/sys/kernel/osrelease")
OSRELEASE_ENV = "HALVOICE_OSRELEASE"
LINUX = "linux"
WSL2 = "wsl2"
UNSUPPORTED = "unsupported"
DOCS_HINT = "docs/howto/run-on-windows-with-wsl2.md"


class UnsupportedPlatform(RuntimeError):
    """hal-voice runs on Linux, or on Windows through WSL2, and nowhere else."""


def osrelease_text(path: Path | None = None) -> str | None:
    """Returns the kernel release string, or None when it cannot be read."""
    path = path or Path(os.environ.get(OSRELEASE_ENV) or OSRELEASE)
    try:
        return path.read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return None


def detect_platform(osrelease: Path | None = None, system: str | None = None) -> str:
    """Returns `wsl2`, `linux` or `unsupported`.

    WSL1 also reports `Microsoft` in its release string; it has no working audio bridge, so a
    WSL1 user finds out from the how-to, not from a separate platform value.
    """
    if (system or sys.platform) != "linux":
        return UNSUPPORTED
    text = osrelease_text(osrelease)
    if text is None:
        return UNSUPPORTED
    if "microsoft" in text.lower() or "WSL" in text:
        return WSL2
    return LINUX


def require_supported(osrelease: Path | None = None, system: str | None = None) -> str:
    """Returns the platform, or raises `UnsupportedPlatform` with what to do instead."""
    found = detect_platform(osrelease, system)
    if found == UNSUPPORTED:
        raise UnsupportedPlatform(
            f"{system or sys.platform} is not supported: hal-voice runs on Linux, and on "
            f"Windows 11 through WSL2 (see {DOCS_HINT}); native Windows and macOS are not"
        )
    return found
