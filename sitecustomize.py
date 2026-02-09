"""Repository-wide console encoding safeguards for Windows/GBK environments."""

from __future__ import annotations

import os
import sys
from typing import TextIO


def _safe_reconfigure(stream: TextIO | None) -> None:
    if stream is None or not hasattr(stream, "reconfigure"):
        return
    try:
        # Prefer UTF-8 but never raise on unsupported glyphs.
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            return


os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

_safe_reconfigure(sys.stdout)
_safe_reconfigure(sys.stderr)
