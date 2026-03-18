"""Franka robot evaluation example."""

from __future__ import annotations

from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
_SRC_ROOT_STR = str(_SRC_ROOT)
if _SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, _SRC_ROOT_STR)
