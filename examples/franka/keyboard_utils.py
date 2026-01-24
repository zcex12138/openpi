"""Non-blocking keyboard detection utilities (Linux only)."""

from __future__ import annotations

import select
import sys
import termios
import tty
from contextlib import contextmanager
from typing import Generator


@contextmanager
def cbreak_terminal() -> Generator[None, None, None]:
    """Context manager for cbreak terminal mode (preserves SIGINT)."""
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def check_key_pressed() -> str | None:
    """Non-blocking key detection. Returns pressed char or None."""
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None
