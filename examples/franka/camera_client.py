"""Camera client for the Python 3.9 IPC camera service."""

from __future__ import annotations

import socket
import threading
from typing import Any

import cv2
import numpy as np

from examples.franka import ipc


class CameraClient:
    def __init__(self, host: str, port: int, *, timeout_s: float = 0.1) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def _connect(self) -> socket.socket:
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout_s)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        sock.settimeout(self._timeout_s)
        return sock

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            self._sock = self._connect()
        return self._sock

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            sock = self._ensure_socket()
            try:
                ipc.send_msg(sock, payload)
                return ipc.recv_msg(sock)
            except (OSError, ConnectionError, TimeoutError):
                self._close_unlocked()
                raise

    def ping(self) -> bool:
        try:
            response = self._request({"type": "ping"})
            return bool(response.get("ok"))
        except Exception:
            return False

    def get_frames(self) -> tuple[dict[str, np.ndarray], int, int]:
        frames, _marker3d, timestamp_ns, seq = self.get_frames_with_markers()
        return frames, timestamp_ns, seq

    def get_frames_with_markers(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int]:
        response = self._request({"type": "get_frames"})
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown camera service error"))

        frames = response.get("frames", {})
        decoded_frames = {key: _decode_array(value) for key, value in frames.items()}

        marker3d = response.get("marker3d", {})
        decoded_marker3d = {key: _decode_array(value) for key, value in marker3d.items()}

        return decoded_frames, decoded_marker3d, int(response.get("timestamp_ns", 0)), int(response.get("seq", 0))

    def get_frames_resized(self, height: int=224, width: int=224) -> tuple[dict[str, np.ndarray], int, int]:
        """Fetch frames and resize with cv2 to match preprocessing behavior (no pad, no crop)."""
        frames, timestamp_ns, seq = self.get_frames()
        resized = {key: _resize_frame_cv2(img, height=height, width=width) for key, img in frames.items()}
        return resized, timestamp_ns, seq

    def __enter__(self) -> CameraClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _decode_array(encoded: dict[str, Any] | None) -> np.ndarray:
    if encoded is None:
        raise RuntimeError("missing array in camera response")
    shape = encoded["shape"]
    dtype = np.dtype(encoded["dtype"])
    data = encoded["data"]
    arr = np.frombuffer(data, dtype=dtype)
    return arr.reshape(shape)


def _resize_frame_cv2(image: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize a single image using cv2.resize (H, W, C) -> (height, width, C)."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
