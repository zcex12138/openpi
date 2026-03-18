from __future__ import annotations

import time
import threading

import numpy as np

from examples.franka.camera_service import FrameStore
from examples.franka.camera_service import _AsyncFrameReader


def test_async_frame_reader_reports_unhealthy_after_failure() -> None:
    attempts = 0
    allow_failure = threading.Event()

    def grab():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return np.zeros((2, 2, 3), dtype=np.uint8)
        allow_failure.wait(timeout=1.0)
        raise RuntimeError("device disconnected")

    reader = _AsyncFrameReader("l500_rgb", grab)
    reader.start()
    try:
        frame, marker3d = reader.get(timeout_s=1.0)
        assert frame.shape == (2, 2, 3)
        assert marker3d is None

        allow_failure.set()
        deadline = time.time() + 1.0
        while True:
            try:
                reader.get(timeout_s=0.1)
            except RuntimeError as exc:
                assert "unhealthy" in str(exc)
                assert "device disconnected" in str(exc)
                break
            if time.time() >= deadline:
                raise AssertionError("Expected reader to become unhealthy after grab failure")
            time.sleep(0.01)
    finally:
        reader.stop()


def test_frame_store_exposes_latest_error_without_dropping_cached_frames() -> None:
    store = FrameStore()
    frames = {"l500_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
    markers = {"xense_1_marker3d": np.zeros((0, 0, 3), dtype=np.float32)}

    store.update(frames, markers)
    store.set_error("camera disconnected")

    cached_frames, cached_markers, timestamp_ns, seq, error = store.get()

    assert cached_frames is frames
    assert cached_markers is markers
    assert timestamp_ns > 0
    assert seq == 1
    assert error == "camera disconnected"
