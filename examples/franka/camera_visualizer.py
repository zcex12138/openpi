"""Camera visualizer subscriber for Franka evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import threading
import time

import cv2
import numpy as np
from openpi_client.runtime import subscriber as _subscriber

from examples.franka import camera_client as _camera_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizerConfig:
    camera_host: str = "127.0.0.1"
    camera_port: int = 5050
    camera_timeout_s: float = 0.1
    display_keys: list[str] = field(default_factory=lambda: ["xense_1_rgb"])
    display_fps: float = 30.0
    window_scale: float = 1.0


class CameraVisualizer(_subscriber.Subscriber):
    """Runtime subscriber that displays camera frames during evaluation."""

    def __init__(self, config: VisualizerConfig) -> None:
        self._config = config
        self._camera: _camera_client.CameraClient | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def on_episode_start(self) -> None:
        self._stop_event.clear()
        self._camera = _camera_client.CameraClient(
            host=self._config.camera_host,
            port=self._config.camera_port,
            timeout_s=self._config.camera_timeout_s,
        )
        if not self._camera.ping():
            logger.warning("Visualizer: camera service not responding")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Camera visualizer started (keys=%s)", self._config.display_keys)

    def on_step(self, observation: dict, action: dict) -> None:
        pass

    def on_episode_end(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        cv2.destroyAllWindows()
        logger.info("Camera visualizer stopped")

    def _run(self) -> None:
        period_s = 1.0 / self._config.display_fps if self._config.display_fps > 0 else 0.0
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frames, _, _ = self._camera.get_frames()
                for key in self._config.display_keys:
                    frame = frames.get(key)
                    if frame is None:
                        continue
                    bgr = frame[:, :, ::-1].copy()
                    if self._config.window_scale != 1.0:
                        h, w = bgr.shape[:2]
                        new_w = int(w * self._config.window_scale)
                        new_h = int(h * self._config.window_scale)
                        bgr = cv2.resize(bgr, (new_w, new_h))
                    cv2.imshow(key, bgr)
                cv2.waitKey(1)
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.debug("Visualizer frame error: %s", exc)
            if period_s > 0:
                next_tick += period_s
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
