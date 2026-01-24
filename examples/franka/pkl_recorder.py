"""Episode-level PKL recorder for Franka evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import pickle
import queue
import re
import threading
import time
from typing import Any

import numpy as np
from openpi_client.runtime import subscriber as _subscriber

from examples.franka import env as _env

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecorderConfig:
    record_dir: Path
    record_fps: float = 30.0
    queue_size: int = 256
    prompt: str = ""


def _downsample_half(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        return np.asarray(image)
    return np.ascontiguousarray(image[::2, ::2, :])


def _empty_marker3d() -> np.ndarray:
    return np.zeros((0, 0, 3), dtype=np.float32)


class EpisodePklRecorder(_subscriber.Subscriber):
    """Runtime subscriber that records episode frames to a PKL file."""

    def __init__(self, environment: _env.FrankaEnvironment, config: RecorderConfig) -> None:
        self._env = environment
        self._config = config

        self._episode_index = self._resolve_start_index()
        self._frame_index = 0
        self._frames: list[dict[str, Any]] = []

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=config.queue_size)
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._record_thread: threading.Thread | None = None

        self._last_record_time: float | None = None
        self._period_s = 1.0 / config.record_fps if config.record_fps > 0 else 0.0
        self._dropped = 0
        self._episode_start_ns: int | None = None

    def on_episode_start(self) -> None:
        self._episode_index += 1
        self._frame_index = 0
        self._frames = []
        self._dropped = 0
        self._last_record_time = None
        self._episode_start_ns = None
        self._stop_event.clear()

        self._worker = threading.Thread(target=self._run_writer, daemon=True)
        self._worker.start()
        self._record_thread = threading.Thread(target=self._run_recorder, daemon=True)
        self._record_thread.start()

    def _resolve_start_index(self) -> int:
        output_dir = Path(self._config.record_dir)
        if not output_dir.exists():
            return -1
        max_idx = -1
        for path in output_dir.glob("episode_*.pkl"):
            match = re.match(r"episode_(\d+)$", path.stem)
            if match is None:
                continue
            try:
                idx = int(match.group(1))
            except ValueError:
                continue
            if idx > max_idx:
                max_idx = idx
        return max_idx

    def on_step(self, observation: dict, action: dict) -> None:
        return

    def on_episode_end(self) -> None:
        self._stop_event.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=2.0)
        if self._worker is not None:
            self._worker.join(timeout=10.0)

        output_dir = Path(self._config.record_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"episode_{self._episode_index:03d}.pkl"

        payload = {
            "version": 1,
            "episode_index": self._episode_index,
            "prompt": self._config.prompt,
            "fps": float(self._config.record_fps),
            "frames": self._frames,
        }

        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved episode PKL: %s (frames=%d, dropped=%d)", output_path, len(self._frames), self._dropped)

    def _should_record(self) -> bool:
        if self._period_s <= 0:
            return False
        now = time.monotonic()
        if self._last_record_time is None:
            self._last_record_time = now
            return True
        if now - self._last_record_time >= self._period_s:
            # Advance by one or more periods to reduce drift.
            steps = int((now - self._last_record_time) / self._period_s)
            self._last_record_time += self._period_s * max(1, steps)
            return True
        return False

    def _record_once(self) -> None:
        try:
            sample = self._env.get_recording_frame()
        except Exception as exc:
            logger.warning("Recorder failed to fetch frame: %s", exc)
            return

        record = self._build_record(sample)
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 50 == 0:
                logger.warning("Recorder queue full, dropped %d frames", self._dropped)

    def _run_recorder(self) -> None:
        if self._period_s <= 0:
            return
        next_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(next_time - now, 0.2))
                continue
            self._record_once()
            next_time += self._period_s
            if next_time < now:
                next_time = now + self._period_s

    def _run_writer(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                record = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._frames.append(record)

    def _build_record(self, sample: dict[str, Any]) -> dict[str, Any]:
        frames = sample.get("frames", {})
        marker3d = sample.get("marker3d", {})

        images = {
            "l500": _downsample_half(frames.get("l500_rgb")),
            "d400": _downsample_half(frames.get("d400_rgb")),
            "xense_1": _downsample_half(frames.get("xense_1_rgb")),
        }

        markers = {
            "xense_1": marker3d.get("xense_1_marker3d", _empty_marker3d()),
        }

        timestamp_ns = int(sample.get("timestamp_ns", 0))
        if timestamp_ns > 0:
            if self._episode_start_ns is None:
                self._episode_start_ns = timestamp_ns
            timestamp = (timestamp_ns - self._episode_start_ns) / 1e9
        else:
            timestamp = float(self._frame_index) / float(self._config.record_fps)
        record = {
            "timestamp": timestamp,
            "timestamp_ns": timestamp_ns,
            "seq": int(sample.get("seq", -1)),
            "frame_index": self._frame_index,
            "images": images,
            "marker3d": markers,
            "tcp_pose": np.asarray(sample.get("tcp_pose", np.zeros(7, dtype=np.float32)), dtype=np.float32),
            "tcp_velocity": np.asarray(sample.get("tcp_velocity", np.zeros(6, dtype=np.float32)), dtype=np.float32),
            "wrench": np.asarray(sample.get("wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32),
            "gripper": np.asarray(sample.get("gripper", np.zeros(1, dtype=np.float32)), dtype=np.float32),
            "is_human_teaching": self._env.is_teaching_mode,
        }
        self._frame_index += 1
        return record
