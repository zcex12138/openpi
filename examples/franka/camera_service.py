"""Python 3.9 camera service with IPC for RealSense/XenseCamera frames.

Run this script in a Python 3.9 environment where the camera drivers and PyYAML are installed.
Camera parameters are loaded from a YAML config file.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
import signal
import socket
import socketserver
import threading
import time
from typing import Any

import numpy as np

from examples.franka import ipc
from examples.franka.utils import load_yaml_config

logger = logging.getLogger(__name__)


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] | None = None
        self._marker3d: dict[str, np.ndarray] | None = None
        self._timestamp_ns: int = 0
        self._seq: int = 0
        self._error: str | None = None

    def update(self, frames: dict[str, np.ndarray], marker3d: dict[str, np.ndarray]) -> None:
        now_ns = time.time_ns()
        with self._lock:
            self._frames = frames
            self._marker3d = marker3d
            self._timestamp_ns = now_ns
            self._seq += 1
            self._error = None

    def set_error(self, error: Exception | str) -> None:
        with self._lock:
            self._error = str(error)

    def get(self) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, int, int, str | None]:
        with self._lock:
            return self._frames, self._marker3d, self._timestamp_ns, self._seq, self._error


def _load_provider(path: str, kwargs: dict[str, Any]) -> Any:
    if ":" not in path:
        raise ValueError("--provider must be in the form 'module:attribute'")
    module_name, attr_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    provider = factory(**kwargs) if callable(factory) else factory
    if not hasattr(provider, "get_frames"):
        raise AttributeError("Camera provider must define get_frames()")
    return provider


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        return image
    return image[..., ::-1]


def _compute_marker3d(marker2d: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
    if marker2d is None or depth_frame is None:
        return np.zeros((0, 0, 3), dtype=np.float32)
    if marker2d.ndim != 3 or marker2d.shape[-1] != 2:
        return np.zeros((0, 0, 3), dtype=np.float32)
    height, width = depth_frame.shape[:2]
    x_coords = marker2d[:, :, 0]
    y_coords = marker2d[:, :, 1]
    x_int = np.round(x_coords).astype(np.int32)
    y_int = np.round(y_coords).astype(np.int32)

    valid_mask = (x_int >= 0) & (x_int < width) & (y_int >= 0) & (y_int < height)
    x_clipped = np.clip(x_int, 0, width - 1)
    y_clipped = np.clip(y_int, 0, height - 1)

    z_coords = depth_frame[y_clipped, x_clipped]
    z_coords = np.where(valid_mask, z_coords, 0.0)
    return np.stack([x_coords, y_coords, z_coords], axis=-1).astype(np.float32)


# Default configurations for different RealSense camera types
_REALSENSE_DEFAULTS: dict[str, dict[str, Any]] = {
    "L500": {
        "rgb_resolution": (960, 540),
        "depth_resolution": (640, 480),
        "fps": 30,
        "exposure": None,  # Auto exposure
        "white_balance": None,  # Auto white balance
    },
    "D400": {
        "rgb_resolution": (640, 480),
        "depth_resolution": (640, 480),
        "fps": 30,
        "exposure": 120,
        "white_balance": 5900,
    },
}


class SimpleRealsenseCamera:
    """Direct RealSense camera interface using pyrealsense2."""

    def __init__(
        self,
        camera_serial_number: str,
        camera_type: str = "D400",
        rgb_resolution: tuple[int, int] | None = None,
        depth_resolution: tuple[int, int] | None = None,
        fps: int | None = None,
        rgb_fps: int | None = None,
        depth_fps: int | None = None,
        enable_depth: bool = True,
        exposure: int | None = None,
        white_balance: int | None = None,
        decimate: int = 1,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type

        # Get default values for this camera type
        defaults = _REALSENSE_DEFAULTS.get(camera_type, _REALSENSE_DEFAULTS["D400"])

        self.rgb_resolution = rgb_resolution if rgb_resolution is not None else defaults["rgb_resolution"]
        self.depth_resolution = depth_resolution if depth_resolution is not None else defaults["depth_resolution"]
        default_fps = fps if fps is not None else defaults["fps"]
        self.rgb_fps = rgb_fps if rgb_fps is not None else default_fps
        self.depth_fps = depth_fps if depth_fps is not None else default_fps
        self.enable_depth = bool(enable_depth)
        self.exposure = exposure if exposure is not None else defaults["exposure"]
        self.white_balance = white_balance if white_balance is not None else defaults["white_balance"]

        # Create decimation filter
        self._decimate_filter = rs.decimation_filter()
        self._decimate_filter.set_option(rs.option.filter_magnitude, 2**decimate)

        self._pipeline: Any = None
        self._align: Any = None
        self._color_sensor: Any = None
        self._options_set = False
        self._started = False
        self._lock = threading.RLock()

    def start(self) -> None:
        with self._lock:
            self._started = False
            if self._pipeline is not None:
                self.stop()

            rs = self._rs

            context = rs.context()
            devices = context.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No RealSense devices found")

            device = None
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == self.camera_serial_number:
                    device = dev
                    break
            if device is None:
                raise RuntimeError(f"Camera with serial number {self.camera_serial_number} not found")

            if self.enable_depth:
                for rgb_format in (rs.format.rgb8, rs.format.bgr8):
                    try:
                        self._start_with_config(
                            self._make_config(
                                enable_depth=True,
                                rgb_resolution=self.rgb_resolution,
                                rgb_format=rgb_format,
                                rgb_fps=self.rgb_fps,
                                depth_resolution=self.depth_resolution,
                                depth_fps=self.depth_fps,
                            ),
                            enable_depth=True,
                        )
                        return
                    except RuntimeError as exc:
                        if "couldn't resolve requests" not in str(exc).lower():
                            raise
                        self._stop(log_info=False)

                logger.warning(
                    "RealSense camera %s: unsupported stream request (rgb=%sx%s@%s, depth=%sx%s@%s); retrying without depth",
                    self.camera_serial_number,
                    self.rgb_resolution[0],
                    self.rgb_resolution[1],
                    self.rgb_fps,
                    self.depth_resolution[0],
                    self.depth_resolution[1],
                    self.depth_fps,
                )

            for rgb_format in (rs.format.rgb8, rs.format.bgr8):
                try:
                    self._start_with_config(
                        self._make_config(
                            enable_depth=False,
                            rgb_resolution=self.rgb_resolution,
                            rgb_format=rgb_format,
                            rgb_fps=self.rgb_fps,
                            depth_resolution=self.depth_resolution,
                            depth_fps=self.depth_fps,
                        ),
                        enable_depth=False,
                    )
                    return
                except RuntimeError as exc:
                    if "couldn't resolve requests" not in str(exc).lower():
                        raise
                    self._stop(log_info=False)

            color_profile = self._pick_video_profile(
                device,
                stream=rs.stream.color,
                desired_resolution=self.rgb_resolution,
                desired_fps=self.rgb_fps,
                allowed_formats=(rs.format.bgr8, rs.format.rgb8, rs.format.yuyv),
            )
            if color_profile is None:
                raise RuntimeError(f"Unable to find supported color profiles for {self.camera_serial_number}") from None

            logger.warning(
                "RealSense camera %s: auto-selecting color profile %sx%s@%s (%s)",
                self.camera_serial_number,
                color_profile["width"],
                color_profile["height"],
                color_profile["fps"],
                color_profile["format"],
            )
            self._start_with_config(
                self._make_config(
                    enable_depth=False,
                    rgb_resolution=(color_profile["width"], color_profile["height"]),
                    rgb_format=color_profile["format"],
                    rgb_fps=color_profile["fps"],
                    depth_resolution=self.depth_resolution,
                    depth_fps=self.depth_fps,
                ),
                enable_depth=False,
            )

    def stop(self) -> None:
        with self._lock:
            self._stop(log_info=True)

    def _stop(self, *, log_info: bool) -> None:
        self._started = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except (RuntimeError, OSError) as exc:
                logger.debug("Failed to stop RealSense pipeline: %s", exc)
            self._pipeline = None
        self._align = None
        self._color_sensor = None
        if log_info:
            logger.info("RealSense camera %s stopped", self.camera_serial_number)

    def _make_config(
        self,
        *,
        enable_depth: bool,
        rgb_resolution: tuple[int, int],
        rgb_format: Any,
        rgb_fps: int,
        depth_resolution: tuple[int, int],
        depth_fps: int,
    ) -> Any:
        rs = self._rs
        config = rs.config()
        config.enable_device(self.camera_serial_number)
        if enable_depth:
            config.enable_stream(
                rs.stream.depth,
                depth_resolution[0],
                depth_resolution[1],
                rs.format.z16,
                depth_fps,
            )
        config.enable_stream(
            rs.stream.color,
            rgb_resolution[0],
            rgb_resolution[1],
            rgb_format,
            rgb_fps,
        )
        return config

    def _start_with_config(self, config: Any, *, enable_depth: bool) -> None:
        rs = self._rs
        self._pipeline = rs.pipeline()
        try:
            profile = self._pipeline.start(config)

            device = profile.get_device()
            product_line = str(device.get_info(rs.camera_info.product_line))
            if product_line != self.camera_type:
                raise RuntimeError(f"Camera type mismatch: expected {self.camera_type}, got {product_line}")

            self._align = rs.align(rs.stream.color) if enable_depth else None
            self._color_sensor = device.first_color_sensor()
            self._options_set = False
            self._started = True
            logger.info("RealSense camera %s (%s) started", self.camera_serial_number, self.camera_type)
            try:
                streams = profile.get_streams()
                stream_descs = []
                for stream in streams:
                    if stream.stream_type() not in (rs.stream.color, rs.stream.depth):
                        continue
                    vsp = stream.as_video_stream_profile()
                    stream_descs.append(
                        f"{stream.stream_type()} {vsp.width()}x{vsp.height()}@{vsp.fps()} {stream.format()}"
                    )
                if stream_descs:
                    logger.info("RealSense camera %s active streams: %s", self.camera_serial_number, ", ".join(stream_descs))
            except Exception as exc:
                logger.debug("Failed to log active RealSense streams: %s", exc)
        except Exception:
            self._stop(log_info=False)
            raise

    def _pick_video_profile(
        self,
        device: Any,
        *,
        stream: Any,
        desired_resolution: tuple[int, int],
        desired_fps: int,
        allowed_formats: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if profile.stream_type() != stream:
                    continue
                if profile.format() not in allowed_formats:
                    continue
                try:
                    v_profile = profile.as_video_stream_profile()
                except Exception:
                    continue
                candidates.append(
                    {
                        "width": int(v_profile.width()),
                        "height": int(v_profile.height()),
                        "fps": int(v_profile.fps()),
                        "format": profile.format(),
                    }
                )

        if not candidates:
            return None

        desired_w, desired_h = desired_resolution
        format_rank = {fmt: idx for idx, fmt in enumerate(allowed_formats)}
        exact = [c for c in candidates if c["fps"] == desired_fps]
        if exact:
            return min(
                exact,
                key=lambda c: (
                    (c["width"] - desired_w) ** 2 + (c["height"] - desired_h) ** 2,
                    format_rank.get(c["format"], len(allowed_formats)),
                ),
            )

        higher = [c for c in candidates if c["fps"] > desired_fps]
        if higher:
            best_fps = min(c["fps"] for c in higher)
            same_fps = [c for c in higher if c["fps"] == best_fps]
            return min(
                same_fps,
                key=lambda c: (
                    (c["width"] - desired_w) ** 2 + (c["height"] - desired_h) ** 2,
                    format_rank.get(c["format"], len(allowed_formats)),
                ),
            )

        lower = [c for c in candidates if c["fps"] < desired_fps]
        best_fps = max(c["fps"] for c in lower)
        same_fps = [c for c in lower if c["fps"] == best_fps]
        return min(
            same_fps,
            key=lambda c: (
                (c["width"] - desired_w) ** 2 + (c["height"] - desired_h) ** 2,
                format_rank.get(c["format"], len(allowed_formats)),
            ),
        )

    def _set_exposure(self, exposure: int | None = None, gain: int | None = None) -> None:
        rs = self._rs
        if exposure is None and gain is None:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 1.0)
        else:
            self._color_sensor.set_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self._color_sensor.set_option(rs.option.exposure, exposure)
            if gain is not None:
                self._color_sensor.set_option(rs.option.gain, gain)

    def _set_white_balance(self, white_balance: int | None = None) -> None:
        rs = self._rs
        if white_balance is None:
            self._color_sensor.set_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self._color_sensor.set_option(rs.option.enable_auto_white_balance, 0.0)
            self._color_sensor.set_option(rs.option.white_balance, white_balance)

    def get_frame(self, timeout_ms: int = 5000) -> np.ndarray:
        """Get color frame as BGR numpy array."""
        rs = self._rs
        with self._lock:
            if self._pipeline is None or not self._started:
                raise RuntimeError("Camera not started. Call start() first.")

            deadline_s = time.monotonic() + (timeout_ms / 1000.0)
            restarted = False
            while True:
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    raise RuntimeError(f"Timeout waiting for RealSense frames ({timeout_ms}ms)")
                wait_ms = max(1, min(int(remaining_s * 1000.0), 1000))
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=wait_ms)
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if not restarted and ("before start" in msg or "device disconnected" in msg):
                        logger.warning("RealSense camera %s stream stopped; restarting", self.camera_serial_number)
                        self.stop()
                        self.start()
                        restarted = True
                        continue
                    raise

                if self._align is not None:
                    frames = self._align.process(frames)

                depth_frame = frames.get_depth_frame() if self._align is not None else None
                color_frame = frames.get_color_frame()

                if not color_frame or (self._align is not None and not depth_frame):
                    continue

                if not self._options_set:
                    try:
                        self._color_sensor.set_option(rs.option.global_time_enabled, 1)
                    except (RuntimeError, OSError) as exc:
                        logger.debug("Failed to set global_time_enabled: %s", exc)
                    try:
                        self._set_exposure(exposure=self.exposure, gain=0)
                    except (RuntimeError, OSError) as exc:
                        logger.debug("Failed to set exposure: %s", exc)
                    try:
                        self._set_white_balance(white_balance=self.white_balance)
                    except (RuntimeError, OSError) as exc:
                        logger.debug("Failed to set white_balance: %s", exc)
                    self._options_set = True

                color_profile = color_frame.get_profile().as_video_stream_profile()
                color_format = color_profile.format()
                if color_format == rs.format.yuyv:
                    color_image = self._yuyv_to_bgr(
                        color_frame.get_data(),
                        width=int(color_profile.width()),
                        height=int(color_profile.height()),
                    )
                else:
                    color_image = np.asanyarray(color_frame.get_data())

                if color_format == rs.format.rgb8:
                    color_image = color_image[..., ::-1]

                return color_image

    @staticmethod
    def _yuyv_to_bgr(data: Any, *, width: int, height: int) -> np.ndarray:
        frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 2))
        y = frame[:, :, 0].astype(np.int32)
        u = np.repeat(frame[:, 0::2, 1].astype(np.int32), 2, axis=1)
        v = np.repeat(frame[:, 1::2, 1].astype(np.int32), 2, axis=1)

        c = y - 16
        d = u - 128
        e = v - 128

        r = (298 * c + 409 * e + 128) >> 8
        g = (298 * c - 100 * d - 208 * e + 128) >> 8
        b = (298 * c + 516 * d + 128) >> 8

        out = np.stack([b, g, r], axis=-1)
        return np.clip(out, 0, 255).astype(np.uint8)


class SimpleXenseCamera:
    """Direct Xense camera interface using xensesdk."""

    def __init__(self, camera_index: str) -> None:
        self.camera_index = camera_index
        self._sensor: Any = None

    def start(self) -> None:
        from xensesdk import Sensor

        if self._sensor is None:
            self._sensor = Sensor.create(self.camera_index, use_gpu=True)
            logger.info("Xense camera %s started", self.camera_index)
        else:
            logger.warning("Xense camera %s is already running", self.camera_index)

    def stop(self) -> None:
        if self._sensor is not None:
            self._sensor.release()
            self._sensor = None
            logger.info("Xense camera %s stopped", self.camera_index)

    def get_frame(self) -> np.ndarray:
        """Get color frame and marker3d as numpy arrays."""
        from xensesdk import Sensor

        if self._sensor is None:
            raise RuntimeError("Camera not started. Call start() first.")

        rgb_frame, depth_frame, marker2d = self._sensor.selectSensorInfo(
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Marker2D,
        )
        marker3d = _compute_marker3d(marker2d, depth_frame)
        return rgb_frame, marker3d


class _AsyncFrameReader:
    def __init__(self, name: str, grab_fn: Any) -> None:
        self._name = name
        self._grab_fn = grab_fn
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._marker3d: np.ndarray | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_warn_s = 0.0
        self._last_success_s: float | None = None
        self._last_failure_s: float | None = None
        self._last_failure: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"camera-reader:{self._name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        self._thread = None

    def get(self, *, timeout_s: float = 2.0) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._ready.wait(timeout=timeout_s):
            raise RuntimeError(f"Timeout waiting for camera frames: {self._name}")
        with self._lock:
            if self._frame is None:
                raise RuntimeError(f"Missing camera frame: {self._name}")
            if self._last_failure is not None and (
                self._last_success_s is None
                or self._last_failure_s is None
                or self._last_failure_s >= self._last_success_s
            ):
                raise RuntimeError(f"Camera {self._name} unhealthy: {self._last_failure}")
            return self._frame, self._marker3d

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._grab_fn()
                if isinstance(result, tuple) and len(result) == 2:
                    frame, marker3d = result
                else:
                    frame, marker3d = result, None
                frame = np.asarray(frame)
                if not frame.flags["OWNDATA"]:
                    frame = frame.copy()
                now_s = time.monotonic()
                with self._lock:
                    self._frame = frame
                    self._marker3d = marker3d
                    self._last_success_s = now_s
                    self._last_failure_s = None
                    self._last_failure = None
                self._ready.set()
            except Exception as exc:
                now_s = time.monotonic()
                with self._lock:
                    self._last_failure_s = now_s
                    self._last_failure = str(exc)
                if now_s - self._last_warn_s > 5.0:
                    logger.warning("Camera %s grab failed: %s", self._name, exc)
                    self._last_warn_s = now_s
                time.sleep(0.05)


def _unpack_frames_result(result: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Unpack get_frames() result into (frames, markers) tuple."""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


class CameraProvider:
    """Unified camera provider for L500, D400, and optional Xense cameras."""

    def __init__(
        self,
        *,
        l500_serial: str,
        d400_kind: str,
        d400_serial: str | None = None,
        xense_index: str | None = None,
        xense_indices: list[str] | None = None,
        xense_frame_keys: list[str] | None = None,
        xense_marker_keys: list[str] | None = None,
        l500_kwargs: dict[str, Any] | None = None,
        d400_kwargs: dict[str, Any] | None = None,
        convert_bgr: bool = True,
        l500_key: str = "l500_rgb",
        d400_key: str = "d400_rgb",
    ) -> None:
        self._lock = threading.Lock()
        self._convert_bgr = convert_bgr
        self._l500_key = l500_key
        self._d400_key = d400_key
        self._xense_frame_keys = xense_frame_keys or []
        self._xense_marker_keys = xense_marker_keys or []

        self._l500 = SimpleRealsenseCamera(
            camera_serial_number=l500_serial,
            camera_type="L500",
            **(l500_kwargs or {}),
        )
        self._d400_is_bgr = False

        if d400_kind == "realsense":
            if d400_serial is None:
                raise ValueError("d400_serial is required when d400_kind='realsense'")
            self._d400 = SimpleRealsenseCamera(
                camera_serial_number=d400_serial,
                camera_type="D400",
                **(d400_kwargs or {}),
            )
            self._d400_is_bgr = True
        elif d400_kind == "xense":
            idx = xense_index or (xense_indices[0] if xense_indices else None)
            if idx is None:
                raise ValueError("xense_index or xense_indices required when d400_kind='xense'")
            self._d400 = SimpleXenseCamera(camera_index=idx)
            self._d400_is_bgr = True
        else:
            raise ValueError(f"Unsupported d400 kind: {d400_kind}")

        self._xense_cameras = [SimpleXenseCamera(camera_index=idx) for idx in (xense_indices or [])]
        self._l500_reader: _AsyncFrameReader | None = None
        self._d400_reader: _AsyncFrameReader | None = None
        self._xense_readers: list[_AsyncFrameReader] = []

    def start(self) -> None:
        with self._lock:
            self._l500.start()
            self._d400.start()
            for cam in self._xense_cameras:
                cam.start()

            self._l500_reader = _AsyncFrameReader(self._l500_key, lambda: self._l500.get_frame(timeout_ms=1000))
            if isinstance(self._d400, SimpleRealsenseCamera):
                d400_grab = lambda: self._d400.get_frame(timeout_ms=1000)
            else:
                d400_grab = self._d400.get_frame
            self._d400_reader = _AsyncFrameReader(self._d400_key, d400_grab)
            self._xense_readers = [
                _AsyncFrameReader(key, cam.get_frame)
                for key, cam in zip(self._xense_frame_keys, self._xense_cameras)
            ]

            self._l500_reader.start()
            self._d400_reader.start()
            for reader in self._xense_readers:
                reader.start()

    def get_frames(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        with self._lock:
            l500_reader = self._l500_reader
            d400_reader = self._d400_reader
            xense_readers = list(self._xense_readers)

        if l500_reader is None or d400_reader is None:
            raise RuntimeError("Camera provider not started")

        frames: dict[str, np.ndarray] = {}
        markers: dict[str, np.ndarray] = {}

        l500_img, _ = l500_reader.get()
        d400_img, _ = d400_reader.get()

        if self._convert_bgr:
            l500_img = _bgr_to_rgb(l500_img)
            if self._d400_is_bgr:
                d400_img = _bgr_to_rgb(d400_img)

        frames[self._l500_key] = l500_img
        frames[self._d400_key] = d400_img

        for idx, reader in enumerate(xense_readers):
            if idx >= len(self._xense_frame_keys) or idx >= len(self._xense_marker_keys):
                break
            frame_key = self._xense_frame_keys[idx]
            marker_key = self._xense_marker_keys[idx]
            rgb_frame, marker3d = reader.get()
            if self._convert_bgr:
                rgb_frame = _bgr_to_rgb(rgb_frame)
            frames[frame_key] = rgb_frame
            markers[marker_key] = marker3d if marker3d is not None else np.zeros((0, 0, 3), dtype=np.float32)

        return frames, markers

    def close(self) -> None:
        with self._lock:
            l500_reader = self._l500_reader
            d400_reader = self._d400_reader
            xense_readers = list(self._xense_readers)
            self._l500_reader = None
            self._d400_reader = None
            self._xense_readers = []

        if l500_reader is not None:
            l500_reader.stop()
        if d400_reader is not None:
            d400_reader.stop()
        for reader in xense_readers:
            reader.stop()

        with self._lock:
            self._l500.stop()
            self._d400.stop()
            for cam in self._xense_cameras:
                cam.stop()


# Backward compatibility aliases
DualCameraProvider = CameraProvider
MultiCameraProvider = CameraProvider


def _get_service_config(config: dict[str, Any]) -> dict[str, Any]:
    if "camera_service" in config and isinstance(config["camera_service"], dict):
        return config["camera_service"]
    camera_cfg = config.get("camera")
    if isinstance(camera_cfg, dict):
        service_cfg = camera_cfg.get("camera_service")
        if isinstance(service_cfg, dict):
            return service_cfg
    return {}


def _get_camera_config(config: dict[str, Any]) -> dict[str, Any]:
    camera_cfg = config.get("camera")
    if isinstance(camera_cfg, dict):
        return camera_cfg
    return config


def _find_realsense_config(
    configs: list[dict[str, Any]],
    *,
    camera_type: str | None = None,
    camera_name: str | None = None,
) -> dict[str, Any] | None:
    for cam in configs:
        if camera_name is not None and cam.get("camera_name") != camera_name:
            continue
        if camera_type is not None:
            cam_type = str(cam.get("camera_type", "")).upper()
            if cam_type != camera_type.upper():
                continue
        return cam
    return None


def _find_xense_config(
    configs: list[dict[str, Any]],
    *,
    camera_name: str | None = None,
) -> dict[str, Any] | None:
    for cam in configs:
        if camera_name is not None and cam.get("camera_name") != camera_name:
            continue
        return cam
    return None


def _realsense_kwargs(cam_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract RealSense camera kwargs from config."""
    kwargs: dict[str, Any] = {}
    if cam_cfg.get("decimate") is not None:
        kwargs["decimate"] = cam_cfg["decimate"]
    if cam_cfg.get("rgb_resolution") is not None:
        kwargs["rgb_resolution"] = tuple(cam_cfg["rgb_resolution"])
    if cam_cfg.get("depth_resolution") is not None:
        kwargs["depth_resolution"] = tuple(cam_cfg["depth_resolution"])
    if cam_cfg.get("fps") is not None:
        kwargs["fps"] = cam_cfg["fps"]
    if cam_cfg.get("rgb_fps") is not None:
        kwargs["rgb_fps"] = cam_cfg["rgb_fps"]
    if cam_cfg.get("depth_fps") is not None:
        kwargs["depth_fps"] = cam_cfg["depth_fps"]
    if cam_cfg.get("enable_depth") is not None:
        kwargs["enable_depth"] = bool(cam_cfg["enable_depth"])
    elif cam_cfg.get("use_depth") is not None:
        kwargs["enable_depth"] = bool(cam_cfg["use_depth"])
    if cam_cfg.get("exposure") is not None:
        kwargs["exposure"] = cam_cfg["exposure"]
    if cam_cfg.get("white_balance") is not None:
        kwargs["white_balance"] = cam_cfg["white_balance"]
    return kwargs


def _xense_kwargs(cam_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract Xense camera kwargs from config (currently none needed)."""
    return {}


def _build_provider_from_config(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    camera_cfg = _get_camera_config(config)
    if not camera_cfg.get("enable", True):
        raise ValueError("Camera config is disabled (camera.enable=false)")

    realsense_configs = camera_cfg.get("realsense_cameras", [])
    xense_configs = camera_cfg.get("xense_cameras", [])
    if not isinstance(realsense_configs, list) or not isinstance(xense_configs, list):
        raise ValueError("Camera config must define realsense_cameras/xense_cameras as lists")

    service_cfg = _get_service_config(config)
    l500_name = service_cfg.get("l500_name")
    d400_name = service_cfg.get("d400_name")
    d400_kind = service_cfg.get("d400_kind")
    l500_key = service_cfg.get("l500_key", "l500_rgb")
    d400_key = service_cfg.get("d400_key", "d400_rgb")

    xense_names = service_cfg.get("xense_names") or []
    xense_keys = service_cfg.get("xense_keys") or []
    xense_marker_keys = service_cfg.get("xense_marker_keys") or []

    l500_cfg = _find_realsense_config(realsense_configs, camera_name=l500_name, camera_type="L500")
    if l500_cfg is None:
        raise ValueError("Unable to find L500 realsense config in YAML")

    d400_cfg: dict[str, Any] | None = None
    d400_is_xense = False
    if d400_name:
        d400_cfg = _find_realsense_config(realsense_configs, camera_name=d400_name)
        if d400_cfg is None:
            d400_cfg = _find_xense_config(xense_configs, camera_name=d400_name)
            d400_is_xense = d400_cfg is not None
    elif d400_kind == "xense":
        d400_cfg = _find_xense_config(xense_configs)
        d400_is_xense = d400_cfg is not None
    elif d400_kind == "realsense":
        d400_cfg = _find_realsense_config(realsense_configs, camera_type="D400")
    else:
        d400_cfg = _find_realsense_config(realsense_configs, camera_type="D400")
        if d400_cfg is None:
            d400_cfg = _find_xense_config(xense_configs)
            d400_is_xense = d400_cfg is not None

    if d400_cfg is None:
        raise ValueError("Unable to find D400/Xense config in YAML")

    l500_serial = l500_cfg.get("camera_serial_number")
    if l500_serial is None:
        raise ValueError("L500 config missing camera_serial_number")

    l500_cam_name = l500_cfg.get("camera_name") or "l500"
    d400_cam_name = d400_cfg.get("camera_name") or ("xense" if d400_is_xense else "d400")

    if d400_is_xense:
        xense_index = d400_cfg.get("camera_index")
        if xense_index is None:
            raise ValueError("Xense config missing camera_index")
        provider = DualCameraProvider(
            l500_serial=str(l500_serial),
            d400_kind="xense",
            d400_serial=None,
            xense_index=str(xense_index),
            l500_kwargs=_realsense_kwargs(l500_cfg),
            d400_kwargs=_xense_kwargs(d400_cfg),
            convert_bgr=service_cfg.get("convert_bgr", True),
            l500_name=l500_cam_name,
            d400_name=d400_cam_name,
        )
    else:
        d400_serial = d400_cfg.get("camera_serial_number")
        if d400_serial is None:
            raise ValueError("D400 config missing camera_serial_number")
        if xense_names:
            if d400_kind == "xense":
                raise ValueError("d400_kind cannot be 'xense' when xense_names are configured")
            if xense_keys and len(xense_keys) != len(xense_names):
                raise ValueError("xense_keys length must match xense_names length")
            if xense_marker_keys and len(xense_marker_keys) != len(xense_names):
                raise ValueError("xense_marker_keys length must match xense_names length")

            if not xense_keys:
                xense_keys = [f"xense_{idx+1}_rgb" for idx in range(len(xense_names))]
            if not xense_marker_keys:
                xense_marker_keys = [f"xense_{idx+1}_marker3d" for idx in range(len(xense_names))]
            service_cfg["xense_keys"] = xense_keys
            service_cfg["xense_marker_keys"] = xense_marker_keys

            xense_indices: list[str] = []
            for name in xense_names:
                xense_cfg = _find_xense_config(xense_configs, camera_name=name)
                if xense_cfg is None:
                    raise ValueError(f"Unable to find Xense config for camera_name={name}")
                xense_index = xense_cfg.get("camera_index")
                if xense_index is None:
                    raise ValueError(f"Xense config missing camera_index for {name}")
                xense_indices.append(str(xense_index))

            provider = MultiCameraProvider(
                l500_serial=str(l500_serial),
                d400_kind="realsense",
                d400_serial=str(d400_serial),
                xense_indices=xense_indices,
                xense_frame_keys=xense_keys,
                xense_marker_keys=xense_marker_keys,
                l500_kwargs=_realsense_kwargs(l500_cfg),
                d400_kwargs=_realsense_kwargs(d400_cfg),
                convert_bgr=service_cfg.get("convert_bgr", True),
                l500_key=l500_key,
                d400_key=d400_key,
            )
        else:
            provider = DualCameraProvider(
                l500_serial=str(l500_serial),
                d400_kind="realsense",
                d400_serial=str(d400_serial),
                xense_index=None,
                l500_kwargs=_realsense_kwargs(l500_cfg),
                d400_kwargs=_realsense_kwargs(d400_cfg),
                convert_bgr=service_cfg.get("convert_bgr", True),
                l500_name=l500_cam_name,
                d400_name=d400_cam_name,
            )

    return provider, service_cfg


def _encode_array(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    }


def _encode_frames(frames: dict[str, np.ndarray], keys: list[str]) -> dict[str, Any]:
    missing = [key for key in keys if key not in frames]
    if missing:
        available = ", ".join(sorted(frames.keys()))
        raise KeyError(f"Missing frames. Expected {missing}, got: {available}")
    return {key: _encode_array(frames[key]) for key in keys}


def _poll_frames(
    provider: Any,
    store: FrameStore,
    poll_hz: float,
    stop_event: threading.Event,
    active_event: threading.Event | None = None,
) -> None:
    period_s = 1.0 / poll_hz if poll_hz > 0 else 0.0
    next_tick_s = time.monotonic()
    while not stop_event.is_set():
        try:
            if active_event is not None and not active_event.is_set():
                active_event.wait(timeout=0.2)
                next_tick_s = time.monotonic()
                continue
            result = provider.get_frames()
            if result is not None:
                frames, marker3d = _unpack_frames_result(result)
                store.update(frames, marker3d)
        except Exception as exc:
            if stop_event.is_set():
                break
            store.set_error(exc)
            logger.exception("Camera provider get_frames failed")
        if period_s > 0:
            next_tick_s += period_s
            sleep_s = next_tick_s - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick_s = time.monotonic()


class _RequestHandler(socketserver.BaseRequestHandler):
    def setup(self) -> None:
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        server: CameraServer = self.server  # type: ignore[assignment]
        server.client_connected()

    def handle(self) -> None:
        server: CameraServer = self.server  # type: ignore[assignment]
        try:
            while True:
                try:
                    request = ipc.recv_msg(self.request)
                except ConnectionError:
                    break
                req_type = request.get("type")
                try:
                    if req_type == "ping":
                        ipc.send_msg(self.request, {"ok": True})
                        continue
                    if req_type != "get_frames":
                        ipc.send_msg(self.request, {"ok": False, "error": f"unknown request type: {req_type}"})
                        continue
                except (BrokenPipeError, ConnectionResetError):
                    break

                try:
                    if server.poll_hz > 0:
                        frames, marker3d, ts_ns, seq, error = server.store.get()
                        if error is not None:
                            raise RuntimeError(error)
                        if frames is None:
                            frames, marker3d = _unpack_frames_result(server.provider.get_frames())
                            ts_ns = time.time_ns()
                            seq = 0
                            server.store.update(frames, marker3d)
                    else:
                        frames, marker3d = _unpack_frames_result(server.provider.get_frames())
                        ts_ns = time.time_ns()
                        seq = 0
                    encoded = _encode_frames(frames, server.frame_keys)
                    encoded_marker3d = _encode_frames(marker3d, server.marker_keys) if server.marker_keys else {}
                    ipc.send_msg(
                        self.request,
                        {
                            "ok": True,
                            "timestamp_ns": ts_ns,
                            "seq": seq,
                            "frames": encoded,
                            "marker3d": encoded_marker3d,
                        },
                    )
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected, exit handler gracefully
                    break
                except Exception as exc:
                    try:
                        ipc.send_msg(self.request, {"ok": False, "error": str(exc)})
                    except (BrokenPipeError, ConnectionResetError):
                        # Client disconnected while sending error, exit gracefully
                        break
        finally:
            server.client_disconnected()


class CameraServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        provider: Any,
        store: FrameStore,
        *,
        frame_keys: list[str],
        marker_keys: list[str],
        poll_hz: float,
    ) -> None:
        super().__init__(server_address, _RequestHandler)
        self.provider = provider
        self.store = store
        self.frame_keys = frame_keys
        self.marker_keys = marker_keys
        self.poll_hz = poll_hz
        self._active_lock = threading.Lock()
        self._active_clients = 0
        self.active_event = threading.Event()

    def client_connected(self) -> None:
        with self._active_lock:
            self._active_clients += 1
            self.active_event.set()

    def client_disconnected(self) -> None:
        with self._active_lock:
            if self._active_clients > 0:
                self._active_clients -= 1
            if self._active_clients == 0:
                self.active_event.clear()


def main() -> None:
    default_config = Path(__file__).parent / "camera_config.yaml"
    parser = argparse.ArgumentParser(description="Franka camera IPC service (Python 3.9)")
    parser.add_argument(
        "--config",
        default=str(default_config) if default_config.exists() else None,
        help=f"YAML config path (default: {default_config})",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--provider",
        default=None,
        help="Python path to camera provider factory, e.g. 'mymodule:make_camera_provider'",
    )
    parser.add_argument("--provider-kwargs", default="{}", help="JSON string of kwargs for the provider factory")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    service_cfg: dict[str, Any] = {}
    if args.config is not None:
        config = load_yaml_config(args.config)
        service_cfg = _get_service_config(config)

    if args.provider is not None:
        provider_kwargs = json.loads(args.provider_kwargs)
        provider = _load_provider(args.provider, provider_kwargs)
    else:
        if args.config is None:
            raise ValueError("--config is required when --provider is not set")
        provider, service_cfg = _build_provider_from_config(config)
    if hasattr(provider, "start"):
        provider.start()

    store = FrameStore()
    stop_event = threading.Event()
    poll_hz = service_cfg.get("poll_hz")
    if poll_hz is None:
        poll_hz = config.get("recording", {}).get("fps", 30.0)
    poll_hz = float(poll_hz)

    host = args.host or service_cfg.get("host", "0.0.0.0")
    port = int(args.port or service_cfg.get("port", 5050))
    l500_key = service_cfg.get("l500_key", "l500_rgb")
    d400_key = service_cfg.get("d400_key", "d400_rgb")
    xense_keys = service_cfg.get("xense_keys") or []
    xense_marker_keys = service_cfg.get("xense_marker_keys") or []
    frame_keys = [l500_key, d400_key, *xense_keys]

    server = CameraServer(
        (host, port),
        provider,
        store,
        frame_keys=frame_keys,
        marker_keys=xense_marker_keys,
        poll_hz=poll_hz,
    )

    poll_thread: threading.Thread | None = None
    if poll_hz > 0:
        poll_thread = threading.Thread(
            target=_poll_frames,
            args=(provider, store, poll_hz, stop_event, server.active_event),
            daemon=True,
        )
        poll_thread.start()

    shutdown_event = threading.Event()

    def _shutdown(signum: int, _frame: Any) -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        logger.info("Shutting down camera service (signal %s)", signum)
        # server.shutdown() must run on a different thread than serve_forever().
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Camera service listening on %s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        stop_event.set()
        if poll_thread is not None:
            poll_thread.join(timeout=15.0)
        server.server_close()
        if hasattr(provider, "close"):
            if poll_thread is None or not poll_thread.is_alive():
                provider.close()
            else:
                logger.warning("Poll thread did not shut down; skipping provider.close()")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
