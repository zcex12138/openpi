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
        self._timestamp_ns: int = 0
        self._seq: int = 0

    def update(self, frames: dict[str, np.ndarray]) -> None:
        now_ns = time.time_ns()
        with self._lock:
            self._frames = frames
            self._timestamp_ns = now_ns
            self._seq += 1

    def get(self) -> tuple[dict[str, np.ndarray] | None, int, int]:
        with self._lock:
            return self._frames, self._timestamp_ns, self._seq


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
        self.fps = fps if fps is not None else defaults["fps"]
        self.exposure = exposure if exposure is not None else defaults["exposure"]
        self.white_balance = white_balance if white_balance is not None else defaults["white_balance"]

        # Create decimation filter
        self._decimate_filter = rs.decimation_filter()
        self._decimate_filter.set_option(rs.option.filter_magnitude, 2**decimate)

        self._pipeline: Any = None
        self._align: Any = None
        self._color_sensor: Any = None
        self._options_set = False

    def start(self) -> None:
        rs = self._rs

        # Verify camera exists
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found")

        found = any(d.get_info(rs.camera_info.serial_number) == self.camera_serial_number for d in devices)
        if not found:
            raise RuntimeError(f"Camera with serial number {self.camera_serial_number} not found")

        # Configure pipeline
        config = rs.config()
        config.enable_device(self.camera_serial_number)
        config.enable_stream(
            rs.stream.depth,
            self.depth_resolution[0],
            self.depth_resolution[1],
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.rgb_resolution[0],
            self.rgb_resolution[1],
            rs.format.rgb8,
            self.fps,
        )

        self._pipeline = rs.pipeline()
        self._pipeline.start(config)

        # Verify camera type and setup alignment
        device = self._pipeline.get_active_profile().get_device()
        product_line = str(device.get_info(rs.camera_info.product_line))
        if product_line != self.camera_type:
            raise RuntimeError(f"Camera type mismatch: expected {self.camera_type}, got {product_line}")

        self._align = rs.align(rs.stream.color)
        self._color_sensor = device.first_color_sensor()
        self._options_set = False
        logger.info("RealSense camera %s (%s) started", self.camera_serial_number, self.camera_type)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except (RuntimeError, OSError) as exc:
                logger.debug("Failed to stop RealSense pipeline: %s", exc)
            self._pipeline = None
        logger.info("RealSense camera %s stopped", self.camera_serial_number)

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
        if self._pipeline is None:
            raise RuntimeError("Camera not started. Call start() first.")

        while True:
            frames = self._pipeline.wait_for_frames(timeout_ms=timeout_ms)
            frames = self._align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not color_frame or not depth_frame:
                continue

            # Set options after first successful frame
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

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Handle color format: L500 uses RGB8, D400 uses BGR8 - convert to BGR
            if color_frame.get_profile().as_video_stream_profile().format() == rs.format.rgb8:
                color_image = color_image[..., ::-1]  # RGB to BGR

            return color_image


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
        """Get RGB frame as numpy array."""
        from xensesdk import Sensor

        if self._sensor is None:
            raise RuntimeError("Camera not started. Call start() first.")

        rgb_frame, _depth_frame, _marker2d = self._sensor.selectSensorInfo(
            Sensor.OutputType.Rectify,
            Sensor.OutputType.Depth,
            Sensor.OutputType.Marker2D,
        )
        return rgb_frame


class DualCameraProvider:
    """Camera provider using local SimpleRealsenseCamera and SimpleXenseCamera."""

    def __init__(
        self,
        *,
        l500_serial: str,
        d400_kind: str,
        d400_serial: str | None,
        xense_index: str | None,
        l500_kwargs: dict[str, Any],
        d400_kwargs: dict[str, Any],
        convert_bgr: bool,
        l500_name: str = "l500",
        d400_name: str = "d400",
    ) -> None:
        self._convert_bgr = convert_bgr
        self._l500_name = l500_name
        self._d400_name = d400_name

        # Create L500 camera (always RealSense)
        self._l500 = SimpleRealsenseCamera(
            camera_serial_number=l500_serial,
            camera_type="L500",
            **l500_kwargs,
        )
        self._d400_is_bgr = False

        # Create D400 camera (RealSense or Xense)
        if d400_kind == "realsense":
            if d400_serial is None:
                raise ValueError("d400_serial is required when d400_kind='realsense'")
            self._d400 = SimpleRealsenseCamera(
                camera_serial_number=d400_serial,
                camera_type="D400",
                **d400_kwargs,
            )
            self._d400_is_bgr = True  # RealSense D400 returns BGR
        elif d400_kind == "xense":
            if xense_index is None:
                raise ValueError("xense_index is required when d400_kind='xense'")
            self._d400 = SimpleXenseCamera(camera_index=xense_index)
            self._d400_is_bgr = False  # Xense returns RGB
        else:
            raise ValueError(f"Unsupported d400 kind: {d400_kind}")

    def start(self) -> None:
        """Start both cameras."""
        self._l500.start()
        self._d400.start()

    def get_frames(self) -> dict[str, np.ndarray]:
        """Get frames from both cameras."""
        l500_img = self._l500.get_frame()  # BGR
        d400_img = self._d400.get_frame()  # BGR or RGB depending on camera type

        if l500_img is None or d400_img is None:
            raise RuntimeError("Camera returned empty frame")

        if self._convert_bgr:
            l500_img = _bgr_to_rgb(l500_img)
            if self._d400_is_bgr:
                d400_img = _bgr_to_rgb(d400_img)

        return {"l500_rgb": l500_img, "d400_rgb": d400_img}

    def close(self) -> None:
        """Stop both cameras."""
        self._l500.stop()
        self._d400.stop()


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


def _encode_frame(frame: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.tobytes(),
    }


def _encode_frames(frames: dict[str, np.ndarray], l500_key: str, d400_key: str) -> dict[str, Any]:
    if l500_key not in frames or d400_key not in frames:
        available = ", ".join(sorted(frames.keys()))
        raise KeyError(f"Missing frames. Expected '{l500_key}' and '{d400_key}', got: {available}")
    return {
        "l500_rgb": _encode_frame(frames[l500_key]),
        "d400_rgb": _encode_frame(frames[d400_key]),
    }


def _poll_frames(provider: Any, store: FrameStore, poll_hz: float, stop_event: threading.Event) -> None:
    sleep_s = 1.0 / poll_hz if poll_hz > 0 else 0.0
    while not stop_event.is_set():
        try:
            frames = provider.get_frames()
            if frames is not None:
                store.update(frames)
        except Exception:
            if stop_event.is_set():
                break
            logger.exception("Camera provider get_frames failed")
        if sleep_s > 0:
            time.sleep(sleep_s)


class _RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: CameraServer = self.server  # type: ignore[assignment]
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
                    frames, ts_ns, seq = server.store.get()
                    if frames is None:
                        ipc.send_msg(self.request, {"ok": False, "error": "no frames yet"})
                        continue
                else:
                    frames = server.provider.get_frames()
                    ts_ns = time.time_ns()
                    seq = 0
                encoded = _encode_frames(frames, server.l500_key, server.d400_key)
                ipc.send_msg(
                    self.request,
                    {
                        "ok": True,
                        "timestamp_ns": ts_ns,
                        "seq": seq,
                        "frames": encoded,
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


class CameraServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        provider: Any,
        store: FrameStore,
        *,
        l500_key: str,
        d400_key: str,
        poll_hz: float,
    ) -> None:
        super().__init__(server_address, _RequestHandler)
        self.provider = provider
        self.store = store
        self.l500_key = l500_key
        self.d400_key = d400_key
        self.poll_hz = poll_hz


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
    poll_thread: threading.Thread | None = None
    if poll_hz > 0:
        poll_thread = threading.Thread(
            target=_poll_frames,
            args=(provider, store, poll_hz, stop_event),
            daemon=True,
        )
        poll_thread.start()

    host = args.host or service_cfg.get("host", "0.0.0.0")
    port = int(args.port or service_cfg.get("port", 5050))
    l500_key = service_cfg.get("l500_key", "l500_rgb")
    d400_key = service_cfg.get("d400_key", "d400_rgb")

    server = CameraServer(
        (host, port),
        provider,
        store,
        l500_key=l500_key,
        d400_key=d400_key,
        poll_hz=poll_hz,
    )

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
            poll_thread.join(timeout=2.0)
        server.server_close()
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
