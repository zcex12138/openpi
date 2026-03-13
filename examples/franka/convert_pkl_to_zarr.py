"""Convert Franka PKL recordings to Zarr format.

PKL → Zarr 转换，输出格式兼容 convert_zarr_to_lerobot_v2.0.py。

Usage:
    python examples/franka/convert_pkl_to_zarr.py \
        --records /path/to/episode_000.pkl \
        --output_dir /path/to/output
"""

from __future__ import annotations

import dataclasses
import gc
import pickle
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import zarr

try:
    from numcodecs import Blosc as _NumcodecsBlosc
except ImportError:  # pragma: no cover
    _NumcodecsBlosc = None

try:
    from zarr.codecs import BloscCodec as _ZarrBloscCodec
except ImportError:  # pragma: no cover
    _ZarrBloscCodec = None

_DEFAULT_RECORDS = "/home/mpi/workspace/yhx/openpi/demo_records/"
_DEFAULT_RECORDS = "/home/mpi/workspace/yhx/openpi/eval_records/pi05_franka_cola_lora/20260207"
_DEFAULT_RECORDS = "data/dataset/dataset_zarr/20260126_失败单帧"
_DEFAULT_RECORDS = "eval_records"

ActionTarget = Literal["auto", "executed", "base", "corrected"]

_DEFAULT_IMAGE_SHAPE = (224, 224, 3)
_EMPTY_IMAGE = np.zeros((0, 0, 3), dtype=np.uint8)
_EMPTY_MARKER3D = np.zeros((0, 0, 3), dtype=np.float32)
_REQUIRED_DATA_KEYS = (
    "timestamp",
    "timestamp_ns",
    "control_timestamp",
    "seq",
    "frame_index",
    "l500_camera_img",
    "d400_camera_img",
    "robot_tcp_pose",
    "robot_tcp_velocity",
    "robot_tcp_wrench",
    "action",
    "executed_action",
    "base_action",
    "corrected_action",
    "corrected_action_valid",
    "is_human_teaching",
    "teaching_segment_id",
    "teaching_step",
)
_OPTIONAL_DATA_KEYS = ("xense1_camera_img", "xense1_marker3d")
_ZARR_MAJOR_VERSION = int(str(zarr.__version__).split(".", maxsplit=1)[0])


@dataclasses.dataclass
class Args:
    records: str = _DEFAULT_RECORDS
    output_dir: str | None = None
    temporal_downsample_ratio: int = 1
    drop_frames_after_human_teaching: int = 0
    action_target: ActionTarget = "auto"


def _collect_pkl_files(records_path: Path) -> list[Path]:
    if records_path.is_file():
        return [records_path]
    if records_path.is_dir():
        candidates = sorted(records_path.glob("episode_*.pkl"))
        if candidates:
            return candidates
    raise FileNotFoundError(f"No episode_*.pkl found under {records_path}")


def _load_pkl(pkl_path: Path) -> dict | None:
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f"Failed to load {pkl_path}: {e}")
        return None


def _vector_or_none(value: object, *, size: int) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    out = np.zeros(size, dtype=np.float32)
    out[: min(size, arr.size)] = arr[:size]
    return out


def _get_frame_images(frame: dict) -> dict:
    images = frame.get("images")
    if isinstance(images, dict):
        return images

    legacy_frames = frame.get("frames")
    if not isinstance(legacy_frames, dict):
        return {}

    return {
        "l500": legacy_frames.get("l500", legacy_frames.get("l500_rgb")),
        "d400": legacy_frames.get("d400", legacy_frames.get("d400_rgb")),
        "xense_1": legacy_frames.get("xense_1", legacy_frames.get("xense_1_rgb")),
    }


def _get_frame_marker3d(frame: dict) -> dict:
    marker3d = frame.get("marker3d")
    if not isinstance(marker3d, dict):
        return {}
    return {
        "xense_1": marker3d.get("xense_1", marker3d.get("xense_1_marker3d")),
    }


def _resolve_image_shape(frames: list[dict], camera_key: str, *, required: bool) -> tuple[int, int, int] | None:
    for frame in frames:
        image = _get_frame_images(frame).get(camera_key, _EMPTY_IMAGE)
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.size > 0:
            return arr.shape
    return _DEFAULT_IMAGE_SHAPE if required else None


def _resolve_marker3d_shape(frames: list[dict], marker_key: str) -> tuple[int, int, int] | None:
    shapes = {
        tuple(np.asarray(_get_frame_marker3d(frame).get(marker_key, _EMPTY_MARKER3D), dtype=np.float32).shape)
        for frame in frames
        if np.asarray(_get_frame_marker3d(frame).get(marker_key, _EMPTY_MARKER3D), dtype=np.float32).size > 0
    }
    if not shapes:
        return None
    if len(shapes) > 1:
        print(f"Skipping {marker_key} marker3d because non-empty frame shapes are inconsistent: {sorted(shapes)}")
        return None
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[-1] != 3:
        print(f"Skipping {marker_key} marker3d because shape is invalid: {shape}")
        return None
    return shape


def _normalize_image(value: object, *, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value if value is not None else _EMPTY_IMAGE, dtype=np.uint8)
    if arr.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (*, *, 3), got {arr.shape}")
    if arr.shape != shape:
        raise ValueError(f"Expected image shape {shape}, got {arr.shape}")
    return arr


def _normalize_marker3d(value: object, *, shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(value if value is not None else _EMPTY_MARKER3D, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(shape, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected marker3d with shape (*, *, 3), got {arr.shape}")
    if arr.shape != shape:
        raise ValueError(f"Expected marker3d shape {shape}, got {arr.shape}")
    return arr


def _extract_state_components(frame: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = _vector_or_none(frame.get("state"), size=14)
    tcp_pose = _vector_or_none(frame.get("tcp_pose"), size=7)
    gripper = _vector_or_none(frame.get("gripper"), size=1)
    wrench = _vector_or_none(frame.get("wrench"), size=6)
    tcp_velocity = _vector_or_none(frame.get("tcp_velocity"), size=6)

    if tcp_pose is None and state is not None:
        tcp_pose = state[:7].astype(np.float32, copy=True)
    if gripper is None and state is not None:
        gripper = state[7:8].astype(np.float32, copy=True)
    if wrench is None and state is not None:
        wrench = state[8:14].astype(np.float32, copy=True)

    if tcp_pose is None:
        tcp_pose = np.zeros(7, dtype=np.float32)
    if gripper is None:
        gripper = np.zeros(1, dtype=np.float32)
    if wrench is None:
        wrench = np.zeros(6, dtype=np.float32)
    if tcp_velocity is None:
        tcp_velocity = np.zeros(6, dtype=np.float32)
    return tcp_pose, gripper, wrench, tcp_velocity


def _select_action_target(
    *,
    action_target: ActionTarget,
    executed_action: np.ndarray | None,
    base_action: np.ndarray | None,
    corrected_action: np.ndarray | None,
    corrected_action_valid: bool,
) -> tuple[np.ndarray | None, bool]:
    if action_target == "executed":
        candidates = (
            executed_action,
            corrected_action if corrected_action_valid else None,
            base_action,
        )
    elif action_target == "base":
        candidates = (
            base_action,
            corrected_action if corrected_action_valid else None,
            executed_action,
        )
    elif action_target == "corrected":
        candidates = (
            corrected_action if corrected_action_valid else None,
            base_action,
            executed_action,
        )
    else:
        candidates = (
            executed_action,
            corrected_action if corrected_action_valid else None,
            base_action,
        )

    for candidate in candidates:
        if candidate is not None:
            return candidate.astype(np.float32, copy=True), True
    return None, False


def _infer_actions_from_next_pose(tcp_pose_arr: np.ndarray) -> np.ndarray:
    if len(tcp_pose_arr) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    actions = np.zeros_like(tcp_pose_arr, dtype=np.float32)
    if len(tcp_pose_arr) == 1:
        actions[0] = tcp_pose_arr[0]
        return actions
    actions[:-1] = tcp_pose_arr[1:]
    actions[-1] = tcp_pose_arr[-1]
    return actions


def _frame_timestamp(
    frame: dict,
    *,
    first_control_timestamp: float | None,
    index: int,
    fps: float,
) -> tuple[float, float | None]:
    if "timestamp" in frame:
        return float(frame["timestamp"]), first_control_timestamp

    control_timestamp = frame.get("control_timestamp")
    if control_timestamp is not None:
        current_control_timestamp = float(control_timestamp)
        if first_control_timestamp is None:
            first_control_timestamp = current_control_timestamp
        return float(current_control_timestamp - first_control_timestamp), first_control_timestamp

    safe_fps = fps if fps > 0 else 30.0
    return float(index / safe_fps), first_control_timestamp


def _process_episode(episode: dict, *, action_target: ActionTarget = "auto") -> dict[str, np.ndarray] | None:
    """Process single episode, return arrays dict."""
    frames = episode.get("frames", [])
    if not frames:
        return None

    n = len(frames)
    fps = float(episode.get("fps", 30.0))
    l500_shape = _resolve_image_shape(frames, "l500", required=True)
    d400_shape = _resolve_image_shape(frames, "d400", required=True)
    xense1_shape = _resolve_image_shape(frames, "xense_1", required=False)
    xense1_marker3d_shape = _resolve_marker3d_shape(frames, "xense_1")

    l500_list, d400_list = [], []
    xense1_list: list[np.ndarray] = []
    xense1_marker3d_list: list[np.ndarray] = []
    timestamp_arr = np.zeros(n, dtype=np.float32)
    timestamp_ns_arr = np.zeros(n, dtype=np.int64)
    control_timestamp_arr = np.zeros(n, dtype=np.float64)
    seq_arr = np.full(n, -1, dtype=np.int64)
    frame_index_arr = np.arange(n, dtype=np.int64)
    tcp_pose_arr = np.zeros((n, 8), dtype=np.float32)
    tcp_velocity_arr = np.zeros((n, 6), dtype=np.float32)
    tcp_wrench_arr = np.zeros((n, 6), dtype=np.float32)
    action_arr = np.zeros((n, 8), dtype=np.float32)
    executed_action_arr = np.zeros((n, 8), dtype=np.float32)
    base_action_arr = np.zeros((n, 8), dtype=np.float32)
    corrected_action_arr = np.zeros((n, 8), dtype=np.float32)
    corrected_action_valid_arr = np.zeros(n, dtype=np.uint8)
    is_human_teaching_arr = np.zeros(n, dtype=np.uint8)
    teaching_segment_id_arr = np.full(n, -1, dtype=np.int64)
    teaching_step_arr = np.full(n, -1, dtype=np.int64)
    has_target_action = False
    selected_action_mask = np.zeros(n, dtype=bool)
    first_control_timestamp: float | None = None

    for i, frame in enumerate(frames):
        images = _get_frame_images(frame)
        l500_list.append(_normalize_image(images.get("l500"), shape=l500_shape))
        d400_list.append(_normalize_image(images.get("d400"), shape=d400_shape))
        if xense1_shape is not None:
            xense1_list.append(_normalize_image(images.get("xense_1"), shape=xense1_shape))

        marker3d = _get_frame_marker3d(frame)
        if xense1_marker3d_shape is not None:
            xense1_marker3d_list.append(_normalize_marker3d(marker3d.get("xense_1"), shape=xense1_marker3d_shape))

        timestamp, first_control_timestamp = _frame_timestamp(
            frame,
            first_control_timestamp=first_control_timestamp,
            index=i,
            fps=fps,
        )
        timestamp_arr[i] = timestamp
        timestamp_ns_arr[i] = int(frame.get("timestamp_ns", 0))
        control_timestamp_arr[i] = float(frame.get("control_timestamp", 0.0))
        seq_arr[i] = int(frame.get("seq", -1))
        frame_index_arr[i] = int(frame.get("frame_index", i))

        tcp_pose, gripper, wrench, tcp_velocity = _extract_state_components(frame)
        tcp_pose_arr[i] = np.concatenate([tcp_pose, gripper], axis=0)
        tcp_velocity_arr[i] = tcp_velocity
        tcp_wrench_arr[i] = wrench
        is_human_teaching_arr[i] = 1 if bool(frame.get("is_human_teaching", False)) else 0
        teaching_segment_id_arr[i] = int(
            frame.get("teaching_segment_id", -1) if frame.get("teaching_segment_id") is not None else -1
        )
        teaching_step_arr[i] = int(frame.get("teaching_step", -1) if frame.get("teaching_step") is not None else -1)

        executed_action = _vector_or_none(frame.get("action"), size=8)
        base_action = _vector_or_none(frame.get("base_action"), size=8)
        corrected_action = _vector_or_none(frame.get("corrected_action"), size=8)
        corrected_action_valid = bool(frame.get("corrected_action_valid", corrected_action is not None))
        if base_action is None:
            base_action = executed_action

        if executed_action is not None:
            executed_action_arr[i] = executed_action
        if base_action is not None:
            base_action_arr[i] = base_action
        if corrected_action is not None:
            corrected_action_arr[i] = corrected_action

        corrected_action_valid_arr[i] = 1 if corrected_action_valid else 0

        selected_action, selected_valid = _select_action_target(
            action_target=action_target,
            executed_action=executed_action,
            base_action=base_action,
            corrected_action=corrected_action,
            corrected_action_valid=corrected_action_valid,
        )
        if selected_valid and selected_action is not None:
            action_arr[i] = selected_action
            selected_action_mask[i] = True
            has_target_action = True

    if not has_target_action or not selected_action_mask.all():
        fallback_actions = _infer_actions_from_next_pose(tcp_pose_arr)
        action_arr[~selected_action_mask] = fallback_actions[~selected_action_mask]

    result = {
        "timestamp": timestamp_arr,
        "timestamp_ns": timestamp_ns_arr,
        "control_timestamp": control_timestamp_arr,
        "seq": seq_arr,
        "frame_index": frame_index_arr,
        "l500_camera_img": np.stack(l500_list),
        "d400_camera_img": np.stack(d400_list),
        "robot_tcp_pose": tcp_pose_arr,
        "robot_tcp_velocity": tcp_velocity_arr,
        "robot_tcp_wrench": tcp_wrench_arr,
        "action": action_arr,
        "executed_action": executed_action_arr,
        "base_action": base_action_arr,
        "corrected_action": corrected_action_arr,
        "corrected_action_valid": corrected_action_valid_arr,
        "is_human_teaching": is_human_teaching_arr,
        "teaching_segment_id": teaching_segment_id_arr,
        "teaching_step": teaching_step_arr,
    }

    if xense1_shape is not None:
        xense1_arr = np.stack(xense1_list)
        result["xense1_camera_img"] = xense1_arr

    if xense1_marker3d_shape is not None:
        result["xense1_marker3d"] = np.stack(xense1_marker3d_list)

    return result


def _drop_after_teaching(data: dict[str, np.ndarray], n_drop: int) -> dict[str, np.ndarray]:
    """Drop n_drop frames after every human-teaching segment starts."""
    teaching = data["is_human_teaching"]
    ep_len = len(teaching)
    mask = np.ones(ep_len, dtype=bool)
    teaching_bool = np.asarray(teaching, dtype=bool)
    rising_edges = np.flatnonzero(teaching_bool & ~np.concatenate([[False], teaching_bool[:-1]]))
    for start in rising_edges:
        drop_end = min(int(start) + 1 + n_drop, ep_len)
        mask[int(start) + 1 : drop_end] = False
    if mask.all():
        return data
    return {k: v[mask] for k, v in data.items()}


def _apply_downsample(data: dict[str, np.ndarray], ratio: int) -> dict[str, np.ndarray]:
    if ratio <= 1:
        return data
    n = len(data["action"])
    if n <= 2:
        return data
    mid_idx = np.arange(1, n - 1)[::ratio]
    keep = np.concatenate([[0], mid_idx, [n - 1]])
    return {k: v[keep] for k, v in data.items()}


def _dataset_chunks(name: str, values: np.ndarray) -> tuple[int, ...]:
    if values.ndim == 1:
        return (min(10000, values.shape[0]),)
    if name.endswith("_img") or "marker3d" in name:
        return (min(100, values.shape[0]),) + values.shape[1:]
    return (min(10000, values.shape[0]),) + values.shape[1:]


def _make_compressor() -> object:
    if _ZARR_MAJOR_VERSION >= 3 and _ZarrBloscCodec is not None:
        return _ZarrBloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    if _NumcodecsBlosc is not None:
        return _NumcodecsBlosc(cname="zstd", clevel=3, shuffle=_NumcodecsBlosc.SHUFFLE)
    raise ImportError("A Blosc codec implementation is required to write Zarr output.")


def _create_dataset(group: zarr.Group, name: str, values: np.ndarray, compressor: object) -> None:
    chunks = _dataset_chunks(name, values)
    if _ZARR_MAJOR_VERSION >= 3:
        arr = group.create_array(
            name,
            shape=values.shape,
            chunks=chunks,
            dtype=values.dtype,
            compressors=(compressor,),
            fill_value=0,
        )
        arr[:] = values
        return
    group.create_dataset(
        name,
        data=values,
        chunks=chunks,
        dtype=values.dtype,
        compressor=compressor,
    )


def _append_or_create_optional_dataset(
    group: zarr.Group,
    name: str,
    values: np.ndarray | None,
    *,
    previous_total: int,
    current_frames: int,
    compressor: object,
) -> None:
    if values is not None:
        if name not in group:
            prefix_shape = (previous_total,) + values.shape[1:]
            prefix = np.zeros(prefix_shape, dtype=values.dtype)
            _create_dataset(group, name, np.concatenate([prefix, values], axis=0), compressor)
            return
        group[name].append(values)
        return

    if name not in group:
        return

    if current_frames == 0:
        return
    tail_shape = (current_frames,) + group[name].shape[1:]
    group[name].append(np.zeros(tail_shape, dtype=group[name].dtype))


def main(args: Args) -> None:
    records_path = Path(args.records)
    pkl_files = _collect_pkl_files(records_path)

    output_dir = (
        Path(args.output_dir) if args.output_dir else records_path if records_path.is_dir() else records_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / "replay_buffer.zarr"

    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    print(f"Converting {len(pkl_files)} PKL files to {zarr_path}")

    zarr_root = zarr.group(str(zarr_path))
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = _make_compressor()

    episode_ends: list[int] = []
    trajectory_ids: list[str] = []
    prompts: list[str] = []
    episode_fps: list[float] = []
    total_frames = 0
    initialized = False

    for idx, pkl_path in enumerate(pkl_files):
        print(f"[{idx + 1}/{len(pkl_files)}] {pkl_path.name}")
        episode = _load_pkl(pkl_path)
        if episode is None:
            continue

        ep_data = _process_episode(episode, action_target=args.action_target)
        episode_prompt = str(episode.get("prompt", ""))
        episode_fps_value = float(episode.get("fps", 30.0))
        del episode
        gc.collect()

        if ep_data is None:
            continue

        if args.drop_frames_after_human_teaching > 0:
            before = len(ep_data["action"])
            ep_data = _drop_after_teaching(ep_data, args.drop_frames_after_human_teaching)
            dropped = before - len(ep_data["action"])
            if dropped > 0:
                print(f"  Dropped {dropped} frames after human_teaching")

        if args.temporal_downsample_ratio > 1:
            ep_data = _apply_downsample(ep_data, args.temporal_downsample_ratio)

        n_frames = len(ep_data["action"])
        if n_frames == 0:
            print("  Skipping empty episode after filtering")
            continue

        if not initialized:
            for key in _REQUIRED_DATA_KEYS:
                _create_dataset(zarr_data, key, ep_data[key], compressor)
            for key in _OPTIONAL_DATA_KEYS:
                if key in ep_data:
                    _create_dataset(zarr_data, key, ep_data[key], compressor)
            initialized = True
        else:
            for key in _REQUIRED_DATA_KEYS:
                zarr_data[key].append(ep_data[key])
            for key in _OPTIONAL_DATA_KEYS:
                _append_or_create_optional_dataset(
                    zarr_data,
                    key,
                    ep_data.get(key),
                    previous_total=total_frames,
                    current_frames=n_frames,
                    compressor=compressor,
                )

        total_frames += n_frames
        episode_ends.append(total_frames)
        trajectory_ids.append(pkl_path.stem)
        prompts.append(episode_prompt)
        episode_fps.append(episode_fps_value)
        del ep_data
        gc.collect()

    if episode_ends:
        _create_dataset(zarr_meta, "episode_ends", np.array(episode_ends, dtype=np.int64), compressor)
        zarr_meta.attrs["trajectory_ids"] = trajectory_ids
        zarr_meta.attrs["prompts"] = prompts
        zarr_meta.attrs["fps"] = episode_fps
        zarr_meta.attrs["action_target"] = args.action_target

    print(f"\nDone. Episodes: {len(episode_ends)}, Frames: {total_frames}")
    print(f"Zarr saved to: {zarr_path}")


if __name__ == "__main__":
    tyro.cli(main)
