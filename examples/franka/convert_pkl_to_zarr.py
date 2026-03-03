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

import numpy as np
import tyro
import zarr

_DEFAULT_RECORDS = "/home/mpi/workspace/yhx/openpi/demo_records/"
_DEFAULT_RECORDS = "/home/mpi/workspace/yhx/openpi/eval_records/pi05_franka_cola_lora/20260207"
_DEFAULT_RECORDS = "data/dataset/dataset_zarr/20260126_失败单帧"


@dataclasses.dataclass
class Args:
    records: str = _DEFAULT_RECORDS
    output_dir: str | None = None
    temporal_downsample_ratio: int = 1
    drop_frames_after_human_teaching: int = 40


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


def _process_episode(episode: dict) -> dict[str, np.ndarray] | None:
    """Process single episode, return arrays dict."""
    frames = episode.get("frames", [])
    if not frames:
        return None

    n = len(frames)
    l500_list, d400_list, xense1_list = [], [], []
    xense1_marker3d_list = []
    timestamp_arr = np.zeros(n, dtype=np.float32)
    tcp_pose_arr = np.zeros((n, 8), dtype=np.float32)
    tcp_wrench_arr = np.zeros((n, 6), dtype=np.float32)
    action_arr = np.zeros((n, 8), dtype=np.float32)
    is_human_teaching_arr = np.zeros(n, dtype=np.uint8)
    has_action = False

    for i, frame in enumerate(frames):
        images = frame.get("images", {})
        l500_list.append(np.asarray(images.get("l500", np.zeros((224, 224, 3), dtype=np.uint8)), dtype=np.uint8))
        d400_list.append(np.asarray(images.get("d400", np.zeros((224, 224, 3), dtype=np.uint8)), dtype=np.uint8))
        xense1_list.append(np.asarray(images.get("xense_1", np.zeros((224, 224, 3), dtype=np.uint8)), dtype=np.uint8))

        marker3d = frame.get("marker3d", {})
        xense1_m3d = marker3d.get("xense_1", np.zeros((0, 0, 3), dtype=np.float32))
        xense1_marker3d_list.append(np.asarray(xense1_m3d, dtype=np.float32))

        timestamp_arr[i] = float(frame.get("timestamp", i / 30.0))

        tcp_pose = np.asarray(frame.get("tcp_pose", np.zeros(7)), dtype=np.float32)
        gripper_raw = frame.get("gripper", 0.0)
        gripper = float(np.asarray(gripper_raw).flat[0])
        wrench = np.asarray(frame.get("wrench", np.zeros(6)), dtype=np.float32)

        tcp_pose_arr[i] = np.concatenate([tcp_pose, [gripper]])
        tcp_wrench_arr[i] = wrench
        is_human_teaching_arr[i] = 1 if frame.get("is_human_teaching", False) else 0
        frame_action = frame.get("action")
        if frame_action is not None:
            action_vec = np.asarray(frame_action, dtype=np.float32).reshape(-1)
            if action_vec.size == 8:
                action_arr[i] = action_vec
                has_action = True

    if not has_action:
        # action = next frame's pose (shift by 1)
        action_arr[:-1] = tcp_pose_arr[1:]
        action_arr[-1] = tcp_pose_arr[-1]

    result = {
        "timestamp": timestamp_arr,
        "l500_camera_img": np.stack(l500_list),
        "d400_camera_img": np.stack(d400_list),
        "robot_tcp_pose": tcp_pose_arr,
        "robot_tcp_wrench": tcp_wrench_arr,
        "action": action_arr,
        "is_human_teaching": is_human_teaching_arr,
    }

    # xense1 image (only if non-empty)
    xense1_arr = np.stack(xense1_list)
    if xense1_arr.size > 0 and xense1_arr.max() > 0:
        result["xense1_camera_img"] = xense1_arr

    # xense1 marker3d (only if shapes consistent and non-empty)
    if xense1_marker3d_list and xense1_marker3d_list[0].size > 0:
        shapes = [m.shape for m in xense1_marker3d_list]
        if len(set(shapes)) == 1:
            result["xense1_marker3d"] = np.stack(xense1_marker3d_list)

    return result


def _drop_after_teaching(data: dict[str, np.ndarray], n_drop: int) -> dict[str, np.ndarray]:
    """Drop n_drop frames immediately after the first human_teaching frame."""
    teaching = data["is_human_teaching"]
    ep_len = len(teaching)
    mask = np.ones(ep_len, dtype=bool)
    first = np.argmax(teaching)  # 0 if none found
    if teaching[first]:
        drop_end = min(first + 1 + n_drop, ep_len)
        mask[first + 1 : drop_end] = False
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
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    episode_ends: list[int] = []
    total_frames = 0
    initialized = False

    for idx, pkl_path in enumerate(pkl_files):
        print(f"[{idx + 1}/{len(pkl_files)}] {pkl_path.name}")
        episode = _load_pkl(pkl_path)
        if episode is None:
            continue

        ep_data = _process_episode(episode)
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

        if not initialized:
            l500_shape = ep_data["l500_camera_img"].shape[1:]
            d400_shape = ep_data["d400_camera_img"].shape[1:]
            zarr_data.create_dataset(
                "timestamp", data=ep_data["timestamp"], chunks=(10000,), dtype="float32", compressor=compressor
            )
            zarr_data.create_dataset(
                "l500_camera_img",
                data=ep_data["l500_camera_img"],
                chunks=(100,) + l500_shape,
                dtype="uint8",
                compressor=compressor,
            )
            zarr_data.create_dataset(
                "d400_camera_img",
                data=ep_data["d400_camera_img"],
                chunks=(100,) + d400_shape,
                dtype="uint8",
                compressor=compressor,
            )
            zarr_data.create_dataset(
                "robot_tcp_pose",
                data=ep_data["robot_tcp_pose"],
                chunks=(10000, 8),
                dtype="float32",
                compressor=compressor,
            )
            zarr_data.create_dataset(
                "robot_tcp_wrench",
                data=ep_data["robot_tcp_wrench"],
                chunks=(10000, 6),
                dtype="float32",
                compressor=compressor,
            )
            zarr_data.create_dataset(
                "action", data=ep_data["action"], chunks=(10000, 8), dtype="float32", compressor=compressor
            )
            zarr_data.create_dataset(
                "is_human_teaching",
                data=ep_data["is_human_teaching"],
                chunks=(10000,),
                dtype="uint8",
                compressor=compressor,
            )
            if "xense1_camera_img" in ep_data:
                xense1_shape = ep_data["xense1_camera_img"].shape[1:]
                zarr_data.create_dataset(
                    "xense1_camera_img",
                    data=ep_data["xense1_camera_img"],
                    chunks=(100,) + xense1_shape,
                    dtype="uint8",
                    compressor=compressor,
                )
            if "xense1_marker3d" in ep_data:
                m3d_shape = ep_data["xense1_marker3d"].shape[1:]
                zarr_data.create_dataset(
                    "xense1_marker3d",
                    data=ep_data["xense1_marker3d"],
                    chunks=(100,) + m3d_shape,
                    dtype="float32",
                    compressor=compressor,
                )
            initialized = True
        else:
            zarr_data["timestamp"].append(ep_data["timestamp"])
            zarr_data["l500_camera_img"].append(ep_data["l500_camera_img"])
            zarr_data["d400_camera_img"].append(ep_data["d400_camera_img"])
            zarr_data["robot_tcp_pose"].append(ep_data["robot_tcp_pose"])
            zarr_data["robot_tcp_wrench"].append(ep_data["robot_tcp_wrench"])
            zarr_data["action"].append(ep_data["action"])
            zarr_data["is_human_teaching"].append(ep_data["is_human_teaching"])
            if "xense1_camera_img" in ep_data and "xense1_camera_img" in zarr_data:
                zarr_data["xense1_camera_img"].append(ep_data["xense1_camera_img"])
            if "xense1_marker3d" in ep_data and "xense1_marker3d" in zarr_data:
                zarr_data["xense1_marker3d"].append(ep_data["xense1_marker3d"])

        total_frames += n_frames
        episode_ends.append(total_frames)
        del ep_data
        gc.collect()

    if episode_ends:
        zarr_meta.create_dataset(
            "episode_ends",
            data=np.array(episode_ends, dtype=np.int64),
            chunks=(10000,),
            dtype="int64",
            compressor=compressor,
        )

    print(f"\nDone. Episodes: {len(episode_ends)}, Frames: {total_frames}")
    print(f"Zarr saved to: {zarr_path}")


if __name__ == "__main__":
    tyro.cli(main)
