"""Convert Franka PKL recordings to LeRobot v3.0 dataset format."""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
import pickle
import shutil

import numpy as np
import tyro

_DEFAULT_RECORDS = "/home/mpi/workspace/yhx/openpi/eval_records/test_config/episode_000.pkl"


@dataclasses.dataclass
class Args:
    """Command-line arguments for PKL to LeRobot conversion."""

    records: str = _DEFAULT_RECORDS
    repo_id: str | None = None
    fps: float | None = None
    task_index: int = 0
    push_to_hub: bool = False
    image_writer_threads: int = 10
    image_writer_processes: int = 5


def _collect_pkl_files(records_path: Path) -> list[Path]:
    if records_path.is_file():
        return [records_path]
    if records_path.is_dir():
        candidates = sorted(records_path.glob("episode_*.pkl"))
        if candidates:
            return candidates
    raise FileNotFoundError(f"No episode_*.pkl found under {records_path}")


def _build_features(sample_frame: dict) -> dict:
    l500 = np.asarray(sample_frame["images"]["l500"])
    d400 = np.asarray(sample_frame["images"]["d400"])

    return {
        "observation.images.l500": {
            "dtype": "image",
            "shape": l500.shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.d400": {
            "dtype": "image",
            "shape": d400.shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["action"],
        },
    }


def _import_lerobot() -> tuple[Path, type]:
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "未检测到 lerobot v3.0 相关模块，请安装 lerobot>=0.4.0（或 main 分支）。"
        ) from exc
    return HF_LEROBOT_HOME, LeRobotDataset


def _create_dataset(LeRobotDataset: type, **kwargs):
    create_sig = inspect.signature(LeRobotDataset.create)
    filtered = {k: v for k, v in kwargs.items() if k in create_sig.parameters}
    return LeRobotDataset.create(**filtered)


def main(args: Args) -> None:
    HF_LEROBOT_HOME, LeRobotDataset = _import_lerobot()
    records_path = Path(args.records)
    pkl_files = _collect_pkl_files(records_path)

    with pkl_files[0].open("rb") as handle:
        first_episode = pickle.load(handle)

    fps = float(args.fps) if args.fps is not None else float(first_episode.get("fps", 30.0))

    repo_id = args.repo_id
    if repo_id is None:
        base_name = records_path.parent.name if records_path.is_file() else records_path.name
        repo_id = f"franka_eval_{base_name}"

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    frames = first_episode.get("frames", [])
    if not frames:
        raise ValueError("First PKL file contains no frames")

    dataset = _create_dataset(
        LeRobotDataset,
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features=_build_features(frames[0]),
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    global_index = 0
    for pkl_path in pkl_files:
        with pkl_path.open("rb") as handle:
            episode = pickle.load(handle)
        prompt = episode.get("prompt", "")
        for frame in episode.get("frames", []):
            tcp_pose = np.asarray(frame.get("tcp_pose", np.zeros(7, dtype=np.float32)), dtype=np.float32)
            gripper = np.asarray(frame.get("gripper", np.zeros(1, dtype=np.float32)), dtype=np.float32)
            wrench = np.asarray(frame.get("wrench", np.zeros(6, dtype=np.float32)), dtype=np.float32)
            state = np.concatenate([tcp_pose, gripper, wrench]).astype(np.float32)

            dataset.add_frame(
                {
                    "observation.images.l500": np.asarray(frame["images"]["l500"], dtype=np.uint8),
                    "observation.images.d400": np.asarray(frame["images"]["d400"], dtype=np.uint8),
                    "observation.state": state,
                    "action": np.zeros((8,), dtype=np.float32),
                    "task": prompt,
                }
            )
            global_index += 1

        dataset.save_episode()

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "_wait_image_writer"):
        dataset._wait_image_writer()

    print(f"\nDataset saved to: {output_path}")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["franka", "panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
