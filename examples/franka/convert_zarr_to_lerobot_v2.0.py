"""
将 Zarr 格式数据集转换为 LeRobot v2.0 数据集格式（兼容 LeRobot >= v2.1）。

Usage:
    uv run examples/convert_zarr_to_lerobot.py \
        --zarr_path /path/to/replay_buffer.zarr \
        --annotations_path /path/to/annotations.json \
        --repo_id your_name/dataset_name
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import tqdm
import tyro
import zarr
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main(
    zarr_path: str = "/home/mpi/workspace/yhx/openpi/dataset/2026_0105_224/replay_buffer.zarr",
    annotations_path: str = "/home/mpi/workspace/yhx/openpi/dataset/2026_0105_224/annotations.json",
    repo_id: str = "local/single_arm_screwdriver",
    task: str = "使用螺丝刀将可乐罐撬开",
    fps: int = 30,
    success_only: bool = True,
    push_to_hub: bool = False,
):
    # 清理已存在的数据集
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    # 打开 Zarr 数据集
    z = zarr.open(zarr_path, mode="r")

    # 获取数据 (嵌套结构: data/ 和 meta/)
    episode_ends = z["meta/episode_ends"][:]
    actions = z["data/action"][:]

    # 图像数据
    d400_imgs = z["data/d400_camera_img"]
    l500_imgs = z["data/l500_camera_img"]

    # 状态数据: TCP pose (8维) + wrench (6维) = 14维
    tcp_pose = z["data/robot_tcp_pose"][:]  # (N, 8)
    tcp_wrench = z["data/robot_tcp_wrench"][:]  # (N, 6)

    # 获取图像尺寸
    d400_img_shape = d400_imgs.shape[1:]  # (H, W, C)
    l500_img_shape = l500_imgs.shape[1:]  # (H, W, C)

    print("数据集信息:")
    print(f"  总帧数: {len(actions)}")
    print(f"  Episode 数: {len(episode_ends)}")
    print(f"  D400 图像尺寸: {d400_img_shape}")
    print(f"  L500 图像尺寸: {l500_img_shape}")
    print(f"  Action 维度: {actions.shape[1]}")
    print(f"  State 维度: {tcp_pose.shape[1] + tcp_wrench.shape[1]}")

    annotations = None
    selected_episodes = list(range(len(episode_ends)))

    if success_only:
        annotations_path = Path(annotations_path)
        if not annotations_path.exists():
            raise FileNotFoundError(f"annotations.json 未找到: {annotations_path}")

        annotations = json.loads(annotations_path.read_text())
        annotation_map: dict[int, dict] = {}

        if isinstance(annotations, list):
            annotation_map = {idx: ann for idx, ann in enumerate(annotations)}
        elif isinstance(annotations, dict):
            for key, ann in annotations.items():
                if isinstance(key, int):
                    annotation_map[key] = ann
                    continue
                if not isinstance(key, str):
                    continue
                if key.isdigit():
                    annotation_map[int(key)] = ann
                    continue
                match = re.search(r"\d+", key)
                if match:
                    annotation_map[int(match.group())] = ann
        else:
            raise ValueError("annotations.json 的格式应为 list 或 dict")

        selected_episodes = []
        missing_annotations = []
        for ep_idx in range(len(episode_ends)):
            ann = annotation_map.get(ep_idx)
            if ann is None:
                missing_annotations.append(ep_idx)
                continue
            if ann.get("is_success") is True:
                selected_episodes.append(ep_idx)

        if missing_annotations:
            print(f"警告: 有 {len(missing_annotations)} 个 episode 缺少标注，将被跳过。")
        print(f"成功轨迹数: {len(selected_episodes)} / {len(episode_ends)}")
        if not selected_episodes:
            raise ValueError("未找到标注为成功的轨迹，无法生成数据集。")

    # 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="single_arm",
        fps=fps,
        features={
            "observation.images.d400": {
                "dtype": "image",
                "shape": d400_img_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.l500": {
                "dtype": "image",
                "shape": l500_img_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (tcp_pose.shape[1] + tcp_wrench.shape[1],),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (actions.shape[1],),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 转换数据
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    for ep_idx in tqdm.tqdm(selected_episodes, desc="Converting episodes"):
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]

        for i in range(start_idx, end_idx):
            # 拼接状态: pose + wrench
            state = np.concatenate([tcp_pose[i], tcp_wrench[i]]).astype(np.float32)

            dataset.add_frame({
                "observation.images.d400": d400_imgs[i],
                "observation.images.l500": l500_imgs[i],
                "observation.state": state,
                "action": actions[i].astype(np.float32),
                "task": task,
            })

        dataset.save_episode()

    # LeRobot >= v2.1 writes metadata/stats incrementally in save_episode, so consolidate is a no-op.
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
    elif hasattr(dataset, "_wait_image_writer"):
        dataset._wait_image_writer()
    print(f"\n数据集已保存到: {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["single_arm", "screwdriver"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
