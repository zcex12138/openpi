"""
将 Zarr 格式数据集转换为 LeRobot v2.0 数据集格式（兼容 LeRobot >= v2.1）。

Usage:
    uv run examples/convert_zarr_to_lerobot.py \
        --zarr_path /path/to/replay_buffer.zarr \
        --repo_id your_name/dataset_name
"""

import shutil

import numpy as np
import tqdm
import tyro
import zarr
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main(
    zarr_path: str = "/home/mpi/workspace/yhx/openpi/eval_records/pi05_franka_position_control_lora/20260126/replay_buffer.zarr",
    repo_id: str = "2026_0126_pi05_franka_cola_lerobot_v2.1",
    task: str = "open the can with the screwdriver",
    fps: int = 30,
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
    push_to_hub: bool = False,
    drop_frames_after_human_teaching: int = 30,
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
    is_human_teaching = z["data/is_human_teaching"][:]

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

    selected_episodes = list(range(len(episode_ends)))

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
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    # 转换数据
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    total_frames = 0
    dropped_frames = 0
    for ep_idx in tqdm.tqdm(selected_episodes, desc="Converting episodes"):
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]

        # 批量预读取整个 episode 的数据（优化 I/O 性能）
        d400_batch = d400_imgs[start_idx:end_idx]
        l500_batch = l500_imgs[start_idx:end_idx]
        tcp_pose_batch = tcp_pose[start_idx:end_idx]
        tcp_wrench_batch = tcp_wrench[start_idx:end_idx]
        actions_batch = actions[start_idx:end_idx]
        teaching_batch = is_human_teaching[start_idx:end_idx]

        # 计算每帧的有效掩码：仅在首次切换到示教模式后丢弃 N 帧
        ep_len = len(d400_batch)
        valid_mask = np.ones(ep_len, dtype=bool)
        if drop_frames_after_human_teaching > 0:
            first_teaching_idx = None
            for i in range(ep_len):
                if teaching_batch[i]:
                    first_teaching_idx = i
                    break
            if first_teaching_idx is not None:
                drop_end = min(first_teaching_idx + 1 + drop_frames_after_human_teaching, ep_len)
                valid_mask[first_teaching_idx + 1 : drop_end] = False

        total_frames += ep_len
        dropped_frames += int(np.sum(~valid_mask))

        frame_added = False
        for i in range(ep_len):
            if not valid_mask[i]:
                continue
            state = np.concatenate([tcp_pose_batch[i], tcp_wrench_batch[i]]).astype(np.float32)

            dataset.add_frame(
                {
                    "observation.images.d400": d400_batch[i],
                    "observation.images.l500": l500_batch[i],
                    "observation.state": state,
                    "action": actions_batch[i].astype(np.float32),
                    "task": task,
                }
            )
            frame_added = True

        if frame_added:
            dataset.save_episode()

    if drop_frames_after_human_teaching > 0:
        print(
            f"\n帧过滤统计: 总帧数={total_frames}, 丢弃帧数={dropped_frames}, 保留帧数={total_frames - dropped_frames}"
        )

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
