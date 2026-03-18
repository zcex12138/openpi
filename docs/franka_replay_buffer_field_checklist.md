# Franka replay_buffer.zarr 字段最小保留清单

适用范围：
- 仅基于当前仓库里的训练与转换链路。
- 不覆盖用户自定义脚本、离线分析脚本或纯可视化用途。
- 结论针对 `/home/mpi/workspace/yhx/openpi/eval_records/replay_buffer.zarr` 与当前代码实现。

## 1. residual 训练最小保留字段

残差训练实际只依赖以下字段：

- `data/robot_tcp_pose`
- `data/base_action`
- `data/corrected_action`
- `data/corrected_action_valid`
- `data/is_human_teaching`
- `meta/episode_ends`

说明：
- `corrected_action_valid` 不是 `is_human_teaching` 的重复字段。
- 当前 residual 采样逻辑使用 `is_human_teaching & corrected_action_valid` 定义 correction interval。
- 以上 Franka canonical pose/action 字段当前统一使用 `pose10 = [xyz, r6d, gripper]`。

相关代码：
- `src/residual_policy/dataset.py`

## 2. 主策略训练最小保留字段

如果使用当前 `Zarr -> LeRobot -> 训练` 链路，转换脚本实际读取以下字段：

基础必需：
- `data/action`
- `data/d400_camera_img`
- `data/l500_camera_img`
- `data/robot_tcp_pose`
- `data/robot_tcp_wrench`
- `meta/episode_ends`

条件必需：
- `data/is_human_teaching`
  - 仅当 `drop_frames_after_human_teaching > 0` 时需要。
  - 当前 `examples/convert_zarr_to_lerobot_v2.0.py` 默认值为 `40`，因此按默认跑法需要保留。
- `data/xense1_marker3d`
  - 仅 tactile 训练需要。

说明：
- 普通 Franka 配置直接使用数据集里的 `action`。
- `LeRobotFrankaDataConfig` / `LeRobotFrankaTactileDataConfig` 在训练阶段会从 `observation.state` 生成 action target，不直接依赖数据集里的 `action`。
- 但当前 Zarr 转 LeRobot 脚本仍然先读取 `data/action`，所以在“不改转换脚本”的前提下，`action` 仍需保留。

相关代码：
- `examples/convert_zarr_to_lerobot_v2.0.py`
- `src/openpi/training/config.py`

## 3. 同时兼容 residual 训练与当前主训练转换的推荐最小集合

推荐保留：

- `data/action`
- `data/base_action`
- `data/corrected_action`
- `data/corrected_action_valid`
- `data/is_human_teaching`
- `data/d400_camera_img`
- `data/l500_camera_img`
- `data/robot_tcp_pose`
- `data/robot_tcp_wrench`
- `meta/episode_ends`

可选保留：
- `data/xense1_camera_img`
  - 仅可视化、回放或排查 xense 相机问题时需要；当前转换脚本在源 PKL 存在该字段时会一并导出。
- `data/xense1_marker3d`
  - 仅 tactile 训练需要。

## 4. 当前可安全裁掉的字段

在“保留 residual 训练 + 当前主训练转换能力”的前提下，可裁掉：

- `data/timestamp`
- `data/timestamp_ns`
- `data/control_timestamp`
- `data/seq`
- `data/frame_index`
- `data/teaching_segment_id`
- `data/teaching_step`
- `data/robot_tcp_velocity`
- `data/xense1_camera_img`

这些字段当前不进入训练张量，主要用于运行时、录制、索引或追踪。
其中 `data/xense1_camera_img` 虽然当前转换脚本会保留，但对现有训练链路仍不是必需字段。
其中 `data/executed_action` 不属于 canonical 训练张量，但它仍是 Franka env-only 8D quaternion 执行边界的调试/回放字段，不建议在需要执行 provenance 时删除。

## 5. 这份具体数据中的明显冗余项

针对 `/home/mpi/workspace/yhx/openpi/eval_records/replay_buffer.zarr` 的实际检查结果：

- `timestamp`
  - 可由 `control_timestamp - 每个 episode 起点` 完整恢复。
- `frame_index`
  - 每个 episode 内等于 `0..T-1`。
- `teaching_step`
  - 每段 teaching 内等于局部 `0..len-1`。
- `teaching_segment_id`
  - 当前每个 episode 只有一段 teaching，非 teaching 为 `-1`，teaching 全为 `0`，信息量很低。
- `action`
  - 在这份数据里与 `corrected_action` 完全相同。

注意：
- `action == corrected_action` 是这份具体数据的结果，不是通用 schema 约束。
- 如果后续导出策略变化，这一条可能不再成立。
