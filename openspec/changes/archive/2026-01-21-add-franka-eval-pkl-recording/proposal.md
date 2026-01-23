# Change: Add Franka Evaluation PKL Recording

## Why
评估过程需要同时采集数据用于二次训练，现有流程仅支持策略记录（policy_records），未覆盖多相机/Xense marker3d 与 TCP 速度数据，也缺少可直接转 LeRobot 的中间格式。

## What Changes
- 在 Franka 评估流程中新增可选的 **每条轨迹 pkl 录制**（episode 级文件），录制与评估解耦异步运行，录制帧率可独立配置。
- 在 `FrankaEnvironment` 中新增 **每帧数据获取接口**（相机帧+状态+时间戳），评估侧以异步方式调用并投递到录制队列/进程。
- 相机服务扩展为输出 **L500、D400、Xense1、Xense2** 的 RGB 图像与 **Xense1/Xense2 marker3d**。
- Xense marker3d 获取参考 `/home/mpi/workspace/yhx/reactive_diffusion_policy/reactive_diffusion_policy/real_world/simple_camera/simple_xense_camera.py` 的 API 与数据形状。
- TCP 6D 速度优先读取控制回路缓存的 `robot_state.O_dP_EE_c`，避免在阻抗/笛卡尔控制中直接调用 `Robot.get_state()`。
- 新增独立录制脚本，可在新进程中运行录制逻辑，与评估分离。
- 新增 pkl→LeRobot 的转换脚本，**保持现有 LeRobot key 格式**，动作字段写零。
- 增加 CLI 开关与默认输出目录，便于评估或独立录制时一键启动。

## Impact
- Affected specs: `franka-eval-recording`（新增）
- Affected code:
  - `examples/franka/main.py`
  - `examples/franka/env.py`
  - `examples/franka/camera_service.py`
  - `examples/franka/camera_client.py`
  - `examples/franka/camera_config.yaml`
  - `packages/openpi-client/src/openpi_client/runtime/*`（新增 Subscriber 或放置 recorder）
  - `examples/franka/convert_pkl_to_lerobot.py`（新增）
  - `examples/franka/`（新增独立录制脚本）
  - `examples/franka/README.md`（如需更新使用说明）
