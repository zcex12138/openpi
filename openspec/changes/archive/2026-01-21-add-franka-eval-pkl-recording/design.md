## Context
Franka 评估流程当前仅依赖 L500/D400 两路图像与 14D state，且录制机制集中在 PolicyRecorder（policy_records），无法满足评估阶段同步采集多相机与 marker3d/TCP 速度数据的需求。

## Goals / Non-Goals
- Goals:
  - 评估时可选录制每条轨迹的 pkl 文件（episode 级）。
  - 采集 L500/D400/Xense1/Xense2 图像（原始分辨率 2×降采样），Xense1/Xense2 marker3d，TCP pose、TCP 速度（6D）、6D 力。
  - 提供 pkl→LeRobot 的转换脚本，并保持现有 LeRobot key 格式。
- Non-Goals:
  - 不改变策略推理的输入/输出结构与预处理流程。
  - 不在 LeRobot 里新增额外 observation keys（先保持现有格式）。
  - 不引入新的训练管线或复杂数据管理系统。

## Decisions
- **录制入口**：使用 runtime `Subscriber` 实现评估侧录制，避免重新引入已移除的 ClientRecorder。
- **异步录制**：评估线程仅做轻量采样/入队，录制线程/进程按独立 fps 写入 pkl；评估不因序列化或 I/O 阻塞。
- **独立录制脚本**：提供单独脚本启动录制进程，复用相机/状态来源，允许评估与录制分离运行。
- **pkl 文件结构**（每个 episode 一个文件）：
  ```python
  {
    "version": 1,
    "config_name": str,
    "episode_index": int,
    "prompt": str,
    "fps": float,
    "frames": [
      {
        "timestamp": float,
        "frame_index": int,
        "images": {
          "l500": uint8[H/2, W/2, 3],
          "d400": uint8[H/2, W/2, 3],
          "xense_1": uint8[H/2, W/2, 3],
          "xense_2": uint8[H/2, W/2, 3],
        },
        "marker3d": {
          "xense_1": float32[N, M, 3],
          "xense_2": float32[N, M, 3],
        },
        "tcp_pose": float32[7],
        "tcp_velocity": float32[6],
        "wrench": float32[6],
        "gripper": float32[1],
      },
      ...
    ]
  }
  ```
  - timestamp 采用 `frame_index / record_fps`，record_fps 可配置并与评估频率解耦（默认 30Hz）。
- **TCP 速度来源**：优先从控制回路缓存的 `robot_state.O_dP_EE_c` 读取 6D 速度，
  避免在阻抗/笛卡尔控制中直接调用 `Robot.get_state()`。
  当控制未启动且无缓存时，才考虑使用 `Robot.get_state()` 兜底。
- **Xense marker3d 获取**：参考 `/home/mpi/workspace/yhx/reactive_diffusion_policy/reactive_diffusion_policy/real_world/simple_camera/simple_xense_camera.py`，
  使用 `Sensor.selectSensorInfo(Rectify, Depth, Marker2D)` 获取 `rgb_frame`、`depth_frame`、`marker2d`，
  再用深度补齐得到 `marker3d`（marker2d 形状 `(26, 14, 2)`，depth 形状 `(700, 400)`，marker3d 形状 `(26, 14, 3)`）。
- **下采样**：录制时对所有图像做 2×降采样（原始分辨率 → 1/2）。推理输入仍使用现有 224×224 处理链路。
- **LeRobot 转换**：转换脚本只写当前已有 key：
  - `observation.images.l500`, `observation.images.d400`, `observation.state`, `action`, `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`
  - `action` 使用 8D 全零以维持 schema；`observation.state` 由 `tcp_pose + gripper + wrench` 组合。
  - Xense 图像、marker3d、tcp_velocity 暂不写入 LeRobot（后续可扩展）。

## Risks / Trade-offs
- Xense marker3d 输出格式依赖 SDK，可能需要额外适配；若获取失败，需要写空数组并记录日志。
- 多相机/marker 同步存在时序偏差；本阶段仅保证“同一录制步取最新帧”。
- pkl 逐步追加可能增大 I/O 开销；保持单 episode 文件以降低碎片。
- 异步队列可能积压导致丢帧或内存增长，需要设置队列上限与丢帧策略。

## Migration Plan
1. 新增录制开关与输出目录；默认关闭，避免影响现有评估流程。
2. 先落地 pkl 录制，再提供转换脚本生成 LeRobot 数据集。

## Open Questions
- 暂无。
