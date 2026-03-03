<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

openpi 是 Physical Intelligence 开源的机器人 VLA 模型仓库（π₀ / π₀-FAST / π₀.₅），本项目仅在 **Franka** 平台上实验。

## 安装

参考：`INSTALL.md`（主环境）、`examples/franka/INSTALL.md`（Franka 子环境 Python 3.9）

## 代码结构

| 层 | 路径 | 说明 |
|---|------|------|
| 模型 | `src/openpi/models/` | `model.py` 基础接口，`pi0.py` 流匹配实现 |
| 策略 | `src/openpi/policies/` | `policy.py` 推理封装，`*_policy.py` 平台适配 |
| 训练 | `src/openpi/training/config.py` | 所有配置定义在 `_CONFIGS` 列表 |
| 变换 | `src/openpi/transforms.py` | 数据管道：`RepackTransform` → `data_transforms` → 归一化 → `model_transforms` |

## Franka 配置

配置定义：`src/openpi/training/config.py`，命名模式 `pi05_franka_*_lora`

- 数据：`LeRobotFrankaDataConfig`，动作维度 8
- 模型：`pi05=True`，`action_horizon=30`，LoRA 微调
- 变换：`FrankaInputs` / `FrankaOutputs`

## Franka 机器人控制

frankx 库：`/home/mpi/workspace/yhx/frankx`（API 详见仓库 README）

约定：四元数 `(w,x,y,z)`，单位 [m]/[rad]。`real_env.py` 通过 `set_EE(constants.DEFAULT_EE_TRANSFORM)` 设置末端坐标系，修改前需确认与数据采集时一致。

**安全**：控制真实机器人前确认急停可用、速度/力矩限制合理、工作空间无碰撞风险。

## 工作流

权威流程文档：`examples/franka/评估_录制_转换_可视化.md`

快速参考（远程推理模式）：
```bash
# 1. 启动相机服务（Python 3.9 环境，在项目根目录执行）
python examples/franka/camera_service.py

# 2. 启动策略服务（主环境）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=<config_name> --policy.dir=<checkpoint_path>

# 3. 运行评估（Python 3.9 环境，在项目根目录执行）
python examples/franka/main.py \
    --args.remote-host localhost --args.remote-port 8000 \
    --args.control-mode impedance
```

关键文件：

| 文件 | 说明 |
|------|------|
| `main.py` | 评估入口，websocket 连接策略服务 |
| `real_env.py` | 机器人底层控制（frankx） |
| `camera_service.py` | IPC 相机服务（Python 3.9） |
| `collect_data.py` | 零重力示教连续数据采集（30Hz） |
| `collect_single_frame.py` | 零重力示教单帧数据采集（按 Enter 采集） |
| `convert_pkl_to_zarr.py` | PKL 转 Zarr |
| `camera_config.yaml` | 相机配置（设备序列号、分辨率、帧率） |
| `real_env_config.yaml` | 机器人环境配置（IP、限位、控制参数） |

## 数据格式

参考 Zarr 数据集：`eval_records/pi05_franka_position_control_lora/20260126/replay_buffer.zarr`

| 键 | 形状 | dtype | 说明 |
|----|------|-------|------|
| `robot_tcp_pose` | (N,8) | float32 | [x,y,z,qw,qx,qy,qz,gripper]，绝对位姿，gripper∈[0,1] |
| `robot_tcp_wrench` | (N,6) | float32 | 力/力矩 [fx,fy,fz,tx,ty,tz] |
| `action` | (N,8) | float32 | 同 tcp_pose 格式，绝对目标位姿 |
| `l500_camera_img` | (N,270,480,3) | uint8 | 底座相机 RGB |
| `d400_camera_img` | (N,240,320,3) | uint8 | 腕部相机 RGB |
| `xense1_camera_img` | (N,350,200,3) | uint8 | Xense 触觉相机 RGB |
| `xense1_marker3d` | (N,26,14,3) | float32 | Xense 3D 标记点 |
| `timestamp` | (N,) | float32 | 秒级时间戳 |
| `is_human_teaching` | (N,) | uint8 | 示教标记 |
| `meta/episode_ends` | (E,) | int64 | episode 结束索引 |

## RTC 模式

Real-Time Chunking 通过重叠推理与执行降低控制延迟：
- `--args.rtc`：启用
- `--args.rtc-inference-delay N`：推理期间执行旧 chunk 动作数（默认 3）
- `--args.rtc-execute-horizon N`：每轮执行总动作数（默认 5）
