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

openpi 是 Physical Intelligence 开源的机器人 VLA 仓库（π₀ / π₀-FAST / π₀.₅）。当前这个 fork 的主线集中在 Franka 实机评估、PKL/Zarr 数据链路，以及独立 residual policy 训练/推理。

## 先看哪些文件

- 仓库与 Franka 总览：`CLAUDE.md`
- Franka 实操流程：`examples/franka/使用方法说明.md`
- Franka 运行时参数：`examples/franka/real_env_config.yaml`
- replay_buffer 字段裁剪边界：`docs/franka_replay_buffer_field_checklist.md`
- residual 训练模板：`scripts/train_residual_policy.example.yaml`
- 如果文档与代码冲突：以 `examples/franka/main.py`、`examples/franka/convert_pkl_to_zarr.py`、`scripts/train_residual_policy.py`、`src/residual_policy/` 当前实现为准

## 近期已落地的变化

- Franka canonical state/action 已统一为 `pose10 = [x, y, z, r1, r2, r3, r4, r5, r6, gripper]`
- `pose8 = [x, y, z, qw, qx, qy, qz, gripper]` 只在真实执行边界和 `executed_action` 调试字段中保留
- `examples/franka/main.py` 现在把“策略来源”和“执行模式”拆开：
  - policy source: `service` / `local`
  - execution mode: `sync` / `rtc` / `cr_dagger_baseline`
- residual policy 已独立到 `src/residual_policy/`；训练入口是 `scripts/train_residual_policy.py`，运行时可在 Franka evaluation 中按 step 叠加
- `examples/franka/convert_pkl_to_zarr.py` 现在会导出训练所需字段，保留 `base_action`、`corrected_action`、`corrected_action_valid`，并支持 `action_target`
- 通用 YAML 辅助在 `src/openpi/shared/yaml_config.py`；checkpoint 加载与 warmup 入口在 `src/openpi/serving/policy_loading.py`

## 安装

- 主环境参考根目录 `INSTALL.md`
- Franka 子环境参考 `examples/franka/INSTALL.md`
- `camera_service.py` / Franka 运行链路通常依赖 Python 3.9 子环境
- `serve_policy.py` / 训练链路使用项目主环境

## 代码结构

| 层 | 路径 | 说明 |
|---|---|---|
| 模型 | `src/openpi/models/` | π₀ / π₀.₅ / π₀-FAST 模型定义 |
| 策略 | `src/openpi/policies/` | 推理封装与平台适配；`franka_policy.py` 负责 Franka 输入输出变换 |
| 服务 | `src/openpi/serving/` | checkpoint policy 加载、warmup、websocket 服务 |
| 训练配置 | `src/openpi/training/config.py` | 所有 train config；Franka LoRA config 定义在这里 |
| residual | `src/residual_policy/` | `action_repr`、`dataset`、`inference`、`trainer` |
| 客户端执行 | `packages/openpi-client/src/openpi_client/` | `ActionChunkBroker`、`RealTimeChunkBroker`、`CrDaggerChunkBroker`、runtime/agent |
| Franka 实机 | `examples/franka/` | 评估、录制、转换、可视化、相机/机器人运行时 |

## Franka 训练与动作表示

- Franka 训练配置定义在 `src/openpi/training/config.py`
- 当前常用配置：
  - `pi05_franka_cola_lora`
  - `pi05_franka_cola_relative_lora`
  - `pi05_franka_tactile_lora`
- `LeRobotFrankaDataConfig` / `LeRobotFrankaTactileDataConfig` 通过 `ShiftedStateToAction` 从未来 state 生成 action target；当前 `state_to_action_shift=10`
- 训练内部会把 quaternion 转成 `rotate6d`，公开 Franka canonical 动作维度是 10 维 `pose10`
- tactile 训练额外读取 `observation.tactile.xense1_marker3d`
- Franka π₀.₅ LoRA 配置当前 `action_horizon=30`

## Franka 评估 / 推理工作流

当前真实入口：
1. 相机服务：`python examples/franka/camera_service.py`
2. 策略服务：`uv run scripts/serve_policy.py ...`
3. 评估：`python examples/franka/main.py ...`

远程推理常用命令：

```bash
python examples/franka/main.py \
    --args.remote-host localhost \
    --args.remote-port 8000 \
    --args.execution-mode rtc
```

本地 checkpoint 常用命令：

```bash
python examples/franka/main.py \
    --args.checkpoint-dir data/checkpoints/<run>/<step> \
    --args.config pi05_franka_cola_relative_lora
```

说明：
- `examples/franka/main.py` 当前通过 Tyro 暴露参数，CLI 以 `--args.*` 形式传入
- 当前真正会被解析的 YAML 是 `examples/franka/real_env_config.yaml`
- `examples/franka/eval_config.example.yaml` 已存在，可作为整理评估参数的模板；在 `main.py` 明确接入前，不要把它当成唯一权威入口
- `scripts/serve_policy.py` 启动前会 warmup policy；websocket server 由 `src/openpi/serving/websocket_policy_server.py` 提供，并包含 `/healthz`
- `observation["__openpi"]` 是运行时元数据通道，`PolicyAgent` 通过它读取 `reset_policy`、`manual_override_action` 等控制信号

## 执行模式

三种模式：
- `sync`：`ActionChunkBroker`
- `rtc`：`RealTimeChunkBroker`
- `cr_dagger_baseline`：`CrDaggerChunkBroker`

约束：
- 推荐使用 `--args.execution-mode` 或 `real_env_config.yaml: execution.mode`
- `--args.rtc` 只保留为 legacy shorthand
- 当前 `examples/franka/main.py` 在 RTC 中固定 `use_action_prefix=False`
- RTC / CR-Dagger 具体参数来自 `real_env_config.yaml` 的 `rtc` / `cr_dagger` 段，CLI 可覆盖

## Residual Policy

训练入口：

```bash
uv run scripts/train_residual_policy.py --config scripts/train_residual_policy.example.yaml
```

当前实现要点：
- 训练数据直接读取 Zarr，不走 openpi 主训练框架
- 输入特征是 `state_pose10 + base_action_pose10`
- 监督目标是 `corrected_action` 相对 `base_action` 的 residual `pose10`
- 采样逻辑是 CR-Dagger 风格，核心字段为 `is_human_teaching & corrected_action_valid`
- 运行时在 `examples/franka/main.py` 里通过 `FrankaPolicyPose10Wrapper` + `FrankaResidualStepPolicy` 叠加
- 残差限幅参数：
  - `scale`
  - `translation_cap_m`
  - `rotation_cap_rad`
  - `gripper_cap`
- 当前默认 `apply_gripper_delta=False`

## 数据与导出

`examples/franka/convert_pkl_to_zarr.py` 当前行为：
- `robot_tcp_pose` / `action` / `base_action` / `corrected_action` 使用 `pose10`
- `executed_action` 保留 `pose8`，主要用于执行 provenance / debug
- `action_target` 支持 `auto | executed | base | corrected`
- 导出字段固定为当前训练链路需要的最小集合，并在存在时保留 `xense1_marker3d`
- `meta.attrs` 当前写入：
  - `trajectory_ids`
  - `prompts`
  - `fps`
  - `action_target`
  - `schema_version=franka_replay_buffer_v2`

字段最小保留集合和可裁剪项，以 `docs/franka_replay_buffer_field_checklist.md` 为准。

## Franka 控制与安全

- frankx 仓库：`/home/mpi/workspace/yhx/frankx`
- 四元数顺序统一按 `(w, x, y, z)`
- `examples/franka/real_env.py` 会通过 `set_EE(...)` / YAML 配置应用末端变换，修改前必须确认与采集标定一致
- `translation_scale` / `rotation_scale` 会直接影响真实执行补偿，不要把它们误当成训练侧数据变换
- 真实机器人测试前必须确认急停、工作空间、速度/力矩限制、夹爪阈值与 reset 位姿
