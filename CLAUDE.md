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

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

openpi 是 Physical Intelligence 开源的机器人视觉-语言-动作（VLA）模型仓库。包含三种模型架构：
- **π₀ (pi0)**：基于流匹配的 VLA 模型
- **π₀-FAST (pi0_fast)**：使用 FAST 动作分词器的自回归 VLA
- **π₀.₅ (pi05)**：通过知识隔离技术升级的 π₀，具有更好的开放世界泛化能力

代码库同时支持 JAX（主要）和 PyTorch 实现。

## 常用命令

### 环境配置
```bash
# 克隆仓库（含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 安装依赖（需要先安装 uv）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 训练
```bash
# 训练前先计算归一化统计量（必需）
uv run scripts/compute_norm_stats.py --config-name <config_name>

# JAX 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name>

# PyTorch 单卡训练
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# PyTorch 多卡训练
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

### 推理服务
```bash
# 启动默认策略服务（按环境选择）
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO | ALOHA_SIM]

# 启动自定义检查点服务
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>
```

### 测试
```bash
# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest src/openpi/models/pi0_test.py

# 详细输出
uv run pytest -v
```

### 代码检查
```bash
ruff check .
ruff format .
pre-commit run --all-files
```

## 架构概览

### 核心模块

**模型层 (`src/openpi/models/`)**：
- `model.py`：基础模型接口，定义 `Observation`、`Actions` 和 `BaseModel` 抽象类
- `pi0.py`：流匹配 π₀ 实现，使用 PaliGemma 主干 + 动作专家网络
- `pi0_fast.py`：自回归 FAST 变体
- `gemma.py`、`siglip.py`：底层 Transformer 和视觉编码器
- `tokenizer.py`：文本和动作分词（PaligemmaTokenizer、FASTTokenizer）

**策略层 (`src/openpi/policies/`)**：
- `policy.py`：`Policy` 类，封装模型和变换用于推理
- `policy_config.py`：`create_trained_policy()` 工厂函数，用于加载检查点
- `*_policy.py`（aloha、droid、libero）：机器人平台特定的输入/输出变换

**训练层 (`src/openpi/training/`)**：
- `config.py`：所有训练配置定义在 `_CONFIGS` 列表中，使用 `get_config(name)` 获取
- `data_loader.py`：LeRobot 和 RLDS 数据加载
- `weight_loaders.py`：检查点权重加载工具
- `checkpoints.py`：检查点保存/恢复逻辑

**变换层 (`src/openpi/transforms.py`)**：
- 数据管道变换（归一化、分词、缩放、增量动作）
- `Group` 类组织输入/输出变换
- `RepackTransform` 将数据集键映射到模型期望的键

### 数据流

1. **训练**：原始数据 → `repack_transforms` → `data_transforms` → 归一化 → `model_transforms` → 模型
2. **推理**：观测字典 → 变换（同训练）→ `policy.infer()` → `output_transforms` → 反归一化 → 动作

### 关键设计模式

**配置系统**：所有配置都是 `src/openpi/training/config.py` 中的 dataclass。训练配置包含：
- 模型配置（架构、维度）
- 数据配置工厂（数据集、变换）
- 权重加载器（预训练检查点）
- 超参数（学习率、批大小、步数）

**机器人适配器**：每个机器人平台有专门的输入/输出变换类（如 `AlohaInputs`、`DroidOutputs`），处理：
- 机器人传感器与模型期望之间的键映射
- 状态/动作空间转换
- 坐标系变换

**远程推理**：`openpi-client` 包（`packages/openpi-client/`）提供轻量级 websocket 客户端，用于从机器人代码查询策略服务器。

## 可用训练配置

主要配置（完整列表见 `src/openpi/training/config.py`）：
- `pi0_aloha`、`pi05_aloha`：ALOHA 机器人推理
- `pi0_droid`、`pi05_droid`、`pi0_fast_droid`：DROID 机器人推理
- `pi0_libero`、`pi05_libero`、`pi0_fast_libero`：LIBERO 基准训练
- `pi05_franka_screwdriver_lora`：Franka Panda LoRA 微调示例（LeRobot v2，repo_id `single_arm_screwdriver`）
- `pi0_aloha_sim`：ALOHA 仿真训练
- `debug`、`debug_pi05`：快速迭代调试配置

### Franka（LeRobot v2）配置片段

配置名：`pi05_franka_screwdriver_lora`（定义位置：`src/openpi/training/config.py`）。要点摘要：
- 数据配置：`LeRobotFrankaDataConfig`（LeRobot v2），`repo_id=single_arm_screwdriver`；动作维度 8（末尾 gripper 1 维），`use_delta_joint_actions=False`，`default_prompt="open the can with the screwdriver"`。
- 关键映射：`observation/image -> observation.images.l500`，`observation/wrist_image -> observation.images.d400`，`observation/state -> observation.state`，`actions -> action`。
- 变换：`FrankaInputs/FrankaOutputs`，可选 delta/absolute 动作变换；`ModelTransformFactory(default_prompt=...)`。
- 模型：`pi05=True`，`action_horizon=30`，`paligemma_variant="gemma_2b_lora"`，`action_expert_variant="gemma_300m_lora"`（`action_dim` 默认 32，需匹配 `pi05_base` checkpoint）。
- 训练：`weight_loader="./data/checkpoints/pi05_base/params"`；`CosineDecaySchedule(warmup=500, peak_lr=1.5e-5, decay_steps=12000, decay_lr=1e-6)`；`num_train_steps=12000`，`batch_size=64`，`num_workers=8`，`log_interval=100`，`save_interval=500`，`keep_period=2000`，`ema_decay=None`；`freeze_filter` 来自 `Pi0Config(pi05=True, action_dim=8, action_horizon=30, ...).get_freeze_filter()`。

## Franka 机器人控制

### frankx 运动库

位置：`/home/mpi/workspace/yhx/frankx`

frankx 是 Franka Emika 机器人的高层次运动库，提供 C++ 和 Python 双接口。它封装了 libfranka，使用 Ruckig 实现在线轨迹生成（OTG），支持 jerk 约束。

**核心 API**：
- `Robot`：机器人连接与运动控制（`robot.move(motion)`、`set_dynamic_rel()`、`current_pose()`）
- `Gripper`：夹爪控制（`clamp()`、`release()`、`move()`）
- `Affine`：SE(3) 变换封装（Eigen::Affine3d 的薄包装），支持欧拉角（ZYX 约定）
- 运动类型：`JointMotion`、`LinearMotion`、`LinearRelativeMotion`、`WaypointMotion`、`PositionHold`
- `MotionData`：动态参数调整与实时反应（`with_reaction()`）

**构建与安装**：
```bash
cd /home/mpi/workspace/yhx/frankx
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DBUILD_TYPE=Release .. && make && make install
# 或 Python 模块安装
pip install .
```

**坐标与单位**：距离单位 [m]，旋转单位 [rad]，四元数顺序 `(w, x, y, z)`。

### Franka 坐标系与四元数注意事项
- 观测/动作四元数顺序为 `(w, x, y, z)`。
- `frankx` 默认行为在 `setDefaultBehavior()` 中设置 `NE_T_EE = Rx(pi)`，但本项目在 `examples/franka/real_env.py` 会调用 `set_EE(constants.DEFAULT_EE_TRANSFORM)` 覆盖该值。
- 若出现姿态偏差，优先核对 `examples/franka/constants.py` 中的 `DEFAULT_EE_TRANSFORM` 与采集时一致（通常为单位阵），避免在观测/动作上叠加手动补偿。

## Franka 评估示例 (`examples/franka/`)

### 虚拟环境

位置：`/home/mpi/workspace/yhx/openpi/examples/franka/.venv`（Python 3.9）

该环境独立于主 openpi 环境，用于相机服务和机器人控制，因为 RealSense/Xense 驱动需要 Python 3.9。

**安装依赖**（见 `INSTALL.md`）：
```bash
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
# RealSense
uv pip install pyrealsense2-2.53.1.4623-cp39-cp39-manylinux1_x86_64
# Xense（可选）
uv pip install nvidia-cudnn-cu11==8.9.2.26 nvidia-cuda-runtime-cu11 onnxruntime-gpu==1.18.0
pip install xensesdk==1.6.0 -i https://repo.huaweicloud.com/repository/pypi/simple/
# frankx
cd /home/mpi/workspace/yhx/frankx && uv pip install -e .
# openpi-client
uv pip install -e packages/openpi-client && uv pip install tyro
```

### 架构与流程

```
+----------------+     +------------------+     +--------------------+
| Policy Server  |<--->| examples/franka  |<--->| Camera Service     |
| (serve_policy) | WS  | /main.py         | IPC | (Python 3.9, RDP)  |
+----------------+     +--------+---------+     +--------------------+
                                |
                                | FCI/IP
                                v
                       +----------------+
                       | Franka Robot   |
                       | (frankx 直连)   |
                       +----------------+
```

**评估流程**：
1. **启动相机服务**（Python 3.9 环境）：
   ```bash
   source examples/franka/.venv/bin/activate
   PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/camera_service.py
   ```
2. **确认 FCI 可连接**：机器人处于可控状态，FCI IP 可访问（frankx 直连，无需额外服务）。
3. **启动 Policy Server**（主 openpi 环境）：
   ```bash
   source .venv/bin/activate
   uv run scripts/serve_policy.py policy:checkpoint \
       --policy.config=pi05_franka_screwdriver_lora \
       --policy.dir=./data/checkpoints/pi05_franka_screwdriver_lora/10000
   ```
4. **运行评估**：
   ```bash
   PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/main.py \
       --args.remote-host localhost --args.remote-port 8000 \
       --args.control-mode cartesian --args.prompt "open the can with the screwdriver"
   ```

### 关键文件

| 文件 | 说明 |
|------|------|
| `main.py` | 评估主入口，支持本地/远程推理 |
| `real_env.py` | 机器人状态/动作底层处理，调用 frankx |
| `env.py` | 面向 openpi runtime 的 FrankaEnvironment 封装 |
| `camera_service.py` | IPC 相机服务（Python 3.9） |
| `camera_client.py` | 相机服务客户端 |
| `constants.py` | 默认配置常量（从 `camera_config.yaml` 加载） |
| `visualize_online_trajectory.py` | 实时可视化 TCP 位姿与目标 |
| `visualize_wrench.py` | 实时可视化力/力矩 |

### 数据格式

**观测（Observation）**：
- `observation/image`：底座相机（L500），224×224×3 uint8
- `observation/wrist_image`：腕部相机（D400/Xense），224×224×3 uint8
- `observation/state`：14 维 float32（TCP 位姿 7 维 + GRIPPER 1 维 + 6 维力）

**动作（Action）**：8 维 float32 `[x, y, z, qw, qx, qy, qz, gripper]`


### 本地数据集路径
- 使用 HF_LEROBOT_HOME 将默认数据集地址设置为本地
`export HF_LEROBOT_HOME=/home/mpi/workspace/yhx/openpi/data/dataset`

## 近期整理记录（2026-01-20）

- `examples/franka/main.py` 移除相对路径点（relative waypoint）相关功能与 CLI 参数，精简评估流程与输出。
- `examples/franka/main.py` 删除客户端记录功能（ClientRecorder）及相关参数/订阅逻辑。
- `examples/franka/real_env.py` cartesian 控制统一使用绝对 Waypoint，移除相对路径点分支。
- `examples/franka/README.md` 命令行参数表更新为与 `real_env_config.yaml` / `camera_config.yaml` 的默认值对齐，并补充互斥与覆盖关系说明。

## 项目 AI 提示词（建议）

```text
你是一名专注机器人与机器学习的资深代码架构工程师，目标是让 openpi 代码高性能、可维护、健壮且简洁易用。
工作时严格遵循 KISS / YAGNI / DRY / SOLID 原则；避免无谓功能扩展与冗余日志。
当涉及规划、提案或架构变更时，必须先查看 @/openspec/AGENTS.md 的规范。
围绕 examples/franka 的修改需保持 CLI、README 与配置文件默认值一致，不要重新引入已移除的相对路径点与客户端记录功能，除非用户明确要求。
以中文沟通；代码标识、命令与日志保持原语言。
```
