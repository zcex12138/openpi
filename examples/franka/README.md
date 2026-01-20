# Franka 机器人评估

本目录包含用于在真实 Franka Panda 机器人上评估微调后的 openpi checkpoint 的脚本。

## 使用方法

### 1. 启动相机服务（Python 3.9）

相机服务在带有 RealSense/Xense 驱动的独立 Python 3.9 环境中运行：

```bash
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/camera_service.py

# 可视化相机画面
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/camera_test_visualize.py
```

### 2. 确认 FCI 可连接

确保机器人处于可控制状态，FCI IP 可从本机访问。

### 3. 运行评估

**注意**：请在仓库根目录下运行，并确保已安装本项目（或将其加入 `PYTHONPATH`）。

#### 远程推理（Policy Server 模式）

首先启动 policy server：
```bash
source /home/mpi/workspace/yhx/openpi/.venv/bin/activate
uv run scripts/serve_policy.py \
    --record \
    policy:checkpoint \
    --policy.config=pi05_franka_screwdriver_lora \
    --policy.dir=./data/checkpoints/pi05_franka_screwdriver_lora/10000
```

然后运行评估：
```bash
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/main.py \
    --remote-host localhost \
    --remote-port 8000 \
    --control-mode cartesian \
    --prompt "open the can with the screwdriver" \
    --open-loop-horizon 3 \
    --max-pos-speed 0.1 \
    --max-episode-time 120.0 \
    --num-episodes 1
```

#### 本地推理

```bash
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
uv run examples/franka/main.py \
    --checkpoint-dir ./checkpoints/10000 \
    --config pi05_franka_screwdriver_lora \
    --prompt "open the can with the screwdriver" \
    --num-episodes 10
```

### 4. 可视化工具

```bash
# 实时可视化 TCP 位姿与目标
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/visualize_online_trajectory.py \
    --robot-ip 127.0.0.1

# 实时可视化力/力矩
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/visualize_wrench.py \
    --robot-ip 127.0.0.1
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint-dir` | None | checkpoint 目录路径（本地推理，需与 `--config` 同时使用；与 `--remote-host` 互斥） |
| `--config` | None | 训练配置名（例如 `pi05_franka_screwdriver_lora`，本地推理必填） |
| `--real-env-config` | `examples/franka/real_env_config.yaml` | 环境配置文件路径（None = 使用默认文件） |
| `--robot-ip` | `172.16.0.2` | Franka 控制器 FCI IP（覆盖 `robot.ip`） |
| `--control-mode` | `impedance` | 控制模式：`impedance` 或 `cartesian`（覆盖 `control.mode`） |
| `--control-fps` | `30.0` | 控制环频率 Hz（覆盖 `control.fps`） |
| `--open-loop-horizon` | None | 动作分块大小；本地推理使用模型 `action_horizon`，远程默认 30 |
| `--max-episode-time` | `30.0` | 单回合最长时长（秒，覆盖 `evaluation.max_episode_time`） |
| `--num-episodes` | `10` | 评估回合数（覆盖 `evaluation.num_episodes`） |
| `--prompt` | `open the can with the screwdriver` | 任务指令（覆盖 `evaluation.default_prompt`） |
| `--max-pos-speed` | `0.5` | TCP 最大速度 m/s（覆盖 `motion.max_pos_speed`） |
| `--action-smoothing-alpha` | `0.5` | 动作平滑系数（覆盖 `smoothing.alpha`；0.0=无平滑，0.9=重度平滑） |
| `--cartesian-velocity-factor` | `0.05` | cartesian 模式速度因子（覆盖 `cartesian.velocity_factor`） |
| `--camera-host` | `127.0.0.1` | 相机服务 host（`camera_config.yaml`） |
| `--camera-port` | `5050` | 相机服务端口（`camera_config.yaml`） |
| `--camera-timeout-s` | `0.1` | 相机服务超时时间（秒，`camera_config.yaml`） |
| `--remote-host` | None | 远程 policy server host（与 `--checkpoint-dir` 互斥） |
| `--remote-port` | `8000` | 远程 policy server 端口 |