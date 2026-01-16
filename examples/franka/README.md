# Franka 机器人评估

本目录包含用于在真实 Franka Panda 机器人上评估微调后的 openpi checkpoint 的脚本。

## 先决条件

### 硬件
- Franka Panda 机器人，并运行 C++ 的 `franka_control` 服务器
- Intel RealSense L500 相机（底座/第三人称视角）
- Intel RealSense D400 或 Xense 相机（腕部视角）

### 软件
- 安装了 openpi 的 Python 3.11+ 环境
- 用于相机服务的独立 Python 3.9 环境（RealSense/Xense 驱动）
- 用于机器人客户端的 `franka_control` 包，位于 `"/home/mpi/workspace/yhx/franka_control"`

## 架构

```
                              +-----------------+
                              | serve_policy.py |
                              | (Policy Server) |
                              +--------+--------+
                                       |
                                       | WebSocket (optional)
                                       |
+----------------+             +-------v--------+         +--------------------+
| Franka Robot   |<----------->| examples/franka|<------->| Camera Service     |
| C++ Controller |   TCP/IP    | /main.py       |  IPC    | (Python 3.9, RDP)  |
+----------------+             +----------------+         +--------------------+
```

## 使用方法

### 1. 启动相机服务（Python 3.9）

相机服务在带有 RealSense/Xense 驱动的独立 Python 3.9 环境中运行：

```bash
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
# In Python 3.9 environment
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/camera_service.py

# Visualize
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/camera_test_visualize.py

# Or with custom provider
PYTHONPATH=/home/mpi/workspace/yhx/openpi
python examples/franka/camera_service.py \
    --provider 'reactive_diffusion_policy.local_franka:make_camera_provider' \
    --provider-kwargs '{"l500_serial": "xxx", "d400_serial": "yyy"}'

### 1.1 可视化阻抗控制目标与 TCP 位姿（3D 坐标架）

```bash
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/visualize_impedance_pose.py \
    --robot-ip 127.0.0.1 \
    --robot-port 8888
```
```

### 2. 启动 Franka C++ 控制器

确保 Franka C++ 控制器（`franka_control`）正在运行并可接受连接。

### 3. 运行评估

**注意**：请在仓库根目录下运行，并确保已安装本项目（或将其加入 `PYTHONPATH`）。

#### 本地推理（推荐）

```bash
source /home/mpi/workspace/yhx/openpi/examples/franka/.venv/bin/activate
uv run examples/franka/main.py \
    --checkpoint-dir ./checkpoints/10000 \
    --config pi05_franka_screwdriver_lora \
    --prompt "open the can with the screwdriver" \
    --num-episodes 10
```

#### 远程推理（Policy Server 模式，默认使用该模式）

首先启动 policy server：
```bash
source /home/mpi/workspace/yhx/openpi/.venv/bin/activate
uv run scripts/serve_policy.py \
    --record \
    policy:checkpoint \
    --policy.config=pi05_franka_screwdriver_lora \
    --policy.dir=./checkpoints/pi05_franka_screwdriver_lora/10000
```

然后运行评估：
```bash
proxy_off
PYTHONPATH=/home/mpi/workspace/yhx/openpi python examples/franka/main.py \
    --args.remote-host localhost \
    --args.remote-port 8000 \
    --args.prompt "open the can with the screwdriver" \
    --args.open-loop-horizon 10 \
    --args.max_episode_time 120.0 \
    --args.num-episodes 1
```

## 配置选项

### 命令行参数

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--checkpoint-dir` | None | checkpoint 目录路径（本地推理） |
| `--config` | None | 配置名（例如 `pi05_franka_screwdriver_lora`） |
| `--robot-ip` | `127.0.0.1` | Franka 控制器 IP |
| `--robot-port` | `8888` | Franka 控制器端口 |
| `--control-fps` | `30.0` | 控制环频率 |
| `--open-loop-horizon` | None | 动作分块大小（None = 使用模型默认值） |
| `--max-episode-time` | `30.0` | 单回合最长时长（秒） |
| `--num-episodes` | `10` | 评估回合数 |
| `--prompt` | `"open the can..."` | 任务指令 |
| `--max-pos-speed` | `0.5` | TCP 最大速度（m/s） |
| `--output-dir` | `./eval_results` | 结果输出目录 |
| `--camera-host` | `127.0.0.1` | 相机服务 host |
| `--camera-port` | `5050` | 相机服务端口 |
| `--remote-host` | None | 远程 policy server host |
| `--remote-port` | None | 远程 policy server 端口 |

### 相机配置（YAML）

创建一个 `camera_config.yaml` 文件：

```yaml
camera:
  realsense_cameras:
    - camera_name: l500
      camera_type: L500
      camera_serial_number: "f0123456"
      rgb_resolution: [960, 540]
      depth_resolution: [640, 480]
      fps: 30
    - camera_name: d400
      camera_type: D400
      camera_serial_number: "d0123456"
      rgb_resolution: [640, 480]
      depth_resolution: [640, 480]
      fps: 30

  camera_service:
    host: "0.0.0.0"
    port: 5050
    l500_name: l500
    d400_name: d400
    convert_bgr: true
    poll_hz: 30

robot_server:
  ip: "127.0.0.1"
  port: 8888

recording:
  fps: 30

evaluation:
  workspace_bounds:
    min: [0.2, -0.5, 0.0]
    max: [0.8, 0.5, 0.6]
  max_pos_speed: 0.5
  max_episode_time: 30.0
  num_episodes: 10
```

## 数据格式

### 观测（Observation）
- `observation/image`：底座相机（L500），224x224x3 uint8
- `observation/wrist_image`：腕部相机（D400），224x224x3 uint8
- `observation/state`：14 维 float32（TCP 位姿 7 维 + 额外 7 维）

### 动作（Action）
- 8 维 float32：`[x, y, z, qw, qx, qy, qz, gripper]`

## 安全指南

1. **工作空间边界**：默认边界较为保守；如有需要可在配置中调整。
2. **速度限制**：TCP 最大速度限制为 `max_pos_speed`。
3. **用户确认**：每个回合开始前都需要按 Enter 键确认启动。
4. **紧急停止**：按 Ctrl+C 可中止评估。
5. **初始位置**：开始前请确认机器人处于安全位置。

## 输出

结果会保存到 `output_dir/results.csv`：

```csv
episode,steps,elapsed_time,success
1,450,15.2,False
2,380,12.8,False
...
```

注意：`success` 需要在观看回合过程后进行人工标注。

## 故障排查

### 相机服务无响应
- 检查 Python 3.9 的相机服务是否在运行
- 核对配置中的相机序列号
- 在相机客户端侧尝试使用 `ping` 命令

### 机器人连接失败
- 确认 Franka C++ 控制器正在运行
- 检查 IP 与端口设置
- 确保机器人处于可控制状态

### 动作被大量裁剪（clipped）
- 检查配置中的工作空间边界
- 确认模型训练时使用了相近的边界
- 考虑调整 `max_pos_speed`

## 文件

| 文件 | 说明 |
|------|-------------|
| `main.py` | 评估主入口 |
| `env.py` | 面向 openpi runtime 的 FrankaEnvironment 封装 |
| `real_env.py` | 机器人状态/动作的底层处理 |
| `camera_client.py` | Python 3.9 相机服务的客户端 |
| `camera_service.py` | IPC 相机服务（Python 3.9） |
| `constants.py` | 默认配置常量 |
| `ipc.py` | 带长度前缀的 msgpack IPC 辅助函数 |
