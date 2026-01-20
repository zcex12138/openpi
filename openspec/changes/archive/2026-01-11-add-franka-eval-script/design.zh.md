# 设计：Franka 机器人评估脚本

## 架构

```
examples/franka/
├── main.py           # 含 CLI 参数的入口
├── real_env.py       # FrankaRealEnv（机器人状态 + 动作）
├── camera_service.py # 相机服务（Python 3.9, IPC）
├── camera_client.py  # 相机服务客户端（Python 3.9 服务）
├── env.py            # FrankaEnvironment（openpi runtime 封装）
├── constants.py      # 机器人/相机配置
└── README.md         # 使用文档
```

## 组件设计

### 1. FrankaRealEnv 类

`FrankaRealEnv` 负责：
1. 通过 `RobotClient` 进行机器人通信
2. 采集机器人状态
3. 在安全约束下执行底层动作

策略推理由 runtime/agent 层负责。

```python
class FrankaRealEnv:
    """低层 Franka 机器人环境（机器人状态 + 动作）。"""

    def __init__(
        self,
        robot_ip: str,
        robot_port: int,
        camera_config: dict,
        control_fps: float,
        workspace_bounds: tuple[np.ndarray, np.ndarray],
        max_pos_speed: float,
    ):
        ...

    def get_state(self) -> np.ndarray:
        """获取当前 TCP 状态（前 7 维为姿态）。"""
        ...

    def execute_action(self, action: np.ndarray) -> None:
        """执行动作并包含安全检查。"""
        # action: (8,) - [x,y,z,qw,qx,qy,qz,gripper]
        ...

    def reset(self) -> None:
        """可选复位到安全姿态。"""
        ...
```

### 2. CameraService（IPC）

相机服务在 Python 3.9 环境运行 RealSense/XenseCamera 驱动，提供最新 RGB 帧的 IPC 接口。

协议（TCP + length-prefixed msgpack）：
- 请求：`{\"type\": \"get_frames\"}`
- 响应：`{\"ok\": true, \"timestamp_ns\": int, \"frames\": {\"l500_rgb\": {...}, \"d400_rgb\": {...}}}`
- 每帧编码为 `{ \"shape\": [H, W, C], \"dtype\": \"uint8\", \"data\": bytes }`

### 3. CameraClient 类

`CameraClient` 连接 Python 3.9 相机服务（RDP 栈）并获取 RGB 帧。

```python
class CameraClient:
    def __init__(self, host: str, port: int, *, timeout_s: float = 0.1):
        ...

    def get_frames(self) -> dict[str, np.ndarray]:
        """返回最新帧，包含 l500_rgb 与 d400_rgb。"""
        ...
```

### 4. FrankaEnvironment 类

`FrankaEnvironment` 实现 `openpi_client.runtime.environment.Environment`，并封装 `FrankaRealEnv`。

```python
class FrankaEnvironment(_environment.Environment):
    def __init__(self, real_env: FrankaRealEnv, camera: CameraClient, *, prompt: str):
        ...

    def reset(self) -> None:
        ...

    def is_episode_complete(self) -> bool:
        ...

    def get_observation(self) -> dict:
        """返回匹配 FrankaInputs 的观测。"""
        ...

    def apply_action(self, action: dict) -> None:
        """应用单步动作（动作分块由 ActionChunkBroker 处理）。"""
        ...
```

### 5. 观测格式映射

训练数据采用 LeRobot 格式：
- `observation.images.l500` → `observation/image`（基座相机）
- `observation.images.d400` → `observation/wrist_image`（腕部相机）
- `observation.state` → `observation/state`（包含 TCP pose 与可选额外状态）
- `action` → 8 维：position[3] + quaternion[4] + gripper[1]

`LeRobotFrankaDataConfig.repack_transforms` 将其映射到模型输入格式：
```python
{
    "observation/image": "observation.images.l500",
    "observation/wrist_image": "observation.images.d400",
    "observation/state": "observation.state",
    "actions": "action",
}
```

`FrankaInputs` 进一步转换为模型格式：
- `state`: `observation/state` 的前 7 维（TCP pose）
- `image["base_0_rgb"]`: 基座相机图像
- `image["left_wrist_0_rgb"]`: 腕部相机图像（对 pi0/pi05 映射为 left_wrist）

### 6. 动作执行链路

```
Policy Output (action_horizon x action_dim)
    → FrankaOutputs (裁剪为 8 维动作)
    → Unnormalize
    → ActionChunkBroker (裁剪为单步)
    → FrankaEnvironment.apply_action()
        → FrankaRealEnv.execute_action()
        → 工作空间裁剪
        → 速度限制
        → RobotClient.send_pose()
```

### 7. 控制循环时序

使用 `openpi_client.runtime.Runtime`，并设置 `max_hz=control_fps`。`ActionChunkBroker` 保证每次循环仅执行单步动作；
若某步超过目标周期，则记录警告并继续（不 sleep）。

## 数据流

```
┌─────────────┐     ┌──────────────────┐
│ Robot State │────►│ FrankaRealEnv    │
└─────────────┘     └────────┬─────────┘
                             │ state
                             ▼
┌──────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Camera Service   │────►│ CameraClient  │────►│ FrankaInputs    │
│ (Python 3.9)     │ IPC │ get_frames()  │     │ (transform)     │
└──────────────────┘     └──────┬────────┘     └────────┬────────┘
                                │                        │
                                ▼                        ▼
                         ┌──────────────────┐     ┌─────────────────┐
                         │ Policy.infer()   │◄────│ Normalize +     │
                         │                  │     │ Model transforms│
                         └────────┬─────────┘     └─────────────────┘
                                  │
                                  ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Robot       │◄────│ FrankaRealEnv    │◄────│ FrankaOutputs   │
│ Execution   │     │ execute_action() │     │ + Unnormalize   │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## 配置

### CLI 参数（main.py）

```python
@dataclasses.dataclass
class Args:
    # Checkpoint
    checkpoint_dir: str          # checkpoint 路径（如 ./checkpoints/11999）
    config: str                  # 配置名（如 pi05_franka_screwdriver_lora）

    # Robot
    robot_ip: str = "localhost"
    robot_port: int = 8888

    # Control
    control_fps: float = 30.0
    open_loop_horizon: int | None = None  # None = 使用模型 action_horizon
    max_episode_time: float = 30.0
    num_episodes: int = 10

    # Task
    prompt: str = "open the can with the screwdriver"

    # Safety
    max_pos_speed: float = 0.5   # m/s

    # Output
    save_video: bool = True
    output_dir: str = "./eval_results"
    save_summary: bool = True

    # Camera service (Python 3.9)
    camera_host: str = "127.0.0.1"
    camera_port: int = 5050
    camera_timeout_s: float = 0.1

    # Remote policy (optional)
    remote_host: str | None = None
    remote_port: int | None = None
    api_key: str | None = None
```

### 相机配置

默认相机映射（匹配训练数据）：
- `l500_camera` → 基座相机（observation/image）
- `d400_camera` → 腕部相机（observation/wrist_image）

## 输出

- episode 视频（可选）保存到 `output_dir`
- episode 汇总信息保存到 `output_dir` 下的 CSV 或 JSON 文件（如 `results.csv`）

## 安全考虑

1. **工作空间边界**：使用保守默认值，可覆盖
2. **速度限制**：限制每步 TCP 最大速度
3. **用户确认**：每个 episode 前需按 Enter 确认
4. **夹爪控制**：可选择禁用夹爪
5. **紧急停止**：Ctrl+C 立即停止阻抗控制

## 错误处理

- 机器人连接失败 → 退避重试后退出
- 相机服务失败 → 记录警告，使用零图像（允许降级运行）
- 推理超时 → 保持上一次动作或在下一轮重新推理
- 动作裁剪 → 发生显著裁剪时记录日志
