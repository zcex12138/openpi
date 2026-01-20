# Design: Franka Robot Evaluation Script

## Architecture

```
src/openpi/policies/
└── franka_policy.py      # FrankaInputs/FrankaOutputs transforms

examples/franka/
├── main.py           # Entry point with CLI args
├── real_env.py       # FrankaRealEnv (robot state + actions)
├── camera_service.py # Camera service (Python 3.9, IPC)
├── camera_client.py  # Camera service client (Python 3.9 service)
├── env.py            # FrankaEnvironment (openpi runtime wrapper)
├── constants.py      # Robot/camera configuration
└── README.md         # Usage documentation
```

## Component Design

### 0. FrankaInputs/FrankaOutputs Transforms

The `franka_policy.py` module provides data transforms for the Franka policy, following the pattern of `droid_policy.py`.

```python
@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Transform observation dict to model input format."""
    model_type: _model.ModelType
    base_image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    state_key: str = "observation/state"

    def __call__(self, data: dict) -> dict:
        # Extract first 7D from 14D state (TCP pose: x,y,z,qw,qx,qy,qz)
        state = np.asarray(data[self.state_key])[:7]

        # Map images to model input names
        # base_0_rgb, left_wrist_0_rgb (pi0/pi05)
        ...


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Extract 8D actions from model output."""
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
```

### 1. FrankaRealEnv Class

The `FrankaRealEnv` is responsible for:
1. Robot communication via `RobotClient`
2. Robot state collection
3. Low-level action execution with safety checks

Policy inference is handled outside the real env by the runtime/agent.

```python
class FrankaRealEnv:
    """Low-level Franka robot environment (robot state + actions)."""

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
        """Get current TCP state (first 7 dims are pose)."""
        ...

    def execute_action(self, action: np.ndarray) -> None:
        """Execute action on robot with safety checks."""
        # action: (8,) - [x,y,z,qw,qx,qy,qz,gripper]
        ...

    def reset(self) -> None:
        """Optional reset to a safe pose."""
        ...
```

### 2. CameraService (IPC)

The camera service runs in a Python 3.9 environment with RealSense/XenseCamera drivers. It provides a lightweight
IPC interface that returns the latest RGB frames.

Protocol (length-prefixed msgpack over TCP):
- Request: `{"type": "get_frames"}`
- Response: `{"ok": true, "timestamp_ns": int, "frames": {"l500_rgb": {...}, "d400_rgb": {...}}}`
- Each frame is encoded as `{ "shape": [H, W, C], "dtype": "uint8", "data": bytes }`

### 3. CameraClient Class

The `CameraClient` connects to a Python 3.9 camera service (RDP stack) and fetches RGB frames.

```python
class CameraClient:
    def __init__(self, host: str, port: int, *, timeout_s: float = 0.1):
        ...

    def get_frames(self) -> dict[str, np.ndarray]:
        """Return latest frames with keys l500_rgb and d400_rgb."""
        ...
```

### 4. FrankaEnvironment Class

`FrankaEnvironment` implements `openpi_client.runtime.environment.Environment` and wraps `FrankaRealEnv`.

```python
class FrankaEnvironment(_environment.Environment):
    def __init__(self, real_env: FrankaRealEnv, camera: CameraClient, *, prompt: str):
        ...

    def reset(self) -> None:
        ...

    def is_episode_complete(self) -> bool:
        ...

    def get_observation(self) -> dict:
        """Return observation formatted for FrankaInputs."""
        ...

    def apply_action(self, action: dict) -> None:
        """Apply a single-step action (already chunked)."""
        ...
```

### 5. Observation Format Mapping

The training data uses LeRobot format with keys:
- `observation.images.l500` → `observation/image` (base camera, 224x224x3)
- `observation.images.d400` → `observation/wrist_image` (wrist camera, 224x224x3)
- `observation.state` → `observation/state` (14D: TCP pose 7D + additional state 7D)
- `action` → 8D: position[3] + quaternion[4] + gripper[1]

The `LeRobotFrankaDataConfig.repack_transforms` maps these to model input format:
```python
{
    "observation/image": "observation.images.l500",
    "observation/wrist_image": "observation.images.d400",
    "observation/state": "observation.state",
    "actions": "action",
}
```

`FrankaInputs` then transforms to model format:
- `state`: first 7 dims of observation/state (TCP pose: x, y, z, qw, qx, qy, qz)
- `image["base_0_rgb"]`: base camera image (L500)
- `image["left_wrist_0_rgb"]`: wrist camera image (D400, mapped to left_wrist for pi0/pi05)

### 6. Action Execution Pipeline

```
Policy Output (action_horizon x action_dim)
    → FrankaOutputs (slice to 8D actions)
    → Unnormalize
    → ActionChunkBroker (slice to single step)
    → FrankaEnvironment.apply_action()
        → FrankaRealEnv.execute_action()
        → Workspace clipping
        → Velocity limiting
        → RobotClient.send_pose()
```

### 7. Control Loop Timing

Use `openpi_client.runtime.Runtime` with `max_hz=control_fps`. `ActionChunkBroker` ensures one-step actions
per loop; if a step exceeds the target period, log a warning and continue without sleeping.

## Data Flow

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

## Configuration

### CLI Arguments (main.py)

```python
@dataclasses.dataclass
class Args:
    # Checkpoint
    checkpoint_dir: str          # Path to checkpoint (e.g., ./checkpoints/11999)
    config: str                  # Config name (e.g., pi05_franka_screwdriver_lora)

    # Robot
    robot_ip: str = "localhost"
    robot_port: int = 8888

    # Control
    control_fps: float = 30.0
    open_loop_horizon: int | None = None  # None = use model action_horizon
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

### Camera Configuration

Default camera mapping (matching training data):
- `l500_camera` → base camera (observation/image)
- `d400_camera` → wrist camera (observation/wrist_image)

## Outputs

- Episode videos (optional) are written under `output_dir`.
- Episode summaries are written to a CSV or JSON file in `output_dir` (e.g., `results.csv`).

## Safety Considerations

1. **Workspace Bounds**: Default conservative bounds, can be overridden
2. **Velocity Limiting**: Cap maximum TCP velocity per step
3. **User Confirmation**: Require Enter press before each episode
4. **Gripper Control**: Option to disable gripper during evaluation
5. **Emergency Stop**: Ctrl+C immediately stops impedance control

## Error Handling

- Robot connection failure → retry with backoff, then exit
- Camera service failure → log warning, use zero image (allow degraded operation)
- Policy inference timeout → hold last action or re-run inference on next loop
- Action clipping → log when actions are clipped significantly
