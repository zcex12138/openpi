# Proposal: Add Franka Robot Evaluation Script

## Summary

Add a Franka Panda robot evaluation script that allows users to evaluate fine-tuned openpi checkpoints (such as `pi05_franka_screwdriver_lora`) on a real Franka robot. The script uses an `Environment` class (aligned with `examples/aloha_real`) and openpi's synchronous inference patterns (single-loop, chunked actions), while leveraging the robot communication infrastructure from `reactive_diffusion_policy`. Because the required RealSense/XenseCamera packages are Python 3.9–only, camera capture runs in a separate Python 3.9 process and openpi connects via a lightweight camera client.

## Motivation

- Users have fine-tuned openpi models on custom Franka datasets (e.g., `single_arm_screwdriver`) and need to evaluate these checkpoints on real hardware
- The `config.py` defines `LeRobotFrankaDataConfig` and `pi05_franka_screwdriver_lora` but the required `franka_policy.py` transforms are not yet implemented
- Reference implementation exists in `reactive_diffusion_policy/eval_real_robot_franka.py` using `FrankaRealRunner`, which provides tested robot communication code

## Scope

### In Scope
1. Create `src/openpi/policies/franka_policy.py` - FrankaInputs/FrankaOutputs transforms (prerequisite for training config)
2. Create `examples/franka/main.py` - main evaluation script following openpi style
3. Create `examples/franka/real_env.py` - low-level Franka robot integration
4. Create `examples/franka/camera_service.py` - Python 3.9 camera service with IPC
5. Create `examples/franka/camera_client.py` - camera client for external Python 3.9 camera service
6. Create `examples/franka/env.py` - `Environment` wrapper adapted for openpi runtime
7. Create `examples/franka/constants.py` - shared constants for Franka robot
8. Create `examples/franka/README.md` - usage documentation, including camera service startup

### Out of Scope
- Changes to core policy inference code
- Camera/sensor driver implementations (reuse from `reactive_diffusion_policy`)
- Running camera drivers inside openpi (Python 3.11)
- ROS2 integration (script is ROS2-independent)
- Multi-robot or multi-arm support

## Design Overview

The evaluation script supports local in-process inference by default, with an optional client-server mode similar to other openpi examples (diagram shows remote mode). Camera capture runs in a separate Python 3.9 process:

```
                              +-----------------+
                              | serve_policy.py |
                              | (Policy Server) |
                              +--------+--------+
                                       |
                                       | WebSocket
                                       |
+----------------+             +-------v--------+         +--------------------+
| Franka Robot   |<----------->| examples/franka|<------->| Camera Service     |
| C++ Controller |   TCP/IP    | /main.py       |  IPC    | (Python 3.9, RDP)  |
+----------------+             +----------------+         +--------------------+
```

### Key Components

1. **franka_policy.py**: Policy transforms that:
   - `FrankaInputs`: Maps observation keys to model input format (base_0_rgb, left_wrist_0_rgb, state)
   - `FrankaOutputs`: Extracts 8D actions from model output

2. **main.py**: Entry point that:
   - Loads trained policy via `policy_config.create_trained_policy()`
   - Builds a `Runtime` with `FrankaEnvironment`, `PolicyAgent`, and `ActionChunkBroker`
   - Runs evaluation episodes

3. **real_env.py**: `FrankaRealEnv` class that:
   - Connects to Franka robot via `RobotClient` (from `reactive_diffusion_policy`)
   - Collects robot state and executes robot actions with safety checks

4. **camera_service.py**: Camera service that:
   - Runs in Python 3.9 with RealSense/XenseCamera drivers
   - Exposes latest L500/D400 RGB frames via a lightweight IPC protocol

5. **camera_client.py**: Camera client that:
   - Connects to the external Python 3.9 camera service
   - Retrieves L500/D400 RGB frames for observation formatting

6. **env.py**: `FrankaEnvironment` class that:
   - Implements `openpi_client.runtime.environment.Environment`
   - Formats observations to match `FrankaInputs` expectations
   - Applies per-step actions (already chunked by `ActionChunkBroker`)

7. **constants.py**: Robot configuration constants (IP, ports, FPS, workspace bounds, etc.)

## Alternatives Considered

1. **Remote Policy Server Mode**: Use `serve_policy.py` + websocket client (like DROID example)
   - Pros: Separates inference from robot control, enables remote deployment
   - Cons: Adds latency, more complex setup
   - Decision: Support both modes (local inference as default, remote as option). Both modes remain synchronous.

2. **Integrate directly into serve_policy.py**
   - Cons: Mixes serving concerns with robot-specific code
   - Decision: Keep robot-specific code in examples/

## Dependencies

- `reactive_diffusion_policy.local_franka.bootstrap` - for `RobotClient` initialization
- `robot_client` - TCP client for Franka C++ controller
- Python 3.9 camera service from `reactive_diffusion_policy` (RealSense + XenseCamera)
- OpenCV - for camera image processing
- `openpi_client` runtime + websocket client (remote mode)

## Risks

1. **Robot Safety**: Workspace bounds and velocity limiting must be properly configured
   - Mitigation: Use conservative default bounds, require user confirmation before each episode

2. **Hardware Compatibility**: Different Franka setups may have different camera configurations
   - Mitigation: Configurable camera names via command-line args

3. **Loop Timing**: Synchronous inference may reduce control frequency on slow hardware
   - Mitigation: Use action chunks (open-loop horizon) to reduce inference calls and log when step time exceeds target

4. **Cross-process Camera Latency**: Camera service IPC may introduce latency or stale frames
   - Mitigation: Use latest-frame semantics, include timestamps, log if frames are stale/missing

## Success Criteria

- User can evaluate a checkpoint with: `uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora`
- Script maintains target control loop frequency (best-effort) with synchronous inference
- Action chunks are executed sequentially at control frequency
- Actions are properly clipped to workspace bounds
- Video recording of evaluation episodes is saved
- Camera frames are sourced from the external Python 3.9 camera service
