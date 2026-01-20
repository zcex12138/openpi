# franka-evaluation Specification

## Purpose
TBD - created by archiving change add-franka-eval-script. Update Purpose after archive.
## Requirements
### Requirement: Policy Transforms
The system SHALL provide FrankaInputs and FrankaOutputs transforms for processing observations and actions.

#### Scenario: Transform observations to model format
- **WHEN** the policy receives an observation dict with keys `observation/image`, `observation/wrist_image`, `observation/state`
- **THEN** `FrankaInputs` maps images to model input names (`base_0_rgb`, `left_wrist_0_rgb`)
- **AND** extracts the first 7 dimensions of state as TCP pose

#### Scenario: Transform model output to actions
- **WHEN** the model outputs actions of shape `[action_horizon, action_dim]`
- **THEN** `FrankaOutputs` extracts the first 8 dimensions as robot actions

---

### Requirement: Local Policy Loading
The evaluation script SHALL load trained checkpoints locally using openpi's standard `policy_config.create_trained_policy()` function when remote mode is not enabled.

#### Scenario: Load local JAX checkpoint
- **WHEN** the user runs `uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora`
- **THEN** the policy is created via `policy_config.create_trained_policy()`
- **AND** normalization stats are loaded from the checkpoint assets

#### Scenario: Load local PyTorch checkpoint
- **WHEN** the checkpoint directory contains `model.safetensors`
- **THEN** the system detects PyTorch format and loads the PyTorch weights

---

### Requirement: Remote Policy Inference
The evaluation script SHALL support an optional remote inference mode via the websocket policy server.

#### Scenario: Connect to remote policy server
- **WHEN** the user provides `--remote-host` and `--remote-port`
- **THEN** the script uses `WebsocketClientPolicy` to send observations and receive action chunks
- **AND** local checkpoint loading is skipped

---

### Requirement: Observation Collection
The evaluation script SHALL collect observations matching the training data format.

#### Scenario: Camera service integration
- **WHEN** the external camera service (Python 3.9) is running and reachable
- **THEN** the script retrieves the latest L500/D400 RGB frames via the camera client
- **AND** the frames are used to populate `observation/image` and `observation/wrist_image`

#### Scenario: Collect camera observations
- **WHEN** the robot has L500 (base) and D400 (wrist) cameras connected
- **THEN** `observation/image` contains the L500 RGB image resized to 224x224
- **AND** `observation/wrist_image` contains the D400 RGB image resized to 224x224

#### Scenario: Collect robot state
- **WHEN** an observation is collected
- **THEN** `observation/state` includes the 7D TCP pose [x, y, z, qw, qx, qy, qz] in the first 7 dimensions
- **AND** additional state dimensions MAY be present

#### Scenario: Camera service unavailable
- **WHEN** the camera service is unreachable or times out
- **THEN** the script logs a warning
- **AND** a zero image MAY be substituted to allow degraded operation

---

### Requirement: Action Execution
The evaluation script SHALL execute policy actions on the robot with safety constraints.

#### Scenario: Execute TCP pose action
- **WHEN** the policy outputs an action chunk of shape `[action_horizon, 8]`
- **THEN** each action uses the format [x, y, z, qw, qx, qy, qz, gripper]
- **AND** the TCP position is clipped to workspace bounds
- **AND** the TCP velocity is limited to `max_pos_speed` m/s
- **AND** the quaternion is normalized before sending

#### Scenario: Gripper control
- **WHEN** gripper control is enabled and the policy predicts gripper > 0.7
- **THEN** the gripper closes for that step

---

### Requirement: Control Loop Timing
The evaluation script SHALL use a synchronous control loop consistent with openpi examples.

#### Scenario: Chunked synchronous loop
- **WHEN** `control_fps=30` and `open_loop_horizon=10`
- **THEN** the script executes one action every 1/30s (best-effort)
- **AND** policy inference is called once every 10 control steps

#### Scenario: Over-budget step
- **WHEN** a control step exceeds the target period
- **THEN** the script logs a warning and continues without sleeping

---

### Requirement: Episode Management
The evaluation script SHALL manage evaluation episodes with user interaction.

#### Scenario: Run multiple episodes
- **WHEN** `num_episodes=10`
- **THEN** the script waits for user confirmation before each episode
- **AND** optional reset-between-episodes behavior is applied when enabled
- **AND** an episode summary is saved after evaluation completes

#### Scenario: Early termination
- **WHEN** the user presses Ctrl+C during an episode
- **THEN** impedance control is stopped safely
- **AND** the gripper opens if gripper control is enabled
- **AND** the script exits cleanly

---

### Requirement: Video Recording
The evaluation script SHALL optionally record evaluation episodes.

#### Scenario: Save episode video
- **WHEN** `save_video=True` and an episode completes
- **THEN** a video file is saved to `output_dir` with camera views
- **AND** the filename includes episode number and timestamp

### Requirement: Waypoint 轨迹旋转使用四元数表示
在 Frankx 的 waypoint 轨迹生成中，系统 SHALL 采用基于四元数的旋转表示（log/exp 或等价形式），
避免欧拉角分支跳变导致的大幅旋转。

#### Scenario: 旋转连续性
- **GIVEN** 目标姿态与当前姿态非常接近
- **WHEN** 轨迹生成器更新 waypoint 目标
- **THEN** 旋转插值保持连续，不出现大幅绕圈

#### Scenario: 平移与肘部保持兼容
- **GIVEN** 轨迹生成使用 waypoint
- **WHEN** 切换到四元数旋转表示
- **THEN** 平移与肘部的运动行为保持不变

