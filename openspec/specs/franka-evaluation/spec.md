# franka-evaluation Specification

## Purpose
TBD - created by archiving change add-franka-eval-script. Update Purpose after archive.
## Requirements
### Requirement: Policy Transforms
The system SHALL support configurable rotation representation (`"quat"` or `"r6d"`) in FrankaInputs and FrankaOutputs transforms, controlled by training config.

#### Scenario: Transform observations with rotate6d enabled
- **WHEN** the training config has `rotation_representation="r6d"`
- **THEN** `FrankaInputs` skips quaternion sign normalization (`normalize_quat_sign=False`)
- **AND** `QuatToRotate6d` converts state from 7D `[xyz, qwqxqyqz]` to 9D `[xyz, r1-r6]`
- **AND** `QuatToRotate6d` converts actions from (H, 8D) to (H, 10D) with gripper shifted from index 7 to index 9

#### Scenario: Transform model output with rotate6d enabled
- **WHEN** the model outputs actions in 10D rotate6d format `[xyz, r1-r6, gripper]`
- **THEN** `Rotate6dToQuat` converts actions only (not state) from (H, 10D) to (H, 8D)
- **AND** `FrankaOutputs` extracts the first 8 dimensions as robot actions in quaternion format
- **AND** the output quaternions are unit-normalized and sign-canonicalized

#### Scenario: Transform with quaternion representation (backward compat)
- **WHEN** the training config has `rotation_representation="quat"` (default)
- **THEN** the transform pipeline behaves identically to the existing implementation
- **AND** no rotate6d conversion transforms are inserted

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

### Requirement: Rotation Conversion Utilities
The system SHALL provide vectorized rotation conversion functions in `src/openpi/shared/rotation.py` supporting arbitrary batch dimensions.

#### Scenario: Quaternion to rotate6d conversion
- **WHEN** `quat_to_rotate6d(quat)` is called with shape `(…, 4)` wxyz quaternion
- **THEN** it returns shape `(…, 6)` rotate6d representation
- **AND** internal computation uses float64 for numerical stability
- **AND** output dtype matches input dtype

#### Scenario: Rotate6d to quaternion conversion
- **WHEN** `rotate6d_to_quat(r6d)` is called with shape `(…, 6)` rotate6d
- **THEN** it returns shape `(…, 4)` wxyz quaternion
- **AND** Gram-Schmidt orthogonalization uses `eps=1e-6` with collinear fallback
- **AND** output quaternion is L2-normalized and sign-canonicalized

#### Scenario: Round-trip identity
- **WHEN** `rotate6d_to_quat(quat_to_rotate6d(q))` is computed
- **THEN** the result represents the same rotation as `q` (dot product > 1-ε)
- **AND** this holds for edge cases: identity rotation, 180° rotations, near-collinear inputs

#### Scenario: SO(3) membership guarantee
- **WHEN** `rotate6d_to_rotmat(r6d)` produces rotation matrix `R`
- **THEN** `R^T @ R ≈ I` (orthogonality)
- **AND** `det(R) ≈ +1` (proper rotation, not reflection)
- **AND** no NaN or Inf values in output

### Requirement: Delta Rotation in SO(3)
The system SHALL support relative rotation actions via rotation matrix multiplication when `use_relative_rotation=True`.

#### Scenario: Convert absolute to delta rotation
- **WHEN** `DeltaRotate6dActions` processes state and actions in rotate6d format
- **THEN** it computes `R_delta = R_target @ R_current^T` for each action frame
- **AND** the delta is stored back as rotate6d in the actions array
- **AND** translation (xyz) and gripper dimensions are unmodified

#### Scenario: Convert delta to absolute rotation
- **WHEN** `AbsoluteRotate6dActions` processes delta actions with current state
- **THEN** it computes `R_target = R_delta @ R_current` for each action frame
- **AND** round-trip `Absolute(Delta(actions, state), state)` recovers original actions (within float tolerance)

### Requirement: Dimension Validation
The system SHALL validate state and action dimensions at pipeline boundaries via `ValidateDims` transform.

#### Scenario: Dimension mismatch detection
- **WHEN** state or action dimensions do not match expected values after transform chain
- **THEN** `ValidateDims` raises `ValueError` with clear message including expected vs actual dimensions
- **AND** this prevents silent dimension errors from propagating to normalization or model

### Requirement: Franka Training Config with Rotate6D
The system SHALL provide Franka training configs with `rotation_representation="r6d"` that correctly wire the transform chain.

#### Scenario: Config with rotate6d and relative rotation
- **WHEN** `LeRobotFrankaDataConfigV2` is created with `rotation_representation="r6d"` and `use_relative_rotation=True`
- **THEN** the input transform chain is: ShiftedStateToAction → SelectStateFrame → FrankaInputs → QuatToRotate6d → DeltaActions(xyz) → DeltaRotate6dActions → ValidateDims
- **AND** the output transform chain is: AbsoluteRotate6dActions → AbsoluteActions(xyz) → Rotate6dToQuat → FrankaOutputs
- **AND** norm stats use 10D action and 9D state dimensions

#### Scenario: Norm stats dimension consistency
- **WHEN** norm stats are computed for a rotate6d config
- **THEN** action stats have shape `(10,)` and state stats have shape `(9,)`
- **AND** loading 8D norm stats with a 10D config raises an error (dimension mismatch)

---

### Requirement: Canonical Execution Mode Selection
The Franka evaluation flow SHALL resolve a single canonical `execution.mode` before constructing any execution broker.

#### Scenario: Explicit mode selects the broker
- **GIVEN** the user provides `execution.mode = "cr_dagger_baseline"`
- **WHEN** the evaluation script resolves execution behavior
- **THEN** it instantiates the CR-Dagger baseline broker
- **AND** it does not instantiate `ActionChunkBroker` or `RealTimeChunkBroker`

#### Scenario: Legacy RTC shorthand is normalized
- **GIVEN** `execution.mode` is unset
- **AND** legacy RTC config or CLI shorthand is enabled
- **WHEN** the evaluation script resolves execution behavior
- **THEN** it normalizes the selection to canonical `execution.mode = "rtc"`

#### Scenario: Explicit mode conflicts with legacy RTC input
- **GIVEN** the user provides an explicit `execution.mode`
- **AND** legacy RTC config or CLI shorthand implies a different mode
- **WHEN** the evaluation script resolves execution behavior
- **THEN** setup fails fast with a clear configuration error

### Requirement: CR-Dagger Baseline Execution Mode
The Franka evaluation flow SHALL provide a CR-Dagger-style baseline execution mode that disables RTC and executes a cached base-policy chunk for a fixed execution horizon before re-inference.

#### Scenario: Start a new base horizon
- **GIVEN** Franka evaluation is running with CR-Dagger baseline mode enabled
- **AND** there is no active cached horizon
- **WHEN** the control loop requests the next action
- **THEN** the system calls the base policy exactly once
- **AND** caches the returned action chunk as the authoritative base horizon
- **AND** sets `horizon_start_timestamp` to the canonical `control_timestamp` of the observation that triggered that inference
- **AND** derives `planned_timestamps` from that same canonical control clock
- **AND** returns the first cached action step to the robot

#### Scenario: Re-infer only after the execution window ends
- **GIVEN** an active cached base horizon with `execution_horizon = 10`
- **WHEN** the control loop remains within that horizon's wall-clock execution window
- **THEN** the system SHALL NOT call the base policy again
- **AND** it SHALL continue selecting actions from the cached base horizon
- **AND** once the execution window ends, the next control step SHALL trigger a fresh base-policy inference from the newest observation

### Requirement: Time-Aligned Cached Action Selection
The CR-Dagger baseline mode SHALL align cached action selection to wall-clock time and only tolerate bounded stale-step skipping.

#### Scenario: Control loop keeps pace
- **GIVEN** a cached base horizon and a control loop running near the configured control frequency
- **WHEN** the next action is requested on time
- **THEN** the system returns the next logical cached action step
- **AND** `chunk_idx` increases monotonically by one

#### Scenario: Control loop lags behind within the safety threshold
- **GIVEN** a cached base horizon and a delayed control loop iteration
- **AND** the implied `skip_count` does not exceed `max_skip_steps`
- **WHEN** elapsed wall-clock time advances by more than one control period
- **THEN** the system skips stale cached action steps
- **AND** returns the latest valid action for the current logical time
- **AND** reports the number of skipped steps in metadata

#### Scenario: Control loop lag exceeds the safety threshold
- **GIVEN** a cached base horizon and a delayed control loop iteration
- **AND** the implied `skip_count` exceeds `max_skip_steps`
- **WHEN** the broker evaluates the next control step
- **THEN** the episode is terminated safely
- **AND** the system does not apply a new robot action for that control step
- **AND** only the current episode is terminated
- **AND** the system does not continue chasing future cached poses

### Requirement: RTC Mutual Exclusion
The Franka evaluation flow MUST prevent CR-Dagger baseline mode and RTC mode from being active at the same time.

#### Scenario: Conflicting settings supplied
- **GIVEN** the user enables both CR-Dagger baseline mode and RTC through CLI or config
- **WHEN** evaluation setup begins
- **THEN** the system fails fast with a clear configuration error

### Requirement: Horizon Length Safety
The CR-Dagger baseline mode SHALL validate requested execution length against the available base chunk length and expose the effective horizon explicitly.

#### Scenario: Requested horizon exceeds known model horizon
- **GIVEN** the requested `execute_horizon` is greater than the known model action horizon
- **WHEN** evaluation setup begins
- **THEN** setup fails fast with a clear configuration error

#### Scenario: Returned chunk is shorter than requested at runtime
- **GIVEN** the broker receives a base chunk shorter than the requested `execute_horizon`
- **WHEN** it starts a new base horizon
- **THEN** it sets `effective_horizon = min(requested_execute_horizon, len(base_chunk))`
- **AND** it records both the requested and effective values in metadata
- **AND** it does not index beyond the returned chunk

### Requirement: Remote Policy Compatibility
The CR-Dagger baseline mode SHALL work with both local and websocket-backed policies without requiring realtime sampling support.

#### Scenario: Remote policy in baseline mode
- **GIVEN** the user runs Franka evaluation with `--remote-host` and CR-Dagger baseline mode enabled
- **WHEN** the broker requests a new base horizon
- **THEN** it calls the standard policy `infer()` path
- **AND** no `infer_realtime()` support is required from the server
