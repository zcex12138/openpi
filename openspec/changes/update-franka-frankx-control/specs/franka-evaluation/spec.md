## ADDED Requirements
### Requirement: Frankx Backend Interface
The evaluation scripts SHALL use frankx `Robot` and `Gripper` APIs for Franka robot communication and SHALL not require a robot port parameter.

#### Scenario: Connect via FCI IP only
- **WHEN** the user provides a robot IP
- **THEN** the scripts connect using frankx `Robot(fci_ip)`
- **AND** no robot port configuration is required

---

## MODIFIED Requirements
### Requirement: Action Execution
The evaluation script SHALL execute policy actions on the robot with safety constraints and a selectable control mode.

#### Scenario: Execute TCP pose action in impedance mode
- **WHEN** `control_mode=impedance` and the policy outputs an action chunk of shape `[action_horizon, 8]`
- **THEN** each action uses the format [x, y, z, qw, qx, qy, qz, gripper]
- **AND** the TCP position is clipped to workspace bounds
- **AND** the TCP velocity is limited to `max_pos_speed` m/s
- **AND** the quaternion is normalized before sending
- **AND** the impedance target pose is updated via frankx `ImpedanceMotion`

#### Scenario: Execute TCP pose action in cartesian mode
- **WHEN** `control_mode=cartesian` and the policy outputs an action chunk of shape `[action_horizon, 8]`
- **THEN** each action uses the format [x, y, z, qw, qx, qy, qz, gripper]
- **AND** the TCP position is clipped to workspace bounds
- **AND** the TCP velocity is limited to `max_pos_speed` m/s
- **AND** the quaternion is normalized before sending
- **AND** the control loop updates a frankx `WaypointMotion` via `set_next_waypoint` at `control_fps`

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
- **AND** cartesian mode updates the waypoint target at every control step

#### Scenario: Over-budget step
- **WHEN** a control step exceeds the target period
- **THEN** the script logs a warning and continues without sleeping
