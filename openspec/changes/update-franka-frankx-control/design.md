## Context
The Franka evaluation scripts currently depend on a custom `RobotClient` (franka_control) with IP/port sockets. We must migrate to frankx (FCI IP only) and add a cartesian position control mode while retaining impedance control behavior.

## Goals / Non-Goals
- Goals:
  - Use frankx `Robot`/`Gripper` for all Franka robot interactions.
  - Provide a selectable control mode (`impedance` default, `cartesian` optional).
  - Maintain the existing observation/action semantics and safety checks.
- Non-Goals:
  - Rework training data formats or policy interfaces.
  - Add new sensors or change camera service protocols.

## Decisions
- **Robot interface**: Use frankx `Robot(fci_ip)` and `robot.get_gripper()`; drop robot port parameters entirely.
- **Impedance control**: Implement via `ImpedanceMotion(translational_stiffness, rotational_stiffness)` and update targets; remove damping ratio parameters to align with frankx API.
- **Cartesian position control**: Maintain a long-running `WaypointMotion(return_when_finished=False)` and update targets each control step via `set_next_waypoint` using absolute poses; this supports fixed-rate control at `control_fps`.
- **State mapping**: Build the 14D observation from `RobotState` (`O_T_EE` for pose, `O_F_ext_hat_K` for wrench) plus gripper state.
- **Visualization**: Use `O_T_EE` (current) and `O_T_EE_d` (target) for pose plotting; use `O_F_ext_hat_K` for wrench plots.

## Risks / Trade-offs
- frankx state fields are low-level and require correct frame interpretation; mapping mistakes could affect control or visualization.
- Waypoint-based cartesian control may behave differently than impedance targets; careful tuning of dynamics and update rate is required.

## Migration Plan
1. Update configs/CLI to remove robot port parameters.
2. Swap in frankx interfaces and verify impedance mode.
3. Add cartesian control mode and validate on hardware.
4. Update visualization tools and docs.

## Open Questions
- None (parameters and defaults confirmed).
