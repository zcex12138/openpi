# Change: Replace Franka control backend with frankx and add cartesian control mode

## Why
The Franka examples must use the frankx interfaces available at `~/workspace/yhx/frankx` instead of the legacy franka_control client. We also need a selectable cartesian position control mode with a fixed control frequency while keeping the existing impedance mode.

## What Changes
- Replace Franka robot communication in `examples/franka` with frankx `Robot`/`Gripper` APIs.
- Remove robot port configuration and CLI flags; use FCI IP only.
- Add a control mode switch (`impedance` vs `cartesian`) with default `impedance`.
- Implement cartesian position control via `WaypointMotion.set_next_waypoint` at `control_fps`.
- Keep impedance control via `ImpedanceMotion` with translational/rotational stiffness only (no damping parameters).
- Use frankx `RobotState` fields for wrench and target pose visualization.
- Update docs/configs to reflect frankx usage.

## Impact
- Affected specs: `specs/franka-evaluation/spec.md`
- Affected code: `examples/franka/*.py`, `examples/franka/README.md`, `examples/franka/INSTALL.md`, `examples/franka/constants.py`, `examples/franka/camera_config.yaml`
