## 1. Implementation
- [x] 1.1 Remove robot port configuration and CLI flags; update constants and camera_config defaults.
- [x] 1.2 Replace RobotClient usage in `FrankaRealEnv` with frankx `Robot`/`Gripper` and map robot state to the expected 14D observation.
- [x] 1.3 Implement control mode switch with default `impedance` and add cartesian position control using `WaypointMotion.set_next_waypoint` at `control_fps`.
- [x] 1.4 Update visualization scripts to use frankx `RobotState` (`O_F_ext_hat_K`, `O_T_EE`, `O_T_EE_d`).
- [x] 1.5 Update README/INSTALL docs to reflect frankx setup and CLI changes.
- [x] 1.6 Add/adjust manual validation steps (robot-connected smoke runs) in docs or comments.

## 2. Validation
- [ ] 2.1 Manual: run `uv run examples/franka/main.py --checkpoint-dir ... --config ...` in impedance mode.
- [ ] 2.2 Manual: run the same script with `--control-mode cartesian` and verify target tracking.
- [ ] 2.3 Manual: run `uv run examples/franka/visualize_wrench.py` and `uv run examples/franka/visualize_online_trajectory.py` to confirm wrench/pose updates.
