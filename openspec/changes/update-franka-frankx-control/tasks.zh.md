## 1. 实现
- [x] 1.1 移除 robot port 配置与 CLI 参数；更新 constants 与 camera_config 默认值。
- [x] 1.2 在 `FrankaRealEnv` 中用 frankx `Robot`/`Gripper` 替换 `RobotClient`，并将机器人状态映射为期望的 14D 观测。
- [x] 1.3 实现控制模式开关（默认 `impedance`），新增使用 `WaypointMotion.set_next_waypoint` 在 `control_fps` 下的笛卡尔位置控制。
- [x] 1.4 更新可视化脚本以使用 frankx `RobotState`（`O_F_ext_hat_K`、`O_T_EE`、`O_T_EE_d`）。
- [x] 1.5 更新 README/INSTALL 文档以反映 frankx 设定与 CLI 变化。
- [x] 1.6 在文档或注释中补充/调整手动验证步骤（连接机器人运行的 smoke 测试）。

## 2. 验证
- [ ] 2.1 手动：在阻抗模式运行 `uv run examples/franka/main.py --checkpoint-dir ... --config ...`。
- [ ] 2.2 手动：使用 `--control-mode cartesian` 运行同脚本并验证目标跟踪。
- [ ] 2.3 手动：运行 `uv run examples/franka/visualize_wrench.py` 与 `uv run examples/franka/visualize_online_trajectory.py` 验证力/位姿更新。
