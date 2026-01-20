# 变更：用 frankx 替换 Franka 控制后端，并新增笛卡尔控制模式

## 为什么
Franka 示例必须使用 `~/workspace/yhx/frankx` 中的 frankx 接口替代旧的 franka_control 客户端。同时需要在保留现有阻抗控制模式的前提下，新增一个可选择的、固定频率的笛卡尔位置控制模式。

## 变更内容
- 将 `examples/franka` 中的 Franka 机器人通信替换为 frankx 的 `Robot`/`Gripper` API。
- 移除 robot port 配置与 CLI 参数，仅使用 FCI IP。
- 新增控制模式切换（`impedance` 与 `cartesian`），默认 `impedance`。
- 使用 `WaypointMotion.set_next_waypoint` 在 `control_fps` 下实现笛卡尔位置控制。
- 使用 `ImpedanceMotion` 保持阻抗控制，仅保留平移/旋转刚度（不含阻尼参数）。
- 使用 frankx `RobotState` 字段进行力/力矩与目标位姿可视化。
- 更新文档与配置以反映 frankx 用法。

## 影响范围
- 受影响的规范：`specs/franka-evaluation/spec.md`
- 受影响的代码：`examples/franka/*.py`、`examples/franka/README.md`、`examples/franka/INSTALL.md`、`examples/franka/constants.py`、`examples/franka/camera_config.yaml`
