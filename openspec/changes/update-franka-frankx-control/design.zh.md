## 背景
Franka 评估脚本目前依赖自定义 `RobotClient`（franka_control）并通过 IP/port socket 通信。需要迁移至 frankx（仅 FCI IP）并新增笛卡尔位置控制模式，同时保留阻抗控制行为。

## 目标 / 非目标
- 目标：
  - Franka 交互统一使用 frankx `Robot`/`Gripper`。
  - 提供可选控制模式（默认 `impedance`，可选 `cartesian`）。
  - 维持现有观测/动作语义与安全检查。
- 非目标：
  - 重构训练数据格式或策略接口。
  - 新增传感器或修改相机服务协议。

## 关键决策
- **机器人接口**：使用 frankx `Robot(fci_ip)` 与 `robot.get_gripper()`；完全移除 robot port 参数。
- **阻抗控制**：通过 `ImpedanceMotion(translational_stiffness, rotational_stiffness)` 实现并更新目标；移除阻尼比参数以匹配 frankx API。
- **笛卡尔位置控制**：维护一个长期运行的 `WaypointMotion(return_when_finished=False)`，并在每个控制步通过 `set_next_waypoint` 更新目标（绝对位姿），以 `control_fps` 固定频率控制。
- **状态映射**：从 `RobotState` 构建 14D 观测（`O_T_EE` 作为位姿、`O_F_ext_hat_K` 作为力矩）并加入夹爪状态。
- **可视化**：用 `O_T_EE`（当前）与 `O_T_EE_d`（目标）绘制位姿；用 `O_F_ext_hat_K` 绘制力/力矩。

## 风险 / 权衡
- frankx 状态字段较底层，坐标系/帧解释若出错会影响控制或可视化。
- 基于 Waypoint 的笛卡尔控制与阻抗目标行为可能不同，需要调参与更新率验证。

## 迁移计划
1. 更新配置/CLI，移除 robot port 参数。
2. 替换为 frankx 接口并验证阻抗模式。
3. 新增笛卡尔控制模式并在硬件上验证。
4. 更新可视化工具与文档。

## 开放问题
- 无（参数与默认值已确认）。
