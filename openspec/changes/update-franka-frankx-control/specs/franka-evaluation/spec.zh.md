## 新增需求
### 需求：Frankx 后端接口
评估脚本 **必须** 使用 frankx 的 `Robot` 与 `Gripper` API 进行 Franka 通信，并且 **不得** 需要 robot port 参数。

#### 场景：仅通过 FCI IP 连接
- **当** 用户提供 robot IP  
- **则** 脚本使用 frankx `Robot(fci_ip)` 连接  
- **且** 不需要 robot port 配置  

---

## 修改需求
### 需求：动作执行
评估脚本 **必须** 在安全约束下执行策略动作，并提供可选控制模式。

#### 场景：阻抗模式下执行 TCP 位姿动作
- **当** `control_mode=impedance` 且策略输出形状为 `[action_horizon, 8]` 的动作块  
- **则** 每个动作为 [x, y, z, qw, qx, qy, qz, gripper]  
- **且** TCP 位置裁剪到工作空间边界  
- **且** TCP 速度限制为 `max_pos_speed` m/s  
- **且** 发送前归一化四元数  
- **且** 通过 frankx `ImpedanceMotion` 更新阻抗目标位姿  

#### 场景：笛卡尔模式下执行 TCP 位姿动作
- **当** `control_mode=cartesian` 且策略输出形状为 `[action_horizon, 8]` 的动作块  
- **则** 每个动作为 [x, y, z, qw, qx, qy, qz, gripper]  
- **且** TCP 位置裁剪到工作空间边界  
- **且** TCP 速度限制为 `max_pos_speed` m/s  
- **且** 发送前归一化四元数  
- **且** 控制循环在 `control_fps` 下通过 `set_next_waypoint` 更新 frankx `WaypointMotion`  

#### 场景：夹爪控制
- **当** 启用夹爪控制且策略预测 gripper > 0.7  
- **则** 夹爪在该步执行闭合  

---

### 需求：控制循环时序
评估脚本 **必须** 使用与 openpi 示例一致的同步控制循环。

#### 场景：分块同步循环
- **当** `control_fps=30` 且 `open_loop_horizon=10`  
- **则** 脚本以 1/30s（尽力而为）执行一次动作  
- **且** 每 10 个控制步调用一次策略推理  
- **且** 笛卡尔模式在每个控制步更新 waypoint 目标  

#### 场景：超时控制步
- **当** 某控制步超过目标周期  
- **则** 脚本记录警告并继续，不再 sleep  
