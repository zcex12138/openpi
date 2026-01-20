# 规格：Franka 机器人评估

## 概述

该能力用于在 Franka Panda 真实机器人上评估 openpi 训练的策略。它提供命令行接口来加载 checkpoint 并以实时控制方式运行评估 episode。

## 新增需求

### 需求：本地策略加载
在未启用远程模式时，评估脚本 SHALL 使用 openpi 标准的 `policy_config.create_trained_policy()` 进行本地 checkpoint 加载。

#### Scenario: 加载本地 JAX checkpoint
- **WHEN** 用户运行 `uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora`
- **THEN** 策略通过 `policy_config.create_trained_policy()` 创建
- **AND** 归一化统计量从 checkpoint assets 中加载

#### Scenario: 加载本地 PyTorch checkpoint
- **WHEN** checkpoint 目录中包含 `model.safetensors`
- **THEN** 系统检测到 PyTorch 格式并加载 PyTorch 权重

---

### 需求：远程策略推理
评估脚本 SHALL 支持通过 websocket 策略服务器进行可选的远程推理。

#### Scenario: 连接远程策略服务器
- **WHEN** 用户提供 `--remote-host` 与 `--remote-port`
- **THEN** 脚本使用 `WebsocketClientPolicy` 发送观测并接收动作块
- **AND** 跳过本地 checkpoint 加载

---

### 需求：观测采集
评估脚本 SHALL 采集与训练数据格式一致的观测。

#### Scenario: 相机服务集成
- **WHEN** 外部相机服务（Python 3.9）运行且可访问
- **THEN** 脚本通过相机客户端获取最新的 L500/D400 RGB 帧
- **AND** 将帧用于填充 `observation/image` 与 `observation/wrist_image`

#### Scenario: 采集相机观测
- **WHEN** 机器人连接 L500（基座）与 D400（腕部）相机
- **THEN** `observation/image` 为 224x224 的 L500 RGB 图像
- **AND** `observation/wrist_image` 为 224x224 的 D400 RGB 图像

#### Scenario: 采集机器人状态
- **WHEN** 采集观测
- **THEN** `observation/state` 的前 7 维包含 TCP pose [x, y, z, qw, qx, qy, qz]
- **AND** 可以包含额外状态维度

#### Scenario: 相机服务不可用
- **WHEN** 相机服务不可达或超时
- **THEN** 脚本记录警告
- **AND** 可使用零图像作为降级运行方案

---

### 需求：动作执行
评估脚本 SHALL 在安全约束下执行策略动作。

#### Scenario: 执行 TCP pose 动作
- **WHEN** 策略输出形状为 `[action_horizon, 8]` 的动作块
- **THEN** 每步动作格式为 [x, y, z, qw, qx, qy, qz, gripper]
- **AND** TCP 位置被裁剪到工作空间边界
- **AND** TCP 速度限制为 `max_pos_speed` m/s
- **AND** quaternion 在发送前归一化

#### Scenario: 夹爪控制
- **WHEN** 夹爪控制启用且策略预测 gripper > 0.7
- **THEN** 该步夹爪闭合

---

### 需求：控制循环时序
评估脚本 SHALL 使用与 openpi 示例一致的同步控制循环。

#### Scenario: 同步动作块循环
- **WHEN** `control_fps=30` 且 `open_loop_horizon=10`
- **THEN** 脚本以 1/30s 执行一次动作（尽力保证）
- **AND** 每 10 个控制步调用一次策略推理

#### Scenario: 超时步处理
- **WHEN** 某次控制步超过目标周期
- **THEN** 脚本记录警告并继续，不进行 sleep

---

### 需求：Episode 管理
评估脚本 SHALL 以用户交互方式管理评估 episode。

#### Scenario: 运行多个 episode
- **WHEN** `num_episodes=10`
- **THEN** 每个 episode 前等待用户确认
- **AND** 若启用重置，则在 episode 间执行复位
- **AND** 评估结束后保存 episode 汇总

#### Scenario: 提前终止
- **WHEN** 用户在 episode 中按下 Ctrl+C
- **THEN** 安全停止阻抗控制
- **AND** 若启用夹爪控制则打开夹爪
- **AND** 脚本干净退出

---

### 需求：视频录制
评估脚本 SHALL 可选地录制评估 episode。

#### Scenario: 保存 episode 视频
- **WHEN** `save_video=True` 且 episode 结束
- **THEN** 视频文件保存到 `output_dir` 并包含相机视角
- **AND** 文件名包含 episode 编号与时间戳
