# 提案：新增 Franka 机器人评估脚本

## 摘要

新增一个 Franka Panda 真实机器人评估脚本，使用户能够在真实硬件上评估微调后的 openpi checkpoints（如 `pi05_franka_screwdriver_lora`）。该脚本采用 `Environment` 类（对齐 `examples/aloha_real` 的结构）以及 openpi 的同步推理模式（单循环、动作块执行），并复用 `reactive_diffusion_policy` 的机器人通信基础设施。由于指定版本的 RealSense/XenseCamera 仅支持 Python 3.9，相机采集通过独立的 Python 3.9 进程提供服务，openpi 侧通过轻量相机客户端连接。

## 动机

- 用户已经在自定义 Franka 数据集（如 `single_arm_screwdriver`）上微调了 openpi 模型，需要在真实硬件上评估这些 checkpoint
- 现有的 `franka_policy.py` 提供了输入/输出变换，但缺少完整的评估流程
- `reactive_diffusion_policy/eval_real_robot_franka.py` 中的 `FrankaRealRunner` 提供了经过验证的机器人通信实现，可作为参考

## 范围

### 包含
1. 创建 `examples/franka/main.py`：遵循 openpi 风格的主评估脚本
2. 创建 `examples/franka/real_env.py`：底层 Franka 机器人集成
3. 创建 `examples/franka/camera_service.py`：Python 3.9 相机服务（IPC）
4. 创建 `examples/franka/camera_client.py`：外部 Python 3.9 相机服务客户端
5. 创建 `examples/franka/env.py`：适配 openpi runtime 的 `Environment` 封装
6. 创建 `examples/franka/constants.py`：Franka 机器人共享常量
7. 创建 `examples/franka/README.md`：使用说明文档，包含相机服务启动说明

### 不包含
- 修改核心策略推理代码
- 相机/传感器驱动实现（复用 `reactive_diffusion_policy`）
- 在 openpi（Python 3.11）内直接运行相机驱动
- ROS2 集成（脚本保持 ROS2 无依赖）
- 多机器人或多臂支持

## 设计概览

评估脚本默认支持本地进程内推理，同时也提供可选的客户端-服务端模式（下图展示远程模式），与其他 openpi 示例一致。相机采集由独立的 Python 3.9 服务进程提供：

```
                              +-----------------+
                              | serve_policy.py |
                              | (Policy Server) |
                              +--------+--------+
                                       |
                                       | WebSocket
                                       |
+----------------+             +-------v--------+         +--------------------+
| Franka Robot   |<----------->| examples/franka|<------->| Camera Service     |
| C++ Controller |   TCP/IP    | /main.py       |  IPC    | (Python 3.9, RDP)  |
+----------------+             +----------------+         +--------------------+
```

### 关键组件

1. **main.py**：入口脚本，负责：
   - 通过 `policy_config.create_trained_policy()` 加载训练策略
   - 组装 `Runtime` + `FrankaEnvironment` + `PolicyAgent` + `ActionChunkBroker`
   - 运行评估 episode

2. **real_env.py**：`FrankaRealEnv` 类，负责：
   - 通过 `RobotClient`（来自 `reactive_diffusion_policy`）连接 Franka 机器人
   - 采集机器人状态并在安全约束下执行动作

3. **camera_service.py**：相机服务，负责：
   - 在 Python 3.9 环境运行 RealSense/XenseCamera 驱动
   - 通过轻量 IPC 协议提供最新的 L500/D400 RGB 帧

4. **camera_client.py**：相机客户端，负责：
   - 连接外部 Python 3.9 相机服务
   - 获取 L500/D400 RGB 帧用于观测

5. **env.py**：`FrankaEnvironment` 类，负责：
   - 实现 `openpi_client.runtime.environment.Environment`
   - 格式化观测以匹配 `FrankaInputs` 期望
   - 应用单步动作（动作分块由 `ActionChunkBroker` 处理）

6. **constants.py**：机器人配置常量（IP、端口、FPS、工作空间边界等）

## 备选方案

1. **远程策略服务器模式**：使用 `serve_policy.py` + websocket 客户端（类似 DROID 示例）
   - 优点：推理与机器人控制分离，支持远端部署
   - 缺点：增加延迟，设置更复杂
   - 结论：支持本地为默认，远程为可选；两种模式均保持同步执行

2. **直接集成到 serve_policy.py**
   - 缺点：将服务端职责与机器人特定代码混在一起
   - 结论：机器人特定逻辑保持在 examples/ 中

## 依赖

- `reactive_diffusion_policy.local_franka.bootstrap`：初始化 `RobotClient`
- `robot_client`：Franka C++ 控制器 TCP 客户端
- `reactive_diffusion_policy` 的 Python 3.9 相机服务（RealSense + XenseCamera）
- OpenCV：相机图像处理
- `openpi_client` runtime 与 websocket 客户端（远程模式需要）

## 风险

1. **机器人安全**：必须正确配置工作空间边界与速度限制
   - 缓解：使用保守默认边界，每个 episode 前要求用户确认

2. **硬件兼容性**：不同 Franka 设置可能使用不同相机配置
   - 缓解：支持通过命令行参数配置相机名称

3. **循环时序**：同步推理在低性能硬件上可能降低控制频率
   - 缓解：使用动作块（open-loop horizon）减少推理调用，并在步长超时时记录日志

4. **跨进程相机延迟**：相机服务 IPC 可能引入延迟或帧过期
   - 缓解：使用最新帧语义，附带时间戳并在帧过期/缺失时记录日志

## 成功标准

- 用户可通过 `uv run examples/franka/main.py --checkpoint-dir ./checkpoints/11999 --config pi05_franka_screwdriver_lora` 进行评估
- 脚本以同步方式尽力保持目标控制频率
- 动作块按控制频率顺序执行
- 动作被正确裁剪到工作空间边界
- 评估 episode 的视频记录被保存
- 相机帧来自外部 Python 3.9 相机服务
