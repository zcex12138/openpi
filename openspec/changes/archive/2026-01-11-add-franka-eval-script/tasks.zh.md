# 任务：新增 Franka 机器人评估脚本

## Phase 1：核心基础设施

- [ ] **T1.1** 创建 `examples/franka/` 目录结构
- [ ] **T1.2** 创建 `examples/franka/constants.py`，包含默认机器人/相机/控制常量
- [ ] **T1.3** 创建 `examples/franka/real_env.py` 并实现 `FrankaRealEnv`
  - 基于 `reactive_diffusion_policy` 适配机器人集成
  - 实现机器人状态采集
  - 实现带速度限制与工作空间裁剪的动作执行
- [ ] **T1.4** 创建 `examples/franka/camera_service.py`（Python 3.9）并实现 IPC
  - 通过 length-prefixed msgpack 协议提供 L500/D400 RGB 帧
  - 以 provider 接口对接 RDP 相机驱动
- [ ] **T1.5** 创建 `examples/franka/camera_client.py` 作为 Python 3.9 相机服务客户端
  - 连接外部相机服务（RDP 栈）
  - 带超时的 L500/D400 RGB 帧获取
- [ ] **T1.6** 创建 `examples/franka/env.py` 并实现 `FrankaEnvironment`
  - 实现 `openpi_client.runtime.environment.Environment`
  - 使观测与 `FrankaInputs` 期望一致（observation/image、observation/wrist_image、observation/state）
  - 应用单步动作（动作分块由 ActionChunkBroker 处理）

## Phase 2：主脚本

- [ ] **T2.1** 创建 `examples/franka/main.py` 入口脚本
  - 通过 `policy_config.create_trained_policy()` 加载 checkpoint
  - 使用 tyro 解析命令行参数
  - 同步支持本地推理与远程服务模式
  - 使用 `Runtime` + `PolicyAgent` + `ActionChunkBroker`
  - 向 `output_dir` 写入 episode 汇总（CSV/JSON）

## Phase 3：文档与测试

- [ ] **T3.1** 创建 `examples/franka/README.md`
  - 前置条件（C++ franka_control 服务、相机设置）
  - 本地与远程模式的使用示例
  - 参数说明
  - 安全指南

- [ ] **T3.2** Manual test on real robot (example: `./checkpoints/11999`)
  - 验证观测格式与训练数据一致
  - 验证动作格式与维度（8D：position[3] + quaternion[4] + gripper[1]）

## 依赖关系

- T1.3 依赖 T1.1、T1.2
- T1.4 依赖 T1.3
- T1.5 依赖 T1.4
- T1.6 依赖 T1.5
- T2.1 依赖 T1.6
- T3.1 可与 T2.1 并行
- T3.2 依赖 T2.1

## 验证

每项任务需通过以下验证：
- T1.x：代码可编译，import 正常
- T2.1：脚本 `--help` 能正常运行
- T3.1：文档完整且准确
- T3.2：真实机器人端到端评估运行成功
