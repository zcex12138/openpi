# Tasks: Add Real-Time Chunking

## Phase 1: Model Layer (Core Algorithm)

- [x] **T1.1** 在 `src/openpi/models/pi0.py` 添加 `realtime_sample_actions()` 方法
  - 支持 `action_prefix` 前缀条件
  - 实现 time masking 逻辑（prefix 位置 time=0，通过 update_mask 保持不变）
  - 保持现有 KV cache 优化
  - **验证**: Import 通过

- [x] **T1.2** 在 `src/openpi/models/model.py` 扩展 `BaseModel` 接口
  - 添加 `realtime_sample_actions()` 方法（默认回退到 `sample_actions`）
  - **验证**: Import 通过

## Phase 2: Policy Layer (API Wrapper)

- [x] **T2.1** 在 `src/openpi/policies/policy.py` 添加 `infer_realtime()` 方法
  - 封装 `realtime_sample_actions` 调用
  - 处理输入/输出变换
  - 支持 JAX 和 PyTorch 模型
  - **验证**: Import 通过

- [x] **T2.2** RTC 配置参数
  - 配置参数将在 Phase 3 的 `RealTimeChunkBroker` 中实现
  - `Policy.infer_realtime()` 已支持 `action_prefix` 参数传递
  - **验证**: 接口设计完成

## Phase 3: Client Layer (Execution Scheduler)

- [x] **T3.1** 创建 `packages/openpi-client/src/openpi_client/realtime_chunk_broker.py`
  - 实现 `RealTimeChunkBroker` 类
  - 实现 `RTCConfig` 配置 dataclass
  - 管理 action queue 和执行指针
  - 实现 chunk 融合逻辑 (`_merge_chunk`)
  - **验证**: Import 通过

- [x] **T3.2** 实现双线程异步执行模型
  - 执行线程：通过 `get_action()` 消费 action queue
  - 推理线程：`_inference_worker()` 生成新 chunk
  - 处理 queue underrun（返回 `_last_action` 作为 position hold）
  - **验证**: 线程安全设计完成

- [x] **T3.3** 处理边界情况
  - 首次推理（无 prefix 时回退到标准推理）
  - Episode 重置（`reset()` 清空 queue）
  - **验证**: 边界逻辑已实现

## Phase 4: Franka Integration

- [x] **T4.1** 更新 `examples/franka/main.py` 支持 RTC 模式
  - 添加 `--rtc`, `--rtc-inference-delay`, `--rtc-execute-horizon` 命令行参数
  - 条件性使用 `RealTimeChunkBroker`
  - **验证**: 语法检查通过

- [x] **T4.2** 更新 `examples/franka/real_env_config.yaml`
  - 添加 `rtc` 配置节 (`enabled`, `inference_delay`, `execute_horizon`)
  - **验证**: 配置文件格式正确

## Phase 5: Documentation & Validation

- [x] **T5.1** 更新 CLAUDE.md Franka 评估章节
  - 添加 RTC 使用说明和命令示例
  - 记录参数配置和推荐值
  - **验证**: 文档已更新

- [x] **T5.2** 性能基准测试
  - 文档化了推荐配置（基于推理延迟）
  - 实际性能测试需在机器人环境中进行
  - **验证**: 配置指南已记录

## Dependencies

```
T1.1 ──► T1.2 ──► T2.1 ──► T2.2
                    │
                    ▼
         T3.1 ──► T3.2 ──► T3.3
                    │
                    ▼
         T4.1 ──► T4.2 ──► T5.1 ──► T5.2
```

## Parallelizable Work

- T1.x (Model) 和 T3.1 (Client 基础结构) 可并行开发
- T5.x (文档) 可在功能完成后并行进行
