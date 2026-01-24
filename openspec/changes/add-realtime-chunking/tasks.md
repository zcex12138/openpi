# Tasks: Add Real-Time Chunking

## Phase 1: Model Layer (Core Algorithm)

- [ ] **T1.1** 在 `src/openpi/models/pi0.py` 添加 `realtime_sample_actions()` 方法
  - 支持 `prev_chunk` 前缀条件
  - 实现 time masking 逻辑（prefix 位置 time=0）
  - 保持现有 KV cache 优化
  - **验证**: 单元测试确认 prefix 位置动作不变

- [ ] **T1.2** 在 `src/openpi/models/model.py` 扩展 `BaseModel` 接口
  - 添加 `realtime_sample_actions()` 抽象方法（可选实现）
  - **验证**: 类型检查通过

## Phase 2: Policy Layer (API Wrapper)

- [ ] **T2.1** 在 `src/openpi/policies/policy.py` 添加 `infer_realtime()` 方法
  - 封装 `realtime_sample_actions` 调用
  - 处理输入/输出变换
  - 返回完整 chunk 及 `inference_delay` 元数据
  - **验证**: 集成测试确认端到端推理正确

- [ ] **T2.2** 添加 RTC 配置参数到 `PolicyConfig`
  - `rtc_enabled: bool = False`
  - `inference_delay: int = 3`
  - `execute_horizon: int = 5`
  - `prefix_attention_horizon: int = 10`
  - **验证**: 配置加载测试

## Phase 3: Client Layer (Execution Scheduler)

- [ ] **T3.1** 创建 `packages/openpi-client/src/openpi_client/realtime_chunk_broker.py`
  - 实现 `RealTimeChunkBroker` 类
  - 管理 action queue 和执行指针
  - 实现 chunk 融合逻辑
  - **验证**: 单元测试模拟推理/执行交替

- [ ] **T3.2** 实现双线程异步执行模型
  - 执行线程：按 Hz 消费 action queue
  - 推理线程：生成新 chunk 并融合
  - 处理 queue underrun（保持最后动作）
  - **验证**: 压力测试确认无死锁/竞态

- [ ] **T3.3** 处理边界情况
  - 首次推理（无 prev_chunk）
  - Episode 重置（清空 queue）
  - **验证**: 边界测试用例

## Phase 4: Franka Integration

- [ ] **T4.1** 更新 `examples/franka/main.py` 支持 RTC 模式
  - 添加 `--rtc` 命令行参数
  - 条件性使用 `RealTimeChunkBroker`
  - **验证**: 手动评估确认机器人运动更平滑

- [ ] **T4.2** 更新 `examples/franka/real_env_config.yaml`
  - 添加 RTC 相关配置项
  - **验证**: 配置加载正确

## Phase 5: Documentation & Validation

- [ ] **T5.1** 更新 CLAUDE.md Franka 评估章节
  - 添加 RTC 使用说明
  - 记录推荐参数配置

- [ ] **T5.2** 性能基准测试
  - 对比 RTC vs 标准 chunking 的有效控制频率
  - 记录推理延迟分布
  - **验证**: 文档化测试结果

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
