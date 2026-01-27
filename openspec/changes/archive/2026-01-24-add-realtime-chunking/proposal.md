# Proposal: Add Real-Time Chunking to π₀.5 Inference

## Summary

引入 Real-Time Chunking (RTC) 方法到 π₀.5 推理流程，解决 action chunking 在实际机器人控制中的延迟问题。RTC 允许在模型推理期间继续执行先前生成的动作，同时通过 prefix attention 机制保持动作序列的时序一致性。

## Motivation

当前 `ActionChunkBroker` 实现存在以下问题：

1. **阻塞式推理**：每次推理生成完整 action chunk (30 步)，但在推理期间机器人无法执行任何动作
2. **推理延迟浪费**：典型 π₀.5 推理耗时 ~100-200ms，期间机器人处于静止状态
3. **动作不连续**：chunk 边界处可能出现动作跳变

RTC 通过以下机制解决这些问题：
- **重叠执行**：在生成新 chunk 时，继续执行旧 chunk 中的动作
- **Prefix Attention**：新 chunk 的生成条件化于正在执行的动作前缀，保证时序一致性
- **Guidance Correction**：可选的梯度引导修正，进一步提升轨迹平滑度

## Scope

### In Scope
- `RealTimeChunkBroker` 类：替代现有 `ActionChunkBroker`
- `Pi0.realtime_sample_actions()` 方法：支持 prefix-conditioned 推理
- 配置参数：`inference_delay`, `execute_horizon`, `prefix_attention_horizon`
- Franka 评估示例集成

### Out of Scope
- Training-Time RTC（需要重新训练模型，不在本次范围）
- Guidance Correction（可作为后续增强）
- BID (Best-of-N with Importance weighting and Diversity) 采样

## Design Decisions

### 1. 推理模式选择

**选项 A: Naive RTC**
- 仅实现执行重叠，不修改模型推理
- 优点：实现简单，无需修改 `pi0.py`
- 缺点：chunk 边界处可能不连续

**选项 B: Prefix-Conditioned RTC** ✓ 推荐
- 推理时将正在执行的动作作为前缀条件
- 优点：动作序列更平滑
- 缺点：需要修改模型采样逻辑

选择 B，因为这是 RTC 论文的核心贡献，能显著提升轨迹质量。

### 2. 实现层级

**选项 A: 仅在 Client 层**
- 修改 `ActionChunkBroker`，不触及模型代码
- 限制：无法实现 prefix attention

**选项 B: Model + Client 协作** ✓ 推荐
- Model 层新增 `realtime_sample_actions()`
- Client 层新增 `RealTimeChunkBroker` 调度执行
- 优点：完整实现 RTC 算法

### 3. 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `inference_delay` | 3 | 推理期间执行的旧 chunk 动作数 |
| `execute_horizon` | 5 | 每次迭代总执行动作数 |
| `prefix_attention_horizon` | 10 | prefix attention 衰减范围 |
| `prefix_attention_schedule` | "linear" | 衰减方式 (linear/exp/ones/zeros) |

## References

- [Real-Time Chunking Kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- Paper: "Real-Time Execution of Action Chunking Flow Policies"
- Paper: "Training-Time Action Conditioning for Efficient Real-Time Chunking"
