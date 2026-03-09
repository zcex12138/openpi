# realtime-chunking Specification

## Purpose
TBD - created by archiving change add-realtime-chunking. Update Purpose after archive.
## Requirements
### Requirement: Prefix-Conditioned Sampling
系统 SHALL 支持在 rotate6d 表示下的前缀条件化采样，自动处理 action_prefix 的维度转换。

#### Scenario: RTC with rotate6d action prefix auto-conversion
- **GIVEN** 模型使用 `rotation_representation="r6d"`（内部 10D 动作）
- **AND** broker 缓存的 action_prefix 是 8D quat 格式（经 output_transform 后）
- **WHEN** `policy.infer_realtime(obs, action_prefix=prefix_8d)` 被调用
- **THEN** policy 检测到 `action_prefix.shape[-1]`（8）与模型期望的动作维度（10）不匹配
- **AND** 自动将 8D quat prefix 转换为 10D r6d prefix（通过 QuatToRotate6d）
- **AND** 转换后的 prefix 传入模型进行条件化采样
- **AND** 模型输出经 output_transform 转回 8D quat 返回给 broker

#### Scenario: RTC with quaternion representation (backward compat)
- **GIVEN** 模型使用 `rotation_representation="quat"`（内部 8D 动作）
- **WHEN** `policy.infer_realtime(obs, action_prefix=prefix_8d)` 被调用
- **THEN** 行为与现有实现完全一致（无维度转换）

### Requirement: RealTimeChunkBroker Execution Scheduling
系统 SHALL 提供 `RealTimeChunkBroker` 类，协调推理与执行的时序，实现动作队列管理和 chunk 融合。

#### Scenario: 推理与执行并行
- **Given** `RealTimeChunkBroker` 配置 `inference_delay=3, execute_horizon=5`
- **When** 机器人以 10Hz 执行动作
- **And** 模型推理耗时 300ms
- **Then** 在推理期间，机器人继续执行旧 chunk 的 3 个动作
- **And** 推理完成后，无缝切换到新 chunk

#### Scenario: Queue Underrun 处理
- **Given** action queue 中剩余动作少于 `min_buffer`
- **When** 执行线程请求下一个动作
- **Then** 返回最后一个有效动作（position hold）
- **And** 记录 underrun 警告日志

#### Scenario: Episode 重置
- **Given** episode 结束触发 `broker.reset()`
- **When** 新 episode 开始
- **Then** action queue 被清空
- **And** prev_chunk 重置为 None

---

### Requirement: RTC Configuration
系统 SHALL 支持通过配置参数控制 RTC 行为。

#### Scenario: 启用 RTC 模式
- **Given** 配置 `rtc_enabled=True`
- **When** 创建 Policy 实例
- **Then** 使用 `RealTimeChunkBroker` 而非 `ActionChunkBroker`

#### Scenario: 自定义 RTC 参数
- **Given** 配置:
  ```yaml
  rtc_enabled: true
  inference_delay: 5
  execute_horizon: 8
  prefix_attention_horizon: 15
  ```
- **When** 运行评估
- **Then** 每次迭代执行 8 个动作
- **And** 推理期间执行旧 chunk 的 5 个动作
- **And** prefix attention 在 15 步内衰减

---

### Requirement: Policy RTC Interface
`Policy` 类 SHALL 提供 `infer_realtime()` 方法，封装 prefix-conditioned 推理逻辑。

#### Scenario: 调用 realtime 推理
- **Given** 已加载 π₀.5 模型的 Policy 实例
- **When** 调用 `policy.infer_realtime(obs, prev_chunk, inference_delay=3)`
- **Then** 返回字典包含:
  - `actions`: shape `[action_horizon, action_dim]`
  - `inference_delay`: 使用的 delay 值
  - `policy_timing`: 推理耗时

#### Scenario: 模型不支持 RTC
- **Given** 模型未实现 `realtime_sample_actions()`
- **When** 调用 `policy.infer_realtime()`
- **Then** 回退到标准 `infer()` 并记录警告

---

### Requirement: Thread-Safe Action Queue
RTC 的 action queue MUST 支持多线程安全访问。

#### Scenario: 并发读写
- **Given** 执行线程以 10Hz 消费动作
- **And** 推理线程以 ~3Hz 生成新 chunk
- **When** 两线程并发运行
- **Then** 无数据竞争或死锁
- **And** 动作顺序正确

---

### Requirement: Franka RTC Evaluation
Franka 评估示例 SHALL 支持 RTC 模式。

#### Scenario: 命令行启用 RTC
- **Given** 运行命令:
  ```bash
  python examples/franka/main.py --rtc --inference-delay 3 --execute-horizon 5
  ```
- **When** 评估开始
- **Then** 使用 RTC 模式执行
- **And** 日志显示 RTC 配置

#### Scenario: RTC 性能提升
- **Given** 标准模式有效控制频率 ~5Hz（受推理延迟限制）
- **When** 启用 RTC (`execute_horizon=5`)
- **Then** 有效控制频率提升至 ~15-25Hz
- **And** 机器人运动更平滑

