# Design: Real-Time Chunking for π₀.5

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RealTimeChunkBroker                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Action Queue                                                    │   │
│  │  [a₀, a₁, a₂, ..., a_{h-1}]  ← 当前 chunk                       │   │
│  │       ↑                                                          │   │
│  │  execute_ptr (正在执行的位置)                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────┐     ┌──────────────────────────────────────────┐  │
│  │ 执行线程         │     │ 推理线程                                  │  │
│  │ while running:  │     │ while running:                           │  │
│  │   action = pop()│     │   prev_chunk = get_executed_prefix()     │  │
│  │   env.step()    │     │   new_chunk = model.realtime_sample(     │  │
│  │   wait(1/Hz)    │     │       obs, prev_chunk, inference_delay)  │  │
│  └─────────────────┘     │   merge_chunks()                         │  │
│                          └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Algorithm

### 1. 时序关系

```
时间轴 ───────────────────────────────────────────────────────────────►

      ├── Chunk 0 生成 ──┤
      │                  │
      [  推理延迟期间    ][   执行 chunk 0 剩余动作  ]
      │                  │
      │                  ├── Chunk 1 生成 ──┤
      │                  │                  │
执行  [      静止       ][ a₀ a₁ a₂ ][ a₃ a₄ ][ b₃ b₄ ]
                         ↑           ↑        ↑
                         │           │        └─ 新 chunk 接续
                         │           └─ inference_delay 结束
                         └─ chunk 0 开始执行
```

### 2. Chunk 融合逻辑

```python
# 每次推理迭代：
# 1. 收集正在执行的动作前缀
prefix = action_queue[:inference_delay]

# 2. 生成新 chunk（条件化于 prefix）
new_chunk = model.realtime_sample_actions(obs, prefix, inference_delay)

# 3. 融合执行
# - 继续执行旧 chunk 的 [0:inference_delay]
# - 切换到新 chunk 的 [inference_delay:execute_horizon]
executed = concat(prefix, new_chunk[inference_delay:execute_horizon])

# 4. 更新 action queue
action_queue = new_chunk[execute_horizon:]
```

### 3. Prefix-Conditioned 采样

在 `Pi0.realtime_sample_actions()` 中：

```python
def realtime_sample_actions(self, rng, observation, prev_chunk, inference_delay, ...):
    # 标准 ODE 求解，但在每步中：
    def step(carry):
        x_t, time = carry

        # 关键：将前 inference_delay 个动作固定为 prev_chunk
        mask = jnp.arange(action_horizon) < inference_delay
        x_t_masked = jnp.where(mask[None, :, None], prev_chunk[:, :inference_delay], x_t)

        # 对应的时间也设为 0（已完成去噪）
        time_chunk = jnp.where(mask, 0.0, time)

        # 计算速度场
        v_t = self.predict_velocity(observation, x_t_masked, time_chunk)

        # 仅更新非 prefix 部分
        v_t_masked = jnp.where(mask[None, :, None], 0.0, v_t)

        return x_t + dt * v_t_masked, time + dt

    return jax.lax.while_loop(cond, step, (noise, 1.0))
```

## Key Components

### 1. RealTimeChunkBroker (packages/openpi-client)

```
openpi_client/
├── action_chunk_broker.py      # 现有阻塞式实现
└── realtime_chunk_broker.py    # 新增 RTC 实现
```

**职责**：
- 管理 action queue 和执行指针
- 协调推理与执行的时序
- 处理 chunk 边界融合

### 2. Pi0.realtime_sample_actions (src/openpi/models)

```
models/
└── pi0.py
    ├── sample_actions()           # 现有标准采样
    └── realtime_sample_actions()  # 新增 prefix-conditioned 采样
```

**职责**：
- 支持 prev_chunk 前缀条件
- 实现 time masking 逻辑
- 保持 KV cache 优化

### 3. Policy.infer_realtime (src/openpi/policies)

```
policies/
└── policy.py
    ├── infer()           # 现有接口
    └── infer_realtime()  # 新增 RTC 接口
```

**职责**：
- 封装 realtime_sample_actions 调用
- 处理输入/输出变换
- 返回完整 chunk 及元数据

## Data Flow

```
┌─────────────┐    obs    ┌──────────────────────┐
│ Environment │ ────────► │ RealTimeChunkBroker  │
└─────────────┘           │                      │
      ▲                   │  ┌────────────────┐  │
      │                   │  │ Action Queue   │  │
   action                 │  │ [a₀,a₁,...,aₙ] │  │
      │                   │  └───────┬────────┘  │
      │                   │          │           │
      │                   │          ▼           │
      │                   │  ┌────────────────┐  │
      └───────────────────┤  │ pop_action()   │  │
                          │  └────────────────┘  │
                          │                      │
                          │  ┌────────────────┐  │   prev_chunk
                          │  │ Policy.infer   │◄─┼─────────────┐
                          │  │ _realtime()    │  │             │
                          │  └───────┬────────┘  │             │
                          │          │           │             │
                          │          ▼           │             │
                          │  ┌────────────────┐  │             │
                          │  │ merge_chunk()  │──┼─────────────┘
                          │  └────────────────┘  │
                          └──────────────────────┘
```

## Threading Model

### Option A: 单线程顺序执行
```python
while running:
    action = broker.get_action(obs)  # 阻塞直到有动作可用
    env.step(action)
```
- 简单可靠
- 适合推理速度快于控制频率的场景

### Option B: 双线程异步执行 ✓ 推荐
```python
# 执行线程
def execute_loop():
    while running:
        action = queue.get()
        env.step(action)
        time.sleep(1/control_hz)

# 推理线程
def infer_loop():
    while running:
        obs = env.observe()
        prefix = get_executed_prefix()
        new_chunk = policy.infer_realtime(obs, prefix)
        queue.merge(new_chunk)
```
- 真正实现推理与执行并行
- 充分利用 RTC 优势

## Error Handling

### 1. Queue Underrun
当推理速度慢于执行消耗时：
- **检测**：`queue.size() < min_buffer`
- **处理**：保持最后一个动作（position hold）或线性外推

### 2. 首次推理延迟
首个 chunk 生成时无历史可参考：
- **处理**：使用零向量作为 prev_chunk，或执行静止动作

### 3. Episode 边界
Episode 重置时：
- **处理**：清空 queue，重置 prev_chunk

## Performance Considerations

### Memory
- 额外存储 prev_chunk: `O(action_horizon × action_dim) ≈ 240 floats`
- 可忽略

### Latency
- Prefix masking 操作：`O(1)` 额外开销
- 整体推理时间基本不变

### Throughput
- 有效控制频率提升：从 `1/inference_time` 到 `execute_horizon/inference_time`
- 典型提升 3-5x
