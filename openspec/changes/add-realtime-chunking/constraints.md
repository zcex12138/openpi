# Constraints: Real-Time Chunking for π₀.5

本文档记录所有已确认的约束，消除实现阶段的决策点。

## 环境约束

| 约束项 | 确定值 | 依据 |
|--------|--------|------|
| 控制频率 | **30 Hz** (33ms/周期) | 用户确认 |
| 运行模式 | **Remote Policy Server** (WebSocket) | 用户确认 |
| RTT 分布 | **p90 100-200ms** | 用户确认 |
| 实现范围 | **最小版本** | 硬前缀固定 + client 侧融合，暂不实现 token 级 attention decay |

## WebSocket 协议约束

| 约束项 | 确定值 |
|--------|--------|
| 兼容策略 | **双格式兼容**：旧 client 发裸 obs dict，新 client 发 envelope，server 自动识别 |
| 顶层键命名 | **`__openpi_*` 前缀**（`__openpi_method`, `__openpi_rtc`） |
| 并发模式 | **单 in-flight**：每连接只允许 1 个在途请求，无需 request_id |
| prev_chunk 格式 | **仅传 prefix**：shape `[inference_delay, action_dim]` |
| 响应内容 | **返回两份**：`actions_env` + `actions_model` |
| Reset 机制 | **增加 reset RPC**：server 可做 recorder 分段/清状态 |

### 消息格式

**请求 (RTC envelope)**:
```json
{
  "__openpi_method": "infer_realtime",
  "__openpi_rtc": {
    "prev_prefix_actions": [[...], [...], ...],  // shape [inference_delay, action_dim]
    "inference_delay": 6
  },
  "observation": { ... }
}
```

**请求 (reset)**:
```json
{
  "__openpi_method": "reset"
}
```

**响应 (RTC)**:
```json
{
  "actions_env": [[...], ...],    // shape [action_horizon, action_dim], 环境空间
  "actions_model": [[...], ...],  // shape [action_horizon, action_dim], 模型空间
  "server_timing": {
    "infer_ms": 150,
    "total_ms": 155
  }
}
```

**响应 (错误)**:
```json
{
  "__openpi_error": {
    "code": "METHOD_NOT_SUPPORTED",
    "message": "..."
  }
}
```

## 错误处理与安全约束

| 约束项 | 确定值 |
|--------|--------|
| Underrun fallback | **Hold last pose**：保持最后一个目标位姿 |
| 超时阈值 | **固定 300ms** |
| 断线恢复 | **终止 episode**：机器人进入安全停 |
| 升级阈值 | **连续 30 次 underrun** 后终止 episode |
| 错误处理 | 结构化错误 + 连接继续（不关闭连接） |

### 错误处理流程

```
超时/断线 → 进入 underrun fallback (hold last pose)
         → 累计 underrun_count
         → if underrun_count >= 30: 终止 episode + 安全停
         → else: 继续尝试恢复
```

## 调度与参数约束

| 约束项 | 确定值 |
|--------|--------|
| inference_delay | **固定 6**（覆盖 p90 RTT，单一 JAX 编译） |
| execute_horizon | **8**（≥ inference_delay） |
| action_horizon | **30**（模型默认） |
| 推理触发 | **阈值触发**：queue_len < min_buffer 时触发 |
| min_buffer | = inference_delay = **6** |
| Prefix 来源 | **计划队列动作**（非实际执行动作） |

### 参数校验规则（硬约束）

```python
assert 0 < inference_delay <= action_horizon
assert execute_horizon <= action_horizon
assert execute_horizon >= inference_delay
assert min_buffer == inference_delay
```

## Blending 与配置约束

| 约束项 | 确定值 |
|--------|--------|
| Blending 位置 | **Franka 层特化**：SE(3) blend |
| Blend 算法 | translation: lerp, quaternion: slerp |
| Blend 窗口 | **固定 3 步**线性衰减 |
| EMA 交互 | **RTC 开启时关闭 EMA smoothing** |
| Gripper 处理 | **Hold last**：仅保持最后状态，不参与 blending |
| 配置位置 | `examples/franka/real_env_config.yaml` 新增 `rtc:` 块 |

### Blending 算法

```python
def blend_actions(old_action, new_action, step, blend_horizon=3):
    if step >= blend_horizon:
        return new_action

    alpha = step / blend_horizon  # 线性衰减

    # Translation: lerp
    pos_blended = (1 - alpha) * old_action[:3] + alpha * new_action[:3]

    # Quaternion: slerp
    quat_blended = slerp(old_action[3:7], new_action[3:7], alpha)

    # Gripper: hold last (不参与 blending)
    gripper = old_action[7]

    return concat(pos_blended, quat_blended, [gripper])
```

### YAML 配置格式

```yaml
# examples/franka/real_env_config.yaml
rtc:
  enabled: true
  inference_delay: 6
  execute_horizon: 8
  min_buffer: 6
  blend_horizon: 3
  recv_timeout_ms: 300
  max_underrun_count: 30

smoothing:
  # RTC 开启时自动禁用
  enabled: false  # 或 alpha: 0.0
```

## 监控与日志约束

| 约束项 | 确定值 |
|--------|--------|
| 遥测数据 | 结构化指标 |
| 落地位置 | **写入 pkl**：追加到 pkl_recorder |

### 遥测字段

```python
@dataclass
class RTCTelemetry:
    queue_len: int           # 当前队列长度
    underrun_count: int      # 累计 underrun 次数
    late_switch_count: int   # 迟到切换次数
    inference_delay_used: int  # 实际使用的 delay
    rtt_ms: float            # 本次推理 RTT
    step_dt_ms: float        # 控制周期实际耗时
    is_holding: bool         # 是否处于 hold 状态
```

## PBT 属性（Property-Based Testing）

| 属性名 | 不变式 | 反例生成策略 |
|--------|--------|--------------|
| **Prefix 一致性** | `new_chunk[:delay] == prev_prefix` | 生成随机 prev_prefix，验证新 chunk 前 delay 步完全匹配 |
| **队列单调性** | 每次 `get_action()` 后 queue_len 递减或因新 chunk 跳增 | 模拟高频调用，验证无负数/异常跳变 |
| **Underrun 幂等性** | 连续 underrun 返回相同 hold 动作 | 模拟空队列连续调用，验证输出一致 |
| **Reset 原子性** | reset 后首次 get_action 不返回旧 episode 动作 | reset 期间异步推理完成，验证 generation_id 过滤有效 |
| **超时边界** | 超时后 ≤1 个控制周期内进入 fallback | 模拟网络延迟，验证 fallback 触发时机 |
| **Blend 连续性** | blend 窗口内相邻动作 delta < 阈值 | 生成跳变 chunk 对，验证 blend 后平滑 |
| **参数合法性** | 配置加载时校验通过当且仅当满足硬约束 | 生成边界/越界参数组合 |

### PBT 伪代码

```python
# Prefix 一致性
@given(prev_prefix=arrays(float, (6, 8)))
def test_prefix_consistency(prev_prefix):
    new_chunk = model.realtime_sample_actions(obs, prev_prefix, delay=6)
    assert np.allclose(new_chunk[:6], prev_prefix)

# Reset 原子性
@given(old_chunk=arrays(float, (30, 8)))
def test_reset_atomicity(old_chunk):
    broker.queue = old_chunk
    broker.start_inference()  # 异步
    broker.reset()            # 立即 reset
    time.sleep(0.5)           # 等待推理完成
    action = broker.get_action(obs)
    assert broker.generation_id > old_generation_id
    assert action not in old_chunk
```

## 实现优先级

1. **P0 (阻塞项)**
   - `RealTimeChunkBroker` 核心逻辑
   - WebSocket 协议扩展（双格式兼容）
   - `Policy.infer_realtime()` 接口

2. **P1 (核心功能)**
   - `Pi0.realtime_sample_actions()` 模型方法
   - Franka 层 SE(3) blending
   - 错误处理与 underrun fallback

3. **P2 (可观测性)**
   - RTCTelemetry 结构化指标
   - pkl_recorder 集成
   - YAML 配置解析

4. **P3 (测试)**
   - PBT 属性测试
   - 集成测试
