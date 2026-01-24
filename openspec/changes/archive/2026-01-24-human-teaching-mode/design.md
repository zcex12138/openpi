# Design: Human Teaching Mode

## Context

在 Franka 机器人评估过程中，当策略遇到困难时，操作员无法平滑介入引导。本设计实现按空格键切换到零刚度阻抗控制，允许人类手动引导机器人，同时保持推理和录制功能。

**约束条件**：
- frankx `ImpedanceMotion` 刚度为 const，不支持运行时修改
- `Runtime.run()` 是阻塞循环，无法在外层插入 per-step 逻辑
- 录制线程 `EpisodePklRecorder` 独立运行，调用 `env.get_recording_frame()`
- 必须保留 Ctrl+C 终止能力

## Goals / Non-Goals

**Goals**:
- 空格键触发零刚度示教模式
- 推理持续运行并记录
- 标记录制数据中的示教段
- 无线程安全问题

**Non-Goals**:
- 双向切换（恢复自动控制）
- Cartesian 控制模式支持
- Windows 平台支持
- 录制人类实际动作（仅记录 policy 输出）

## Decisions

### D1: 按键检测集成点

**决策**: 在 `FrankaEnvironment.apply_action()` 开头调用 `_check_teaching_trigger()`

**理由**:
- 与控制在同一线程，无竞态风险
- 无需修改 `openpi-client` 包的 Runtime
- 无需新增线程

**备选方案**:
- ❌ 独立守护线程：引入竞态，需加锁
- ❌ 修改 Runtime 添加 hook：跨包改动大
- ❌ Subscriber 模式：需要修改 Runtime 订阅机制

### D2: 刚度切换机制

**决策**: stop 当前 ImpedanceMotion → 创建新的 `ImpedanceMotion(0.0, 0.0)` → move_async

**理由**:
- frankx `ImpedanceMotion.translational_stiffness` 和 `rotational_stiffness` 是 const 成员
- 无法运行时修改，必须重建

**关键步骤**:
```
1. _stop_impedance_motion()        # 停止当前运动
2. robot.set_load(...)             # 设置末端负载（失败仅 warning）
3. motion = ImpedanceMotion(0, 0)  # 创建零刚度运动
4. motion.target = current_affine  # 设置当前位姿为目标（防跳动）
5. robot.move_async(motion)        # 启动异步运动
6. _teaching_mode = True           # 置位标志
```

### D3: 终端模式

**决策**: 使用 `tty.setcbreak()` 而非 `tty.setraw()`

**理由**:
- `setcbreak` 保留 ISIG，Ctrl+C 仍触发 SIGINT
- `setraw` 会屏蔽信号，导致无法终止

**作用域**: 仅包裹 `runtime.run()` 区间，不影响 episode 开始前的 `input()`

### D4: 超时禁用策略

**决策**: `FrankaEnvironment` 新增 `_teaching_mode_active` 标志，`is_episode_complete()` 中若为 True 则跳过超时检查

**理由**:
- 不需要修改 Runtime 的 `max_episode_steps` 逻辑
- 本地标志，简单可控

### D5: 错误处理

| 操作 | 失败处理 | 理由 |
|------|----------|------|
| `robot.set_load()` | warning 并继续 | 非致命，可能下垂但可用 |
| `robot.move_async()` | raise RuntimeError | 无法进入零刚度是严重错误 |

### D6: 状态复位

**决策**: 每 episode 复位 `_teaching_mode = False`

**实现位置**:
- `FrankaRealEnv.reset()`
- `FrankaEnvironment.reset()`

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 切换瞬间机器人跳动 | 低 | 中 | 切换前将 target 设为当前位姿 |
| 负载参数不准导致下垂 | 中 | 低 | 提供配置项，用户可调整 |
| 长时间 teaching 内存增长 | 中 | 低 | 维持现有 drop 策略，用户控制时长 |
| 非 TTY 环境无法使用 | 低 | 低 | 检测 isatty() 并 warning，功能降级 |
| 录制线程在切换期间异常 | 低 | 低 | get_state() 返回缓存值 |

## Thread Safety Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                    Control Thread                           │
│  (Runtime.run() → apply_action() → execute_action())        │
│                                                             │
│  _check_teaching_trigger()                                  │
│       │                                                     │
│       ├─ check_key_pressed()                                │
│       │                                                     │
│       └─ enable_teaching_mode()                             │
│              │                                              │
│              ├─ _stop_impedance_motion()                    │
│              ├─ robot.set_load()                            │
│              ├─ ImpedanceMotion(0.0, 0.0)                   │
│              ├─ robot.move_async()                          │
│              └─ _teaching_mode = True                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Recorder Thread                          │
│  (EpisodePklRecorder._run_recorder())                       │
│                                                             │
│  env.get_recording_frame()                                  │
│       │                                                     │
│       └─ real_env.get_state()  ← 读取 _last_state 缓存      │
│              │                                              │
│              └─ env.is_teaching_mode  ← 读取 _teaching_mode │
└─────────────────────────────────────────────────────────────┘
```

**安全性分析**:
- `_teaching_mode` 是 bool，Python 中 bool 赋值是原子的
- 录制线程只读取，控制线程只写入
- `get_state()` 已有 `_last_state` 缓存机制，异常时返回缓存值

## Migration Plan

无破坏性变更，向后兼容：
- 新增配置项有默认值
- 不按空格时行为不变
- 录制文件新增字段，旧版工具忽略即可

## Open Questions

无 —— 所有决策点已通过多模型分析和用户确认解决。
