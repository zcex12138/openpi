# Human Teaching Mode During Evaluation

## Summary

在评估过程中按空格键切换到人类示教模式：将阻抗控制刚度设为 0.0，允许操作员手动引导机器人，同时保持推理运行和录制功能，并在录制数据中标记示教段。

## Motivation

当前评估流程中，策略完全控制机器人运动。当策略遇到困难或需要人类干预时，无法平滑切换到人类示教。此功能允许：
1. 在策略卡住时人类接管引导
2. 收集混合数据（策略执行 + 人类示教）用于后续分析
3. 通过录制标记区分数据来源

## Requirements

### R1: 空格键触发示教模式
- **场景**: 评估运行中，用户按下空格键
- **行为**: 系统切换到零刚度阻抗控制，机器人可被手动拖动
- **约束**: 仅在 `control_mode == "impedance"` 时有效；非 impedance 模式下按空格仅 warning，不抛异常

### R2: 推理持续运行
- **场景**: 进入示教模式后
- **行为**: 策略推理继续执行，`execute_action()` 继续调用 `ImpedanceMotion.target` 更新
- **约束**: 刚度为 0.0 时，target 更新不产生实际控制力

### R3: 设置末端负载
- **场景**: 进入示教模式时
- **行为**: 调用 `robot.set_load(mass, F_x_Cload, load_inertia)` 设置末端负载
- **默认值**: `mass=0.3kg`, `F_x_Cload=[0.0, 0.0, 0.0]`, `load_inertia=[0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]`
- **约束**: 防止机器人因重力补偿不足而下垂；`set_load()` 失败仅 warning 并继续

### R4: 录制不中断
- **场景**: 切换示教模式前后
- **行为**: `EpisodePklRecorder` 持续录制
- **约束**: 队列满时允许丢帧（维持现有设计），仅 warning 计数

### R5: 录制标记示教段
- **场景**: 进入示教模式后的每一帧
- **行为**: 录制帧包含 `is_human_teaching: bool` 字段
- **约束**: `_teaching_mode=True` 置位后的帧标记为 `true`，置位前为 `false`

### R6: 单向切换
- **场景**: 示教模式激活后
- **行为**: 保持示教模式直到 Ctrl+C 结束 episode
- **约束**: 不支持再次按键恢复自动控制；teaching 模式后禁用 `max_episode_time` 超时

---

## Explicit Constraints (Zero-Decision Implementation)

以下约束经多模型分析和用户确认，实施时**禁止偏离**：

### C1: 按键检测集成点
- **位置**: `FrankaEnvironment.apply_action()` 方法开头
- **理由**: 与控制同线程，无竞态风险；无需新增线程或修改 Runtime
- **实现**: 在 `apply_action()` 开头调用 `_check_teaching_trigger()`

### C2: 切换帧行为
- **决策**: 检测到空格的那一帧，**先切换到 teaching 模式，再执行本帧 action**
- **理由**: 零刚度下 action 无实际控制力，但满足 R2（target 持续更新）

### C3: teaching 后 execute_action 语义
- **决策**: 继续调用 `execute_action()` 更新 `ImpedanceMotion.target`
- **理由**: 满足 R2，保持推理输出可记录

### C4: 超时/步数限制处理
- **决策**: teaching 模式激活后，`is_episode_complete()` 仅响应 Ctrl+C（`_episode_complete` 标志）
- **实现**: `FrankaEnvironment` 新增 `_teaching_mode_active` 标志，`is_episode_complete()` 中若为 True 则跳过超时检查

### C5: 状态复位策略
- **决策**: 每 episode 复位 `_teaching_mode = False`
- **实现**: `FrankaRealEnv.reset()` 和 `FrankaEnvironment.reset()` 中复位标志

### C6: set_load 失败处理
- **决策**: 仅 warning 并继续进入 teaching 模式（可能下垂）
- **实现**: try-except 包裹 `set_load()`，失败时 `logger.warning()`

### C7: move_async 失败处理
- **决策**: 失败时终止 episode（raise RuntimeError）
- **理由**: 无法进入零刚度控制是严重错误，不应继续

### C8: 终端模式
- **决策**: 使用 `tty.setcbreak()`（保留 ISIG，Ctrl+C 仍触发 SIGINT）
- **作用域**: 仅包裹 `runtime.run()` 区间
- **异常处理**: finally 块恢复终端状态

### C9: 录制线程安全
- **策略**: `get_state()` 在切换期间保持"异常安全"
- **实现**: `get_state()` 若遇异常返回 `_last_state`（已有缓存机制）

### C10: `_teaching_mode` 置位时机
- **决策**: 仅当 `ImpedanceMotion(0.0, 0.0)` 启动成功后置位
- **顺序**: `_stop_impedance_motion()` → `set_load()` → `ImpedanceMotion(0.0, 0.0)` → `move_async()` → `_teaching_mode = True`

### C11: 控制重启行为
- **约束**: 一旦 `_teaching_mode=True`，任何自动控制重启必须重建**零刚度** motion，禁止回到配置刚度

### C12: 非 TTY 环境
- **处理**: `sys.stdin.isatty() == False` 时禁用按键检测功能，打印 warning

---

## Design

### 架构变更

```
┌─────────────────────────────────────────────────────────────┐
│                     FrankaEnvironment                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  apply_action():                                     │   │
│  │    1. _check_teaching_trigger()  ← 按键检测          │   │
│  │    2. real_env.execute_action()                      │   │
│  └─────────────────────────────────────────────────────┘   │
│  - is_teaching_mode: bool (属性, 委托 real_env)            │
│  - _teaching_mode_active: bool (本地标志, 禁用超时)         │
│  - enable_teaching_mode() → 委托给 real_env               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FrankaRealEnv                          │
│  - _teaching_mode: bool                                     │
│  - enable_teaching_mode():                                  │
│      1. _stop_impedance_motion()                            │
│      2. try: robot.set_load(...) except: warning           │
│      3. _impedance_motion = ImpedanceMotion(0.0, 0.0)       │
│      4. _impedance_thread = robot.move_async(motion)        │
│      5. _teaching_mode = True                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   EpisodePklRecorder                        │
│  - _build_record():                                         │
│      record["is_human_teaching"] = env.is_teaching_mode     │
└─────────────────────────────────────────────────────────────┘
```

### 文件修改清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `examples/franka/real_env.py` | 修改 | 添加 `enable_teaching_mode()`, `_teaching_mode` 属性, teaching 配置, `reset()` 复位 |
| `examples/franka/real_env_config.yaml` | 修改 | 添加 `teaching.load_mass`, `teaching.load_com`, `teaching.load_inertia` 配置项 |
| `examples/franka/env.py` | 修改 | 添加 `is_teaching_mode` 属性, `enable_teaching_mode()`, `_check_teaching_trigger()`, 超时禁用逻辑 |
| `examples/franka/pkl_recorder.py` | 修改 | `_build_record()` 添加 `is_human_teaching` 字段 |
| `examples/franka/keyboard_utils.py` | 新增 | 非阻塞按键检测工具函数 |
| `examples/franka/main.py` | 修改 | 在 runtime.run() 外包裹 `cbreak_terminal` context |

### 关键实现

#### 1. 非阻塞按键检测 (keyboard_utils.py)

```python
"""Non-blocking keyboard detection utilities (Linux only)."""

from __future__ import annotations

import sys
import select
import termios
import tty
from contextlib import contextmanager
from typing import Generator


@contextmanager
def cbreak_terminal() -> Generator[None, None, None]:
    """Context manager for cbreak terminal mode (preserves SIGINT)."""
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def check_key_pressed() -> str | None:
    """Non-blocking key detection. Returns pressed char or None."""
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None
```

#### 2. RealEnvConfig 扩展 (real_env.py)

```python
@dataclass
class RealEnvConfig:
    # ... existing fields ...

    # Teaching mode
    teaching_load_mass: float = 0.3
    teaching_load_com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    teaching_load_inertia: list[float] = field(
        default_factory=lambda: [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]
    )
```

#### 3. 示教模式切换 (real_env.py)

```python
def enable_teaching_mode(self) -> None:
    """Switch to zero-stiffness teaching mode."""
    if self._control_mode != "impedance":
        logger.warning("Teaching mode only available in impedance control mode")
        return
    if self._teaching_mode:
        return  # Idempotent

    logger.info("Enabling teaching mode...")
    self._stop_impedance_motion()

    # Set end-effector load (failure is non-fatal)
    try:
        self._robot.set_load(
            self._config.teaching_load_mass,
            self._config.teaching_load_com,
            self._config.teaching_load_inertia,
        )
    except Exception as e:
        logger.warning("set_load failed (robot may sag): %s", e)

    # Create zero-stiffness impedance motion
    self._impedance_motion = ImpedanceMotion(0.0, 0.0)
    current_affine = self._get_current_affine(force_robot_state=True)
    self._impedance_motion.target = current_affine

    try:
        self._impedance_thread = self._robot.move_async(self._impedance_motion)
    except Exception as e:
        raise RuntimeError(f"Failed to start teaching motion: {e}") from e

    self._teaching_mode = True
    logger.info("Teaching mode enabled - robot can be guided by hand")

@property
def is_teaching_mode(self) -> bool:
    return self._teaching_mode
```

#### 4. FrankaEnvironment 集成 (env.py)

```python
class FrankaEnvironment(_environment.Environment):
    def __init__(self, ...):
        # ... existing init ...
        self._teaching_mode_active: bool = False
        self._keyboard_enabled: bool = sys.stdin.isatty()
        if not self._keyboard_enabled:
            logger.warning("stdin is not a TTY, keyboard teaching disabled")

    def reset(self) -> None:
        # ... existing reset ...
        self._teaching_mode_active = False

    @property
    def is_teaching_mode(self) -> bool:
        return self._real_env.is_teaching_mode

    def enable_teaching_mode(self) -> None:
        self._real_env.enable_teaching_mode()
        self._teaching_mode_active = True

    def _check_teaching_trigger(self) -> None:
        """Check for spacebar press and trigger teaching mode."""
        if not self._keyboard_enabled or self._teaching_mode_active:
            return
        from examples.franka.keyboard_utils import check_key_pressed
        key = check_key_pressed()
        if key == ' ':
            self.enable_teaching_mode()

    @override
    def apply_action(self, action: dict) -> None:
        # Check teaching trigger BEFORE action execution
        self._check_teaching_trigger()

        # ... existing apply_action code ...

    @override
    def is_episode_complete(self) -> bool:
        if self._episode_complete:
            return True
        # Skip timeout check if teaching mode is active
        if self._teaching_mode_active:
            return False
        elapsed = time.time() - self._episode_start_time
        if elapsed > self._max_episode_time:
            logger.info("Episode timeout after %.1fs", elapsed)
            self._episode_complete = True
            return True
        return False
```

#### 5. Recorder 标记 (pkl_recorder.py)

```python
def _build_record(self, sample: dict[str, Any]) -> dict[str, Any]:
    # ... existing code ...
    record = {
        # ... existing fields ...
        "is_human_teaching": self._env.is_teaching_mode,
    }
    self._frame_index += 1
    return record
```

#### 6. main.py 集成

```python
from examples.franka.keyboard_utils import cbreak_terminal

def _run_episode(...):
    # ... setup code ...

    with cbreak_terminal():
        runtime.run()

    # ... cleanup code ...
```

---

## Property-Based Testing (PBT) Properties

### P1: 单调 Latch 属性 (R1, R6)
- **[INVARIANT]**: ∀t₁, t₂ ∈ Episode: t₂ > t₁ ⟹ (is_teaching(t₁) ⟹ is_teaching(t₂))
- **[FALSIFICATION]**: 生成随机按键序列（空格、其他键、无），验证 `is_teaching` 一旦为 True 永不回退
- **[BOUNDARY]**: episode 开始时按空格；episode 即将结束时按空格

### P2: 刚度-状态同步属性 (R1, R3)
- **[INVARIANT]**: is_teaching ⟺ (K_trans == 0.0 ∧ K_rot == 0.0)
- **[FALSIFICATION]**: 切换后查询 `_impedance_motion` 的刚度参数
- **[BOUNDARY]**: 切换瞬间的那一帧

### P3: 控制模式排他性 (R1)
- **[INVARIANT]**: control_mode ≠ "impedance" ⟹ ∀input, is_teaching ≡ False
- **[FALSIFICATION]**: 在 cartesian 模式下持续按空格
- **[BOUNDARY]**: 不同控制模式切换（若支持）

### P4: 超时抑制属性 (C4)
- **[INVARIANT]**: (is_teaching ∧ elapsed > max_time) ⟹ ¬is_episode_complete
- **[FALSIFICATION]**: 设置短超时，进入 teaching 后等待超过超时时间
- **[BOUNDARY]**: 恰好在超时阈值时刻进入 teaching

### P5: 录制数据完整性 (R4, R5)
- **[INVARIANT]**: ∀frame_i: record[i].is_human_teaching == state(i).is_teaching
- **[FALSIFICATION]**: 检查 PKL 中切换点前后帧的标记正确性
- **[BOUNDARY]**: 切换期间队列满时的帧

### P6: 幂等性 (C10)
- **[INVARIANT]**: enable_teaching_mode(state_teaching) ≡ state_teaching (无错误、无副作用)
- **[FALSIFICATION]**: 连续调用 100 次 `enable_teaching_mode()`
- **[BOUNDARY]**: 快速连续按键

---

## Configuration

### real_env_config.yaml 新增项

```yaml
teaching:
  load_mass: 0.3                    # 末端负载质量 [kg]
  load_com: [0.0, 0.0, 0.0]         # 质心位置 [m]
  load_inertia: [0.001, 0, 0, 0, 0.001, 0, 0, 0, 0.001]  # 惯性矩阵 [kg·m²]
```

---

## Success Criteria

| 判据 | 验证方式 |
|------|----------|
| 按空格后机器人可被手动拖动 | 手动测试：尝试推动末端，感受零阻力 |
| 推理持续更新 | 日志输出 `step=N` 持续增长 |
| 录制文件包含示教标记 | 检查 PKL: `any(f['is_human_teaching'] for f in data['frames'])` |
| 示教前帧标记为 false | 检查 PKL: 切换前 `is_human_teaching == False` |
| Ctrl+C 正常结束并保存 | 检查 episode PKL 文件完整且帧数正确 |
| 机器人无下垂 | 观察：启用示教后机器人保持位置不下沉 |
| 超时被禁用 | teaching 后等待超过 max_episode_time，episode 不自动结束 |
| 幂等性 | 连续按空格不产生错误或重复日志 |

---

## Risks & Mitigations

| 风险 | 缓解措施 |
|------|----------|
| 终端 cbreak 模式影响日志输出 | 使用 `setcbreak` 保留回显和 SIGINT |
| 示教切换时机器人跳动 | 切换前将 target 设为当前位姿 |
| 负载参数不准确导致下垂 | 提供配置项，用户可调整；`set_load` 失败仅 warning |
| 长时间 teaching 导致内存增长 | Recorder 维持现有 drop 策略，用户自行控制时长 |
| 非 TTY 环境无法检测按键 | 检测 `isatty()` 并 warning，功能降级 |

---

## Out of Scope

- 再次按键恢复自动控制（双向切换）
- Cartesian 控制模式下的示教
- Windows 平台支持（当前仅 Linux）
- 录制 teaching 段的"人类动作"（当前仅记录 policy 输出）
- 零丢帧保证（维持现有 drop 策略）
