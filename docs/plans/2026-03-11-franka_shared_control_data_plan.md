# Franka shared-control 数据定义与采集改造方案

## 目标
- 将当前 Franka 的 `human_teaching` 直接替换为“低刚度 base policy + 人类纠正”的 shared-control 语义。
- 在保持当前简化 RTC（`use_action_prefix=False`）的前提下，定义用于后续 residual 训练的最小数据字段。
- 清理当前“manual override 替代 policy”与 `shadow_policy` 相关逻辑，改为 base policy 始终在线执行。

## 设计约束
- 仅改动必要路径，不影响现有非 Franka 路径。
- 当前只支持 `impedance` 模式下的 shared-control。
- 直接替换旧的纯人工接管模式，不保留并行旧模式入口。
- `corrected_action_shift` 固定为 `10`，不单独暴露为当前阶段的用户配置。

## 方案
### 1) 控制语义：切换为 shared-control
- `Space` 进入 shared-control。
- 再按一次 `Space` 退出 shared-control。
- 进入 teaching 后：
  - base policy 继续在线推理并输出动作；
  - 机器人切到 teaching stiffness；
  - 人类在低刚度下对 base trajectory 做物理纠正；
  - 不再使用 `manual_override_action` 替代 policy。
- 退出 teaching 后：
  - 恢复 normal stiffness；
  - 不再触发 `policy reset`。

### 2) `frames` 数据定义
新增且仅新增以下字段：
- `base_action`
  - 当前控制步的 base policy raw output。
- `corrected_action`
  - 自主段：等于 `base_action`。
  - teaching 段：由同一 teaching segment 内未来 `+10` 帧 state 投影回填。
- `corrected_action_valid`
  - 自主段恒为 `True`。
  - teaching 段仅在该帧之后同段内仍至少有 10 帧时为 `True`。

保留现有字段语义：
- `action`
  - 继续表示该控制步的 `executed_action`。
- `is_human_teaching`
  - 只表示当前步处于低刚度人类纠正窗口。
- `action_source`
  - 保持现有字段，但不再承担 residual 训练语义。

不再新增或保留以下冗余训练字段：
- `base_action_valid`
- `base_action_source`
- `corrected_action_shift`
- `residual_valid`
- `action_semantics`

### 3) `corrected_action@+10` 的定义
- 固定采用与当前 Franka position-control 训练一致的定义：
  - `corrected_action_t = project(state_{t+10})`
- `project(state) = state[:7] + gripper`，即 Franka 外部 8D absolute action space。
- future state 必须来自**同一 teaching segment**。
- 如果该 teaching segment 剩余帧数不足 10：
  - `corrected_action_valid = False`
  - `corrected_action` 保留占位值，不参与训练。

### 4) 顶层记录结构
- `policy_steps` 与 `policy_horizons` 保留，继续用于 RTC 调试与可视化分析。
- `human_teaching_steps` 如继续输出，仅保留轻量索引信息：
  - `frame_index`
  - `control_timestamp`
  - `teaching_segment_id`
  - `teaching_step`
- residual 训练语义只依赖 `frames.base_action / frames.corrected_action / frames.corrected_action_valid`。

## 代码改动清单
- `packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py`
  - 删除 manual override 分支。
  - 删除 `shadow_policy` 参数与相关逻辑。
  - 恢复为纯 policy agent。
- `examples/franka/env.py`
  - `_poll_teaching_controls()` 改为 `Space` toggle。
  - `get_observation()` 不再注入 `manual_override_action`、`manual_override_meta`、`reset_policy`。
  - teaching 期间仅打标签，不改变动作来源。
- `examples/franka/real_env.py`
  - teaching mode 继续负责切换刚度，但不再承载“纯人工接管”语义。
  - `execute_action()` 在 teaching 期间继续吃 base policy 动作。
- `examples/franka/pkl_recorder.py`
  - 在 `frames` 中写入 `base_action / corrected_action / corrected_action_valid`。
  - 在 `on_episode_end()` 中按 teaching segment 做 `+10` future-state 回填。
  - `human_teaching_steps` 精简为索引型记录。

## 验证
- 单测：
  - teaching 期间 `PolicyAgent` 仍走正常 policy 推理。
  - `Space` 进入/退出 shared-control 行为正确。
  - 自主段 `base_action == corrected_action` 且 `corrected_action_valid=True`。
  - teaching 段前 `len(segment)-10` 帧 `corrected_action_valid=True`。
  - teaching 段最后 10 帧 `corrected_action_valid=False`。
  - `corrected_action` 精确匹配同段未来第 10 帧 state 投影。
- 实机验收：
  - teaching 期间控制频率不再因 `shadow_policy` 降到约 10Hz。
  - 进入 teaching 后 base policy 不停。
  - 退出 teaching 后不再出现当前由 `policy reset` 引入的冷启动空档。
