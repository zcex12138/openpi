<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Repository Guidelines

先读 `CLAUDE.md`。

## 当前优先文档

- Franka 实操流程：`examples/franka/使用方法说明.md`
- Franka 运行时参数：`examples/franka/real_env_config.yaml`
- replay_buffer 字段边界：`docs/franka_replay_buffer_field_checklist.md`
- residual 训练模板：`scripts/train_residual_policy.example.yaml`

## 协作约束

- 若文档与代码冲突，以当前代码为准，重点看：
  - `examples/franka/main.py`
  - `examples/franka/convert_pkl_to_zarr.py`
  - `scripts/train_residual_policy.py`
  - `src/residual_policy/`
  - `packages/openpi-client/src/openpi_client/`
- 不要继续引用已经删除或退役的旧入口：
  - `examples/franka/README.md`
  - `examples/franka/collect_data.py`
  - `examples/franka/collect_single_frame.py`
  - `examples/franka/camera_visualizer.py`
  - `examples/franka/visualize_offline_trajectory.py`
- Franka canonical pose/action 现在是 `pose10`；`pose8` 只保留在真实执行边界和 `executed_action` 调试字段
- `examples/franka/main.py` 当前 CLI 仍以 Tyro `--args.*` 为主；`examples/franka/eval_config.example.yaml` 已出现，但除非代码明确接入，否则不要假设它已经被主入口解析
