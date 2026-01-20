# Change: Add executed-action recording for Franka client records

## Why
在 Franka 评估中观察到“末端姿态与发送动作不一致”的现象，需要区分模型输出与实际下发动作，便于定位限速/裁剪/控制异常。

## What Changes
- 在 Franka 客户端记录中新增执行后动作字段（例如 `outputs/executed_action`），记录经过安全约束后的最终下发动作。
- 保留 `outputs/actions` 为原始策略输出（未经过裁剪/限速/四元数规范化）。

## Impact
- Affected specs: policy-recording
- Affected code: examples/franka/real_env.py, examples/franka/env.py, examples/franka/main.py
