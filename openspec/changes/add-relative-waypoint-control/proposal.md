# Change: Add relative pose waypoint option for Franka cartesian control

## Why
在笛卡尔控制中出现“绕大圈”旋转的问题，可能由欧拉角分支跳变导致。使用相对姿态（Relative Waypoint）能让每步目标更接近当前姿态，从而缓解大幅旋转。

## What Changes
- 在 Franka 评估的 cartesian 控制中新增“相对姿态 waypoint”模式，可选启用。
- 默认保持现有绝对 waypoint 行为，确保兼容。

## Impact
- Affected specs: franka-evaluation
- Affected code: examples/franka/real_env.py, examples/franka/main.py
