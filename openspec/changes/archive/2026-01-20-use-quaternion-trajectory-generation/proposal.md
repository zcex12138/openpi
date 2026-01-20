# Change: Use quaternion-based rotation representation in Frankx waypoint trajectory generation

## Why
当前 Frankx 的 waypoint 轨迹生成以欧拉角作为旋转表示，容易出现分支跳变，导致机器人在姿态很接近时仍绕大圈旋转。
需要改为基于四元数的旋转表示，以保证姿态插值连续且避免“摇头”。

## What Changes
- 在 Frankx waypoint 轨迹生成中用四元数（log/exp）替代欧拉角表示，避免分支跳变。
- 输出笛卡尔位姿时由旋转向量还原四元数并生成矩阵。

## Impact
- Affected specs: franka-evaluation
- Affected code: /home/mpi/workspace/yhx/frankx/include/frankx/motion_waypoint_generator.hpp
