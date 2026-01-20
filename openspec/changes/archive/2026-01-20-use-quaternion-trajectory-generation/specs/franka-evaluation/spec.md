## ADDED Requirements
### Requirement: Waypoint 轨迹旋转使用四元数表示
在 Frankx 的 waypoint 轨迹生成中，系统 SHALL 采用基于四元数的旋转表示（log/exp 或等价形式），
避免欧拉角分支跳变导致的大幅旋转。

#### Scenario: 旋转连续性
- **GIVEN** 目标姿态与当前姿态非常接近
- **WHEN** 轨迹生成器更新 waypoint 目标
- **THEN** 旋转插值保持连续，不出现大幅绕圈

#### Scenario: 平移与肘部保持兼容
- **GIVEN** 轨迹生成使用 waypoint
- **WHEN** 切换到四元数旋转表示
- **THEN** 平移与肘部的运动行为保持不变
