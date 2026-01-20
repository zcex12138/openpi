## ADDED Requirements
### Requirement: 支持相对姿态 waypoint 控制模式
在笛卡尔控制模式下，系统 SHALL 支持使用相对姿态 waypoint（基于当前末端姿态的相对变换）来更新目标，
以降低因欧拉角分支跳变导致的大幅旋转风险。该模式默认关闭，用户显式启用后才生效。

#### Scenario: 默认保持绝对 waypoint
- **GIVEN** cartesian 控制开启且未启用相对姿态模式
- **WHEN** 控制循环发送目标姿态
- **THEN** 系统使用绝对姿态 waypoint 更新目标

#### Scenario: 启用相对姿态 waypoint
- **GIVEN** cartesian 控制开启且启用了相对姿态模式
- **WHEN** 控制循环发送目标姿态
- **THEN** 系统使用相对姿态 waypoint 更新目标
