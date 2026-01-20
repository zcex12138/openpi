## ADDED Requirements
### Requirement: 客户端记录包含执行后动作
当客户端录制启用且动作执行包含安全约束（如裁剪、限速、四元数归一化）时，系统 SHALL 在记录的
`records_client.npy` 中保存执行后的 8D 动作（位置+姿态+夹爪），以便对比模型输出与实际下发动作。
该字段 SHALL 作为 `outputs/executed_action` 存储，且不影响原始 `outputs/actions`。

#### Scenario: 记录执行后动作
- **GIVEN** 客户端录制已启用
- **AND** 机器人动作在执行前经过裁剪/限速/四元数归一化处理
- **WHEN** 录制器写入一条 step 记录
- **THEN** 该记录包含 `outputs/executed_action`
- **AND** `outputs/actions` 仍保持为原始策略输出
