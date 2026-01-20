## 1. Implementation
- [x] 1.1 在 FrankaRealEnv.execute_action 中返回执行后的 8D 动作（裁剪/限速/归一化后）
- [x] 1.2 在 FrankaEnvironment.apply_action 中将执行后动作写入 action dict（如 `executed_action`）
- [x] 1.3 在客户端记录器中持久化新增字段（无需额外转换）
- [x] 1.4（可选）补充说明/日志用于定位执行动作
