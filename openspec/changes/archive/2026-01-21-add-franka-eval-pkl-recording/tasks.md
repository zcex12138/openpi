## 1. Implementation
- [x] 1.1 评估侧录制开关与挂载
  - [x] 1.1.1 在 `examples/franka/main.py` 增加录制开关、输出目录、录制 fps、队列长度等 CLI 参数
  - [x] 1.1.2 初始化 recorder subscriber 并挂载到 `Runtime`（开关开启时）
  - [x] 1.1.3 在 `FrankaEnvironment` 中补充“每帧数据获取”接口（相机帧 + 状态 + 时间戳）
- [x] 1.2 相机服务/客户端扩展（L500/D400/Xense1/Xense2 + marker3d）
  - [x] 1.2.1 `camera_service.py` 扩展 provider：支持两路 Xense 同时采集，并输出 marker3d
  - [x] 1.2.2 `camera_service.py` 编码/响应协议扩展（新增 xense1/xense2 与 marker3d 字段）
  - [x] 1.2.3 `camera_client.py` 解码并暴露新字段（含 marker3d）
  - [x] 1.2.4 `camera_config.yaml` 增加 xense1/xense2 配置与输出 key 定义
- [x] 1.3 Episode 级 pkl 录制器
  - [x] 1.3.1 实现 recorder subscriber（异步队列/线程或进程写盘）
  - [x] 1.3.2 图像 2×降采样与帧打包结构（按 design.md 结构）
  - [x] 1.3.3 TCP 6D 速度读取（优先使用控制回路缓存的 `robot_state.O_dP_EE_c`）
  - [x] 1.3.4 生成 `eval_records/<config_name>/episode_###.pkl`
- [x] 1.4 独立录制脚本（新进程运行）
  - [x] 1.4.1 复用相机/状态来源，支持单独启动录制
  - [x] 1.4.2 CLI 参数与默认输出目录对齐
- [x] 1.5 pkl→LeRobot 转换脚本
  - [x] 1.5.1 解析 pkl 并创建 LeRobot v2 数据集（保持现有 key）
  - [x] 1.5.2 写入 `action` 为 8D 全零，填充 timestamp/frame_index/episode_index/index/task_index
- [x] 1.6 配置与文档更新
  - [x] 1.6.1 更新 `camera_config.yaml`（新增 xense1/xense2 与输出 key）
  - [x] 1.6.2 更新 `examples/franka/README.md`（新增录制与转换说明）
  - [x] 1.6.3 视需要更新 `CLAUDE.md` 的 Franka 评估说明

## 2. Validation
- [x] 2.1 运行一条短评估录制，确认 pkl 输出结构、字段齐全与图像尺寸
- [x] 2.2 运行转换脚本生成 LeRobot 数据集并核对 meta/schema
