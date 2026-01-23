## ADDED Requirements
### Requirement: Episode PKL Recording
在 Franka 评估流程中，当启用录制时，系统 SHALL 按 episode 生成独立的 pkl 文件，
默认输出到 `eval_records/<config_name>/episode_###.pkl`（3 位补零，从 `episode_000` 开始）。
若无法解析配置名，系统 SHALL 使用 `unknown` 作为 `<config_name>`。

#### Scenario: Episode 文件创建
- **GIVEN** 录制已启用且 episode 开始
- **WHEN** 评估 loop 进入新 episode
- **THEN** 创建 `eval_records/<config_name>/episode_###.pkl` 并开始写入

---

### Requirement: Recorded Frame Content
每个录制 step SHALL 写入以下内容：
- L500/D400/Xense1/Xense2 的 RGB 图像（uint8）
- Xense1/Xense2 marker3d（两组 NxMx3 数组）
- TCP pose（7D）
- TCP 6D 速度（线速度 3D + 角速度 3D）
- 6D 力（wrench）与 gripper 状态
- timestamp、frame_index、episode_index

#### Scenario: Step 记录内容完整
- **GIVEN** 相机与机器人状态可用
- **WHEN** 录制器写入当前 step
- **THEN** step 记录包含上述全部字段

---

### Requirement: Image Downsampling
录制图像 SHALL 以相机原始分辨率进行 2×降采样后写入 pkl，
以减少存储空间。

#### Scenario: L500/D400 图像尺寸
- **GIVEN** L500 原始分辨率 960×540、D400 原始分辨率 640×480
- **WHEN** 录制器写入图像
- **THEN** L500 写入 480×270，D400 写入 320×240

---

### Requirement: Async Recording
录制模块 SHALL 与评估控制流程解耦，录制写盘不阻塞评估控制步。
录制 SHOULD 通过独立线程/进程处理序列化与写盘，并通过队列传递数据。

#### Scenario: 写盘不阻塞评估
- **GIVEN** 录制写盘耗时超过评估步长
- **WHEN** 评估 loop 持续运行
- **THEN** 评估仍按原频率输出，录制在后台处理或按策略丢帧

---

### Requirement: Recording Timing
录制频率 SHALL 可配置并独立于评估控制频率（默认 30Hz）。
timestamp SHALL 以 `frame_index / record_fps` 生成，保证单调递增。

#### Scenario: 15Hz 时间戳
- **GIVEN** 录制 fps 为 15
- **WHEN** 连续写入 step
- **THEN** timestamp 以 1/15s 递增

---

### Requirement: LeRobot Conversion
系统 SHALL 提供脚本将 pkl 录制转换为 LeRobot v2 数据集，并保持现有 key 格式：
`observation.images.l500`、`observation.images.d400`、`observation.state`、`action`、
`timestamp`、`frame_index`、`episode_index`、`index`、`task_index`。
转换时 `action` SHALL 写入 8D 全零。

#### Scenario: 转换输出兼容
- **GIVEN** 一组 episode pkl
- **WHEN** 运行转换脚本
- **THEN** 生成的 LeRobot 数据集 schema 与参考数据集一致
