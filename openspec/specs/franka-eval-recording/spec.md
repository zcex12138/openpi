# franka-eval-recording Specification

## Purpose
TBD - created by archiving change add-franka-eval-pkl-recording. Update Purpose after archive.
## Requirements
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
Franka evaluation recording SHALL store canonical Franka policy actions in pose10 format while preserving the executable robot command separately.

#### Scenario: Record canonical Franka actions
- **GIVEN** Franka evaluation recording is enabled
- **WHEN** a control step is recorded
- **THEN** canonical policy-derived action fields are stored as 10D `[xyz, r1-r6, gripper]`
- **AND** the executed robot command remains available as the 8D quaternion action sent to the robot

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
The system SHALL convert Franka PKL recordings to Zarr/LeRobot-compatible datasets using pose10 canonical action fields for Franka-specific action records.

#### Scenario: Convert Franka PKL to Zarr with pose10 canonical actions
- **GIVEN** a Franka PKL recording produced after this change
- **WHEN** the Zarr conversion script runs
- **THEN** Franka pose/action arrays used for residual-policy training are written in pose10 format
- **AND** residual-policy consumers no longer rely on quaternion-only canonical action arrays

### Requirement: CR-Dagger Policy Provenance Recording
When Franka evaluation recording is enabled in CR-Dagger baseline mode, the episode payload SHALL preserve pose10 canonical policy provenance for each control step and cached horizon.

#### Scenario: Record pose10 base horizon provenance
- **GIVEN** recording is enabled and a new CR-Dagger base horizon is inferred
- **WHEN** the recorder persists the episode payload
- **THEN** the cached Franka base horizon is stored in pose10 format
- **AND** per-step canonical policy action provenance is also stored in pose10 format

### Requirement: Shared Provenance Time Base
CR-Dagger baseline recording SHALL use a single canonical control clock for `frames`, `policy_steps`, and `policy_horizons`.

#### Scenario: Frame and policy records share the same clock
- **GIVEN** recording is enabled in CR-Dagger baseline mode
- **WHEN** the recorder writes the episode PKL
- **THEN** every frame record contains `control_timestamp`
- **AND** every policy-step record contains `control_timestamp`
- **AND** every policy-horizon record declares the same `time_base`
- **AND** every policy-horizon record uses the triggering observation step's `control_timestamp` as `horizon_start_timestamp`
- **AND** the horizon `planned_timestamps` are expressed in that same time base

### Requirement: Backward-Compatible Frame Recording
CR-Dagger baseline provenance recording SHALL extend the episode PKL format without removing the existing frame-sampled hardware records.

#### Scenario: Existing frame payload retained
- **GIVEN** recording is enabled in CR-Dagger baseline mode
- **WHEN** the recorder writes the episode PKL
- **THEN** the existing `frames` list remains present
- **AND** the new policy provenance data is stored alongside it rather than replacing it

#### Scenario: Non-baseline mode recording
- **GIVEN** recording is enabled but CR-Dagger baseline mode is not active
- **WHEN** an episode PKL is written
- **THEN** the existing frame recording behavior remains unchanged
- **AND** any new provenance fields may be omitted or left empty

