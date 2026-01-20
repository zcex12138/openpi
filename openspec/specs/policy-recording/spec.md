# policy-recording Specification

## Purpose
TBD - created by archiving change update-policy-recording-output. Update Purpose after archive.
## Requirements
### Requirement: 以配置名分组的 episode 目录与输出文件
当启用录制且观测中包含 episode 索引时，系统 SHALL 在录制根目录下以配置名分组创建
`policy_records/<config_name>/episode_###` 目录（3 位补零，从 `episode_000` 开始）。
若无法解析配置名，系统 SHALL 使用 `unknown` 作为 `<config_name>`。
每个 episode 目录 SHALL 包含 `records.npy` 与 `video.mp4`。

#### Scenario: Episode 目录创建
- **GIVEN** 已启用录制且输入包含 episode 索引
- **WHEN** 新 episode 开始
- **THEN** 创建 `policy_records/<config_name>/episode_###/records.npy` 并开始写入
  `policy_records/<config_name>/episode_###/video.mp4`。

### Requirement: 视频使用处理后的输入图像
episode 视频 SHALL 由输入模型的处理后图像流生成，而不是原始观测图像。若存在多路图像流，录制器 SHALL
按 key 的字典序排序，并优先填充成接近正方形的网格；空余网格位置 SHALL 以黑色填充。

#### Scenario: 网格合成
- **GIVEN** 处理后图像包含多路图像流
- **WHEN** 录制器写入视频帧
- **THEN** 视频帧为字典序排序后的近似正方形网格合成结果。

### Requirement: 录制期间禁止逐帧图片写盘
录制启用时，系统 SHALL 不写入逐帧图片文件。

#### Scenario: 无逐帧图片输出
- **GIVEN** 录制已启用
- **WHEN** 录制进行中
- **THEN** 不写入逐帧图片文件。

### Requirement: 默认视频帧率基于真实时长计算
当未设置帧率覆盖环境变量时，episode 视频 SHALL 以真实时长计算的帧率保存，
即 `fps = 帧数 / (结束时间 - 开始时间)`。若无法获取有效时长（例如仅 1 帧或时长非正），
系统 SHALL 回退为 30 fps。

#### Scenario: 真实时长帧率
- **GIVEN** 录制已启用且未设置帧率覆盖环境变量
- **WHEN** 写入 `video.mp4`
- **THEN** 视频帧率为基于真实时长计算得到的值，且时长无效时回退为 30 fps。

### Requirement: 录制默认配置无需环境变量
启用录制后，系统 SHALL 使用默认配置完成输出（`policy_records/<config_name>/episode_###` 目录、`records.npy`、
`video.mp4`、基于真实时长计算的帧率），且 SHALL NOT 要求用户通过命令行设置环境变量才能生效。

#### Scenario: 默认配置生效
- **GIVEN** 录制已启用且未设置任何录制相关环境变量
- **WHEN** 录制进行中
- **THEN** 仍生成 `policy_records/<config_name>/episode_###/records.npy` 与
  `policy_records/<config_name>/episode_###/video.mp4`。

### Requirement: 环境变量仅覆盖输出路径与帧率
系统 SHALL 仅允许通过环境变量覆盖录制输出路径与视频帧率，并且这些覆盖是可选的。

#### Scenario: 可选覆盖
- **GIVEN** 录制已启用且设置了覆盖环境变量
- **WHEN** 录制进行中
- **THEN** 输出路径与帧率按环境变量生效，其它录制参数不受环境变量影响。

