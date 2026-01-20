# 变更：更新策略录制输出布局

## Why
当前录制会写逐帧图片且多个 episode 混在同一个 records 文件中，难以管理且体积大。需要按 episode
独立存储，并输出单个合成视频以便回放与审查。

## What Changes
- 每个配置写入独立目录 `policy_records/<config_name>/episode_###`（3 位补零，从 `episode_000` 开始）。
- 每个 episode 目录包含 `records.npy` 与单个 `video.mp4`。
- `video.mp4` 使用输入到模型的处理后图像流（非原始图像）。
- 录制期间不再保存逐帧图片。
- 录制参数使用默认值即可生效，无需通过命令行设置环境变量；未设置覆盖时，视频帧率按真实时长计算。
- 环境变量仅允许覆盖输出路径与视频帧率。

## Impact
- 影响规格：policy-recording（新增）
- 影响代码：src/openpi/policies/policy.py，packages/openpi-client/src/openpi_client/runtime/runtime.py，
  scripts/print_record_data.py，examples/franka/visualize_offline_trajectory.py
