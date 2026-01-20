## 1. Implementation
- [x] 1.1 创建 `policy_records/<config_name>/episode_###` 目录（3 位补零、从 000 开始），并在目录内写入 `records.npy`。
- [x] 1.2 以处理后的输入图像流生成单个 `video.mp4`，多路图像按字典序拼接成近似正方形网格。
- [x] 1.3 录制时禁用逐帧图片写盘，并在 episode 切换时关闭视频写入。
- [x] 1.4 更新记录读取工具以支持新的 episode 目录结构。
- [x] 1.5 录制参数使用默认值，无需通过命令行设置环境变量；环境变量仅允许覆盖输出路径与视频帧率。
- [x] 1.6 在未设置帧率覆盖时，基于 episode 实际时长计算视频帧率并写入 `video.mp4`。
- [x] 1.7 会话断开时刷新当前 episode 视频，保证单次运行也能生成 `video.mp4`。

## 2. Validation
- [x] 2.1 手动验证录制结果为 `episode_###/records.npy` 与 `episode_###/video.mp4`，且无 PNG 输出。
- [x] 2.2 手动验证视频时长与实际录制时长一致（未设置帧率覆盖时）。
