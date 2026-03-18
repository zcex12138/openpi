# pi05 残差模块（CR-Dagger 风格）实施计划与当前状态

## 目标
- 在 openpi 的 Franka 路径上引入一条 CR-Dagger 风格的残差训练链。
- 残差训练只依赖纠错数据，不重训 base policy。
- 后续推理目标仍然是 `final = base + residual`，但当前阶段先完成离线训练入口，再接推理链路。

## 当前状态
### 已完成
- 已新增独立顶层包 `src/residual_policy/`，不再把 residual 训练主体塞进 `src/openpi/...`。
- 已实现 Zarr 训练数据读取：
  - 使用 `robot_tcp_pose / base_action / corrected_action / corrected_action_valid / is_human_teaching / meta/episode_ends`
  - 内部将 `[xyz, quat, gripper]` 转成 `[xyz, r6d, gripper]`
  - 在编码前对四元数做 sign canonicalization，消除 `q/-q` 伪残差
- 已实现 CR-Dagger 风格采样：
  - 不再使用“自主帧零残差 + 低权重”的方案
  - 改为通过 `correction_start / correction_interval / additional_horizon` 构造训练索引
  - 默认 `weighted_sampling=4`、`correction_horizon=10`、`num_initial_episodes=0`
  - loss 仍是统一 MSE，不对非示教帧单独做 loss reweight
- 已实现独立训练入口 `scripts/train_residual_policy.py`
- 已完成单测、烟测和一次正式训练

### 尚未完成
- openpi 推理链路还没有接入 residual policy
- `examples/franka/main.py` 与 `scripts/serve_policy.py` 目前仍只加载 base policy
- 尚未提供 residual checkpoint 的在线加载、残差叠加、scale/cap 和安全限幅逻辑

## 已落地方案
### 1) 目录与职责边界
- residual 训练相关内容统一放在 `src/residual_policy/`
- 当前实际文件：
  - `action_repr.py`：残差动作表征与编解码
  - `dataset.py`：Zarr 数据读取与 CR-Dagger 风格采样
  - `model.py`：Residual MLP
  - `trainer.py`：训练循环、验证、checkpoint 保存/恢复
  - `config.py`：训练/采样/模型配置 dataclass
- `src/openpi/...` 当前不承载 residual 训练主体，只继续提供共享旋转工具等底层能力

### 2) 训练数据与目标
- 当前训练直接吃 Zarr，不经过 LeRobot 转换
- 模型输入：`state + base_action`
  - 具体为 `robot_tcp_pose(8) + base_action(8)`，内部转成 `20` 维 `state_10d + base_10d`
- 模型输出：`10` 维 residual target
  - 平移：`corrected_xyz - base_xyz`
  - 旋转：`R_delta = R_corrected @ R_base^T`，再转 `r6d`
  - gripper：`corrected_gripper - base_gripper`
- 归一化统计针对训练索引池上的输入和目标直接计算

### 3) 采样策略
- 当前方案已改成更接近 CR-Dagger 代码库的方式：
  - `correction_interval`：`is_human_teaching=1 && corrected_action_valid=1`
  - `correction_start`：每个 teaching segment 的起始帧
  - `additional_horizon`：从 `correction_start` 向后扩展 `correction_horizon` 帧，只保留仍在同一 teaching segment 的有效帧
- 默认行为：
  - `num_initial_episodes=0` 时，不保留普通自主帧池
  - `correction_interval` 加入一次
  - `correction_start` 和 `additional_horizon` 按 `weighted_sampling=4` 重复加入
- 这与旧版提案中的“非纠错区间 delta=0 或低权重采样”已不一致，现已废弃

### 4) 训练入口与产物
- 当前训练脚本：`scripts/train_residual_policy.py`
- checkpoint 目录结构：
  - `best/`
  - `latest/`
- 每个目录保存：
  - `model.safetensors`
  - `optimizer.pt`
  - `metadata.pt`
  - `residual_stats.pt`
- `metadata.pt` 已记录：
  - `action_representation='xyz_r6d_gripper'`
  - `sampling_style='cr_dagger_like'`
  - `weighted_sampling`
  - `correction_horizon`
  - `num_initial_episodes`

## 当前验证结果
### 单测与烟测
- `src/residual_policy/action_repr_test.py`
- `src/residual_policy/dataset_test.py`
- `scripts/train_residual_policy_test.py`
- 当前通过：
  - residual 编解码
  - CR-Dagger 风格采样索引构造
  - 训练脚本 smoke test
  - resume 行为

### 真实数据试训
- 已在 `eval_records/replay_buffer.zarr` 上做短程对比试训
- 当前选中的正式训练参数：
  - `hidden_dims=(128, 128, 128)`
  - `lr=1e-3`
  - `dropout=0.0`
  - `batch_size=64`
  - `weighted_sampling=4`
  - `correction_horizon=10`
  - `num_initial_episodes=0`
- 已完成一次正式训练，当前最优结果：
  - 输出目录：`checkpoints/residual_policy/franka_residual_crdagger_like_20260312_2356/`
  - 最优 checkpoint：`best/`
  - `best_val_loss = 0.2414630949497223`
  - 最优 epoch = 43

## 与旧版提案的关键差异
- 旧版计划中的这些内容目前 **没有** 落地：
  - `src/openpi/models_pytorch/residual_mlp.py`
  - `src/openpi/policies/residual_policy.py`
  - `src/openpi/transforms.py` 内的 `ResidualDeltaActions`
  - `src/openpi/training/config.py` 内的 residual 训练配置
  - 扩展 `scripts/train_pytorch.py`
- 当前实际落地的是：
  - 独立包 `src/residual_policy/`
  - 独立脚本 `scripts/train_residual_policy.py`
  - 直接读取 Zarr
  - CR-Dagger 风格采样
- 旧版“先把 residual 训练并到 openpi 主训练框架”这条线，当前被明确推迟

## 剩余工作
### 1) 推理链路接入
- 为 residual checkpoint 增加在线加载逻辑
- 在 base policy 输出之后执行 residual 解码与叠加
- 将 residual wrapper 接到：
  - `examples/franka/main.py`
  - `scripts/serve_policy.py`

### 2) 推理时的安全控制
- 增加 residual scale / cap
- 增加 residual 开关和 checkpoint 路径参数
- 明确 `best/` 与 `latest/` 的加载优先级

### 3) 线上验证
- 对比 base vs base+residual
- 指标至少包括：
  - 任务成功率
  - 人工介入时长
  - 接触失败/抖动情况

## 风险
- 当前只有 3 条 episode，训练链已经打通，但泛化能力仍然有限
- 当前 residual 训练与 openpi 主训练框架是分离的，后续推理接入仍需补一层桥接
- 训练最优点出现在 epoch 43，后续 epoch 有过拟合迹象，推理接入时应优先加载 `best/`

## 回滚
- 不加载 residual checkpoint，即保持当前 base policy 推理链不变
- 当前 residual 训练实现独立于 `src/openpi` 主链，回滚成本低
