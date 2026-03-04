# pi05 残差模块（CR-DAgger 风格）实施计划

## 目标
- 在 openpi 的 pi05 上叠加残差策略：冻结 base policy，仅用纠错数据训练残差网络。
- 推理时输出 `final = base + residual`，支持 scale/cap，默认残差可关闭。

## 依据（已定位代码）
- cr-dagger 残差网络：`PyriteML/diffusion_policy/model/residual/mlp.py`
- cr-dagger 残差训练：`PyriteML/diffusion_policy/workspace/train_online_residual_mlp_workspace.py`
- cr-dagger 残差叠加：`PyriteEnvSuites/env_runners/residual_online_learning_env_runner.py`
- openpi pi05：`src/openpi/models/pi0_config.py`, `src/openpi/models/pi0.py`, `src/openpi/models_pytorch/pi0_pytorch.py`
- openpi 推理/变换链：`src/openpi/policies/policy.py`, `src/openpi/transforms.py`, `src/openpi/training/config.py`

## 设计约束
- 仅改动必要路径，不影响现有功能。
- 残差在数据集动作空间叠加（例如 Franka 8 维），不侵入 pi05 token/模型内部。
- 残差默认可关闭（scale=0 或不加载权重）。

## 方案
### 1) 残差网络
- 新增 PyTorch MLP 残差模块：`src/openpi/models_pytorch/residual_mlp.py`。
- 结构对齐 cr-dagger 的 MLPResidual：`Linear -> GELU -> Dropout -> ... -> Linear -> reshape`。
- 输入维度：`state + base_action`（先做最小闭环）；若纠错数据含 wrench/tactile，再扩展。

### 2) 训练数据与目标
- 纠错数据需包含 `base_action` 与 `corrected_action`。
- 新增 transform：`ResidualDeltaActions`：
  - `delta = corrected_action - base_action`
  - 可选：非纠错区间 `delta = 0` 或低权重采样。
- 归一化统计针对 delta actions。

### 3) 推理叠加
- 新增 `ResidualPolicy`：`src/openpi/policies/residual_policy.py`。
- 流程：
  1. base policy 先输出动作序列；
  2. residual 输入（state + base_action）预测 delta；
  3. 可选 `scale`/`cap`（参考 cr-dagger 的 `scale_and_cap_residual_action`）；
  4. 输出 `base + delta`。

### 4) 配置与入口
- 训练配置新增 `pi05_franka_residual`（或类似）：
  - 指向纠错数据集
  - residual MLP 超参
  - base policy checkpoint 路径
- 训练入口：
  - 新建 `scripts/train_residual_pytorch.py` 或扩展现有 `scripts/train_pytorch.py` 增加 residual 分支。

## 代码改动清单
- `src/openpi/models_pytorch/residual_mlp.py`（新增）
- `src/openpi/policies/residual_policy.py`（新增）
- `src/openpi/transforms.py`（新增 ResidualDeltaActions，可选 MaskResidualOutsideCorrection）
- `src/openpi/training/config.py`（新增 residual 训练配置）
- `scripts/train_residual_pytorch.py`（新增）或扩展 `scripts/train_pytorch.py`

## 验证
- 单测：delta 构造正确、无纠错时 residual 近 0。
- 线下：纠错集 delta MSE。
- 线上：成功率、接触失败率对比。

## 风险
- 动作空间对齐：pi05 内部 32 维 vs Franka 8 维，必须在输出变换后叠加。
- 纠错覆盖不足导致残差发散，需要样本重权或 mask。
- 额外推理延迟：残差网络必须轻量。

## 回滚
- 通过配置不加载 residual 权重或 scale=0 关闭残差。
