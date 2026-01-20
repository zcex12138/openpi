# 常用命令

## 依赖/安装
- `uv sync`
- `uv pip install -e .`

## 训练/推理
- `uv run scripts/compute_norm_stats.py --config-name <config>`
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config> --exp-name=<name>`
- `uv run scripts/train_pytorch.py <config> --exp_name <name>`
- `uv run scripts/serve_policy.py --env=[DROID|ALOHA|LIBERO|ALOHA_SIM]`

## 测试/质量
- `uv run pytest`
- `ruff format .`
- `ruff check .`
- `pre-commit run --all-files`

## 常用系统/开发
- `ls`, `cd`, `pwd`
- `rg <pattern> <path>`
- `git status`, `git diff`, `git log --oneline`