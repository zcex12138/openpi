# 代码风格与约定

- Python 3.11，4 空格缩进，行宽 120
- 命名：模块/函数 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE_CASE`
- 测试文件以 `*_test.py` 命名（如 `src/openpi/models/pi0_test.py`）
- 质量工具：`ruff format` + `ruff check`，配合 `pre-commit`