"""打印 policy_records 中每个 step 的观测 state 和第一帧 action."""

import argparse
from pathlib import Path
import numpy as np


def _normalize_records(records, max_steps: int | None) -> list[dict]:
    if isinstance(records, dict):
        records = [records]
    if max_steps is not None:
        records = records[:max_steps]
    return records


def _print_records(records: list[dict], header: str) -> None:
    print(header)
    print("=" * 80)
    for step_num, data in enumerate(records):
        state = data.get("inputs/observation/state")
        actions = data.get("outputs/actions")

        print(f"Step {step_num}:")
        print("-" * 40)

        if state is not None:
            print(f"  观测 State (shape={state.shape}):")
            state_str = np.array2string(state, precision=4, suppress_small=True, separator=", ")
            print(f"    {state_str}")
        else:
            print("  观测 State: 无数据")

        if actions is not None:
            first_action = actions[0]
            print(f"  第一帧 Action (shape={first_action.shape}):")
            action_str = np.array2string(first_action, precision=4, suppress_small=True, separator=", ")
            print(f"    {action_str}")
        else:
            print("  第一帧 Action: 无数据")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="打印 policy records 数据")
    parser.add_argument(
        "--records_dir",
        type=str,
        default="policy_records",
        help="policy records 目录路径",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="最多打印的 step 数量（默认打印全部）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置名子目录（例如 pi05_franka_screwdriver_lora）",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="指定 episode 索引（从 0 开始，仅用于 episode 目录结构）",
    )
    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    if not records_dir.exists():
        print(f"错误：目录 {records_dir} 不存在")
        return

    records_file = records_dir / "records.npy"
    if records_file.exists():
        records = np.load(records_file, allow_pickle=True).tolist()
        records = _normalize_records(records, args.max_steps)
        _print_records(records, f"找到 {len(records)} 个 step（来自 records.npy）\n")
        return

    episode_record_files = sorted(records_dir.glob("records_ep_*.npy"))
    if episode_record_files:
        for record_file in episode_record_files:
            episode_idx = int(record_file.stem.split("_")[-1])
            records = np.load(record_file, allow_pickle=True).tolist()
            records = _normalize_records(records, args.max_steps)
            _print_records(
                records,
                f"Episode {episode_idx}: {len(records)} 个 step（来自 {record_file.name}）\n",
            )
        return

    episode_records = []
    episode_records.extend(sorted(records_dir.glob("episode_*/records.npy")))
    episode_records.extend(sorted(records_dir.glob("*/episode_*/records.npy")))
    if episode_records:
        filtered: list[Path] = []
        for record_file in episode_records:
            rel = record_file.relative_to(records_dir)
            parts = rel.parts
            if parts[0].startswith("episode_"):
                config_name = records_dir.name
                episode_name = parts[0]
            else:
                config_name = parts[0]
                episode_name = parts[1]
            if args.config is not None and config_name != args.config:
                continue
            if args.episode is not None and episode_name != f"episode_{args.episode:03d}":
                continue
            filtered.append(record_file)

        if not filtered:
            print("错误：未找到匹配的 episode 记录文件")
            return

        for record_file in filtered:
            rel = record_file.relative_to(records_dir)
            parts = rel.parts
            if parts[0].startswith("episode_"):
                config_name = records_dir.name
                episode_name = parts[0]
            else:
                config_name = parts[0]
                episode_name = parts[1]
            records = np.load(record_file, allow_pickle=True).tolist()
            records = _normalize_records(records, args.max_steps)
            header = (
                f"Config {config_name} {episode_name}: {len(records)} 个 step（来自 {record_file}）\n"
            )
            _print_records(records, header)
        return

    # 获取所有 step 文件并按数字排序
    episode_step_files = sorted(
        records_dir.glob("episode_*_step_*.npy"),
        key=lambda x: (int(x.stem.split("_")[1]), int(x.stem.split("_")[3])),
    )
    step_files = sorted(
        records_dir.glob("step_*.npy"),
        key=lambda x: int(x.stem.split("_")[1]),
    )

    if not episode_step_files and not step_files:
        print(f"错误：在 {records_dir} 中未找到 step_*.npy 文件")
        return

    if episode_step_files:
        if args.max_steps is not None:
            episode_step_files = episode_step_files[: args.max_steps]

        print(f"找到 {len(episode_step_files)} 个 step 文件\n")
        print("=" * 80)

        for step_file in episode_step_files:
            parts = step_file.stem.split("_")
            episode_idx = int(parts[1])
            step_num = int(parts[3])
            data = np.load(step_file, allow_pickle=True).item()

            state = data.get("inputs/observation/state")
            actions = data.get("outputs/actions")

            print(f"Episode {episode_idx} Step {step_num}:")
            print("-" * 40)

            if state is not None:
                print(f"  观测 State (shape={state.shape}):")
                # 格式化打印 state，每行显示多个值
                state_str = np.array2string(
                    state, precision=4, suppress_small=True, separator=", "
                )
                print(f"    {state_str}")
            else:
                print("  观测 State: 无数据")

            if actions is not None:
                first_action = actions[0]
                print(f"  第一帧 Action (shape={first_action.shape}):")
                action_str = np.array2string(
                    first_action, precision=4, suppress_small=True, separator=", "
                )
                print(f"    {action_str}")
            else:
                print("  第一帧 Action: 无数据")

            print("=" * 80)
        return

    if args.max_steps is not None:
        step_files = step_files[: args.max_steps]

    print(f"找到 {len(step_files)} 个 step 文件\n")
    print("=" * 80)

    for step_file in step_files:
        step_num = int(step_file.stem.split("_")[1])
        data = np.load(step_file, allow_pickle=True).item()

        state = data.get("inputs/observation/state")
        actions = data.get("outputs/actions")

        print(f"Step {step_num}:")
        print("-" * 40)

        if state is not None:
            print(f"  观测 State (shape={state.shape}):")
            # 格式化打印 state，每行显示多个值
            state_str = np.array2string(
                state, precision=4, suppress_small=True, separator=", "
            )
            print(f"    {state_str}")
        else:
            print("  观测 State: 无数据")

        if actions is not None:
            first_action = actions[0]
            print(f"  第一帧 Action (shape={first_action.shape}):")
            action_str = np.array2string(
                first_action, precision=4, suppress_small=True, separator=", "
            )
            print(f"    {action_str}")
        else:
            print("  第一帧 Action: 无数据")

        print("=" * 80)


if __name__ == "__main__":
    main()
