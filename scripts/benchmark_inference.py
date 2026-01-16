"""Benchmark inference frequency for a trained policy."""

import argparse
import logging
import time

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.policies.franka_policy import make_franka_example
from openpi.training import config as _config


def benchmark_inference(
    config_name: str,
    checkpoint_dir: str,
    num_warmup: int = 10,
    num_iterations: int = 100,
    default_prompt: str | None = None,
) -> dict:
    """Benchmark inference frequency for a policy.

    Args:
        config_name: Training config name (e.g., "pi05_franka_screwdriver_lora").
        checkpoint_dir: Path to checkpoint directory.
        num_warmup: Number of warmup iterations (not counted in stats).
        num_iterations: Number of iterations to benchmark.
        default_prompt: Default prompt to use.

    Returns:
        Dictionary with benchmark results.
    """
    logging.info(f"Loading config: {config_name}")
    train_config = _config.get_config(config_name)

    logging.info(f"Loading checkpoint from: {checkpoint_dir}")
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
    )

    # Create example input
    example = make_franka_example()
    logging.info(f"Example input keys: {list(example.keys())}")

    # Warmup runs (JIT compilation happens here)
    logging.info(f"Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        _ = policy.infer(example.copy())
        if (i + 1) % 5 == 0:
            logging.info(f"  Warmup {i + 1}/{num_warmup}")

    # Benchmark runs
    logging.info(f"Running {num_iterations} benchmark iterations...")
    latencies_ms = []
    model_times_ms = []

    for i in range(num_iterations):
        start = time.perf_counter()
        result = policy.infer(example.copy())
        end = time.perf_counter()

        total_latency_ms = (end - start) * 1000
        latencies_ms.append(total_latency_ms)

        # Extract model-only time from policy output
        if "policy_timing" in result:
            model_times_ms.append(result["policy_timing"]["infer_ms"])

        if (i + 1) % 20 == 0:
            logging.info(f"  Iteration {i + 1}/{num_iterations}")

    latencies = np.array(latencies_ms)
    model_times = np.array(model_times_ms) if model_times_ms else None

    results = {
        "num_iterations": num_iterations,
        "total_latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
            "median": float(np.median(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "frequency_hz": {
            "mean": float(1000 / np.mean(latencies)),
            "max": float(1000 / np.min(latencies)),
            "min": float(1000 / np.max(latencies)),
        },
    }

    if model_times is not None:
        results["model_only_ms"] = {
            "mean": float(np.mean(model_times)),
            "std": float(np.std(model_times)),
            "min": float(np.min(model_times)),
            "max": float(np.max(model_times)),
            "median": float(np.median(model_times)),
        }
        results["model_frequency_hz"] = {
            "mean": float(1000 / np.mean(model_times)),
            "max": float(1000 / np.min(model_times)),
            "min": float(1000 / np.max(model_times)),
        }

    return results


def print_results(results: dict) -> None:
    """Print benchmark results in a formatted way."""
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nIterations: {results['num_iterations']}")

    print("\n--- Total Latency (including transforms) ---")
    lat = results["total_latency_ms"]
    print(f"  Mean:   {lat['mean']:.2f} ms")
    print(f"  Std:    {lat['std']:.2f} ms")
    print(f"  Min:    {lat['min']:.2f} ms")
    print(f"  Max:    {lat['max']:.2f} ms")
    print(f"  Median: {lat['median']:.2f} ms")
    print(f"  P95:    {lat['p95']:.2f} ms")
    print(f"  P99:    {lat['p99']:.2f} ms")

    print("\n--- Total Inference Frequency ---")
    freq = results["frequency_hz"]
    print(f"  Mean: {freq['mean']:.2f} Hz")
    print(f"  Max:  {freq['max']:.2f} Hz")
    print(f"  Min:  {freq['min']:.2f} Hz")

    if "model_only_ms" in results:
        print("\n--- Model-Only Latency (excluding transforms) ---")
        model = results["model_only_ms"]
        print(f"  Mean:   {model['mean']:.2f} ms")
        print(f"  Std:    {model['std']:.2f} ms")
        print(f"  Min:    {model['min']:.2f} ms")
        print(f"  Max:    {model['max']:.2f} ms")
        print(f"  Median: {model['median']:.2f} ms")

        print("\n--- Model-Only Frequency ---")
        mfreq = results["model_frequency_hz"]
        print(f"  Mean: {mfreq['mean']:.2f} Hz")
        print(f"  Max:  {mfreq['max']:.2f} Hz")
        print(f"  Min:  {mfreq['min']:.2f} Hz")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference frequency")
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_franka_screwdriver_lora",
        help="Training config name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Default prompt to use",
    )

    args = parser.parse_args()

    results = benchmark_inference(
        config_name=args.config,
        checkpoint_dir=args.checkpoint,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        default_prompt=args.prompt,
    )

    print_results(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
