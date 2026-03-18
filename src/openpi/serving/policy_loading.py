from __future__ import annotations

import logging
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.policies import policy as _policy


def load_checkpoint_policy(
    checkpoint_dir: str,
    config_name: str,
    *,
    default_prompt: str | None = None,
) -> tuple[object, object]:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(cfg, checkpoint_dir, default_prompt=default_prompt)
    return policy, cfg


def make_dummy_observation(config_name: str) -> dict:
    from openpi.policies import aloha_policy
    from openpi.policies import droid_policy
    from openpi.policies import franka_policy
    from openpi.policies import libero_policy

    if "aloha" in config_name:
        return aloha_policy.make_aloha_example()
    if "droid" in config_name:
        return droid_policy.make_droid_example()
    if "libero" in config_name:
        return libero_policy.make_libero_example()
    if "franka" in config_name:
        return franka_policy.make_franka_example()
    return franka_policy.make_franka_example()


def warmup_policy(policy: object, config_name: str) -> None:
    logging.info("Warming up policy (JIT compilation)...")
    dummy_obs = make_dummy_observation(config_name)
    start = time.monotonic()
    policy.infer(dummy_obs)
    elapsed = time.monotonic() - start
    logging.info("Warmup complete in %.1f seconds.", elapsed)
