import logging
import threading
import time

from openpi_client import cr_dagger_chunk_broker as _cr_dagger_chunk_broker
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber


class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        self._episode_index = -1

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()
        episode_started = True

        self._in_episode = True
        self._episode_steps = 0
        self._episode_index += 1
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        try:
            while self._in_episode:
                if self._step():
                    self._episode_steps += 1

                # Sleep to maintain the desired frame rate
                now = time.time()
                dt = now - last_step_time
                if dt < step_time:
                    time.sleep(step_time - dt)
                    now = time.time()
                last_step_time = now
        except KeyboardInterrupt:
            logging.info("Episode interrupted by user.")
            self._in_episode = False
            raise
        finally:
            if episode_started:
                logging.info("Episode finished.")
                for subscriber in self._subscribers:
                    subscriber.on_episode_end()

    def _step(self) -> bool:
        """A single step of the runtime loop."""
        observation = self._environment.get_observation()
        control_timestamp = time.time()
        observation_for_subscribers = self._augment_observation(observation, control_timestamp=control_timestamp)
        observation_for_agent = self._strip_subscriber_only_metadata(observation_for_subscribers)
        try:
            action = self._agent.get_action(observation_for_agent)
        except _cr_dagger_chunk_broker.CrDaggerLagExceeded as exc:
            logging.warning("CR-Dagger lag safety stop: %s", exc)
            if hasattr(self._environment, "mark_episode_complete"):
                self._environment.mark_episode_complete()
            self.mark_episode_complete()
            return False

        action = dict(action)
        action_meta = {}
        existing_action_meta = action.get("__openpi")
        if isinstance(existing_action_meta, dict):
            action_meta.update(existing_action_meta)
        action_meta.update(observation_for_agent["__openpi"])
        action["__openpi"] = action_meta
        self._environment.apply_action(action)
        for subscriber in self._subscribers:
            subscriber.on_step(observation_for_subscribers, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):
            self.mark_episode_complete()
        return True

    def _augment_observation(self, observation: dict, *, control_timestamp: float) -> dict:
        """Attach episode metadata for downstream consumers (e.g., recorders)."""
        meta = {}
        existing_meta = observation.get("__openpi")
        if isinstance(existing_meta, dict):
            meta.update(existing_meta)
        meta["episode_index"] = self._episode_index
        meta["episode_step"] = self._episode_steps
        meta["control_timestamp"] = float(control_timestamp)
        augmented = dict(observation)
        augmented["__openpi"] = meta
        return augmented

    def _strip_subscriber_only_metadata(self, observation: dict) -> dict:
        """Remove large recorder-only payloads before policy inference."""
        meta = observation.get("__openpi")
        if not isinstance(meta, dict) or "recording_snapshot" not in meta:
            return observation

        filtered_meta = dict(meta)
        filtered_meta.pop("recording_snapshot", None)

        filtered = dict(observation)
        filtered["__openpi"] = filtered_meta
        return filtered
