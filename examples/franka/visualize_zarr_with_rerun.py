#!/usr/bin/env python3
"""Interactive zarr visualizer using rerun."""

import argparse
import gc
import json
import os
import os.path as osp
import re
import signal
import subprocess
import uuid
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import zarr
from loguru import logger
from tqdm import tqdm

# Import rerun-sdk
import rerun as rr
from rerun import Image, Points3D, LineStrips3D, Transform3D, Quaternion, Scalars, Arrows3D
import rerun.blueprint as rrb
import rerun_bindings


class ZarrDatasetViewer:
    """Interactive viewer for zarr datasets with lazy loading."""

    _XENSE_CANON_RE = re.compile(r"^xense[_-]?(\d+)(?:_camera)?$")

    def __init__(
        self,
        zarr_path: str,
        fps: float = 30.0,
    ):
        """
        Initialize the viewer with lazy loading

        Args:
            zarr_path: Path to zarr dataset
            fps: Playback frame rate
        """
        self.zarr_path = zarr_path
        self.fps = fps
        self.current_episode = 0
        self.viewer_port = 9876
        self.annotate_mode = True
        self.annotations_path: Optional[str] = None
        self.keep_only_current = True
        self.use_episode_prefix = False
        self.manage_viewer_process = True
        self._viewer_process: Optional[subprocess.Popen] = None
        self._grpc_url = f"rerun+http://127.0.0.1:{self.viewer_port}/proxy"
        self._recording_id: Optional[uuid.UUID] = None
        self._annotations: Dict[str, Dict] = {}
        self._trajectory_ids: Optional[List[str]] = None
        self._axes_logged = set()

        # Lazy load zarr (does not load data into memory)
        logger.info(f"Opening zarr dataset: {zarr_path}")
        self._validate_zarr_path(zarr_path)
        self.zarr_root = zarr.open(zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        self.meta_group = self.zarr_root.get("meta", {})

        # Load only metadata (small)
        self.episode_ends = None
        if "episode_ends" in self.meta_group:
            self.episode_ends = np.array(self.meta_group["episode_ends"][:])
            self.num_episodes = len(self.episode_ends)
        else:
            # Single episode mode
            self.num_episodes = 1
            first_key = list(self.data_group.keys())[0]
            self.episode_ends = np.array([len(self.data_group[first_key])])

        self._trajectory_ids = self._load_trajectory_ids()
        if self._trajectory_ids is None:
            logger.warning("trajectory_id metadata not found; using episode id as trajectory_id")
        self.annotations_path = self._default_annotations_path(self.zarr_path)
        self._annotations = self._load_annotations(self.annotations_path)

        # Discover data structure (without loading)
        self.data_keys = list(self.data_group.keys())
        self._log_dataset_structure()

        # Identify data keys by type
        self.image_keys = [
            k for k in self.data_keys if "img" in k.lower() or "image" in k.lower() or "camera" in k.lower()
        ]
        # Support both old 'left_robot_*' and new 'robot_*' naming (exclude 'right_robot_*')
        self.pose_keys = [k for k in self.data_keys if "tcp_pose" in k.lower() and not k.startswith("right")]
        self.marker3d_keys = [k for k in self.data_keys if "marker3d" in k.lower()]
        self.wrench_keys = [k for k in self.data_keys if "wrench" in k.lower() and not k.startswith("right")]
        self.vel_keys = [
            k for k in self.data_keys if "vel" in k.lower() and "tcp" in k.lower() and not k.startswith("right")
        ]
        self.gripper_width_keys = [
            k for k in self.data_keys if "gripper_width" in k.lower() and not k.startswith("right")
        ]
        self.has_human_teaching = "is_human_teaching" in self.data_keys

        # Current episode data (for memory management)
        self._current_data: Optional[Dict[str, np.ndarray]] = None

        # Episode-specific namespace to avoid cross-episode time-series mixing
        self._visualization_seq = 0
        self._episode_prefix = self._make_episode_prefix(self.current_episode)
        self._timeline_name = self._make_timeline_name()
        self._current_n_frames = 0

        # Initialize rerun
        self.rec = None
        self._init_rerun()

        logger.info(f"Dataset ready: {self.num_episodes} episodes")
        self._print_episode_summary()

    def _canonicalize_camera_name(self, name: str) -> str:
        if name.startswith("xense_camera_"):
            return name
        match = self._XENSE_CANON_RE.match(name)
        if match:
            return f"xense_camera_{int(match.group(1))}"
        return name

    def _validate_zarr_path(self, zarr_path: str):
        """Validate that the path is a valid zarr store"""
        if osp.isdir(zarr_path):
            if not osp.exists(osp.join(zarr_path, ".zgroup")):
                raise FileNotFoundError(f"Directory '{zarr_path}' is not a Zarr store (.zgroup missing).")

    def _log_dataset_structure(self):
        """Log dataset structure without loading data"""
        logger.info("Dataset structure:")
        for key in self.data_keys:
            arr = self.data_group[key]
            logger.info(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    def _default_annotations_path(self, zarr_path: str) -> str:
        parent_dir = osp.dirname(osp.abspath(zarr_path.rstrip(osp.sep)))
        return osp.join(parent_dir, "annotations.json")

    def _load_annotations(self, path: str) -> Dict[str, Dict]:
        if not osp.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("annotations file must contain a JSON object")
        return data

    def _save_annotations(self):
        if not self.annotations_path:
            raise ValueError("annotations_path is not set")
        os.makedirs(osp.dirname(self.annotations_path), exist_ok=True)
        with open(self.annotations_path, "w", encoding="utf-8") as f:
            json.dump(self._annotations, f, indent=2, sort_keys=True)

    def _decode_str_value(self, value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _load_trajectory_ids(self) -> Optional[List[str]]:
        for key in ("trajectory_id", "trajectory_ids"):
            if key in self.meta_group:
                arr = np.array(self.meta_group[key][:])
                return [self._decode_str_value(v) for v in arr]
        for key in ("trajectory_id", "trajectory_ids"):
            if key in self.meta_group.attrs:
                attr_val = self.meta_group.attrs[key]
                if isinstance(attr_val, (list, tuple, np.ndarray)):
                    return [self._decode_str_value(v) for v in list(attr_val)]
                return [self._decode_str_value(attr_val)]
        return None

    def _print_episode_summary(self):
        """Print summary of all episodes"""
        episode_lengths = []
        for i in range(self.num_episodes):
            start = 0 if i == 0 else self.episode_ends[i - 1]
            end = self.episode_ends[i]
            episode_lengths.append(end - start)

        logger.info(f"Episode lengths: {episode_lengths}")
        logger.info(f"Total frames: {self.episode_ends[-1]}")

    def _get_blueprint(self):
        """Create rerun blueprint layout."""
        traj_origin = self._with_prefix("trajectory")
        tactile_origin = self._with_prefix("tactile_view")
        state_origin = self._with_prefix("state")
        tcp_origin = self._with_prefix("tcp_pose")
        human_teaching_origin = self._with_prefix("human_teaching")
        gripper_origin = self._with_prefix("gripper")
        wrench_origin = self._with_prefix("wrench")
        img_origin = self._with_prefix("images")

        # Use human teaching panel if available, otherwise gripper
        middle_panel_name = "Human Teaching" if self.has_human_teaching else "Gripper"
        middle_panel_origin = human_teaching_origin if self.has_human_teaching else gripper_origin

        return rrb.Blueprint(
            rrb.Horizontal(
                # Left column: 3D views + Action
                rrb.Vertical(
                    rrb.Spatial3DView(name="Robot Trajectory", origin=traj_origin),
                    rrb.Spatial3DView(name="Tactile Markers", origin=tactile_origin),
                    rrb.TimeSeriesView(name="State", origin=state_origin),
                    row_shares=[1, 1, 1],
                ),
                # Middle column: Time series
                rrb.Vertical(
                    rrb.TimeSeriesView(name="TCP Position", origin=tcp_origin),
                    rrb.TimeSeriesView(name=middle_panel_name, origin=middle_panel_origin),
                    rrb.TimeSeriesView(name="Forces & Torques", origin=wrench_origin),
                    row_shares=[1, 1, 1],
                ),
                # Right column: Camera images (2x2 grid layout)
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="D400 Camera", origin=f"{img_origin}/d400_camera"),
                        rrb.Spatial2DView(name="L500 Camera", origin=f"{img_origin}/l500_camera"),
                        column_shares=[1, 1],
                    ),
                    rrb.Horizontal(
                        rrb.Spatial2DView(name="Xense Camera 1", origin=f"{img_origin}/xense_camera_1"),
                        rrb.Spatial2DView(name="Xense Camera 2", origin=f"{img_origin}/xense_camera_2"),
                        column_shares=[1, 1],
                    ),
                    row_shares=[1, 1],
                ),
                column_shares=[2, 2, 3],
            ),
            rrb.TimePanel(timeline=self._timeline_name, fps=self.fps),
            collapse_panels=True,
        )

    def _init_rerun(self):
        """Initialize rerun viewer with blueprint."""
        # Use a fresh recording id when we want to drop previous episode data.
        if self.keep_only_current:
            self._recording_id = uuid.uuid4()
        self.rec = rr.RecordingStream(
            "Zarr Dataset Viewer",
            recording_id=self._recording_id,
            make_default=True,
            make_thread_default=True,
        )
        self._axes_logged.clear()

        self._spawn_and_connect_viewer()
        rr.send_blueprint(self._get_blueprint(), recording=self.rec)

    def cleanup(self):
        """Clean up all resources."""
        self._cleanup_episode_data()
        if self.rec is not None:
            try:
                self.rec.disconnect()
            except Exception:
                pass
        self._shutdown_viewer_process()
        try:
            rerun_bindings.flush_and_cleanup_orphaned_recordings()
        except Exception:
            pass
        gc.collect()

    def _reset_recording(self, episode_id: int):
        """Reset timeline state for a new episode."""
        self.rec.reset_time()
        try:
            self.rec.flush_blocking()
        except Exception:
            pass
        logger.info(f"Reset timeline for new episode {episode_id}")

    def _make_episode_prefix(self, episode_id: int) -> str:
        if not self.use_episode_prefix:
            return ""
        return f"session_{self._visualization_seq:04d}/episode_{episode_id:04d}"

    def _with_prefix(self, suffix: str) -> str:
        if not self._episode_prefix:
            return suffix
        return f"{self._episode_prefix}/{suffix}"

    def _make_timeline_name(self) -> str:
        if self.keep_only_current:
            return "frame"
        return f"frame_{self._visualization_seq:04d}"

    def _ensure_viewer_process(self) -> bool:
        if self._viewer_process is not None and self._viewer_process.poll() is None:
            return True
        cmd = [
            "rerun",
            "--port",
            str(self.viewer_port),
        ]
        self._viewer_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(0.2)
        return self._viewer_process.poll() is None

    def _shutdown_viewer_process(self):
        if self._viewer_process is None:
            return
        if self._viewer_process.poll() is None:
            try:
                os.killpg(self._viewer_process.pid, signal.SIGTERM)
            except Exception:
                self._viewer_process.terminate()
            try:
                self._viewer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._viewer_process.pid, signal.SIGKILL)
                except Exception:
                    self._viewer_process.kill()
        self._viewer_process = None

    def _connect_to_viewer(self, retries: int = 50, delay: float = 0.1):
        last_err = None
        for _ in range(retries):
            try:
                rr.connect_grpc(self._grpc_url, recording=self.rec)
                return
            except Exception as exc:
                last_err = exc
                time.sleep(delay)
        if last_err is not None:
            raise last_err

    def _spawn_and_connect_viewer(self, attempts: int = 5):
        last_err = None
        for _ in range(attempts):
            if self.manage_viewer_process:
                if not self._ensure_viewer_process():
                    last_err = RuntimeError("Failed to start rerun viewer process")
                    self._shutdown_viewer_process()
                    time.sleep(0.2)
                    continue
            try:
                self._connect_to_viewer()
                return
            except Exception as exc:
                last_err = exc
                if self.manage_viewer_process:
                    self._shutdown_viewer_process()
                time.sleep(0.2)
        if last_err is not None:
            raise last_err

    def get_episode_range(self, episode_id: int) -> Tuple[int, int]:
        """Get start and end indices for an episode"""
        if episode_id < 0 or episode_id >= self.num_episodes:
            raise ValueError(f"Episode {episode_id} out of range [0, {self.num_episodes - 1}]")

        start_idx = 0 if episode_id == 0 else int(self.episode_ends[episode_id - 1])
        end_idx = int(self.episode_ends[episode_id])
        return start_idx, end_idx

    def _cleanup_episode_data(self):
        """Clean up memory from previously loaded episode data."""
        if self._current_data is not None:
            for key in list(self._current_data.keys()):
                del self._current_data[key]
            self._current_data.clear()
            self._current_data = None
            gc.collect()
            logger.debug("Previous episode data cleaned up")

    def load_episode_data(self, episode_id: int) -> Dict[str, np.ndarray]:
        """
        Load only data for a specific episode (lazy loading)

        Args:
            episode_id: Episode to load

        Returns:
            Dictionary of numpy arrays for this episode only
        """
        start_idx, end_idx = self.get_episode_range(episode_id)
        n_frames = end_idx - start_idx

        logger.info(f"Loading episode {episode_id}: frames {start_idx}-{end_idx} ({n_frames} frames)")

        data = {}
        for key in tqdm(self.data_keys, desc="Loading episode", unit="array"):
            data[key] = np.array(self.data_group[key][start_idx:end_idx])

        return data

    def visualize_episode(self, episode_id: int, first_run: bool = False):
        """
        Visualize a specific episode

        Args:
            episode_id: Episode to visualize
            first_run: Whether this is the first episode being visualized
        """
        if episode_id < 0 or episode_id >= self.num_episodes:
            logger.error(f"Episode {episode_id} out of range [0, {self.num_episodes - 1}]")
            return

        if not first_run and self.keep_only_current:
            # Hard reset viewer to drop previous episode completely
            try:
                self.rec.disconnect()
            except Exception:
                pass
            if self.manage_viewer_process:
                self._shutdown_viewer_process()
            self._init_rerun()

        self.current_episode = episode_id
        self._visualization_seq += 1
        self._episode_prefix = self._make_episode_prefix(episode_id)
        self._timeline_name = self._make_timeline_name()
        start_idx, end_idx = self.get_episode_range(episode_id)
        n_frames = end_idx - start_idx
        self._current_n_frames = n_frames

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Visualizing Episode {episode_id} ({n_frames} frames)")
        logger.info(f"{'=' * 50}")

        self._cleanup_episode_data()

        rr.send_blueprint(self._get_blueprint(), recording=self.rec)

        if not first_run and not self.keep_only_current:
            self._reset_recording(episode_id)

        # Load episode data (lazy - only this episode)
        data = self.load_episode_data(episode_id)
        self._current_data = data

        rr_log = self.rec

        if self.pose_keys:
            for pose_key in self.pose_keys:
                if pose_key not in data:
                    continue
                poses = data[pose_key]
                robot_name = pose_key.split("_")[0]
                if poses.shape[1] >= 3:
                    positions = poses[:, :3]
                    rr_log.log(
                        self._with_prefix(f"trajectory/{robot_name}/path"),
                        LineStrips3D([positions], colors=[[100, 180, 255]], radii=[0.0012]),
                        static=True,
                    )
        action_positions = None
        if "action" in data:
            action = data["action"]
            if len(action.shape) == 2 and action.shape[1] >= 3:
                action_positions = action[:, :3]
        if action_positions is not None:
            action_robot_name = self.pose_keys[0].split("_")[0] if self.pose_keys else "robot"
            rr_log.log(
                self._with_prefix(f"trajectory/{action_robot_name}/action_path"),
                LineStrips3D([action_positions], colors=[[255, 140, 0]], radii=[0.0009]),
                static=True,
            )

        logger.info("Logging frames to rerun...")
        for i in tqdm(range(n_frames), desc=f"Episode {episode_id}", unit="frame"):
            rr_log.set_time(self._timeline_name, sequence=i)

            self._log_poses(rr_log, data, i)
            self._log_action_pose(rr_log, data, i)
            self._log_images(rr_log, data, i)
            self._log_wrench(rr_log, data, i)
            self._log_velocities(rr_log, data, i)
            self._log_gripper(rr_log, data, i)
            self._log_human_teaching(rr_log, data, i)
            self._log_pose_components(rr_log, data, i)
            self._log_state(rr_log, data, i)
            self._log_marker3d(rr_log, data, i)

        logger.info(f"Episode {episode_id} visualization complete")
        self._print_command_hints()

    def _print_command_hints(self):
        """Print available commands as a reminder"""
        print("\n" + "-" * 50)
        print("Commands: <number> | n(ext) | p(rev) | l(ist) | i(nfo) | r(eload) | a(nnotate) | q(uit)")
        print("-" * 50)

    def _get_trajectory_id(self, episode_id: int) -> str:
        if self._trajectory_ids is None:
            return f"{episode_id:03d}"
        if episode_id < 0 or episode_id >= len(self._trajectory_ids):
            logger.warning("trajectory_id index out of range; using episode id as trajectory_id")
            return f"{episode_id:03d}"
        return self._trajectory_ids[episode_id]

    def _prompt_bool(self, prompt: str) -> bool:
        while True:
            val = input(prompt).strip().lower()
            if val in ("y", "yes", "true", "t", "1"):
                return True
            if val in ("n", "no", "false", "f", "0"):
                return False
            print("Please enter y/yes or n/no.")

    def _prompt_int(self, prompt: str, min_value: int, max_value: int, allow_empty: bool = False) -> Optional[int]:
        while True:
            val = input(prompt).strip()
            if allow_empty and val == "":
                return None
            try:
                parsed = int(val)
            except ValueError:
                print("Please enter an integer.")
                continue
            if parsed < min_value or parsed > max_value:
                print(f"Value out of range [{min_value}, {max_value}].")
                continue
            return parsed

    def _prompt_keyframes(self, prompt: str, min_value: int, max_value: int) -> List[int]:
        while True:
            raw = input(prompt).strip()
            if raw == "":
                return []
            raw = raw.replace(",", " ")
            parts = [p for p in raw.split() if p]
            keyframes = []
            invalid = False
            for part in parts:
                try:
                    val = int(part)
                except ValueError:
                    invalid = True
                    break
                if val < min_value or val > max_value:
                    invalid = True
                    break
                keyframes.append(val)
            if invalid:
                print(f"Keyframes must be integers in range [{min_value}, {max_value}].")
                continue
            return sorted(set(keyframes))

    def annotate_current_episode(self):
        """Annotate the current episode with success/failure labels."""
        traj_id = self._get_trajectory_id(self.current_episode)
        start_idx, end_idx = self.get_episode_range(self.current_episode)
        total_frames = self._current_n_frames or (end_idx - start_idx)
        max_frame = max(total_frames - 1, 0)

        print("\n" + "=" * 50)
        print(f"Annotating trajectory_id: {traj_id}")
        print(f"Total frames (0-based): {total_frames}")
        print("=" * 50)

        is_success = self._prompt_bool("is_success? (y/n): ")
        success_frame = None
        if is_success:
            success_frame = self._prompt_int(
                f"success_frame (0-{max_frame}): ",
                min_value=0,
                max_value=max_frame,
                allow_empty=False,
            )
        else:
            success_frame = self._prompt_int(
                f"success_frame (0-{max_frame}, empty for none): ",
                min_value=0,
                max_value=max_frame,
                allow_empty=True,
            )

        retry_keyframes = self._prompt_keyframes(
            f"retry_keyframes (comma/space separated, 0-{max_frame}, empty for none): ",
            min_value=0,
            max_value=max_frame,
        )

        record = {
            "trajectory_id": traj_id,
            "is_success": is_success,
            "success_frame": success_frame,
            "total_frames": total_frames,
            "retry_keyframes": retry_keyframes,
        }
        self._annotations[traj_id] = record
        self._save_annotations()
        print(f"Saved annotation for {traj_id} -> {self.annotations_path}")

    def _log_poses(self, rr_log, data: Dict, frame_idx: int):
        """Log robot poses"""
        for pose_key in self.pose_keys:
            if pose_key not in data:
                continue
            poses = data[pose_key]
            robot_name = pose_key.split("_")[0]
            if poses.shape[1] >= 7:
                translation = poses[frame_idx, :3]
                rotation_quat = poses[frame_idx, 3:7]

                rr_log.log(
                    self._with_prefix(f"robot/{robot_name}/tcp"),
                    Transform3D(translation=translation, rotation=Quaternion(xyzw=rotation_quat)),
                )
                axes_entity = self._with_prefix(f"trajectory/{robot_name}/current_pose")
                rr_log.log(axes_entity, Transform3D(translation=translation, rotation=Quaternion(xyzw=rotation_quat)))
                if axes_entity not in self._axes_logged:
                    axis_len = 0.05
                    axis_radius = 0.002
                    rr_log.log(
                        axes_entity,
                        Arrows3D(
                            origins=[[0.0, 0.0, 0.0]] * 3,
                            vectors=[
                                [axis_len, 0.0, 0.0],
                                [0.0, axis_len, 0.0],
                                [0.0, 0.0, axis_len],
                            ],
                            colors=[
                                [255, 0, 0],
                                [0, 255, 0],
                                [0, 0, 255],
                            ],
                            radii=[axis_radius] * 3,
                        ),
                        static=True,
                    )
                    self._axes_logged.add(axes_entity)

    def _log_action_pose(self, rr_log, data: Dict, frame_idx: int):
        """Log action pose axes"""
        if "action" not in data:
            return
        action = data["action"]
        if len(action.shape) != 2 or action.shape[1] < 7:
            return
        translation = action[frame_idx, :3]
        rotation_quat = action[frame_idx, 3:7]
        robot_name = self.pose_keys[0].split("_")[0] if self.pose_keys else "robot"
        axes_entity = self._with_prefix(f"trajectory/{robot_name}/action_pose")
        rr_log.log(axes_entity, Transform3D(translation=translation, rotation=Quaternion(xyzw=rotation_quat)))
        if axes_entity not in self._axes_logged:
            axis_len = 0.05
            axis_radius = 0.0018
            rr_log.log(
                axes_entity,
                Arrows3D(
                    origins=[[0.0, 0.0, 0.0]] * 3,
                    vectors=[
                        [axis_len, 0.0, 0.0],
                        [0.0, axis_len, 0.0],
                        [0.0, 0.0, axis_len],
                    ],
                    colors=[
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                    ],
                    radii=[axis_radius] * 3,
                ),
                static=True,
            )
            self._axes_logged.add(axes_entity)

    def _log_images(self, rr_log, data: Dict, frame_idx: int):
        """Log camera images"""
        for img_key in self.image_keys:
            if img_key not in data:
                continue
            images = data[img_key]
            if len(images.shape) == 4:
                camera_name = img_key.replace("_img", "").replace("_image", "")
                camera_name = self._canonicalize_camera_name(camera_name)
                img = images[frame_idx]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                rr_log.log(self._with_prefix(f"images/{camera_name}"), Image(img))

    def _log_wrench(self, rr_log, data: Dict, frame_idx: int):
        """Log wrench data"""
        for wrench_key in self.wrench_keys:
            if wrench_key not in data:
                continue
            wrenches = data[wrench_key]
            robot_name = wrench_key.split("_")[0]
            forces = wrenches[frame_idx, :3] if wrenches.shape[1] >= 3 else wrenches[frame_idx]
            torques = wrenches[frame_idx, 3:6] if wrenches.shape[1] >= 6 else np.zeros(3)

            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                if axis_idx < len(forces):
                    rr_log.log(self._with_prefix(f"wrench/{robot_name}/force_{axis_name}"), Scalars([forces[axis_idx]]))
                if axis_idx < len(torques):
                    rr_log.log(
                        self._with_prefix(f"wrench/{robot_name}/torque_{axis_name}"), Scalars([torques[axis_idx]])
                    )

    def _log_velocities(self, rr_log, data: Dict, frame_idx: int):
        """Log velocity data"""
        for vel_key in self.vel_keys:
            if vel_key not in data:
                continue
            velocities = data[vel_key]
            robot_name = vel_key.split("_")[0]
            linear_vel = velocities[frame_idx, :3] if velocities.shape[1] >= 3 else velocities[frame_idx]
            angular_vel = velocities[frame_idx, 3:6] if velocities.shape[1] >= 6 else np.zeros(3)

            for axis_idx, axis_name in enumerate(["x", "y", "z"]):
                if axis_idx < len(linear_vel):
                    rr_log.log(
                        self._with_prefix(f"velocities/{robot_name}/linear_{axis_name}"),
                        Scalars([linear_vel[axis_idx]]),
                    )
                if axis_idx < len(angular_vel):
                    rr_log.log(
                        self._with_prefix(f"velocities/{robot_name}/angular_{axis_name}"),
                        Scalars([angular_vel[axis_idx]]),
                    )

    def _log_gripper(self, rr_log, data: Dict, frame_idx: int):
        """Log gripper data"""
        for width_key in self.gripper_width_keys:
            if width_key not in data:
                continue
            widths = data[width_key].flatten()
            robot_name = width_key.split("_")[0]
            rr_log.log(self._with_prefix(f"gripper/{robot_name}/width"), Scalars([widths[frame_idx]]))

        # Also log gripper state from 8D pose
        for pose_key in self.pose_keys:
            if pose_key not in data:
                continue
            poses = data[pose_key]
            if poses.shape[1] == 8:
                robot_name = pose_key.split("_")[0]
                rr_log.log(self._with_prefix(f"gripper/{robot_name}/state"), Scalars([poses[frame_idx, 7]]))

    def _log_human_teaching(self, rr_log, data: Dict, frame_idx: int):
        """Log human teaching flag"""
        if "is_human_teaching" not in data:
            return
        value = float(data["is_human_teaching"][frame_idx])
        rr_log.log(self._with_prefix("human_teaching/is_human"), Scalars([value]))

    def _log_pose_components(self, rr_log, data: Dict, frame_idx: int):
        """Log TCP pose position components"""
        for pose_key in self.pose_keys:
            if pose_key not in data:
                continue
            poses = data[pose_key]
            robot_name = pose_key.split("_")[0]
            if poses.shape[1] >= 3:
                rr_log.log(self._with_prefix(f"tcp_pose/{robot_name}/position/x"), Scalars([poses[frame_idx, 0]]))
                rr_log.log(self._with_prefix(f"tcp_pose/{robot_name}/position/y"), Scalars([poses[frame_idx, 1]]))
                rr_log.log(self._with_prefix(f"tcp_pose/{robot_name}/position/z"), Scalars([poses[frame_idx, 2]]))

    def _log_action(self, rr_log, data: Dict, frame_idx: int):
        """Log action data"""
        if "action" not in data:
            return
        action = data["action"][frame_idx]
        for dim in range(min(len(action), 10)):
            rr_log.log(self._with_prefix(f"action/dim_{dim}"), Scalars([action[dim]]))

    def _log_state(self, rr_log, data: Dict, frame_idx: int):
        """Log state data"""
        state_vec = None
        tcp_pose = data.get("robot_tcp_pose")
        if tcp_pose is None:
            for pose_key in self.pose_keys:
                tcp_pose = data.get(pose_key)
                if tcp_pose is not None:
                    break

        if tcp_pose is not None:
            state_vec = tcp_pose[frame_idx].reshape(-1)[:7]
        elif "state" in data:
            state_vec = np.asarray(data["state"][frame_idx]).reshape(-1)[:7]

        if state_vec is None:
            return

        for dim in range(len(state_vec)):
            rr_log.log(self._with_prefix(f"state/dim_{dim}"), Scalars([state_vec[dim]]))

    def _log_marker3d(self, rr_log, data: Dict, frame_idx: int):
        """Log marker3d data"""
        for camera_idx, marker3d_key in enumerate(self.marker3d_keys):
            if marker3d_key not in data:
                continue
            marker3d_frame = data[marker3d_key][frame_idx]
            camera_name = marker3d_key.replace("_marker3d", "")
            camera_name = self._canonicalize_camera_name(camera_name)

            points_3d = self._process_marker3d_to_points(marker3d_frame, camera_idx)
            if points_3d is not None:
                rr_log.log(self._with_prefix(f"tactile_view/{camera_name}/points"), Points3D(positions=points_3d))

    def _process_marker3d_to_points(
        self,
        marker3d_frame: np.ndarray,
        camera_idx: int = 0,
        camera_offset_x: float = 0.0,
        camera_offset_z: float = 300.0,
        scale_z: float = 0.1,
    ) -> Optional[np.ndarray]:
        """Convert marker3d data to 3D point cloud for visualization

        Args:
            marker3d_frame: Marker3d data (26, 14, 3) with (x, y, z) coordinates
            camera_idx: Camera index for offset
            camera_offset_x: X offset between cameras
            camera_offset_z: Z offset between cameras
            scale_z: Scale factor for z-axis (depth) visualization

        Returns:
            3D points array for rerun visualization
        """
        if len(marker3d_frame.shape) == 3 and marker3d_frame.shape[2] == 3:
            # Shape: (26, 14, 3) - marker3d format
            num_markers = marker3d_frame.shape[0]
            camera_x_offset = camera_idx * camera_offset_x
            camera_z_offset = camera_idx * camera_offset_z

            points_3d_list = []
            for marker_idx in range(num_markers):
                marker_points = marker3d_frame[marker_idx]  # (14, 3)
                # Check for valid points (not all zeros and not nan)
                valid_mask = ~(np.isnan(marker_points).any(axis=1) | (marker_points[:, :2] == 0).all(axis=1))

                if valid_mask.sum() > 0:
                    valid_points = marker_points[valid_mask]
                    # Use x, y from marker, and use marker row index for visual z-spread
                    # Add the actual depth (z) as additional visual offset
                    points_3d = np.zeros((len(valid_points), 3), dtype=np.float32)
                    points_3d[:, 0] = valid_points[:, 0] + camera_x_offset  # x
                    points_3d[:, 1] = valid_points[:, 1]  # y
                    # z: combine row index spread + actual depth value scaled
                    points_3d[:, 2] = marker_idx * scale_z + camera_z_offset + valid_points[:, 2] * 100
                    points_3d_list.append(points_3d)

            if points_3d_list:
                return np.vstack(points_3d_list)
            return None

        elif len(marker3d_frame.shape) == 2 and marker3d_frame.shape[1] == 3:
            # Shape: (N, 3) - flat marker3d format
            valid_mask = ~(np.isnan(marker3d_frame).any(axis=1) | (marker3d_frame[:, :2] == 0).all(axis=1))
            if valid_mask.sum() > 0:
                valid_points = marker3d_frame[valid_mask]
                camera_x_offset = camera_idx * camera_offset_x
                camera_z_offset = camera_idx * camera_offset_z
                points_3d = np.zeros((len(valid_points), 3), dtype=np.float32)
                points_3d[:, 0] = valid_points[:, 0] + camera_x_offset
                points_3d[:, 1] = valid_points[:, 1]
                points_3d[:, 2] = camera_z_offset + valid_points[:, 2] * 100
                return points_3d
            return None

        return None

    def list_episodes(self):
        """Print list of all episodes with their lengths"""
        print(f"\n{'=' * 50}")
        print(f"Episodes in dataset: {self.num_episodes}")
        print(f"{'=' * 50}")
        for i in range(self.num_episodes):
            start, end = self.get_episode_range(i)
            length = end - start
            marker = " <-- current" if i == self.current_episode else ""
            print(f"  Episode {i:4d}: {length:6d} frames (indices {start:7d} - {end:7d}){marker}")
        print(f"{'=' * 50}\n")

    def show_info(self):
        """Show info about current episode"""
        start, end = self.get_episode_range(self.current_episode)
        length = end - start
        print(f"\nCurrent: Episode {self.current_episode}")
        print(f"  Frames: {length} (indices {start} - {end})")
        print(f"  Total episodes: {self.num_episodes}")
        print()

    def run_interactive(self, start_episode: int = 0):
        """
        Run interactive mode

        Args:
            start_episode: Episode to start with
        """
        print("\n" + "=" * 50)
        print("Interactive Zarr Viewer")
        print(f"Dataset: {self.zarr_path}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Annotation: {self.annotations_path}")
        print("=" * 50)

        # Visualize initial episode (will print command hints after)
        self.visualize_episode(start_episode, first_run=True)

        while True:
            try:
                cmd = input(f"Episode [{self.current_episode}/{self.num_episodes - 1}] > ").strip().lower()

                if not cmd:
                    continue

                if cmd in ("q", "quit", "exit"):
                    print("Exiting...")
                    break

                elif cmd in ("n", "next"):
                    next_ep = self.current_episode + 1
                    if next_ep >= self.num_episodes:
                        print(f"Already at last episode ({self.current_episode})")
                    else:
                        self.visualize_episode(next_ep)

                elif cmd in ("p", "prev", "previous"):
                    prev_ep = self.current_episode - 1
                    if prev_ep < 0:
                        print(f"Already at first episode (0)")
                    else:
                        self.visualize_episode(prev_ep)

                elif cmd in ("l", "list"):
                    self.list_episodes()

                elif cmd in ("i", "info"):
                    self.show_info()

                elif cmd in ("r", "reload"):
                    self.visualize_episode(self.current_episode)

                elif cmd in ("a", "annotate"):
                    self.annotate_current_episode()

                elif cmd.isdigit():
                    ep_id = int(cmd)
                    if 0 <= ep_id < self.num_episodes:
                        self.visualize_episode(ep_id)
                    else:
                        print(f"Episode {ep_id} out of range [0, {self.num_episodes - 1}]")

                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'l' to list episodes, or a number to switch episode")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except EOFError:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        # Clean up resources on exit
        self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive zarr dataset visualizer using rerun",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Commands:
  <number>  - Switch to episode <number>
  n/next    - Next episode
  p/prev    - Previous episode
  l/list    - List all episodes
  i/info    - Show current episode info
  r/reload  - Reload current episode
  a/annotate- Annotate current episode
  q/quit    - Exit

Example:
  python visualize_zarr_with_rerun.py --zarr_path ./data/replay_buffer.zarr
  python visualize_zarr_with_rerun.py --zarr_path ./data/replay_buffer.zarr --episode 5
        """,
    )

    parser.add_argument(
        "--zarr_path",
        type=str,
        default="eval_records/pi05_franka_position_control_lora/20260126/replay_buffer.zarr",
        # default="demo_records/replay_buffer.zarr",
        help="Path to zarr dataset file",
    )

    parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate")

    parser.add_argument("--episode", "-e", type=int, default=0, help="Starting episode ID (0-indexed)")

    args = parser.parse_args()

    # Create viewer
    viewer = ZarrDatasetViewer(
        args.zarr_path,
        args.fps,
    )

    try:
        viewer.run_interactive(args.episode)
    finally:
        # Ensure cleanup on any exit
        viewer.cleanup()


if __name__ == "__main__":
    main()
