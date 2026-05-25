# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replay a LeRobot dataset on G2 robot with inverse kinematics conversion.

This script is designed for replaying datasets whose actions are in end-effector (EE)
space (e.g. l.ee.x, l.ee.y, ...) onto the G2 robot, which expects joint position commands.
An IK processor converts EE poses to joint angles on-the-fly.

By default EE poses are interpreted in the base/world frame.  Set ``reference_frame``
to a URDF frame (e.g. ``arm_base_link``) to use chest-frame EE poses instead.

Example (world frame):
    python -m lerobot.scripts.replay_g2_dataset \
        --robot.type=g2 \
        --robot.use_left_arm=true \
        --robot.use_right_arm=true \
        --dataset.root=/home/ck/finetune-data/g2_dual_arm_g3 \
        --dataset.episode=0 \
        --dataset.speed_factor=0.5

Example (chest frame):
    python -m lerobot.scripts.replay_g2_dataset \
        --robot.type=g2 \
        --robot.use_left_arm=true \
        --robot.use_right_arm=true \
        --dataset.root=/home/ck/finetune-data/g2_dual_arm_g3 \
        --dataset.episode=0 \
        --reference_frame=arm_base_link
"""

import logging
import numpy as np
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

from pynput import keyboard

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import DualArmKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    RobotAction,
    RobotObservation,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import (
    RobotConfig,
    g2,
    make_robot_from_config,
)
from lerobot.robots.g2.g2_constants import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from lerobot.robots.g2.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEObservationG2,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, log_say


@dataclass
class G2DatasetReplayConfig:
    # Local root directory of the dataset (e.g. '/home/ck/finetune-data/g2_dual_arm_g3')
    root: str | Path
    # Episode index to replay
    episode: int = 0
    # Replay speed multiplier. 1.0 = real-time, 0.5 = half speed (safer for validation)
    speed_factor: float = 1.0
    # Frame index to start from (useful for skipping initial frames)
    start_frame: int = 0
    # Before replaying, smoothly move the robot to the dataset episode's first
    # observation state (start pose). This calibrates the initial configuration so
    # that absolute EE actions from the dataset can be replayed directly.
    calibrate_start: bool = True


@dataclass
class G2ReplayConfig:
    robot: RobotConfig
    dataset: G2DatasetReplayConfig
    # Kinematics reference frame. None (or "world") uses the base/world frame;
    # set to a URDF frame name (e.g. "arm_base_link") to use chest-frame EE poses.
    reference_frame: str | None = None
    # Use vocal synthesis to read events
    play_sounds: bool = False
    ee_tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.007])


@parser.wrap()
def replay(cfg: G2ReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    # Load dataset from local root
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # repo_id is used only as an identifier when root is provided
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        episodes=[cfg.dataset.episode],
        download_videos=False,
    )
    actions = dataset.select_columns(ACTION)

    # Build IK action processor for G2
    robot_observation_processor = None
    if robot.name == "g2":
        g2_urdf_path = "src/lerobot/robots/g2/G2/genie_robot/urdf/G2_t2_crs/model.urdf"
        kinematics_solver = DualArmKinematics(
            urdf_path=g2_urdf_path,
            left_frame_name="arm_l_end_link" if robot.config.use_left_arm else None,
            left_joint_names=LEFT_ARM_JOINT_NAMES if robot.config.use_left_arm else None,
            right_frame_name="arm_r_end_link" if robot.config.use_right_arm else None,
            right_joint_names=RIGHT_ARM_JOINT_NAMES if robot.config.use_right_arm else None,
            reference_frame_name=cfg.reference_frame,
            left_tcp_offset=cfg.ee_tcp_offset,
            right_tcp_offset=cfg.ee_tcp_offset,
            max_iterations=10
        )
        robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
            steps=[
                ForwardKinematicsJointsToEEObservationG2(
                    kinematics=kinematics_solver,
                ),
            ],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )

        robot_action_processor = RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ](
            steps=[
                InverseKinematicsEEToJoints(
                    motor_names=list(robot.motors),
                    kinematics=kinematics_solver,
                    initial_guess_current_joints=True,
                    use_relative_actions=False,
                    ik_position_weight=1.0,
                    ik_orientation_weight=1,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )
    else:
        from lerobot.processor import make_default_robot_action_processor
        robot_action_processor = make_default_robot_action_processor()
        logging.warning(
            f"Robot '{robot.name}' is not g2. Replaying without IK conversion. "
            "Make sure the dataset action space matches the robot's expected action space."
        )

    # Verify action feature compatibility
    dataset_action_names = dataset.features[ACTION]["names"]
    robot_action_names = list(robot.action_features.keys())
    logging.info(f"Dataset action names: {dataset_action_names}")
    logging.info(f"Robot action names:   {robot_action_names}")

    # Determine effective FPS
    effective_fps = dataset.fps * cfg.dataset.speed_factor
    if cfg.dataset.speed_factor != 1.0:
        logging.info(
            f"Speed factor: {cfg.dataset.speed_factor}. Replaying at {effective_fps:.1f} FPS "
            f"(original: {dataset.fps} FPS)."
        )

    robot.connect()

    # Optionally calibrate to the dataset episode's start pose
    if cfg.dataset.calibrate_start:
        obs_states = dataset.select_columns(OBS_STATE)
        first_obs: dict[str, float] = {}
        for i, name in enumerate(dataset.features[OBS_STATE]["names"]):
            first_obs[name] = float(obs_states[0][OBS_STATE][i])
        logging.info(f"fist obs for calibrate:{obs_states[0][OBS_STATE]}")

        calibration_start_obs = robot.get_observation()
        if robot_observation_processor is not None:
            calibration_start_obs = robot_observation_processor(calibration_start_obs)

        steps = 50
        duration_s = 5.0
        print(f"Calibrating to dataset start pose ({steps} steps, {duration_s}s)...")
        for step in range(1, steps + 1):
            # Refresh the live observation for IK, but keep the calibration target
            # trajectory anchored to the initial calibration pose. Otherwise the
            # interpolation target drifts together with solver error.
            current_obs = robot.get_observation()
            if robot_observation_processor is not None:
                current_obs = robot_observation_processor(current_obs)

            alpha = step / steps
            # Interpolate EE pose between the initial calibration observation and
            # the dataset start pose.
            interp_action: dict[str, float] = {}
            for prefix in ("l", "r"):
                # Linear interpolation for translation
                for axis in ("x", "y", "z"):
                    key = f"{prefix}.ee.{axis}"
                    if key in first_obs and key in calibration_start_obs:
                        interp_action[key] = float(
                            calibration_start_obs[key]
                            + alpha * (first_obs[key] - calibration_start_obs[key])
                        )
                # Spherical linear interpolation for rotation (rotvec)
                rot_keys = [f"{prefix}.ee.{a}" for a in ("wx", "wy", "wz")]
                if all(k in first_obs and k in calibration_start_obs for k in rot_keys):
                    curr_rot = Rotation.from_rotvec(
                        np.array([float(calibration_start_obs[k]) for k in rot_keys])
                    )
                    target_rot = Rotation.from_rotvec(
                        np.array([float(first_obs[k]) for k in rot_keys])
                    )
                    rel_rot = target_rot * curr_rot.inv()
                    interp_rot = Rotation.from_rotvec(alpha * rel_rot.as_rotvec()) * curr_rot
                    for k, v in zip(rot_keys, interp_rot.as_rotvec()):
                        interp_action[k] = float(v)
                gripper_key = f"{prefix}.ee.gripper.pos"
                if gripper_key in first_obs:
                    interp_action[gripper_key] = float(first_obs[gripper_key])

            calibrate_action = robot_action_processor((interp_action, current_obs))
            robot.send_action(calibrate_action)
            time.sleep(duration_s / steps)

        print("Calibration done.")
        final_obs = robot_observation_processor(robot.get_observation())
        for prefix in ("l", "r"):
            err = np.linalg.norm([final_obs[f"{prefix}.ee.{a}"] - first_obs[f"{prefix}.ee.{a}"] for a in "xyz"])
            logging.info(f"[{prefix}] post-calibration pos error: {err*1000:.2f} mm")
            if err > 0.01:
                raise RuntimeError(f"{prefix} arm calibration failed: residual {err*1000:.1f} mm")

    # Space-as-trigger: hold SPACE to replay, release to pause
    running = threading.Event()
    stop_replay = threading.Event()

    def on_press(key):
        if key == keyboard.Key.space:
            running.set()
        elif key == keyboard.Key.esc:
            print("[ESC] Stopping replay...")
            stop_replay.set()
            running.set()  # unblock loop so it can exit
            return False

    def on_release(key):
        if key == keyboard.Key.space:
            running.clear()

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    total_frames = dataset.num_frames
    start_frame = max(0, min(cfg.dataset.start_frame, total_frames - 1))

    try:
        print("=" * 50)
        print("Robot connected.")
        if cfg.dataset.calibrate_start:
            print("Calibrated to dataset start pose.")
        print("Hold SPACE to replay, release to pause.")
        print("Press ESC to exit.")
        print("=" * 50)

        for idx in range(start_frame, total_frames):
            if stop_replay.is_set():
                print("Replay stopped by user.")
                break

            # Block here while SPACE is not held
            while not running.is_set() and not stop_replay.is_set():
                time.sleep(0.01)

            if stop_replay.is_set():
                break

            loop_t = time.perf_counter()

            action_array = actions[idx][ACTION]
            action: dict[str, float] = {}
            for i, name in enumerate(dataset_action_names):
                action[name] = float(action_array[i])

            robot_obs = robot.get_observation()
            if robot_observation_processor is not None:
                robot_obs = robot_observation_processor(robot_obs)
            processed_action = robot_action_processor((action, robot_obs))
            _ = robot.send_action(processed_action)

            dt_s = time.perf_counter() - loop_t
            sleep_s = max(1.0 / effective_fps - dt_s, 0.0)
            precise_sleep(sleep_s)

        log_say("Replay finished", cfg.play_sounds)
    finally:
        kb_listener.stop()
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
