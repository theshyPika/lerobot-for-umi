# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Reset the G2 arms to a target EE pose, e.g. the observation EE mean of a pi05 model.

The reset trajectory mirrors the ``calibrate_start`` block of
``replay_g2_dataset.py``: we read the current EE pose via FK, then interpolate
linearly in translation and slerp in orientation over a short trajectory,
streaming joint commands solved by the dual-arm IK processor at each step.

The target pose can come from either:
- A JSON file written by ``extract_obs_ee_mean.py`` (``{"values": {...}}``), or
- A pretrained_model directory (the script will load the normalizer mean itself).

Example (using a pretrained model directly):
    python -m lerobot.scripts.reset_g2_to_ee_mean \
        --robot.type=g2 \
        --robot.use_left_arm=true \
        --robot.use_right_arm=true \
        --target.model_path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model

Example (using a pre-extracted JSON):
    python -m lerobot.scripts.reset_g2_to_ee_mean \
        --robot.type=g2 \
        --robot.use_left_arm=true \
        --robot.use_right_arm=true \
        --target.json_path=/tmp/g2_ee_mean.json
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

from lerobot.configs import parser
from lerobot.model.kinematics import DualArmKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.g2.g2_constants import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from lerobot.robots.g2.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEObservationG2,
    InverseKinematicsEEToJoints,
)
from lerobot.scripts.extract_obs_ee_mean import extract_obs_ee_mean
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging, log_say


@dataclass
class G2ResetTargetConfig:
    # One of these must be set.
    model_path: str | Path | None = None
    json_path: str | Path | None = None
    # Which statistic to read from the model normalizer (mean, q50, ...).
    stat: str = "mean"


@dataclass
class G2ResetConfig:
    robot: RobotConfig
    target: G2ResetTargetConfig
    # Kinematics reference frame; must match the policy/dataset frame.
    # None (or "world") uses the base/world frame; e.g. "arm_base_link" for chest-frame poses.
    reference_frame: str | None = None
    # Interpolation parameters for the reset motion.
    steps: int = 50
    duration_s: float = 5.0
    # Acceptable post-reset positional error per arm (meters). Set <= 0 to skip the check.
    position_tolerance: float = 0.01
    # Use vocal synthesis to read events.
    play_sounds: bool = False
    ee_tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.007])


def _load_target_ee(target: G2ResetTargetConfig) -> dict[str, float]:
    if target.json_path and target.model_path:
        raise ValueError("Specify only one of target.json_path or target.model_path.")
    if target.json_path:
        path = Path(target.json_path)
        with path.open() as f:
            payload = json.load(f)
        if "values" in payload and isinstance(payload["values"], dict):
            values = payload["values"]
        else:
            values = payload  # Allow a flat {name: value} JSON.
        return {str(k): float(v) for k, v in values.items()}
    if target.model_path:
        return extract_obs_ee_mean(Path(target.model_path), stat=target.stat)
    raise ValueError("Must specify either target.json_path or target.model_path.")


@parser.wrap()
def reset(cfg: G2ResetConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    target_ee = _load_target_ee(cfg.target)
    logging.info("Target EE pose:\n%s", pformat(target_ee))

    robot = make_robot_from_config(cfg.robot)
    if robot.name != "g2":
        raise RuntimeError(f"This script only supports robot.type=g2 (got '{robot.name}').")

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
        max_iterations=10,
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEEObservationG2(kinematics=kinematics_solver)],
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
                ik_position_weight=1.0,
                ik_orientation_weight=1.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    active_prefixes: list[str] = []
    if robot.config.use_left_arm:
        active_prefixes.append("l")
    if robot.config.use_right_arm:
        active_prefixes.append("r")

    robot.connect()
    try:
        start_obs = robot_observation_processor(robot.get_observation())

        steps = max(int(cfg.steps), 1)
        duration_s = max(float(cfg.duration_s), 1e-3)
        step_dt = duration_s / steps
        print(f"Resetting G2 to target EE pose ({steps} steps, {duration_s:.2f}s)...")

        # Precompute the start->target rotation once per arm so each step is a
        # plain slerp factor on the same rotation axis.
        rot_starts: dict[str, Rotation] = {}
        rot_deltas: dict[str, Rotation] = {}
        for prefix in active_prefixes:
            rot_keys = [f"{prefix}.ee.{a}" for a in ("wx", "wy", "wz")]
            if not all(k in start_obs and k in target_ee for k in rot_keys):
                continue
            start_rot = Rotation.from_rotvec(np.array([float(start_obs[k]) for k in rot_keys]))
            target_rot = Rotation.from_rotvec(np.array([float(target_ee[k]) for k in rot_keys]))
            rot_starts[prefix] = start_rot
            rot_deltas[prefix] = target_rot * start_rot.inv()

        for step in range(1, steps + 1):
            current_obs = robot_observation_processor(robot.get_observation())
            alpha = step / steps

            interp_action: dict[str, float] = {}
            for prefix in active_prefixes:
                for axis in ("x", "y", "z"):
                    key = f"{prefix}.ee.{axis}"
                    if key in target_ee and key in start_obs:
                        interp_action[key] = float(
                            start_obs[key] + alpha * (target_ee[key] - start_obs[key])
                        )
                if prefix in rot_starts:
                    rot_keys = [f"{prefix}.ee.{a}" for a in ("wx", "wy", "wz")]
                    interp_rot = Rotation.from_rotvec(alpha * rot_deltas[prefix].as_rotvec()) * rot_starts[prefix]
                    for k, v in zip(rot_keys, interp_rot.as_rotvec()):
                        interp_action[k] = float(v)
                gripper_key = f"{prefix}.ee.gripper.pos"
                if gripper_key in target_ee:
                    interp_action[gripper_key] = float(target_ee[gripper_key])

            processed_action = robot_action_processor((interp_action, current_obs))
            robot.send_action(processed_action)
            time.sleep(step_dt)

        print("Reset done.")
        final_obs = robot_observation_processor(robot.get_observation())
        for prefix in active_prefixes:
            pos_keys = [f"{prefix}.ee.{a}" for a in ("x", "y", "z")]
            if not all(k in final_obs and k in target_ee for k in pos_keys):
                continue
            err = float(np.linalg.norm([final_obs[k] - target_ee[k] for k in pos_keys]))
            logging.info("[%s] post-reset pos error: %.2f mm", prefix, err * 1000)
            if cfg.position_tolerance > 0 and err > cfg.position_tolerance:
                raise RuntimeError(
                    f"{prefix} arm reset failed: residual {err * 1000:.1f} mm > "
                    f"tolerance {cfg.position_tolerance * 1000:.1f} mm"
                )

        log_say("Reset finished", cfg.play_sounds)
    finally:
        robot.disconnect()


def main():
    register_third_party_plugins()
    reset()


if __name__ == "__main__":
    main()
