#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

r"""G2-specific entry point for ``lerobot-rollout``.

The generic ``lerobot-rollout`` script does not know about the G2's
end-effector action space.  This script wires the G2 forward/inverse
kinematics processors into the rollout context so that:

* observations expose EE poses (``l.ee.x`` … ``r.ee.wz``) computed by FK,
* policy actions in the same EE space are converted to joint targets via
  IK before ``robot.send_action``.

All four rollout strategies (``base``, ``sentry``, ``highlight``,
``dagger``) and both inference backends (``sync``, ``rtc``) work
unchanged on top of this — only the processor pipelines change.

Examples
--------

base_panic (autonomous rollout, no recording, no reset windows;
``space`` enters PAUSED, ``n`` resumes, ESC exits)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
        --fps=20 \
        --strategy.type=base_panic \
        --return_to_initial_position=false \
        --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/045000/pretrained_model \
        --policy.chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["gripper"]' \
        --inference.type=rtc \
        --inference.rtc.execution_horizon=10 \
        --inference.rtc.max_guidance_weight=5.0 \
        --robot.type=g2 \
        --robot.use_left_arm=true --robot.use_right_arm=true \
        --robot.use_gripper=true \
        --task="Pick up the paper cup." --duration=90 \
        --rename_map='{"observation.images.head_color": "observation.images.base_0_rgb", "observation.images.hand_left": "observation.images.left_wrist_0_rgb", "observation.images.hand_right": "observation.images.right_wrist_0_rgb"}' \
        --display_data=true

sentry_panic (eval mode: continuous recording with manual episode
control. ``space`` enters PAUSED, ``n`` resumes / skips reset window,
``→`` ends current episode, ``←`` discards buffer and re-records,
ESC exits. ``--dataset.push_to_hub=false`` keeps the run local)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
        --fps=20 \
        --strategy.type=sentry_panic \
        --strategy.reset_time_s=10 \
        --strategy.upload_every_n_episodes=2 \
        --return_to_initial_position=false \
        --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/045000/pretrained_model \
        --policy.chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["gripper"]' \
        --inference.type=rtc \
        --inference.rtc.execution_horizon=10 \
        --inference.rtc.max_guidance_weight=5.0 \
        --robot.type=g2 \
        --robot.use_left_arm=true --robot.use_right_arm=true \
        --robot.use_gripper=true \
        --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_45k_cup_1"     --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_45k_cup_1" \
        --dataset.single_task="Pick up the paper cup." \
        --dataset.streaming_encoding=true \
        --dataset.push_to_hub=false \
        --duration=0 \
        --rename_map='{"observation.images.head_color": "observation.images.base_0_rgb", "observation.images.hand_left": "observation.images.left_wrist_0_rgb", "observation.images.hand_right": "observation.images.right_wrist_0_rgb"}' \
        --display_data=true

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
        --fps=20 \
        --strategy.type=sentry_panic \
        --strategy.reset_time_s=10 \
        --strategy.upload_every_n_episodes=2 \
        --return_to_initial_position=false \
        --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/045000/pretrained_model \
        --policy.chunk_size=50 \
        --policy.n_action_steps=50 \
        --policy.use_relative_actions=true \
        --policy.relative_exclude_joints='["gripper"]' \
        --inference.type=rtc \
        --inference.rtc.execution_horizon=10 \
        --inference.rtc.max_guidance_weight=5.0 \
        --robot.type=g2 \
        --robot.use_left_arm=true --robot.use_right_arm=true \
        --robot.use_gripper=true \
        --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_45k_cup_2"     --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_45k_cup_2" \
        --dataset.single_task="Pick up the paper cup with your right hand." \
        --dataset.streaming_encoding=true \
        --dataset.push_to_hub=false \
        --duration=0 \
        --rename_map='{"observation.images.head_color": "observation.images.base_0_rgb", "observation.images.hand_left": "observation.images.left_wrist_0_rgb", "observation.images.hand_right": "observation.images.right_wrist_0_rgb"}' \
        --display_data=true

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=15 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
            --policy.chunk_size=40 \
            --policy.n_action_steps=40 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=10 \
            --inference.rtc.max_guidance_weight=8.0 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_cup_to_bin" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_cup_to_bin" \
            --dataset.single_task="把纸杯放进收纳盒里。" \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true


    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=20 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
            --policy.chunk_size=50 \
            --policy.n_action_steps=50 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=10 \
            --inference.rtc.max_guidance_weight=5.0 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_cup_to_coaster" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_cup_to_coaster" \
            --dataset.single_task="Pick up the banana and gently place it into the box." \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true


    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=20 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
            --policy.chunk_size=50 \
            --policy.n_action_steps=40 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=10 \
            --inference.rtc.max_guidance_weight=5.0 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_toolbox" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_toolbox" \
            --dataset.single_task="Open the lid of the black toolbox." \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=10 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --interpolation_multiplier=3 \
            --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
            --policy.chunk_size=40 \
            --policy.n_action_steps=40 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=10 \
            --inference.rtc.max_guidance_weight=5.0 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_g3-3t" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_g3-3t" \
            --dataset.single_task="将游标卡尺、千分卡尺放到钢管旁" \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true

   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=6 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --interpolation_multiplier=4 \
            --policy.path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
            --policy.chunk_size=50 \
            --policy.n_action_steps=50 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=10 \
            --inference.rtc.max_guidance_weight=15 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_group4_1debug12" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_g2_dual_arm_g1_7_relstats_clean_38k_group4_1debug12" \
            --dataset.single_task="将多个轴承同轴堆叠成塔形" \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true \
            > src/lerobot/exp/stop_and_move_large13.log 2>&1 | tail -f src/lerobot/exp/stop_and_move_large13.log 


    LEROBOT_DIAG_RTC_ROTVEC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 scripts/run-quiet-gdk.sh lerobot-rollout-g2 \
            --fps=6 \
            --strategy.type=sentry_panic \
            --strategy.reset_time_s=10 \
            --strategy.upload_every_n_episodes=2 \
            --return_to_initial_position=false \
            --interpolation_multiplier=1 \
            --policy.path=/home/ck/models/finetune_models/pi05_clean_nostill_ep_fr/checkpoints/045000/pretrained_model \
            --policy.chunk_size=50 \
            --policy.n_action_steps=50 \
            --policy.use_relative_actions=true \
            --policy.relative_exclude_joints='["gripper"]' \
            --inference.type=rtc \
            --inference.rtc.execution_horizon=4 \
            --inference.rtc.max_guidance_weight=3 \
            --ik_orientation_weight=0.03 \
            --ik_regularization_weight=0.003 \
            --robot.type=g2 \
            --robot.use_left_arm=true --robot.use_right_arm=true \
            --robot.use_gripper=true \
            --dataset.repo_id="luck4ck/rollout_pi05_clean_nostill_ep_fr_45k_group4_1debug8" \
            --dataset.root="/home/ck/.cache/huggingface/rollout_pi05_clean_nostill_ep_fr_45k_group4_1debug8" \
            --dataset.single_task="将多个轴承同轴堆叠成塔形" \
            --dataset.streaming_encoding=true \
            --dataset.push_to_hub=false \
            --duration=0 \
            --rename_map="{\"observation.images.head_color\": \"observation.images.base_0_rgb\", \"observation.images.hand_left\": \"observation.images.left_wrist_0_rgb\", \"observation.images.hand_right\": \"observation.images.right_wrist_0_rgb\"}" \
            --display_data=true \
            > src/lerobot/exp/stop_and_move_clean_nostill_ep_fr_45k_group4_1debug8.log 2>&1 | tail -f src/lerobot/exp/stop_and_move_clean_nostill_ep_fr_45k_group4_1debug8.log

        
""" 

 

import logging
from dataclasses import dataclass, field

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.model.kinematics import DualArmKinematics
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    g2,
    hope_jr,
    koch_follower,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.g2.g2_constants import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
from lerobot.robots.g2.robot_kinematic_processor import (
    ForwardKinematicsJointsToEEObservationG2,
    InverseKinematicsEEToJoints,
)
from lerobot.rollout import RolloutConfig, build_rollout_context, create_strategy
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    omx_leader,
    openarm_leader,
    openarm_mini,
    pico,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfigG2(RolloutConfig):
    """``RolloutConfig`` extended with G2 FK/IK knobs.

    Field names mirror those in ``lerobot_record_g2.RecordConfig`` so the
    same values used at recording time can be reused for rollout.
    """

    # URDF and end-effector frames consumed by ``DualArmKinematics``.
    g2_urdf_path: str = "src/lerobot/robots/g2/G2/genie_robot/urdf/G2_t2_crs/model.urdf"
    left_ee_target_frame: str = "arm_l_end_link"
    right_ee_target_frame: str = "arm_r_end_link"
    # Reference frame for EE poses. ``arm_base_link`` matches the dataset/teleop
    # convention; set to ``None`` (or ``"world"``) for absolute base/world poses.
    ee_reference_frame: str | None = "arm_base_link"
    # Prefer placo's native relative frame task when ``ee_reference_frame`` is set.
    use_relative_frame_task: bool = True
    # Local-z TCP offset (m) applied to each arm's end link.
    # ee_tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.007])
    ee_tcp_offset: list[float] = field(default_factory=lambda: [0.0, -0.007, 0.0])
    # Skip IK when the absolute EE target is already close to the current EE
    # observation. Prevents near-hold commands from being re-solved into a
    # different redundant joint posture.
    ee_hold_position_threshold_m: float = 0.005
    ee_hold_rotation_threshold_rad: float = 0.005
    # IK frame task weights. With null-space stabilisation handled by the velocity
    # regularization task (see ``ik_regularization_weight``), orientation can be
    # tracked accurately without destabilising position. ``0.2`` keeps the wrist
    # orientation error well under 1 deg in the reachable interior while position
    # stays dominant (see docs/source/g2_dualarm_ik_fix.md for the sweep data).
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.2
    # Velocity (Tikhonov / DLS) regularization weight. This is the null-space
    # stabiliser that prevents the end-effector from being flung off-target near
    # reach/wrist singularities. Required: with it at 0 the redundant arm is
    # unstable in the extended-reach region. Raise toward 3e-3 for smoother joints
    # at a small position-accuracy cost.
    ik_regularization_weight: float = 1e-3
    # Manipulability task weight. 0 = OFF (recommended). Any positive value biases
    # the solution away from the commanded EE target and degrades accuracy.
    ik_manipulability_weight: float = 0.0
    # placo solver iteration budget and integration step per IK call.
    ik_max_iterations: int = 10
    ik_dt: float = 5e-2


def _build_g2_processors(cfg: RolloutConfigG2, motor_names: list[str]):
    """Return ``(teleop_action, robot_action, robot_observation)`` processors for G2.

    The teleop pipeline keeps the default identity step (rollout has no
    teleop action stream unless ``--strategy.type=dagger``; even then, the
    teleop side is the policy fallback and needs no IK).  The observation
    pipeline runs FK to publish EE keys, and the action pipeline runs IK
    to convert policy EE commands into joint targets before ``send_action``.
    """
    use_left = bool(getattr(cfg.robot, "use_left_arm", False))
    use_right = bool(getattr(cfg.robot, "use_right_arm", False))
    if not (use_left or use_right):
        raise ValueError(
            "G2 rollout requires at least one arm enabled. Set --robot.use_left_arm=true "
            "and/or --robot.use_right_arm=true."
        )

    kinematics_kwargs = {
        "urdf_path": cfg.g2_urdf_path,
        "left_frame_name": cfg.left_ee_target_frame if use_left else None,
        "left_joint_names": LEFT_ARM_JOINT_NAMES if use_left else None,
        "right_frame_name": cfg.right_ee_target_frame if use_right else None,
        "right_joint_names": RIGHT_ARM_JOINT_NAMES if use_right else None,
        "reference_frame_name": cfg.ee_reference_frame,
        "use_relative_frame_task": cfg.use_relative_frame_task,
        "left_tcp_offset": cfg.ee_tcp_offset,
        "right_tcp_offset": cfg.ee_tcp_offset,
        "max_iterations": cfg.ik_max_iterations,
        "dt": cfg.ik_dt,
        "regularization_weight": cfg.ik_regularization_weight,
        "manipulability_weight": cfg.ik_manipulability_weight,
    }
    fk_solver = DualArmKinematics(**kinematics_kwargs)
    ik_solver = DualArmKinematics(**kinematics_kwargs)

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[ForwardKinematicsJointsToEEObservationG2(kinematics=fk_solver)],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    robot_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            InverseKinematicsEEToJoints(
                motor_names=motor_names,
                kinematics=ik_solver,
                initial_guess_current_joints=True,
                # Policy outputs absolute EE targets in our G2 datasets.
                use_relative_actions=False,
                ee_hold_position_threshold_m=cfg.ee_hold_position_threshold_m,
                ee_hold_rotation_threshold_rad=cfg.ee_hold_rotation_threshold_rad,
                ik_position_weight=cfg.ik_position_weight,
                ik_orientation_weight=cfg.ik_orientation_weight,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    teleop_action_processor, _, _ = make_default_processors()
    return teleop_action_processor, robot_action_processor, robot_observation_processor


def _resolve_motor_names(robot_cfg) -> list[str]:
    """Compute the motor-name list the IK processor needs to emit ``*.pos`` keys.

    Mirrors the logic inside ``G2Robot.__init__`` so we can build processors
    *before* constructing the robot in ``build_rollout_context`` (the robot
    instance is created inside that helper).
    """
    use_left = bool(getattr(robot_cfg, "use_left_arm", False))
    use_right = bool(getattr(robot_cfg, "use_right_arm", False))
    motors: list[str] = []
    if use_left and use_right:
        motors.extend(f"l.joint{i}" for i in range(1, 8))
        motors.append("l.gripper")
        motors.extend(f"r.joint{i}" for i in range(1, 8))
        motors.append("r.gripper")
    elif use_left:
        motors.extend(f"l.joint{i}" for i in range(1, 8))
        motors.append("l.gripper")
    elif use_right:
        motors.extend(f"r.joint{i}" for i in range(1, 8))
        motors.append("r.gripper")
    return motors


@parser.wrap()
def rollout_g2(cfg: RolloutConfigG2):
    """Main entry point for G2 policy deployment."""
    init_logging()

    if cfg.display_data:
        logger.info("Initializing Rerun visualization (ip=%s, port=%s)", cfg.display_ip, cfg.display_port)
        init_rerun(session_name="rollout_g2", ip=cfg.display_ip, port=cfg.display_port)

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event

    motor_names = _resolve_motor_names(cfg.robot)
    logger.info("G2 IK motor names (%d): %s", len(motor_names), motor_names)

    teleop_action_processor, robot_action_processor, robot_observation_processor = _build_g2_processors(
        cfg, motor_names
    )

    logger.info("Building rollout context with G2 FK/IK processors...")
    ctx = build_rollout_context(
        cfg,
        shutdown_event,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    strategy = create_strategy(cfg.strategy)
    logger.info("Rollout strategy: %s", cfg.strategy.type)
    logger.info(
        "Robot: %s | FPS: %.0f | Duration: %s",
        cfg.robot.type if cfg.robot else "?",
        cfg.fps,
        f"{cfg.duration}s" if cfg.duration > 0 else "infinite",
    )

    try:
        strategy.setup(ctx)
        logger.info("Rollout setup complete, starting rollout...")
        strategy.run(ctx)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        strategy.teardown(ctx)

    logger.info("Rollout finished")


def main():
    """CLI entry point for ``lerobot-rollout-g2``."""
    register_third_party_plugins()
    rollout_g2()


if __name__ == "__main__":
    main()
