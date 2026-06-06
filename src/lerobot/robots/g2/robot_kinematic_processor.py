import logging
from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import DualArmKinematics
from lerobot.processor import (
    ObservationProcessorStep,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.robots.g2.g2_constants import FULL_JOINT_POSITIONS_OBS_KEY
from lerobot.utils.rotation import Rotation


# TODO(g2-frame-refactor): 这一步存在是为了把 GDK 自带 frame 的 EE 覆写成
# arm_base_link 系，但同时引入了 "raw vs processed observation" 二元性，
# 触发了 strategies/core.py:_dispatch_action 和 base_panic 里的 frame mismatch bug
# （见 plan hashed-roaming-riddle.md）。
# 等 G2Robot.get_observation() 直接产出 arm_base_link 系 EE 后，本步骤即可删除。
@ProcessorStepRegistry.register("g2_forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservationG2(ObservationProcessorStep):
    """Replace SDK EE observations with FK poses from the configured kinematics reference frame."""

    kinematics: DualArmKinematics | None = None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        observation = dict(observation)
        if self.kinematics is None:
            return observation

        full_joint_positions_deg_by_name = observation.get(FULL_JOINT_POSITIONS_OBS_KEY)
        if not isinstance(full_joint_positions_deg_by_name, dict) or len(full_joint_positions_deg_by_name) == 0:
            return observation

        left_pose, right_pose = self.kinematics.forward_kinematics(full_joint_positions_deg_by_name)
        if self.kinematics.left_frame_name:
            observation.update(self._pose_to_observation(left_pose, "l", observation))
        if self.kinematics.right_frame_name:
            observation.update(self._pose_to_observation(right_pose, "r", observation))
        return observation

    def _pose_to_observation(
        self,
        pose: np.ndarray,
        prefix: str,
        observation: RobotObservation,
    ) -> dict[str, float]:
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()  # xyzw
        result = {
            f"{prefix}.ee.x": float(pose[0, 3]),
            f"{prefix}.ee.y": float(pose[1, 3]),
            f"{prefix}.ee.z": float(pose[2, 3]),
            f"{prefix}.ee.qx": float(quat[0]),
            f"{prefix}.ee.qy": float(quat[1]),
            f"{prefix}.ee.qz": float(quat[2]),
            f"{prefix}.ee.qw": float(quat[3]),
        }
        # EE gripper mirrors the joint-space gripper state (l.gripper.pos / r.gripper.pos);
        # GDK does not provide a separate EE gripper reading.
        joint_gripper_key = f"{prefix}.gripper.pos"
        if joint_gripper_key in observation:
            result[f"{prefix}.ee.gripper.pos"] = float(observation[joint_gripper_key])
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from an absolute end-effector pose via IK.

    Uses a unified ``DualArmKinematics`` solver so that both arms are solved in a single
    placo iteration when running in dual-arm mode.  Single-arm mode is supported by simply
    requesting only one side. Actions are absolute EE poses (position + xyzw quaternion)
    in the solver's configured reference frame, e.g. ``arm_base_link`` for G2 — the policy
    postprocessor has already restored absolute actions by this stage, regardless of
    whether the policy was trained on relative actions.

    Attributes:
        kinematics: The unified dual-arm kinematic model.
        motor_names: A list of motor names for which to compute joint positions.
        left_q_curr: Internal state storing the last left joint positions.
        right_q_curr: Internal state storing the last right joint positions.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
    """

    motor_names: list[str]
    kinematics: DualArmKinematics | None = None
    left_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    right_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    ee_hold_position_threshold_m: float = 0.001
    ee_hold_rotation_threshold_rad: float = 0.01
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 1e-2
    # If the solved EE position still misses the target by more than this (m),
    # the arm is treated as unreachable and holds its current joints instead of
    # sending the diverged IK solution (prevents the end-effector being flung
    # off-target near / beyond the reach boundary). 5 mm sits well above the
    # ~1.2 mm reachable error ceiling and well below the >16 mm unreachable band.
    ee_unreachable_hold_threshold_m: float = 2e-2

    def action(self, action: RobotAction) -> RobotAction:
        if self.kinematics is None:
            return action

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for computing robot kinematics")
        observation = observation.copy()
        logging.info(f"action in robot_kinematic_processor: {action}")
        obs_ee_pose = {k: v for k, v in observation.items() if "ee" in k}
        logging.info(f"observation ee pose: {obs_ee_pose}")

        full_joint_positions_deg_by_name = observation.get(FULL_JOINT_POSITIONS_OBS_KEY)
        if not isinstance(full_joint_positions_deg_by_name, dict) or len(full_joint_positions_deg_by_name) == 0:
            full_joint_positions_deg_by_name = None

        # Prepare per-arm absolute EE targets
        left_target, left_q, left_joints = self._prepare_arm(action, "l", observation)
        right_target, right_q, right_joints = self._prepare_arm(action, "r", observation)

        left_requested = left_target is not None
        right_requested = right_target is not None

        if left_requested or right_requested:
            left_result, right_result = self.kinematics.inverse_kinematics_dual(
                left_current_joint_pos=left_q,
                left_target_pose=left_target,
                right_current_joint_pos=right_q,
                right_target_pose=right_target,
                position_weight=self.ik_position_weight,
                orientation_weight=self.ik_orientation_weight,
                current_joint_pos_by_name=full_joint_positions_deg_by_name,
                hold_on_unreachable_pos_m=self.ee_unreachable_hold_threshold_m,
            )

            if left_result is not None:
                left_joint_delta = left_result - left_q
                logging.info(
                    "[l] IK joint_delta_deg max=%.4f values=%s",
                    float(np.max(np.abs(left_joint_delta))),
                    np.array2string(left_joint_delta, precision=4, suppress_small=True),
                )
                self.left_q_curr = left_result
                for i, name in enumerate(left_joints):
                    action[f"{name}.pos"] = float(left_result[i])

            if right_result is not None:
                right_joint_delta = right_result - right_q
                logging.info(
                    "[r] IK joint_delta_deg max=%.4f values=%s",
                    float(np.max(np.abs(right_joint_delta))),
                    np.array2string(right_joint_delta, precision=4, suppress_small=True),
                )
                self.right_q_curr = right_result
                for i, name in enumerate(right_joints):
                    action[f"{name}.pos"] = float(right_result[i])

        # For arms that were not solved (e.g., deadband-held), command current joint
        # positions so downstream consumers see a complete joint action dict.
        if left_joints and left_q is not None:
            for i, name in enumerate(left_joints):
                action.setdefault(f"{name}.pos", float(left_q[i]))
        if right_joints and right_q is not None:
            for i, name in enumerate(right_joints):
                action.setdefault(f"{name}.pos", float(right_q[i]))

        # Pop gripper actions after IK (they pass through)
        for prefix in ("l", "r"):
            gripper_key = f"{prefix}.ee.gripper.pos"
            if gripper_key in action:
                action[f"{prefix}.gripper.pos"] = action.pop(gripper_key)

        return action

    def _prepare_arm(
        self,
        action: RobotAction,
        prefix: str,
        observation: dict,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
        """Extract the absolute EE target pose and current joints for one arm.

        The action carries an absolute EE pose as position + xyzw quaternion. Returns
        ``(target_pose, q_curr, joint_motor_names)`` where ``target_pose`` is a 4x4
        homogeneous TCP matrix in the solver's reference frame, or ``None`` when the
        arm has no EE action keys (q_curr also None) or the target is within the hold
        deadband (q_curr set so the caller can command current joints).
        """
        ee_keys = [f"{prefix}.ee.{a}" for a in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        if not any(k in action for k in ee_keys):
            return None, None, []

        x = action.pop(f"{prefix}.ee.x", 0.0)
        y = action.pop(f"{prefix}.ee.y", 0.0)
        z = action.pop(f"{prefix}.ee.z", 0.0)
        qx = action.pop(f"{prefix}.ee.qx", 0.0)
        qy = action.pop(f"{prefix}.ee.qy", 0.0)
        qz = action.pop(f"{prefix}.ee.qz", 0.0)
        qw = action.pop(f"{prefix}.ee.qw", 1.0)
        target_rot = Rotation.from_quat([qx, qy, qz, qw])
        target_pose = np.eye(4, dtype=float)
        target_pose[:3, :3] = target_rot.as_matrix()
        target_pose[:3, 3] = [x, y, z]

        motor_names = [name for name in self.motor_names if name.startswith(f"{prefix}.")]
        joint_motor_names = [name for name in motor_names if "gripper" not in name]
        joint_keys = [f"{name}.pos" for name in joint_motor_names]
        missing_joint_keys = [key for key in joint_keys if key not in observation]
        if missing_joint_keys:
            raise ValueError(
                f"Missing joint positions for {prefix} arm in observation: {missing_joint_keys}"
            )
        q_raw = np.array([float(observation[key]) for key in joint_keys], dtype=float)

        q_curr = self.left_q_curr if prefix == "l" else self.right_q_curr
        if self.initial_guess_current_joints or q_curr is None:
            q_curr = q_raw

        # Hold deadband: skip IK when the absolute target is within thresholds of the
        # current observed EE pose, so a near-hold target is not re-solved into a
        # different redundant posture.
        obs_keys = [f"{prefix}.ee.{a}" for a in ("x", "y", "z", "qx", "qy", "qz", "qw")]
        if all(k in observation for k in obs_keys):
            obs_vals = [float(observation[k]) for k in obs_keys]
            obs_rot = Rotation.from_quat(obs_vals[3:7])
            pos_norm = float(np.linalg.norm(np.array([x, y, z]) - np.array(obs_vals[:3])))
            rot_norm = float(np.linalg.norm((target_rot * obs_rot.inv()).as_rotvec()))
            logging.info(
                f"[{prefix}] absolute EE delta pos_norm={pos_norm:.6f}, rot_norm={rot_norm:.6f}"
            )
            if (
                self.ee_hold_position_threshold_m > 0.0
                and self.ee_hold_rotation_threshold_rad > 0.0
                and pos_norm < self.ee_hold_position_threshold_m
                and rot_norm < self.ee_hold_rotation_threshold_rad
            ):
                logging.info(
                    f"[{prefix}] EE target within hold deadband "
                    f"({self.ee_hold_position_threshold_m:.4f} m, "
                    f"{self.ee_hold_rotation_threshold_rad:.4f} rad); skipping IK"
                )
                return None, q_curr, joint_motor_names

        return target_pose, q_curr, joint_motor_names

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for prefix in ("l", "r"):
            for feat in ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper.pos"]:
                features[PipelineFeatureType.ACTION].pop(f"{prefix}.ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.left_q_curr = None
        self.right_q_curr = None
