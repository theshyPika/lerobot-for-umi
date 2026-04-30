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
        rotvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        result = {
            f"{prefix}.ee.x": float(pose[0, 3]),
            f"{prefix}.ee.y": float(pose[1, 3]),
            f"{prefix}.ee.z": float(pose[2, 3]),
            f"{prefix}.ee.wx": float(rotvec[0]),
            f"{prefix}.ee.wy": float(rotvec[1]),
            f"{prefix}.ee.wz": float(rotvec[2]),
        }
        gripper_key = f"{prefix}.ee.gripper.pos"
        if gripper_key in observation:
            result[gripper_key] = float(observation[gripper_key])
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_delta_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from an end-effector pose using inverse kinematics (IK).

    Uses a unified ``DualArmKinematics`` solver so that both arms are solved in a single
    placo iteration when running in dual-arm mode.  Single-arm mode is supported by simply
    requesting only one side. End-effector actions are interpreted in the kinematics solver's
    configured reference frame, e.g. ``arm_base_link`` for G2 recording.

    Attributes:
        kinematics: The unified dual-arm kinematic model.
        motor_names: A list of motor names for which to compute joint positions.
        left_q_curr: Internal state storing the last left joint positions.
        right_q_curr: Internal state storing the last right joint positions.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
        use_relative_actions: If True, actions are interpreted as delta end-effector poses.
    """

    motor_names: list[str]
    kinematics: DualArmKinematics | None = None
    left_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    right_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    use_relative_actions: bool = True

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

        # Prepare per-arm inputs
        left_delta, left_target, left_q, left_joints = self._prepare_arm(
            action, "l", observation
        )
        right_delta, right_target, right_q, right_joints = self._prepare_arm(
            action, "r", observation
        )

        left_requested = left_delta is not None or left_target is not None
        right_requested = right_delta is not None or right_target is not None

        if left_requested or right_requested:
            left_result, right_result = self.kinematics.inverse_kinematics_dual(
                left_current_joint_pos=left_q,
                left_delta_ee=left_delta,
                left_target_pose=left_target,
                right_current_joint_pos=right_q,
                right_delta_ee=right_delta,
                right_target_pose=right_target,
                current_joint_pos_by_name=full_joint_positions_deg_by_name,
            )

            if left_result is not None:
                self.left_q_curr = left_result
                for i, name in enumerate(left_joints):
                    action[f"{name}.pos"] = float(left_result[i])

            if right_result is not None:
                self.right_q_curr = right_result
                for i, name in enumerate(right_joints):
                    action[f"{name}.pos"] = float(right_result[i])

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
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str]]:
        """Extract delta/target values and current joint positions for one arm.

        Returns:
            (delta_ee, target_pose, q_curr, joint_motor_names).
            ``delta_ee`` is set when ``use_relative_actions`` is True,
            ``target_pose`` is set otherwise. Both are None when the arm has no action keys.
        """
        ee_keys = [f"{prefix}.ee.{a}" for a in ("x", "y", "z", "wx", "wy", "wz")]
        if not any(k in action for k in ee_keys):
            return None, None, None, []

        x = action.pop(f"{prefix}.ee.x", 0.0)
        y = action.pop(f"{prefix}.ee.y", 0.0)
        z = action.pop(f"{prefix}.ee.z", 0.0)
        wx = action.pop(f"{prefix}.ee.wx", 0.0)
        wy = action.pop(f"{prefix}.ee.wy", 0.0)
        wz = action.pop(f"{prefix}.ee.wz", 0.0)

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

        if self.use_relative_actions:
            pos_norm = np.linalg.norm([x, y, z])
            rot_norm = np.linalg.norm([wx, wy, wz])
            logging.info(f"[{prefix}] pos_norm={pos_norm:.4f}, rot_norm={rot_norm:.4f}")

            if pos_norm < 2e-3 and rot_norm < 1e-1:
                logging.info(f"All delta values are zero for {prefix} arm, skipping IK")
                return None, None, q_curr, joint_motor_names

            delta_ee = np.array([x, y, z, wx, wy, wz])
            return delta_ee, None, q_curr, joint_motor_names
        else:
            target_pose = np.array([x, y, z, wx, wy, wz])
            return None, target_pose, q_curr, joint_motor_names

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for prefix in ("l", "r"):
            for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper.pos"]:
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
