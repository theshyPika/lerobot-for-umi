import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    TransitionKey,
)
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
    """

    motor_names: list[str]
    left_kinematics: RobotKinematics = None
    right_kinematics: RobotKinematics = None
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    left_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    right_q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True
    
    def action(self, action: RobotAction) -> RobotAction:
        if self.left_kinematics:
            left_result = self._action(
                action=action,
                prefix="l",
                kinematics=self.left_kinematics,
                motor_names=[name for name in self.motor_names if "l" in name],
                q_curr_ref="left_q_curr"
            )
            action.update(left_result)

        if self.right_kinematics:
            right_result = self._action(
                action=action,
                prefix="r",
                kinematics=self.right_kinematics,
                motor_names=[name for name in self.motor_names if "r" in name],
                q_curr_ref="right_q_curr"
            )
            action.update(right_result)
            
        return action
    
    def _action(self, action: RobotAction, prefix: str, 
                    kinematics: RobotKinematics, motor_names: list[str],
                    q_curr_ref: str) -> dict[str, float]:
        logging.info(f"action in robot_kinematic_processor: {action}")
        x = action.pop(f"{prefix}.ee.x")
        y = action.pop(f"{prefix}.ee.y")
        z = action.pop(f"{prefix}.ee.z")
        wx = action.pop(f"{prefix}.ee.wx")
        wy = action.pop(f"{prefix}.ee.wy")
        wz = action.pop(f"{prefix}.ee.wz")
        gripper_pos = action.pop(f"{prefix}.ee.gripper.pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                f"Missing required end-effector pose components for {prefix} arm: "
                f"{prefix}.ee.x, {prefix}.ee.y, {prefix}.ee.z, {prefix}.ee.wx, "
                f"{prefix}.ee.wy, {prefix}.ee.wz, {prefix}.ee.gripper_pos must all be present"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        logging.info(f"observation : {observation}")
        q_raw = np.array(
            [float(v) for k, v in observation.items() 
             if isinstance(k, str) and k.startswith(f"{prefix}.") and k.endswith(".pos")],
            dtype=float,
        )
        
        if len(q_raw) == 0:
            raise ValueError(f"No joint positions found for {prefix} arm in observation")

        q_curr = getattr(self, q_curr_ref)
        if self.initial_guess_current_joints:  
            q_curr = q_raw[:8] if prefix == 'l' or None in (self.left_kinematics, self.right_kinematics) else q_raw[8:]
        else:  
            if q_curr is None:
                q_curr = q_raw[:8] if prefix == 'l' or None in (self.left_kinematics, self.right_kinematics) else q_raw[8:]

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = kinematics.inverse_kinematics(q_curr, t_des)
        
        setattr(self, q_curr_ref, q_target)

        result = {}
        for i, name in enumerate(motor_names):
            if "gripper" not in name:
                result[f"{name}.pos"] = float(q_target[i])
            else:
                result[f"{prefix}.gripper.pos"] = float(gripper_pos)
        return result


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
