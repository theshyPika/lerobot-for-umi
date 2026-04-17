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


@ProcessorStepRegistry.register("inverse_kinematics_delta_ee_to_joints")
@dataclass
class InverseKinematicsDeltaEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a delta end-effector pose in world frame using inverse kinematics (IK).
    
    This processor handles delta_ee actions (incremental end-effector pose changes in world frame) by:
    1. Getting current end-effector pose from observation (in world frame)
    2. Applying delta rotation and translation in world frame to get target pose
    3. Computing inverse kinematics to get joint positions
    
    This is specifically designed for G2 robot which expects delta_ee actions in world frame.

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
        logging.info(f"action in robot_delta_kinematic_processor: {action}")
        
        # Get delta values from action (in world frame)
        delta_x = action.pop(f"{prefix}.ee.x", 0.0)
        delta_y = action.pop(f"{prefix}.ee.y", 0.0)
        delta_z = action.pop(f"{prefix}.ee.z", 0.0)
        delta_wx = action.pop(f"{prefix}.ee.wx", 0.0)
        delta_wy = action.pop(f"{prefix}.ee.wy", 0.0)
        delta_wz = action.pop(f"{prefix}.ee.wz", 0.0)
        gripper_pos = action.pop(f"{prefix}.ee.gripper.pos", 0.0)

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is required for computing robot kinematics")

        # logging.info(f"observation : {observation}")
        
        # Get current joint positions for IK initial guess
        q_raw = np.array(
            [float(v) for k, v in observation.items() 
             if isinstance(k, str) and k.startswith(f"{prefix}.") and k.endswith(".pos")],
            dtype=float,
        )
        
        if len(q_raw) == 0:
            raise ValueError(f"No joint positions found for {prefix} arm in observation")

        q_curr = getattr(self, q_curr_ref)
        if self.initial_guess_current_joints:  
            q_curr = q_raw[:7] if prefix == 'l' or None in (self.left_kinematics, self.right_kinematics) else q_raw[8:]
        else:  
            if q_curr is None:
                q_curr = q_raw[:7] if prefix == 'l' or None in (self.left_kinematics, self.right_kinematics) else q_raw[8:]

        # Check if all delta values are zero (including gripper)
        # If all delta values are zero, skip IK and use current joint positions
        # delta_values = [delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz]
        pos_norm = np.linalg.norm([delta_x, delta_y, delta_z,])
        rot_norm = np.linalg.norm([delta_wx, delta_wy, delta_wz])
        logging.info(f"rot_norm: {rot_norm}")
        all_delta_zero = pos_norm < 2e-3 and rot_norm < 1e-1 
        
        if all_delta_zero:
            logging.info(f"All delta values are zero for {prefix} arm, skipping IK and using current joint positions")
            q_target = q_curr
        else:
            # Create delta_ee array: [dx, dy, dz, dwx, dwy, dwz]
            delta_ee = np.array([delta_x, delta_y, delta_z, delta_wx, delta_wy, delta_wz])
                        
            # Compute inverse kinematics using placo forward kinematics for current pose
            # This will use placo's forward kinematics to get the current EE pose,
            # then apply the delta to compute target pose, and finally solve IK
            q_target = kinematics.inverse_kinematics(current_joint_pos=q_curr, delta_ee=delta_ee)
        
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