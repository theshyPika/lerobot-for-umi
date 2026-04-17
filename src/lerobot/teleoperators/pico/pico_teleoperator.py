import sys
from typing import Any, Dict, Tuple
import logging

import numpy as np

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_pico import PicoTeleoperatorConfig


# Constants for action feature dimensions
DOF_PER_ARM = 6  # 3 position + 3 axis-angle rotation
GRIPPER_PER_ARM = 1

# Action feature names
LEFT_POSITION_NAMES = ["l.ee.x", "l.ee.y", "l.ee.z"]
RIGHT_POSITION_NAMES = ["r.ee.x", "r.ee.y", "r.ee.z"]
LEFT_ROTATION_NAMES = ["l.ee.wx", "l.ee.wy", "l.ee.wz"]
RIGHT_ROTATION_NAMES = ["r.ee.wx", "r.ee.wy", "r.ee.wz"]
LEFT_GRIPPER_NAME = "l.ee.gripper.pos"
RIGHT_GRIPPER_NAME = "r.ee.gripper.pos"


class PicoTeleoperator(Teleoperator):
    """
    Teleop class to use PICO XR controller inputs for dual-arm control.
    Based on PICO official xrobotoolkit_sdk teleoperation examples.
    """

    config_class = PicoTeleoperatorConfig
    name = "pico"

    def __init__(self, config: PicoTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.pico = None
        self.logger = logging.getLogger(__name__)

    @property
    def action_features(self) -> dict:
        """
        Define action features for dual-arm control.
        
        Returns different action spaces based on configuration:
        - Single arm mode: 6 DOF + optional gripper
        - Dual arm mode: 12 DOF (left 6DOF + right 6DOF) + optional grippers
        """
        if self.config.dual_arm:
            return self._get_dual_arm_action_features()
        else:
            return self._get_single_arm_action_features()
    
    def _get_dual_arm_action_features(self) -> dict:
        """Get action features for dual-arm mode."""
        names = {}
        idx = 0
        
        # Left arm position
        for name in LEFT_POSITION_NAMES:
            names[name] = idx
            idx += 1
        
        # Left arm rotation (axis-angle)
        for name in LEFT_ROTATION_NAMES:
            names[name] = idx
            idx += 1
        
        # Right arm position
        for name in RIGHT_POSITION_NAMES:
            names[name] = idx
            idx += 1
        
        # Right arm rotation (axis-angle)
        for name in RIGHT_ROTATION_NAMES:
            names[name] = idx
            idx += 1
        
        # Grippers
        if self.config.use_gripper:
            names[LEFT_GRIPPER_NAME] = idx
            idx += 1
            names[RIGHT_GRIPPER_NAME] = idx
            idx += 1
        
        shape = (idx,)
        return {
            "dtype": "float32",
            "shape": shape,
            "names": names
        }
    
    def _get_single_arm_action_features(self) -> dict:
        """Get action features for single-arm mode."""
        prefix = self.config.single_arm_prefix
        names = {}
        idx = 0
        
        # Position
        for i, axis in enumerate(["x", "y", "z"]):
            names[f"{prefix}.ee.{axis}"] = idx
            idx += 1
        
        # Rotation (axis-angle)
        for i, axis in enumerate(["wx", "wy", "wz"]):
            names[f"{prefix}.ee.{axis}"] = idx
            idx += 1
        
        # Gripper
        if self.config.use_gripper:
            names[f"{prefix}.ee.gripper.pos"] = idx
            idx += 1
        
        shape = (idx,)
        return {
            "dtype": "float32",
            "shape": shape,
            "names": names
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        """Connect to PICO XR controller."""
        from .pico_utils import PicoController
        # 创建 PICO 控制器实例，传入配置参数
        # 使用新的参数：缩放因子、坐标系转换等
        self.pico = PicoController(
            deadzone=self.config.deadzone,
            scale_factor=self.config.scale_factor,
            use_coordinate_transform=self.config.use_coordinate_transform
        )
        self.pico.start()
        
        if calibrate:
            self.calibrate()

    def get_action(self) -> dict[str, Any]:
        """Get action from PICO XR controller."""
        if self.pico is None:
            self.logger.error("Cannot get action: PICO controller is not connected")
            return self._get_zero_action()
            
        # Update the controller to get fresh inputs
        self.pico.update()

        if self.config.dual_arm:
            return self._get_dual_arm_action()
        else:
            return self._get_single_arm_action()
    
    def _get_dual_arm_action(self) -> dict[str, Any]:
        """Get action for dual-arm mode."""
        # 确保pico不为None
        if self.pico is None:
            return self._get_zero_action()
            
        # 获取12DOF增量（左臂6DOF + 右臂6DOF）
        deltas = self.pico.get_deltas()
        
        # 解包增量
        (left_delta_x, left_delta_y, left_delta_z, left_delta_rot_x, left_delta_rot_y, left_delta_rot_z,
         right_delta_x, right_delta_y, right_delta_z, right_delta_rot_x, right_delta_rot_y, right_delta_rot_z) = deltas
        
        # 构建动作字典
        action_dict = {
            "l.ee.x": left_delta_x,
            "l.ee.y": left_delta_y,
            "l.ee.z": left_delta_z,
            "l.ee.wx": left_delta_rot_x,
            "l.ee.wy": left_delta_rot_y,
            "l.ee.wz": left_delta_rot_z,
            "r.ee.x": right_delta_x,
            "r.ee.y": right_delta_y,
            "r.ee.z": right_delta_z,
            "r.ee.wx": right_delta_rot_x,
            "r.ee.wy": right_delta_rot_y,
            "r.ee.wz": right_delta_rot_z,
        }
        
        # 添加夹爪控制
        if self.config.use_gripper:
            left_gripper_value, right_gripper_value = self.pico.get_gripper_values()
            action_dict["l.ee.gripper.pos"] = left_gripper_value
            action_dict["r.ee.gripper.pos"] = right_gripper_value
        
        return action_dict
    
    def _get_single_arm_action(self) -> dict[str, Any]:
        """Get action for single-arm mode."""
        # 确保pico不为None
        if self.pico is None:
            return self._get_zero_action()
            
        prefix = self.config.single_arm_prefix
        is_left_arm = self.config.use_left_arm
        
        # 获取增量
        deltas = self.pico.get_deltas()
        
        # 根据配置选择使用左臂还是右臂的增量
        if is_left_arm:
            # 使用左臂增量
            (delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z,
             _, _, _, _, _, _) = deltas
            grip_active = self.pico.left_grip
            gripper_value = self.pico.left_gripper_value if self.config.use_gripper else 0.0
        else:
            # 使用右臂增量
            (_, _, _, _, _, _,
             delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z) = deltas
            grip_active = self.pico.right_grip
            gripper_value = self.pico.right_gripper_value if self.config.use_gripper else 0.0
        
        # 如果抓握键未激活，返回零动作
        if not grip_active:
            action_dict = {
                f"{prefix}.ee.x": 0.0,
                f"{prefix}.ee.y": 0.0,
                f"{prefix}.ee.z": 0.0,
                f"{prefix}.ee.wx": 0.0,
                f"{prefix}.ee.wy": 0.0,
                f"{prefix}.ee.wz": 0.0,
            }
        else:
            action_dict = {
                f"{prefix}.ee.x": delta_x,
                f"{prefix}.ee.y": delta_y,
                f"{prefix}.ee.z": delta_z,
                f"{prefix}.ee.wx": delta_rot_x,
                f"{prefix}.ee.wy": delta_rot_y,
                f"{prefix}.ee.wz": delta_rot_z,
            }
        
        # 添加夹爪控制
        if self.config.use_gripper:
            action_dict[f"{prefix}.ee.gripper.pos"] = gripper_value
        
        return action_dict

    def _get_zero_action(self) -> dict[str, Any]:
        """Return zero action when controller is not connected."""
        if self.config.dual_arm:
            zero_action = {
                "l.ee.x": 0.0,
                "l.ee.y": 0.0,
                "l.ee.z": 0.0,
                "l.ee.wx": 0.0,
                "l.ee.wy": 0.0,
                "l.ee.wz": 0.0,
                "r.ee.x": 0.0,
                "r.ee.y": 0.0,
                "r.ee.z": 0.0,
                "r.ee.wx": 0.0,
                "r.ee.wy": 0.0,
                "r.ee.wz": 0.0,
            }
            if self.config.use_gripper:
                zero_action["l.ee.gripper.pos"] = 0.0  # 0.0 = 打开
                zero_action["r.ee.gripper.pos"] = 0.0
            return zero_action
        else:
            prefix = "l" if self.config.use_left_arm else "r"
            zero_action = {
                f"{prefix}.ee.x": 0.0,
                f"{prefix}.ee.y": 0.0,
                f"{prefix}.ee.z": 0.0,
                f"{prefix}.ee.wx": 0.0,
                f"{prefix}.ee.wy": 0.0,
                f"{prefix}.ee.wz": 0.0,
            }
            if self.config.use_gripper:
                zero_action[f"{prefix}.ee.gripper.pos"] = 0.0  # 0.0 = 打开
            return zero_action

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the PICO controller such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.pico is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update controller state to get fresh inputs
        self.pico.update()

        # Check if intervention is active
        is_intervention = self.pico.should_intervene()

        # Get episode end status
        episode_end_status = self.pico.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the PICO controller."""
        if self.pico is not None:
            self.pico.stop()
            self.pico = None

    @property
    def is_connected(self) -> bool:
        """Check if PICO controller is connected."""
        return self.pico is not None and self.pico.running

    def calibrate(self) -> None:
        """Calibrate the PICO controller."""
        pass

    @property
    def is_calibrated(self) -> bool:
        """Check if PICO controller is calibrated."""
        return True

    def configure(self) -> None:
        """Configure the PICO controller."""
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the PICO controller."""
        pass
