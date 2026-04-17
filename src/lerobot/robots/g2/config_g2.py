from dataclasses import dataclass, field
from typing import Any, Dict

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


def g2_cameras_config() -> dict[str, CameraConfig]:
    """Default camera configuration for G2 robot."""
    return {
        "head_color": OpenCVCameraConfig(
            index_or_path="",  # Will be set by agibot_gdk
            fps=30,
            width=640,
            height=400,
        ),
        "hand_left": OpenCVCameraConfig(
            index_or_path="",  # Will be set by agibot_gdk
            fps=50,
            width=1280,
            height=1056,
        ),
        "hand_right": OpenCVCameraConfig(
            index_or_path="",  # Will be set by agibot_gdk
            fps=50,
            width=1280,
            height=1056,
        ),
    }


@RobotConfig.register_subclass("g2")
@dataclass
class G2RobotConfig(RobotConfig):
    # arm config
    use_left_arm: bool = field(default=True)
    use_right_arm: bool = field(default=True)
    use_gripper: bool = field(default=True)
    gripper_config: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "left": {
                "type": "g2",
                "open_position": 1.0,
                "close_position": 0.0,
                "device": "/dev/ttyUSB0",
                "baud": 115200,
                "slave_id": 1,
            },
            "right": {
                "type": "g2",
                "open_position": 1.0,
                "close_position": 0.0,
                "device": "/dev/ttyUSB0",
                "baud": 115200,
                "slave_id": 1,
            },
        }
    )

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(default_factory=g2_cameras_config)

    # Target joint positions (radians) for homing, same ordering as observation.state joint_* block:
    # dual: joint_1..7 left arm, joint_8..14 right arm, joint_15 left grip, joint_16 right grip.
    # single: joint_1..7 arm, joint_8 gripper.
    # Empty tuple: use all zeros. Physical motion requires a future GDK joint API; see G2Robot.reset().
    default_positions: tuple[float, ...] = field(default_factory=tuple)

    # Timestep (s) when interpolating toward default_positions if joint motion API is added.
    reset_control_dt: float = 1.0 / 30.0
    
    @property
    def dual_arm(self) -> bool:
        """Whether dual-arm mode is enabled (both left and right arms are used)."""
        return self.use_left_arm and self.use_right_arm
    
    @property
    def single_arm_prefix(self) -> str:
        """Get the prefix for single arm mode based on configuration."""
        return "l" if self.use_left_arm else "r"
