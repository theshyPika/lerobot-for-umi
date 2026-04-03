from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from ..config import RobotConfig


@RobotConfig.register_subclass("g2")
@dataclass
class G2RobotConfig(RobotConfig):
    # arm config
    use_left_arm: bool = field(default=True)
    use_right_arm: bool = field(default=True)
    use_gripper: bool = field(default=True)
    gripper_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "left": {
            "type": "g2",
            "open_position": 1.0,
            "close_position": 0.0,
            "device": "/dev/ttyUSB0",  # optional(device, baud, slave_id), not need for g2 original gripper
            "baud": 115200,
            "slave_id": 1
        },
        "right": {
            "type": "g2",
            "open_position": 1.0,
            "close_position": 0.0,
            "device": "/dev/ttyUSB0",  # optional(device, baud, slave_id), not need for g2 original gripper
            "baud": 115200,
            "slave_id": 1
        },
        # "right": {
        #     "type": "dh",
        #     "open_position": 930,
        #     "close_position": 0,
        #     "device": "/dev/ttyUSB0",
        #     "baud": 115200,
        #     "slave_id": 1
        # }
    })
    
    @property
    def dual_arm(self) -> bool:
        """Whether dual-arm mode is enabled (both left and right arms are used)."""
        return self.use_left_arm and self.use_right_arm
    
    @property
    def single_arm_prefix(self) -> str:
        """Get the prefix for single arm mode based on configuration."""
        return "left_" if self.use_left_arm else "right_"
