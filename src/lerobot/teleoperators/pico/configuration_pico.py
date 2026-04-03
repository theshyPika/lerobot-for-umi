
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

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("pico")
@dataclass
class PicoTeleoperatorConfig(TeleoperatorConfig):
    """
    Configuration for PICO XR controller teleoperation.
    
    Attributes:
        use_gripper: Whether to enable gripper control
        use_left_arm: Whether to use left arm for teleoperation
        use_right_arm: Whether to use right arm for teleoperation
        deadzone: Controller deadzone threshold (0.0 to 1.0)
        scale_factor: Motion scaling factor for position deltas
        use_coordinate_transform: Whether to apply coordinate transformation from headset to world frame
        R_headset_to_world: Rotation matrix for coordinate transformation
        controller_deadzone: Deadzone threshold for controller inputs (0.0 to 1.0)
    """
    
    # Teleoperation mode settings
    use_gripper: bool = field(default=True)
    use_left_arm: bool = field(default=True)
    use_right_arm: bool = field(default=True)
    
    # Controller sensitivity settings
    deadzone: float = field(default=0.1)
    scale_factor: float = field(default=1.0)
    
    # Coordinate transformation settings
    use_coordinate_transform: bool = field(default=True)
    R_headset_to_world: Tuple[Tuple[float, float, float], 
                              Tuple[float, float, float], 
                              Tuple[float, float, float]] = field(
        default=((0, 0, -1),
                 (-1, 0, 0),
                 (0, 1, 0))
    )
    
    # Controller input settings
    controller_deadzone: float = field(default=0.1)
    
    def __post_init__(self):
        """Initialize after dataclass fields are set."""
        super().__init__(id="pico", calibration_dir=None)
    
    @property
    def dual_arm(self) -> bool:
        """Whether dual-arm mode is enabled (both left and right arms are used)."""
        return self.use_left_arm and self.use_right_arm
    
    @property
    def single_arm_prefix(self) -> str:
        """Get the prefix for single arm mode based on configuration."""
        return "left_" if self.use_left_arm else "right_"
