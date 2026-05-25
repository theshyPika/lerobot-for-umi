# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Constants for G2 robot integration."""

# Minimum absolute change in normalized gripper command [0,1] to send a new command.
GRIPPER_COMMAND_MIN_DELTA = 0.01

# Hidden observation key used to pass the full-body joint state through the
# processor pipeline without exposing it as a dataset feature.
FULL_JOINT_POSITIONS_OBS_KEY = "_g2.full_joint_positions_deg_by_name"

# SDK joint names in control order (radians from GDK).
LEFT_ARM_JOINT_NAMES = [
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    # "l_gripper_joint"
]

RIGHT_ARM_JOINT_NAMES = [
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
    # "r_gripper_joint"
]

RAD_TO_DEG = 180.0 / 3.141592653589793

# Mapping from LeRobot action keys (degrees) to GDK SDK joint names (radians at the SDK boundary).
ARM_JOINT_MAPPING = {
    "l.joint1.pos": "idx21_arm_l_joint1",
    "l.joint2.pos": "idx22_arm_l_joint2",
    "l.joint3.pos": "idx23_arm_l_joint3",
    "l.joint4.pos": "idx24_arm_l_joint4",
    "l.joint5.pos": "idx25_arm_l_joint5",
    "l.joint6.pos": "idx26_arm_l_joint6",
    "l.joint7.pos": "idx27_arm_l_joint7",
    "r.joint1.pos": "idx61_arm_r_joint1",
    "r.joint2.pos": "idx62_arm_r_joint2",
    "r.joint3.pos": "idx63_arm_r_joint3",
    "r.joint4.pos": "idx64_arm_r_joint4",
    "r.joint5.pos": "idx65_arm_r_joint5",
    "r.joint6.pos": "idx66_arm_r_joint6",
    "r.joint7.pos": "idx67_arm_r_joint7",
}

# G2 omnipicker gripper joint names (servo range: -0.785 open .. 0 closed, radians).
G2_GRIPPER_JOINT_NAMES = {
    "left": "idx31_gripper_l_inner_joint1",
    "right": "idx71_gripper_r_inner_joint1",
}

# Action/observation gripper convention for G2 (matches the legacy interface and
# fine-tuned policy outputs): normalized value in [0, 1] where 0 = fully open
# and 1 = fully closed. Mapped linearly to the omnipicker inner joint angle,
# whose servo range is [-0.785, 0] rad with -0.785 = open and 0 = closed.
G2_GRIPPER_OPEN_RAD = -0.785
G2_GRIPPER_CLOSED_RAD = 0.0


def g2_gripper_normalized_to_rad(value: float) -> float:
    """Map normalized gripper command in [0, 1] → omnipicker joint angle (rad)."""
    v = float(value)
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return G2_GRIPPER_OPEN_RAD + v * (G2_GRIPPER_CLOSED_RAD - G2_GRIPPER_OPEN_RAD)


def g2_gripper_rad_to_normalized(rad: float) -> float:
    """Map omnipicker joint angle (rad) → normalized gripper value in [0, 1]."""
    span = G2_GRIPPER_CLOSED_RAD - G2_GRIPPER_OPEN_RAD
    if span == 0.0:
        return 0.0
    v = (float(rad) - G2_GRIPPER_OPEN_RAD) / span
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
