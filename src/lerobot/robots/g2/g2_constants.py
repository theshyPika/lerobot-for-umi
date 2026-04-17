# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Constants for G2 robot integration."""

# Minimum absolute change in normalized gripper command [0,1] to send a new command.
GRIPPER_COMMAND_MIN_DELTA = 0.01

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
