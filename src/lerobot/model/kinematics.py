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

import logging
import numpy as np
from scipy import signal
from collections import deque

from lerobot.utils.rotation import Rotation


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
        max_iterations: int = 3,
        dt: float = 1e-2,
        eps: float = 1e-4,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
            max_iterations (int): Maximum IK iterations per solve
            dt (float): Time step for velocity-based IK (larger = faster convergence, less stable)
            eps (float): Convergence threshold for IK
        """
        try:
            import placo  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.max_iterations = max_iterations
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base
        self.solver.enable_joint_limits = True

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

        # Mask irrelevant joints
        for j_name in self.robot.joint_names():
            if j_name not in self.joint_names:
                self.solver.mask_dof(j_name)

        # Joint regularization task (soft constraint to stay near initial guess)
        # self.joint_reg_task = self.solver.add_joints_task()
        # self.joint_reg_task.configure("joint_reg", "soft", 1e-6)

        # Manipulability task (optional, helps avoid singularities)
        self.manipulability = self.solver.add_manipulability_task(target_frame_name, "both", 1.0)
        self.manipulability.configure("manipulability", "soft", 1e-2)

        # position task
        self.position_task = None
        # Get current position of the target frame
        self.position_task = self.solver.add_position_task(self.target_frame_name, np.zeros(3))
        self.position_task.configure(f"{self.target_frame_name}_position", "soft", 1.0)
        

        # Solver parameters
        self.solver.dt = dt
        self.solver.eps = eps
        self.solver.damping = 0.6
        

        # Cache for joint regularization dictionary to avoid repeated construction
        self._joint_reg_dict = {name: 0.0 for name in self.joint_names}

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
            self,
            current_joint_pos: np.ndarray,
            delta_ee: np.ndarray | None = None,
            target_pose: np.ndarray | None = None,
            position_weight: float = 1.0,
            orientation_weight: float = 1.0,
        ) -> np.ndarray:
            """
            Compute inverse kinematics from either delta or absolute target pose.

            Args:
                current_joint_pos: Current joint positions in degrees (initial guess).
                delta_ee: Delta end-effector pose as [dx, dy, dz, dwx, dwy, dwz] in world frame.
                target_pose: Absolute target pose as 4x4 transformation matrix (used if delta_ee is None).
                position_weight: Weight for position constraint in IK.
                orientation_weight: Weight for orientation constraint in IK.

            Returns:
                Joint positions in degrees that achieve the desired end-effector pose.
            """
            # Validate inputs
            if delta_ee is None and target_pose is None:
                raise ValueError("Either delta_ee or target_pose must be provided.")
            if delta_ee is not None and target_pose is not None:
                raise ValueError("Only one of delta_ee or target_pose should be provided.")

            # Set initial guess (critical for fast convergence)
            current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
            for i, joint_name in enumerate(self.joint_names):
                self.robot.set_joint(joint_name, current_joint_rad[i])
            self.robot.update_kinematics()

            # Determine desired_ee_pose
            if delta_ee is not None:
                current_ee_pose = self.robot.get_T_world_frame(self.target_frame_name)
                current_pos = current_ee_pose[:3, 3]
                current_rot = Rotation.from_matrix(current_ee_pose[:3, :3])

                target_pos = current_pos + delta_ee[:3]
                delta_rot = Rotation.from_rotvec(delta_ee[3:])
                target_rot = delta_rot * current_rot

                desired_ee_pose = np.eye(4, dtype=float)
                desired_ee_pose[:3, :3] = target_rot.as_matrix()
                desired_ee_pose[:3, 3] = target_pos

                # Inline readable logging
                # if logging.getLogger().isEnabledFor(logging.DEBUG):
                #     def _format_pose(mat):
                #         x, y, z = mat[:3, 3]
                #         r = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
                #         return f"({x:.4f}, {y:.4f}, {z:.4f}, {r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})"
                #     logging.debug(f"FK current: {_format_pose(current_ee_pose)}")
                #     logging.debug(f"Delta: {delta_ee}")
                #     logging.debug(f"Target:  {_format_pose(desired_ee_pose)}")
            else:
                desired_ee_pose = target_pose

            # Configure IK tasks
            self.tip_frame.T_world_frame = desired_ee_pose
            self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

            # Update position task target if it exists
            if self.position_task is not None:
                self.position_task.target_world = desired_ee_pose[:3, 3]
            # undo constraint
            # for i, joint_name in enumerate(self.joint_names):
            #     self._joint_reg_dict[joint_name] = current_joint_rad[i]
            # self.joint_reg_task.set_joints(self._joint_reg_dict)

            # Iterative solve
            converged = False
            for iteration in range(self.max_iterations):
                try:
                    self.solver.solve(True)
                    self.robot.update_kinematics()

                    pos_err = self.tip_frame.position().error_norm()
                    orient_err = np.linalg.norm(self.tip_frame.orientation().error())
                    if pos_err < 1e-2 and orient_err < 1e-1:
                        converged = True
                        logging.debug(f"IK converged after {iteration + 1} iterations")
                        break
                except RuntimeError as e:
                    logging.warning(f"IK iteration {iteration + 1} failed: {e}")
                    break

            if not converged:
                logging.debug(f"IK did not converge within {self.max_iterations} iterations")

            # Extract result
            joint_pos_rad = [self.robot.get_joint(name) for name in self.joint_names]
            joint_pos_deg = np.rad2deg(joint_pos_rad)

            if len(current_joint_pos) > len(self.joint_names):
                result = np.zeros_like(current_joint_pos)
                result[: len(self.joint_names)] = joint_pos_deg
                result[len(self.joint_names):] = current_joint_pos[len(self.joint_names):]
                return result
            return joint_pos_deg