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
from collections.abc import Mapping
import time

import numpy as np

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
        self.all_joint_names = list(self.robot.joint_names())
        self._robot_joint_name_set = set(self.all_joint_names)
        self.joint_names = self.all_joint_names if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

        # Mask irrelevant joints
        for j_name in self.all_joint_names:
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

    def _set_robot_joint_positions(
        self,
        joint_pos_deg: np.ndarray | Mapping[str, float],
        joint_names: list[str] | None = None,
    ) -> None:
        """Apply joint positions to the placo robot in degrees."""
        if isinstance(joint_pos_deg, Mapping):
            for joint_name, joint_value_deg in joint_pos_deg.items():
                if joint_name in self._robot_joint_name_set:
                    self.robot.set_joint(joint_name, np.deg2rad(float(joint_value_deg)))
            return

        joint_names = self.joint_names if joint_names is None else joint_names
        joint_values_deg = np.asarray(joint_pos_deg, dtype=float)
        if len(joint_values_deg) < len(joint_names):
            raise ValueError(
                f"Expected at least {len(joint_names)} joint values, got {len(joint_values_deg)}."
            )

        joint_pos_rad = np.deg2rad(joint_values_deg[: len(joint_names)])
        for i, joint_name in enumerate(joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

    def forward_kinematics(self, joint_pos_deg: np.ndarray | Mapping[str, float]) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees, either as an ordered numpy array
                for ``self.joint_names`` or as a full-body mapping keyed by URDF joint name.

        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        self._set_robot_joint_positions(joint_pos_deg)

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
            current_joint_pos_by_name: Mapping[str, float] | None = None,
        ) -> np.ndarray:
            """
            Compute inverse kinematics from either delta or absolute target pose.

            Args:
                current_joint_pos: Current joint positions in degrees (initial guess).
                delta_ee: Delta end-effector pose as [dx, dy, dz, dwx, dwy, dwz] in world frame.
                target_pose: Absolute target pose as 4x4 transformation matrix (used if delta_ee is None).
                position_weight: Weight for position constraint in IK.
                orientation_weight: Weight for orientation constraint in IK.
                current_joint_pos_by_name: Optional full-body joint positions in degrees,
                    keyed by URDF joint name. These joints are loaded into the robot state
                    before IK, while only ``self.joint_names`` remain movable.

            Returns:
                Joint positions in degrees that achieve the desired end-effector pose.
            """
            # Validate inputs
            if delta_ee is None and target_pose is None:
                raise ValueError("Either delta_ee or target_pose must be provided.")
            if delta_ee is not None and target_pose is not None:
                raise ValueError("Only one of delta_ee or target_pose should be provided.")

            # Set the full-body state first so FK starts from the robot's real pose,
            # then override the controllable arm joints with the IK initial guess.
            if current_joint_pos_by_name is not None:
                self._set_robot_joint_positions(current_joint_pos_by_name)
            self._set_robot_joint_positions(current_joint_pos, joint_names=self.joint_names)
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
                def _format_pose(mat):
                    x, y, z = mat[:3, 3]
                    r = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
                    return f"({x:.4f}, {y:.4f}, {z:.4f}, {r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})"
                logging.info(f"FK current: {_format_pose(current_ee_pose)}")
                logging.info(f"Delta: {delta_ee}")
                logging.info(f"Target:  {_format_pose(desired_ee_pose)}")
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


class DualArmKinematics:
    """Dual-arm kinematics using a single placo solver for both arms.

    Supports single-arm (left-only, right-only) and dual-arm configurations.
    When only one arm is enabled, behavior is equivalent to ``RobotKinematics``
    for that arm.
    """

    def __init__(
        self,
        urdf_path: str,
        left_frame_name: str | None = None,
        left_joint_names: list[str] | None = None,
        right_frame_name: str | None = None,
        right_joint_names: list[str] | None = None,
        max_iterations: int = 3,
        dt: float = 1e-2,
        eps: float = 1e-6,
    ):
        try:
            import placo  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "placo is required for DualArmKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        if not (left_frame_name or right_frame_name):
            raise ValueError("At least one of left_frame_name or right_frame_name must be provided.")

        self.max_iterations = max_iterations
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits = True

        self.left_frame_name = left_frame_name
        self.right_frame_name = right_frame_name
        self.left_joint_names = list(left_joint_names) if left_joint_names is not None else []
        self.right_joint_names = list(right_joint_names) if right_joint_names is not None else []

        self.all_joint_names = list(self.robot.joint_names())
        self._robot_joint_name_set = set(self.all_joint_names)

        enabled_joints = set(self.left_joint_names + self.right_joint_names)
        for j_name in self.all_joint_names:
            if j_name not in enabled_joints:
                self.solver.mask_dof(j_name)

        self.left_tip = None
        self.left_manip = None
        self.left_position_task = None
        if self.left_frame_name:
            self.left_tip = self.solver.add_frame_task(self.left_frame_name, np.eye(4))
            self.left_manip = self.solver.add_manipulability_task(self.left_frame_name, "both", 1.0)
            self.left_manip.configure("manip_left", "soft", 1e-2)
            self.left_position_task = self.solver.add_position_task(self.left_frame_name, np.zeros(3))
            self.left_position_task.configure(f"{self.left_frame_name}_position", "soft", 1.0)

        self.right_tip = None
        self.right_manip = None
        self.right_position_task = None
        if self.right_frame_name:
            self.right_tip = self.solver.add_frame_task(self.right_frame_name, np.eye(4))
            self.right_manip = self.solver.add_manipulability_task(self.right_frame_name, "both", 1.0)
            self.right_manip.configure("manip_right", "soft", 1e-2)
            self.right_position_task = self.solver.add_position_task(self.right_frame_name, np.zeros(3))
            self.right_position_task.configure(f"{self.right_frame_name}_position", "soft", 1.0)

        self.solver.dt = dt
        self.solver.eps = eps
        self.solver.damping = 0.2

    def _set_robot_joint_positions(
        self,
        joint_pos_deg: np.ndarray | Mapping[str, float],
        joint_names: list[str] | None = None,
    ) -> None:
        """Apply joint positions to the placo robot in degrees."""
        if isinstance(joint_pos_deg, Mapping):
            for joint_name, joint_value_deg in joint_pos_deg.items():
                if joint_name in self._robot_joint_name_set:
                    self.robot.set_joint(joint_name, np.deg2rad(float(joint_value_deg)))
            return

        joint_names = self.left_joint_names + self.right_joint_names if joint_names is None else joint_names
        joint_values_deg = np.asarray(joint_pos_deg, dtype=float)
        if len(joint_values_deg) < len(joint_names):
            raise ValueError(
                f"Expected at least {len(joint_names)} joint values, got {len(joint_values_deg)}."
            )

        joint_pos_rad = np.deg2rad(joint_values_deg[: len(joint_names)])
        for i, joint_name in enumerate(joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

    def _compute_target_pose(
        self,
        delta_ee: np.ndarray | None,
        target_pose: np.ndarray | None,
        frame_name: str,
    ) -> np.ndarray:
        """Compute desired target pose from delta or absolute pose."""
        if delta_ee is not None:
            current_ee_pose = self.robot.get_T_world_frame(frame_name)
            current_pos = current_ee_pose[:3, 3]
            current_rot = Rotation.from_matrix(current_ee_pose[:3, :3])

            target_pos = current_pos + delta_ee[:3]
            delta_rot = Rotation.from_rotvec(delta_ee[3:])
            target_rot = delta_rot * current_rot

            desired_ee_pose = np.eye(4, dtype=float)
            desired_ee_pose[:3, :3] = target_rot.as_matrix()
            desired_ee_pose[:3, 3] = target_pos
            
            # Inline readable logging #remove later
            def _format_pose(mat):
                x, y, z = mat[:3, 3]
                r = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
                return f"({x:.6f}, {y:.6f}, {z:.6f}, {r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f})"
            logging.info(f"FK current: {_format_pose(current_ee_pose)}")
            return desired_ee_pose

        # target_pose is an absolute pose as [x, y, z, wx, wy, wz]
        target_pos = target_pose[:3]
        target_rot = Rotation.from_rotvec(target_pose[3:])
        desired_ee_pose = np.eye(4, dtype=float)
        desired_ee_pose[:3, :3] = target_rot.as_matrix()
        desired_ee_pose[:3, 3] = target_pos
        return desired_ee_pose

    def forward_kinematics(
        self,
        joint_pos_deg: Mapping[str, float],
    ) -> tuple[np.ndarray]:
        """Compute forward kinematics for enabled end-effector frames.

        Only ``Mapping[str, float]`` is accepted because full-body joint names
        are required for an accurate pose (torso joints affect arm base pose).

        Args:
            joint_pos_deg: Full-body joint positions in degrees, keyed by URDF joint name.

        Returns:
            Dictionary with fixed keys ``"left"`` and ``"right"``.
            Each value is a 4x4 pose matrix; for an unenabled arm a zero matrix
            ``np.zeros((4, 4))`` is returned as a placeholder.
        """
        self._set_robot_joint_positions(joint_pos_deg)
        self.robot.update_kinematics()

        left_pose = np.zeros((4, 4), dtype=float)
        right_pose = np.zeros((4, 4), dtype=float)
        if self.left_frame_name:
            left_pose = self.robot.get_T_world_frame(self.left_frame_name)
        if self.right_frame_name:
            right_pose = self.robot.get_T_world_frame(self.right_frame_name)
        return (left_pose, right_pose)

    def inverse_kinematics_dual(
        self,
        left_current_joint_pos: np.ndarray | None = None,
        left_delta_ee: np.ndarray | None = None,
        left_target_pose: np.ndarray | None = None,
        right_current_joint_pos: np.ndarray | None = None,
        right_delta_ee: np.ndarray | None = None,
        right_target_pose: np.ndarray | None = None,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
        current_joint_pos_by_name: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Compute inverse kinematics for one or both arms using a single solver.

        Returns:
            Tuple of (left_joint_pos_deg, right_joint_pos_deg).  Each element is
            ``None`` if the corresponding arm was not requested.
        """
        t_ik_start = time.perf_counter()
        # Validate inputs
        if left_delta_ee is not None and left_target_pose is not None:
            raise ValueError("Only one of left_delta_ee or left_target_pose should be provided.")
        if right_delta_ee is not None and right_target_pose is not None:
            raise ValueError("Only one of right_delta_ee or right_target_pose should be provided.")

        left_requested = left_current_joint_pos is not None and (left_delta_ee is not None or left_target_pose is not None)
        right_requested = right_current_joint_pos is not None and (right_delta_ee is not None or right_target_pose is not None)

        if left_requested and not self.left_frame_name:
            raise ValueError("Left arm IK requested but left_frame_name was not configured.")
        if right_requested and not self.right_frame_name:
            raise ValueError("Right arm IK requested but right_frame_name was not configured.")

        # Set full-body state first, then override the controllable arm joints
        if current_joint_pos_by_name is not None:
            self._set_robot_joint_positions(current_joint_pos_by_name)
        if left_requested:
            self._set_robot_joint_positions(left_current_joint_pos, joint_names=self.left_joint_names)
        if right_requested:
            self._set_robot_joint_positions(right_current_joint_pos, joint_names=self.right_joint_names)
        self.robot.update_kinematics()

        # Compute target poses
        left_target = None
        if left_requested:
            left_target = self._compute_target_pose(left_delta_ee, left_target_pose, self.left_frame_name)
            self.left_tip.T_world_frame = left_target
            self.left_tip.configure(self.left_frame_name, "soft", position_weight, orientation_weight)
            if self.left_position_task is not None:
                self.left_position_task.target_world = left_target[:3, 3]

        right_target = None
        if right_requested:
            right_target = self._compute_target_pose(right_delta_ee, right_target_pose, self.right_frame_name)
            self.right_tip.T_world_frame = right_target
            self.right_tip.configure(self.right_frame_name, "soft", position_weight, orientation_weight)
            if self.right_position_task is not None:
                self.right_position_task.target_world = right_target[:3, 3]

        # Freeze inactive arms by pinning them at their current FK pose.
        # This prevents stale frame_task targets from conflicting with the
        # updated joint state when only one arm is being controlled.
        if not left_requested and self.left_frame_name:
            current_left_pose = self.robot.get_T_world_frame(self.left_frame_name)
            self.left_tip.T_world_frame = current_left_pose
            if self.left_position_task is not None:
                self.left_position_task.target_world = current_left_pose[:3, 3]

        if not right_requested and self.right_frame_name:
            current_right_pose = self.robot.get_T_world_frame(self.right_frame_name)
            self.right_tip.T_world_frame = current_right_pose
            if self.right_position_task is not None:
                self.right_position_task.target_world = current_right_pose[:3, 3]

        # Iterative solve
        converged = False
        for iteration in range(self.max_iterations):
            try:
                self.solver.solve(True)
                self.robot.update_kinematics()

                left_ok = not left_requested
                right_ok = not right_requested

                if left_requested:
                    lp = self.left_tip.position().error_norm()
                    lo = np.linalg.norm(self.left_tip.orientation().error())
                    left_ok = lp < 1e-2 and lo < 1e-1
                if right_requested:
                    rp = self.right_tip.position().error_norm()
                    ro = np.linalg.norm(self.right_tip.orientation().error())
                    right_ok = rp < 1e-2 and ro < 1e-1

                if left_ok and right_ok:
                    converged = True
                    logging.debug(f"Dual IK converged after {iteration + 1} iterations")
                    break
            except RuntimeError as e:
                logging.warning(f"Dual IK iteration {iteration + 1} failed: {e}")
                break

        if not converged:
            logging.debug(f"Dual IK did not converge within {self.max_iterations} iterations")

        left_result = None
        if left_requested:
            left_rad = [self.robot.get_joint(name) for name in self.left_joint_names]
            left_result = np.rad2deg(left_rad)

        right_result = None
        if right_requested:
            right_rad = [self.robot.get_joint(name) for name in self.right_joint_names]
            right_result = np.rad2deg(right_rad)
        t_ik_end = time.perf_counter()
        logging.info(f"IK took {(t_ik_end - t_ik_start)*1000:.2f} ms")
        return left_result, right_result
