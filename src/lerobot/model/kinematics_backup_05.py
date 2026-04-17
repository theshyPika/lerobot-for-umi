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
from typing import Optional


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics.
    
    This is an optimized version incorporating improvements from XRoboToolkit-Python-sample.
    It maintains backward compatibility with the original API while providing better IK performance.
    """

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
        enable_manipulability: bool = True,  
        manipulability_weight: float = 1e-2,
        joint_reg_weight: float = 1e-2,  # Increased from 1e-6 for better regularization
        max_iterations: int = 1,  # Reduced from 100 for faster computation
        tolerance: float = 1e-2,
        dt: float = 0.04,
        add_position_task: bool = False,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
            enable_manipulability (bool): Whether to enable manipulability optimization (new parameter)
            manipulability_weight (float): Weight for manipulability task (new parameter)
            joint_reg_weight (float): Weight for joint regularization task (new parameter)
            max_iterations (int): Maximum iterations for IK convergence (new parameter)
            tolerance (float): Convergence tolerance (new parameter)
            dt (float): Time step for IK solver (new parameter)
            add_position_task (bool): If True, add a position task for the target frame (like motion tracker tasks)
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base
        # self.solver.dt = dt

        self.target_frame_name = target_frame_name
        self.enable_manipulability = enable_manipulability
        self.manipulability_weight = manipulability_weight
        self.joint_reg_weight = joint_reg_weight
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.add_position_task = add_position_task

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))
        
        # Initialize manipulability task if enabled
        self.manipulability_task = None
        if self.enable_manipulability:
            self.manipulability_task = self.solver.add_manipulability_task(
                self.target_frame_name, "both", 1.0
            )
            self.manipulability_task.configure(
                f"{self.target_frame_name}_manipulability", "soft", self.manipulability_weight
            )
        
        # Initialize joint regularization task
        self.joint_reg_task = self.solver.add_joints_task()
        
        # Initialize position task if requested
        self.position_task = None
        if self.add_position_task:
            # Get current position of the target frame
            self.robot.update_kinematics()
            T_world_frame = self.robot.get_T_world_frame(self.target_frame_name)
            target_position = T_world_frame[:3, 3]
            self.position_task = self.solver.add_position_task(self.target_frame_name, target_position)
            self.position_task.configure(f"{self.target_frame_name}_position", "soft", 1.0)
            logging.info(f"Added position task for {self.target_frame_name}")

        

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
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using optimized placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        logging.info(f"current_joint_pos: {current_joint_pos}")
        
        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Update joint regularization task with current joint positions
        self.joint_reg_task.set_joints({
            joint_name: current_joint_rad[i]
            for i, joint_name in enumerate(self.joint_names)
        })
        self.joint_reg_task.configure("joint_reg", "soft", self.joint_reg_weight)

        # Update position task target if it exists
        if self.position_task is not None:
            self.position_task.target_world = desired_ee_pose[:3, 3]
            logging.debug(f"Updated position task target to {desired_ee_pose[:3, 3]}")

        
        # 2. add soft contraints, keep current joint position
        current_joints = {}
        for joint in self.robot.joint_names():
            # 排除 6 自由度的浮动基座，只处理普通关节
            if joint not in ["universe", "root_joint"]:
                current_joints[joint] = 0.0
        if hasattr(self, 'joint_reg_task'):
            self.solver.remove_task(self.joint_reg_task)
        self.joint_reg_task = self.solver.add_joints_task()
        # current_joints.update({
        #     joint_name: current_joint_rad[i]
        #     for i, joint_name in enumerate(self.joint_names)
        # })
        self.joint_reg_task.set_joints(current_joints)
        self.joint_reg_task.configure("joint_reg", "soft", 1e-2)

        # # lock
        # current_base_pose = self.robot.get_T_world_frame("base_link")
        # base_task = self.solver.add_frame_task("base_link", current_base_pose)
        # base_task.configure("base_lock", "soft", 1.0)

        # Solve IK with convergence checking (improved from original fixed 5 iterations)
        converged = False
        for iteration in range(self.max_iterations):
            try:
                self.solver.solve(True)
                self.robot.update_kinematics()
                
                # Check convergence
                # current_joints = np.array([self.robot.get_joint(name) for name in self.joint_names])
                if self._check_convergence():
                    converged = True
                    logging.info(f"IK converged after {iteration + 1} iterations")
                    break
                    
                # self.prev_joint_pos = current_joints
                
            except RuntimeError as e:
                logging.warning(f"IK iteration {iteration + 1} failed: {e}")
                break
        
        if not converged:
            logging.warning(f"IK did not converge within {self.max_iterations} iterations")

        # Extract joint positions
        joint_pos_rad = [self.robot.get_joint(name) for name in self.joint_names]
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg
        

    # def _check_convergence(self,) -> bool:
    #     """Check if IK has converged based on end-effector pose error."""
    #     pos_task = self.tip_frame.position()
    #     orient_task = self.tip_frame.orientation()
        
    #     pos_err = pos_task.error_norm()
    #     orient_err = np.linalg.norm(orient_task.error())
        
    #     logging.debug(f"IK errors: pos={pos_err:.6f}, orient={orient_err:.6f}")
        
    #     return pos_err < self.tolerance and orient_err < self.tolerance

    def _check_convergence(self,) -> bool:
        # 获取当前实际末端位姿（通过正向运动学）
        self.robot.update_kinematics()
        T_current = self.robot.get_T_world_frame(self.target_frame_name)
        T_target = self.tip_frame.T_world_frame
        pos_err_real = np.linalg.norm(T_current[:3,3] - T_target[:3,3])
        # 方向误差可以用旋转矩阵差或轴角表示
        from scipy.spatial.transform import Rotation as R
        rot_err_real = R.from_matrix(T_current[:3,:3]).inv() * R.from_matrix(T_target[:3,:3])
        orient_err_real = np.linalg.norm(rot_err_real.as_rotvec())
        
        # 同时打印任务对象返回的误差
        pos_err_task = self.tip_frame.position().error_norm()
        orient_err_task = np.linalg.norm(self.tip_frame.orientation().error())
        
        logging.info(f"Real error: pos={pos_err_real:.6f}, orient={orient_err_real:.6f}")
        logging.info(f"Task error: pos={pos_err_task:.6f}, orient={orient_err_task:.6f}")
        
        return pos_err_task < self.tolerance and orient_err_task < self.tolerance
        