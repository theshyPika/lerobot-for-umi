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
import pinocchio as pin


class RobotKinematics:
    """Robot kinematics using pinocchio library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize pinocchio-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "pinocchio is required for RobotKinematics. "
                "Please install pinocchio: pip install pin"
            ) from e
        
        # Load model from URDF
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        # Get frame ID for target frame
        self.target_frame_id = self.model.getFrameId(target_frame_name)
        if self.target_frame_id == len(self.model.frames):
            raise ValueError(f"Frame '{target_frame_name}' not found in URDF")
        
        self.target_frame_name = target_frame_name
        
        # Set joint names
        if joint_names is None:
            # Use all joint names except universe (fixed base)
            self.joint_names = [name for name in self.model.names[1:]]  # Skip universe
        else:
            self.joint_names = joint_names
        
        # Get joint indices for the specified joints
        self.joint_indices = []
        self.joint_q_indices = []
        for joint_name in self.joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id == 0:  # 0 is universe joint
                raise ValueError(f"Joint '{joint_name}' not found in URDF")
            self.joint_indices.append(joint_id)
            self.joint_q_indices.append(self.model.joints[joint_id].idx_q)
        
        # IK parameters (matching planner_tool.hpp)
        self.max_iterations = 10000
        self.tolerance = 1e-3  # More strict tolerance
        self.dt = 1e-2  # Integration step size
        self.damp = 1e-6  # Damping factor
        
        # Joint limits
        self.lower_limits = self.model.lowerPositionLimit
        self.upper_limits = self.model.upperPositionLimit

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
        
        # Create full configuration vector (neutral position)
        q = pin.neutral(self.model)
        
        # Set joint positions
        for i, q_idx in enumerate(self.joint_q_indices):
            q[q_idx] = joint_pos_rad[i]
        
        # Compute forward kinematics
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        # Get transformation matrix for target frame
        T = self.data.oMf[self.target_frame_id]
        
        # Convert to numpy array
        return T.homogeneous

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 1.0,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using pinocchio solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK (not used in current implementation)
            orientation_weight: Weight for orientation constraint in IK (not used in current implementation)

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """
        logging.info(f"current_joint_pos: {current_joint_pos}")
        
        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        
        # Create full configuration vector (neutral position)
        q = pin.neutral(self.model)
        
        # Set initial joint positions
        for i, q_idx in enumerate(self.joint_q_indices):
            q[q_idx] = current_joint_rad[i]
        
        # Convert desired pose to pinocchio SE3
        # desired_ee_pose is 4x4 transformation matrix
        R = desired_ee_pose[:3, :3]
        p = desired_ee_pose[:3, 3]
        oMdes = pin.SE3(R, p)
        
        # Initialize Jacobian matrix 6 x nv
        nv = self.model.nv
        J = np.zeros((6, nv))
        
        # Initialize result vector for group joints
        joint_num = len(self.joint_indices)
        q_result = np.zeros(joint_num)
        
        # IK loop (matching planner_tool.hpp implementation)
        success = False
        err = 0.0
        
        for iteration in range(self.max_iterations):
            # Compute forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # Compute error between current and desired pose
            iMd = self.data.oMf[self.target_frame_id].actInv(oMdes)
            err_vec = pin.log6(iMd).vector  # 6D error in joint frame
            err = np.linalg.norm(err_vec)
            
            # Check convergence
            if err < self.tolerance:
                success = True
                logging.info(f"IK converged after {iteration + 1} iterations, error: {err:.6f}")
                break
            
            # Compute Jacobian for the frame
            J = pin.computeFrameJacobian(self.model, self.data, q, self.target_frame_id, pin.LOCAL)
            
            # Compute Jlog (Jacobian of log map)
            Jlog = pin.Jlog6(iMd.inverse())
            J = -Jlog @ J
            
            # Extract Jacobian for relevant joints
            J_group = np.zeros((6, joint_num))
            for i, q_idx in enumerate(self.joint_q_indices):
                # Copy column from J to J_group
                J_group[:, i] = J[:, q_idx]
            
            # Damped least squares: v = -J^T (J J^T + λI)^(-1) e
            JJt = J_group @ J_group.transpose()
            # Add damping to diagonal
            for i in range(6):
                JJt[i, i] += self.damp
            v_group = -J_group.transpose() @ np.linalg.solve(JJt, err_vec)
            
            # Update full velocity vector
            v = np.zeros(nv)
            for i, q_idx in enumerate(self.joint_q_indices):
                v[q_idx] = v_group[i]
            
            # Integrate: q = q ⊕ (v * dt)
            q = pin.integrate(self.model, q, v * self.dt)
            
            # Apply joint limits
            for i, q_idx in enumerate(self.joint_q_indices):
                lo = self.lower_limits[q_idx]
                hi = self.upper_limits[q_idx]
                if np.isfinite(lo):
                    q[q_idx] = max(q[q_idx], lo)
                if np.isfinite(hi):
                    q[q_idx] = min(q[q_idx], hi)
        
        if not success:
            logging.warning(f"IK did not converge within {self.max_iterations} iterations, final error: {err:.6f}")
        
        # Extract result for group joints
        for i, q_idx in enumerate(self.joint_q_indices):
            q_result[i] = q[q_idx]
        
        # Convert back to degrees
        joint_pos_deg = np.rad2deg(q_result)
        
        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg