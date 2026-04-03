import logging
import threading
import numpy as np
import meshcat.transformations as tf
from typing import Optional, Tuple, List

from lerobot.utils.rotation import Rotation
from ..utils import TeleopEvents

# 全局变量跟踪SDK初始化状态（使用线程锁保证线程安全）
_XROBOTOOLKIT_INITIALIZED = False
_XROBOTOOLKIT_LOCK = threading.Lock()


class InputController:
    """Base class for input controllers that generate motion deltas.
    
    参考 PICO 官方示例的设计理念，提供更稳定的增量计算。
    """

    def __init__(self, deadzone=0.1, scale_factor=0.6):
        """
        Initialize the controller.

        Args:
            deadzone: Controller deadzone threshold
            scale_factor: Motion scaling factor (参考官方示例)
        """
        self.deadzone = deadzone
        self.scale_factor = scale_factor
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        
        # 夹爪控制（连续值，参考官方示例）
        self.left_gripper_value = 0.0  # 0.0 = 打开, 1.0 = 关闭
        self.right_gripper_value = 0.0

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_pose_deltas(self) -> Tuple[float, ...]:
        """
        Get pose deltas (position + quaternion).
        
        Returns:
            Tuple of pose deltas. Base implementation returns 7DOF for single arm.
            Subclasses can override to return different numbers of DOF.
        """
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def get_gripper_values(self):
        """Return current gripper values (continuous)."""
        return self.left_gripper_value, self.right_gripper_value


class PicoController(InputController):
    """Generate motion deltas from PICO XR controller input for dual-arm teleoperation.
    
    参考官方示例 dual_arm_ur_controller.py 的实现：
    - 当抓握键按下时，记录当前控制器姿态作为初始姿态
    - 计算相对于初始姿态的增量，而不是相对于上一时刻
    - 当抓握键释放时，重置初始姿态
    """

    def __init__(self, deadzone=0.1, scale_factor=0.6, use_coordinate_transform=True):
        """
        Initialize PICO controller.
        
        Args:
            deadzone: Controller deadzone threshold (参考官方示例的 CONTROLLER_DEADZONE)
            scale_factor: Motion scaling factor (参考官方示例的 DEFAULT_SCALE_FACTOR)
            use_coordinate_transform: Whether to apply coordinate transformation
        """
        super().__init__(deadzone, scale_factor)
        self.use_coordinate_transform = use_coordinate_transform
        self.xrt = None
        
        # 控制器状态
        self.left_pose = None  # [x, y, z, rot_x, rot_y, rot_z]
        self.right_pose = None
        
        # 按钮状态
        self.left_grip = False
        self.right_grip = False
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.a_button = False
        self.b_button = False
        self.x_button = False
        self.y_button = False
        
        # 摇杆状态
        self.left_axis = [0.0, 0.0]
        self.right_axis = [0.0, 0.0]
        
        # 瞬时增量动作：记录上一帧姿态
        self.prev_left_controller_xyz = None  # 左控制器上一帧位置
        self.prev_left_controller_quat = None  # 左控制器上一帧四元数
        self.prev_right_controller_xyz = None  # 右控制器上一帧位置
        self.prev_right_controller_quat = None  # 右控制器上一帧四元数
        
        # 坐标系转换矩阵（参考官方示例）
        self.R_headset_to_world = np.array([
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

    def start(self):
        """Initialize xrobotoolkit_sdk."""
        global _XROBOTOOLKIT_INITIALIZED
        
        try:
            import xrobotoolkit_sdk as xrt
            self.xrt = xrt
            
            # 检查是否已经初始化过
            if not _XROBOTOOLKIT_INITIALIZED:
                try:
                    # 直接初始化，参考官方示例的 XrClient 实现
                    self.xrt.init()
                    _XROBOTOOLKIT_INITIALIZED = True
                    logging.info("PICO XR controller SDK initialized")
                except Exception as e:
                    logging.error(f"Failed to initialize PICO SDK: {e}")
                    self.running = False
                    return
            else:
                logging.debug("SDK already initialized, skipping")
            
            logging.info("PICO XR controller initialized successfully")
            
            print("PICO XR Controller controls for dual-arm teleoperation:")
            print("  Left grip: Control left arm (when pressed)")
            print("  Right grip: Control right arm (when pressed)")
            print("  Left trigger: Control left gripper (continuous)")
            print("  Right trigger: Control right gripper (continuous)")
            print("  A button: End episode with FAILURE")
            print("  B button: ")
            print("  X button: Rerecord episode")
            print("  Y button: End episode with SUCCESS")
            
        except ImportError as e:
            logging.error(f"Failed to import xrobotoolkit_sdk: {e}")
            self.running = False
        except Exception as e:
            logging.error(f"Failed to initialize PICO controller: {e}")
            self.running = False

    def stop(self):
        """Clean up resources."""
        global _XROBOTOOLKIT_INITIALIZED
        
        if self.xrt is not None and hasattr(self.xrt, 'close'):
            try:
                self.xrt.close()
            except Exception as e:
                logging.error(f"Error closing xrobotoolkit_sdk: {e}")
        
        self.xrt = None

    def update(self):
        """Update controller state from PICO XR SDK.
        
        参考官方示例 dual_arm_ur_controller.py 的激活检测逻辑：
        active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
        其中 CONTROLLER_DEADZONE = 0.1，所以阈值是 0.9
        """
        if self.xrt is None:
            return
            
        try:
            # 获取姿态信息（参考官方示例的 XrClient 接口）
            self.left_pose = self.xrt.get_left_controller_pose()
            self.right_pose = self.xrt.get_right_controller_pose()
            
            # 获取按钮和触发器状态
            self.left_trigger = self.xrt.get_left_trigger()
            self.right_trigger = self.xrt.get_right_trigger()
            left_grip_val = self.xrt.get_left_grip()
            right_grip_val = self.xrt.get_right_grip()
            
            # 使用与官方示例一致的激活检测逻辑
            self.left_grip = left_grip_val > (1.0 - self.deadzone)  # 阈值 = 1.0 - deadzone
            self.right_grip = right_grip_val > (1.0 - self.deadzone)
            
            # 获取按钮状态
            self.a_button = self.xrt.get_A_button()
            self.b_button = self.xrt.get_B_button()
            self.x_button = self.xrt.get_X_button()
            self.y_button = self.xrt.get_Y_button()
            
            # 获取摇杆状态并应用死区
            left_axis_raw = self.xrt.get_left_axis()
            right_axis_raw = self.xrt.get_right_axis()
            
            self.left_axis = [
                0.0 if abs(left_axis_raw[0]) < self.deadzone else left_axis_raw[0],
                0.0 if abs(left_axis_raw[1]) < self.deadzone else left_axis_raw[1]
            ]
            
            self.right_axis = [
                0.0 if abs(right_axis_raw[0]) < self.deadzone else right_axis_raw[0],
                0.0 if abs(right_axis_raw[1]) < self.deadzone else right_axis_raw[1]
            ]
            
            # 设置干预标志（当任一抓握键按下时）
            self.intervention_flag = self.left_grip or self.right_grip
            
            # 设置夹爪值（连续控制，参考官方示例）
            self.left_gripper_value = self.left_trigger
            self.right_gripper_value = self.right_trigger
            
            # 设置剧集结束状态
            if self.a_button:
                self.episode_end_status = TeleopEvents.FAILURE
                print(f"A按钮按下: 设置剧集结束状态为 FAILURE")
            elif self.y_button:
                self.episode_end_status = TeleopEvents.SUCCESS
                print(f"Y按钮按下: 设置剧集结束状态为 SUCCESS")
            elif self.x_button:
                self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                print(f"X按钮按下: 设置剧集结束状态为 RERECORD_EPISODE")
                
        except Exception as e:
            logging.error(f"Error updating PICO controller state: {e}")

    def _process_pose(self, pose: Optional[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理控制器姿态，应用坐标系转换和缩放因子。
        
        严格按照官方示例 dual_arm_ur_controller.py 的 _process_xr_pose 方法实现。
        使用 meshcat.transformations 库，与官方示例完全一致。
        
        Args:
            pose: 控制器姿态列表 [x, y, z, qx, qy, qz, qw]，如果为None则返回零
            
        Returns:
            Tuple of (position_xyz, quaternion_wxyz)  # [w, x, y, z] 格式
        """
        if pose is None or len(pose) < 7:
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        
        # 提取位置和四元数 - 按照官方示例转换为 [w, x, y, z] 格式
        controller_xyz = np.array([pose[0], pose[1], pose[2]])
        controller_quat = np.array([pose[6], pose[3], pose[4], pose[5]])  # [w, x, y, z]
        
        # 应用坐标系转换（如果需要）
        if self.use_coordinate_transform:
            # 位置转换
            controller_xyz = self.R_headset_to_world @ controller_xyz
            
            # 四元数转换：严格按照官方示例
            # 官方示例：controller_quat = tf.quaternion_multiply(
            #            tf.quaternion_multiply(R_quat, controller_quat),
            #            tf.quaternion_conjugate(R_quat))
            
            # 创建4x4变换矩阵，如官方示例
            R_transform = np.eye(4)
            R_transform[:3, :3] = self.R_headset_to_world
            
            # 将4x4矩阵转换为四元数
            R_quat = tf.quaternion_from_matrix(R_transform)  # [w, x, y, z] 格式
            
            # 应用转换：q_result = R * q * R_inv
            # controller_quat = tf.quaternion_multiply(
            #     tf.quaternion_multiply(R_quat, controller_quat),
            #     tf.quaternion_conjugate(R_quat)
            # )

            controller_quat = tf.quaternion_multiply(R_quat, controller_quat) 
            
        return controller_xyz, controller_quat

    def reset(self):
        self.prev_left_controller_xyz = None
        self.prev_left_controller_quat = None
        self.prev_right_controller_xyz = None
        self.prev_right_controller_quat = None
        logging.info("PICO controller history has been reset")

    def get_pose_deltas(self):
        """
        Get 12DOF pose deltas for dual-arm control (左臂6DOF + 右臂6DOF).
        
        返回格式: (left_delta_x, left_delta_y, left_delta_z, left_delta_rot_x, left_delta_rot_y, left_delta_rot_z,
                 right_delta_x, right_delta_y, right_delta_z, right_delta_rot_x, right_delta_rot_y, right_delta_rot_z)
        
        改进的瞬时增量动作实现：
        - 当抓握键激活时，计算相对于上一帧的增量
        - 当抓握键不激活时，重置上一帧姿态
        - 增量是瞬时的，不会累积
        - 第一次记录姿态时返回零增量，避免抖动
        - 使用轴角表示旋转（6DOF：平移3 + 旋转3）
        """
        if self.xrt is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        try:
            # 左臂增量计算
            left_delta_x, left_delta_y, left_delta_z = 0.0, 0.0, 0.0
            left_delta_rot_x, left_delta_rot_y, left_delta_rot_z = 0.0, 0.0, 0.0
            
            if self.left_grip and self.left_pose is not None:
                # 处理当前姿态
                curr_xyz, curr_quat = self._process_pose(self.left_pose)
                
                # 如果上一帧姿态未记录，则记录当前姿态作为上一帧姿态，并返回零增量
                if self.prev_left_controller_xyz is None:
                    self.prev_left_controller_xyz = curr_xyz.copy()
                    self.prev_left_controller_quat = curr_quat.copy()
                    logging.debug(f"left arm by pico：记录左臂上一帧姿态，返回零增量")
                    # 返回零增量，避免第一次运动抖动
                    left_delta_x, left_delta_y, left_delta_z = 0.0, 0.0, 0.0
                    left_delta_rot_x, left_delta_rot_y, left_delta_rot_z = 0.0, 0.0, 0.0
                else:
                    # 计算相对于上一帧的增量（瞬时增量）
                    delta_xyz = (curr_xyz - self.prev_left_controller_xyz) * self.scale_factor
                    left_delta_x, left_delta_y, left_delta_z = delta_xyz
                    
                    # 计算四元数增量（使用meshcat.transformations，与官方示例一致）
                    if self.prev_left_controller_quat is None:
                        # 这不应该发生，因为我们在上面检查了prev_left_controller_xyz不为None
                        delta_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
                    else:
                        # 计算四元数差异：delta_q = curr_quat * prev_quat_inv 
                        prev_quat_inv = tf.quaternion_conjugate(self.prev_left_controller_quat)
                        delta_quat = tf.quaternion_multiply(curr_quat, prev_quat_inv)
                    
                    # 将四元数增量转换为轴角表示（旋转向量）
                    # delta_quat是[w, x, y, z]格式，需要转换为[x, y, z, w]格式供Rotation类使用
                    delta_quat_xyzw = np.array([delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0]])
                    rot = Rotation.from_quat(delta_quat_xyzw)
                    delta_rotvec = rot.as_rotvec()  # [rx, ry, rz] 轴角表示
                    
                    left_delta_rot_x, left_delta_rot_y, left_delta_rot_z = delta_rotvec
                    
                    # 更新上一帧姿态为当前姿态
                    self.prev_left_controller_xyz = curr_xyz.copy()
                    self.prev_left_controller_quat = curr_quat.copy()
                    
            elif not self.left_grip:
                # 左抓握键不激活：重置上一帧姿态
                if self.prev_left_controller_xyz is not None:
                    self.prev_left_controller_xyz = None
                    self.prev_left_controller_quat = None
            
            # 右臂增量计算
            right_delta_x, right_delta_y, right_delta_z = 0.0, 0.0, 0.0
            right_delta_rot_x, right_delta_rot_y, right_delta_rot_z = 0.0, 0.0, 0.0
            
            if self.right_grip and self.right_pose is not None:
                # 处理当前姿态
                curr_xyz, curr_quat = self._process_pose(self.right_pose)
                
                # 如果上一帧姿态未记录，则记录当前姿态作为上一帧姿态，并返回零增量
                if self.prev_right_controller_xyz is None:
                    self.prev_right_controller_xyz = curr_xyz.copy()
                    self.prev_right_controller_quat = curr_quat.copy()
                    logging.debug(f"right arm by pico：记录右臂上一帧姿态，返回零增量")
                    # 返回零增量，避免第一次运动抖动
                    right_delta_x, right_delta_y, right_delta_z = 0.0, 0.0, 0.0
                    right_delta_rot_x, right_delta_rot_y, right_delta_rot_z = 0.0, 0.0, 0.0
                else:
                    # 计算相对于上一帧的增量（瞬时增量）
                    delta_xyz = (curr_xyz - self.prev_right_controller_xyz) * self.scale_factor
                    right_delta_x, right_delta_y, right_delta_z = delta_xyz
                    
                    # 计算四元数增量（使用meshcat.transformations，与官方示例一致）
                    if self.prev_right_controller_quat is None:
                        # 这不应该发生，因为我们在上面检查了prev_right_controller_xyz不为None
                        delta_quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
                    else:
                        # 计算四元数差异：delta_q = curr_quat * prev_quat_inv
                        prev_quat_inv = tf.quaternion_conjugate(self.prev_right_controller_quat)
                        delta_quat = tf.quaternion_multiply(curr_quat, prev_quat_inv)
                    
                    # 将四元数增量转换为轴角表示（旋转向量）
                    # delta_quat是[w, x, y, z]格式，需要转换为[x, y, z, w]格式供Rotation类使用
                    delta_quat_xyzw = np.array([delta_quat[1], delta_quat[2], delta_quat[3], delta_quat[0]])
                    rot = Rotation.from_quat(delta_quat_xyzw)
                    delta_rotvec = rot.as_rotvec()  # [rx, ry, rz] 轴角表示
                    
                    right_delta_rot_x, right_delta_rot_y, right_delta_rot_z = delta_rotvec
                    
                    # 更新上一帧姿态为当前姿态
                    self.prev_right_controller_xyz = curr_xyz.copy()
                    self.prev_right_controller_quat = curr_quat.copy()
                    
            elif not self.right_grip:
                # 右抓握键不激活：重置上一帧姿态
                if self.prev_right_controller_xyz is not None:
                    self.prev_right_controller_xyz = None
                    self.prev_right_controller_quat = None
            
            return (left_delta_x, left_delta_y, left_delta_z, left_delta_rot_x, left_delta_rot_y, left_delta_rot_z,
                    right_delta_x, right_delta_y, right_delta_z, right_delta_rot_x, right_delta_rot_y, right_delta_rot_z)
            
        except Exception as e:
            logging.error(f"Error getting pose deltas from PICO controller: {e}")
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # 向后兼容的方法
    def get_deltas(self):
        """向后兼容：与原有接口保持一致"""
        return self.get_pose_deltas()
    
    def gripper_command(self):
        """向后兼容：返回离散的夹爪命令"""
        left_val, right_val = self.get_gripper_values()
        # 转换为离散命令
        if left_val > 0.5:
            return "close"
        elif left_val < 0.5:
            return "open"
        else:
            return "stay"
