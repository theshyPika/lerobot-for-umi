import os
import sys 
import numpy as np
import threading
import time
import cv2
import logging
import meshcat.transformations as tf
from typing import Dict, Any

# 保持原有的路径导入逻辑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .config_gripper import DHGripper
from .config_g2 import G2RobotConfig
from ..robot import Robot
from lerobot.utils.rotation import Rotation

# 使用标准 logging 模块
logger = logging.getLogger(__name__)

class G2Robot(Robot):
    config_class = G2RobotConfig
    name = "g2"
    
    def __init__(self, config: G2RobotConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = False
        self.robot = None
        self.camera = None
        self._agibot_gdk = None  # 延迟导入
        
        # 夹爪控制器字典，key为"left"或"right"
        self.grippers = {
            "left": None,
            "right": None
        }

        # 夹爪位置缓存，用于避免重复发送相同位置
        self._last_gripper_positions = {
            "left": None,
            "right": None
        }
        
        # 夹爪控制阈值，只有位置变化超过此阈值时才发送新命令
        self._gripper_threshold = 0.01  # 阈值1%
        
        # 夹爪控制错误计数和重试机制
        self._gripper_error_count = {
            "left": 0,
            "right": 0
        }
        self._max_gripper_errors = 3  # 最大错误次数

        # 相机配置映射（将在 connect 中填充）
        self.camera_types = None
        
        # 根据配置选择摄像头
        if config.dual_arm:
            # 双臂模式：使用所有摄像头
            self.selected_cameras = {
                'head_color': 'head_color',
                'hand_left': 'hand_left',  
                'hand_right': 'hand_right',  
            }
        else:
            # 单臂模式：只使用对应的摄像头
            if config.use_left_arm:
                # 左臂模式：使用 head_color 和 hand_left
                self.selected_cameras = {
                    'head_color': 'head_color',
                    'hand_left': 'hand_left',  
                }
            else:
                # 右臂模式：使用 head_color 和 hand_right
                self.selected_cameras = {
                    'head_color': 'head_color',
                    'hand_right': 'hand_right',  
                }

        self.camera_dimensions = {}

        # --- 异步采集相关变量 ---
        self._frames = {cam: None for cam in self.selected_cameras.keys()} # 图像缓存
        self._locks = {cam: threading.Lock() for cam in self.selected_cameras.keys()} # 线程锁
        self._running = False # 线程运行标志
        self._threads = [] # 存储线程对象
        
        self.left_arm_active = False
        self.right_arm_active = False

        # --- 遥操状态管理 ---
        self.curr_left_ee_pose = None 
        self.curr_right_ee_pose = None
        
        # 激活阈值
        self.activation_threshold = 0.001  # 位置激活阈值
        self.quaternion_threshold = 0.01  # 四元数激活阈值
        
        # 为兼容性添加 bus 属性
        # self.bus = self.G2RobotBus(self)

    def connect(self, calibrate: bool = False) -> None:
        if self._is_connected:
            return

        try:
            # 延迟导入 agibot_gdk
            import agibot_gdk
            self._agibot_gdk = agibot_gdk
            
            # 初始化相机类型映射
            self.camera_types = {
                'head_color': agibot_gdk.CameraType.kHeadColor,
                'hand_left': agibot_gdk.CameraType.kHandLeftColor,
                'hand_right': agibot_gdk.CameraType.kHandRightColor,
            }
            
            logger.info("正在初始化 GDK 并连接机器人...")
            print("DEBUG: 调用 agibot_gdk.gdk_init() 之前")
            if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
                raise RuntimeError("GDK 初始化失败")
            print("DEBUG: agibot_gdk.gdk_init() 成功")
            
            self.robot = agibot_gdk.Robot()
            print("DEBUG: 创建 robot 实例")
            self.camera = agibot_gdk.Camera()
            print("DEBUG: 创建 camera 实例")
            
            # 根据配置初始化夹爪
            self._init_grippers(calibrate)
            
            time.sleep(2)
            
            # 启动相机采集线程
            self._running = True
            for cam_name in self.selected_cameras.keys():
                t = threading.Thread(
                    target=self._camera_update_loop, 
                    args=(cam_name,), 
                    daemon=True
                )
                t.start()
                self._threads.append(t)
            
            self._is_connected = True
            for cam_name, cam_type in self.camera_types.items():
                if cam_name in self.selected_cameras.keys():
                    
                    if shape:=self.camera.get_image_shape(cam_type):
                        self.camera_dimensions[cam_name] = {
                            'width': shape[0],
                            'height': shape[1],
                        }
                        logger.info(f"相机 {cam_name}: {shape[0]}x{shape[1]}")
                    else:
                        # 使用默认尺寸
                        self.camera_dimensions[cam_name] = {
                            'height': 480,
                            'width': 640
                        }
                        logger.warning(f"相机 {cam_name} 获取失败，使用默认尺寸")
            logger.info("机器人及多线程相机采集已就绪")
            print("DEBUG: connect 完成")
            
            if calibrate:
                self.calibrate()
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.disconnect()
            raise e

    def _camera_update_loop(self, cam_name):
        """每个相机独立的采集线程"""
        cam_type = self.camera_types[cam_name]
        logger.info(f"启动相机线程: {cam_name}")
        
        # 使用更短的超时时间，更容易停止
        timeout_ms = 10.0  # 从20ms减少到10ms
        
        while self._running:
            try:
                # 使用更短的超时时间
                image = self.camera.get_latest_image(cam_type, timeout_ms)
                if image is None:
                    # 如果没有图像，检查是否需要停止
                    if not self._running:
                        break
                    time.sleep(0.002)  # 更短的休眠时间
                    continue

                # --- 颜色空间转换 (修正点 1) ---
                frame = None
                if hasattr(image, 'encoding') and image.encoding == self._agibot_gdk.Encoding.JPEG:
                    nparr = np.frombuffer(image.data, np.uint8)
                    # cv2.imdecode 得到 BGR，必须转为 RGB 给 LeRobot
                    bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                
                elif hasattr(image, 'color_format'):
                    # 如果是原始 RGB 数据，直接 reshape
                    img_array = np.frombuffer(image.data, dtype=np.uint8)
                    expected_size = image.width * image.height * 3
                    if img_array.size >= expected_size:
                        frame = img_array[:expected_size].reshape((image.height, image.width, 3))

                if frame is not None:
                    with self._locks[cam_name]:
                        self._frames[cam_name] = frame.copy()
                
                # 更短的休眠时间，更容易响应停止信号
                time.sleep(0.01)
                
            except Exception as e:
                # 如果发生异常，检查是否需要停止
                if not self._running:
                    break
                logger.error(f"相机 {cam_name} 采集异常: {e}")
                time.sleep(0.05)  # 异常后稍微休眠
        
        logger.info(f"相机线程 {cam_name} 已停止")

    def _cleanup_grippers(self):
        """清理夹爪资源"""
        logger.info("正在清理夹爪资源...")
        for arm in ["left", "right"]:
            gripper = self.grippers.get(arm)
            if gripper is not None:
                try:
                    if hasattr(gripper, 'disconnect'):
                        logger.info(f"断开 {arm} 臂夹爪连接...")
                        gripper.disconnect()
                    # 将夹爪引用设为 None，让垃圾回收器处理
                    self.grippers[arm] = None
                except Exception as e:
                    logger.warning(f"清理 {arm} 臂夹爪时出错: {e}")
        logger.info("夹爪资源清理完成")

    def disconnect(self):
        """断开机器人连接，确保所有资源被正确清理
        
        注意：这个方法必须被调用，否则程序退出时会出现
        "terminate called without an active exception" 错误
        """
        # 如果已经断开连接，直接返回
        if not self._is_connected:
            logger.info("机器人未连接，无需断开")
            return
            
        logger.info("开始断开机器人连接...")
        
        # 首先设置运行标志为False，让所有线程知道应该停止
        self._running = False
        
        try:
            # 1. 停止所有相机线程 - 使用更可靠的方法
            logger.info(f"正在停止 {len(self._threads)} 个相机线程...")
            
            # 给线程一些时间自然停止
            time.sleep(0.1)
            
            # 等待所有线程停止，但不要无限等待
            alive_threads = []
            for i, t in enumerate(self._threads):
                if t.is_alive():
                    logger.info(f"等待线程 {i} 停止...")
                    t.join(timeout=0.5)  # 减少超时时间到0.5秒
                    if t.is_alive():
                        logger.warning(f"线程 {i} 仍在运行，将强制停止")
                        alive_threads.append(t)
                else:
                    logger.info(f"线程 {i} 已停止")
            
            # 如果有线程仍然存活，记录警告但继续执行
            if alive_threads:
                logger.warning(f"{len(alive_threads)} 个线程仍在运行，但将继续断开连接流程")
            
            self._threads = []
            logger.info("相机线程处理完成")
            
            # 2. 清理夹爪资源（在释放GDK资源之前）
            logger.info("正在清理夹爪资源...")
            self._cleanup_grippers()
            
            # 3. 关闭相机（如果存在）
            if self.camera and self._agibot_gdk:
                try:
                    logger.info("正在关闭相机...")
                    result = self.camera.close_camera()
                    if result == self._agibot_gdk.GDKRes.kSuccess:
                        logger.info("相机关闭成功")
                    else:
                        logger.warning(f"相机关闭失败，错误码: {result}")
                except Exception as e:
                    logger.warning(f"关闭相机时出错: {e}")
            
            # 4. 释放 GDK 资源（如果存在）
            if self._agibot_gdk:
                try:
                    logger.info("正在释放 GDK 资源...")
                    # 添加延迟，确保相机资源完全释放
                    time.sleep(0.1)
                    if self._agibot_gdk.gdk_release() != self._agibot_gdk.GDKRes.kSuccess:
                        logger.warning("GDK 释放失败")
                    else:
                        logger.info("GDK 释放成功")
                except Exception as e:
                    logger.warning(f"释放 GDK 资源时出错: {e}")
                    
        except Exception as e:
            logger.error(f"断开连接过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            # 5. 无论如何都要清理所有资源引用
            logger.info("正在清理所有资源引用...")
            
            # 确保运行标志为False
            self._running = False
            
            # 清理线程列表
            self._threads = []
            
            # 清理帧缓存（不尝试获取锁，避免死锁）
            for cam_name in list(self._frames.keys()):
                try:
                    # 直接设置帧为None，不尝试获取锁
                    self._frames[cam_name] = None
                except Exception as e:
                    logger.warning(f"清理帧缓存 {cam_name} 时出错: {e}")
            
            # 清理 GDK 相关资源
            self.robot = None
            self.camera = None
            self._agibot_gdk = None
            
            # 清理夹爪资源
            for arm in ["left", "right"]:
                self.grippers[arm] = None
                self._last_gripper_positions[arm] = None
            
            # 更新状态标志
            self._is_connected = False
            self._is_calibrated = False
            
            # 清理激活状态
            self.curr_left_ee_pose = None
            self.curr_right_ee_pose = None
            
            logger.info("机器人已完全断开连接，所有资源已清理")
            
    @property
    def observation_features(self) -> Dict:
        """观测特征 - 根据配置返回对应手臂的关节角度和摄像头"""
        if self.config.dual_arm:
            return self._get_dual_arm_observation_features()
        else:
            return self._get_single_arm_observation_features()
    
    def _get_dual_arm_observation_features(self) -> Dict:
        """获取双臂模式的观测特征"""
        # 总共16个关节：左臂7个 + 右臂7个 + 左夹爪1个 + 右夹爪1个
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),  # 从14改为16
                "names": [f"joint_{i}" for i in range(1, 17)]  # joint_1 到 joint_16
            }
        }
        
        # 添加末端执行器位姿特征（封装到frame_poses中）- 使用平移+轴角表示
        features["observation.state.frame_poses"] = {
            "dtype": "float32",
            "shape": (12,),  # 左臂6DOF + 右臂6DOF (平移3 + 轴角3)
            "names": [
                "left_ee_x", "left_ee_y", "left_ee_z", 
                "left_ee_rx", "left_ee_ry", "left_ee_rz",
                "right_ee_x", "right_ee_y", "right_ee_z",
                "right_ee_rx", "right_ee_ry", "right_ee_rz"
            ]
        }
        
        # 添加相机特征
        for cam_name in self.selected_cameras.keys():
            key = f"observation.images.{cam_name}"
            # 使用存储的尺寸信息
            if cam_name in self.camera_dimensions:
                dims = self.camera_dimensions[cam_name]
                shape = (dims['height'], dims['width'], 3)
            else:
                shape = (480, 640, 3)
                
            features[key] = {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            }
            
        return features
    
    def _get_single_arm_observation_features(self) -> Dict:
        """获取单臂模式的观测特征"""
        # 单臂模式：只返回对应手臂的关节和夹爪
        # 总共8个关节：手臂7个 + 夹爪1个
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"joint_{i}" for i in range(1, 9)]  # joint_1 到 joint_8
            }
        }
        
        # 添加末端执行器位姿特征（封装到frame_poses中）- 使用平移+轴角表示
        # 对于瞬时增量动作，我们需要当前末端执行器位姿信息
        if self.config.use_left_arm:
            # 左臂模式：只添加左臂当前位姿特征
            features["observation.state.frame_poses"] = {
                "dtype": "float32",
                "shape": (6,),  # 左臂6DOF (平移3 + 轴角3)
                "names": [
                    "left_ee_x", "left_ee_y", "left_ee_z", 
                    "left_ee_rx", "left_ee_ry", "left_ee_rz"
                ]
            }
        else:
            # 右臂模式：只添加右臂当前位姿特征
            features["observation.state.frame_poses"] = {
                "dtype": "float32",
                "shape": (6,),  # 右臂6DOF (平移3 + 轴角3)
                "names": [
                    "right_ee_x", "right_ee_y", "right_ee_z",
                    "right_ee_rx", "right_ee_ry", "right_ee_rz"
                ]
            }
        
        # 添加相机特征 - 只返回对应手臂的摄像头
        for cam_name in self.selected_cameras.keys():
            if self.config.use_left_arm:
                if cam_name in ['head_color', 'hand_left']:
                    key = f"observation.images.{cam_name}"
                    if cam_name in self.camera_dimensions:
                        dims = self.camera_dimensions[cam_name]
                        shape = (dims['height'], dims['width'], 3)
                    else:
                        shape = (480, 640, 3)
                        
                    features[key] = {
                        "dtype": "video",
                        "shape": shape,
                        "names": ["height", "width", "channel"],
                    }
            else:  # 右臂模式
                if cam_name in ['head_color', 'hand_right']:
                    key = f"observation.images.{cam_name}"
                    if cam_name in self.camera_dimensions:
                        dims = self.camera_dimensions[cam_name]
                        shape = (dims['height'], dims['width'], 3)
                    else:
                        shape = (480, 640, 3)
                        
                    features[key] = {
                        "dtype": "video",
                        "shape": shape,
                        "names": ["height", "width", "channel"],
                    }
            
        return features

    @property
    def action_features(self) -> Dict:
        """动作特征 - 根据配置返回单臂或双臂的delta末端位姿以及夹爪"""
        if self.config.dual_arm:
            return self._get_dual_arm_action_features()
        else:
            return self._get_single_arm_action_features()
    
    def _get_dual_arm_action_features(self) -> Dict:
        """获取双臂模式的动作特征（6DoF轴角表示）"""
        names = {}
        idx = 0
        
        # 左臂位置增量
        left_position_names = ["left_delta_x", "left_delta_y", "left_delta_z"]
        for name in left_position_names:
            names[name] = idx
            idx += 1
        
        # 左臂轴角增量（旋转向量）
        left_rotation_names = ["left_delta_rot_x", "left_delta_rot_y", "left_delta_rot_z"]
        for name in left_rotation_names:
            names[name] = idx
            idx += 1
        
        # 右臂位置增量
        right_position_names = ["right_delta_x", "right_delta_y", "right_delta_z"]
        for name in right_position_names:
            names[name] = idx
            idx += 1
        
        # 右臂轴角增量（旋转向量）
        right_rotation_names = ["right_delta_rot_x", "right_delta_rot_y", "right_delta_rot_z"]
        for name in right_rotation_names:
            names[name] = idx
            idx += 1
        
        # 夹爪控制
        if self.config.use_gripper:
            names["left_gripper"] = idx
            idx += 1
            names["right_gripper"] = idx
            idx += 1
        
        shape = (idx,)
        return {
            "action": {
                "dtype": "float32",
                "shape": shape,
                "names": names
            }
        }
    
    def _get_single_arm_action_features(self) -> Dict:
        """获取单臂模式的动作特征（6DoF轴角表示）"""
        prefix = self.config.single_arm_prefix
        names = {}
        idx = 0
        
        # 位置增量
        for axis in ["x", "y", "z"]:
            names[f"{prefix}delta_{axis}"] = idx
            idx += 1
        
        # 轴角增量（旋转向量）
        for axis in ["rot_x", "rot_y", "rot_z"]:
            names[f"{prefix}delta_{axis}"] = idx
            idx += 1
        
        # 夹爪控制
        if self.config.use_gripper:
            names[f"{prefix}gripper"] = idx
            idx += 1
        
        shape = (idx,)
        return {
            "action": {
                "dtype": "float32",
                "shape": shape,
                "names": names
            }
        }
    
    def get_latest_image(self, camera_type, timeout_ms=200.0): 
        try:
            if self.camera is None or self._agibot_gdk is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
            image = self.camera.get_latest_image(camera_type, timeout_ms)
            if image is not None:
                frame = None
                
                if hasattr(image, 'data'):
                    if hasattr(image, 'encoding') and image.encoding == self._agibot_gdk.Encoding.JPEG:
                        # JPEG 解码
                        nparr = np.frombuffer(image.data if not isinstance(image.data, np.ndarray) else image.data.tobytes(), np.uint8)
                        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            # 【修正】将 OpenCV 默认的 BGR 转回 RGB
                            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                    elif hasattr(image, 'color_format') and (image.color_format in [self._agibot_gdk.ColorFormat.kRGB8, self._agibot_gdk.ColorFormat.RGB]):
                        # 如果原始数据就是 RGB
                        if isinstance(image.data, np.ndarray):
                            img_array = image.data
                        else:
                            img_array = np.frombuffer(image.data, dtype=np.uint8)
                        
                        expected_size = image.width * image.height * 3
                        if img_array.size >= expected_size:
                            frame = img_array[:expected_size].reshape((image.height, image.width, 3))
                            # 【修正】不要再转成 BGR，直接保持 RGB
                
                if frame is not None:
                    return frame
                else:
                    return np.zeros((image.height, image.width, 3), dtype=np.uint8)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
        except Exception as e:
            logger.error(f"获取相机图像失败: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_observation(self) -> Dict[str, Any]:
        """异步版本 - 从缓存读取数据，返回角度，根据配置返回对应手臂的关节角度和摄像头"""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected")
            
        obs_dict = {}
        
        # 1. 获取关节数据
        try:
            joints_states = self.robot.get_joint_states()
            
            # 调试：打印所有关节名称和位置
            logger.debug(f"获取到 {len(joints_states['states'])} 个关节状态")
            
            # 创建一个字典来按名称查找关节位置
            joint_positions_by_name = {}
            for state in joints_states['states']:
                joint_name = state['name']
                motor_position = state['motor_position']
                joint_positions_by_name[joint_name] = motor_position
                logger.debug(f"关节 {joint_name}: {motor_position:.3f} 弧度 ({motor_position * (180.0 / 3.141592653589793):.1f} 度)")
            
            # 定义手臂关节名称（按照start_init.cpp中的顺序）
            left_arm_joint_names = [
                "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
                "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7"
            ]
            
            right_arm_joint_names = [
                "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
                "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7"
            ]
            
            # 按名称提取左臂关节位置
            left_arm_positions = []
            for joint_name in left_arm_joint_names:
                if joint_name in joint_positions_by_name:
                    left_arm_positions.append(joint_positions_by_name[joint_name])
                else:
                    logger.warning(f"找不到关节: {joint_name}")
                    left_arm_positions.append(0.0)
            
            # 按名称提取右臂关节位置
            right_arm_positions = []
            for joint_name in right_arm_joint_names:
                if joint_name in joint_positions_by_name:
                    right_arm_positions.append(joint_positions_by_name[joint_name])
                else:
                    logger.warning(f"找不到关节: {joint_name}")
                    right_arm_positions.append(0.0)
            
            # 将弧度转换为角度
            left_arm_positions_deg = [pos * (180.0 / 3.141592653589793) for pos in left_arm_positions]
            right_arm_positions_deg = [pos * (180.0 / 3.141592653589793) for pos in right_arm_positions]
            
            # 根据配置返回关节数据
            if self.config.dual_arm:
                # 双臂模式：返回所有关节
                arm_positions_deg = left_arm_positions_deg + right_arm_positions_deg
                gripper_positions = [0.0, 0.0]
                positions = arm_positions_deg + gripper_positions  # 总共16个位置
                
                # 存储到 observation.state
                obs_dict["observation.state"] = np.array(positions, dtype=np.float32)
                
                # 为兼容性添加 joint_i.pos 键
                for i, pos in enumerate(positions, start=1):
                    obs_dict[f"joint_{i}.pos"] = float(pos)
            else:
                # 单臂模式：只返回对应手臂的关节
                if self.config.use_left_arm:
                    arm_positions_deg = left_arm_positions_deg
                    gripper_positions = [0.0]  # 左夹爪
                    positions = arm_positions_deg + gripper_positions  # 总共8个位置
                    
                    # 存储到 observation.state
                    obs_dict["observation.state"] = np.array(positions, dtype=np.float32)
                    
                    # 为兼容性添加 joint_i.pos 键（重新编号为1-8）
                    for i, pos in enumerate(positions, start=1):
                        obs_dict[f"joint_{i}.pos"] = float(pos)
                else:
                    arm_positions_deg = right_arm_positions_deg
                    gripper_positions = [0.0]  # 右夹爪
                    positions = arm_positions_deg + gripper_positions  # 总共8个位置
                    
                    # 存储到 observation.state
                    obs_dict["observation.state"] = np.array(positions, dtype=np.float32)
                    
                    # 为兼容性添加 joint_i.pos 键（重新编号为1-8）
                    for i, pos in enumerate(positions, start=1):
                        obs_dict[f"joint_{i}.pos"] = float(pos)
            
            # 添加夹爪状态
            gripper_config = self.config.gripper_config
            if self.config.dual_arm:
                # 双臂模式：处理两个夹爪
                for arm in ["left", "right"]:
                    if arm in gripper_config:
                        gripper_type = gripper_config[arm].get("type", "g2")
                        if gripper_type == "dh":
                            try:
                                gripper = self.grippers.get(arm)
                                if gripper and hasattr(gripper, 'get_position'):
                                    gripper_pos = gripper.get_position()
                                    config = gripper_config[arm]
                                    open_pos = config.get("open_position", 1000)
                                    close_pos = config.get("close_position", 0)
                                    if open_pos != close_pos:
                                        normalized_pos = (gripper_pos - close_pos) / (open_pos - close_pos)
                                        joint_idx = 15 if arm == "right" else 14
                                        obs_dict[f"joint_{joint_idx}.pos"] = float(normalized_pos)
                            except Exception as e:
                                logger.warning(f"获取{arm}夹爪位置失败: {e}")
            else:
                # 单臂模式：只处理对应手臂的夹爪
                arm = "left" if self.config.use_left_arm else "right"
                if arm in gripper_config:
                    gripper_type = gripper_config[arm].get("type", "g2")
                    if gripper_type == "dh":
                        try:
                            gripper = self.grippers.get(arm)
                            if gripper and hasattr(gripper, 'get_position'):
                                gripper_pos = gripper.get_position()
                                config = gripper_config[arm]
                                open_pos = config.get("open_position", 1000)
                                close_pos = config.get("close_position", 0)
                                if open_pos != close_pos:
                                    normalized_pos = (gripper_pos - close_pos) / (open_pos - close_pos)
                                    # 单臂模式下，夹爪是第8个关节
                                    obs_dict["joint_8.pos"] = float(normalized_pos)
                                    # 更新 observation.state 中的夹爪位置
                                    if "observation.state" in obs_dict:
                                        positions = obs_dict["observation.state"].copy()
                                        positions[7] = normalized_pos
                                        obs_dict["observation.state"] = positions
                        except Exception as e:
                            logger.warning(f"获取{arm}夹爪位置失败: {e}")
                
        except Exception as e:
            logger.warning(f"获取关节状态失败: {e}")
            # 根据配置返回零值
            if self.config.dual_arm:
                obs_dict["observation.state"] = np.zeros(16, dtype=np.float32)
                for i in range(1, 17):
                    obs_dict[f"joint_{i}.pos"] = 0.0
            else:
                obs_dict["observation.state"] = np.zeros(8, dtype=np.float32)
                for i in range(1, 9):
                    obs_dict[f"joint_{i}.pos"] = 0.0

        # 2. 从缓存获取图像（非阻塞）
        if self.config.dual_arm:
            # 双臂模式：返回所有摄像头
            for cam_name in self.selected_cameras.keys():
                key = f"observation.images.{cam_name}"
                with self._locks[cam_name]:
                    frame = self._frames[cam_name]
                    if frame is not None:
                        obs_dict[key] = frame.copy()  # 复制以避免线程竞争
                    else:
                        # 返回占位图像
                        if cam_name in self.camera_dimensions:
                            dims = self.camera_dimensions[cam_name]
                            obs_dict[key] = np.zeros((dims['height'], dims['width'], 3), dtype=np.uint8)
                        else:
                            obs_dict[key] = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # 为兼容性添加简化的相机键（如 head_color）
                obs_dict[cam_name] = obs_dict[key]
        else:
            # 单臂模式：只返回对应手臂的摄像头
            if self.config.use_left_arm:
                # 左臂模式：返回 head_color 和 hand_left
                for cam_name in ['head_color', 'hand_left']:
                    key = f"observation.images.{cam_name}"
                    with self._locks[cam_name]:
                        frame = self._frames[cam_name]
                        if frame is not None:
                            obs_dict[key] = frame.copy()
                        else:
                            if cam_name in self.camera_dimensions:
                                dims = self.camera_dimensions[cam_name]
                                obs_dict[key] = np.zeros((dims['height'], dims['width'], 3), dtype=np.uint8)
                            else:
                                obs_dict[key] = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    obs_dict[cam_name] = obs_dict[key]
            else:
                # 右臂模式：返回 head_color 和 hand_right
                for cam_name in ['head_color', 'hand_right']:
                    key = f"observation.images.{cam_name}"
                    with self._locks[cam_name]:
                        frame = self._frames[cam_name]
                        if frame is not None:
                            obs_dict[key] = frame.copy()
                        else:
                            if cam_name in self.camera_dimensions:
                                dims = self.camera_dimensions[cam_name]
                                obs_dict[key] = np.zeros((dims['height'], dims['width'], 3), dtype=np.uint8)
                            else:
                                obs_dict[key] = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    obs_dict[cam_name] = obs_dict[key]

        # 3. 添加当前末端执行器位姿作为观测特征（封装到frame_poses中）- 使用平移+轴角表示
        try:
            # 获取当前末端执行器位姿
            current_poses = self.get_end_effector_pose()
            
            if self.config.dual_arm:
                # 双臂模式：返回左右臂的当前位姿
                if len(current_poses) >= 2:
                    left_pose = current_poses[0]
                    right_pose = current_poses[1]
                    
                    # 将四元数转换为轴角表示
                    left_quat = np.array([
                        float(left_pose.orientation.x),
                        float(left_pose.orientation.y),
                        float(left_pose.orientation.z),
                        float(left_pose.orientation.w)
                    ])
                    right_quat = np.array([
                        float(right_pose.orientation.x),
                        float(right_pose.orientation.y),
                        float(right_pose.orientation.z),
                        float(right_pose.orientation.w)
                    ])
                    
                    left_rot = Rotation.from_quat(left_quat)
                    right_rot = Rotation.from_quat(right_quat)
                    
                    left_rotvec = left_rot.as_rotvec()
                    right_rotvec = right_rot.as_rotvec()
                    
                    # 创建frame_poses数组：左臂6DOF + 右臂6DOF (平移3 + 轴角3)
                    frame_poses = np.array([
                        float(left_pose.position.x), float(left_pose.position.y), float(left_pose.position.z),
                        float(left_rotvec[0]), float(left_rotvec[1]), float(left_rotvec[2]),
                        float(right_pose.position.x), float(right_pose.position.y), float(right_pose.position.z),
                        float(right_rotvec[0]), float(right_rotvec[1]), float(right_rotvec[2])
                    ], dtype=np.float32)
                    
                    obs_dict["observation.state.frame_poses"] = frame_poses
                else:
                    # 如果无法获取位姿，返回零值
                    obs_dict["observation.state.frame_poses"] = np.zeros(12, dtype=np.float32)
            else:
                # 单臂模式：只返回对应手臂的当前位姿
                if len(current_poses) >= 2:
                    if self.config.use_left_arm:
                        # 左臂模式：返回左臂当前位姿
                        left_pose = current_poses[0]
                        
                        # 将四元数转换为轴角表示
                        left_quat = np.array([
                            float(left_pose.orientation.x),
                            float(left_pose.orientation.y),
                            float(left_pose.orientation.z),
                            float(left_pose.orientation.w)
                        ])
                        left_rot = Rotation.from_quat(left_quat)
                        left_rotvec = left_rot.as_rotvec()
                        
                        frame_poses = np.array([
                            float(left_pose.position.x), float(left_pose.position.y), float(left_pose.position.z),
                            float(left_rotvec[0]), float(left_rotvec[1]), float(left_rotvec[2])
                        ], dtype=np.float32)
                    else:
                        # 右臂模式：返回右臂当前位姿
                        right_pose = current_poses[1]
                        
                        # 将四元数转换为轴角表示
                        right_quat = np.array([
                            float(right_pose.orientation.x),
                            float(right_pose.orientation.y),
                            float(right_pose.orientation.z),
                            float(right_pose.orientation.w)
                        ])
                        right_rot = Rotation.from_quat(right_quat)
                        right_rotvec = right_rot.as_rotvec()
                        
                        frame_poses = np.array([
                            float(right_pose.position.x), float(right_pose.position.y), float(right_pose.position.z),
                            float(right_rotvec[0]), float(right_rotvec[1]), float(right_rotvec[2])
                        ], dtype=np.float32)
                    
                    obs_dict["observation.state.frame_poses"] = frame_poses
                else:
                    # 如果无法获取位姿，返回零值
                    obs_dict["observation.state.frame_poses"] = np.zeros(6, dtype=np.float32)
                    
        except Exception as e:
            logger.warning(f"获取末端执行器位姿失败: {e}")
            # 返回零值
            if self.config.dual_arm:
                obs_dict["observation.state.frame_poses"] = np.zeros(12, dtype=np.float32)
            else:
                obs_dict["observation.state.frame_poses"] = np.zeros(6, dtype=np.float32)

        return obs_dict

    def get_end_effector_pose(self):
        while (status := self.robot.get_motion_control_status()) is None:
            pass
        return status.frame_poses

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Supports two action formats:
        1. Joint position mode: {"joint_1.pos": value1, "joint_2.pos": value2, ...} - uses joint control
        2. End effector pose control mode: controls robot end effector pose based on delta increments
        
        Supports single-arm and dual-arm modes, automatically handles based on configuration.
        
        Note: Now supports instantaneous delta actions, delta is relative to the previous frame, not initial pose.
        
        Safety: Added action magnitude limits to prevent abnormally large actions.
        """
        
        # First handle gripper control (if any)
        self._control_grippers(action)
                
        # Check action format
        is_ee_pose_mode = self._is_ee_pose_mode(action)
                
        if not is_ee_pose_mode:
            logger.error(f"Unsupported action format: {list(action.keys())}")
            return {"success": False, "error": "Unsupported action format, only end effector pose control mode supported"}
        
        # Get activation status based on configuration
        if self.config.dual_arm:
            # Dual-arm mode: check activation status of both arms
            left_active, right_active = self._get_dual_arm_activation(action)
        else:
            # Single-arm mode: only check enabled arm based on configuration
            left_active, right_active = self._get_single_arm_activation(action)
        
        # Update activation state, record or reset initial pose (for observation features)
        self._update_activation_state(left_active, right_active)
        
        try:
            # Create end effector pose control command
            target_pose = self._agibot_gdk.EndEffectorPose()

            current_poses = None
            if left_active or right_active:
                current_poses = self.get_end_effector_pose()
                if len(current_poses) < 2:
                    logger.error("Cannot get current end effector pose")
                    return {"success": False, "error": "Cannot get current end effector pose"}

            # Process left arm
            if left_active:
                target_pose.group = self._agibot_gdk.EndEffectorControlGroup.kLeftArm
                # Get delta values (instantaneous increments) - 6DoF axis-angle representation
                delta_x = action.get("left_delta_x", 0.0)
                delta_y = action.get("left_delta_y", 0.0)
                delta_z = action.get("left_delta_z", 0.0)
                delta_rot_x = action.get("left_delta_rot_x", 0.0)
                delta_rot_y = action.get("left_delta_rot_y", 0.0)
                delta_rot_z = action.get("left_delta_rot_z", 0.0)
                
                current_left_pose = current_poses[0]
                
                # 计算目标位姿：当前位姿 + 瞬时delta
                target_pose.left_end_effector_pose.position.x = (
                    current_left_pose.position.x + delta_x
                )
                target_pose.left_end_effector_pose.position.y = (
                    current_left_pose.position.y + delta_y
                )
                target_pose.left_end_effector_pose.position.z = (
                    current_left_pose.position.z + delta_z
                )
                
                # 将轴角增量转换为四元数增量
                # 轴角表示：[rx, ry, rz] 旋转向量
                delta_rotvec = np.array([delta_rot_x, delta_rot_y, delta_rot_z])
                rot = Rotation.from_rotvec(delta_rotvec)
                delta_quat = rot.as_quat()  # [x, y, z, w] 格式
                
                # 当前姿态的四元数表示
                current_q = np.array([
                    current_left_pose.orientation.x,  # x分量
                    current_left_pose.orientation.y,  # y分量
                    current_left_pose.orientation.z,  # z分量
                    current_left_pose.orientation.w,  # w分量
                ])  # [x, y, z, w] 格式
                
                # 四元数乘法：target_q = delta_q * current_q
                # 注意：这里使用 [w, x, y, z] 格式进行乘法
                delta_q_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])  # [w, x, y, z]
                current_q_wxyz = np.array([current_q[3], current_q[0], current_q[1], current_q[2]])  # [w, x, y, z]
                target_q_wxyz = tf.quaternion_multiply(delta_q_wxyz, current_q_wxyz)  # [w, x, y, z]
                
                # 转换回[x, y, z, w]格式
                target_q = np.array([target_q_wxyz[1], target_q_wxyz[2], target_q_wxyz[3], target_q_wxyz[0]])
                
                # 将结果设置回Pose对象，保持[x, y, z, w]格式
                target_pose.left_end_effector_pose.orientation.x = target_q[0]  # x分量
                target_pose.left_end_effector_pose.orientation.y = target_q[1]  # y分量
                target_pose.left_end_effector_pose.orientation.z = target_q[2]  # z分量
                target_pose.left_end_effector_pose.orientation.w = target_q[3]  # w分量
            
            # 处理右臂
            if right_active:
                target_pose.group = self._agibot_gdk.EndEffectorControlGroup.kRightArm
                # 获取delta值（瞬时增量）- 6DoF轴角表示
                delta_x = action.get("right_delta_x", 0.0)
                delta_y = action.get("right_delta_y", 0.0)
                delta_z = action.get("right_delta_z", 0.0)
                delta_rot_x = action.get("right_delta_rot_x", 0.0)
                delta_rot_y = action.get("right_delta_rot_y", 0.0)
                delta_rot_z = action.get("right_delta_rot_z", 0.0)
                
                current_right_pose = current_poses[1]
                
                # 计算目标位姿：当前位姿 + 瞬时delta
                target_pose.right_end_effector_pose.position.x = (
                    current_right_pose.position.x + delta_x
                )
                target_pose.right_end_effector_pose.position.y = (
                    current_right_pose.position.y + delta_y
                )
                target_pose.right_end_effector_pose.position.z = (
                    current_right_pose.position.z + delta_z
                )
                
                # 将轴角增量转换为四元数增量
                # 轴角表示：[rx, ry, rz] 旋转向量
                delta_rotvec = np.array([delta_rot_x, delta_rot_y, delta_rot_z])
                rot = Rotation.from_rotvec(delta_rotvec)
                delta_quat = rot.as_quat()  # [x, y, z, w] 格式
                
                # 当前姿态的四元数表示
                current_q = np.array([
                    current_right_pose.orientation.x,  # x分量
                    current_right_pose.orientation.y,  # y分量
                    current_right_pose.orientation.z,  # z分量
                    current_right_pose.orientation.w,  # w分量
                ])  # [x, y, z, w] 格式
                
                # 四元数乘法：target_q = delta_q * current_q
                # 注意：这里使用 [w, x, y, z] 格式进行乘法
                delta_q_wxyz = np.array([delta_quat[3], delta_quat[0], delta_quat[1], delta_quat[2]])  # [w, x, y, z]
                current_q_wxyz = np.array([current_q[3], current_q[0], current_q[1], current_q[2]])  # [w, x, y, z]
                target_q_wxyz = tf.quaternion_multiply(delta_q_wxyz, current_q_wxyz)  # [w, x, y, z]
                
                # 转换回[x, y, z, w]格式
                target_q = np.array([target_q_wxyz[1], target_q_wxyz[2], target_q_wxyz[3], target_q_wxyz[0]])
                
                # 将结果设置回Pose对象，保持[x, y, z, w]格式
                target_pose.right_end_effector_pose.orientation.x = target_q[0]  # x分量
                target_pose.right_end_effector_pose.orientation.y = target_q[1]  # y分量
                target_pose.right_end_effector_pose.orientation.z = target_q[2]  # z分量
                target_pose.right_end_effector_pose.orientation.w = target_q[3]  # w分量
            
            target_pose.life_time = 0.1  # 短暂保持时间，用于增量控制 (15FPS)

            if left_active and right_active:
                target_pose.group = self._agibot_gdk.EndEffectorControlGroup.kBothArms
            
            # 发送控制命令
            if left_active or right_active:
                # success = self.robot.end_effector_pose_control(target_pose)
                success = False
                logger.info(f"send_action: {action}")
                logger.info(f"target_pose: {target_pose}")
            else:
                success = False
            
            if success:
                return {"success": True, "mode": "pico_teleop"}
            else:
                return {"success": False, "error": "末端执行器位姿控制命令发送失败"}
                
        except Exception as e:
            logger.error(f"send_action error: {e}")
            return {"success": False, "error": str(e)}
    
    def _control_grippers(self, action: dict[str, Any]):
        """根据配置控制夹爪，添加位置缓存、变化检测和线程安全"""
        
        # 检查机器人是否已连接，如果未连接则跳过夹爪控制
        if not self._is_connected:
            logger.debug("机器人未连接，跳过夹爪控制")
            return
        
        gripper_config = self.config.gripper_config
        
        for arm in ["left", "right"]:
            if arm not in gripper_config:
                continue
                
            config = gripper_config[arm]
            gripper_type = config.get("type", "g2")
            
            # 检查是否有该臂的夹爪控制信号
            gripper_key = f"{arm}_gripper"
            if gripper_key not in action:
                continue
                
            gripper_value = action[gripper_key]
            
            # 检查位置变化是否超过阈值
            last_position = self._last_gripper_positions[arm]
            if last_position is not None and abs(gripper_value - last_position) < self._gripper_threshold:
                # 位置变化小于阈值，跳过发送命令
                logger.debug(f"{arm}臂夹爪位置变化小于阈值，跳过控制")
                continue
            
            # 更新位置缓存
            self._last_gripper_positions[arm] = gripper_value
            
            # 检查错误计数，如果错误太多则跳过
            if self._gripper_error_count[arm] >= self._max_gripper_errors:
                logger.warning(f"{arm}臂夹爪错误计数过多({self._gripper_error_count[arm]})，跳过控制")
                continue
            
            # 根据夹爪类型进行控制
            if gripper_type == "g2":
                # 使用agibot_gdk控制G2夹爪
                self._control_g2_gripper(arm, gripper_value, config)
            elif gripper_type == "dh":
                # 使用DHGripper控制
                self._control_dh_gripper(arm, gripper_value, config)
            else:
                logger.warning(f"{arm}臂未知夹爪类型: {gripper_type}")
    
    def _control_g2_gripper(self, arm: str, gripper_value: float, config: dict):
        """控制G2本体夹爪"""
        try:
            # 根据错误信息，move_ee_pos可能期望同时有left_ee_state和right_ee_state
            # 即使只控制一侧，也需要提供完整的末端执行器状态
            action = {
                "left_ee_state": {
                    "joint_position": 0.0,  # 默认值
                },
                "right_ee_state": {
                    "joint_position": 0.0,  # 默认值
                }
            }
            
            # 设置要控制的臂的位置
            if arm == "left":
                action["left_ee_state"]["joint_position"] = gripper_value
            else:  # right
                action["right_ee_state"]["joint_position"] = gripper_value
            
            # 调用agibot_gdk的夹爪控制接口
            result = self.robot.move_ee_pos(action)
            logger.info(f"{arm}臂G2夹爪控制: 位置={gripper_value}, 结果={result}")
            
            # 如果成功，重置错误计数
            self._gripper_error_count[arm] = 0
            
        except Exception as e:
            logger.error(f"{arm}臂G2夹爪控制失败: {e}")
            # 增加错误计数
            self._gripper_error_count[arm] = self._gripper_error_count.get(arm, 0) + 1

    def _control_dh_gripper(self, arm: str, gripper_value: float, config: dict):
        """控制DH夹爪 - 修复超时问题"""
        try:
            gripper = self.grippers.get(arm)
            if gripper is None:
                logger.error(f"{arm}臂DH夹爪未初始化")
                return
            
            
            # 将浮点数转换为整数位置 (0-1000)
            # gripper_value范围通常是0.0-1.0，对应闭合到张开
            open_position = config.get("open_position", 1000)
            close_position = config.get("close_position", 0)
            
            # 计算实际位置
            position_range = open_position - close_position
            gripper_position = int(close_position + gripper_value * position_range)
            
            # 限制在有效范围内
            gripper_position = max(close_position, min(open_position, gripper_position))
            
            logger.info(f"{arm}臂DH夹爪控制: 原始值={gripper_value}, 位置={gripper_position}")
            gripper.set_position(gripper_position)
            gripper.wait_move(300)
            
        except Exception as e:
            logger.error(f"{arm}臂DH夹爪控制失败: {e}")
    
    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接"""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已校准"""
        return self._is_calibrated

    def calibrate(self) -> None:
        """Calibrate the robot"""
        self._is_calibrated = True
        logger.info("Robot calibration completed")

    def reset(self) -> None:
        """Reset robot state, clean up episode-related state variables
        
        This method should be called at the beginning of each episode to reset:
        1. End effector initial poses
        2. Activation states
        3. Other episode-related states
        """
        logger.info("Resetting robot state")
        
        # Reset end effector initial poses
        self.curr_left_ee_pose = None
        self.curr_right_ee_pose = None
        
        # Reset activation states
        self.left_arm_active = False
        self.right_arm_active = False
        
        # Reset gripper position cache (optional, based on requirements)
        # self._last_gripper_positions = {"left": None, "right": None}
        # self._gripper_error_count = {"left": 0, "right": 0}
        
        logger.info("Robot state reset completed: cleaned up end effector initial poses and activation states")

    def _is_arm_active_by_delta(self, delta_values):
        """
        通过delta值判断手臂是否激活（7DoF四元数版本）
        
        Args:
            delta_values: 包含7个值的列表或元组 [delta_x, delta_y, delta_z, delta_qx, delta_qy, delta_qz, delta_qw]
        
        Returns:
            bool: 如果delta值超过阈值则返回True，否则返回False
        """
        if len(delta_values) < 7:
            return False
            
        # 计算位置变化幅度
        pos_magnitude = abs(delta_values[0]) + abs(delta_values[1]) + abs(delta_values[2])
        
        # 计算四元数变化幅度
        # 对于单位四元数 [0, 0, 0, 1]，delta_qw应该接近1.0，其他分量接近0.0
        quat_magnitude = abs(delta_values[3]) + abs(delta_values[4]) + abs(delta_values[5]) + abs(1.0 - delta_values[6])
        
        # 判断是否超过阈值
        return (pos_magnitude > self.activation_threshold) or (quat_magnitude > self.quaternion_threshold)
    
    def _is_arm_active_by_delta_6dof(self, delta_values):
        """
        通过delta值判断手臂是否激活（6DoF轴角）
        
        Args:
            delta_values: 包含6个值的列表或元组 [delta_x, delta_y, delta_z, delta_rot_x, delta_rot_y, delta_rot_z]
        
        Returns:
            bool: 如果delta值超过阈值则返回True，否则返回False
        """
        if len(delta_values) < 6:
            return False
            
        # 计算位置变化幅度
        pos_magnitude = abs(delta_values[0]) + abs(delta_values[1]) + abs(delta_values[2])
        
        # 计算轴角变化幅度（旋转向量）
        rot_magnitude = abs(delta_values[3]) + abs(delta_values[4]) + abs(delta_values[5])
        
        # 判断是否超过阈值
        return (pos_magnitude > self.activation_threshold) or (rot_magnitude > self.quaternion_threshold)

    def _update_activation_state(self, left_active: bool, right_active: bool):
        """更新激活状态，记录或重置当前位姿"""
        
        # 左臂激活状态变化
        if left_active and not self.left_arm_active:
            # 左臂刚刚激活：记录初始位姿
            logger.info("左臂激活，记录初始位姿")
            self._record_curr_ee_pose("left")
            self.left_arm_active = True
        elif not left_active and self.left_arm_active:
            # 左臂刚刚去激活：重置初始位姿
            logger.info("左臂去激活，重置初始位姿")
            self.curr_left_ee_pose = None
            self.left_arm_active = False
        
        # 右臂激活状态变化
        if right_active and not self.right_arm_active:
            # 右臂刚刚激活：记录初始位姿
            logger.info("右臂激活，记录初始位姿")
            self._record_curr_ee_pose("right")
            self.right_arm_active = True
        elif not right_active and self.right_arm_active:
            # 右臂刚刚去激活：重置初始位姿
            logger.info("右臂去激活，重置初始位姿")
            self.curr_right_ee_pose = None
            self.right_arm_active = False
    
    def _record_curr_ee_pose(self, arm: str):
        """记录末端执行器初始位姿"""
        try:
            current_poses = self.get_end_effector_pose()
            if len(current_poses) < 2:
                logger.error(f"无法获取{arm}臂当前位姿")
                return
            
            if arm == "left":
                self.curr_left_ee_pose = current_poses[0]
                logger.info(f"左臂初始位姿: pos=({self.curr_left_ee_pose.position.x:.3f}, "
                          f"{self.curr_left_ee_pose.position.y:.3f}, "
                          f"{self.curr_left_ee_pose.position.z:.3f})")
            else:
                self.curr_right_ee_pose = current_poses[1]
                logger.info(f"右臂初始位姿: pos=({self.curr_right_ee_pose.position.x:.3f}, "
                          f"{self.curr_right_ee_pose.position.y:.3f}, "
                          f"{self.curr_right_ee_pose.position.z:.3f})")
                
        except Exception as e:
            logger.error(f"记录{arm}臂初始位姿失败: {e}")
    
    def _is_ee_pose_mode(self, action: dict[str, Any]) -> bool:
        """检查是否为末端执行器位姿控制模式（6DoF轴角表示）"""
        # 检查是否有delta动作键（轴角表示）
        delta_keys = ["left_delta_x", "right_delta_x", "left_delta_rot_x", "right_delta_rot_x"]
        for key in delta_keys:
            if key in action:
                return True
        
        # 检查单臂模式的前缀
        prefix = self.config.single_arm_prefix
        single_arm_keys = [f"{prefix}delta_x", f"{prefix}delta_rot_x"]
        for key in single_arm_keys:
            if key in action:
                return True
        
        return False
    
    def _get_dual_arm_activation(self, action: dict[str, Any]) -> tuple[bool, bool]:
        """获取双臂模式的激活状态（6DoF轴角表示）"""
        # 从动作中提取左臂delta值（6DoF：平移3 + 旋转3）
        left_delta_values = [
            action.get("left_delta_x", 0.0),
            action.get("left_delta_y", 0.0),
            action.get("left_delta_z", 0.0),
            action.get("left_delta_rot_x", 0.0),
            action.get("left_delta_rot_y", 0.0),
            action.get("left_delta_rot_z", 0.0)
        ]
        
        # 从动作中提取右臂delta值（6DoF：平移3 + 旋转3）
        right_delta_values = [
            action.get("right_delta_x", 0.0),
            action.get("right_delta_y", 0.0),
            action.get("right_delta_z", 0.0),
            action.get("right_delta_rot_x", 0.0),
            action.get("right_delta_rot_y", 0.0),
            action.get("right_delta_rot_z", 0.0)
        ]
        
        # 基于delta值判断激活状态（6DoF版本）
        left_active = self._is_arm_active_by_delta_6dof(left_delta_values)
        right_active = self._is_arm_active_by_delta_6dof(right_delta_values)
        
        return left_active, right_active
    
    def _get_single_arm_activation(self, action: dict[str, Any]) -> tuple[bool, bool]:
        """获取单臂模式的激活状态（6DoF轴角表示）"""
        prefix = self.config.single_arm_prefix
        is_left_arm = self.config.use_left_arm
        
        # 提取delta值（6DoF：平移3 + 旋转3）
        delta_values = [
            action.get(f"{prefix}delta_x", 0.0),
            action.get(f"{prefix}delta_y", 0.0),
            action.get(f"{prefix}delta_z", 0.0),
            action.get(f"{prefix}delta_rot_x", 0.0),
            action.get(f"{prefix}delta_rot_y", 0.0),
            action.get(f"{prefix}delta_rot_z", 0.0)
        ]
        
        # 基于delta值判断激活状态（6DoF版本）
        arm_active = self._is_arm_active_by_delta_6dof(delta_values)
        
        # 根据配置返回激活状态
        if is_left_arm:
            return arm_active, False  # 左臂激活，右臂不激活
        else:
            return False, arm_active  # 左臂不激活，右臂激活
    
    def _init_grippers(self, calibrate: bool = True):
        """根据配置初始化夹爪"""
        # 检查是否启用夹爪
        if not self.config.use_gripper:
            logger.info("夹爪功能已关闭，跳过夹爪初始化")
            return
        gripper_config = self.config.gripper_config
        
        for arm in ["left", "right"]:
            if arm not in gripper_config:
                logger.warning(f"夹爪配置中缺少{arm}臂配置")
                continue
                
            config = gripper_config[arm]
            gripper_type = config.get("type", "g2")
            
            if gripper_type == "dh":
                # 初始化DH夹爪
                try:
                    device = config.get("device", "/dev/ttyUSB0")
                    baud = config.get("baud", 115200)
                    slave_id = config.get("slave_id", 1)
                    
                    gripper = DHGripper()
                    if gripper.connect(device=device, baud=baud, slave_id=slave_id):
                        logger.info(f"{arm}臂DH夹爪连接成功")
                        if calibrate:
                            logger.info(f"{arm}臂DH夹爪校准中...")
                            gripper.calibrate()
                        self.grippers[arm] = gripper
                    else:
                        logger.error(f"{arm}臂DH夹爪连接失败")
                except Exception as e:
                    logger.error(f"{arm}臂DH夹爪初始化失败: {e}")
            elif gripper_type == "g2":
                # G2夹爪不需要特殊初始化，通过agibot_gdk控制
                logger.info(f"{arm}臂使用G2本体夹爪")
                self.grippers[arm] = None  # 标记为G2夹爪
            else:
                logger.error(f"{arm}臂未知夹爪类型: {gripper_type}")

    def configure(self) -> None:
        """配置机器人"""
        # 暂时没有额外的配置需要
        logger.info("机器人配置完成")
    
    @property
    def cameras(self):
        """为兼容性提供 cameras 属性，返回一个包含相机名称的字典"""
        # 创建一个简单的字典，键是相机名称，值是包含 keys() 方法的对象
        class CameraWrapper:
            def __init__(self, name):
                self.name = name
            
            def keys(self):
                # 返回包含相机名称的列表
                return [self.name]
        
        cameras_dict = {}
        for cam_name in self.selected_cameras.keys():
            cameras_dict[cam_name] = CameraWrapper(cam_name)
        
        return cameras_dict
    