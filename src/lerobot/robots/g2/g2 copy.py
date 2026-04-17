from __future__ import annotations

import logging
import threading
import time
from typing import Any

import cv2
import meshcat.transformations as tf
import numpy as np
import torch
from torchvision import transforms

from lerobot.motors.motors_bus import Motor
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.rotation import Rotation

from ..robot import Robot
from .config_g2 import G2RobotConfig
from .config_gripper import DHGripper
from .g2_constants import (
    GRIPPER_COMMAND_MIN_DELTA,
    LEFT_ARM_JOINT_NAMES,
    RAD_TO_DEG,
    RIGHT_ARM_JOINT_NAMES,
)

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

        self._last_gripper_positions = {
            "left": 0,
            "right": 0,
        }
        
        # 夹爪控制错误计数和重试机制
        self._gripper_error_count = {
            "left": 0,
            "right": 0
        }
        self._max_gripper_errors = 3  # 最大错误次数

        # 相机配置映射（将在 connect 中填充）
        self.camera_types = None
        
        self.motors = None
        # 根据配置选择摄像头
        if config.dual_arm:
            # 双臂模式：使用所有摄像头
            self.selected_cameras = {
                'head_color': 'head_color',
                'hand_left': 'hand_left',  
                'hand_right': 'hand_right',  
            }
            _motors = {f"{p}.joint{i}": None for i in range(1, 8) for p in ("l", "r")}
            _motors.update({f"{p}.gripper": None for p in ("l", "r")})
        else:
            # 单臂模式：只使用对应的摄像头
            if config.use_left_arm:
                # 左臂模式：使用 head_color 和 hand_left
                self.selected_cameras = {
                    'head_color': 'head_color',
                    'hand_left': 'hand_left',  
                }
                _motors= {f"l.joint{i}": None for i in range(1,8)}
                _motors.update({"l.gripper": None})
            else:
                # 右臂模式：使用 head_color 和 hand_right
                self.selected_cameras = {
                    'head_color': 'head_color',
                    'hand_right': 'hand_right',  
                }
                _motors= {f"r.joint{i}": None for i in range(1,8)}
                _motors.update({"r.gripper": None})
        self.motors = _motors

        self.camera_dimensions = {}

        # --- 异步采集相关变量 ---
        self._frames = {cam: None for cam in self.selected_cameras.keys()} # 图像缓存
        self._locks = {cam: threading.Lock() for cam in self.selected_cameras.keys()} # 线程锁
        self._running = False # 线程运行标志
        self._threads = [] # 存储线程对象
        
        # 当前关节角度存储（度）
        self._current_joint_positions = {
            'l.joint1.pos': 0.0,
            'l.joint2.pos': 0.0,
            'l.joint3.pos': 0.0,
            'l.joint4.pos': 0.0,
            'l.joint5.pos': 0.0,
            'l.joint6.pos': 0.0,
            'l.joint7.pos': 0.0,
            'r.joint1.pos': 0.0,
            'r.joint2.pos': 0.0,
            'r.joint3.pos': 0.0,
            'r.joint4.pos': 0.0,
            'r.joint5.pos': 0.0,
            'r.joint6.pos': 0.0,
            'r.joint7.pos': 0.0,
        }
        
        # 激活阈值
        self.activation_threshold = 0.0001  # 位置激活阈值
        self.quaternion_threshold = 0.001  # 四元数激活阈值
        self.joint_activation_threshold = 0.2  # 关节角度变化激活阈值（度）
        
    def _enabled_camera_names(self) -> list[str]:
        return list(self.selected_cameras.keys())

    def _joint_observation_key_names(self) -> list[str]:
        if self.config.dual_arm:
            return [f"joint_{i}" for i in range(1, 17)]
        return [f"joint_{i}" for i in range(1, 9)]

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos" : float for motor in self.motors}    

    def _ee_observation_key_names(self) -> list[str]:
        if self.config.dual_arm:
            return [
                "l.ee.x",
                "l.ee.y",
                "l.ee.z",
                "l.ee.wx",
                "l.ee.wy",
                "l.ee.wz",
                "l.ee.gripper.pos"
                "r.ee.x",
                "r.ee.y",
                "r.ee.z",
                "r.ee.wx",
                "r.ee.wy",
                "r.ee.wz",
                "r.ee.gripper.pos"
            ]
        if self.config.use_left_arm:
            return [
                "l.ee.x",
                "l.ee.y",
                "l.ee.z",
                "l.ee.wx",
                "l.ee.wy",
                "l.ee.wz",
                "l.ee.gripper.pos"
            ]
        return [
            "r.ee.x",
            "r.ee.y",
            "r.ee.z",
            "r.ee.wx",
            "r.ee.wy",
            "r.ee.wz",
            "r.ee.gripper.pos"
        ]

    def _ordered_state_scalar_keys(self) -> list[str]:
        # return self._joint_observation_key_names() + self._ee_observation_key_names() # option 1
        return self._ee_observation_key_names() # option2 ee pose only 


    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征：从配置中获取相机尺寸，返回 tensor 格式 (channels, height, width)"""
        cameras_ft = {}
        for cam_name in self._enabled_camera_names():
            if cam_name in self.config.cameras:
                cam_config = self.config.cameras[cam_name]
                cameras_ft[cam_name] = (3, cam_config.height, cam_config.width)
                logger.info(f"相机 {cam_name} 尺寸: {cam_config.width}x{cam_config.height} (tensor格式: 3x{cam_config.height}x{cam_config.width})")
            else:
                # 如果配置中没有，使用默认尺寸
                logger.warning(f"相机 {cam_name} 尺寸未在配置中定义，使用默认尺寸 640x480")
                cameras_ft[cam_name] = (3, 480, 640)
        return cameras_ft


    def _ordered_action_feature_names(self) -> list[str]:
        if self.config.dual_arm:
            names = [
                "l.ee.x",
                "l.ee.y",
                "l.ee.z",
                "l.ee.wx",
                "l.ee.wy",
                "l.ee.wz",
                "r.ee.x",
                "r.ee.y",
                "r.ee.z",
                "r.ee.wx",
                "r.ee.wy",
                "r.ee.wz",
            ]
            if self.config.use_gripper:
                names.extend(["l.ee.gripper.pos", "r.ee.gripper.pos"])
            return names
        prefix = self.config.single_arm_prefix
        names = [
            f"{prefix}.ee.x",
            f"{prefix}.ee.y",
            f"{prefix}.ee.z",
            f"{prefix}.ee.wx",
            f"{prefix}.ee.wy",
            f"{prefix}.ee.wz",
        ]
        if self.config.use_gripper:
            names.append(f"{prefix}.ee.gripper.pos")
        return names

    def _resolve_default_joint_positions(self) -> tuple[float, ...]:
        n = len(self._joint_observation_key_names())
        if self.config.default_positions and len(self.config.default_positions) == n:
            return tuple(float(x) for x in self.config.default_positions)
        return (0.0,) * n

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
            if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
                raise RuntimeError("GDK 初始化失败")

            self.robot = agibot_gdk.Robot()
            self.camera = agibot_gdk.Camera()
            
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
            logger.info("机器人及多线程相机采集已就绪")
            
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
        timeout_ms = 2.0  # 从20ms减少到5ms
        
        while self._running:
            try:
                # 使用更短的超时时间
                image = self.camera.get_latest_image(cam_type, timeout_ms)
                if image is None:
                    # 如果没有图像，检查是否需要停止
                    if not self._running:
                        break
                    time.sleep(0.001)
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
                # time.sleep(0.001)
                
            except Exception as e:
                # 如果发生异常，检查是否需要停止
                if not self._running:
                    break
                logger.error(f"相机 {cam_name} 采集异常: {e}")
                time.sleep(0.01)  # 异常后稍微休眠
        
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
    def observation_features(self) -> dict[str, type | tuple]:
        """LeRobot 标准硬件特征：标量状态 + 图像形状"""
        features: dict[str, type | tuple] = {}
        for key in self._ordered_state_scalar_keys():
            features[key] = float
        # camera features
        cameras_ft = self._cameras_ft
        for cam, shape in cameras_ft.items():
            features[cam] = shape
            logger.debug(f"camera {cam}'s expected shape = {shape}")
        return features

    @property
    def action_features(self) -> dict[str, type]:
        """末端 delta 位姿（轴角）+ 可选夹爪，与 Pico 等遥操键名一致。"""
        return {name: float for name in self._ordered_action_feature_names()}
    
    def get_latest_image(self, camera_type, timeout_ms=10.0): 
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

    def get_observation(self) -> RobotObservation:
        """与 observation_features 键名一致的扁平观测（关节度 + 末端轴角位姿 + 相机 HWC uint8）。"""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected")

        obs: dict[str, Any] = {}
        joint_keys = self._motors_ft
        ee_keys = self._ee_observation_key_names()

        try:
            joints_states = self.robot.get_joint_states()
            joint_positions_by_name: dict[str, float] = {}
            for state in joints_states["states"]:
                joint_positions_by_name[state["name"]] = state["motor_position"]

            left_rad = [
                float(joint_positions_by_name.get(n, 0.0)) for n in LEFT_ARM_JOINT_NAMES
            ]
            right_rad = [
                float(joint_positions_by_name.get(n, 0.0)) for n in RIGHT_ARM_JOINT_NAMES
            ]
            left_deg = np.rad2deg(left_rad)
            right_deg = np.rad2deg(right_rad)

            if self.config.dual_arm:
                joint_values_deg = list(left_deg) + [0.0] + list(right_deg) + [0.0]
                # joints_rad = left_rad + [0.0] + right_rad + [0.0]
            elif self.config.use_left_arm:
                joint_values_deg = list(left_deg) + [0.0]
                # joints_rad = left_rad + [0.0]
            else:
                joint_values_deg = list(right_deg) + [0.0]
                # joints_rad = right_rad + [0.0]

            gripper_config = self.config.gripper_config
            if self.config.dual_arm:
                for arm, j_idx in (("left", 7), ("right", 15)):
                    if arm not in gripper_config:
                        continue
                    if gripper_config[arm].get("type", "g2") == "g2":
                        joint_values_deg[j_idx]=self._last_gripper_positions.get(arm)
                    elif gripper_config[arm].get("type", "g2") == "dh":
                        try:
                            gripper = self.grippers.get(arm)
                            if gripper and hasattr(gripper, "get_position"):
                                gcfg = gripper_config[arm]
                                open_pos = gcfg.get("open_position", 1000)
                                close_pos = gcfg.get("close_position", 0)
                                if open_pos != close_pos:
                                    raw = gripper.get_position()
                                    joint_values_deg[j_idx] = float(
                                        (raw - close_pos) / (open_pos - close_pos)
                                    )
                        except Exception as e:
                            logger.warning("获取%s臂夹爪位置失败: %s", arm, e)
            else:
                arm_side = "left" if self.config.use_left_arm else "right"
                if arm_side in gripper_config and gripper_config[arm_side].get("type", "g2") == "dh":
                    joint_values_deg[-1] = self._last_gripper_positions.get(arm_side)
                else:
                    try:
                        gripper = self.grippers.get(arm_side)
                        if gripper and hasattr(gripper, "get_position"):
                            gcfg = gripper_config[arm_side]
                            open_pos = gcfg.get("open_position", 1000)
                            close_pos = gcfg.get("close_position", 0)
                            if open_pos != close_pos:
                                raw = gripper.get_position()
                                joint_values_deg[-1] = float(
                                    (raw - close_pos) / (open_pos - close_pos)
                                )
                    except Exception as e:
                        logger.warning("获取%s臂夹爪位置失败: %s", arm_side, e)
            logger.info(f"joint_values_deg' len: {len(joint_values_deg)}")
            for k, v in zip(joint_keys, joint_values_deg, strict=True):
                obs[k] = float(v)
                # 更新当前关节角度存储
                if k in self._current_joint_positions:
                    self._current_joint_positions[k] = float(v)

        except Exception as e:
            logger.warning("获取关节状态失败: %s", e)
            for k in joint_keys:
                obs[k] = 0.0

        try:
            current_poses = self.get_end_effector_pose()
            if self.config.dual_arm:
                if len(current_poses) >= 2:
                    lp, rp = current_poses[0], current_poses[1]
                    lq = np.array(
                        [
                            float(lp.orientation.x),
                            float(lp.orientation.y),
                            float(lp.orientation.z),
                            float(lp.orientation.w),
                        ]
                    )
                    rq = np.array(
                        [
                            float(rp.orientation.x),
                            float(rp.orientation.y),
                            float(rp.orientation.z),
                            float(rp.orientation.w),
                        ]
                    )
                    lv = Rotation.from_quat(lq).as_rotvec()
                    rv = Rotation.from_quat(rq).as_rotvec()
                    ee_vals = [
                        float(lp.position.x),
                        float(lp.position.y),
                        float(lp.position.z),
                        float(lv[0]),
                        float(lv[1]),
                        float(lv[2]),
                        0.0, # TODO：add gripper values later
                        float(rp.position.x),
                        float(rp.position.y),
                        float(rp.position.z),
                        float(rv[0]),
                        float(rv[1]),
                        float(rv[2]),
                        0.0,
                    ]
                else:
                    ee_vals = [0.0] * len(ee_keys)
            elif len(current_poses) >= 2:
                p = current_poses[0] if self.config.use_left_arm else current_poses[1]
                q = np.array(
                    [float(p.orientation.x), float(p.orientation.y), float(p.orientation.z), float(p.orientation.w)]
                )
                v = Rotation.from_quat(q).as_rotvec()
                ee_vals = [
                    float(p.position.x),
                    float(p.position.y),
                    float(p.position.z),
                    float(v[0]),
                    float(v[1]),
                    float(v[2]),
                    0.0,
                ]
            else:
                ee_vals = [0.0] * len(ee_keys)
            logger.info(f"current_poses:{ee_vals}")
            for k, v in zip(ee_keys, ee_vals, strict=True):
                obs[k] = float(v)
        except Exception as e:
            logger.warning("获取末端执行器位姿失败: %s", e)
            for k in ee_keys:
                obs[k] = 0.0

        for cam_name in self._enabled_camera_names():
            with self._locks[cam_name]:
                frame = self._frames[cam_name]
            if frame is not None:
                # save as [0,1] tensor, shape (3, H, W)
                to_tensor = transforms.ToTensor()
                obs[cam_name] = to_tensor(frame)
                
            elif cam_name in self._cameras_ft:
                d = self._cameras_ft[cam_name]
                obs[cam_name] = torch.zeros(d, dtype=torch.float32)
            else:
                obs[cam_name] = torch.zeros((3, 480, 640), dtype=torch.float32)

        return obs

    def get_end_effector_pose(self):
        while (status := self.robot.get_motion_control_status()) is None:
            pass
        return status.frame_poses

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send joint actions to robot via GDK joint control and gripper interfaces.
        
        Action format includes joint positions (e.g., 'r.joint1.pos': -99.12) 
        and gripper positions (e.g., 'r.gripper.pos': 0.0).
        
        Includes activation check for joint control.
        """
        logger.info(f"Sending action: {action}")

        if not self._is_connected:
            logger.warning("Robot not connected, skipping action")
            return action
        
        # # Get activation status for joint control
        left_active, right_active = self._get_joint_activation(action)
        
        try:
            # Control joints based on action (only if activated)
            if left_active or right_active:
                self._control_joints(action)
            
            # Control grippers based on action
            self._control_grippers_from_action(action)
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")
        
        return action

    def _control_joints(self, action: RobotAction) -> None:
        """Control robot joints based on action dictionary.
        
        Action keys expected: '{l|r}.joint{1-7}.pos' (degrees)
        Converts to radians and sends via GDK joint_control_request.
        """
        if not self._agibot_gdk or not self.robot:
            logger.warning("GDK not initialized, skipping joint control")
            return
        
        # Map joint names from action keys to GDK joint names
        joint_mapping = {
            'l.joint1.pos': 'idx21_arm_l_joint1',
            'l.joint2.pos': 'idx22_arm_l_joint2',
            'l.joint3.pos': 'idx23_arm_l_joint3',
            'l.joint4.pos': 'idx24_arm_l_joint4',
            'l.joint5.pos': 'idx25_arm_l_joint5',
            'l.joint6.pos': 'idx26_arm_l_joint6',
            'l.joint7.pos': 'idx27_arm_l_joint7',
            'r.joint1.pos': 'idx61_arm_r_joint1',
            'r.joint2.pos': 'idx62_arm_r_joint2',
            'r.joint3.pos': 'idx63_arm_r_joint3',
            'r.joint4.pos': 'idx64_arm_r_joint4',
            'r.joint5.pos': 'idx65_arm_r_joint5',
            'r.joint6.pos': 'idx66_arm_r_joint6',
            'r.joint7.pos': 'idx67_arm_r_joint7',
        }
        
        # Collect joints to control based on configuration
        joints_to_control = []
        
        if self.config.dual_arm:
            # Dual-arm: control both arms if joints present in action
            arm_prefixes = ['l', 'r']
        else:
            # Single-arm: control only configured arm
            arm_prefixes = [self.config.single_arm_prefix]
        
        for prefix in arm_prefixes:
            for joint_num in range(1, 8):
                key = f"{prefix}.joint{joint_num}.pos"
                if key in action:
                    gdk_joint_name = joint_mapping.get(key)
                    if gdk_joint_name:
                        joints_to_control.append((gdk_joint_name, action[key]))
        
        if not joints_to_control:
            logger.debug("No joint positions found in action")
            return
        
        try:
            # Create joint control request
            joint_control_req = self._agibot_gdk.JointControlReq()
            joint_control_req.joint_names = [name for name, _ in joints_to_control]
            
            # Convert degrees to radians for GDK
            joint_control_req.joint_positions = [
                np.deg2rad(pos) for _, pos in joints_to_control
            ]
            
            # Set default velocity (0.1 rad/s) and lifetime (5.0s)
            joint_control_req.joint_velocities = [0.4] * len(joints_to_control)
            joint_control_req.life_time = 0.01
            joint_control_req.detail = "Joint control from LeRobot"
            
            # Send control request
            result = self.robot.joint_control_request(joint_control_req)
            logger.info(f"Joint control sent for {len(joints_to_control)} joints, result: {result}")
            
        except Exception as e:
            logger.error(f"Joint control failed: {e}")

    def _get_joint_activation(self, action: RobotAction) -> tuple[bool, bool]:
        """Get activation status for joint control by comparing target and current positions.
        
        An arm is considered active if any of its joints has a target position
        that differs from the current position by more than the activation threshold.
        """
        left_active = False
        right_active = False
        
        # Check left arm joints
        for joint_num in range(1, 8):
            key = f"l.joint{joint_num}.pos"
            if key in action:
                target_pos = action[key]
                current_pos = self._current_joint_positions.get(key, 0.0)
                
                # Check if the difference exceeds threshold
                if abs(target_pos - current_pos) > self.joint_activation_threshold:
                    left_active = True
                    break
        
        # Check right arm joints
        for joint_num in range(1, 8):
            key = f"r.joint{joint_num}.pos"
            if key in action:
                target_pos = action[key]
                current_pos = self._current_joint_positions.get(key, 0.0)
                
                # Check if the difference exceeds threshold
                if abs(target_pos - current_pos) > self.joint_activation_threshold:
                    right_active = True
                    break
        
        # Apply configuration constraints
        if not self.config.dual_arm:
            if self.config.use_left_arm:
                right_active = False  # Single left arm mode
            else:
                left_active = False   # Single right arm mode
        
        logger.debug(f"Joint activation: left={left_active}, right={right_active}")
        return left_active, right_active

    def _control_grippers_from_action(self, action: RobotAction) -> None:
        """Control grippers based on action dictionary.
        
        Action keys expected: '{l|r}.gripper.pos' (normalized 0.0-1.0)
        Adapts to existing _control_grippers method interface.
        """
        if not self.config.use_gripper:
            logger.debug("Gripper control disabled in config")
            return
        
        # Convert action format to match _control_grippers expected format
        gripper_action = {}
        
        for arm in ['left', 'right']:
            prefix = 'l' if arm == 'left' else 'r'
            key = f"{prefix}.gripper.pos"
            
            if key in action:
                # Map to expected format: left_gripper, right_gripper
                gripper_action[f"{arm}_gripper"] = action[key]
        
        if gripper_action:
            # Use existing gripper control logic
            self._control_grippers(gripper_action)
        else:
            logger.debug("No gripper positions found in action")

    
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
            if last_position is not None and abs(gripper_value - last_position) < GRIPPER_COMMAND_MIN_DELTA:
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
        """Calibrate the robot and set gripper positions to 0 (fully open)."""
        try:
            # Set calibration flag
            self._is_calibrated = True
            
            # Set gripper positions to 0 (fully open) for both arms
            if self._is_connected and self.robot:
                logger.info("Setting gripper positions to 0 (fully open) after calibration")
                
                # Create action to set both grippers to position 0
                action = {
                    "left_ee_state": {
                        "joint_position": 0.0,    # Left gripper fully open
                    },
                    "right_ee_state": {
                        "joint_position": 0.0,    # Right gripper fully open
                    }
                }
                
                # Send gripper control command
                result = self.robot.move_ee_pos(action)
                logger.info(f"Gripper calibration command sent, result: {result}")
                
                # Update last gripper positions
                self._last_gripper_positions["left"] = 0.0
                self._last_gripper_positions["right"] = 0.0
            else:
                logger.info("Robot not connected or gripper disabled, skipping gripper calibration")
            
            logger.info("Robot calibration completed")
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            # Still mark as calibrated even if gripper control fails
            self._is_calibrated = True

    def reset(self) -> None:
        """每回合开始清理末端参考位姿与臂激活状态。关节回零目标见 config.default_positions（待 GDK 关节接口对接后实现插值运动）。"""
        logger.info("Resetting robot state")
        for key in self._current_joint_positions:
            self._current_joint_positions[key] = 0.0     
        _ = self._resolve_default_joint_positions()
        logger.info("Robot state reset completed")

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
        delta_keys = ["l.ee.x", "r.ee.x", "l.ee.wx", "r.ee.wx"]
        for key in delta_keys:
            if key in action:
                return True
        
        # 检查单臂模式的前缀
        prefix = self.config.single_arm_prefix
        single_arm_keys = [f"{prefix}.ee.x", f"{prefix}.ee.wx"]
        for key in single_arm_keys:
            if key in action:
                return True
        
        return False
    
    def _get_dual_arm_activation(self, action: dict[str, Any]) -> tuple[bool, bool]:
        """获取双臂模式的激活状态（6DoF轴角表示）"""
        # 从动作中提取左臂delta值（6DoF：平移3 + 旋转3）
        left_delta_values = [
            action.get("l.ee.x", 0.0),
            action.get("l.ee.y", 0.0),
            action.get("l.ee.z", 0.0),
            action.get("l.ee.wx", 0.0),
            action.get("l.ee.wy", 0.0),
            action.get("l.ee.wz", 0.0)
        ]
        
        # 从动作中提取右臂delta值（6DoF：平移3 + 旋转3）
        right_delta_values = [
            action.get("r.ee.x", 0.0),
            action.get("r.ee.y", 0.0),
            action.get("r.ee.z", 0.0),
            action.get("r.ee.wx", 0.0),
            action.get("r.ee.wy", 0.0),
            action.get("r.ee.wz", 0.0)
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
            action.get(f"{prefix}.ee.x", 0.0),
            action.get(f"{prefix}.ee.y", 0.0),
            action.get(f"{prefix}.ee.z", 0.0),
            action.get(f"{prefix}.ee.wx", 0.0),
            action.get(f"{prefix}.ee.wy", 0.0),
            action.get(f"{prefix}.ee.wz", 0.0)
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
    