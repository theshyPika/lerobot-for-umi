import agibot_gdk
import time
# 初始化 GDK 系统
if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
    print("GDK 初始化失败")
    exit(1)
print("GDK 初始化成功")
robot = agibot_gdk.Robot()
time.sleep(2) # 等待机器人初始化

def control_gripper():
    # 控制夹爪开合
    action = {
    # "left_ee_state": {
    # "joint_position": 0.5,
    # },
    "right_ee_state": {
    "joint_position": 0.5,
    }
    }
    # 左手夹爪位置
    # 右手夹爪位置
    try:
        result = robot.move_ee_pos(action)
        print("夹爪控制成功")
    except Exception as e:
        print(f"夹爪控制失败: {e}")

def control_gripper2():
    # # 控制左夹爪（omnipicker类型，需要1个关节）
    joint_states_right = agibot_gdk.JointStates()
    # joint_states_right.group = "right_tool"
    # joint_states_right.target_type = "omnipicker"

    joint_state = agibot_gdk.JointState()
    joint_state.position = 0  # 取值范围 [0, 1]  
    joint_states_right.states = [joint_state]
    joint_states_right.nums = len(joint_states_right.states)

    try:
        result = robot.move_ee_pos(joint_states_right)
        print("右夹爪控制成功")
    except Exception as e:
        print(f"右夹爪控制失败: {e}")

def control_arm():
    # 获取末端执行器状态
    while(status := robot.get_motion_control_status()) is None:
        continue
    current_poses = status.frame_poses
    print(f"{status.frame_poses}")

    target_pose = agibot_gdk.EndEffectorPose()
    target_pose.group = agibot_gdk.EndEffectorControlGroup.kRightArm
    # 获取delta值（瞬时增量）- 6DoF轴角表示
    delta_x = 0.0
    delta_y = 0.0
    delta_z = 0.01

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
    target_pose.life_time = 0.01
    success = robot.end_effector_pose_control(target_pose)          


if __name__ == "__main__":
    control_gripper()
    # control_gripper2()
    control_arm()