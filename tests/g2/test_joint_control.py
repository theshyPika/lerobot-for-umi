import agibot_gdk
import time

# 控制参数
CONTROL_PERIOD = 0.01  # 控制周期（秒）
RATE_HZ = 100.0        # 发送频率（Hz）
DURATION = 3.0         # 单次移动持续时间（秒）
HOLD_DURATION = 0.5    # 在 0 或 1 处保持的时间（秒）
NUM_CYCLES = 3         # 往复运动次数

# 左臂 7 个关节名称（与 get_joint_states 中一致）
ARM_L_JOINT_NAMES = [
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
]

def get_arm_positions_by_name(joint_states, joint_names):
    """从 get_joint_states 的返回中按关节名取出位置列表"""
    name_to_pos = {s["name"]: s["motor_position"] for s in joint_states["states"]}
    return [name_to_pos[name] for name in joint_names]

def get_ee_names_and_positions(end_state, side="left"):
    """从 get_end_state 的返回中取出指定侧末端的关节名列表和位置列表"""
    key = f"{side}_end_state"
    if key not in end_state:
        raise RuntimeError(f"get_end_state 中未找到 {key}")
    state = end_state[key]
    names = state.get("names", [])
    positions = [s["position"] for s in state.get("end_states", [])]
    if len(names) != len(positions):
        raise RuntimeError(f"{key} 中 names 与 end_states 长度不一致")
    return names, positions

def main():
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK初始化失败")
        return
    print("GDK初始化成功")

    try:
        robot = agibot_gdk.Robot()
        time.sleep(2)

        # 1) 从 get_joint_states 获取全身关节（含机械臂）
        joint_states = robot.get_joint_states()
        # 2) 从 get_end_state 获取末端关节名与当前位置（如左夹爪/左手）
        end_state = robot.get_end_state()
        ee_names, ee_positions = get_ee_names_and_positions(end_state, "left")

        # 合并：机械臂关节 + 末端关节
        all_names = ARM_L_JOINT_NAMES + ee_names
        arm_positions = get_arm_positions_by_name(joint_states, ARM_L_JOINT_NAMES)

        # 机械臂小幅运动，末端关节在 -0.785 和 0 之间往复运动(omnipicker)
        arm_start = arm_positions[:]                           # 机械臂初始姿态
        arm_target = [p + 0.05 for p in arm_positions]         # 机械臂目标姿态（小幅偏移）
        ee_low = -0.785                                        # 末端下限
        ee_high = 0.0                                          # 末端上限

        print(f"控制关节数: 机械臂 {len(ARM_L_JOINT_NAMES)} + 末端 {len(ee_names)} = {len(all_names)}")
        print(f"机械臂将在关节空间 arm_start ↔ arm_target 之间运动，末端关节在 {ee_low} 和 {ee_high} 之间往复 {NUM_CYCLES} 次")
        
        n_steps = int(DURATION * RATE_HZ)           # 单次移动的步数
        hold_steps = int(HOLD_DURATION * RATE_HZ)   # 保持阶段的步数
        dt = 1.0 / RATE_HZ

        def send_control_command(arm_target_values, ee_target_values):
            """发送控制命令的辅助函数"""
            current = arm_target_values + ee_target_values
            req = agibot_gdk.JointServoControlReq()
            req.control_period = CONTROL_PERIOD
            req.joint_names = all_names
            req.joint_positions = current
            return robot.joint_servo_control(req)

        for cycle in range(NUM_CYCLES):
            print(f"\n=== 周期 {cycle+1}/{NUM_CYCLES}: 机械臂 arm_start -> arm_target, 末端 {ee_low} -> {ee_high} ===")
            start_time = time.time()
            
            # 第一段：机械臂 arm_start -> arm_target，末端 ee_low -> ee_high
            for i in range(n_steps):
                t = float(i) / (n_steps - 1) if n_steps > 1 else 1.0
                current_arm = [
                    arm_start[j] + t * (arm_target[j] - arm_start[j])
                    for j in range(len(ARM_L_JOINT_NAMES))
                ]
                current_ee = [ee_low + t * (ee_high - ee_low) for _ in ee_names]
                
                result = send_control_command(current_arm, current_ee)
                if result != 0:
                    print(f"发送失败，周期={cycle+1}, 阶段=0->1, 步数={i}")
                    raise RuntimeError("joint_servo_control failed")
                
                elapsed = time.time() - start_time
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 第二段：在 arm_target / ee_high 处保持
            print(f"=== 周期 {cycle+1}/{NUM_CYCLES}: 在 arm_target / {ee_high} 处保持 {HOLD_DURATION} 秒 ===")
            start_time = time.time()
            current_arm = arm_target[:]
            current_ee = [ee_high for _ in ee_names]
            
            for i in range(hold_steps):
                result = send_control_command(current_arm, current_ee)
                if result != 0:
                    print(f"发送失败，周期={cycle+1}, 阶段=保持@1, 步数={i}")
                    raise RuntimeError("joint_servo_control failed")
                
                elapsed = time.time() - start_time
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 第三段：机械臂 arm_target -> arm_start，末端 ee_high -> ee_low
            print(f"=== 周期 {cycle+1}/{NUM_CYCLES}: 机械臂 arm_target -> arm_start, 末端 {ee_high} -> {ee_low} ===")
            start_time = time.time()
            
            for i in range(n_steps):
                t = float(i) / (n_steps - 1) if n_steps > 1 else 1.0
                current_arm = [
                    arm_target[j] + t * (arm_start[j] - arm_target[j])
                    for j in range(len(ARM_L_JOINT_NAMES))
                ]
                current_ee = [ee_high + t * (ee_low - ee_high) for _ in ee_names]
                
                result = send_control_command(current_arm, current_ee)
                if result != 0:
                    print(f"发送失败，周期={cycle+1}, 阶段=1->0, 步数={i}")
                    raise RuntimeError("joint_servo_control failed")
                
                elapsed = time.time() - start_time
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # 第四段：在 arm_start / ee_low 处保持
            print(f"=== 周期 {cycle+1}/{NUM_CYCLES}: 在 arm_start / {ee_low} 处保持 {HOLD_DURATION} 秒 ===")
            start_time = time.time()
            current_arm = arm_start[:]
            current_ee = [ee_low for _ in ee_names]
            
            for i in range(hold_steps):
                result = send_control_command(current_arm, current_ee)
                if result != 0:
                    print(f"发送失败，周期={cycle+1}, 阶段=保持@0, 步数={i}")
                    raise RuntimeError("joint_servo_control failed")
                
                elapsed = time.time() - start_time
                sleep_time = (i + 1) * dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"\n机械臂 arm_start↔arm_target、末端 {ee_low}↔{ee_high} 往复控制结束")
    except Exception as e:
        print(f"执行错误: {e}")
    finally:
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            print("GDK释放失败")
        else:
            print("GDK释放成功")

if __name__ == "__main__":
    main()
