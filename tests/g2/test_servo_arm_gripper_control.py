# tests/g2/test_servo_with_move_ee.py
import agibot_gdk
import time
import math
import sys

# —— 官方 servo_control.py 的 22 关节列表（用于上电使能） ——
INIT_JOINT_NAMES = [
    "idx01_body_joint1", "idx02_body_joint2", "idx03_body_joint3",
    "idx04_body_joint4", "idx05_body_joint5",
    "idx11_head_joint1", "idx13_head_joint3", "idx12_head_joint2",
    "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
    "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
    "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

# omnipicker: -0.785 = 开, 0.0 = 关
GRIPPER_OPEN   = -0.785
GRIPPER_CLOSED =  0.0

GRIPPER_TOGGLE_PERIOD = 1.5   # 开/关切换周期 (秒)
ARM_SEND_HZ           = 50.0  # servo_control_arm_pos 下发频率
GRIPPER_SEND_HZ       = 10.0  # move_ee_pos 下发频率
DURATION              = 8.0   # 总运行时间; 改成 None 则无限循环 (Ctrl+C 退出)


def make_dual_omnipicker_req(left_pos: float, right_pos: float):
    """构造 dual_tool / omnipicker 的 move_ee_pos 请求 (左+右一次发)."""
    js_l = agibot_gdk.JointState(); js_l.position = float(left_pos)
    js_r = agibot_gdk.JointState(); js_r.position = float(right_pos)
    req = agibot_gdk.JointStates()
    req.group = "dual_tool"
    req.target_type = "omnipicker"
    req.states = [js_l, js_r]
    req.nums = len(req.states)
    return req


def main():
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败")
        sys.exit(1)
    print("GDK 初始化成功")
    robot = agibot_gdk.Robot()
    time.sleep(1)

    # —— 与官方示例完全一致的初始化：拉到零位 (有运动风险, 请确认空间安全) ——
    init_req = agibot_gdk.JointControlReq()
    init_req.life_time = 0.1
    init_req.joint_names = INIT_JOINT_NAMES
    init_req.joint_positions = [0.0] * len(INIT_JOINT_NAMES)
    init_req.joint_velocities = [0.3] * len(INIT_JOINT_NAMES)
    robot.joint_control_request(init_req)
    time.sleep(1)

    # —— 读当前 14 个臂关节作为正弦基线 (照搬官方逻辑) ——
    joint_states = robot.get_joint_states()
    joint_positions = [s["motor_position"] for s in joint_states["states"]]
    base = joint_positions[8:22]  # 索引 8-21 对应 14 个臂关节

    # —— 控制循环参数 ——
    t = 0.0
    dt = 1.0 / ARM_SEND_HZ   # 50 Hz arm
    freq = 0.5
    omega = 2 * math.pi * freq
    amp = 0.087
    phases = [i * 0.2 for i in range(14)]
    fade_duration = 1.0

    gripper_send_interval = 1.0 / GRIPPER_SEND_HZ
    last_gripper_send_t = -1.0

    arm_calls = 0
    gripper_calls = 0
    gripper_errors = 0

    print(
        f"开始: 臂 {ARM_SEND_HZ:.0f}Hz 正弦扫动 + 夹爪 {GRIPPER_SEND_HZ:.0f}Hz dual_tool/omnipicker, "
        f"开/关切换周期 {GRIPPER_TOGGLE_PERIOD}s, 左右反相"
    )

    try:
        while True:
            # 1) 臂伺服 (官方示例的扫动)
            fade = min(1.0, t / fade_duration)
            arm_positions = [
                base[i] + (fade * amp) * math.sin(omega * t + phases[i])
                for i in range(14)
            ]
            robot.servo_control_arm_pos(arm_positions, 2)
            arm_calls += 1

            # 2) 夹爪 (独立通道, 节流)
            if t - last_gripper_send_t >= gripper_send_interval:
                # 在 [0, PERIOD) 内左开右关; 在 [PERIOD, 2*PERIOD) 内左关右开
                phase = (t % (2 * GRIPPER_TOGGLE_PERIOD)) < GRIPPER_TOGGLE_PERIOD
                left_target  = GRIPPER_OPEN   if phase else GRIPPER_CLOSED
                right_target = GRIPPER_CLOSED if phase else GRIPPER_OPEN
                try:
                    robot.move_ee_pos(make_dual_omnipicker_req(left_target, right_target))
                    gripper_calls += 1
                except Exception as e:
                    gripper_errors += 1
                    if gripper_errors <= 3:
                        print(f"move_ee_pos 失败 (t={t:.2f}s): {e}")
                last_gripper_send_t = t

            t += dt
            time.sleep(dt)

            if DURATION is not None and t >= DURATION:
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # —— 末态读回, 直接看夹爪到位情况 ——
        try:
            end_state = robot.get_end_state()
            for side in ("left_end_state", "right_end_state"):
                if side in end_state:
                    st = end_state[side]
                    names = st.get("names")
                    positions = [s["position"] for s in st.get("end_states", [])]
                    print(f"末态 {side}: names={names}, positions={positions}")
        except Exception as e:
            print(f"读末态失败: {e}")

        print(
            f"统计: 臂调用 {arm_calls} 次, 夹爪调用 {gripper_calls} 次, "
            f"夹爪异常 {gripper_errors} 次"
        )
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            print("GDK 释放失败")
        else:
            print("GDK 释放成功")


if __name__ == "__main__":
    main()
