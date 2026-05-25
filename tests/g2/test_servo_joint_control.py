# tests/g2/test_servo_joint_control.py
#
# 使用 joint_servo_control 单一接口同时控制 双臂 + 双夹爪 (omnipicker).
# 运动逻辑与 test_servo_arm_gripper_control.py 一致, 便于对照:
#   - 双臂: 围绕当前位置做 0.5Hz 正弦扫动 (amp=0.087rad), 带 fade-in
#   - 双夹爪: 周期 GRIPPER_TOGGLE_PERIOD 秒在 [-0.785, 0] 之间切换, 左右反相
import agibot_gdk
import math
import sys
import time

# —— 官方 servo_control.py 的 22 关节列表 (用于上电使能) ——
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

ARM_L_JOINT_NAMES = [n for n in INIT_JOINT_NAMES if n.startswith("idx2")]
ARM_R_JOINT_NAMES = [n for n in INIT_JOINT_NAMES if n.startswith("idx6")]

# omnipicker: -0.785 = 开, 0.0 = 关
GRIPPER_OPEN   = -0.785
GRIPPER_CLOSED =  0.0

GRIPPER_TOGGLE_PERIOD = 1.5    # 开/关切换周期 (秒)
SERVO_SEND_HZ         = 10.0   # joint_servo_control 下发频率
CONTROL_PERIOD        = 0.1  # 略大于 1/SERVO_SEND_HZ
DURATION              = 8.0    # 总运行时间; None 则无限循环 (Ctrl+C 退出)


def get_ee_names(end_state: dict, side: str) -> list[str]:
    key = f"{side}_end_state"
    if key not in end_state:
        raise RuntimeError(f"get_end_state 中未找到 {key}")
    return list(end_state[key].get("names", []))


def main():
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败")
        sys.exit(1)
    print("GDK 初始化成功")
    robot = agibot_gdk.Robot()
    time.sleep(1)

    # —— 上电使能: joint_control_request 保持当前位置, 避免任何运动 ——
    joint_states = robot.get_joint_states()
    name_to_pos = {s["name"]: s["motor_position"] for s in joint_states["states"]}
    init_positions = [name_to_pos[n] for n in INIT_JOINT_NAMES]

    init_req = agibot_gdk.JointControlReq()
    init_req.life_time = 0.1
    init_req.joint_names = INIT_JOINT_NAMES
    init_req.joint_positions = init_positions
    init_req.joint_velocities = [0.3] * len(INIT_JOINT_NAMES)
    robot.joint_control_request(init_req)
    time.sleep(1)

    # —— 读 14 个臂关节作为正弦基线 ——
    arm_l_base = [name_to_pos[n] for n in ARM_L_JOINT_NAMES]
    arm_r_base = [name_to_pos[n] for n in ARM_R_JOINT_NAMES]
    base = arm_l_base + arm_r_base  # 长度 14, 顺序与 servo 请求中的 joint_names 对齐

    # —— 从 get_end_state 拿到双夹爪关节名 (omnipicker 各 1 个) ——
    end_state = robot.get_end_state()
    left_ee_names  = get_ee_names(end_state, "left")
    right_ee_names = get_ee_names(end_state, "right")
    if not left_ee_names or not right_ee_names:
        raise RuntimeError(
            f"末端关节名缺失: left={left_ee_names}, right={right_ee_names}"
        )

    # —— 单次 joint_servo_control 请求的关节名列表 ——
    servo_names = ARM_L_JOINT_NAMES + ARM_R_JOINT_NAMES + left_ee_names + right_ee_names
    n_arms = 14
    n_left_ee = len(left_ee_names)
    n_right_ee = len(right_ee_names)

    # —— 控制循环参数 ——
    dt = 1.0 / SERVO_SEND_HZ
    freq = 0.5
    omega = 2 * math.pi * freq
    amp = 0.087
    phases = [i * 0.2 for i in range(n_arms)]
    fade_duration = 1.0

    servo_calls = 0
    servo_errors = 0

    print(
        f"开始: joint_servo_control {SERVO_SEND_HZ:.0f}Hz 单请求控制, "
        f"臂 {n_arms} 关节 + 左夹爪 {n_left_ee} + 右夹爪 {n_right_ee} = {len(servo_names)} 关节"
    )
    print(f"夹爪开/关切换周期 {GRIPPER_TOGGLE_PERIOD}s, 左右反相")

    req = agibot_gdk.JointServoControlReq()
    req.control_period = CONTROL_PERIOD
    req.joint_names = servo_names

    t = 0.0
    t0 = time.time()
    try:
        while True:
            # 1) 臂正弦扫动
            fade = min(1.0, t / fade_duration)
            arm_positions = [
                base[i] + (fade * amp) * math.sin(omega * t + phases[i])
                for i in range(n_arms)
            ]
            # 2) 夹爪 phase: [0, PERIOD) 左开右关, [PERIOD, 2*PERIOD) 左关右开
            phase = (t % (2 * GRIPPER_TOGGLE_PERIOD)) < GRIPPER_TOGGLE_PERIOD
            left_target  = GRIPPER_OPEN   if phase else GRIPPER_CLOSED
            right_target = GRIPPER_CLOSED if phase else GRIPPER_OPEN

            req.joint_positions = (
                arm_positions
                + [left_target] * n_left_ee
                + [right_target] * n_right_ee
            )
            try:
                result = robot.joint_servo_control(req)
                if result != 0:
                    servo_errors += 1
                    if servo_errors <= 3:
                        print(f"joint_servo_control 返回 {result} (t={t:.2f}s)")
                else:
                    servo_calls += 1
            except Exception as e:
                servo_errors += 1
                if servo_errors <= 3:
                    print(f"joint_servo_control 异常 (t={t:.2f}s): {e}")

            # 节拍对齐 (基于 wall clock, 避免漂移)
            t += dt
            target_wall = t0 + t
            sleep_t = target_wall - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

            if DURATION is not None and t >= DURATION:
                break
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # —— 末态读回, 看夹爪是否真的到位 ——
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
            f"统计: joint_servo_control 成功 {servo_calls} 次, 异常 {servo_errors} 次"
        )
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            print("GDK 释放失败")
        else:
            print("GDK 释放成功")


if __name__ == "__main__":
    main()
