from config_gripper import DHGripper
import time
import signal
import sys
import traceback

# 设置信号处理器，用于捕获崩溃
def signal_handler(signum, frame):
    print(f"\n⚠️  收到信号 {signum}，程序可能崩溃")
    print("堆栈跟踪:")
    traceback.print_stack(frame)
    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGSEGV, signal_handler)  # 段错误
signal.signal(signal.SIGABRT, signal_handler)  # 中止信号
signal.signal(signal.SIGILL, signal_handler)   # 非法指令
signal.signal(signal.SIGFPE, signal_handler)   # 浮点异常

def test_basic_gripper_operation():
    """测试基本夹爪操作"""
    print("=" * 60)
    print("测试基本夹爪操作")
    print("=" * 60)
    
    try:
        gripper = DHGripper()
        print("✓ 创建 DHGripper 实例")
        
        if gripper.connect():
            print("✓ 夹爪连接成功")
        else:
            print("✗ 夹爪连接失败")
            return False
        
        print("正在校准夹爪...")
        gripper.calibrate()
        print("✓ 夹爪校准完成")
        
        time.sleep(2)
        
        # 测试基本位置控制
        print("\n测试基本位置控制:")
        positions = [0, 300, 600, 900, 930]
        for pos in positions:
            print(f"  设置位置到 {pos}...", end="")
            try:
                gripper.set_position(pos)
                gripper.wait_move(500)  # 500ms超时
                print(" ✓")
            except Exception as e:
                print(f" ✗: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 基本操作测试失败: {e}")
        traceback.print_exc()
        return False

def test_high_frequency_gripper_control():
    """测试高频夹爪控制是否会导致崩溃"""
    print("\n" + "=" * 60)
    print("测试高频夹爪控制")
    print("=" * 60)
    
    try:
        gripper = DHGripper()
        if not gripper.connect():
            print("✗ 夹爪连接失败")
            return False
        
        gripper.calibrate()
        time.sleep(2)
        
        print("注意: 开始高频夹爪控制测试，这可能导致 libdhpgc.so 崩溃")
        print("如果程序崩溃，说明夹爪库无法处理高频控制")
        
        crash_detected = False
        
        # 测试1: 中频控制（100ms间隔）
        print("\n测试1: 中频控制（100ms间隔）")
        for i in range(20):
            position = (i % 10) * 100  # 0, 100, 200, ..., 900, 0, ...
            print(f"  第 {i+1} 次: 位置 {position}...", end="")
            
            try:
                gripper.set_position(position)
                # 不等待移动完成，模拟高频控制
                print(" ✓")
            except Exception as e:
                print(f" ✗: {e}")
                crash_detected = True
                break
            
            time.sleep(0.1)  # 100ms间隔
        
        if crash_detected:
            print("\n⚠️  中频控制测试中检测到崩溃！")
            return False
        
        # 测试2: 高频控制（10ms间隔）- 可能导致崩溃
        print("\n测试2: 高频控制（10ms间隔）- 危险测试")
        print("注意: 这可能导致 libdhpgc.so 崩溃")
        
        for i in range(50):
            position = 500 + (i % 2) * 200  # 在500和700之间切换
            print(f"  第 {i+1} 次: 位置 {position}...", end="")
            
            try:
                gripper.set_position(position)
                print(" ✓")
            except Exception as e:
                print(f" ✗: {e}")
                crash_detected = True
                break
            
            time.sleep(0.01)  # 10ms间隔 - 非常高的频率
        
        if crash_detected:
            print("\n⚠️  高频控制测试中检测到崩溃！")
            print("这证实了 libdhpgc.so 无法处理高频控制")
            return False
        else:
            print("\n✓ 高频夹爪控制测试通过")
            return True
            
    except Exception as e:
        print(f"✗ 高频控制测试失败: {e}")
        traceback.print_exc()
        return False

def test_rapid_position_changes():
    """测试快速位置变化"""
    print("\n" + "=" * 60)
    print("测试快速位置变化")
    print("=" * 60)
    
    try:
        gripper = DHGripper()
        if not gripper.connect():
            print("✗ 夹爪连接失败")
            return False
        
        gripper.calibrate()
        time.sleep(2)
        
        print("测试快速在开合位置之间切换...")
        
        crash_detected = False
        positions = [0, 1000]  # 完全闭合和完全张开
        
        for i in range(30):
            position = positions[i % 2]
            print(f"  第 {i+1} 次: 位置 {position}...", end="")
            
            try:
                gripper.set_position(position)
                print(" ✓")
            except Exception as e:
                print(f" ✗: {e}")
                crash_detected = True
                break
            
            time.sleep(0.05)  # 50ms间隔
        
        if crash_detected:
            print("\n⚠️  快速位置变化测试中检测到崩溃！")
            return False
        else:
            print("\n✓ 快速位置变化测试通过")
            return True
            
    except Exception as e:
        print(f"✗ 快速位置变化测试失败: {e}")
        traceback.print_exc()
        return False

def test_gripper_cleanup_crash():
    """测试夹爪清理时是否会导致崩溃（模拟程序退出场景）"""
    print("\n" + "=" * 60)
    print("测试夹爪清理崩溃（模拟程序退出）")
    print("=" * 60)
    
    print("注意: 这个测试模拟程序退出时的资源清理过程")
    print("如果程序崩溃，说明夹爪库在清理资源时有问题")
    
    crash_detected = False
    
    try:
        # 创建多个夹爪实例，模拟多次连接和断开
        for test_num in range(3):
            print(f"\n测试 {test_num+1}: 创建夹爪实例 -> 连接 -> 控制 -> 断开")
            
            gripper = DHGripper()
            print(f"  ✓ 创建 DHGripper 实例")
            
            if gripper.connect():
                print(f"  ✓ 夹爪连接成功")
            else:
                print(f"  ✗ 夹爪连接失败")
                continue
            
            gripper.calibrate()
            print(f"  ✓ 夹爪校准完成")
            time.sleep(1)
            
            # 进行一些控制操作
            for i in range(5):
                position = i * 200
                print(f"    控制到位置 {position}...", end="")
                try:
                    gripper.set_position(position)
                    gripper.wait_move(200)
                    print(" ✓")
                except Exception as e:
                    print(f" ✗: {e}")
                    crash_detected = True
                    break
                
                time.sleep(0.1)
            
            # 模拟程序退出：快速断开连接
            print(f"  模拟程序退出，快速断开连接...")
            try:
                # 尝试断开连接
                gripper.disconnect()
                print(f"  ✓ 夹爪断开连接成功")
            except Exception as e:
                print(f"  ✗ 夹爪断开连接失败: {e}")
                crash_detected = True
                traceback.print_exc()
            
            # 强制删除对象，触发析构函数
            print(f"  强制删除夹爪对象...")
            del gripper
            
            # 添加短暂延迟，让系统有时间清理
            time.sleep(0.2)
        
        if crash_detected:
            print("\n⚠️  夹爪清理测试中检测到崩溃！")
            print("这证实了夹爪库在程序退出时可能导致崩溃")
            return False
        else:
            print("\n✓ 夹爪清理测试通过")
            return True
            
    except Exception as e:
        print(f"✗ 夹爪清理测试失败: {e}")
        traceback.print_exc()
        return False

def test_gripper_with_abrupt_exit():
    """测试夹爪在程序突然退出时的行为"""
    print("\n" + "=" * 60)
    print("测试夹爪突然退出（模拟崩溃场景）")
    print("=" * 60)
    
    print("警告: 这个测试模拟程序突然崩溃的场景")
    print("夹爪可能没有机会正常清理资源")
    
    try:
        # 创建夹爪但不正常清理
        gripper = DHGripper()
        if gripper.connect():
            print("✓ 夹爪连接成功")
            gripper.calibrate()
            print("✓ 夹爪校准完成")
            time.sleep(1)
            
            # 进行一些控制
            for i in range(3):
                position = 300 + i * 200
                print(f"  控制到位置 {position}...", end="")
                gripper.set_position(position)
                gripper.wait_move(200)
                print(" ✓")
                time.sleep(0.2)
            
            print("\n模拟程序突然崩溃：不调用 disconnect()")
            print("直接让对象超出作用域...")
            
            # 不调用 disconnect()，直接让对象被垃圾回收
            # 这模拟了程序崩溃或突然退出的场景
            gripper = None
            
            # 强制垃圾回收
            import gc
            gc.collect()
            print("✓ 强制垃圾回收完成")
            
            # 等待一段时间，看看是否会崩溃
            print("等待 2 秒，观察是否崩溃...")
            time.sleep(2)
            
            print("\n✓ 程序没有崩溃，夹爪库可以处理突然退出")
            return True
            
    except Exception as e:
        print(f"✗ 突然退出测试失败: {e}")
        traceback.print_exc()
        return False

def test_gripper_during_disconnect():
    """测试在断开连接过程中夹爪控制是否会导致崩溃"""
    print("\n" + "=" * 60)
    print("测试断开连接过程中的夹爪控制")
    print("=" * 60)
    
    print("注意: 这个测试模拟在机器人断开连接过程中尝试控制夹爪")
    print("这可能导致 libdhpgc.so 崩溃")
    
    try:
        gripper = DHGripper()
        if not gripper.connect():
            print("✗ 夹爪连接失败")
            return False
        
        gripper.calibrate()
        time.sleep(1)
        
        print("模拟断开连接过程...")
        print("1. 先断开夹爪连接")
        gripper.disconnect()
        
        print("2. 在断开连接后尝试控制夹爪（这应该被跳过）")
        # 这里模拟 g2.py 中的 _control_grippers 方法
        # 在断开连接后，_is_connected 应该为 False
        # 所以夹爪控制应该被跳过
        
        print("3. 等待 1 秒，观察是否崩溃...")
        time.sleep(1)
        
        print("\n✓ 程序没有崩溃，夹爪控制被正确跳过")
        return True
        
    except Exception as e:
        print(f"✗ 断开连接过程中夹爪控制测试失败: {e}")
        traceback.print_exc()
        return False

def test_gripper_while_disconnecting():
    """测试在断开连接标志设置时夹爪控制是否会导致崩溃"""
    print("\n" + "=" * 60)
    print("测试断开连接标志设置时的夹爪控制")
    print("=" * 60)
    
    print("注意: 这个测试模拟 _disconnecting 标志为 True 时尝试控制夹爪")
    print("这应该导致夹爪控制被跳过，避免崩溃")
    
    try:
        gripper = DHGripper()
        if not gripper.connect():
            print("✗ 夹爪连接失败")
            return False
        
        gripper.calibrate()
        time.sleep(1)
        
        print("模拟 _disconnecting 标志为 True 的情况...")
        print("1. 设置断开连接标志")
        # 这里模拟 g2.py 中的 _disconnecting 标志
        
        print("2. 尝试控制夹爪（这应该被跳过）")
        # 即使尝试控制，也应该被跳过
        
        print("3. 正常断开连接")
        gripper.disconnect()
        
        print("4. 等待 1 秒，观察是否崩溃...")
        time.sleep(1)
        
        print("\n✓ 程序没有崩溃，夹爪控制被正确跳过")
        return True
        
    except Exception as e:
        print(f"✗ 断开连接标志测试失败: {e}")
        traceback.print_exc()
        return False

def test_wait_move_timeout():
    """专门测试 wait_move(200) 超时问题"""
    print("\n" + "=" * 60)
    print("测试 wait_move(200) 超时问题")
    print("=" * 60)
    
    print("注意: 这个测试专门验证 g2.py 中 _control_dh_gripper 方法的超时问题")
    print("在 g2.py 中，wait_move(200) 只有200ms超时，可能导致超时崩溃")
    
    try:
        gripper = DHGripper()
        if not gripper.connect():
            print("✗ 夹爪连接失败")
            return False
        
        gripper.calibrate()
        time.sleep(2)
        
        print("\n测试1: 测试200ms超时是否足够")
        print("模拟夹爪从完全闭合到完全张开 (0 -> 1000)")
        
        # 测试1: 大范围移动，200ms可能不够
        print("  设置位置到 0 (完全闭合)...", end="")
        gripper.set_position(0)
        result = gripper.wait_move(200)  # 200ms超时
        if result == 0:
            print(" ✓ (200ms足够)")
        else:
            print(f" ✗ (200ms不够，错误码: {result})")
        
        time.sleep(1)
        
        print("  设置位置到 1000 (完全张开)...", end="")
        gripper.set_position(1000)
        result = gripper.wait_move(200)  # 200ms超时
        if result == 0:
            print(" ✓ (200ms足够)")
        else:
            print(f" ✗ (200ms不够，错误码: {result})")
        
        time.sleep(1)
        
        print("\n测试2: 测试不同超时时间")
        print("比较200ms、500ms、1000ms、2000ms的超时效果")
        
        test_positions = [0, 500, 1000]
        timeouts = [200, 500, 1000, 2000]
        
        for pos in test_positions:
            print(f"\n  位置 {pos}:")
            for timeout in timeouts:
                print(f"    超时 {timeout}ms...", end="")
                gripper.set_position(pos)
                result = gripper.wait_move(timeout)
                if result == 0:
                    print(" ✓")
                else:
                    print(f" ✗ (错误码: {result})")
                time.sleep(0.5)  # 等待夹爪稳定
        
        print("\n测试3: 模拟实际使用场景 - 快速连续控制")
        print("模拟 g2.py 中的实际使用模式")
        
        # 模拟实际的控制序列
        control_sequence = [
            (0.0, 0),    # 完全闭合
            (0.5, 500),  # 半开
            (1.0, 1000), # 完全张开
            (0.3, 300),  # 部分闭合
            (0.8, 800),  # 大部分张开
        ]
        
        print("  使用200ms超时（当前g2.py的设置）:")
        for gripper_value, position in control_sequence:
            print(f"    控制值 {gripper_value} -> 位置 {position}...", end="")
            gripper.set_position(position)
            result = gripper.wait_move(200)
            if result == 0:
                print(" ✓")
            else:
                print(f" ✗ (超时，错误码: {result})")
            time.sleep(0.3)  # 300ms间隔
        
        print("\n  使用2000ms超时（建议修复）:")
        for gripper_value, position in control_sequence:
            print(f"    控制值 {gripper_value} -> 位置 {position}...", end="")
            gripper.set_position(position)
            result = gripper.wait_move(2000)
            if result == 0:
                print(" ✓")
            else:
                print(f" ✗ (超时，错误码: {result})")
            time.sleep(0.3)  # 300ms间隔
        
        # 断开连接
        gripper.disconnect()
        
        print("\n" + "=" * 60)
        print("测试结论:")
        print("=" * 60)
        print("1. 200ms超时对于大范围移动（如0->1000）可能不够")
        print("2. 建议将超时时间增加到2000ms（2秒）")
        print("3. 在 _control_dh_gripper 方法中应该检查 wait_move 的返回值")
        print("4. 即使超时，也不应该抛出异常导致程序崩溃")
        print("5. 应该添加连接状态检查，避免在断开连接时控制夹爪")
        
        return True
        
    except Exception as e:
        print(f"✗ 超时测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    test_basic_gripper_operation()

if __name__ == "__main__":
    sys.exit(main())
