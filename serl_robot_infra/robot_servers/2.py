import requests
import json
import sys
import os

# 针对 Linux 终端的单键捕获（无需按回车）
if os.name == 'posix':
    import tty
    import termios
    def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
else:
    # Windows 备用
    import msvcrt
    def getch():
        return msvcrt.getch().decode('utf-8')

BASE_URL = "http://127.0.0.1:5000"
MOVE_STEP = 0.1  # 单次按键移动步长：1mm

def test_endpoint(endpoint, method="POST", data=None):
    url = f"{BASE_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data) if data else requests.post(url)
        else:
            response = requests.get(url)
        
        print(f"[{method}] {endpoint} -> 状态码: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(f"返回值: {response.text}")
    except Exception as e:
        print(f"[{method}] {endpoint} -> 请求失败: {e}")
    print("-" * 50)

def keyboard_control_loop():
    print("\n" + "="*50)
    print(" 进入键盘控制模式 (位移步长: 5mm)")
    print(" 机械臂键位:")
    print("   W / S : X轴 前 / 后")
    print("   A / D : Y轴 左 / 右")
    print("   R / F : Z轴 上 / 下")
    print(" 夹爪键位:")
    print("   O     : 打开夹爪 (Open)")
    print("   C     : 快速关闭夹爪 (Close)")
    print("   V     : 慢速关闭夹爪 (Slow Close)")
    print(" 系统键位:")
    print("   X     : 退出控制")
    print("="*50)
    print("⚠️ 提示: 移动指令会阻塞直到位姿就位，请【点按】按键，切勿长按！\n")

    while True:
        # 1. 获取当前最新位姿
        try:
            res = requests.post(f"{BASE_URL}/getpos").json()
            curr_pose = res["pose"]
        except Exception as e:
            print(f"\n[错误] 无法获取当前机器人位姿: {e}")
            break

        # 2. 获取当前夹爪状态
        try:
            g_res = requests.post(f"{BASE_URL}/get_gripper").json()
            gripper_pos = g_res.get("gripper", 0.0)
        except Exception:
            gripper_pos = 0.0

        # 在行首实时打印最新的位置与夹爪开度
        print(f"\r当前状态 -> X: {curr_pose[0]:.4f}, Y: {curr_pose[1]:.4f}, Z: {curr_pose[2]:.4f} | 夹爪开度: {gripper_pos:.4f}m | 指令...", end="")
        
        # 3. 捕获按键
        key = getch().lower()

        # 4. 解析按键
        target_pose = curr_pose.copy()
        is_move_command = False
        is_gripper_command = False
        action_desc = ""
        gripper_endpoint = ""

        # 机械臂移动分支
        if key == 'w':
            target_pose[0] += MOVE_STEP
            action_desc = "X轴 +5mm"
            is_move_command = True
        elif key == 's':
            target_pose[0] -= MOVE_STEP
            action_desc = "X轴 -5mm"
            is_move_command = True
        elif key == 'a':
            target_pose[1] += MOVE_STEP
            action_desc = "Y轴 +5mm"
            is_move_command = True
        elif key == 'd':
            target_pose[1] -= MOVE_STEP
            action_desc = "Y轴 -5mm"
            is_move_command = True
        elif key == 'r':
            target_pose[2] += MOVE_STEP
            action_desc = "Z轴 +5mm"
            is_move_command = True
        elif key == 'f':
            target_pose[2] -= MOVE_STEP
            action_desc = "Z轴 -5mm"
            is_move_command = True
            
        # 夹爪控制分支
        elif key == 'o':
            action_desc = "打开夹爪"
            gripper_endpoint = "/open_gripper"
            is_gripper_command = True
        elif key == 'c':
            action_desc = "快速关闭夹爪"
            gripper_endpoint = "/close_gripper"
            is_gripper_command = True
        elif key == 'v':
            action_desc = "慢速关闭夹爪"
            gripper_endpoint = "/close_gripper_slow"
            is_gripper_command = True
            
        elif key == 'x':
            print("\n\n退出键盘控制模式。")
            break
        else:
            continue  # 无效按键直接跳过

        # 5. 执行指令
        if is_move_command:
            print(f"\n执行动作: {action_desc} -> 正在移动机械臂...")
            try:
                response = requests.post(f"{BASE_URL}/pose", json={"arr": target_pose})
                if response.status_code == 200:
                    print("机械臂移动就位。")
                else:
                    print(f"服务端异常状态码: {response.status_code}")
            except Exception as e:
                print(f"发送移动指令失败: {e}")

        elif is_gripper_command:
            print(f"\n执行动作: {action_desc} -> 正在驱动夹爪...")
            try:
                response = requests.post(f"{BASE_URL}{gripper_endpoint}")
                if response.status_code == 200:
                    print(f"夹爪动作完成: {response.text}")
                else:
                    print(f"夹爪服务端异常状态码: {response.status_code}")
            except Exception as e:
                print(f"发送夹爪指令失败: {e}")


if __name__ == "__main__":
    print("开始测试带有夹爪的机械臂控制接口...\n")
    
    # 1. 基础连通性检查
    print("测试 1: 获取当前末端位姿...")
    test_endpoint("/getpos")
    
    print("测试 2: 获取当前夹爪位置...")
    test_endpoint("/get_gripper")
    
    # 2. 激活/初始化夹爪 (如果是 Robotiq 或需要归零的官方夹爪，先对其进行 Homing)
    print("测试 3: 初始化/激活夹爪驱动...")
    test_endpoint("/activate_gripper")
    
    # 3. 进入交互遥控
    keyboard_control_loop()