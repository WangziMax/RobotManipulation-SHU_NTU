import requests
import json
import time
from pynput import keyboard

BASE_URL = "http://127.0.0.1:5000"
CONTROL_HZ = 30
MOVE_SPEED = 0.02  # m/s

pressed_keys = set()
running = True

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

def get_pose():
    response = requests.post(f"{BASE_URL}/getpos", timeout=2)
    response.raise_for_status()
    data = response.json()
    return response.json()["pose"]

def get_gripper_pos():
    try:
        response = requests.post(f"{BASE_URL}/get_gripper", timeout=1)
        response.raise_for_status()
        return response.json().get("gripper", 0.0)
    except Exception:
        return 0.0

def send_pose(pose):
    response = requests.post(f"{BASE_URL}/pose", json={"arr": pose}, timeout=1)
    response.raise_for_status()

def run_gripper(endpoint, desc):
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", timeout=2)
        response.raise_for_status()
        print(f"\n夹爪动作完成: {desc}")
    except Exception as e:
        print(f"\n发送夹爪指令失败 ({desc}): {e}")

def on_press(key):
    global running

    try:
        key_name = key.char.lower()
    except AttributeError:
        if key == keyboard.Key.esc:
            running = False
            return False
        return

    pressed_keys.add(key_name)

    if key_name == 'o':
        run_gripper("/open_gripper", "打开夹爪")
    elif key_name == 'c':
        run_gripper("/close_gripper", "快速关闭夹爪")
    elif key_name == 'v':
        run_gripper("/close_gripper_slow", "慢速关闭夹爪")
    elif key_name == 'x':
        running = False
        return False

def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        pass

def keyboard_control_loop():
    global running

    print("\n" + "="*50)
    print(" 进入键盘连续控制模式")
    print(" 机械臂键位:")
    print("   W / S : X轴 前 / 后")
    print("   A / D : Y轴 左 / 右")
    print("   R / F : Z轴 上 / 下")
    print(" 夹爪键位:")
    print("   O     : 打开夹爪 (Open)")
    print("   C     : 快速关闭夹爪 (Close)")
    print("   V     : 慢速关闭夹爪 (Slow Close)")
    print(" 系统键位:")
    print("   X / ESC : 退出控制")
    print("="*50)
    print(f"提示: 按住按键连续移动，速度约 {MOVE_SPEED * 1000:.0f}mm/s。\n")

    try:
        target_pose = get_pose()
        target_pose[3:7] = [1, 0, 0, 0]
        # print(target_pose)
        send_pose(target_pose)
        # response = requests.post(f"{BASE_URL}/getpos", timeout=2)
        # response.raise_for_status()
        # data = response.json()
        # print(f"数据类型: {type(data['pose'])}")
        # print(f"数据内容: {data['pose']}")
        # print(f"数据长度: {len(data['pose'])}")
    except Exception as e:
        print(f"\n[错误] 无法获取当前机器人位姿: {e}")
        return

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    dt = 1.0 / CONTROL_HZ
    last_status_time = 0.0

    try:
        while running:
            loop_start = time.time()
            delta = [0.0, 0.0, 0.0]

            if 'w' in pressed_keys:
                delta[0] += MOVE_SPEED * dt
            if 's' in pressed_keys:
                delta[0] -= MOVE_SPEED * dt
            if 'a' in pressed_keys:
                delta[1] += MOVE_SPEED * dt
            if 'd' in pressed_keys:
                delta[1] -= MOVE_SPEED * dt
            if 'r' in pressed_keys:
                delta[2] += MOVE_SPEED * dt
            if 'f' in pressed_keys:
                delta[2] -= MOVE_SPEED * dt

            moved = any(abs(v) > 0.0 for v in delta)
            if moved:
                for i in range(3):
                    target_pose[i] += delta[i]
                try:
                    # current_pose = get_pose()
                    # print(current_pose)
                    # current_pose[0:3] = target_pose[0:3]
                    send_pose(target_pose)
                except Exception as e:
                    print(f"\n发送移动指令失败: {e}")

            now = time.time()
            if now - last_status_time > 0.2:
                gripper_pos = get_gripper_pos()
                # print(
                #     f"\r目标状态 -> X: {target_pose[0]:.4f}, Y: {target_pose[1]:.4f}, "
                #     f"Z: {target_pose[2]:.4f} | 夹爪开度: {gripper_pos:.4f}m | "
                #     f"按住移动...",
                #     end="",
                # )
                actual_pose = get_pose()
                # print(actual_pose)
                print(
                    f"\rreal状态 -> X: {actual_pose[0]:.4f}, Y: {actual_pose[1]:.4f}, "
                    f"Z: {actual_pose[2]:.4f} | 夹爪开度: {gripper_pos:.4f}m | "
                    f"按住移动...",
                    end="",
                )
                last_status_time = now

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    finally:
        listener.stop()
        print("\n\n退出键盘控制模式。")


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