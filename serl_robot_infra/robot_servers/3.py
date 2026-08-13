import json
import time
import requests

# 配置服务端的基本 URL（请根据实际情况修改 IP）
SERVER_URL = "http://127.0.0.1:5000"


def test_route(endpoint, data=None):
    """通用请求测试函数"""
    url = f"{SERVER_URL}{endpoint}"
    print(f"📡 Sending POST request to: {url}")
    try:
        if data:
            response = requests.post(url, json=data)
        else:
            response = requests.post(url)

        print(f"⏱️ Status Code: {response.status_code}")

        # 如果返回的是 JSON，优雅打印；否则打印文本
        try:
            print("📦 Response Data:")
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print(f"📦 Response Text: {response.text}")

    except Exception as e:
        print(f"❌ Request failed: {e}")
    print("-" * 60)


print("🚀 Starting Franka Server API Tests...\n")

# 1. 测试获取单个状态接口
print("=== 1. Testing State Getters ===")
test_route("/getpos")
test_route("/getpos_euler")
test_route("/getq")
test_route("/getforce")

# 2. 测试一键获取打包状态接口
print("\n=== 2. Testing Full State Pack ===")
test_route("/getstate")

# 3. 测试夹爪控制接口 (仅当设置了夹爪时生效)
print("\n=== 3. Testing Gripper Control ===")
test_route("/open_gripper")
time.sleep(1.0)  # 等待夹爪动作

# 模拟把夹爪控制到指定位置 (例如 0-255 映射，这里测试发送 128)
test_route("/move_gripper", data={"gripper_pos": 128})
time.sleep(1.0)

test_route("/get_gripper")

# 4. 测试修改动力学速度因子
print("\n=== 4. Testing Parameter Update ===")
test_route("/update_param", data={"relative_dynamics_factor": 0.15})

# 5. 测试机器人复位与动作接口（⚠️ 请确保周围安全再取消注释）
print("\n=== 5. Testing Robot Motion (Optional) ===")

# 恢复错误
test_route("/clearerr")

# 关节复位测试 (取消注释将控制实体机器人运动！)
# while True:
#     test_route("/jointreset")

# 笛卡尔空间异步移动测试 (取消注释将控制实体机器人运动！)
# 示例数据：[x, y, z, qx, qy, qz, qw]
# test_pose = [0.4, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0]
# test_route("/pose", data={"arr": test_pose})