#!/usr/bin/env python3
"""
Franka Server 完整测试脚本
测试顺序：
  1. 状态读取 & 格式验证
  2. 夹爪控制
  3. 微动运动（z轴 +1cm）
  4. 关节复位
"""
from franky import CartesianImpedanceMotion
import requests
import numpy as np
import time
import sys

BASE = "http://localhost:5000"
TIMEOUT = 10

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def post(endpoint, data=None, silent=False):
    try:
        r = requests.post(f"{BASE}{endpoint}", json=data, timeout=TIMEOUT)
        if not silent:
            status = PASS if r.ok else FAIL
            print(f"  {status} {endpoint:30s} → {r.text[:100]}")
        return r
    except requests.exceptions.ConnectionError:
        print(f"  {FAIL} {endpoint:30s} → 无法连接服务器，请先启动 franka_server.py")
        sys.exit(1)
    except Exception as e:
        print(f"  {FAIL} {endpoint:30s} → {e}")
        return None


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(condition, msg_pass, msg_fail):
    if condition:
        print(f"  {PASS} {msg_pass}")
    else:
        print(f"  {FAIL} {msg_fail}")
    return condition


# ============================================================
# 0. 连通性检查
# ============================================================
section("0. 连通性检查")
try:
    r = requests.post(f"{BASE}/getpos", timeout=5)
    print(f"  {PASS} 服务器响应正常")
except Exception:
    print(f"  {FAIL} 无法连接 {BASE}，请先启动 franka_server.py")
    sys.exit(1)


# ============================================================
# 1. 状态读取 & 格式验证
# ============================================================
section("1. 状态读取 & 格式验证")

# 各端点基本可达
for ep in ["/getpos", "/getpos_euler", "/getq", "/getdq",
           "/getvel", "/getforce", "/gettorque", "/getjacobian",
           "/getstate", "/get_gripper"]:
    post(ep)

# 详细格式检查
print(f"\n  {INFO} 详细格式检查...")
state = requests.post(f"{BASE}/getstate", timeout=TIMEOUT).json()

pose    = state.get("pose", [])
q       = state.get("q", [])
dq      = state.get("dq", [])
vel     = state.get("vel", [])
force   = state.get("force", [])
torque  = state.get("torque", [])
jac     = state.get("jacobian", [])
gripper = state.get("gripper_pos", None)

check(len(pose) == 7,           f"pose 维度正确: {len(pose)}D",            f"pose 维度错误: 期望7, 得到{len(pose)}")
check(len(q) == 7,              f"q 维度正确: {len(q)}D",                  f"q 维度错误: 期望7, 得到{len(q)}")
check(len(dq) == 7,             f"dq 维度正确: {len(dq)}D",                f"dq 维度错误: 期望7, 得到{len(dq)}")
check(len(vel) == 6,            f"vel 维度正确: {len(vel)}D",               f"vel 维度错误: 期望6, 得到{len(vel)}")
check(len(force) == 3,          f"force 维度正确: {len(force)}D",           f"force 维度错误: 期望3, 得到{len(force)}")
check(len(torque) == 3,         f"torque 维度正确: {len(torque)}D",         f"torque 维度错误: 期望3, 得到{len(torque)}")
check(len(jac) == 6,            f"jacobian 行数正确: {len(jac)}行",         f"jacobian 行数错误: 期望6, 得到{len(jac)}")
check(len(jac[0]) == 7,         f"jacobian 列数正确: {len(jac[0])}列",      f"jacobian 列数错误: 期望7, 得到{len(jac[0])}")
check(gripper is not None,      f"gripper_pos 存在: {gripper:.4f} m",       "gripper_pos 缺失")

# 四元数范数检查
quat = np.array(pose[3:])
quat_norm = np.linalg.norm(quat)
check(abs(quat_norm - 1.0) < 0.01,
      f"四元数范数正常: {quat_norm:.6f}",
      f"四元数范数异常: {quat_norm:.6f} (期望≈1.0)")

# # 关节角范围检查（Franka FR3 限制）
# q_arr = np.array(q)
# q_limits = np.array([2.7925, 1.7453, 2.7925, 0.0873, 2.8798, 4.6251, 3.0718])
# in_range = np.all(np.abs(q_arr) <= q_limits + 0.01)
# check(in_range,
#       f"关节角在安全范围内: {np.round(q_arr, 3)}",
#       f"关节角超出范围: {np.round(q_arr, 3)}")

# 位置合理性检查（机械臂工作空间粗略判断）
pos = np.array(pose[:3])
pos_norm = np.linalg.norm(pos)
check(0.2 < pos_norm < 1.2,
      f"末端位置合理: xyz={np.round(pos, 4)}, |r|={pos_norm:.3f}m",
      f"末端位置异常: xyz={np.round(pos, 4)}, |r|={pos_norm:.3f}m (期望0.2~1.2m)")

print(f"\n  {INFO} Euler角验证...")
euler_resp = requests.post(f"{BASE}/getpos_euler", timeout=TIMEOUT).json()
euler = euler_resp.get("pose", [])
check(len(euler) == 6,
      f"Euler pose 维度正确: {len(euler)}D → {np.round(euler, 4)}",
      f"Euler pose 维度错误: 期望6, 得到{len(euler)}")


# ============================================================
# 2. 夹爪控制
# ============================================================
section("2. 夹爪控制")
print(f"  {INFO} 开爪 → 等待2s → 读取位置...")
post("/open_gripper")
time.sleep(2.5)
g_open = requests.post(f"{BASE}/get_gripper", timeout=TIMEOUT).json().get("gripper", -1)
check(g_open > 0.03,
      f"开爪成功: {g_open:.4f} m (期望≈0.04)",
      f"开爪异常: {g_open:.4f} m (期望>0.03)")

print(f"  {INFO} 关爪 → 等待2s → 读取位置...")
post("/close_gripper")
time.sleep(2.5)
g_close = requests.post(f"{BASE}/get_gripper", timeout=TIMEOUT).json().get("gripper", -1)
check(g_close < 0.01,
      f"关爪成功: {g_close:.4f} m (期望≈0.00)",
      f"关爪异常: {g_close:.4f} m (期望<0.01)")

print(f"  {INFO} 移动到中间位置 (0.02m)...")
post("/move_gripper", {"gripper_pos": 50})   # 128/255 * 0.04 ≈ 0.02m
time.sleep(2.0)
g_mid = requests.post(f"{BASE}/get_gripper", timeout=TIMEOUT).json().get("gripper", -1)
check(0.005 < g_mid < 0.035,
      f"中间位置正常: {g_mid:.4f} m",
      f"中间位置异常: {g_mid:.4f} m (期望0.005~0.035)")

# 恢复开爪
post("/open_gripper", silent=True)
time.sleep(2.0)


# ============================================================
# 3. 微动运动测试
# ============================================================
section("3. 微动运动测试 (z轴 +1cm)")
print(f"  {WARN} 机器人将移动 1cm，手放急停旁边！")
ans = input("  按 Enter 继续，输入 s 跳过: ").strip().lower()

if ans == "s":
    print(f"  {INFO} 跳过运动测试")
else:
    pose_before = requests.post(f"{BASE}/getpos", timeout=TIMEOUT).json()["pose"]
    target = pose_before.copy()
    target[2] += 0.03   # z + 1cm

    print(f"  {INFO} 当前位置: {np.round(pose_before[:3], 4)}")
    print(f"  {INFO} 目标位置: {np.round(target[:3], 4)}")

    post("/pose", {"arr": target})
    time.sleep(3.0)

    pose_after = requests.post(f"{BASE}/getpos", timeout=TIMEOUT).json()["pose"]
    dz = pose_after[2] - pose_before[2]
    print(f"  {INFO} 实际位移: dz = {dz*1000:.2f} mm")

    check(abs(dz - 0.01) < 0.003,
          f"运动精度正常: dz={dz*1000:.2f}mm (期望10mm, 误差<3mm)",
          f"运动偏差过大: dz={dz*1000:.2f}mm (期望10mm)")

    # # 返回原位
    # print(f"  {INFO} 返回原位...")
    # post("/pose", {"arr": pose_before})
    # time.sleep(3.0)


# ============================================================
# 4. 错误恢复
# ============================================================
section("4. 错误恢复接口")
post("/clearerr")
post("/startimp")
post("/stopimp")
post("/startimp")


# ============================================================
# 5. 关节复位（可选）
# ============================================================
section("5. 关节复位（可选）")
print(f"  {WARN} 机器人将运动到初始姿态，确保周围无障碍！")
ans = input("  按 Enter 继续，输入 s 跳过: ").strip().lower()

if ans == "s":
    print(f"  {INFO} 跳过关节复位")
else:
    post("/jointreset")
    print(f"  {INFO} 关节复位中（约10s）...")
    time.sleep(12.0)
    q_after = requests.post(f"{BASE}/getq", timeout=TIMEOUT).json().get("q", [])
    target_q = [0, 0, 0, -1.9, 0, 2, 0]
    if len(q_after) == 7:
        err = np.max(np.abs(np.array(q_after) - np.array(target_q)))
        check(err < 0.1,
              f"复位成功: max_err={err:.4f} rad",
              f"复位偏差过大: max_err={err:.4f} rad")
    else:
        print(f"  {FAIL} 无法读取复位后关节角")


# ============================================================
# 汇总
# ============================================================
section("测试完成")
print(f"  如有 {FAIL} 项，将对应报错贴出继续排查。\n")

