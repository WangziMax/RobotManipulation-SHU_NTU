"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""

from flask import Flask, request, jsonify
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
from absl import app, flags

# 引入 franky 库组件
from franky import Robot, Gripper, Affine, CartesianMotion, JointMotion

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Franka", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [      -0.15152642130851746,
    -0.2778744399547577,
    -0.03075450286269188,
    -1.9316643476486206,
    -0.1036304458975792,
    1.5131351947784424,
    -0.011792900040745735],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string(
    "flask_url", "127.0.0.1", "URL for the flask server to run on."
)


class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    using franky library native interface."""

    def __init__(self, robot_ip, gripper_type, reset_joint_target):
        self.robot_ip = robot_ip
        self.reset_joint_target = np.array(reset_joint_target, dtype=np.float64)
        self.gripper_type = gripper_type

        self.lock = threading.Lock()

        # 1. 初始化 franky 机器人连接
        print(f"[Franka] Connecting to Franka via franky at {self.robot_ip}...")
        try:
            self.robot = Robot(self.robot_ip)
            self.robot.recover_from_errors()
            self.robot.relative_dynamics_factor = 0.2  # 基础动力学速度因子
            
            try:
                self.model = self.robot.model
            except Exception as e:
                print(f"[Franka WARN] Model interface unavailable: {e}")
                self.model = None
                
            print("[Franka] Robot connection established.")
        except Exception as e:
            print(f"[Franka FATAL] Failed to connect to robot: {e}")
            raise

        # 2. 内部缓存状态变量
        self.pos = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float64)
        self.vel = np.zeros(6, dtype=np.float64)
        self.force = np.zeros(3, dtype=np.float64)
        self.torque = np.zeros(3, dtype=np.float64)
        self.q = np.zeros(7, dtype=np.float64)
        self.dq = np.zeros(7, dtype=np.float64)
        self.jacobian = np.zeros((6, 7), dtype=np.float64)
        
        # 首次拉取状态
        self._update_states()

    def _update_states(self):
        """利用 franky 原生 API 正确读取类对象中的机械臂状态信息"""
        try:
            if not self.robot:
                return

            # 1. 读取笛卡尔状态 (通过专属的 current_cartesian_state 属性)
            cs = self.robot.current_cartesian_state
            ee = cs.pose.end_effector_pose
            pos_arr = np.asarray(ee.translation, dtype=np.float64)

            # franky 内部四元数顺序为 [w, x, y, z]，此处统一转换为标准 [x, y, z, w] 传出
            q_raw = np.asarray(ee.quaternion, dtype=np.float64)
            if q_raw.shape[0] == 4:
                q_xyzw = np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]])
            else:
                q_xyzw = q_raw.copy()
            norm = np.linalg.norm(q_xyzw)
            if norm > 1e-8:
                q_xyzw /= norm

            # 2. 读取外力和外力矩
            new_force = np.zeros(3, dtype=np.float64)
            new_torque = np.zeros(3, dtype=np.float64)
            if hasattr(cs, 'wrench'):
                w = cs.wrench
                if hasattr(w, 'force'):
                    new_force = np.asarray(w.force, dtype=np.float64)
                if hasattr(w, 'torque'):
                    new_torque = np.asarray(w.torque, dtype=np.float64)

            # 3. 读取关节状态 (通过专属的 current_joint_state 属性)
            js = self.robot.current_joint_state
            q_arr = np.asarray(js.position, dtype=np.float64).reshape(-1)
            new_q = q_arr.copy() if q_arr.size == 7 else self.q.copy()
            
            new_dq = np.zeros(7, dtype=np.float64)
            if hasattr(js, 'velocity'):
                dq_arr = np.asarray(js.velocity, dtype=np.float64).reshape(-1)
                if dq_arr.size == 7:
                    new_dq = dq_arr.copy()

            # 4. 计算雅可比矩阵与末端速度
            new_jac = np.zeros((6, 7), dtype=np.float64)
            new_vel = np.zeros(6, dtype=np.float64)
            if self.model is not None and new_q.size == 7:
                try:
                    jac_raw = np.asarray(self.model.zero_jacobian(js), dtype=np.float64)
                    if jac_raw.size == 42:
                        new_jac = jac_raw.reshape(6, 7)
                        new_vel = new_jac @ new_dq
                except Exception:
                    pass

            # 线程锁保护：统一更新到类属性中
            with self.lock:
                self.pos = np.concatenate([pos_arr, q_xyzw])
                self.force = new_force
                self.torque = new_torque
                self.q = new_q
                self.dq = new_dq
                self.jacobian = new_jac
                self.vel = new_vel

        except Exception as e:
            print(f"[Franka ERROR] Error updating states: {e}")

    def start_impedance(self):
        self.clear()
        print("franky control layer is ready.")

    def stop_impedance(self):
        print("franky control layer stopped.")
        try:
            self.robot.stop()
        except Exception:
            pass

    def clear(self):
        """错误恢复"""
        try:
            self.robot.recover_from_errors()
            print("franky: Recovered from errors.")
        except Exception as e:
            print(f"franky recovery failed: {e}")

    def reset_joint(self):
        """利用 franky 的 JointMotion 移动到目标复位关节角"""
        self.clear()
        print("RUNNING JOINT RESET via franky...")
        try:
            motion = JointMotion(self.reset_joint_target.tolist())
            self.robot.move(motion)
            print("RESET DONE")
        except Exception as e:
            print(f"Joint reset failed: {e}")
            
        self.clear()
        self._update_states()
        print("RESET FINISHED, Current Pos:", self.pos)

    def move(self, pose: list):
        """发送一次异步动作命令，不等待到位。"""
        assert len(pose) == 7

        try:
            self.robot.poll_motion()
        except Exception:
            self.robot.recover_from_errors()

        target = np.asarray(pose, dtype=np.float64)
        quat = target[3:]
        norm = np.linalg.norm(quat)
        target[3:] = quat / norm if norm > 1e-8 else np.array([0., 0., 0., 1.])

        x, y, z = target[:3]
        qx, qy, qz, qw = target[3:]

        motion = CartesianMotion(Affine([x, y, z], [qw, qx, qy, qz]))

        self.robot.move(motion, asynchronous=True)


class FrankaGripperWrapper:
    """包装真实的 Franka 官方夹爪硬件接口"""
    def __init__(self, robot_ip):
        print(f"[Gripper] Connecting to Franka Gripper at {robot_ip}...")
        try:
            self.gripper = Gripper(robot_ip)
            self.gripper.homing()
            self.gripper_pos = 0.08
            print("[Gripper] Franka Gripper initialized and homed.")
        except Exception as e:
            print(f"[Gripper ERROR] Failed to initialize Franka Gripper: {e}")
            self.gripper = None
            self.gripper_pos = 0.0

    def activate_gripper(self):
        if self.gripper:
            try: self.gripper.homing()
            except Exception: pass

    def reset_gripper(self):
        if self.gripper:
            try: self.gripper.stop()
            except Exception: pass

    def open(self):
        if self.gripper:
            self.gripper.move(0.08, 0.1)
            self.gripper_pos = 0.08

    def close(self):
        if self.gripper:
            self.gripper.move(0.002, 0.1)
            self.gripper_pos = 0.002

    def close_slow(self):
        if self.gripper:
            self.gripper.move(0.002, 0.05)
            self.gripper_pos = 0.002

    def move(self, pos_val):
        if self.gripper:
            # 将 0-255 映射到 0.0 到 0.08 米
            meters = (np.clip(pos_val, 0, 255) / 255.0) * 0.08
            self.gripper.move(meters, 0.1)
            self.gripper_pos = meters


###############################################################################


def main(_):
    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    # 初始化机器人服务端
    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    robot_server.start_impedance()

    # 初始化夹爪控制器
    if GRIPPER_TYPE == "Robotiq":
        from robot_servers.robotiq_gripper_server import RobotiqGripperServer
        gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
    elif GRIPPER_TYPE == "Franka":
        gripper_server = FrankaGripperWrapper(ROBOT_IP)
    elif GRIPPER_TYPE == "None":
        gripper_server = None
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    # ==================== Flask API 路由接口 ====================

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        print("Set load parameters requested.")
        return "Set Load"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"
    
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        robot_server._update_states()
        with robot_server.lock:
            xyz = robot_server.pos[:3].copy()
            quat = robot_server.pos[3:].copy()
        r = R.from_quat(quat).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"pose": robot_server.pos.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"vel": robot_server.vel.tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"force": robot_server.force.tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"torque": robot_server.torque.tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"q": robot_server.q.tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"dq": robot_server.dq.tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify({"jacobian": robot_server.jacobian.flatten().tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        if gripper_server:
            return jsonify({"gripper": gripper_server.gripper_pos})
        return jsonify({"gripper": 0.0})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        if gripper_server: 
            gripper_server.activate_gripper()
        return "Activated"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        if gripper_server: 
            gripper_server.reset_gripper()
        return "Reset"

    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        if gripper_server: 
            gripper_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        if gripper_server: 
            gripper_server.close()
        return "Closed"

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        if gripper_server: 
            gripper_server.close_slow()
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        if gripper_server:
            gripper_pos = request.json
            pos = int(gripper_pos["gripper_pos"])
            gripper_server.move(pos)
        return "Moved Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])  
        robot_server.move(pos)  # 此时此处会阻塞直到动作结束或超时，随后返回 "Moved"
        return "Moved"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        robot_server._update_states()
        with robot_server.lock:
            return jsonify(
                {
                    "pose": robot_server.pos.tolist(),
                    "vel": robot_server.vel.tolist(),
                    "force": robot_server.force.tolist(),
                    "torque": robot_server.torque.tolist(),
                    "q": robot_server.q.tolist(),
                    "dq": robot_server.dq.tolist(),
                    "jacobian": robot_server.jacobian.flatten().tolist(),
                    "gripper_pos": gripper_server.gripper_pos if gripper_server else 0.0,
                }
            )

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        if "relative_dynamics_factor" in request.json:
            robot_server.robot.relative_dynamics_factor = float(request.json["relative_dynamics_factor"])
        return "Updated compliance parameters"

    print(f"\n{'='*60}")
    print(f"Franka Native Web Server Active")
    print(f"URL: http://{FLAGS.flask_url}:5000")
    print(f"{'='*60}\n")

    webapp.run(host=FLAGS.flask_url, port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    app.run(main)