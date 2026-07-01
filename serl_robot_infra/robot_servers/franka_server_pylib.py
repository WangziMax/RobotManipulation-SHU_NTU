from flask import Flask, request, jsonify
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation as R   # used only outside hot loop
from absl import app, flags
import pylibfranka

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "172.16.0.2", "IP address of the franka robot's controller box")
flags.DEFINE_string("gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used")
flags.DEFINE_string("gripper_type", "Franka", "Type of gripper to use: Robotiq, Franka, or None")
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", "127.0.0.1", "URL for the flask server to run on.")

_TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)


# ---------------------------------------------------------------------------
# Fast pure-numpy rotation helpers (no scipy object creation in the hot loop)
# ---------------------------------------------------------------------------

def _rotmat_to_quat(m):
    """Rotation matrix (3x3) -> quaternion [x, y, z, w]. Pure numpy."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def _orientation_error(R_curr, R_tgt):
    """Rotation vector mapping R_curr -> R_tgt, expressed in base frame.
    Equivalent to scipy (R_tgt * R_curr.inv()).as_rotvec(), pure numpy."""
    R_err = R_tgt @ R_curr.T
    cos_a = (R_err[0, 0] + R_err[1, 1] + R_err[2, 2] - 1.0) * 0.5
    cos_a = min(1.0, max(-1.0, cos_a))
    angle = np.arccos(cos_a)
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    return axis * (angle / (2.0 * np.sin(angle)))


class FrankaServer:
    def __init__(self, robot_ip: str, gripper_type: str, reset_joint_target: list):
        self.robot_ip = robot_ip
        self.gripper_type = gripper_type
        self.reset_joint_target = [float(x) for x in reset_joint_target]

        # Robot state — written by control loop, read by Flask
        self.pos      = np.zeros(7)
        self.vel      = np.zeros(6)
        self.q        = np.zeros(7)
        self.dq       = np.zeros(7)
        self.force    = np.zeros(3)
        self.torque   = np.zeros(3)
        self.jacobian = np.zeros((6, 7))

        # Target pose as 4x4 matrix (row-major)
        self._target_mat = None

        # Compliance params
        self._ks = 200.0
        self._kr = 10.0

        # Control flags
        self._stop_flag  = False
        self._do_reset   = False
        self._reset_done = False

    # --- Flask-facing API ---

    def move(self, pose):
        assert len(pose) == 7
        p = np.array(pose, dtype=float)
        T = np.eye(4)
        T[:3, :3] = R.from_quat(p[3:]).as_matrix()   # outside hot loop, scipy ok
        T[:3, 3]  = p[:3]
        self._target_mat = T

    def update_compliance(self, params: dict):
        if "translational_stiffness" in params:
            self._ks = float(params["translational_stiffness"])
        if "rotational_stiffness" in params:
            self._kr = float(params["rotational_stiffness"])

    def set_load(self, mass, F_x_center_load, load_inertia):
        print("set_load: not supported during active control")

    def reset_joint(self):
        self._reset_done = False
        self._do_reset   = True
        start = time.time()
        while not self._reset_done:
            if time.time() - start > 60:
                print("joint reset TIMEOUT")
                break
            time.sleep(0.2)

    # --- Joint reset (runs between torque control sessions) ---

    def _do_joint_reset(self, robot):
        try:
            robot.automatic_error_recovery()
        except Exception:
            pass
        active   = robot.start_joint_position_control(
            pylibfranka.ControllerMode.JointImpedance
        )
        target_q = np.array(self.reset_joint_target)
        start    = time.time()
        while True:
            state, _ = active.readOnce()
            q_curr   = np.array(state.q)
            if np.allclose(q_curr, target_q, atol=1e-2):
                break
            if time.time() - start > 30:
                print("joint reset TIMEOUT")
                break
            q_cmd = (q_curr + 0.02 * (target_q - q_curr)).tolist()
            active.writeOnce(pylibfranka.JointPositions(q_cmd))
        fin = pylibfranka.JointPositions(target_q.tolist())
        fin.motion_finished = True
        active.writeOnce(fin)

    # --- Main control loop (main thread) ---

    def run_control_loop(self):
        nullspace_q = np.array(self.reset_joint_target)
        kn          = 0.5
        eye7        = np.eye(7)

        robot = pylibfranka.Robot(self.robot_ip, pylibfranka.RealtimeConfig.kIgnore)
        model = robot.load_model()

        while True:
            try:
                robot.automatic_error_recovery()
            except Exception:
                pass
            robot.set_collision_behavior([100]*7, [100]*7, [100]*6, [100]*6)

            active          = robot.start_torque_control()
            target_latched  = False
            last_state      = None
            self._stop_flag = False

            try:
                while not self._stop_flag:
                    state, _ = active.readOnce()
                    last_state = state

                    # --- Raw state (cheap) ---
                    O_T_EE = np.asarray(state.O_T_EE).reshape(4, 4).T
                    R_curr = O_T_EE[:3, :3]
                    p_curr = O_T_EE[:3, 3]
                    q   = np.asarray(state.q)
                    dq  = np.asarray(state.dq)
                    jac = np.asarray(model.zero_jacobian(state)).reshape(6, 7, order="F")
                    coriolis = np.asarray(model.coriolis(state))
                    ext = np.asarray(state.K_F_ext_hat_K)
                    vel = jac @ dq

                    # --- Publish state for Flask (pure numpy quat) ---
                    self.pos      = np.concatenate([p_curr, _rotmat_to_quat(R_curr)])
                    self.vel      = vel
                    self.q        = q
                    self.dq       = dq
                    self.force    = ext[:3]
                    self.torque   = ext[3:]
                    self.jacobian = jac

                    # --- Latch target on first tick ---
                    if not target_latched:
                        self._target_mat = O_T_EE.copy()
                        target_latched   = True

                    T_tgt   = self._target_mat
                    p_tgt   = T_tgt[:3, 3]
                    R_tgt   = T_tgt[:3, :3]

                    # --- Cartesian error (pure numpy) ---
                    pos_err = p_tgt - p_curr
                    ori_err = _orientation_error(R_curr, R_tgt)
                    err     = np.concatenate([pos_err, ori_err])

                    # --- Diagonal stiffness/damping, elementwise (cheap) ---
                    ks = self._ks
                    kr = self._kr
                    K_diag = np.array([ks, ks, ks, kr, kr, kr])
                    D_diag = 2.0 * np.sqrt(K_diag)
                    F = K_diag * err - D_diag * vel

                    # --- Nullspace torque (6x6 inv, cheap) ---
                    JJT_inv     = np.linalg.inv(jac @ jac.T)
                    I_minus_JpJ = eye7 - jac.T @ JJT_inv @ jac
                    tau_null    = I_minus_JpJ @ (
                        kn * (nullspace_q - q) - 2.0 * np.sqrt(kn) * dq
                    )

                    tau = np.clip(jac.T @ F + coriolis + tau_null, -_TAU_MAX, _TAU_MAX)
                    active.writeOnce(pylibfranka.Torques(tau.tolist()))

                    if self._do_reset:
                        break

            except Exception as e:
                print(f"Control loop error: {e}")
            finally:
                if last_state is not None:
                    try:
                        c = pylibfranka.Torques([0.0] * 7)
                        c.motion_finished = True
                        active.writeOnce(c)
                    except Exception:
                        pass

            if self._do_reset:
                print("Running joint reset...")
                try:
                    self._do_joint_reset(robot)
                except Exception as e:
                    print(f"Joint reset error: {e}")
                self._target_mat = None
                self._do_reset   = False
                self._reset_done = True
                print("Joint reset done")
            else:
                time.sleep(2)
                try:
                    robot.automatic_error_recovery()
                except Exception:
                    pass


###############################################################################


def main(_):
    ROBOT_IP           = FLAGS.robot_ip
    GRIPPER_IP         = FLAGS.gripper_ip
    GRIPPER_TYPE       = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    if GRIPPER_TYPE == "Robotiq":
        from robot_servers.robotiq_gripper_server import RobotiqGripperServer
        gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
    elif GRIPPER_TYPE == "Franka":
        from robot_servers.franka_gripper_server import FrankaGripperServer
        gripper_server = FrankaGripperServer(robot_ip=ROBOT_IP)
    elif GRIPPER_TYPE == "None":
        gripper_server = None
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        reset_joint_target=RESET_JOINT_TARGET,
    )

    # Flask in background thread
    flask_thread = threading.Thread(
        target=lambda: webapp.run(host=FLAGS.flask_url),
        daemon=True,
    )
    flask_thread.start()

    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        data = request.json
        robot_server.set_load(data["mass"], data["F_x_center_load"], data["load_inertia"])
        return "Set Load"

    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server._stop_flag = True
        return "Stopped impedance"

    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        pos   = robot_server.pos
        euler = R.from_quat(pos[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([pos[:3], euler]).tolist()})

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": robot_server.pos.tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": robot_server.vel.tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": robot_server.force.tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": robot_server.torque.tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": robot_server.q.tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": robot_server.dq.tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": robot_server.jacobian.flatten().tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.reset_joint()
        return "Reset Joint"

    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        gripper_server.activate_gripper()
        return "Activated"

    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        gripper_server.reset_gripper()
        return "Reset"

    @webapp.route("/open_gripper", methods=["POST"])
    def open_gripper():
        gripper_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close_gripper():
        gripper_server.close()
        return "Closed"

    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        gripper_server.close_slow()
        return "Closed"

    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        pos = np.clip(int(request.json["gripper_pos"]), 0, 255)
        gripper_server.move(pos)
        return "Moved Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clearerr():
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        robot_server.move(np.array(request.json["arr"]))
        return "Moved"

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify({
            "pose":        robot_server.pos.tolist(),
            "vel":         robot_server.vel.tolist(),
            "force":       robot_server.force.tolist(),
            "torque":      robot_server.torque.tolist(),
            "q":           robot_server.q.tolist(),
            "dq":          robot_server.dq.tolist(),
            "jacobian":    robot_server.jacobian.flatten().tolist(),
            "gripper_pos": gripper_server.gripper_pos,
        })

    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        robot_server.update_compliance(request.json)
        return "Updated compliance parameters"

    # Control loop blocks main thread (same execution model as the working demo)
    robot_server.run_control_loop()


if __name__ == "__main__":
    app.run(main)
