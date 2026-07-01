from flask import Flask, request, jsonify
import numpy as np
import time
import threading
from absl import app, flags
from scipy.spatial.transform import Rotation as R
from franky import (
    Robot, Gripper, CartesianImpedanceMotion, JointMotion, Affine, Duration,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("robot_ip", "172.16.0.2", "Franka FCI IP")
flags.DEFINE_string("flask_url", "0.0.0.0", "Flask host")
flags.DEFINE_list("reset_joint_target", [0, 0, 0, -1.9, 0, 2, 0], "reset joint")

GRIPPER_MAX_WIDTH = 0.08
GRIPPER_SPEED     = 0.1

RELATIVE_DYNAMICS_FACTOR = 0.2

# Streaming rate for impedance targets. Keep modest; impedance tolerates this.
MOTION_HZ = 30.0
MOTION_POLL_INTERVAL_SEC = 0.5

# Impedance gains (N/m, Nm/rad)
TRANS_STIFFNESS = 3000.0
ROT_STIFFNESS   = 200.0
# Per-segment duration. Slightly longer than the loop period so consecutive
# impedance motions blend instead of braking to a stop.
SEG_DURATION_MS = 500


def async_call(fn):
    threading.Thread(target=fn, daemon=True).start()


class FrankaServer:

    def __init__(self, robot_ip, reset_joint_target):
        self.robot_ip = robot_ip
        self.reset_joint_target = np.array(reset_joint_target, dtype=np.float64)

        self.lock         = threading.Lock()
        self.gripper_lock = threading.Lock()

        self.pos      = np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float64)
        self.q        = np.zeros(7, dtype=np.float64)
        self.dq       = np.zeros(7, dtype=np.float64)
        self.vel      = np.zeros(6, dtype=np.float64)
        self.force    = np.zeros(3, dtype=np.float64)
        self.torque   = np.zeros(3, dtype=np.float64)
        self.jacobian = np.zeros((6, 7), dtype=np.float64)

        self.gripper_pos   = 0.0
        self.gripper_ready = False

        self.running            = True
        self.impedance_running  = False

        self._latest_target     = None
        self._target_lock       = threading.Lock()
        self._recovering        = False
        self._motion_cmd_count  = 0
        self._last_poll_time    = 0.0

        self.robot   = None
        self.gripper = None
        self.model   = None
        self.connected = False
        self.state_update_count = 0
        self.state_error_count  = 0

        print("[Franka] Connecting to robot at", robot_ip)
        try:
            self.robot = Robot(robot_ip)
            self.robot.recover_from_errors()
            self.robot.relative_dynamics_factor = RELATIVE_DYNAMICS_FACTOR

            try:
                self.robot.set_collision_behavior(
                    [20.0] * 7, [20.0] * 7,
                    [20.0] * 6, [20.0] * 6,
                )
            except Exception as e:
                print(f"[WARN] set_collision_behavior: {e}")

            try:
                self.model = self.robot.model
            except Exception as e:
                print(f"[WARN] model unavailable: {e}")
                self.model = None

            self.gripper = Gripper(robot_ip)
            self.connected = True
            print("[Franka] Connected")
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.connected = False
            raise

        self.state_thread = threading.Thread(target=self._state_loop, daemon=True)
        self.state_thread.start()

        self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
        self._motion_thread.start()
        self._init_gripper_internal()

    # =========================================================
    # State Loop
    # =========================================================
    def _state_loop(self):
        print("[State Loop] Started at ~50 Hz")
        consecutive_errors = 0

        while self.running:
            try:
                if not self.connected or not self.robot:
                    time.sleep(0.1)
                    continue

                try:
                    cs = self.robot.current_cartesian_state
                    ee = cs.pose.end_effector_pose

                    pos_arr = np.asarray(ee.translation, dtype=np.float64)

                    # franky quaternion convention: [w, x, y, z] -> [x, y, z, w]
                    q_raw = np.asarray(ee.quaternion, dtype=np.float64)
                    if q_raw.shape[0] == 4:
                        q_xyzw = np.array([q_raw[1], q_raw[2], q_raw[3], q_raw[0]])
                    else:
                        q_xyzw = q_raw.copy()
                    norm = np.linalg.norm(q_xyzw)
                    if norm > 1e-8:
                        q_xyzw /= norm
                    new_pos = np.concatenate([pos_arr, q_xyzw])

                    new_force  = self.force.copy()
                    new_torque = self.torque.copy()
                    if hasattr(cs, 'wrench'):
                        w = cs.wrench
                        if hasattr(w, 'force'):
                            new_force  = np.asarray(w.force,  dtype=np.float64)
                        if hasattr(w, 'torque'):
                            new_torque = np.asarray(w.torque, dtype=np.float64)

                    js = self.robot.current_joint_state
                    q_arr = np.asarray(js.position, dtype=np.float64).reshape(-1)
                    new_q = q_arr.copy() if q_arr.size == 7 else self.q.copy()
                    new_dq = self.dq.copy()
                    if hasattr(js, 'velocity'):
                        dq_arr = np.asarray(js.velocity, dtype=np.float64).reshape(-1)
                        if dq_arr.size == 7:
                            new_dq = dq_arr.copy()

                    # Jacobian + end-effector velocity via franky Model
                    new_jac = self.jacobian.copy()
                    new_vel = np.zeros(6, dtype=np.float64)
                    if self.model is not None and new_q.size == 7:
                        try:
                            jac_raw = np.asarray(
                                self.model.zero_jacobian(self.robot.current_joint_state),
                                dtype=np.float64,
                            )
                            if jac_raw.size == 42:
                                new_jac = jac_raw.reshape(6, 7)
                                new_vel = new_jac @ new_dq
                        except Exception:
                            pass

                    new_gripper = self.gripper_pos
                    try:
                        if self.gripper and self.gripper_ready:
                            gs = self.gripper.state
                            if hasattr(gs, 'width'):
                                new_gripper = float(gs.width)
                    except Exception:
                        pass

                    with self.lock:
                        self.pos         = new_pos
                        self.force       = new_force
                        self.torque      = new_torque
                        self.q           = new_q
                        self.dq          = new_dq
                        self.vel         = new_vel
                        self.jacobian    = new_jac
                        self.gripper_pos = new_gripper

                    self.state_update_count += 1
                    consecutive_errors = 0

                    if self.state_update_count % 50 == 0:
                        print(f"[State] #{self.state_update_count} "
                            f"pos={np.round(pos_arr, 4)} "
                            f"gripper={new_gripper:.4f}m "
                            f"force={np.linalg.norm(new_force):.3f}N")

                except Exception as e:
                    self.state_error_count += 1
                    consecutive_errors += 1
                    if consecutive_errors <= 3:
                        print(f"[ERROR] State parse: {e}")

            except Exception as e:
                self.state_error_count += 1
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"[ERROR] State loop: {e}")
                if consecutive_errors > 100:
                    print("[ERROR] Too many errors, reconnecting...")
                    try:
                        self.robot = Robot(self.robot_ip)
                        self.robot.relative_dynamics_factor = RELATIVE_DYNAMICS_FACTOR
                        try:
                            self.model = self.robot.model
                        except Exception:
                            self.model = None
                        self.connected = True
                        consecutive_errors = 0
                    except Exception as e2:
                        print(f"[ERROR] Reconnect failed: {e2}")

            time.sleep(0.02)

        print("[State Loop] Stopped")

    # =========================================================
    # Motion Loop — async impedance motion, streamed at MOTION_HZ
    # =========================================================
    def _poll_previous_motion(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_poll_time < MOTION_POLL_INTERVAL_SEC:
            return
        self._last_poll_time = now
        try:
            self.robot.poll_motion()
        except Exception as exc:
            print(f"[Motion] async motion error: {exc!r}")
            try:
                self.robot.recover_from_errors()
                print("[Motion] recovered from async error")
            except Exception as e2:
                print(f"[Motion] recover failed: {e2!r}")

    def _motion_loop(self):
          """
          Fixed-rate async impedance-motion loop. Continuously re-sends the
          current target as a CartesianImpedanceMotion so the internal spring
          keeps pulling the EE toward the target (and holds it there). Impedance
          motion is torque-level and does NOT run through the motion generator's
          continuity checks, so re-sending at MOTION_HZ is safe (no Reflex).
          """
          print(f"[Motion Loop] Started at {MOTION_HZ} Hz (impedance async, continuous)")
          period = 1.0 / MOTION_HZ

          while self.running:
              loop_start = time.monotonic()

              if not self.impedance_running or not self.connected or self._recovering:
                  time.sleep(period)
                  continue

              with self._target_lock:
                  target = self._latest_target

              if target is None:
                  time.sleep(period)
                  continue

              try:
                  t  = target[:3].tolist()
                  qx, qy, qz, qw = target[3], target[4], target[5], target[6]
                  # franky Affine quaternion order: [w, x, y, z]
                  motion = CartesianImpedanceMotion(
                      Affine(t, [qw, qx, qy, qz]),
                      Duration(SEG_DURATION_MS),
                      translational_stiffness=TRANS_STIFFNESS,
                      rotational_stiffness=ROT_STIFFNESS,
                      return_when_finished=False,
                  )

                  self._poll_previous_motion()
                  # Re-send every cycle — keeps the impedance spring active.
                  self.robot.move(motion, asynchronous=True)
                  self._motion_cmd_count += 1

              except Exception as exc:
                  err = str(exc)
                  print(f"[Motion] Error: {exc!r}")
                  if "Reflex" in err or "motion aborted" in err or "command not possible" in err:
                      with self._target_lock:
                          self._latest_target = None
                      self._recover_with_backoff(base_wait=1.0)
                  else:
                      try:
                          self.robot.recover_from_errors()
                      except Exception as e2:
                          print(f"[Motion] recover failed: {e2!r}")
                      time.sleep(0.3)

              elapsed = time.monotonic() - loop_start
              time.sleep(max(0.0, period - elapsed))

          print("[Motion Loop] Stopped")


    def _recover_with_backoff(self, base_wait=1.0, max_attempts=5):
        self._recovering = True
        try:
            t0 = time.time()
            while time.time() - t0 < 5.0:
                with self.lock:
                    if np.linalg.norm(self.dq) < 0.005:
                        break
                time.sleep(0.05)

            for attempt in range(max_attempts):
                wait = base_wait * (2 ** attempt)
                print(f"[Recover] attempt {attempt + 1}/{max_attempts}, waiting {wait:.1f}s")
                time.sleep(wait)
                try:
                    self.robot.recover_from_errors()
                    time.sleep(0.3)
                    print("[Recover] OK")
                    return True
                except Exception as e:
                    if "Reflex" in str(e) or "command not possible" in str(e):
                        print(f"[Recover] Still in Reflex: {e}")
                        continue
                    print(f"[Recover] Unexpected error: {e}")
                    return False
            print("[Recover] Gave up after max attempts")
            return False
        finally:
            self._recovering = False

    # =========================================================
    # Arm Control
    # =========================================================
    def start_impedance(self):
        print("[Franka] Impedance started")
        self.impedance_running = True

    def stop_impedance(self):
        print("[Franka] Impedance stopped")
        self.impedance_running = False
        self._recovering = False
        try:
            if self.robot:
                self.robot.join_motion()
        except Exception:
            pass
        try:
            if self.robot:
                self.robot.stop()
        except Exception as e:
            print(f"[WARN] stop: {e}")

    def clear(self):
        print("[Franka] Clearing errors")
        try:
            if self.robot:
                self.robot.recover_from_errors()
        except Exception as e:
            print(f"[ERROR] clear: {e}")

    def _wait_for_physical_stop(self, timeout=5.0, vel_threshold=0.005):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                dq_norm = np.linalg.norm(self.dq)
            if dq_norm < vel_threshold:
                return True
            time.sleep(0.05)
        with self.lock:
            dq_norm = np.linalg.norm(self.dq)
        print(f"[Franka] wait_for_physical_stop timeout (dq_norm={dq_norm:.4f})")
        return False

    def reset_joint(self):
        try:
            print("[Franka] Joint reset: stopping motion loop")
            self.impedance_running = False
            self._recovering = False

            with self._target_lock:
                self._latest_target = None

            try:
                self.robot.join_motion()
            except Exception:
                pass

            self._wait_for_physical_stop(timeout=5.0)
            time.sleep(0.5)

            try:
                self.robot.stop()
            except Exception:
                pass
            time.sleep(0.2)

            self.robot.recover_from_errors()
            time.sleep(0.3)

            print("[Franka] Joint reset: moving to home...")
            motion = JointMotion(self.reset_joint_target.tolist())
            self.robot.move(motion)
            print("[Franka] Joint reset: done")
            self.start_impedance()

        except Exception as e:
            print(f"[ERROR] Joint reset failed: {e}")
            import traceback; traceback.print_exc()
            raise

    def move(self, pose):
        """Queue target pose [x, y, z, qx, qy, qz, qw]; motion loop fires async."""
        assert len(pose) == 7
        target = np.asarray(pose, dtype=np.float64)
        norm = np.linalg.norm(target[3:])
        target[3:] = target[3:] / norm if norm > 1e-8 else np.array([0., 0., 0., 1.])
        with self._target_lock:
            self._latest_target = target.copy()

    def move_sync(self, pose, timeout=10.0):
        self.move(pose)
        time.sleep(2.0 / MOTION_HZ)
        try:
            self.robot.join_motion()
        except Exception as e:
            print(f"[move_sync] join_motion error (non-fatal): {e}")

    # =========================================================
    # Gripper
    # =========================================================
    def _init_gripper_internal(self):
        try:
            if not self.gripper:
                print("[Gripper] Not connected")
                return
            print("[Gripper] Homing...")
            with self.gripper_lock:
                self.gripper.homing()
                time.sleep(1.0)
            self.gripper_ready = True
            self.gripper_pos = GRIPPER_MAX_WIDTH
            print("[Gripper] Ready")
        except Exception as e:
            print(f"[ERROR] Gripper homing: {e}")
            self.gripper_ready = False

    def _ensure_gripper_ready(self):
        if not self.gripper_ready:
            self._init_gripper_internal()

    def gripper_open(self):
        try:
            self._ensure_gripper_ready()
            with self.gripper_lock:
                self.gripper.move(GRIPPER_MAX_WIDTH, GRIPPER_SPEED)
                self.gripper_pos = GRIPPER_MAX_WIDTH
            print("[Gripper] Opened")
        except Exception as e:
            print(f"[ERROR] Gripper open: {e}")
            self.gripper_ready = False
            raise

    def gripper_close(self):
        try:
            self._ensure_gripper_ready()
            with self.gripper_lock:
                self.gripper.move(0.002, GRIPPER_SPEED)
                self.gripper_pos = 0.002
            print("[Gripper] Closed")
        except Exception as e:
            print(f"[ERROR] Gripper close: {e}")
            self.gripper_ready = False
            raise

    def gripper_move(self, position):
        try:
            self._ensure_gripper_ready()
            pos = float(np.clip(position, 0.002, GRIPPER_MAX_WIDTH))
            with self.gripper_lock:
                self.gripper.move(pos, GRIPPER_SPEED)
                self.gripper_pos = pos
            print(f"[Gripper] -> {pos:.4f} m")
        except Exception as e:
            print(f"[ERROR] Gripper move: {e}")
            self.gripper_ready = False
            raise

    def gripper_reset(self):
        try:
            if self.gripper:
                with self.gripper_lock:
                    self.gripper.stop()
                time.sleep(0.5)
            self.gripper_ready = False
            self._init_gripper_internal()
        except Exception as e:
            print(f"[ERROR] Gripper reset: {e}")
            raise

    def gripper_activate(self):
        self._init_gripper_internal()

    # =========================================================
    # Payload
    # =========================================================
    def set_load(self, mass, F_x_center_load, load_inertia):
        try:
            self.robot.set_load(
                mass,
                np.array(F_x_center_load, dtype=np.float64),
                np.array(load_inertia, dtype=np.float64).reshape(3, 3),
            )
            print(f"[Franka] Load set: {mass} kg")
        except Exception as e:
            print(f"[WARN] set_load: {e}")

    def shutdown(self):
        self.running = False
        try:
            if self.robot:
                self.robot.join_motion()
        except Exception:
            pass
        try:
            if self.robot:   self.robot.stop()
            if self.gripper: self.gripper.stop()
        except Exception:
            pass
        print(f"[Franka] Shutdown. updates={self.state_update_count} errors={self.state_error_count}")


# =========================================================
# Flask
# =========================================================
def main(_):
    flask_app = Flask(__name__)

    try:
        server = FrankaServer(FLAGS.robot_ip, FLAGS.reset_joint_target)
    except Exception as e:
        print(f"\n[FATAL] {e}")
        return

    # ---- Arm ----

    @flask_app.route("/pose", methods=["POST"])
    def pose():
        try:
            data = request.json
            server.move(data["arr"])
            return "Moved"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/pose_sync", methods=["POST"])
    def pose_sync():
        try:
            data = request.json
            timeout = float(data.get("timeout", 10.0))
            server.move_sync(data["arr"], timeout=timeout)
            return "Moved"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/startimp", methods=["POST"])
    def start_impedance():
        try:
            server.clear()
            server.start_impedance()
            return "Started impedance"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/stopimp", methods=["POST"])
    def stop_impedance():
        try:
            server.stop_impedance()
            return "Stopped impedance"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/jointreset", methods=["POST"])
    def joint_reset():
        try:
            async_call(server.reset_joint)
            return "Reset Joint"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/clearerr", methods=["POST"])
    def clear_errors():
        try:
            server.clear()
            return "Clear"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- State ----

    @flask_app.route("/getpos", methods=["POST"])
    def get_pos():
        try:
            with server.lock:
                return jsonify({"pose": server.pos.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getpos_euler", methods=["POST"])
    def get_pos_euler():
        try:
            with server.lock:
                xyz  = server.pos[:3].copy()
                quat = server.pos[3:].copy()
            norm = np.linalg.norm(quat)
            euler = R.from_quat(quat / norm).as_euler("xyz") if norm > 1e-8 else np.zeros(3)
            return jsonify({"pose": np.concatenate([xyz, euler]).tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getvel", methods=["POST"])
    def get_vel():
        try:
            with server.lock:
                return jsonify({"vel": server.vel.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getforce", methods=["POST"])
    def get_force():
        try:
            with server.lock:
                return jsonify({"force": server.force.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/gettorque", methods=["POST"])
    def get_torque():
        try:
            with server.lock:
                return jsonify({"torque": server.torque.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getq", methods=["POST"])
    def get_q():
        try:
            with server.lock:
                return jsonify({"q": server.q.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getdq", methods=["POST"])
    def get_dq():
        try:
            with server.lock:
                return jsonify({"dq": server.dq.tolist()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        with server.lock:
            j = server.jacobian.flatten().tolist()
        return jsonify({"jacobian": j})

    @flask_app.route("/getstate", methods=["POST"])
    def get_state():
        try:
            with server.lock:
                return jsonify({
                    "pose":        server.pos.tolist(),
                    "vel":         server.vel.tolist(),
                    "force":       server.force.tolist(),
                    "torque":      server.torque.tolist(),
                    "q":           server.q.tolist(),
                    "dq":          server.dq.tolist(),
                    "jacobian":    server.jacobian.flatten().tolist(),
                    "gripper_pos": server.gripper_pos,
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- Gripper ----

    @flask_app.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        try:
            async_call(server.gripper_activate)
            return "Activated"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        try:
            async_call(server.gripper_reset)
            return "Reset"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        try:
            async_call(server.gripper_open)
            return "Opened"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        try:
            async_call(server.gripper_close)
            return "Closed"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        try:
            async_call(server.gripper_close)
            return "Closed"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/move_gripper", methods=["POST"])
    def move_gripper():
        try:
            normalized = int(request.json["gripper_pos"])
            meters = (np.clip(normalized, 0, 255) / 255.0) * GRIPPER_MAX_WIDTH
            async_call(lambda: server.gripper_move(meters))
            return "Moved Gripper"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/get_gripper", methods=["POST"])
    def get_gripper():
        try:
            return jsonify({"gripper": server.gripper_pos})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- Payload ----

    @flask_app.route("/set_load", methods=["POST"])
    def set_load():
        try:
            d = request.json
            server.set_load(
                d.get("mass", 0.0),
                d.get("F_x_center_load", [0, 0, 0]),
                d.get("load_inertia", [[0]*3]*3),
            )
            return "Set Load"
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/update_param", methods=["POST"])
    def update_param():
        return jsonify({"error": "update_param not supported with franky backend"}), 501

    print(f"\n{'='*60}")
    print(f"Franka Server (FR3 + Franky / Impedance)")
    print(f"   Flask : http://{FLAGS.flask_url}:5000")
    print(f"   Robot : {FLAGS.robot_ip}")
    print(f"   Gripper max: {GRIPPER_MAX_WIDTH*100:.0f} cm")
    print(f"   Status: {'Connected' if server.connected else 'Not Connected'}")
    print(f"   Motion: CartesianImpedanceMotion streamed @ {MOTION_HZ} Hz")
    print(f"{'='*60}\n")

    flask_app.run(host=FLAGS.flask_url, port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    app.run(main)
