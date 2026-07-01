import time
import threading
import numpy as np
import pylibfranka

from robot_servers.gripper_server import GripperServer


class FrankaGripperServer(GripperServer):
    def __init__(self, robot_ip: str):
        super().__init__()
        self.gripper = pylibfranka.Gripper(robot_ip)
        self.binary_gripper_pose = 0
        self.gripper_pos = 0.0

        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="gripper_poll"
        )
        self._poll_thread.start()

    def open(self):
        if self.binary_gripper_pose == 0:
            return
        self.gripper.stop()
        self.gripper.move(0.09, 0.3)
        self.binary_gripper_pose = 0

    def close(self):
        if self.binary_gripper_pose == 1:
            return
        self.gripper.grasp(0.01, 0.3, 130, 1.0, 1.0)
        self.binary_gripper_pose = 1

    def close_slow(self):
        if self.binary_gripper_pose == 1:
            return
        self.gripper.grasp(0.01, 0.1, 130, 1.0, 1.0)
        self.binary_gripper_pose = 1

    def move(self, position: int):
        width = float(position / (255 * 10))  # [0, 0.1] m
        self.gripper.move(width, 0.3)

    def _poll_loop(self):
        while True:
            try:
                state = self.gripper.read_once()
                self.gripper_pos = state.width / 0.08
            except Exception:
                pass
            time.sleep(0.05)
