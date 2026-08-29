#!/usr/bin/env python3
"""Keyboard test client for the Franka Cartesian velocity servo server."""

import argparse
from collections import deque
import threading
import time

import numpy as np
import requests
from pynput import keyboard


KEY_DIRECTIONS = {
    "w": np.array([1.0, 0.0, 0.0]),
    "s": np.array([-1.0, 0.0, 0.0]),
    "a": np.array([0.0, 1.0, 0.0]),
    "d": np.array([0.0, -1.0, 0.0]),
    "r": np.array([0.0, 0.0, 1.0]),
    "f": np.array([0.0, 0.0, -1.0]),
}

GRIPPER_COMMANDS = {
    "o": ("/open_gripper", "opened"),
    "c": ("/close_gripper", "closed"),
}


class RobotClient:
    def __init__(self, base_url, timeout):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def get_pose(self):
        response = self.session.post(
            f"{self.base_url}/getpos", timeout=self.timeout
        )
        response.raise_for_status()
        pose = np.asarray(response.json()["pose"], dtype=np.float64).reshape(-1)
        if pose.size != 7 or not np.all(np.isfinite(pose)):
            raise ValueError(f"server returned an invalid pose: {pose}")
        if np.linalg.norm(pose[3:]) <= 1e-8:
            raise ValueError("server returned a zero pose quaternion")
        return pose

    def send_pose(self, pose):
        response = self.session.post(
            f"{self.base_url}/pose",
            json={"arr": pose.tolist()},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def send_gripper_command(self, endpoint):
        response = self.session.post(
            f"{self.base_url}{endpoint}", timeout=self.timeout
        )
        response.raise_for_status()


class KeyboardState:
    def __init__(self):
        self._pressed = set()
        self._gripper_commands = deque()
        self._lock = threading.Lock()
        self.stop_event = threading.Event()

    def on_press(self, key):
        try:
            name = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.esc:
                self.stop_event.set()
                return False
            return None

        if name == "x":
            self.stop_event.set()
            return False

        with self._lock:
            first_press = name not in self._pressed
            self._pressed.add(name)
            if first_press and name in GRIPPER_COMMANDS:
                self._gripper_commands.append(GRIPPER_COMMANDS[name])
        return None

    def on_release(self, key):
        try:
            name = key.char.lower()
        except AttributeError:
            return None

        with self._lock:
            self._pressed.discard(name)
        return None

    def direction(self):
        with self._lock:
            pressed = self._pressed.copy()

        direction = sum(
            (value for name, value in KEY_DIRECTIONS.items() if name in pressed),
            np.zeros(3, dtype=np.float64),
        )
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 0.0 else direction

    def pop_gripper_commands(self):
        with self._lock:
            commands = list(self._gripper_commands)
            self._gripper_commands.clear()
        return commands


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move the Franka end effector with the keyboard."
    )
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--hz", type=positive_float, default=10.0)
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=0.02,
        help="target translation speed in m/s (default: 0.02)",
    )
    parser.add_argument(
        "--max-target-offset",
        type=positive_float,
        default=0.01,
        help="maximum target-to-actual position error in m (default: 0.01)",
    )
    parser.add_argument("--timeout", type=positive_float, default=1.0)
    return parser.parse_args()


def clamp_target(
    target_xyz,
    actual_xyz,
    max_target_offset,
):
    offset = target_xyz - actual_xyz
    offset_norm = np.linalg.norm(offset)
    if offset_norm > max_target_offset:
        target_xyz = actual_xyz + offset * (max_target_offset / offset_norm)
    return target_xyz


def print_controls(args, initial_pose):
    print("\nCartesian velocity keyboard test")
    print("  W / S  : +X / -X")
    print("  A / D  : +Y / -Y")
    print("  R / F  : +Z / -Z")
    print("  O / C  : open / close gripper")
    print("  X / Esc: stop and exit")
    print(f"\nTarget speed: {args.speed * 1000.0:.1f} mm/s")
    print(f"Control rate: {args.hz:.1f} Hz")
    print(f"Initial pose: {np.round(initial_pose, 4)}")
    print("XYZ travel limit: disabled\n")


def main():
    args = parse_args()
    client = RobotClient(args.url, args.timeout)
    key_state = KeyboardState()

    try:
        initial_pose = client.get_pose()
    except Exception as exc:
        raise SystemExit(f"cannot read robot pose from {args.url}: {exc}") from exc

    target_pose = initial_pose.copy()
    print_controls(args, initial_pose)

    listener = keyboard.Listener(
        on_press=key_state.on_press,
        on_release=key_state.on_release,
    )
    listener.start()

    period = 1.0 / args.hz
    last_tick = time.monotonic()
    last_status = 0.0
    consecutive_errors = 0

    try:
        while not key_state.stop_event.is_set():
            tick_start = time.monotonic()
            dt = min(tick_start - last_tick, 2.0 * period)
            last_tick = tick_start

            try:
                actual_pose = client.get_pose()
                target_pose[:3] += key_state.direction() * args.speed * dt
                target_pose[:3] = clamp_target(
                    target_pose[:3],
                    actual_pose[:3],
                    args.max_target_offset,
                )

                # Re-send the target so the 200 ms velocity command stays active.
                client.send_pose(target_pose)
                consecutive_errors = 0

                for endpoint, action in key_state.pop_gripper_commands():
                    try:
                        client.send_gripper_command(endpoint)
                        print(f"\nGripper {action}.")
                    except requests.RequestException as exc:
                        print(f"\nGripper command failed ({action}): {exc}")

                if tick_start - last_status >= 0.25:
                    print(
                        "\rActual XYZ: "
                        f"{actual_pose[0]: .4f} {actual_pose[1]: .4f} "
                        f"{actual_pose[2]: .4f} | Target XYZ: "
                        f"{target_pose[0]: .4f} {target_pose[1]: .4f} "
                        f"{target_pose[2]: .4f}",
                        end="",
                        flush=True,
                    )
                    last_status = tick_start
            except (requests.RequestException, KeyError, ValueError) as exc:
                consecutive_errors += 1
                print(f"\nServer request failed ({consecutive_errors}/3): {exc}")
                if consecutive_errors >= 3:
                    raise RuntimeError("stopping after three server errors") from exc

            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        key_state.stop_event.set()
    finally:
        listener.stop()
        try:
            actual_pose = client.get_pose()
            client.send_pose(actual_pose)
        except Exception as exc:
            print(f"\nCould not send the final zero-velocity target: {exc}")
        print("\nKeyboard velocity test stopped.")


if __name__ == "__main__":
    main()
