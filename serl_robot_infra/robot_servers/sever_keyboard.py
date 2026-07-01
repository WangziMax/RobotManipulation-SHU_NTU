#!/usr/bin/env python3

import time
import threading

import numpy as np
import requests
from pynput import keyboard

SERVER = "http://127.0.0.1:5000"

STEP = 0.002          # 2 mm
CONTROL_HZ = 50       # 50 Hz


pressed_keys = set()
running = True


def get_pose():
    r = requests.post(f"{SERVER}/getpos", timeout=2)
    r.raise_for_status()
    return np.array(r.json()["pose"], dtype=np.float64)


def send_pose(pose):
    requests.post(
        f"{SERVER}/pose",
        json={"arr": pose.tolist()},
        timeout=1,
    )


def open_gripper():
    requests.post(f"{SERVER}/open_gripper", timeout=1)


def close_gripper():
    requests.post(f"{SERVER}/close_gripper", timeout=1)


def on_press(key):
    global running

    try:
        k = key.char.lower()
        pressed_keys.add(k)

        if k == "r":
            print("Open gripper")
            open_gripper()

        elif k == "f":
            print("Close gripper")
            close_gripper()

        elif k == "p":
            pose = get_pose()
            print("Current pose:")
            print(np.round(pose, 4))

    except AttributeError:
        if key == keyboard.Key.esc:
            running = False
            return False


def on_release(key):
    try:
        k = key.char.lower()
        pressed_keys.discard(k)
    except Exception:
        pass


def main():
    global running

    print("\n==============================")
    print("FR3 Keyboard Controller")
    print("==============================")
    print("W/S : +X / -X")
    print("A/D : +Y / -Y")
    print("Q/E : +Z / -Z")
    print("R   : Open Gripper")
    print("F   : Close Gripper")
    print("P   : Print Pose")
    print("ESC : Exit")
    print("==============================\n")

    pose = get_pose()

    print("Initial pose:")
    print(np.round(pose, 4))

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )
    listener.start()

    dt = 1.0 / CONTROL_HZ

    while running:

        moved = False

        if "w" in pressed_keys:
            pose[0] += STEP
            moved = True

        if "s" in pressed_keys:
            pose[0] -= STEP
            moved = True

        if "a" in pressed_keys:
            pose[1] += STEP
            moved = True

        if "d" in pressed_keys:
            pose[1] -= STEP
            moved = True

        if "q" in pressed_keys:
            pose[2] += STEP
            moved = True

        if "e" in pressed_keys:
            pose[2] -= STEP
            moved = True

        if moved:
            try:
                send_pose(pose)
            except Exception as exc:
                print("Send error:", exc)

        time.sleep(dt)

    listener.stop()
    print("Exit")


if __name__ == "__main__":
    main()