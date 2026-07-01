"""
Test script for franka_server_pylib.py — checks all Flask endpoints.
Usage: python test_franka_server.py [--host 127.0.0.1] [--port 5000]
"""

import requests
import numpy as np
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", default=5000, type=int)
args = parser.parse_args()

BASE = f"http://{args.host}:{args.port}"
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def post(route, json=None):
    try:
        r = requests.post(f"{BASE}{route}", json=json, timeout=5)
        r.raise_for_status()
        return r
    except Exception as e:
        return e


def check(name, route, json=None, *, key=None, length=None, text=None):
    r = post(route, json)
    if isinstance(r, Exception):
        print(f"{FAIL} {name}: {r}")
        return False

    if text is not None:
        ok = text in r.text
        print(f"{PASS if ok else FAIL} {name}: {r.text!r}")
        return ok

    try:
        data = r.json()
    except Exception:
        print(f"{FAIL} {name}: response not JSON — {r.text!r}")
        return False

    if key is None:
        print(f"{PASS} {name}: {data}")
        return True

    if key not in data:
        print(f"{FAIL} {name}: missing key '{key}' in {data}")
        return False

    val = data[key]
    if length is not None and len(val) != length:
        print(f"{FAIL} {name}: expected length {length}, got {len(val)} — {val}")
        return False

    print(f"{PASS} {name}: {val}")
    return True


results = []

print(f"\n{'='*50}")
print(f"  Testing Franka server at {BASE}")
print(f"{'='*50}\n")

# --- State reads ---
results.append(check("getpos",      "/getpos",      key="pose",    length=7))
results.append(check("getpos_euler","/getpos_euler", key="pose",    length=6))
results.append(check("getvel",      "/getvel",      key="vel",     length=6))
results.append(check("getforce",    "/getforce",    key="force",   length=3))
results.append(check("gettorque",   "/gettorque",   key="torque",  length=3))
results.append(check("getq",        "/getq",        key="q",       length=7))
results.append(check("getdq",       "/getdq",       key="dq",      length=7))
results.append(check("getjacobian", "/getjacobian", key="jacobian", length=42))
results.append(check("get_gripper", "/get_gripper", key="gripper"))

# --- getstate (all-in-one) ---
r = post("/getstate")
if isinstance(r, Exception):
    print(f"{FAIL} getstate: {r}")
    results.append(False)
else:
    d = r.json()
    expected = {"pose": 7, "vel": 6, "force": 3, "torque": 3,
                "q": 7, "dq": 7, "jacobian": 42}
    ok = True
    for k, n in expected.items():
        if k not in d:
            print(f"{FAIL} getstate: missing key '{k}'")
            ok = False
        elif len(d[k]) != n:
            print(f"{FAIL} getstate[{k}]: expected {n}, got {len(d[k])}")
            ok = False
    if ok:
        print(f"{PASS} getstate: all keys present with correct lengths")
    results.append(ok)

# --- Pose command: send current pose back (no movement) ---
r = post("/getpos")
if not isinstance(r, Exception):
    current_pose = r.json()["pose"]
    results.append(check("pose (hold)", "/pose",
                        json={"arr": current_pose}, text="Moved"))
else:
    print(f"{FAIL} pose: could not read current pose first")
    results.append(False)

# --- Compliance update ---
results.append(check("update_param", "/update_param",
                    json={"translational_stiffness": 200.0,
                            "rotational_stiffness": 10.0},
                    text="Updated"))

# --- Error clear ---
results.append(check("clearerr", "/clearerr", text="Clear"))

# --- Gripper ---
results.append(check("open_gripper",       "/open_gripper",       text="Opened"))
results.append(check("close_gripper_slow", "/close_gripper_slow", text="Closed"))
results.append(check("open_gripper (2)",   "/open_gripper",       text="Opened"))
results.append(check("move_gripper(128)",  "/move_gripper",
                    json={"gripper_pos": 128}, text="Moved Gripper"))

# --- Summary ---
passed = sum(results)
total  = len(results)
print(f"\n{'='*50}")
print(f"  {passed}/{total} passed")
print(f"{'='*50}\n")

sys.exit(0 if passed == total else 1)
