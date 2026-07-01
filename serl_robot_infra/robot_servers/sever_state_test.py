#!/usr/bin/env python3
"""
Franka Server State Query Script

Real-time query of all robot state values
Compatible with both ROS and Franky implementations
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any

SERVER_URL = "http://127.0.0.1:5000"

class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def get_state() -> Dict[str, Any]:
    """
    Get all robot state information
    
    Returns:
        Dictionary with all state values or None if failed
    """
    try:
        response = requests.post(f"{SERVER_URL}/getstate", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.RED}✗ Failed to get state: {response.status_code}{Colors.RESET}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"{Colors.RED}✗ Cannot connect to server at {SERVER_URL}{Colors.RESET}")
        print(f"  Make sure the server is running: python3 franka_server_franky.py")
        return None
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {str(e)}{Colors.RESET}")
        return None


def format_vector(vec, precision=4, width=10):
    """Format vector for display"""
    if isinstance(vec, list):
        return "[" + ", ".join(f"{v:{width}.{precision}f}" for v in vec) + "]"
    return str(vec)


def format_matrix(mat, precision=4, width=8):
    """Format matrix for display"""
    if not isinstance(mat, list) or not mat:
        return str(mat)
    
    lines = []
    for row in mat:
        if isinstance(row, list):
            row_str = "[" + ", ".join(f"{v:{width}.{precision}f}" for v in row) + "]"
        else:
            row_str = str(row)
        lines.append(row_str)
    
    return "\n    ".join(lines)


def print_state(state: Dict[str, Any]):
    """Print formatted state information"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
    print(f"ROBOT STATE - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}{Colors.RESET}\n")
    
    # END-EFFECTOR POSE
    print(f"{Colors.BOLD}📍 END-EFFECTOR POSE{Colors.RESET}")
    pose = state.get("pose", [])
    if len(pose) == 7:
        print(f"  Position (x, y, z):        {format_vector(pose[:3], precision=6, width=10)}")
        print(f"  Quaternion (qx, qy, qz, qw): {format_vector(pose[3:], precision=6, width=10)}")
    else:
        print(f"  {pose}")
    
    # JOINT CONFIGURATION
    print(f"\n{Colors.BOLD}🔧 JOINT CONFIGURATION{Colors.RESET}")
    q = state.get("q", [])
    print(f"  Joint Angles (q):          {format_vector(q, precision=6)}")
    dq = state.get("dq", [])
    print(f"  Joint Velocities (dq):     {format_vector(dq, precision=6)}")
    
    # VELOCITIES
    print(f"\n{Colors.BOLD}⚡ VELOCITIES{Colors.RESET}")
    vel = state.get("vel", [])
    if len(vel) == 6:
        print(f"  Linear (vx, vy, vz):       {format_vector(vel[:3], precision=6)}")
        print(f"  Angular (wx, wy, wz):      {format_vector(vel[3:], precision=6)}")
    else:
        print(f"  {format_vector(vel, precision=6)}")
    
    # FORCES & TORQUES
    print(f"\n{Colors.BOLD}💪 FORCES & TORQUES{Colors.RESET}")
    force = state.get("force", [])
    print(f"  Force (Fx, Fy, Fz) [N]:    {format_vector(force, precision=6)}")
    torque = state.get("torque", [])
    print(f"  Torque (Tx, Ty, Tz) [Nm]:  {format_vector(torque, precision=6)}")
    
    # JACOBIAN MATRIX
    print(f"\n{Colors.BOLD}📐 JACOBIAN MATRIX{Colors.RESET}")
    jacobian = state.get("jacobian", [])
    if jacobian and len(jacobian) == 6:
        print(f"  Shape: 6×7")
        print(f"  {{")
        print(f"    {format_matrix(jacobian, precision=4, width=8)}")
        print(f"  }}")
    else:
        print(f"  Invalid Jacobian")
    
    # GRIPPER STATE
    print(f"\n{Colors.BOLD}🤲 GRIPPER STATE{Colors.RESET}")
    gripper_pos = state.get("gripper_pos", 0.0)
    gripper_percent = (gripper_pos / 0.04 * 100) if gripper_pos else 0
    print(f"  Position:                  {gripper_pos:.6f} m ({gripper_percent:.1f}%)")
    
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}\n")


def print_compact_state(state: Dict[str, Any]):
    """Print compact single-line state for monitoring"""
    pose = state.get("pose", [])
    q = state.get("q", [])
    force = state.get("force", [])
    gripper = state.get("gripper_pos", 0.0)
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    # Compact format: position | q_sample | force | gripper
    if len(pose) >= 3 and len(q) >= 7 and len(force) >= 3:
        pos_str = f"[{pose[0]:7.4f}, {pose[1]:7.4f}, {pose[2]:7.4f}]"
        q_str = f"[{q[0]:7.4f}, {q[3]:7.4f}, {q[6]:7.4f}]"
        f_str = f"[{force[0]:6.3f}, {force[1]:6.3f}, {force[2]:6.3f}]"
        g_str = f"{gripper:.4f}"
        
        print(f"{Colors.DIM}{timestamp}{Colors.RESET} | Pos: {pos_str} | Q: {q_str} | F: {f_str} | Grip: {g_str}")


def monitor_state(interval: float = 1.0, duration: float = None):
    """
    Monitor robot state continuously
    
    Args:
        interval: Update interval in seconds
        duration: Duration to monitor (None = infinite)
    """
    start_time = time.time()
    iteration = 0
    
    print(f"{Colors.BOLD}{Colors.CYAN}ROBOT STATE MONITOR (interval={interval}s){Colors.RESET}")
    print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}\n")
    
    try:
        while True:
            state = get_state()
            
            if state:
                iteration += 1
                print_compact_state(state)
            else:
                print(f"{Colors.RED}Failed to retrieve state{Colors.RESET}")
            
            # Check duration
            if duration and (time.time() - start_time) > duration:
                print(f"\n{Colors.YELLOW}Monitor duration reached{Colors.RESET}")
                break
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitoring stopped by user{Colors.RESET}")


def query_specific_state(endpoint: str, name: str):
    """Query a specific state endpoint"""
    try:
        response = requests.post(f"{SERVER_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
            print(f"  {json.dumps(data, indent=2)}")
            return data
        else:
            print(f"{Colors.RED}✗{Colors.RESET} {name} [{response.status_code}]")
            return None
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.RESET} {name} - {str(e)}")
        return None


def interactive_menu():
    """Interactive menu for state queries"""
    while True:
        print(f"\n{Colors.BOLD}{Colors.CYAN}FRANKA SERVER STATE QUERY{Colors.RESET}")
        print(f"{Colors.DIM}{'='*60}{Colors.RESET}")
        print("1. Get full state (all values)")
        print("2. Get pose (position + orientation)")
        print("3. Get pose (Euler angles)")
        print("4. Get joint angles")
        print("5. Get joint velocities")
        print("6. Get end-effector velocity")
        print("7. Get force")
        print("8. Get torque")
        print("9. Get Jacobian matrix")
        print("10. Get gripper position")
        print("11. Monitor state (real-time)")
        print("0. Exit")
        print(f"{Colors.DIM}{'='*60}{Colors.RESET}")
        
        choice = input(f"\n{Colors.CYAN}Select option (0-11): {Colors.RESET}").strip()
        
        if choice == "0":
            print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}\n")
            break
        
        elif choice == "1":
            state = get_state()
            if state:
                print_state(state)
        
        elif choice == "2":
            query_specific_state("/getpos", "End-Effector Pose (Quaternion)")
        
        elif choice == "3":
            query_specific_state("/getpos_euler", "End-Effector Pose (Euler)")
        
        elif choice == "4":
            query_specific_state("/getq", "Joint Angles")
        
        elif choice == "5":
            query_specific_state("/getdq", "Joint Velocities")
        
        elif choice == "6":
            query_specific_state("/getvel", "End-Effector Velocity")
        
        elif choice == "7":
            query_specific_state("/getforce", "Force")
        
        elif choice == "8":
            query_specific_state("/gettorque", "Torque")
        
        elif choice == "9":
            query_specific_state("/getjacobian", "Jacobian Matrix")
        
        elif choice == "10":
            query_specific_state("/get_gripper", "Gripper Position")
        
        elif choice == "11":
            interval_str = input(f"{Colors.CYAN}Update interval (seconds, default=1): {Colors.RESET}").strip()
            try:
                interval = float(interval_str) if interval_str else 1.0
            except ValueError:
                interval = 1.0
            monitor_state(interval=interval)
        
        else:
            print(f"{Colors.RED}Invalid option{Colors.RESET}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Command line mode
        command = sys.argv[1].lower()
        
        if command == "full" or command == "1":
            state = get_state()
            if state:
                print_state(state)
        
        elif command == "monitor":
            interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
            monitor_state(interval=interval)
        
        elif command == "pose":
            query_specific_state("/getpos", "End-Effector Pose")
        
        elif command == "q":
            query_specific_state("/getq", "Joint Angles")
        
        elif command == "force":
            query_specific_state("/getforce", "Force")
        
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python3 query_state.py full          # Get all state")
            print("  python3 query_state.py monitor [interval]  # Monitor continuously")
            print("  python3 query_state.py pose          # Get end-effector pose")
            print("  python3 query_state.py q             # Get joint angles")
            print("  python3 query_state.py force         # Get force")
            print("  python3 query_state.py               # Interactive menu")
    
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
