from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

for package_root in ("serl_robot_infra", "serl_launcher"):
    package_path = str(REPO_ROOT / package_root)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
