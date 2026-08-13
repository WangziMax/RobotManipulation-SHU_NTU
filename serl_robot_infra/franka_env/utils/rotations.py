from scipy.spatial.transform import Rotation as R


def quat_2_euler(quat):
    """Convert an ``[x, y, z, w]`` quaternion to XYZ Euler angles."""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    """Convert XYZ Euler angles to an ``[x, y, z, w]`` quaternion."""
    return R.from_euler("xyz", xyz).as_quat()
