from franky import *
import time

robot = Robot("172.16.0.2")  # Replace this with your robot's IP

# Let's start slow (this lets the robot use a maximum of 5% of its velocity, acceleration, and jerk limits)
robot.relative_dynamics_factor = 0.2
robot.recover_from_errors()

# Move the robot 20cm along the relative X-axis of its end-effector
while True:
    try:
        # z +=1
        # x -= 1   # 每周期后退1 mm

        pose = Affine(
            [0.5, 0.0, 0.4],
            [0.0, 0.0, 0.0, 1.0]
        )

        motion = CartesianMotion(pose)

        try:
            robot.poll_motion()
        except Exception:
            robot.recover_from_errors()

        robot.move(motion, asynchronous=True)

        time.sleep(1 / 30)

    except KeyboardInterrupt:
        break
