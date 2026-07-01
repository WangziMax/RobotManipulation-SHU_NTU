from franky import Robot, Affine, CartesianMotion
import time

robot = Robot("172.16.0.2")
robot.recover_from_errors()
robot.relative_dynamics_factor = 0.2

x = 0.5
y = 0
z = 0.4

while True:
    try:
        # z +=1
        # x -= 1   # 每周期后退1 mm

        pose = Affine(
            [x, y, z],
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