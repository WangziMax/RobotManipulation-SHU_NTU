import pyrobotiqgripper as rq

#Create a Robotiq gripper object.
gripper = rq.RobotiqGripper(com_port="/dev/ttyUSB2")
gripper.activate()
gripper.open()
gripper.close()
