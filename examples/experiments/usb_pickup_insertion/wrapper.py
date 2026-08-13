from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
import cv2
import queue
from franka_env.envs.franka_env import FrankaEnv

class USBEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        caps_by_serial = {}
        for cam_name, kwargs in name_serial_dict.items():
            serial_number = kwargs["serial_number"]
            if serial_number in caps_by_serial:
                self.cap[cam_name] = caps_by_serial[serial_number]
            else:
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
                caps_by_serial[serial_number] = cap

    def close_cameras(self):
        """Close each physical camera once."""
        try:
            for cap in set(self.cap.values()):
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def get_im(self):
        """Get images, reusing one physical camera frame for multiple crops."""
        images = {}
        display_images = {}
        full_res_images = {}
        frames_by_cap = {}

        for key, cap in self.cap.items():
            try:
                if cap not in frames_by_cap:
                    frames_by_cap[cap] = cap.read()
                rgb = frames_by_cap[cap]
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            self.img_queue.put(display_images)
        return images

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(1.0)
        
        # Move above the target pose
        target = copy.deepcopy(self.currpos)
        target[2] = self.config.TARGET_POSE[2] + 0.05
        self.interpolate_move(target, timeout=0.5)
        time.sleep(0.5)
        self.interpolate_move(self.config.TARGET_POSE, timeout=0.5)
        time.sleep(0.5)
        self._send_gripper_command(-1.0)

        self._update_currpos()
        reset_pose = copy.deepcopy(self.config.TARGET_POSE)
        reset_pose[1] += 0.04
        self.interpolate_move(reset_pose, timeout=0.5)

        obs, info = super().reset(**kwargs)
        self._send_gripper_command(1.0)
        time.sleep(1)
        self.success = False
        self._update_currpos()
        obs = self._get_obs()
        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info
