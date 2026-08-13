import time
import os
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        # ========== 键盘控制初始化（替换SpaceMouse）==========
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "Keyboard intervention requires pynput and an available X display. "
                "Run from a graphical session, or set fake_env=True / avoid "
                "SpacemouseIntervention in headless sessions."
            ) from exc

        self.keyboard = keyboard
        self.key_states = {}  # 记录按键状态
        self.key_press_times = {}
        self.key_latch_seconds = float(os.environ.get("SERL_KEY_LATCH_SECONDS", "0.15"))
        self.debug_keyboard = os.environ.get("SERL_KEYBOARD_DEBUG", "0") == "1"
        # 启动键盘监听（非阻塞）
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        if self.debug_keyboard:
            print("[KeyboardControl] listener started. Keys: W/S, A/D, R/F, arrows, O/C/V")
            print("[KeyboardControl] actions are interpreted in the wrapped env action frame")

        self.left, self.right = False, False
        self.action_indices = action_indices

    # 键盘按下
    def _on_press(self, key):
        try:
            key_name = key.char.lower()
        except AttributeError:
            key_name = key
        self.key_states[key_name] = True
        self.key_press_times[key_name] = time.time()
        if self.debug_keyboard:
            print(f"[KeyboardControl] press: {key_name}")

    # 键盘松开
    def _on_release(self, key):
        try:
            key_name = key.char.lower()
        except AttributeError:
            key_name = key
        self.key_states[key_name] = False
        if self.debug_keyboard:
            print(f"[KeyboardControl] release: {key_name}")

    # 获取键盘控制的动作（核心：完全替代SpaceMouse）
    def _get_keyboard_action(self):
        # 6D动作: [x, y, z, roll, pitch, yaw]
        action = np.zeros(6)
        step = 0.5  # 动作强度，可调节
        now = time.time()

        def active(key_name):
            return self.key_states.get(key_name, False) or (
                now - self.key_press_times.get(key_name, -np.inf) < self.key_latch_seconds
            )

        # ====== 位置控制 WASD + RF ======
        # X轴前后：W/S
        if active('w'):
            action[0] = step
        if active('s'):
            action[0] = -step

        # Y轴左右：A/D
        if active('a'):
            action[1] = step
        if active('d'):
            action[1] = -step

        # Z轴上下：R/F
        if active('r'):
            action[2] = step
        if active('f'):
            action[2] = -step

        # ====== 姿态控制 方向键 ======
        # 上下俯仰：↑/↓
        if active(self.keyboard.Key.up):
            action[4] = step
        if active(self.keyboard.Key.down):
            action[4] = -step

        # 左右旋转：←/→
        if active(self.keyboard.Key.left):
            action[5] = step
        if active(self.keyboard.Key.right):
            action[5] = -step

        # ====== 夹爪按键：O = 张开，C/V = 闭合 ======
        left = active('c') or active('v')
        right = active('o')

        return action, (left, right)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        接口完全不变：输入策略动作，返回(新动作, 是否替换)
        """
        # ========== 这里从键盘获取动作，替代原来的SpaceMouse ==========
        expert_a, buttons = self._get_keyboard_action()
        self.left, self.right = buttons
        intervened = False

        # 判断是否有手动控制
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        # 夹爪逻辑完全保留不变
        if self.gripper_enabled:
            if self.left:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            if self.debug_keyboard:
                print(f"[KeyboardControl] action: {np.round(expert_a, 3)}")
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

# class SpacemouseIntervention(gym.ActionWrapper):
#     def __init__(self, env, action_indices=None):
#         super().__init__(env)

#         self.gripper_enabled = True
#         if self.action_space.shape == (6,):
#             self.gripper_enabled = False

#         self.expert = SpaceMouseExpert()
#         self.left, self.right = False, False
#         self.action_indices = action_indices

#     def action(self, action: np.ndarray) -> np.ndarray:
#         """
#         Input:
#         - action: policy action
#         Output:
#         - action: spacemouse action if nonezero; else, policy action
#         """
#         expert_a, buttons = self.expert.get_action()
#         self.left, self.right = tuple(buttons)
#         intervened = False
        
#         if np.linalg.norm(expert_a) > 0.001:
#             intervened = True

#         if self.gripper_enabled:
#             if self.left:  # close gripper
#                 gripper_action = np.random.uniform(-1, -0.9, size=(1,))
#                 intervened = True
#             elif self.right:  # open gripper
#                 gripper_action = np.random.uniform(0.9, 1, size=(1,))
#                 intervened = True
#             else:
#                 gripper_action = np.zeros((1,))
#             expert_a = np.concatenate((expert_a, gripper_action), axis=0)

#         if self.action_indices is not None:
#             filtered_expert_a = np.zeros_like(expert_a)
#             filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
#             expert_a = filtered_expert_a

#         if intervened:
#             return expert_a, True

#         return action, False

#     def step(self, action):

#         new_action, replaced = self.action(action)

#         obs, rew, done, truncated, info = self.env.step(new_action)
#         if replaced:
#             info["intervene_action"] = new_action
#         info["left"] = self.left
#         info["right"] = self.right
#         return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
