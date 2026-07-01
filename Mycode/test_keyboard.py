import numpy as np
from pynput import keyboard

class KeyboardTester:
    def __init__(self):
        self.key_states = {}
        self.gripper_enabled = True

        # 启动键盘监听
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        print("=" * 60)
        print("🎮 键盘控制测试脚本（匹配你的控制逻辑）")
        print("🎯 控制说明：")
        print("   W/S : X轴前后    A/D : Y轴左右    Q/E : Z轴上下")
        print("   ↑/↓ : 姿态俯仰    ←/→ : 姿态旋转")
        print("   F   : 夹爪闭合    G   : 夹爪张开")
        print("=" * 60)  # 这里修复了！把 = 改成 ()

    def _on_press(self, key):
        try:
            self.key_states[key.char] = True
        except AttributeError:
            self.key_states[key] = True

    def _on_release(self, key):
        try:
            self.key_states[key.char] = False
        except AttributeError:
            self.key_states[key] = False

    def get_action(self):
        # 完全匹配你的键盘输出逻辑
        action = np.zeros(6)
        step = 0.5

        # X轴 W/S
        if self.key_states.get('w', False):
            action[0] = step
        if self.key_states.get('s', False):
            action[0] = -step

        # Y轴 A/D
        if self.key_states.get('a', False):
            action[1] = step
        if self.key_states.get('d', False):
            action[1] = -step

        # Z轴 Q/E
        if self.key_states.get('q', False):
            action[2] = step
        if self.key_states.get('e', False):
            action[2] = -step

        # 姿态 ↑/↓
        if self.key_states.get(keyboard.Key.up, False):
            action[4] = step
        if self.key_states.get(keyboard.Key.down, False):
            action[4] = -step

        # 旋转 ←/→
        if self.key_states.get(keyboard.Key.left, False):
            action[5] = step
        if self.key_states.get(keyboard.Key.right, False):
            action[5] = -step

        # 夹爪 F/G
        left = self.key_states.get('f', False)
        right = self.key_states.get('g', False)

        return action, (left, right)

# ================== 测试主循环 ==================
if __name__ == "__main__":
    import time
    tester = KeyboardTester()

    while True:
        expert_a, (left, right) = tester.get_action()
        norm = np.linalg.norm(expert_a)
        intervened = norm > 0.001

        # 实时打印输出
        print(
            f"\r动作: {np.round(expert_a, 2)} | "
            f"夹爪 F={left} G={right} | "
            f"干预={intervened}",
            end="")

        time.sleep(0.05)
