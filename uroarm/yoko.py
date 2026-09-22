from pynput import keyboard
import math
import time

from util import *
from servo import *
from kinematics import *

params = read_params()
init_servo(params)

STEP = 5      # mm
MOVE_TIME = 0.05

hand_idx = nax - 1

x = 150
y = 0
z = 50

gripper = servo_to_angle(hand_idx, servo_angles[hand_idx])
GRIPPER_STEP = 2   # 1ループで2°動かす


def set_angle(ch, deg):
    servo_deg = angle_to_servo(ch, deg)
    set_servo_angle(ch, servo_deg)


def move_xyz_now(x, y, z):
    pose = get_pose_from_xyz(x, y, z)

    rad5s = inverse_kinematics(pose)

    if rad5s is None:
        return

    deg5s = degree(rad5s)

    for ch, deg in enumerate(deg5s):
        set_angle(ch, deg)


def get_pose_from_xyz(x, y, z):
    x_min = 0
    x_max = 500

    pitch_min = 90
    pitch_max = 60

    r = (x - x_min) / (x_max - x_min)

    pitch = r * pitch_max + (1 - r) * pitch_min

    yaw = -math.atan2(y, x)

    return [x, y, z, yaw, radian(pitch)]


# ----------------------------
# pynput キー監視
# ----------------------------

pressed = set()


def on_press(key):
    pressed.add(key)


def on_release(key):
    pressed.discard(key)


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()

# ----------------------------

print("""
↑↓ : X
←→ : Y
Q : Z+
A : Z-
R : Ready
O : Open
C : Close
ESC : Exit
""")

move_xyz_now(x, y, z)

while True:

    changed = False

    if keyboard.Key.up in pressed:
        x += STEP
        changed = True

    elif keyboard.Key.down in pressed:
        x -= STEP
        changed = True

    elif keyboard.Key.left in pressed:
        y += STEP
        changed = True

    elif keyboard.Key.right in pressed:
        y -= STEP
        changed = True

    elif keyboard.KeyCode.from_char('q') in pressed:
        z += STEP
        changed = True

    elif keyboard.KeyCode.from_char('a') in pressed:
        z -= STEP
        changed = True

    elif keyboard.KeyCode.from_char('r') in pressed:

        degs = [0, 0, -90, -90, 0, 0]

        for ch, deg in enumerate(degs):
            set_angle(ch, deg)

    elif keyboard.Key.esc in pressed:
        break

    if keyboard.KeyCode.from_char('o') in pressed:
        gripper += GRIPPER_STEP

        if gripper > 100:
            gripper = 100

        set_angle(hand_idx, gripper)

    if keyboard.KeyCode.from_char('c') in pressed:
        gripper -= GRIPPER_STEP

        if gripper < 30:
            gripper = 30

        set_angle(hand_idx, gripper)

    if changed:
        x = max(0, min(500, x))
        y = max(-150, min(150, y))
        z = max(0, min(150, z))

        move_xyz_now(x, y, z)

        print(f"X={x:.1f} Y={y:.1f} Z={z:.1f}")

    time.sleep(MOVE_TIME)

listener.stop()