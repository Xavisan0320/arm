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

    rads = inverse_kinematics(pose)

    if rads is None:
        return

    for ch, deg in enumerate(degree(rads)):
        set_angle(ch, deg)


# The gripper hangs straight down, same as arm.py.
PICK_PITCH = 90


def get_pose_from_xyz(x, y, z):
    # R1 comes from facing_r1(): a tool roll that cancels the base yaw, so
    # the gripper keeps reading as facing forward at any Y instead of
    # swinging round J1 in an arc - see arm.py's get_pose_from_xyz for the
    # full reasoning (R2 = 90 makes "roll about the tool axis" and "base
    # yaw" the same rotation, so this is an exact cancellation where the
    # wrist's own travel allows it, tapering off only for a wide swing).
    r1 = facing_r1(x, y, z, radian(PICK_PITCH))
    return [x, y, z, r1, radian(PICK_PITCH), 0.0]


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

        # Same Ready pose as arm.py's move_to_ready - see the reasoning there.
        # -90/-90 put the TCP 20mm *below* the table and parked J3 0.1 degrees
        # off its travel limit; -75/-105 keeps the tool vertical but lifts the
        # TCP clear. Keep the two in step.
        #        J1 J2   J3 J4    J5 J6 J7
        degs = [ 0, 0, -75,  0, -105, 0, 0 ]

        for ch, deg in enumerate(degs):
            set_angle(ch, deg)

    elif keyboard.Key.esc in pressed:
        break

    if keyboard.KeyCode.from_char('o') in pressed:
        gripper += GRIPPER_STEP

        if gripper > GRIP_OPEN:
            gripper = GRIP_OPEN

        set_angle(hand_idx, gripper)

    if keyboard.KeyCode.from_char('c') in pressed:
        gripper -= GRIPPER_STEP

        if gripper < GRIP_CLOSE_MIN:
            gripper = GRIP_CLOSE_MIN

        set_angle(hand_idx, gripper)

    if changed:
        x = max(0, min(500, x))
        y = max(-150, min(150, y))
        z = max(0, min(150, z))

        move_xyz_now(x, y, z)

        print(f"X={x:.1f} Y={y:.1f} Z={z:.1f}")

    time.sleep(MOVE_TIME)

listener.stop()