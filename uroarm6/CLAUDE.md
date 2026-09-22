# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

uroarm is a small open-source robot-arm controller: a desktop GUI (PySimpleGUI) drives a physical
pick-and-place arm over serial, using a camera + ArUco markers for hand-eye calibration and a YOLO
model to detect and sort colored/shaped objects into boxes. It's a hobby/contest project, not a
package — there is no `requirements.txt`/`pyproject.toml`; dependencies live only in `venv/`.

## Running

- Python: use `venv/Scripts/python.exe` (Windows). Bare `python` is not on PATH; `py` resolves to a
  different, unrelated system install (3.12) that lacks the project's dependencies.
- Main app: `venv/Scripts/python.exe arm.py` — requires the real hardware (servo controller on the
  `COM` port from `data/arm.json`, plus the camera at `camera-index`). `init_servo()` calls
  `sys.exit(0)` immediately if the serial port can't be opened, and `init_markers()`/vision code
  assumes `data/arm.json` already has a `cameras` calibration entry.
- Other standalone entry points, each with its own `if __name__ == '__main__':` GUI/loop:
  `servo.py` (raw per-servo jog), `angle.py` (servo-to-arm-degree calibration), `infer.py` (YOLO
  detection preview), `board.py`/`print.py` (ArUco/ChArUco board generation for camera calibration).
- No test suite, linter, or build step. `test_yolo.py` is a manual one-off script, not a test file.
- There is no hardware or camera in most dev environments, so `arm.py` cannot simply be launched and
  observed. To validate changes, stub `servo`/`camera`/`marker` (and `infer.Inference` for vision),
  monkeypatch `arm.read_params`/`arm.write_params` to avoid touching the real `data/arm.json`, then
  `exec` the code under `arm.py`'s `if __name__ == '__main__':` block with a scripted sequence of
  `window.read()` events (drive `sg.Window` through a subclass that returns queued events instead of
  waiting on the real GUI).
- `data/arm.json` holds live machine state (current servo positions, calibration, box positions) and
  is gitignored. Treat it as owned by whatever `arm.py` process is running — don't hand-edit it while
  the app might be live, and prefer stubbing it out entirely in throwaway test scripts rather than
  reading/writing the real file.

## Architecture

### Hardware chain

`arm.py`/`servo.py` → serial (`COM` port, 115200 baud, lines like `"<channel>,<degree>\r"`) →
a Raspberry Pi Pico running `pico/main.py` (MicroPython) → I2C → a PCA9685 PWM driver → servos.
The Pico firmware is dumb: it just parses `channel,degrees` and calls `Servos.position()`; all the
kinematics, angle calibration, and motion planning happen on the PC side.

### Two-layer angle model

Every joint has two representations, converted via a per-channel linear fit stored in
`data/arm.json`'s `servo-angle` array (`[coef, intercept]` per channel, indexed 0..`nax-1`):

- **servo units** — whatever raw degree value goes out over serial (`servo_angles` in `servo.py`).
- **arm-degrees** — the geometric joint angle used everywhere else (kinematics, GUI, jogging).

`angle_to_servo`/`servo_to_angle` (`servo.py`) convert between them; `angle.py` is the GUI used to
*derive* that linear fit by jogging each joint to two known reference angles (`datum_angles`) and
reading back the resulting servo values. Because the fit is per-channel and linear, remounting a
servo horn only requires re-running that joint's calibration in `angle.py` - nothing else changes.

### Joint count (`nax`)

`nax` (`util.py`, currently 7) is the single source of truth for the number of servos: it sizes
`data/arm.json`'s `prev-servo`/`servo-angle` arrays, `util.jKeys`/`servo_angle_keys`, and the
per-channel loops in `servo.py`/`arm.py`/`angle.py` (`range(nax)`, or iterating `servo_angles`,
whose length also derives from `nax`). The last channel (`hand_idx = nax - 1`, J7) is
always the gripper open/close actuator, not a positioning joint - `kinematics.py` only models the
other `nax - 1` channels (`kinematics.narm`). **Adding a servo means growing `nax` by one,
inserting the new channel before `hand_idx`, extending `data/arm.json`'s arrays and `angle.py`'s
`datum_angles`, and - the real work - extending `kinematics.py`'s forward/inverse solve** (see
below), since it's hand-derived for this specific arm and doesn't generalize automatically.

### Kinematics (`kinematics.py`)

6 arm joints (`narm = nax - 1`) plus the gripper: **J1** base yaw, **J2** shoulder pitch,
**J3** elbow pitch, **J4** forearm roll, **J5** wrist pitch, **J6** wrist roll, **J7** gripper.
Link lengths `L0..L5` (mm): base height, upper arm, elbow->J4 roll axis, roll axis->wrist,
wrist perpendicular offset (0 here), wrist->TCP.

J4's roll axis runs *along* the forearm, so the wrist centre sits on it and does not move when J4
turns - which means J4/J5/J6 form a **spherical wrist** and position decouples cleanly from
orientation. `inverse_kinematics` exploits that: J1-J3 place the wrist centre (law of cosines for
the elbow, as before), then J4-J6 come out of one X-Y-X euler split of the leftover rotation. It
is a closed-form solve for this specific geometry, not a generic DH/Jacobian solver. Because the
roll axis is colinear with the forearm, only `LF = L2 + L3` matters for position - where along the
forearm the servo sits does not.

The pose is 6-DOF: `pose_keys` in `util.py` are `X, Y, Z, R1, R2, R3` - position, roll about the
tool axis, pitch (90 = straight down), and how far the tool tilts out of the vertical plane the
arm swings in. Orientation is `rot_y(R2) @ rot_z(R3) @ rot_x(R1)`: **pitch is applied before yaw
on purpose**, so that tool-straight-down stays an ordinary value rather than a gimbal-lock
singularity and R2 can still run past 90 as it always could. **R3 = 0 reproduces the pre-J4 arm
exactly** - both FK and IK - so old behaviour is a special case, not an approximation.

`forward_kinematics` self-checks by running `inverse_kinematics` on its own output, re-running
`fk_pose` on that, and warning if the *poses* disagree by >0.1 (mm/deg). It compares poses rather
than joints because a target the tool reaches pointing back over the base genuinely has two valid
arm configurations; IK prefers the forward-reaching one and falls back to the other only if the
shoulder and elbow can't make it. Every branch is tried for an exact fit before any is allowed to
stretch to full extension, so a marginal target isn't answered with the wrong branch pulled
straight.

### Motion as generators

Every move (`move_joint`, `move_all_joints`, `move_linear`/`move_xyz`, and composites like `grab`)
is a Python generator that `yield`s once per interpolation step and sets a shared `is_moving` flag
while running. `arm.py`'s main loop holds at most one active generator in `moving` and advances it
one step (`moving.__next__()`) per GUI tick (`window.read(timeout=20)`), which is what keeps the GUI
responsive and the camera feed live during a multi-second move. `parallel(*gens)` steps several
motion generators together (e.g. opening the gripper while the arm is still translating). Anything
that needs to run "at the same time" as a move should be composed this way rather than blocking.

### Vision pipeline

`infer.py` runs a YOLO model (`contest.pt`) in a separate `multiprocessing.Process`
(`Inference` class) so detection doesn't stall the GUI loop; frames are pushed in and results pulled
back non-blockingly (`img_que`/`result_que`). Model classes are named `"{color}_{shape}"`
(e.g. `red_cube`); `arm.py` matches on either half independently (color-only or shape-only
filtering) by splitting on the first `_`. Separately, `marker.py` detects ArUco markers used for two
things: markers 0-2 define a reference plane on the work surface, and markers 3+ are mounted on the
gripper to give its live screen position, which `arm.py`'s `Adjust XY` (`calibrate_xy`) uses to fit a
linear regression from screen coordinates to arm coordinates (the hand-eye calibration stored under
`data/arm.json`'s `hand-eye` key).

### `arm.py`'s per-item tunables

Several small per-class/per-color knobs (box positions, a constant XYZ pick correction, per-item
gripper close angle) are stored as plain dicts loaded from and saved back to `data/arm.json`,
editable live from the GUI (spins/dialogs) rather than being hardcoded constants - follow that
pattern for new hardware-tuned values instead of adding module-level constants that require a code
edit to adjust.

## File naming note

Files with a `のコピー`/`- コピー` ("- copy") suffix, and the top-level `data/*.json`/`*.pt` model
files, are untracked local backups/artifacts, not part of the maintained codebase - don't treat them
as a second implementation to keep in sync.
