import sys
import time
import math
import numpy as np
import PySimpleGUI as sg
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, closeCamera, getCameraFrame
from util import nax, jKeys, pose_keys, read_params, radian, write_params, get_move_time, set_move_time, spin, degree, Vec2, Vec3, sleep, get_pose, show_pose, GRIP_CLOSE_MIN, GRIP_OPEN
from servo import init_servo, set_servo_angle, angle_to_servo, servo_to_angle, servo_angles
from kinematics import forward_kinematics, inverse_kinematics, nudge_into_reach, joint_travel, facing_r1, normalize_radian
from marker import init_markers, detect_markers

hand_idx = nax - 1

PICK_Z = 15
PLACE_Z = 25

# How high the hand carries a work between picks. Holding the tool vertical is
# what limits this: the wrist pitch J5 has to make up the whole angle between
# the forearm and straight down, and its travel runs out at about -117 arm-deg.
# Z = 35 needs -116 and just fits; Z = 40 wants -119 and Z = 50 wants -127, so
# the old LIFT_Z of 50 had no in-travel solution at all - J5 was being clamped,
# which is why the hand travelled with the wrist half straight. Re-mounting the
# J5 horn one spline (~13 deg) toward the negative side would buy back Z = 50.
LIFT_Z = 35

# How high grab()/goto_box() carry the hand between the item and the box
# (2026-09-21: the user wanted it higher, before and after the pick - LIFT_Z
# only clears the table by 35mm). Height costs reach, though: the vertical
# hand reaches radius ~220 at Z=35 but only ~201 at Z=55 and ~186 at Z=65, so a
# fixed higher carry would drag far items and the red box (radius 195) inward.
# carry_z() instead goes as high as CARRY_Z wherever the arm can still hold
# the exact spot, and only comes down (never below LIFT_Z) near the edge.
CARRY_Z = 55

# Far-reach tilt (2026-09-21). The vertical gripper runs out of reach at a
# radius of ~237mm at pick depth and ~220 at LIFT_Z (the arm is simply fully
# stretched), so anything placed further out used to be missed outright. Only
# for such targets, grab() leans the gripper outward - by the smallest angle
# that reaches the exact spot at both pick depth and LIFT_Z, never more than
# this. Each ~10 degrees of lean buys ~30mm more reach. Near-side targets and
# everything inside vertical reach are untouched (still straight down).
MAX_REACH_TILT_DEG = 30

# Sorting drops the work into the box from this far above the box's
# registered Z, instead of lowering all the way in (the user's call: a box
# doesn't need a placed drop, and it saves a slow descent). Lowered on its
# own where the arm can't hold that height over the box.
BOX_DROP_HEIGHT = 50

# The gripper hangs straight down for every pick and place - 90 is the tool
# pointing at the ground. It used to be raked back progressively (90 at X_MIN
# to 60 at X_MAX) to buy reach, but a tilted gripper meets the work at an
# angle, and it also made the hand-eye fit conflate position with orientation,
# since the marker moves on screen when the hand tilts.
PICK_PITCH = 90

# Reachable X range for get_pose_from_xyz (mm). Holding the tool vertical costs
# reach - the wrist centre then sits L5 = 180mm straight above the TCP - and
# the servo travel costs more. Measured in-travel band, tool vertical, with
# the 2026-09-21 L0 = 85 correction:
#   Z = 5  radius 156..237    Z = 15 (pick) 149..233    Z = 35 (lift) 132..220
# Those are radii, not X: what actually decides reach is sqrt(x*x + y*y), and
# nudge_into_reach() enforces it against the real solve. X_MIN/X_MAX is only a
# coarse box on top of that - and get_pose_from_xyz clamps x into it BEFORE
# the reach check, so X_MAX must not sit inside the real reach. It was 180
# (right for the old L0 = 55), which silently dragged every reachable target
# at x = 180..220 back to x = 180: a pick error growing with distance.
# 215 stays inside the lift-height reach (220) straight ahead.
# (The old X_MAX of 300 was not reachable even with the hand raked back to 60.)
X_MIN = 100
X_MAX = 215


# Adjust XY (calibrate_xy) sample grid for the screen<->arm hand-eye fit:
# CALIB_RADII x CALIB_BEARINGS_DEG, i.e. 15 points spread over the arc the
# vertical gripper can actually reach at table height - which is where the
# work gets placed.
#
# It used to be a 2x2 box (X 155/185 - clamped to 180 by the old X_MAX - and
# Y +-50), measured twice. That covers 25mm x 100mm; everything else was
# extrapolated. Checked against the real run's data (2026-09-21): only 3
# distinct points survived, the same point landed 1-4mm apart between the two
# passes, and fits from each pass alone disagreed by up to ~13mm at the edges
# of the work area - the "error depends on where it is" the user saw. Covering
# the area turns extrapolation into interpolation.
#
# Every point here holds (no reach nudging, IK in travel) at Z_MIN, PICK_Z,
# PLACE_Z and LIFT_Z; radius 160 already fails at Z = 5, and past +-45 degrees
# the inner radii fail. Points the camera cannot see are skipped on their own
# (see MARKER_WAIT_SEC) - the fit just uses the rest.
CALIB_RADII = (165, 188, 210)
CALIB_BEARINGS_DEG = (-45, -22.5, 0, 22.5, 45)

# Adjust XY's far stations (2026-09-21). Past the vertical gripper's reach
# (radius ~215+) every grab leans outward, and the straight-line calibration -
# fitted only on radius 165..210 - was extrapolating out there: leaned picks
# overshot, a lot (confirmed by the user; the tool-length alternative was
# ruled out by measuring the shoulder at ~85mm, matching L0). These stations
# lean too (grab_pitch), which is what lets the calibration see that region
# at all, and stay within +-22.5 deg: a first attempt (a camera-model
# calibration, rolled back) went out to +-45 and the gripper left the
# camera's view in Y.
# 320 reaches the full MAX_REACH_TILT_DEG lean - without it the 30 deg picks
# were still outside everything calibrated, and overshot a little.
CALIB_FAR_RADII = (240, 270, 300, 320)
CALIB_FAR_BEARINGS_DEG = (22.5, 0, -22.5)

# The far stations are fitted with a homography (a plane seen in perspective
# - which the straight-line map cannot represent, hence the overshoot). If it
# also beats the straight line on the near samples (leave-one-out) it is used
# everywhere; otherwise picks inside this radius band keep the straight-line
# fit and are blended over to the homography across it, so near picks are
# never made worse by the far calibration.
HAND_EYE_BLEND = (205.0, 225.0)

# calibrate_xy waits for the camera to hand it a TCP height reading before each
# Z-search step. The main loop clears that reading to NaN every frame and only
# refills it when the arm is still AND every gripper marker (3+) is in view, so
# the wait has to be able to give up - otherwise one hidden marker hangs the
# whole calibration with no way out. Frames are captured at ~10/s, so this is
# about 50 chances to see the markers.
MARKER_WAIT_SEC = 5.0

# ...and the fit at the end of calibrate_xy is written straight back over
# data/arm.json's hand-eye entry, so a run that salvaged only a couple of
# usable points would quietly replace a good calibration with a bad one.
# A 2-input linear fit needs 3 points to even be determined; require more
# than the bare minimum before overwriting anything.
MIN_CALIB_POINTS = 6

# How far off target a Z-search point may still be when it has hit the Z_MIN
# floor and cannot go any lower, and still be used. Past this it is skipped.
Z_SEARCH_FLOOR_TOL = 8.0

# Sorting colors -> one physical box each. Shapes only filter detection; the
# box a work is dropped into is always chosen by its color.
SORT_COLORS = ['red', 'green', 'blue']
SHAPES = ['cube', 'ragby', 'pyramid']

COLOR_BTN_TEXT = {'red': '赤', 'green': '緑', 'blue': '青'}
SHAPE_BTN_TEXT = {'cube': 'キューブ', 'ragby': 'ラグビー', 'pyramid': 'ピラミッド'}
COLOR_LABEL = {None: '全部', 'red': '赤', 'green': '緑', 'blue': '青'}
SHAPE_LABEL = {None: '全部', 'cube': 'キューブ', 'ragby': 'ラグビー', 'pyramid': 'ピラミッド'}

DEFAULT_BOXES = {
    'red':   [150.0, -120.0, float(PLACE_Z)],
    'green': [200.0, -120.0, float(PLACE_Z)],
    'blue':  [250.0, -120.0, float(PLACE_Z)],
}

# Hand angle (arm-degrees) the gripper closes to when picking. Lower = tighter.
# Per class ("red_cube", ...); tuned live in the grip-settings dialog.
DEFAULT_GRIP_CLOSE = 30
GRIP_CLOSE_MAX = 60

# Keyboard jog: how fast the TCP / hand moves while a key is held down, and how
# long to keep coasting after the last repeat event before treating the key as
# released (PySimpleGUI only reports key-down, not key-up).
JOG_SPEED = 60.0        # mm/s
HAND_JOG_SPEED = 120.0  # deg/s
JOG_RELEASE_TIMEOUT = 0.2
Y_LIMIT = 200.0
# With the tool held vertical (jogging always keeps it vertical, same as a
# pick), J5's travel runs out around Z=49 - above that NOTHING is reachable
# at any radius, not just a narrower one, since the wrist pitch alone has to
# span the whole angle from the forearm to straight down. The old Z_MAX of
# 145 let the jog wander deep into that dead zone with no feedback, which is
# what made it look like jogging just stopped working.
Z_MIN, Z_MAX = 5.0, 45.0

# Manual control dialog (open_manual_control): full 6-DOF pose jog, an
# on-screen "joypad" and keyboard alike, both by holding a direction down -
# MANUAL_*_SPEED is the rate while held, same continuous-jog feel as the main
# window's own X/Y/Z jog (a tap is just a very short hold). Position/angle
# ranges match the debug panel's own pose spins.
MANUAL_POS_SPEED, MANUAL_ANGLE_SPEED = 60.0, 60.0                        # mm/s / deg/s while held

# A fixed-size nudge (mm or deg) applied on a plain click of a joypad/hand
# button, in addition to the press-and-hold jog above - belt and suspenders:
# it guarantees a click always does *something* even in an environment where
# the press/release binding this dialog otherwise relies on does not behave
# the way it does in testing (reported once already: held buttons did not
# move at all). Harmless overlap when holding does work - at most one extra
# small step layered on top of the continuous jog.
MANUAL_FALLBACK_STEP = 5.0
# Z's floor is 20, not 0: below that the arm frequently can't hold the pose
# at all (with R1/R2/R3 free to roam, unlike the main jog's fixed-vertical
# tool, there's no single X/Y-independent cutoff the way Z_MIN/Z_MAX are for
# the main window - 20 is a practical floor from hands-on testing, not a
# derived reach limit). Hitting it here just clamps like any other axis limit
# rather than silently failing to move.
MANUAL_RANGES = [ (0.0, 400.0), (-300.0, 300.0), (20.0, 150.0),          # X, Y, Z (mm)
                   (-90.0, 90.0), (0.0, 120.0), (-90.0, 90.0) ]          # R1, R2, R3 (deg)

# Safety cap: refuse a manual-control step that would move any single joint
# more than this in one go. One tick of ordinary jogging never needs anywhere
# near this much - it exists for the case where the target pose still solves,
# but only via a completely different arm configuration (confirmed: the base
# can end up swinging ~180 degrees round from a plain +/-5 degree Pitch
# change alone, when the branch the arm was already on runs out of reach) -
# see try_move()'s use of it.
MANUAL_MAX_STEP_DEG = 30.0

# Manual-control axis bindings: key name -> (index into the 6-value pose,
# sign). The first 3 are position (mm), the last 3 orientation (deg) - same
# split MANUAL_RANGES/MANUAL_POS_SPEED vs MANUAL_ANGLE_SPEED use. X/Y/Z reuse
# the main window's own arrow/PageUp/PageDown convention; R1/R2/R3 (roll,
# pitch, tilt) get their own keys since the main jog never touches them.
MANUAL_AXES = {
    'Up': (0, 1), 'Down': (0, -1), 'Left': (1, 1), 'Right': (1, -1),
    'Prior': (2, 1), 'Next': (2, -1),
    'z': (3, 1), 'x': (3, -1),        # R1 roll +/-
    'i': (4, 1), 'k': (4, -1),        # R2 pitch +/-
    'j': (5, 1), 'l': (5, -1),        # R3 tilt +/-
}

JOG_DIRS = {
    'Up':    (1, 0, 0),
    'Down':  (-1, 0, 0),
    'Left':  (0, 1, 0),
    'Right': (0, -1, 0),
    'Prior': (0, 0, 1),    # PageUp
    'Next':  (0, 0, -1),   # PageDown
}

# On-screen size of the camera panel (px).
CAM_VIEW = 560               # minimum camera view side (px); it grows with the window
CONTROL_COL_WIDTH = 440      # the control column on the right, scrollbar included
SCREEN_MARGIN_Y = 110        # screen height kept free for the title bar + taskbar

# Seconds to take over the final descent onto a work, and over lowering it into
# the box. Deliberately slower than a normal move: the hand is closing on
# something at the bottom of it, and a fast drop knocks the work over.
DESCEND_TIME = 3.0

# Set True to dump the interpolated joint angles of every linear move to data/ik.csv.
WRITE_IK_CSV = False


def set_angle(ch : int, deg : float):
    servo_deg = angle_to_servo(ch, deg)

    set_servo_angle(ch, servo_deg)

def move_joint(ch, dst):
    global is_moving

    is_moving = True

    src = servo_to_angle(ch, servo_angles[ch])

    move_time = get_move_time()
    start_time = time.time()
    while True:
        t = (time.time() - start_time) / move_time
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_angle(ch, deg)

        yield

    set_angle(ch, dst)
    is_moving = False

def move_all_joints(dsts):
    global is_moving

    is_moving = True

    srcs = [ servo_to_angle(ch, servo_angles[ch]) for ch in range(nax) ]

    move_time = get_move_time()
    start_time = time.time()
    while True:
        t = (time.time() - start_time) / move_time
        if 1 <= t:
            break

        for ch in range(nax):

            deg = t * dsts[ch] + (1 - t) * srcs[ch]

            set_angle(ch, deg)

        yield

    for ch in range(nax):
        set_angle(ch, dsts[ch])

    is_moving = False

def open_hand():
    for _ in move_joint(hand_idx, GRIP_OPEN):
        yield

def close_hand(deg=DEFAULT_GRIP_CLOSE):
    for _ in move_joint(hand_idx, deg):
        yield

def move_linear(dst, move_time=None):
    global move_linear_ok, is_moving

    is_moving = True
    move_linear_ok = True

    src = forward_kinematics(servo_angles)

    if move_time is None:
        move_time = get_move_time()

    f = None
    if WRITE_IK_CSV:
        try:
            f = open('data/ik.csv', 'w')
            f.write('time,' + ','.join(jKeys[:nax - 1]) + '\n')
        except OSError as e:
            print('ik.csv:', e)
            f = None

    start_time = time.time()
    while True:
        t = time.time() - start_time
        if move_time <= t:
            break

        r = t / move_time

        pose = [ r * d + (1 - r) * s for s, d in zip(src, dst) ]

        rads = inverse_kinematics(pose)

        # A straight line between two individually-reachable poses can still
        # cut through a spot the arm cannot hold - the joint-limited workspace
        # is not convex - so check every interpolated step against the real
        # travel too, not just whether some solution exists. When one fails,
        # skip commanding it rather than sending a servo past its travel: the
        # arm just holds where it is for this tick and picks the line back up
        # once the interpolation clears the pinch point.
        if rads is not None and any(
                not (lo <= degree(rad) <= hi) for rad, (lo, hi) in zip(rads, joint_travel())):
            rads = None

        if rads is None:
            move_linear_ok = False
        else:
            degs = degree(rads)
            if f is not None:
                f.write(f'{t},{",".join(["%.1f" % x for x in degs])}\n')

            for ch, deg in enumerate(degs):
                set_angle(ch, deg)

        yield

    if f is not None:
        f.close()

    is_moving = False

_last_reach_msg = None


def get_pose_from_xyz(x, y, z, r1=None, r2=None):
    """Build a pose from a target TCP position.

    x is clamped into the reachable range instead of raising, so a bad target
    degrades gracefully during a contest run.

    r1 is the tool roll; leave it as None (the default) to have
    kinematics.facing_r1() work it out. R2 = 90 makes the tool axis vertical,
    which makes "roll about it" and "base yaw" (J1) the same rotation - so
    left uncompensated (r1 = 0), the gripper would swing round with however
    far the base has to turn to reach (x, y): seen from a fixed camera in
    front, it visibly tilts off-square as soon as Y moves off 0, since J6 is
    tracing an arc round J1 rather than holding still. facing_r1 cancels
    that, tapering off only where a wide swing would otherwise exceed J6's
    own travel (see facing_r1) - so the gripper reads as facing forward at
    any Y, not just at calibrate_xy's own sample points. Pass an explicit r1
    to override this for a specific call, e.g. r1 = 0.0 if some future
    caller genuinely wants the old "roll follows the base" behaviour.

    r2 is the tool pitch in degrees; None means PICK_PITCH (straight down).
    grab() passes it explicitly (see grab_pitch), and an explicit r2 also
    lifts the X_MAX clamp: that box is only a coarse limit for jogging, and
    reach proper is enforced by nudge_into_reach below - a tilted grab
    reaches well past X_MAX, which is the whole point of tilting.
    """
    pitch = PICK_PITCH if r2 is None else r2
    xc = max(X_MIN, x) if r2 is not None else min(X_MAX, max(X_MIN, x))
    if abs(xc - x) > 0.5:
        print(f'get_pose_from_xyz: x {x:.1f} clamped to {xc:.1f}')
        try:
            set_state(f'目標Xが範囲外 ({x:.0f} -> {xc:.0f} に制限)')
        except NameError:
            pass  # window not created yet

    # X alone does not say whether a target is in reach: what the arm actually
    # runs out of is radius, and it runs out sooner the higher it is working.
    # A work off to the side can sit well outside the envelope while its X
    # still looks fine.
    #
    # This always checks reach at R1 = 0, regardless of what R1 ends up being
    # asked for below - position accuracy is what actually matters for a pick,
    # so how far out the arm can reach must not depend on how much roll
    # keeping the gripper camera-facing happens to want at this spot. Facing
    # is worked out afterwards, at whatever (xc, yc) reach alone settled on,
    # and degrades on its own (see facing_r1) rather than pulling the target
    # in any further than reach already required.
    xc, yc = nudge_into_reach(xc, y, z, radian(pitch))

    if math.hypot(xc - x, yc - y) > 0.5:
        # Only say it once per distinct adjustment - a retried move (e.g. each
        # Z-search trial in calibrate_xy) used to repeat the identical line
        # every time and bury the log.
        global _last_reach_msg
        msg = (f'get_pose_from_xyz: ({x:.1f},{y:.1f}) moved to '
               f'({xc:.1f},{yc:.1f}) at z {z:.1f} to stay in reach')
        if msg != _last_reach_msg:
            print(msg)
            _last_reach_msg = msg
        try:
            set_state(f'目標が届く範囲外 ({math.hypot(x, y):.0f} -> '
                      f'{math.hypot(xc, yc):.0f} mm に制限)')
        except NameError:
            pass  # window not created yet

    if r1 is None:
        r1 = facing_r1(xc, yc, z, radian(pitch))

    # R3 = 0 keeps the tool in the vertical plane the arm swings in, exactly
    # what the arm did before the J4 roll joint was added. A pitch below 90
    # therefore always leans the gripper radially outward, away from the base.
    pose = [ xc, yc, z, r1, radian(pitch), 0.0]

    return pose

def move_xyz_now(x, y, z, r1=None, r2=None):
    """Command the pose for (x, y, z) outright (no interpolation). Returns
    whether it actually held - i.e. whether the pose was reachable at all -
    so a caller like the jog loop can tell an ignored command from a real
    move rather than just going quiet."""
    pose = get_pose_from_xyz(x, y, z, r1, r2)
    rads = inverse_kinematics(pose)

    if rads is None:
        return False

    for ch, deg in enumerate(degree(rads)):
        set_angle(ch, deg)

    return True


def move_xyz(x, y, z, move_time=None, r1=None, r2=None):
    pose = get_pose_from_xyz(x, y, z, r1, r2)
    for _ in move_linear(pose, move_time):
        yield

def move_to_ready():
    # J3/J5 are -75/-105, not the -90/-90 this was originally. Two separate
    # problems with -90/-90, both found on the real arm:
    #
    #  - It put the forearm *dead horizontal* with the 180mm tool hanging off
    #    the end, so the TCP sat at Z = -20 - 20mm below the table, i.e. Ready
    #    parked the gripper into the work surface. -75/-105 lifts the TCP to
    #    Z = +19 (just clear of the table, and the same ballpark as the manual
    #    dialog's own Z floor) while keeping the tool vertical: the tool stays
    #    vertical as long as sigma + J5 = -90, where sigma is the forearm angle
    #    above horizontal, so J3 and J5 have to move together.
    #  - J3's calibrated travel is only -90.1..90.4, so -90 parked the elbow
    #    0.1 degrees off its hard limit - in the arm's own maximum-torque
    #    posture, where the whole forearm plus tool hangs on it. A servo held
    #    against its end stop under load buzzes and gets back-driven instead
    #    of holding position. Every joint now has 11+ degrees of margin.
    #
    # Anything here has to stay inside every joint's travel and keep the TCP
    # above the table - check both if these are ever retuned.
    #        J1 J2   J3 J4    J5 J6 J7
    degs = [ 0, 0, -75,  0, -105, 0, 0 ]

    set_phase('Ready へ移動中')
    for _ in move_all_joints(degs):
        yield


def parallel(*gens):
    """Step several move generators together, each tick, until all finish.
    Lets e.g. the gripper open while the arm is still translating."""
    gens = list(gens)
    while gens:
        for g in list(gens):
            try:
                next(g)
            except StopIteration:
                gens.remove(g)
        yield


def calib_grid():
    """Adjust XY's sample points, as rows (one per bearing) of (x, y). Rows
    alternate inward/outward so consecutive points are always neighbours -
    the lift-and-traverse between samples is a straight line, and a long
    chord across the arc could cut inside the reach annulus."""
    rows = []
    for i, bearing in enumerate(CALIB_BEARINGS_DEG):
        radii = CALIB_RADII if i % 2 == 0 else CALIB_RADII[::-1]
        rows.append([(r * math.cos(math.radians(bearing)), r * math.sin(math.radians(bearing)))
                     for r in radii])
    return rows


def calib_far_grid():
    """Adjust XY's far stations: CALIB_FAR_RADII x CALIB_FAR_BEARINGS_DEG,
    starting at +22.5 deg innermost - right next to where the near grid ends
    (+45 deg, outermost) - and snaking so each hop is to a neighbour."""
    rows = []
    for i, bearing in enumerate(CALIB_FAR_BEARINGS_DEG):
        radii = CALIB_FAR_RADII if i % 2 == 0 else CALIB_FAR_RADII[::-1]
        rows.append([(r * math.cos(math.radians(bearing)), r * math.sin(math.radians(bearing)))
                     for r in radii])
    return rows


def calib_stations():
    """Adjust XY's stations as rows of (x, y, r2): the near grid straight
    down (r2 None - exactly as before the far stations existed), then the far
    grid at the least outward lean (grab_pitch) that holds the spot from the
    Z floor up to LIFT_Z. A far station no lean can hold is left out."""
    rows = [[(x, y, None) for x, y in row] for row in calib_grid()]
    for row in calib_far_grid():
        far = []
        for x, y in row:
            r2 = grab_pitch(x, y, (Z_MIN, LIFT_Z))
            if r2 is None:
                print(f'calibrate_xy: far station ({x:.0f},{y:.0f}) is out of reach - left out')
            else:
                far.append((x, y, r2))
        rows.append(far)
    return rows


def fit_homography(src, dst):
    """3x3 H with dst ~ H @ src (normalised DLT) - screen pixels to arm XY
    on the marker plane."""
    def norm(pts):
        c = pts.mean(axis=0)
        k = math.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
        return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1.0]])
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    Ts, Td = norm(src), norm(dst)
    a = (Ts @ np.c_[src, np.ones(len(src))].T).T
    b = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (x, y, _), (X, Y, _) in zip(a, b):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])
    Hn = np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    q = (np.asarray(H, float) @ np.c_[pts, np.ones(len(pts))].T).T
    return q[:, :2] / q[:, 2:3]


def calibrate_xy():
    global tcp_height

    screen_coordinates = []
    robot_coordinates = []
    tcp_heights = []
    # One pass over many distinct points (see calib_grid) rather than two
    # passes over the same four: repeating a point only re-measures it, while
    # spreading out is what pins the fit down across the whole pick area.
    num_trial = 1
    move_time = 1

    leans = []

    for _ in range(num_trial):
        for row in calib_stations():
            for arm_x, arm_y, r2 in row:
                lean = 0.0 if r2 is None else PICK_PITCH - r2
                print(f'start move x:{arm_x:.1f} y:{arm_y:.1f}'
                      + (f' lean:{lean:.0f}' if lean else ''))

                # get_pose_from_xyz's default r1 (facing_r1) already keeps
                # the gripper facing forward here, same as everywhere else -
                # no override needed.
                arm_z = LIFT_Z
                for _ in move_xyz(arm_x, arm_y, arm_z, move_time, r2=r2):
                    yield

                print("move xy end")
                for _ in sleep(1):
                    yield
                tcp_height = np.nan
                converged = False

                for trial in range(20):
                    # Wait for a fresh reading, but not forever - see
                    # MARKER_WAIT_SEC. This used to be a bare
                    # `while np.isnan(tcp_height): yield`, which hangs the
                    # calibration indefinitely if a gripper marker stays out
                    # of view.
                    wait_until = time.time() + MARKER_WAIT_SEC
                    while np.isnan(tcp_height) and time.time() < wait_until:
                        yield

                    if np.isnan(tcp_height):
                        print(f'calibrate_xy: no TCP height reading at '
                              f'x:{arm_x:.1f} y:{arm_y:.1f} within '
                              f'{MARKER_WAIT_SEC:.0f}s - are the gripper '
                              f'markers (3+) in view?')
                        break

                    diff = tcp_height - PLACE_Z
                    print(f'move z trial:{trial} height:{tcp_height:.1f}')
                    if abs(diff) < 2:
                        converged = True
                        break

                    # Hard floor: whatever the height reading says, never
                    # command the gripper below the tabletop.
                    new_z = max(Z_MIN, arm_z - diff)

                    # Already on the floor and still asked to go lower: every
                    # remaining trial would command the exact same Z and read
                    # the same height, so stop now instead of burning ~2s a
                    # trial for all 20 (this is what made Adjust XY crawl).
                    # Close enough counts - the floor is as low as it can go.
                    if new_z == arm_z == Z_MIN:
                        converged = abs(diff) <= Z_SEARCH_FLOOR_TOL
                        print(f'calibrate_xy: at the Z floor ({Z_MIN}) with the gripper '
                              f'still {diff:.1f}mm above target - '
                              + ('accepting it' if converged else 'giving up on this point'))
                        break

                    arm_z = new_z
                    for _ in move_xyz(arm_x, arm_y, arm_z, move_time, r2=r2):
                        yield

                    for _ in sleep(1):
                        yield

                    tcp_height = np.nan
                else:
                    print(f'calibrate_xy: Z search did not converge at '
                          f'x:{arm_x:.1f} y:{arm_y:.1f} after 20 trials - '
                          f'check the plane markers (0-2) are steady and in view')

                # Only record a sample the search actually settled on, and
                # only while the gripper is still visible: get_tcp() returns
                # tcp_scr = None the moment any marker 3+ drops out, and this
                # went straight into tcp_scr.x, crashing the whole run with
                # AttributeError on NoneType. Note the loop above always
                # leaves tcp_height NaN when it runs out of trials (the last
                # thing the body does is reset it), which is where the
                # "move z end:nan" in that traceback came from - that NaN was
                # then being recorded as a real measurement.
                #
                # A bad sample is worse than a missing one: every sample here
                # feeds the hand-eye fit that overwrites data/arm.json below.
                if not converged or tcp_scr is None or np.isnan(tcp_height):
                    print(f'calibrate_xy: skipping the sample at '
                          f'x:{arm_x:.1f} y:{arm_y:.1f}'
                          + (' (gripper markers not visible)' if tcp_scr is None else ''))
                    for _ in move_xyz(arm_x, arm_y, LIFT_Z, move_time, r2=r2):
                        yield
                    continue

                print(f'move z end:{tcp_height:.1f}')

                # Record where the arm actually is, not where it was asked to
                # go: get_pose_from_xyz can pull an out-of-reach target in
                # ("moved to ... to stay in reach"), and recording the request
                # instead fed the hand-eye fit a coordinate the gripper was
                # never at.
                p = forward_kinematics(servo_angles)
                tcp_heights.append(tcp_height)
                robot_coordinates.append([float(p[0]), float(p[1]), float(p[2])])
                screen_coordinates.append([tcp_scr.x, tcp_scr.y])
                leans.append(lean)

                for _ in move_xyz(arm_x, arm_y, LIFT_Z, move_time, r2=r2):
                    yield

    for _ in move_to_ready():
        yield

    # The near (vertical) samples feed the straight-line fit exactly as before
    # the far stations existed; the far (leaned) ones only feed the homography.
    leans = np.array(leans)
    near = leans == 0
    far_uv = np.array(screen_coordinates)[~near]
    far_xyz = np.array(robot_coordinates)[~near]
    far_leans = leans[~near]
    all_uv = np.array(screen_coordinates)
    all_xyz = np.array(robot_coordinates)
    all_heights = np.array(tcp_heights)
    screen_coordinates = [c for c, n in zip(screen_coordinates, near) if n]
    robot_coordinates = [c for c, n in zip(robot_coordinates, near) if n]
    tcp_heights = [h for h, n in zip(tcp_heights, near) if n]

    # Refuse to fit from a handful of salvaged points - see MIN_CALIB_POINTS.
    # Leaving the old calibration in place is the safe failure here; silently
    # replacing it with a fit from 1-2 points is not.
    total = num_trial * sum(len(row) for row in calib_grid())
    if len(screen_coordinates) < MIN_CALIB_POINTS:
        print(f'calibrate_xy: only {len(screen_coordinates)} of {total} samples were '
              f'usable (need at least {MIN_CALIB_POINTS}) - NOT updating the hand-eye '
              f'calibration. The existing one is untouched. Check that the plane markers '
              f'(0-2) and the gripper markers (3+) are all steady and in view, then run '
              f'Adjust XY again.')
        return

    if len(screen_coordinates) < total:
        print(f'calibrate_xy: fitting from {len(screen_coordinates)} of {total} samples '
              f'({total - len(screen_coordinates)} skipped)')

    # predict arm coordinate from screen coordinate
    X = np.array(screen_coordinates)
    Y = np.array(robot_coordinates)

    model = LinearRegression().fit(X, Y)

    # An honest accuracy estimate: refit without each point in turn and see how
    # far off that left-out point is predicted. The in-sample residuals printed
    # further down cannot show this (a fit always agrees with the points it was
    # fitted to - with only 3 distinct points it agrees *exactly*). This is
    # roughly the pick error to expect from calibration alone.
    loo = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        m = LinearRegression().fit(X[keep], Y[keep])
        loo.append(np.hypot(*(m.predict(X[i:i + 1])[0, :2] - Y[i, :2])))
    loo = np.array(loo)
    worst = int(np.argmax(loo))
    print(f'calibrate_xy: expected XY accuracy (leave-one-out) - mean {loo.mean():.1f}mm, '
          f'worst {loo.max():.1f}mm at arm ({Y[worst, 0]:.0f}, {Y[worst, 1]:.0f})')

    print('get_params', type(model.get_params()), model.get_params())
    print('coef_', type(model.coef_), model.coef_)
    print('intercept_', type(model.intercept_), model.intercept_)

    hand_eye = {
        'coef': model.coef_.tolist(),
        'intercept': model.intercept_.tolist()
    }

    # ---- the far region: a homography over every sample. A leaned sample's
    # marker is not straight above the TCP: it sits marker_up * sin(lean)
    # back toward the base, marker_up being how far above the TCP the markers
    # ride (measured here from the near samples: camera height - TCP Z). The
    # straight-line fit's pairs are "marker pixel -> TCP of a vertical
    # gripper", so a leaned sample is recorded as the vertical-gripper TCP
    # whose marker would sit where this one does - otherwise the fit learns
    # to put the TCP marker_up * sin(lean) too far out, i.e. overshoot again.
    far_n = len(far_uv)
    if far_n >= 3:
        marker_up = float(np.median(np.array(tcp_heights) - Y[:, 2]))
        targets = all_xyz[:, :2].copy()
        for i, lean in enumerate(leans):
            if lean:
                bearing = math.atan2(all_xyz[i, 1], all_xyz[i, 0])
                back = marker_up * math.sin(math.radians(lean))
                targets[i] -= back * np.array([math.cos(bearing), math.sin(bearing)])

        H = fit_homography(all_uv, targets)

        def loo_h(idx):
            errs = []
            for i in idx:
                keep = np.arange(len(all_uv)) != i
                Hi = fit_homography(all_uv[keep], targets[keep])
                errs.append(np.linalg.norm(apply_homography(Hi, all_uv[i])[0] - targets[i]))
            return np.array(errs)

        h_near = loo_h(np.where(near)[0])
        h_far = loo_h(np.where(~near)[0])
        # how badly the straight-line fit misses the far samples - the overshoot
        line_far = np.linalg.norm((model.predict(far_uv)[:, :2]) - targets[~near], axis=1)
        everywhere = h_near.mean() <= loo.mean() + 0.5

        # The pick depth, per place. Each sample's commanded Z is where the
        # markers read PLACE_Z high - and that is NOT the same everywhere: the
        # arm sags under its own weight the further it reaches (measured on
        # the real arm: ~0-4mm up to radius 210, ~10 at 240, ~25 at 270-300),
        # so out there it had to be commanded ~25mm higher to put the markers
        # at the same real height. The pick depth has to follow that, or far
        # picks get driven ~25mm deeper than near ones - into the table, and
        # a fully stretched arm pushed into the table skids outward: the
        # overshoot seen at a 30 deg lean. (The first far-calibration version
        # clamped every pick to the near samples' Z - exactly that bug.)
        z_samples = np.c_[targets, all_xyz[:, 2]]

        hand_eye.update({
            'homography': H.tolist(),
            'homography_near': bool(everywhere),
            'z_samples': z_samples.tolist(),
        })
        sag = all_heights - all_xyz[:, 2] - marker_up * np.cos(np.radians(leans))
        print(f'calibrate_xy: arm sag (how far below its commanded height it really is) - '
              f'near up to {max(0.0, -sag[near].min()):.0f}mm, far up to '
              f'{max(0.0, -sag[~near].min()):.0f}mm; pick depth follows it')
        print(f'calibrate_xy: far stations: {far_n} samples (leans '
              f'{", ".join(str(int(l)) for l in sorted(set(far_leans)))} deg), marker '
              f'{marker_up:.0f}mm above the TCP')
        print(f'calibrate_xy: straight line misses the far samples by mean {line_far.mean():.1f}mm, '
              f'worst {line_far.max():.1f}mm (the overshoot)')
        print(f'calibrate_xy: homography (leave-one-out) - far mean {h_far.mean():.1f}mm, '
              f'worst {h_far.max():.1f}mm; near mean {h_near.mean():.1f}mm vs straight line '
              f'{loo.mean():.1f}mm -> '
              + ('homography everywhere' if everywhere
                 else f'straight line near, homography past {HAND_EYE_BLEND[0]:.0f}mm'))
        set_state(f'Adjust XY 完了: 遠方 平均{h_far.mean():.1f}mm / 手前 平均'
                  f'{min(h_near.mean(), loo.mean()):.1f}mm')
    else:
        print(f'calibrate_xy: only {far_n} far samples came back (need 3) - far picks keep '
              f'using the straight-line fit. Are the far stations in the camera view?')
        set_state(f'Adjust XY 完了 (遠方のサンプル不足: {far_n}件)')

    params['hand-eye'] = hand_eye

    write_params(params)

    prd = model.predict(X)

    for dx, dy, dz in (Y - prd).tolist():
        print(f'dxyz 1:{dx:.1f} {dy:.1f} {dz:.1f}')

    A = (model.coef_.dot(X.transpose()) + model.intercept_.reshape(3, 1)).transpose()
    B = Y - A
    for i in range(X.shape[0]):
        dx, dy, dz = B[i, :]
        print(f'dxyz 2:{dx:.1f} {dy:.1f} {dz:.1f}')

    with open('data/calibrate-xy.csv', 'w') as f:

        f.write('scr-x,scr-y,tcp-height,arm-x,arm-y,arm-z,prd-arm-x,prd-arm-y,prd-arm-z,lean\n')

        for (scr_x, scr_y), height, (arm_x, arm_y, arm_z), lean in zip(all_uv, all_heights, all_xyz, leans):
            X = np.array([[scr_x, scr_y]])

            prd = model.predict(X)
            prd_arm_x, prd_arm_y, prd_arm_z = prd[0, :]

            f.write(f'{scr_x}, {scr_y}, {height}, {arm_x}, {arm_y}, {arm_z}, {prd_arm_x}, {prd_arm_y}, {prd_arm_z}, {lean:.0f}\n')


def get_arm_xyz_from_screen(scr_x, scr_y):
    if not 'hand-eye' in params:
        print('No hand-eye calibration')
        sys.exit(0)

    he = params['hand-eye']
    coef = np.array(he['coef'])
    intercept = np.array(he['intercept'])

    arm_x, arm_y, arm_z = coef.dot(np.array([scr_x, scr_y])) + intercept

    if 'homography' in he:
        # See HAND_EYE_BLEND / calibrate_xy: the straight line only holds near
        # where it was fitted; out past it the homography takes over.
        hx, hy = apply_homography(he['homography'], [scr_x, scr_y])[0]
        if he.get('homography_near'):
            w = 1.0
        else:
            b0, b1 = HAND_EYE_BLEND
            w = min(1.0, max(0.0, (math.hypot(arm_x, arm_y) - b0) / (b1 - b0)))
        arm_x = (1 - w) * arm_x + w * hx
        arm_y = (1 - w) * arm_y + w * hy
        if 'z_samples' in he:
            # Pick depth from the calibration samples nearest to where the
            # pick is - it follows the arm's sag (see calibrate_xy). Inverse
            # distance weighting: stays inside the measured values instead of
            # extrapolating like the straight line's Z does.
            zs = np.asarray(he['z_samples'], float)
            dist = np.hypot(zs[:, 0] - arm_x, zs[:, 1] - arm_y)
            k = np.argsort(dist)[:4]
            wt = 1.0 / (dist[k] + 1.0) ** 2
            z_near_far = float(np.sum(wt * zs[k, 2]) / np.sum(wt))
            arm_z = (1 - w) * arm_z + w * z_near_far
        else:
            # an older far calibration: it only kept the near Z range
            zlo, zhi = he['z_range']
            arm_z = min(zhi, max(zlo, arm_z))

    return arm_x, arm_y, arm_z

def get_tcp():

    if np.isnan(marker_table[3:, :]).any():
        return None, None, np.nan
    else:
        tcp_cam  = marker_table[3:,  :3].mean(axis=0)
        tcp_scr  = marker_table[3:, 3:5].mean(axis=0)

        tcp_cam = Vec3(* tcp_cam.tolist())
        tcp_scr = Vec2(* tcp_scr.tolist())

        if basis_point is None:
            tcp_height = np.nan

        else:
            # Signed: positive above the table, negative below it. calibrate_xy
            # needs the sign to correct in the right direction if it ever
            # overshoots past the tabletop - abs() here used to erase exactly
            # that, so a slight overshoot read as "still too high" and the
            # search kept driving the gripper further down, through the table.
            tcp_height = normal_vector.dot(tcp_cam - basis_point)

        return tcp_cam, tcp_scr, tcp_height

def set_plane(vecs):
    if any(vec is None for vec in vecs):
        plane_points.clear()
        return None, None

    plane_points.append(vecs)

    if len(plane_points) < 10:
        return None, None

    points_samples = np.array(plane_points)
    assert points_samples.shape == (10, 3, 5)

    points = points_samples.mean(axis=0)
    assert points.shape == (3, 5)

    p1, p2, p3 = [ Vec3(*xyz ) for xyz in points[:, :3].tolist() ]

    normal_vector = (p2 - p1).cross(p3 - p1).unit()

    basis_point = (1.0 / 3.0) * (p1 + p2 + p3)

    # (p2-p1) x (p3-p1) points to whichever side the markers happen to have
    # been listed in - there is nothing in the marker layout that pins it to
    # "up". Camera-space tvecs are relative to the camera at the origin, so
    # the direction from the table up to the camera is (origin - basis_point);
    # orient the normal to agree with that; the sign this decides is what
    # keeps get_tcp()'s signed height meaning "above the table".
    if 0 < normal_vector.dot(basis_point):
        normal_vector = -1 * normal_vector

    return normal_vector, basis_point

def get_arm_xyz_of_work(color, shape):
    """Return (scr_x, scr_y, arm_x, arm_y, arm_z) of the currently detected work
    matching color/shape (either may be None to mean "any"), or five NaNs when
    nothing is detected."""
    if inference is None:
        return [np.nan] * 5

    work_scr_x, work_scr_y, class_name = inference.get(frame, color, shape)

    if np.isnan(work_scr_x):
        return [np.nan] * 5

    arm_x, arm_y, arm_z = get_arm_xyz_from_screen(work_scr_x, work_scr_y)

    # Constant hand-measured correction (marker/gripper offset, camera mount, ...).
    # Tunable live from the GUI; does not affect jogging or the box positions.
    arm_x += pick_offset['x']
    arm_y += pick_offset['y']
    arm_z += pick_offset['z']

    print(f'{class_name} {arm_x:.1f} {arm_y:.1f} {arm_z:.1f}')

    return work_scr_x, work_scr_y, arm_x, arm_y, arm_z

def test_xy():
    global test_pos

    h, w = frame.shape[:2]
    for scr_x in np.linspace(w / 3, w * 2 / 3, 2):
        for scr_y in np.linspace(h // 2, h * 2 // 3, 2):
            arm_x, arm_y, arm_z = get_arm_xyz_from_screen(scr_x, scr_y)

            for _ in move_xyz(arm_x, arm_y, LIFT_Z):
                yield

            for _ in move_xyz(arm_x, arm_y, arm_z):
                yield

            test_pos = Vec2(scr_x, scr_y)
            for _ in sleep(1):
                yield
            test_pos = None

    for _ in move_to_ready():
        yield


def holds_xy(x, y, z, r2=None):
    """Would move_xyz(x, y, z, r2=r2) land on exactly (x, y)? Mirrors
    get_pose_from_xyz's own X clamp and reach nudge, so True means no nudge."""
    pitch = PICK_PITCH if r2 is None else r2
    xc = max(X_MIN, x) if r2 is not None else min(X_MAX, max(X_MIN, x))
    nx, ny = nudge_into_reach(xc, y, float(z), radian(pitch))
    return math.hypot(nx - x, ny - y) <= 0.5


def highest_z(x, y, top, bottom, r2=None):
    """Highest Z from top down to bottom, in 5mm steps, that holds (x, y)
    exactly at pitch r2 - falls back to bottom if none do."""
    z = float(top)
    while z > bottom:
        if holds_xy(x, y, z, r2):
            return z
        z -= 5.0
    return float(bottom)


def carry_z(x, y, r2=None):
    """Highest height, up to CARRY_Z, at which the hand can hold exactly
    (x, y) at pitch r2 - see CARRY_Z. Never below LIFT_Z."""
    return highest_z(x, y, CARRY_Z, LIFT_Z, r2)


def grab_pitch(x, y, zs):
    """Tool pitch (degrees) to grab at (x, y): PICK_PITCH (straight down)
    unless the target is past the vertical gripper's OUTER reach at any of
    the heights zs, in which case the least outward lean - up to
    MAX_REACH_TILT_DEG - that holds the exact spot at every one of them.
    Near-side misses (inside the inner reach) are left alone: leaning outward
    only moves the reachable band further out."""
    too_far = False
    for z in zs:
        if not holds_xy(x, y, z, PICK_PITCH):
            nx, ny = nudge_into_reach(max(X_MIN, x), y, float(z), radian(PICK_PITCH))
            if math.hypot(nx, ny) < math.hypot(x, y) - 0.5:
                too_far = True       # vertical reach pulled it *inward*
    if not too_far:
        return PICK_PITCH

    for tilt in range(2, MAX_REACH_TILT_DEG + 1, 2):
        pitch = PICK_PITCH - tilt
        if all(holds_xy(x, y, z, pitch) for z in zs):
            return pitch
    return None


def grab_plan(arm_x, arm_y, arm_z):
    """How grab() would take the item at (arm_x, arm_y, arm_z), decided
    before anything moves: (pick depth z, pitch r2 - None if out of reach even
    leaning MAX_REACH_TILT_DEG -, radius from the base). Shared with the live
    aim readout in the main loop, so what the screen says it will do is
    exactly what grab() then does.

    Straight down unless the item is past the vertical gripper's outer reach
    - then lean outward just enough (see grab_pitch)."""
    z = arm_z - (PLACE_Z - PICK_Z)
    return z, grab_pitch(arm_x, arm_y, (z, LIFT_Z)), math.hypot(arm_x, arm_y)


def aim_text(arm_x, arm_y, arm_z):
    """(status line, short on-image label, BGR colour) describing grab_plan
    for an item at (arm_x, arm_y, arm_z) - for the live aim readout."""
    z, r2, radius = grab_plan(arm_x, arm_y, arm_z)
    where = f'X{arm_x:.0f} Y{arm_y:.0f} 半径{radius:.0f}mm'
    if r2 is None:
        return (f'狙い: {where} → 届かない (傾けても{MAX_REACH_TILT_DEG}°が上限)',
                f'NG r{radius:.0f}', (0, 0, 255))
    lean = PICK_PITCH - r2
    if lean:
        return (f'狙い: {where} → {lean:.0f}°傾けて掴む',
                f'r{radius:.0f} tilt{lean:.0f}', (0, 165, 255))
    return (f'狙い: {where} → 垂直で掴む', f'r{radius:.0f} OK', (0, 200, 0))


def grab(arm_x, arm_y, arm_z, box_xyz, close_deg=DEFAULT_GRIP_CLOSE):
    """Pick the work at (arm_x, arm_y, arm_z) and drop it into box_xyz.
    close_deg is the hand angle to close to (per-item, set in grip settings)."""
    global is_moving

    # Decide the approach before moving at all (see grab_plan). The same pitch
    # is held over the item for the approach, the descent and the lift, so
    # the gripper doesn't swing while it's low; the traverse to the box
    # straightens it again on the way.
    z, r2, radius = grab_plan(arm_x, arm_y, arm_z)
    if r2 is None:
        # Out of reach even at the full lean. This used to fall back to a
        # vertical grab, which get_pose_from_xyz then pulled in to the reach
        # limit - a guaranteed miss performed anyway, well short of the item
        # (seen on the real arm: a cube computed at 382mm was "grabbed" at
        # 230mm). Refuse instead, and say so.
        print(f'grab: target ({arm_x:.0f},{arm_y:.0f}) radius {radius:.0f}mm is out of '
              f'reach even leaning {MAX_REACH_TILT_DEG} deg - not attempting it')
        try:
            set_state(f'遠すぎて届きません: 半径{radius:.0f}mm '
                      f'(傾けても{MAX_REACH_TILT_DEG}°が上限)')
        except NameError:
            pass
        is_moving = False
        return

    # Say what was decided on every grab, in the GUI too - a lean of a few
    # degrees (all that's needed just past the vertical limit) is not
    # visible by eye, so "did it tilt?" has to be readable somewhere.
    lean = PICK_PITCH - r2
    print(f'grab: target ({arm_x:.0f},{arm_y:.0f}) radius {radius:.0f}mm - '
          + (f'leaning {lean:.0f} deg outward (past vertical reach)' if lean
             else 'straight down (within vertical reach)'))
    how = f'{lean:.0f}°傾けて' if lean else '垂直で'

    # Rise straight up first (gripper opening on the way), then move over to
    # the item, then descend - never translate horizontally while low, or
    # the gripper can clip other items on the table.
    set_phase(f'上昇中 (次: 半径{radius:.0f}mmの対象を{how}掴む)')
    cx, cy = forward_kinematics(servo_angles)[:2]
    for _ in parallel(open_hand(), move_xyz(cx, cy, carry_z(cx, cy))):
        yield

    set_phase(f'対象の上へ移動中 (半径{radius:.0f}mm, {how}掴む)')
    for _ in move_xyz(arm_x, arm_y, carry_z(arm_x, arm_y, r2), r2=r2):
        yield

    set_phase(f'下降中 ({how}掴む)')
    # Lower straight down onto the item, slowly - the gripper is about to close
    # on it and a fast drop knocks it over.
    for _ in move_xyz(arm_x, arm_y, z, DESCEND_TIME, r2=r2):
        yield

    for _ in sleep(0.5):
        yield

    set_phase('つかんでいます')
    for _ in close_hand(close_deg):
        yield

    for _ in sleep(0.5):
        yield

    # Raise the hand clear of the clutter.
    set_phase('持ち上げ中')
    for _ in move_xyz(arm_x, arm_y, carry_z(arm_x, arm_y, r2), r2=r2):
        yield

    # Carry to the box and let go BOX_DROP_HEIGHT above it - it drops in,
    # rather than being lowered all the way down.
    bx, by, bz = box_xyz
    set_phase('箱へ運搬中')
    for _ in move_xyz(bx, by, carry_z(bx, by)):
        yield

    drop_z = highest_z(bx, by, bz + BOX_DROP_HEIGHT, bz)
    for _ in move_xyz(bx, by, drop_z):
        yield

    set_phase(f'箱に投下 ({drop_z - bz:.0f}mm上から)')
    for _ in open_hand():
        yield

    for _ in move_xyz(bx, by, carry_z(bx, by)):
        yield

    for _ in move_to_ready():
        yield

    is_moving = False


def goto_box(box_xyz):
    bx, by, bz = box_xyz
    set_phase('箱の位置へ移動中')
    for _ in move_xyz(bx, by, carry_z(bx, by)):
        yield
    for _ in move_xyz(bx, by, bz):
        yield


# --------------------------------------------------------------------------
# GUI helpers
# --------------------------------------------------------------------------

def ensure_inference():
    global inference
    if inference is None:
        from infer import Inference
        inference = Inference()
        window['-model-'].update('モデル: 起動しました')


def box_label(color):
    x, y, z = boxes[color]
    return f'X{x:6.0f}  Y{y:6.0f}  Z{z:5.0f}'


def register_box(color):
    """Store the current arm position as the box position for color.
    Written to data/arm.json immediately, so it survives a restart."""
    p = forward_kinematics(servo_angles)
    boxes[color] = [round(float(p[0]), 1), round(float(p[1]), 1), round(float(p[2]), 1)]
    params['boxes'] = boxes
    write_params(params)
    window[f'-box-{color}-'].update(box_label(color))
    set_state(f'{COLOR_LABEL[color]} の箱を {box_label(color)} に登録しました')


_state_is_phase = False

def fit_control_column(window):
    """Make the window tall enough to show the whole control column without
    scrolling - but no taller than the screen allows (SCREEN_MARGIN_Y left for
    the title bar and taskbar) - and keep it on screen. Width is untouched.

    It has to be the window's own geometry that changes: setting the column's
    height alone does nothing, since the toplevel keeps its size once shown
    (checked on real Tk). The column is expand_y, so once the toplevel is
    taller and Tk has processed it, the column fills the extra height."""
    try:
        col = window['-control-col-']
        root = window.TKroot
        root.update()
        content_h = col.TKColFrame.TKFrame.winfo_reqheight()
        canvas_h = col.Widget.winfo_height()
        ww, wh = window.size
        overhead = wh - canvas_h                  # everything but the column's viewport
        _, screen_h = window.get_screen_size()
        max_wh = screen_h - SCREEN_MARGIN_Y
        new_wh = int(max(CAM_VIEW + overhead, min(content_h + 4 + overhead, max_wh)))
        x, y = window.current_location()
        y = max(0, min(y, max_wh - new_wh))      # keep the bottom edge on screen
        root.geometry(f'{ww}x{new_wh}+{x}+{y}')
        root.update()
        col.contents_changed()
    except Exception as e:
        print('fit_control_column:', e)


def cam_view_size(window):
    """Side of the (square) camera view, in pixels: as tall as the window
    allows, leaving room for the control column - but never smaller than
    CAM_VIEW. The margins make it settle rather than grow: an image that
    fills the window exactly would let the window grow to fit it, and so on."""
    try:
        ww, wh = window.size
    except Exception:
        return CAM_VIEW
    return int(max(CAM_VIEW, min(wh - 60, ww - CONTROL_COL_WIDTH - 60)))


def set_state(text):
    """A status that should stay put until something replaces it - a result,
    a warning, an error (e.g. "too far to reach")."""
    global _state_is_phase
    _state_is_phase = False
    try:
        window['-state-'].update(f'状態: {text}')
    except NameError:
        pass  # no window (tests / before the GUI exists)


def set_phase(text):
    """Progress of the motion running right now ("descending", "carrying to
    the box", ...). Updated at every step of a sequence and cleared back to
    待機 the moment the motion ends, so the status line always describes what
    the arm is doing *now* - previously a sequence set one message at its
    start and it lingered, stale, through every later step and after."""
    global _state_is_phase
    try:
        window['-state-'].update(f'状態: {text}')
    except NameError:
        return
    _state_is_phase = True


def end_phase():
    """Called when a motion finishes: a leftover progress message goes back to
    待機; a result/warning set with set_state is left for the user to read."""
    if _state_is_phase:
        set_state('待機')


def refresh_selection_labels():
    window['-target-'].update(f'対象: {COLOR_LABEL[sel_color]} / {SHAPE_LABEL[sel_shape]}')

    for c in SORT_COLORS:
        mark = '▶ ' if sel_color == c else '  '
        window[f'selc-{c}'].update(mark + COLOR_BTN_TEXT[c])
    window['selc-all'].update(('▶ ' if sel_color is None else '  ') + '色:全部')

    for s in SHAPES:
        mark = '▶ ' if sel_shape == s else '  '
        window[f'selsh-{s}'].update(mark + SHAPE_BTN_TEXT[s])
    window['selsh-all'].update(('▶ ' if sel_shape is None else '  ') + '形:全部')


def do_pick(color, shape):
    """Detect color+shape, pick it and drop it into its (color) box.
    Returns a generator or None."""
    if inference is None:
        set_state('先に仕分け対象を選んでください')
        return None
    if color is None:
        set_state('赤・青・緑のいずれかを選んでください（箱を決めるため）')
        return None

    _, _, ax, ay, az = get_arm_xyz_of_work(color, shape)
    if np.isnan(ax):
        set_state(f'{COLOR_LABEL[color]}/{SHAPE_LABEL[shape]} が見つかりません')
        return None

    close_deg = grip_close.get(inference.class_name, DEFAULT_GRIP_CLOSE)
    set_state(f'{COLOR_LABEL[color]}/{SHAPE_LABEL[shape]} をつかみます (閉じ{close_deg:.0f})')
    return grab(ax, ay, az, boxes[color], close_deg)


def open_sequence_editor(params):
    """Modal dialog to build an ordered (color, shape) pick list for auto-sort
    - setup only; running it is the separate 自動仕分け 開始 button, which uses
    whatever was last saved here. Saves to data/arm.json ('auto-sequence') and
    returns the saved list, or None if cancelled."""
    saved = params.get('auto-sequence', [])
    seq = [ (c, s) for c, s in saved ]

    color_choices = [('全部', None)] + [(COLOR_BTN_TEXT[c], c) for c in SORT_COLORS]
    shape_choices = [('全部', None)] + [(SHAPE_BTN_TEXT[s], s) for s in SHAPES]
    color_by_name = dict(color_choices)
    shape_by_name = dict(shape_choices)

    def line(i, c, s):
        return f'{i + 1}: {COLOR_LABEL[c]} / {SHAPE_LABEL[s]}'

    def refresh(lb):
        lb.update([line(i, c, s) for i, (c, s) in enumerate(seq)])

    layout = [
        [ sg.Text('仕分ける順番を「追加」で1件ずつ積み上げてください。') ],
        [
            sg.Text('色', size=(3, 1)),
            sg.Combo([n for n, _ in color_choices], default_value=color_choices[0][0],
                      key='-c-', readonly=True, size=(8, 1)),
            sg.Text('形状', size=(4, 1)),
            sg.Combo([n for n, _ in shape_choices], default_value=shape_choices[0][0],
                      key='-s-', readonly=True, size=(10, 1)),
            sg.Button('追加', key='-add-'),
        ],
        [ sg.Listbox([line(i, c, s) for i, (c, s) in enumerate(seq)], size=(38, 12), key='-list-') ],
        [ sg.Button('選択削除', key='-del-'), sg.Button('全消去', key='-clear-') ],
        [ sg.Button('保存', key='-go-'), sg.Button('キャンセル', key='-cancel-') ],
    ]

    win = sg.Window('自動仕分けシーケンス', layout, modal=True, finalize=True)

    result = None
    while True:
        ev, vals = win.read()
        if ev in (sg.WIN_CLOSED, '-cancel-'):
            break
        elif ev == '-add-':
            seq.append((color_by_name[vals['-c-']], shape_by_name[vals['-s-']]))
            refresh(win['-list-'])
        elif ev == '-del-':
            for idx in sorted(win['-list-'].get_indexes(), reverse=True):
                del seq[idx]
            refresh(win['-list-'])
        elif ev == '-clear-':
            seq.clear()
            refresh(win['-list-'])
        elif ev == '-go-':
            # an empty list is a valid thing to save too (clears the sequence)
            result = list(seq)
            params['auto-sequence'] = [ [c, s] for c, s in seq ]
            write_params(params)
            break

    win.close()
    return result


def open_manual_control():
    """Modal dialog for full 6-DOF manual jog: an on-screen "joypad" and
    keyboard alike, both driving the same [x, y, z, r1, r2, r3] pose (mm/deg),
    both by holding a direction down - a tap gives a small nudge, holding it
    keeps moving for as long as it's held, exactly like the keyboard already
    does for X/Y/Z on the main window. On-screen presses and keyboard both
    just set the same (axis, sign) jog_dir; the one continuous-jog loop below
    drives whichever is currently active.

    Unlike the main window's own X/Y/Z jog, R1 (wrist roll) is under direct
    manual control here too - so this does not go through get_pose_from_xyz
    (which would compute R1 itself via facing_r1) but builds the pose
    directly and solves it, exactly like move_xyz_now does for position-only
    jogging.

    A failed step (pose unreachable, or a dangerous joint jump - see
    try_move) is reverted rather than left to drift, same reasoning as the
    main jog's own revert-on-failure."""
    p = forward_kinematics(servo_angles)
    manual_pose = [float(p[0]), float(p[1]), float(p[2]),
                   degree(float(p[3])), degree(float(p[4])), degree(float(p[5]))]

    def try_move(pose6):
        x, y, z, r1, r2, r3 = pose6
        current = [radian(servo_to_angle(ch, servo_angles[ch])) for ch in range(nax - 1)]

        # prefer=current picks the closest-to-here branch when more than one
        # solves this pose - but a single small nudge can still cross a point
        # where the branch it *was* on stops solving at all (confirmed: the
        # forward-reach configuration can run out of reach from a plain +/-5
        # degree Pitch change alone, at plenty of ordinary positions), leaving
        # only a wildly different one, e.g. the base swung ~180 degrees round
        # to reach backwards instead. inverse_kinematics has nothing better to
        # offer there, so the guard has to be here: refuse to execute a
        # solution that moves any joint further than one tick's worth of jog
        # plausibly should, rather than let a routine nudge swing the arm
        # itself around.
        rads = inverse_kinematics([x, y, z, radian(r1), radian(r2), radian(r3)], prefer=current)
        if rads is None:
            return False

        jump = degree(max(abs(normalize_radian(a - b)) for a, b in zip(rads, current)))
        if jump > MANUAL_MAX_STEP_DEG:
            print(f'open_manual_control: refusing a {jump:.0f} deg joint jump for one step '
                  f'(limit {MANUAL_MAX_STEP_DEG:.0f}) - likely a reach-boundary branch flip, '
                  f'not a real destination')
            return False

        for ch, deg in enumerate(degree(rads)):
            set_angle(ch, deg)
        return True

    def cur_hand():
        try:
            return servo_to_angle(hand_idx, servo_angles[hand_idx])
        except Exception:
            return float('nan')

    # ---- layout: a D-pad for X/Y, a throttle-style pair for Z, and three
    # rotate/tilt pairs for R1/R2/R3 - all just bigger buttons with arrow
    # glyphs, not a true circular widget (PySimpleGUI has no round-button
    # element, and drawing one on a Graph would mean hand-rolling the
    # hit-testing math for which wedge was pressed - not worth the risk of
    # quietly getting a direction backwards on a physical arm).
    DBTN = dict(font=('Helvetica', 18, 'bold'), size=(3, 1))
    dpad = [
        [ sg.Text(''), sg.Button('▲', key='-mc-Up-', **DBTN), sg.Text('') ],
        [ sg.Button('◀', key='-mc-Left-', **DBTN),
          sg.Button('●', key='-mc-center-', font=('Helvetica', 10), size=(3, 1),
                     disabled=True, button_color=('gray', sg.theme_background_color())),
          sg.Button('▶', key='-mc-Right-', **DBTN) ],
        [ sg.Text(''), sg.Button('▼', key='-mc-Down-', **DBTN), sg.Text('') ],
    ]
    zpad = [
        [ sg.Button('▲', key='-mc-Prior-', **DBTN) ],
        [ sg.Text('Z', justification='center', size=(3, 1)) ],
        [ sg.Button('▼', key='-mc-Next-', **DBTN) ],
    ]
    orient_pad = [
        [ sg.Text('Roll ', size=(5, 1)),
          sg.Button('↺', key='-mc-x-', **DBTN), sg.Button('↻', key='-mc-z-', **DBTN) ],
        [ sg.Text('Pitch', size=(5, 1)),
          sg.Button('▼', key='-mc-k-', **DBTN), sg.Button('▲', key='-mc-i-', **DBTN) ],
        [ sg.Text('Tilt ', size=(5, 1)),
          sg.Button('◀', key='-mc-l-', **DBTN), sg.Button('▶', key='-mc-j-', **DBTN) ],
    ]
    hand_pad = [
        [ sg.Button('開く', key='-mc-hopen-', font=('Helvetica', 14), size=(6, 1)),
          sg.Button('閉じる', key='-mc-hclose-', font=('Helvetica', 14), size=(6, 1)) ],
    ]

    layout = [
        [ sg.Column([[sg.Text('位置', font=('Helvetica', 11))]] + dpad,
                     element_justification='center'),
          sg.Column(zpad, element_justification='center', pad=((20, 20), (26, 0))),
          sg.VerticalSeparator(),
          sg.Column([[sg.Text('姿勢', font=('Helvetica', 11))]] + orient_pad, pad=((20, 0), (0, 0))) ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('ハンド', font=('Helvetica', 11)) ],
        [ sg.Column(hand_pad), sg.Text('--', key='-mc-hand-', size=(6, 1)) ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('X'), sg.Text('--', key='-mc-x-val-', size=(6, 1)),
          sg.Text('Y'), sg.Text('--', key='-mc-y-val-', size=(6, 1)),
          sg.Text('Z'), sg.Text('--', key='-mc-z-val-', size=(6, 1)) ],
        [ sg.Text('R1'), sg.Text('--', key='-mc-r1-val-', size=(6, 1)),
          sg.Text('R2'), sg.Text('--', key='-mc-r2-val-', size=(6, 1)),
          sg.Text('R3'), sg.Text('--', key='-mc-r3-val-', size=(6, 1)) ],
        [ sg.Text('状態: ', size=(4, 1)), sg.Text('', key='-mc-state-', size=(34, 1)) ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('ボタンは押している間だけ動作します。キーボードも同様:', font=('Helvetica', 8)) ],
        [ sg.Text('↑↓←→:X/Y位置   PgUp/PgDn:Z   z/x:Roll   i/k:Pitch   j/l:Tilt   o/c:ハンド   r:Ready',
                  font=('Helvetica', 8)) ],
        [ sg.Button('Ready', key='-mc-ready-'), sg.Button('閉じる', key='-mc-exit-') ],
    ]

    win = sg.Window('手動操作', layout, modal=True, finalize=True, return_keyboard_events=True)

    def refresh():
        win['-mc-x-val-'].update(f'{manual_pose[0]:.0f}')
        win['-mc-y-val-'].update(f'{manual_pose[1]:.0f}')
        win['-mc-z-val-'].update(f'{manual_pose[2]:.0f}')
        win['-mc-r1-val-'].update(f'{manual_pose[3]:.0f}')
        win['-mc-r2-val-'].update(f'{manual_pose[4]:.0f}')
        win['-mc-r3-val-'].update(f'{manual_pose[5]:.0f}')
        win['-mc-hand-'].update(f'{cur_hand():.0f}')

    refresh()

    moving = None            # Ready's own generator, stepped below
    jog_dir = None           # (axis_index, sign) while a D-pad/orient button or key is held
    jog_last_t = 0.0
    hand_dir = 0              # -1 / 0 / +1 while a hand button/key is held
    hand_last_t = 0.0
    last_tick_t = time.time()

    def press_axis(name):
        nonlocal jog_dir, jog_last_t
        jog_dir = MANUAL_AXES[name]
        jog_last_t = time.time()

    def release_axis():
        nonlocal jog_dir
        jog_dir = None

    def press_hand(sign):
        nonlocal hand_dir, hand_last_t
        hand_dir = sign
        hand_last_t = time.time()

    def release_hand():
        nonlocal hand_dir
        hand_dir = 0

    # Mouse hold on the on-screen buttons drives the exact same jog_dir /
    # hand_dir the keyboard does. This is Element.bind() (needs the widgets
    # finalize=True already built), which routes the tk event through
    # PySimpleGUI's own event queue as an ordinary '<key>+PRESS'/'+RELEASE'
    # event seen below in the normal event, values = win.read() loop - not a
    # raw Widget.bind() callback mutating state out of band, which is what
    # made the first version of this dialog not respond to being held down.
    # <Leave> covers the mouse being dragged off the button while still held
    # down, so that can't leave a direction stuck "on" without a release.
    for name in MANUAL_AXES:
        win[f'-mc-{name}-'].bind('<ButtonPress-1>', '+PRESS')
        win[f'-mc-{name}-'].bind('<ButtonRelease-1>', '+RELEASE')
        win[f'-mc-{name}-'].bind('<Leave>', '+RELEASE')   # dragged off while held

    for key in ('-mc-hopen-', '-mc-hclose-'):
        win[key].bind('<ButtonPress-1>', '+PRESS')
        win[key].bind('<ButtonRelease-1>', '+RELEASE')
        win[key].bind('<Leave>', '+RELEASE')

    while True:
        event, values = win.read(timeout=20)

        if moving is not None:
            try:
                moving.__next__()
            except StopIteration:
                moving = None
                p = forward_kinematics(servo_angles)
                manual_pose[:] = [float(p[0]), float(p[1]), float(p[2]),
                                   degree(float(p[3])), degree(float(p[4])), degree(float(p[5]))]

        if event in (sg.WIN_CLOSED, '-mc-exit-'):
            break

        elif event == '-mc-ready-':
            jog_dir = None
            hand_dir = 0
            moving = move_to_ready()

        elif isinstance(event, str) and event.split(':')[0] in ('r', 'R'):
            jog_dir = None
            hand_dir = 0
            moving = move_to_ready()

        # ---- on-screen buttons: PySimpleGUI's own Element.bind() (not a raw
        # Widget.bind()) so press/release land as ordinary events through
        # win.read() like everything else, rather than a bare tkinter
        # callback mutating state out of band - that's what actually made
        # the first version of this dialog not respond to being held down.
        elif isinstance(event, str) and event.endswith('+PRESS') \
                and event[4:-len('-+PRESS')] in MANUAL_AXES and moving is None:
            press_axis(event[4:-len('-+PRESS')])         # '-mc-<name>-+PRESS' -> <name>

        elif isinstance(event, str) and event.endswith('+RELEASE') \
                and event[4:-len('-+RELEASE')] in MANUAL_AXES:
            release_axis()                  # always allowed, even if moving started meanwhile

        elif event == '-mc-hopen-+PRESS' and moving is None:
            press_hand(1)

        elif event == '-mc-hclose-+PRESS' and moving is None:
            press_hand(-1)

        elif event in ('-mc-hopen-+RELEASE', '-mc-hclose-+RELEASE'):
            release_hand()

        # ---- fallback: a plain click (PySimpleGUI's own native button
        # event, no +PRESS/+RELEASE suffix - fires regardless of whether the
        # bindings above are actually taking effect) does one fixed nudge.
        # See MANUAL_FALLBACK_STEP.
        elif isinstance(event, str) and event.startswith('-mc-') and event.endswith('-') \
                and event[4:-1] in MANUAL_AXES and moving is None:
            idx, sign = MANUAL_AXES[event[4:-1]]
            lo, hi = MANUAL_RANGES[idx]
            prev = list(manual_pose)
            new_val = manual_pose[idx] + MANUAL_FALLBACK_STEP * sign
            if lo <= manual_pose[idx] <= hi:
                new_val = min(hi, max(lo, new_val))
            manual_pose[idx] = new_val
            if try_move(manual_pose):
                win['-mc-state-'].update('')
            else:
                manual_pose[:] = prev
                win['-mc-state-'].update('その方向には移動できません')
            refresh()

        elif event == '-mc-hopen-' and moving is None:
            dst = min(GRIP_OPEN, cur_hand() + MANUAL_FALLBACK_STEP)
            set_angle(hand_idx, dst)
            refresh()

        elif event == '-mc-hclose-' and moving is None:
            dst = max(GRIP_CLOSE_MIN, cur_hand() - MANUAL_FALLBACK_STEP)
            set_angle(hand_idx, dst)
            refresh()

        elif isinstance(event, str) and moving is None:
            # keyboard only, at this point in the chain
            key = event.split(':')[0]
            if key in MANUAL_AXES:
                press_axis(key)
            elif key in ('o', 'O'):
                press_hand(1)
            elif key in ('c', 'C'):
                press_hand(-1)

        # ---- continuous jog: smooth while a key or on-screen button is held,
        # keyboard and mouse both funnelling through the same jog_dir/hand_dir ----
        now = time.time()
        dt = now - last_tick_t
        last_tick_t = now

        if jog_dir is not None:
            if now - jog_last_t > JOG_RELEASE_TIMEOUT:
                jog_dir = None          # keyboard's own release: no key-up event, so time out
            elif moving is None:
                idx, sign = jog_dir
                speed = MANUAL_POS_SPEED if idx < 3 else MANUAL_ANGLE_SPEED
                prev = list(manual_pose)
                lo, hi = MANUAL_RANGES[idx]
                new_val = manual_pose[idx] + sign * speed * dt
                # Only clamp into [lo, hi] when already inside it. Ready (or
                # anything else that lands outside this axis's normal range -
                # confirmed: Ready's own Z is -20, well under this dialog's
                # Z floor of 20) must not make the very first jog tick snap
                # straight to the boundary: that snap is a real, possibly
                # large positional jump, not a routine small nudge, and can
                # by itself demand more than MANUAL_MAX_STEP_DEG of joint
                # travel - tripping try_move's safety cap on every single
                # attempt and making the axis look permanently stuck.
                # Starting outside range just rides the ordinary small step
                # until it re-enters on its own, same speed as always.
                if lo <= manual_pose[idx] <= hi:
                    new_val = min(hi, max(lo, new_val))
                manual_pose[idx] = new_val
                if try_move(manual_pose):
                    win['-mc-state-'].update('')
                else:
                    manual_pose[:] = prev
                    win['-mc-state-'].update('その方向には移動できません')
                refresh()

        if hand_dir:
            if now - hand_last_t > JOG_RELEASE_TIMEOUT:
                hand_dir = 0
            elif moving is None:
                dst = min(GRIP_OPEN, max(GRIP_CLOSE_MIN, cur_hand() + hand_dir * HAND_JOG_SPEED * dt))
                set_angle(hand_idx, dst)
                refresh()

    win.close()


def open_grip_settings(params, grip_close):
    """Modal dialog to tune the per-item close angle. Each item has a slider,
    and moving it closes the real gripper to that angle right away - so the
    number can be judged by feel against a real item in the hand, instead of
    typing a value and pressing a button to find out what it does.
    Mutates grip_close in place and saves to data/arm.json on save; returns
    True if saved, False if cancelled."""
    items = [(c, s) for c in SORT_COLORS for s in SHAPES]

    def cur_hand():
        try:
            return servo_to_angle(hand_idx, servo_angles[hand_idx])
        except Exception:
            return float('nan')

    rows = []
    for c, s in items:
        key = f'{c}_{s}'
        rows.append([
            sg.Text(f'{COLOR_BTN_TEXT[c]} / {SHAPE_BTN_TEXT[s]}', size=(12, 1)),
            sg.Slider(range=(GRIP_CLOSE_MIN, GRIP_CLOSE_MAX), resolution=1, orientation='h',
                      default_value=int(grip_close.get(key, DEFAULT_GRIP_CLOSE)),
                      size=(28, 15), enable_events=True, key=f'-gc-{key}-'),
        ])

    layout = [
        [ sg.Text('スライダーを動かすと、その角度までハンドが実際に閉じます。') ],
        [ sg.Text(f'{GRIP_CLOSE_MIN} = 完全に閉じる（最も強い）   '
                  f'{GRIP_CLOSE_MAX} = ゆるい   （全開は {GRIP_OPEN}）',
                  font=('Helvetica', 9)) ],
        [ sg.Text('物をハンドに挟んで、落ちずにつぶれない所に合わせてください。',
                  font=('Helvetica', 9)) ],
    ] + rows + [
        [ sg.HorizontalSeparator() ],
        [
            sg.Button('開く', key='-gc-open-', size=(8, 1)),
            sg.Text('現在のハンド角度:'),
            sg.Text('--', key='-gc-current-', size=(6, 1)),
            sg.Text('', key='-gc-which-', size=(24, 1)),
        ],
        [ sg.HorizontalSeparator() ],
        [ sg.Button('保存して閉じる', key='-gc-save-'), sg.Button('キャンセル', key='-gc-cancel-') ],
    ]

    win = sg.Window('つかむ強さの設定', layout, modal=True, finalize=True)

    saved = False
    while True:
        ev, vals = win.read(timeout=150)
        win['-gc-current-'].update(f'{cur_hand():.0f}')

        if ev in (sg.WIN_CLOSED, '-gc-cancel-'):
            break
        elif ev == sg.TIMEOUT_EVENT:
            continue
        elif ev == '-gc-open-':
            set_angle(hand_idx, GRIP_OPEN)
            win['-gc-which-'].update('')
        elif isinstance(ev, str) and ev.startswith('-gc-') and ev[len('-gc-'):-1] in \
                {f'{c}_{s}' for c, s in items}:
            # live: the gripper follows the slider as it moves
            key = ev[len('-gc-'):-1]
            deg = float(vals[ev])
            set_angle(hand_idx, deg)
            c, s = key.split('_', 1)
            win['-gc-which-'].update(f'試し中: {COLOR_BTN_TEXT[c]} / {SHAPE_BTN_TEXT[s]} = {deg:.0f}')
        elif ev == '-gc-save-':
            for c, s in items:
                key = f'{c}_{s}'
                grip_close[key] = float(vals[f'-gc-{key}-'])
            params['grip-close'] = grip_close
            write_params(params)
            saved = True
            break

    win.close()
    return saved


if __name__ == '__main__':
    plane_points = []
    normal_vector = None
    basis_point = None
    inference = None
    sel_color = None
    sel_shape = None

    params = read_params()

    marker_ids = params['marker-ids']

    boxes = dict((c, list(DEFAULT_BOXES[c])) for c in SORT_COLORS)
    for c, xyz in params.get('boxes', {}).items():
        if c in boxes:
            boxes[c] = list(xyz)

    # Constant correction applied to every vision-based pick (marker/gripper
    # mounting offset, camera mount, ...). Tuned live from the GUI; persisted.
    pick_offset = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    pick_offset.update(params.get('pick-offset', {}))

    # Per-item gripper close angle. Tuned in the grip-settings dialog; persisted.
    grip_close = {f'{c}_{s}': DEFAULT_GRIP_CLOSE for c in SORT_COLORS for s in SHAPES}
    grip_close.update(params.get('grip-close', {}))

    init_servo(params)
    init_markers(params)
    initCamera(params)

    marker_table = np.array([[0] * 5] * len(marker_ids), dtype=np.float32)

    COLOR_BTN = {'red': ('white', '#c0392b'), 'green': ('white', '#27ae60'), 'blue': ('white', '#2980b9')}

    debug_layout = [
        [ sg.Text('TCP pose', font=('Helvetica', 11)) ],
        spin('X', 'X' , 0,    0, 400 ),
        spin('Y', 'Y' , 0, -300, 300 ),
        spin('Z', 'Z' , 0,    0, 150 ),
        spin('R1', 'R1', 0, -90,  90 ),
        spin('R2', 'R2', 0,   0, 120 ),
        spin('R3', 'R3', 0, -90,  90 ),
        spin('hand', 'hand', 0,   0, 100 ),
        [ sg.Table(marker_table.tolist(), headings=['cam x', 'cam y', 'cam z', 'scr x', 'scr y'],
                   auto_size_columns=False, col_widths=[6] * 5, num_rows=len(marker_ids), key='-marker-table-') ],
    ]

    box_rows = []
    for c in SORT_COLORS:
        box_rows.append([
            sg.Text(COLOR_BTN_TEXT[c], size=(5, 1), text_color=COLOR_BTN[c][1]),
            sg.Text(box_label(c), key=f'-box-{c}-', size=(24, 1), font=('Consolas', 9)),
            sg.Button('ここに登録', key=f'setbox-{c}'),
            sg.Button('移動', key=f'gotobox-{c}'),
        ])

    control_col = sg.Column([
        [ sg.Text('モデル: 未ロード', key='-model-', size=(34, 1)) ],
        [ sg.Text('対象: 全部 / 全部', key='-target-', size=(34, 1)) ],
        [ sg.Text('検出: -', key='-detect-', size=(34, 1)) ],
        [ sg.Text('狙い: -', key='-aim-', size=(34, 2)) ],
        [ sg.Text('TCP height: -', key='-tcp-height-', size=(34, 1)) ],
        [ sg.Text('状態: 待機', key='-state-', size=(34, 2)) ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('仕分け対象 - 色') ],
        [
            sg.Button('赤', key='selc-red', size=(6, 2), button_color=COLOR_BTN['red']),
            sg.Button('青', key='selc-blue', size=(6, 2), button_color=COLOR_BTN['blue']),
            sg.Button('緑', key='selc-green', size=(6, 2), button_color=COLOR_BTN['green']),
            sg.Button('色:全部', key='selc-all', size=(8, 2)),
        ],
        [ sg.Text('仕分け対象 - 形状') ],
        [
            sg.Button('キューブ', key='selsh-cube', size=(9, 2)),
            sg.Button('ラグビー', key='selsh-ragby', size=(9, 2)),
            sg.Button('ピラミッド', key='selsh-pyramid', size=(9, 2)),
            sg.Button('形:全部', key='selsh-all', size=(8, 2)),
        ],
        [
            sg.Button('つかんで置く', key='pick', size=(18, 2)),
            sg.Button('自動仕分け 開始', key='auto', size=(18, 2)),
        ],
        [ sg.Button('つかむ強さの設定', key='grip-settings'),
          sg.Button('自動仕分けの設定', key='auto-setup'),
          sg.Button('手動', key='manual-control') ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('つかみ位置の補正(mm)  ズレる方向と逆に、Enterで確定') ],
        [
            *spin('dX', '-poff-x-', int(pick_offset['x']), -50, 50),
            *spin('dY', '-poff-y-', int(pick_offset['y']), -50, 50),
            *spin('dZ', '-poff-z-', int(pick_offset['z']), -50, 50),
        ],
        [ sg.HorizontalSeparator() ],
        [
            sg.Text('移動時間'),
            sg.Slider(range=(0.5, 4.0), default_value=get_move_time(), resolution=0.1,
                      orientation='h', size=(18, 15), key='-mt-', enable_events=True),
        ],
        [ sg.HorizontalSeparator() ],
        [ sg.Text('箱の位置  （矢印キー等で動かして「ここに登録」）') ],
        [ sg.Text('↑↓:X前後   ←→:Y左右   PgUp/PgDn:Z上下   o/c:ハンド   r:Ready',
                  font=('Helvetica', 8)) ],
    ] + box_rows + [
        [ sg.HorizontalSeparator() ],
        [ sg.Text('キャリブレーション') ],
        [
            sg.Button('Reset', tooltip='平面（法線・基準点）を測り直す'),
            sg.Button('Adjust XY', tooltip='ハンドアイ・キャリブレーション'),
            sg.Button('Test XY', tooltip='画面座標→アーム座標の確認'),
        ],
        [ sg.HorizontalSeparator() ],
        [ sg.Button('Ready'), sg.Button('詳細表示', key='toggle-debug'), sg.Button('終了', key='Close') ],
        [ sg.pin(sg.Column(debug_layout, key='-dbg-', visible=False)) ],
    ], key='-control-col-', vertical_alignment='top', scrollable=True,
       vertical_scroll_only=True, size=(400, CAM_VIEW), expand_x=True, expand_y=True)

    layout = [
        [ sg.Image(key='-cam-', size=(CAM_VIEW, CAM_VIEW)), control_col ]
    ]

    window = sg.Window('仕分けアーム', layout, finalize=True, return_keyboard_events=True,
                        resizable=True)
    # Grow the window downward so the whole control column shows without
    # scrolling (as far as the screen allows) - at its natural size it was
    # only as tall as the 560px camera view and most controls were below the
    # fold. Taller only; not maximized.
    fit_control_column(window)
    refresh_selection_labels()
    ensure_inference()  # start loading the YOLO model right away, not on first click

    pose = forward_kinematics(servo_angles)
    show_pose(window, pose)

    last_capture = time.time()
    is_moving = False

    moving = None
    test_pos = None
    frame = None
    tcp_cam = tcp_scr = None
    tcp_height = np.nan
    cx = cy = np.nan
    class_name = None

    dbg_visible = False

    auto_mode = False
    auto_queue = []
    auto_step_idx = 0
    auto_state = 'idle'      # idle -> settle -> detect
    auto_t = 0.0

    jog_dir = None            # (dx, dy, dz) while an arrow/PgUp/PgDn key is held
    jog_last_t = 0.0
    jog_pos = None            # [x, y, z] target while jogging
    hand_dir = 0              # -1 / 0 / +1 while o/c is held
    hand_last_t = 0.0
    last_tick_t = time.time()

    while True:
        event, values = window.read(timeout=20)

        # ---- step the running motion ----
        if moving is not None:
            try:
                moving.__next__()
            except StopIteration:
                moving = None
                is_moving = False
                print('========== stop moving ==========')
                end_phase()
                params['prev-servo'] = servo_angles
                write_params(params)

        # ---- events ----
        if event == sg.WIN_CLOSED or event == 'Close':
            params['prev-servo'] = servo_angles
            params['boxes'] = boxes
            write_params(params)
            closeCamera()
            if inference is not None:
                inference.close()
            break

        elif event == sg.TIMEOUT_EVENT:
            pass

        elif event in pose_keys:
            moving = move_linear(get_pose(values))

        elif event == 'hand':
            moving = move_joint(hand_idx, float(values['hand']))

        elif event == 'Ready':
            auto_mode = False
            window['auto'].update('自動仕分け 開始')
            jog_dir = None
            hand_dir = 0
            moving = move_to_ready()

        elif event == 'Test XY':
            moving = test_xy()

        elif event == 'Adjust XY':
            moving = calibrate_xy()

        elif event == 'Reset':
            normal_vector = None
            basis_point = None
            plane_points.clear()
            set_state('平面の再計測を開始します')

        elif event == '-mt-':
            set_move_time(float(values['-mt-']))

        elif event in ('-poff-x-', '-poff-y-', '-poff-z-'):
            pick_offset['x'] = float(values['-poff-x-'])
            pick_offset['y'] = float(values['-poff-y-'])
            pick_offset['z'] = float(values['-poff-z-'])
            params['pick-offset'] = pick_offset
            write_params(params)
            set_state(f'つかみ位置補正を更新: '
                      f'dX{pick_offset["x"]:.0f} dY{pick_offset["y"]:.0f} dZ{pick_offset["z"]:.0f}')

        elif event in ('selc-red', 'selc-blue', 'selc-green', 'selc-all'):
            ensure_inference()
            c = event.split('-')[1]
            sel_color = None if c == 'all' else c
            refresh_selection_labels()

        elif event in ('selsh-cube', 'selsh-ragby', 'selsh-pyramid', 'selsh-all'):
            ensure_inference()
            s = event.split('-')[1]
            sel_shape = None if s == 'all' else s
            refresh_selection_labels()

        elif event == 'pick':
            if moving is None:
                g = do_pick(sel_color, sel_shape)
                if g is not None:
                    moving = g

        elif event == 'auto':
            # Run / stop only - the order is set up separately (auto-setup)
            # and run straight from what was saved there.
            if auto_mode:
                auto_mode = False
                window['auto'].update('自動仕分け 開始')
                set_state('自動仕分けを停止しました')
            else:
                seq = [ (c, s) for c, s in params.get('auto-sequence', []) ]
                if not seq:
                    set_state('先に「自動仕分けの設定」で順番を登録してください')
                else:
                    ensure_inference()
                    auto_queue = seq
                    auto_step_idx = 0
                    auto_state = 'idle'
                    auto_mode = True
                    window['auto'].update('自動仕分け 停止')
                    set_state(f'自動仕分けを開始しました（{len(seq)}件）')

        elif event == 'auto-setup':
            if auto_mode:
                set_state('自動仕分けの実行中は設定を変更できません（先に停止してください）')
            else:
                seq = open_sequence_editor(params)
                if seq is not None:
                    set_state(f'自動仕分けの順番を保存しました（{len(seq)}件）')

        elif event == 'grip-settings':
            if moving is None:
                if open_grip_settings(params, grip_close):
                    set_state('つかむ強さの設定を保存しました')

        elif event == 'manual-control':
            if moving is None:
                open_manual_control()
                pose = forward_kinematics(servo_angles)
                show_pose(window, pose)

        elif isinstance(event, str) and event.startswith('setbox-'):
            register_box(event.split('-')[1])

        elif isinstance(event, str) and event.startswith('gotobox-'):
            if moving is None:
                moving = goto_box(boxes[event.split('-')[1]])

        elif event == 'toggle-debug':
            dbg_visible = not dbg_visible
            window['-dbg-'].update(visible=dbg_visible)
            # -dbg- lives inside a scrollable Column now (so long panels like
            # this one get a scrollbar instead of running off the bottom of
            # the window - see control_col below) - sg.pin() alone keeps a
            # toggled element from disturbing a normal layout, but a
            # scrollable Column's own scroll region is a separate tkinter
            # canvas that does not notice a child's visibility changing on
            # its own; without this the button looked like it did nothing.
            window['-control-col-'].contents_changed()
            # grow (or shrink back) to show the panel, as far as the screen allows
            fit_control_column(window)
            if dbg_visible:
                # ...and even with the scroll region correct, -dbg- is the very
                # LAST row of a control panel that is already taller than its
                # own 560px viewport, so revealing it changed nothing the user
                # could actually see - it appeared below the fold, which is why
                # this button still looked dead after the contents_changed()
                # fix. Scroll down so the panel is actually on screen.
                window['-control-col-'].set_vscroll_position(1.0)

        elif isinstance(event, str) and event.split(':')[0] in ('r', 'R'):
            # Always available, even mid-move: pressing Ready should get the
            # arm out of wherever it is, not wait behind whatever is running.
            jog_dir = None
            hand_dir = 0
            moving = move_to_ready()
            set_state('Ready へ移動中')

        elif isinstance(event, str) and moving is None:
            key = event.split(':')[0]
            if key in JOG_DIRS:
                if jog_dir is None:
                    p = forward_kinematics(servo_angles)
                    jog_pos = [float(p[0]), float(p[1]), float(p[2])]
                jog_dir = JOG_DIRS[key]
                jog_last_t = time.time()
            elif key in ('o', 'O'):
                hand_dir = 1
                hand_last_t = time.time()
            elif key in ('c', 'C'):
                hand_dir = -1
                hand_last_t = time.time()

        # ---- continuous keyboard jog (smooth while a key is held) ----
        now = time.time()
        dt = now - last_tick_t
        last_tick_t = now

        if jog_dir is not None:
            if now - jog_last_t > JOG_RELEASE_TIMEOUT:
                jog_dir = None
            elif moving is None:
                prev_pos = list(jog_pos)

                dx, dy, dz = jog_dir
                step = JOG_SPEED * dt
                jog_pos[0] = min(X_MAX - 5, max(X_MIN + 5, jog_pos[0] + dx * step))
                jog_pos[1] = min(Y_LIMIT, max(-Y_LIMIT, jog_pos[1] + dy * step))
                jog_pos[2] = min(Z_MAX, max(Z_MIN, jog_pos[2] + dz * step))

                if move_xyz_now(*jog_pos):
                    set_state(f'ジョグ X{jog_pos[0]:.0f} Y{jog_pos[1]:.0f} Z{jog_pos[2]:.0f}')
                else:
                    # That step landed somewhere the arm cannot actually hold
                    # (tool-vertical reach shrinks as Z changes - it is not a
                    # simple box). Undo it rather than pressing on from there.
                    jog_pos[:] = prev_pos
                    set_state(f'ジョグ範囲外 (X{jog_pos[0]:.0f} Y{jog_pos[1]:.0f} '
                              f'Z{jog_pos[2]:.0f} には届きません)')

        if hand_dir:
            if now - hand_last_t > JOG_RELEASE_TIMEOUT:
                hand_dir = 0
            elif moving is None:
                cur = servo_to_angle(hand_idx, servo_angles[hand_idx])
                dst = min(GRIP_OPEN, max(GRIP_CLOSE_MIN, cur + hand_dir * HAND_JOG_SPEED * dt))
                set_angle(hand_idx, dst)

        # ---- what the camera loop should currently look for ----
        if auto_mode and 0 <= auto_step_idx < len(auto_queue):
            active_color, active_shape = auto_queue[auto_step_idx]
        else:
            active_color, active_shape = sel_color, sel_shape

        # ---- automatic sorting state machine ----
        if auto_mode and moving is None:
            if auto_step_idx >= len(auto_queue):
                auto_mode = False
                window['auto'].update('自動仕分け 開始')
                set_state('自動仕分け シーケンス完了')
            else:
                color, shape = auto_queue[auto_step_idx]
                if auto_state == 'idle':
                    window['-target-'].update(
                        f'対象: {COLOR_LABEL[color]} / {SHAPE_LABEL[shape]}'
                        f'  (自動 {auto_step_idx + 1}/{len(auto_queue)})'
                    )
                    auto_state = 'settle'
                    auto_t = now
                elif auto_state == 'settle':
                    if now - auto_t > 0.8:
                        auto_state = 'detect'
                elif auto_state == 'detect':
                    if color is None:
                        set_state('自動: 色が指定されていないためスキップします（箱が決まりません）')
                    else:
                        _, _, ax, ay, az = get_arm_xyz_of_work(color, shape)
                        if not np.isnan(ax):
                            close_deg = grip_close.get(inference.class_name, DEFAULT_GRIP_CLOSE)
                            set_state(f'自動: {COLOR_LABEL[color]}/{SHAPE_LABEL[shape]} をつかみます (閉じ{close_deg:.0f})')
                            moving = grab(ax, ay, az, boxes[color], close_deg)
                        else:
                            set_state(f'自動: {COLOR_LABEL[color]}/{SHAPE_LABEL[shape]} が見つかりません（スキップ）')
                    auto_step_idx += 1
                    auto_state = 'idle'

        # ---- camera / vision (throttled) ----
        if frame is None or time.time() - last_capture > 0.1:
            last_capture = time.time()

            f = getCameraFrame()
            if f is None:
                set_state('カメラの読み取りに失敗しました')
            else:
                frame = f
                disp = frame.copy()

                cx = cy = np.nan
                class_name = None
                if inference is not None and moving is None:
                    cx, cy, class_name = inference.get(frame, active_color, active_shape)
                    window['-model-'].update('モデル: 稼働中')

                tcp_height = np.nan
                if not is_moving:
                    disp, vecs = detect_markers(marker_ids, disp)

                    for ch, vec in enumerate(vecs):
                        if vec is None:
                            marker_table[ch, :] = [np.nan] * 5
                        else:
                            marker_table[ch, :] = vec
                            marker_table[ch, 2] = np.abs(marker_table[ch, 2])

                    if normal_vector is None:
                        normal_vector, basis_point = set_plane(vecs[:3])

                    tcp_cam, tcp_scr, tcp_height = get_tcp()

                    window['-tcp-height-'].update(f'TCP height: {tcp_height:.1f}')
                    window['-marker-table-'].update(values=np.round(marker_table).tolist())

                    if tcp_scr is not None:
                        cv2.circle(disp, (int(tcp_scr.x), int(tcp_scr.y)), 10, (0, 0, 255), -1)

                if not np.isnan(cx):
                    cv2.circle(disp, (int(cx), int(cy)), 18, (255, 0, 0), 3)
                    cv2.putText(disp, f'{class_name}', (int(cx) + 20, int(cy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    window['-detect-'].update(f'検出: {class_name} ({int(cx)}, {int(cy)})')

                    # Live aim: where a pick would go right now and how - the
                    # same arm coordinates do_pick() would use (hand-eye +
                    # pick offset) run through the same grab_plan() grab()
                    # uses, so the readout matches what a pick then does.
                    ax, ay, az = get_arm_xyz_from_screen(cx, cy)
                    ax += pick_offset['x']
                    ay += pick_offset['y']
                    az += pick_offset['z']
                    text, label, colour = aim_text(ax, ay, az)
                    window['-aim-'].update(text)
                    cv2.putText(disp, label, (int(cx) + 20, int(cy) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                elif moving is None:
                    window['-detect-'].update('検出: -')
                    window['-aim-'].update('狙い: -')

                if test_pos is not None:
                    cv2.circle(disp, (int(test_pos.x), int(test_pos.y)), 5, (0, 0, 255), -1)

                view = cam_view_size(window)
                h, w = disp.shape[:2]
                if (h, w) != (view, view):
                    disp = cv2.resize(disp, (view, view))

                ok, buf = cv2.imencode('.png', disp)
                if ok:
                    window['-cam-'].update(data=buf.tobytes())

    window.close()
