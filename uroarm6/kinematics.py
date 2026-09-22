import math
import numpy as np
from util import nax, radian, degree, Vec2
from servo import servo_to_angle, angle_limits

# Link lengths, mm.
#   L0  base -> shoulder (height)
#   L1  shoulder -> elbow          (upper arm)
#   L2  elbow -> forearm roll axis
#   L3  forearm roll axis -> wrist (pitch) centre
#   L4  wrist perpendicular offset (0 on this arm)
#   L5  wrist centre -> TCP
# The forearm was a single 147mm link until the roll servo (J4) was fitted
# partway along it; it is now L2 + L3 = 152mm, the 5mm being the servo body.
#L0, L1, L2, L3, L4 = [ 75, 105, 147, 0, 180 ]      # before the J4 servo
# L0 +30 (2026-09-21): on the real arm the gripper sat ~3cm higher than
# commanded, everywhere. The camera agreed - calibrate_xy's measured height ran
# a near-constant ~50mm over the commanded Z at every grid point, i.e. a pure
# vertical offset rather than an angle error (an angle error would vary with
# pose). L0 only translates the whole chain vertically, so this lowers every
# commanded position by exactly 30mm and changes nothing else. With the tool
# vertical it is indistinguishable from the tool being 30mm shorter - either
# way, the arm physically cannot bring a vertical gripper to table height
# closer than ~150mm from the base. Check with a ruler: table to the J2
# (shoulder) axis should measure about this much.
L0, L1, L2, L3, L4, L5 = [ 75 - 20 + 30, 105, 105, 47, 0, 180 ]

# Elbow -> wrist centre. The J4 roll axis runs *along* the forearm, so rolling
# it does not move the wrist centre - only the total length matters for
# position, not where along the forearm the servo happens to sit.
LF = L2 + L3

# Number of joints the kinematics models: everything but the gripper.
narm = nax - 1

Y_HAT = np.array([0.0, 1.0, 0.0])

def arr(v):
    if isinstance(v, np.ndarray):
        return v
    else:
        return np.array(v, dtype=np.float32)

def rot_x(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def dir3(t):
    """Unit vector at elevation t in the arm's own vertical plane (the x-z
    plane of the base-yaw frame), matching planar Vec2(1,0).rot(t)."""
    return np.array([math.cos(t), 0.0, math.sin(t)])

def forearm_frame(sigma):
    """Orientation of the wrist at J4 = J5 = J6 = 0, given the forearm
    elevation sigma. Columns are the tool axis, the reference the wrist rolls
    are measured from, and their cross product."""
    e_f = dir3(sigma)
    return np.column_stack([e_f, Y_HAT, np.cross(e_f, Y_HAT)])

def normalize_radian(rad):
    while np.pi < rad:
        rad -= 2 * np.pi

    while rad < -np.pi:
        rad += 2 * np.pi

    return rad

def fk_pose(rads):
    """Forward kinematics proper: the narm arm joints, in radians, to a pose.
    Kept separate from forward_kinematics so the self-check there can run it on
    an IK solution without recursing."""
    j0, j1, j2, j3, j4, j5 = rads

    # ---- shoulder and elbow, in the vertical plane swung round by the base
    # yaw j0. Unchanged from the 5-joint arm except that the forearm is now
    # LF = L2 + L3 rather than a single link.
    a1 = 0.5 * np.pi + j1
    sigma = a1 + j2

    p1 = np.array([0.0, 0.0, L0])
    p2 = p1 + L1 * dir3(a1)
    pw = p2 + LF * dir3(sigma)          # wrist centre, on the J4 roll axis

    # ---- wrist: roll j3 about the forearm, pitch j4, roll j5 about the tool.
    # All three axes meet at pw, so the wrist is spherical and the orientation
    # is completely decoupled from the position.
    R_pre  = forearm_frame(sigma) @ rot_x(j3) @ rot_y(-j4)
    R_tool = R_pre @ rot_x(j5)

    e_t = R_pre[:, 0]                   # tool pointing direction
    e_p = R_pre[:, 2]                   # the L4 offset direction

    tcp = pw + L5 * e_t + L4 * e_p

    x, y, z = rot_z(j0) @ tcp

    # R_tool is the tool orientation in the base-yaw frame, so its yaw is
    # measured from the arm's own vertical plane - which is exactly R3.
    r1, r2, r3 = tool_euler(R_tool)

    return [ x, y, z, r1, r2, r3 ]

def forward_kinematics(servo_angles):
    # rads[narm] is the gripper - it has no effect on where the TCP ends up.
    rads = [radian(servo_to_angle(ch, deg)) for ch, deg in enumerate(servo_angles) ]

    pose = fk_pose(rads[:narm])

    # Self-check: the IK must land back on this pose. It need not land back on
    # the same joint angles - a pose the tool reaches pointing back over the
    # base has a second, equally valid arm configuration - so compare poses.
    rads2 = inverse_kinematics(pose)
    if rads2 is not None:
        pose2 = fk_pose(rads2)

        diffs = [ abs(p2 - p1) for p1, p2 in zip(pose[:3], pose2[:3]) ]
        diffs += [ abs(degree(normalize_radian(t2 - t1))) for t1, t2 in zip(pose[3:], pose2[3:]) ]

        diff_max = max(diffs)
        if 0.1 < diff_max:
            print(f'IK diff:{diff_max:.16f}', [ f'{d:.2f}' for d in diffs ])

    return arr(pose)

def tool_euler(R):
    """Split a tool orientation into (roll R1, pitch R2, yaw R3) such that
    R == rot_y(R2) @ rot_z(R3) @ rot_x(R1).

    Pitch is applied before yaw on purpose. R2 = 90 is the tool pointing
    straight down - the arm's normal picking attitude - and taking the pitch
    first keeps that an ordinary value rather than a singularity, and leaves
    R2 free to run past 90 as it always could. What is left over is R3, the
    angle the tool tilts out of the arm's own vertical plane, which is bounded
    to +/-90 and is 0 for every pose the arm could strike before J4 existed."""
    sin_yaw = min(1.0, max(-1.0, R[1, 0]))

    r3 = math.asin(sin_yaw)

    if abs(R[1, 0]) > 1 - 1e-18:
        # Tool pointing straight out of the arm's plane: the pitch and the roll
        # turn about the same axis, so put the whole turn into the roll.
        r2 = 0.0
        r1 = math.atan2(R[0, 2], -R[0, 1]) if 0 < r3 else -math.atan2(R[0, 2], R[0, 1])
    else:
        r2 = math.atan2(-R[2, 0], R[0, 0])
        r1 = math.atan2(-R[1, 2], R[1, 1])

    return r1, r2, r3

def wrist_angles(M):
    """Split M == rot_x(j3) @ rot_y(-j4) @ rot_x(j5) back into the three wrist
    joints.

    There are always two ways to do it: bend the wrist pitch the way it needs
    to go, or roll the forearm 180 degrees and bend it the other way. They put
    the tool in exactly the same place, but J4 only has about +/-90 degrees of
    travel, so the flipped one gets clamped by the firmware and lands the tool
    somewhere else entirely - and switching between them partway through an
    interpolated move is a violent 180 degree swing. So always take the one
    that keeps the forearm roll near its centre and let the wrist pitch carry
    whatever sign it needs."""
    cos_j4 = min(1.0, max(-1.0, M[0, 0]))

    beta = math.acos(cos_j4)            # magnitude of the wrist pitch, in [0, pi]

    if math.sin(beta) < 1e-9:
        # Tool colinear with the forearm: both rolls turn about the same axis,
        # so pin the forearm roll and let the tool roll take the whole turn.
        t = math.atan2(M[2, 1], M[1, 1])
        return 0.0, -beta, (t if 0 < cos_j4 else -t)

    j3 = math.atan2(M[1, 0], -M[2, 0])
    j5 = math.atan2(M[0, 1], M[0, 2])
    j4 = -beta

    if 0.5 * np.pi < abs(normalize_radian(j3)):
        j3 = normalize_radian(j3 + np.pi)
        j4 = beta
        j5 = normalize_radian(j5 + np.pi)

    return j3, j4, j5

def solve_arm(rho, zw, r1, r2, r3, tolerant, quiet=False):
    """Shoulder, elbow and wrist for a wrist centre at radius rho (signed, in
    the plane the base yaw swings) and height zw. Returns the five joints after
    the base yaw, or None if the shoulder/elbow cannot reach that far.

    `tolerant` accepts a wrist centre that is a whisker out of reach, pulling
    it back to full stretch; without it only an exactly reachable one will
    do."""
    # How far past full stretch still counts as "rounding". Near full
    # extension cos_alpha runs about 0.0027 per mm, so this is worth well
    # under a millimetre - float32 poses only need ~1e-5. It used to be 0.05,
    # which quietly accepted a target 18mm out of reach and answered with the
    # arm pulled straight, i.e. a silent 18mm positioning error.
    REACH_SLACK = 0.002
    p1 = Vec2(0, L0)
    pw = Vec2(rho, zw)

    l = (pw - p1).len()

    cos_alpha = (LF * LF + l * l - L1 * L1) / (2 * LF * l)
    if 1 < abs(cos_alpha):
        if not tolerant:
            return None
        elif REACH_SLACK < abs(cos_alpha) - 1:
            if not quiet:
                print(f'cos alpha:{cos_alpha:.16f} l:{l:.1f}  L1:{L1:.1f}  LF:{LF:.1f} p1:{p1} pw:{pw} rho:{rho:.1f} zw:{zw:.1f}')
            return None
        else:
            if not quiet:
                print(f'cos_alpha:{cos_alpha:.16f} => {np.sign(cos_alpha):.16f}')
            cos_alpha = np.sign(cos_alpha)

    alpha = math.acos(cos_alpha)

    # unit vector from the wrist centre to the shoulder
    ew1 = (p1 - pw).unit()

    # unit vector from the wrist centre to the elbow
    ew2 = ew1.rot(-alpha)

    p2 = pw + LF * ew2

    j1 = normalize_radian((p2 - p1).arctan2p() - 0.5 * np.pi)
    j2 = normalize_radian((pw - p2).arctan2p() - (p2 - p1).arctan2p())

    # ---- Wrist. The forearm orientation is now known and the tool orientation
    # is given, so what is left for the three wrist joints is a single
    # rotation - read off as roll / pitch / roll about the forearm.
    R_f = forearm_frame(0.5 * np.pi + j1 + j2)

    R_target = rot_y(r2) @ rot_z(r3) @ rot_x(r1)

    j3, j4, j5 = wrist_angles(R_f.T @ R_target)

    return [j1, j2, j3, j4, j5]

def facing_r1(x, y, z, r2, travel=None, iterations=8):
    """Tool roll that keeps the gripper (and whatever is mounted on it, e.g.
    tracking markers) facing the same absolute direction regardless of where
    the base has to swing to reach (x, y, z). R2 = 90 makes the tool axis
    vertical, so a roll about it (R1/J6) and the base yaw (J1) are rotations
    about the same physical axis - hence a fixed multiple of the bearing
    cancels the base swing exactly.

    That multiple is -1, not +1: the abstract kinematic model (this file's
    own rot_x/rot_y/rot_z convention) says +atan2(y, x) is the one that keeps
    a *modelled* orientation constant - verified against real hardware,
    though, the gripper visibly swings backwards with that sign. The
    likely reason is that J6's angle.py calibration (which datum servo
    position was recorded as its positive-degree reference) was not done
    against this model's rotation convention for the axis, so "positive
    wrist-roll arm-degrees" in the calibrated servo mapping runs opposite to
    what this file's geometry assumes - invisible before now because R1 had
    never been driven away from 0 in normal use. -atan2(y, x) is what
    actually cancels the swing on the real arm; do not flip this back to
    +atan2 by re-deriving the model math again, it was already re-derived
    once and confirmed self-consistent - the mismatch is between the model
    and this one servo's calibration, not in the math.

    Full cancellation needs more wrist travel the further off the X axis a
    target is, and can run out for a wide swing (a bearing of a few tens of
    degrees, roughly, depending on height) - so this scales the roll back
    toward 0 by bisection until the pose actually holds, rather than either
    ignoring the request (facing rotates with the base again) or asking for
    more than a joint has (silently clamped by the firmware to something
    else entirely). 0 itself must hold - that was the old fixed behaviour -
    or this has nothing to bisect toward and returns 0."""
    full = -math.atan2(y, x)

    if travel is None:
        travel = joint_travel()

    if holds_pose([x, y, z, full, r2, 0.0], travel):
        return full

    lo, hi = 0.0, full
    if not holds_pose([x, y, z, lo, r2, 0.0], travel):
        return 0.0

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if holds_pose([x, y, z, mid, r2, 0.0], travel):
            lo = mid
        else:
            hi = mid

    return lo

def joint_travel():
    """Arm-degree travel of every modelled joint, straight out of the servo
    calibration - a joint can only be commanded where its servo can go."""
    return [ angle_limits(ch) for ch in range(narm) ]

def holds_pose(pose, travel=None):
    """True when the IK solves this pose *and* every joint of the solution is
    inside its servo's travel. Geometry alone is not enough: the firmware
    silently clamps whatever it is sent, so a pose that solves but needs more
    travel than a servo has is one the arm will not actually hold."""
    rads = inverse_kinematics(pose, quiet=True)
    if rads is None:
        return False

    if travel is None:
        travel = joint_travel()

    return all(lo <= degree(t) <= hi for t, (lo, hi) in zip(rads, travel))

def nudge_into_reach(x, y, z, r2, step=2.0, min_radius=20.0, max_reach=400.0,
                      margin_deg=3.0):
    """Move (x, y) along its own bearing to the nearest radius the arm can
    really hold, and return where to aim instead. Staying on the bearing keeps
    the hand pointing at the work rather than sliding off to one side of it.

    Both ends need this, and neither can be had from the geometry alone. Out
    at the far end, holding the tool vertical leaves the wrist pitch to make up
    the whole angle from the forearm to straight down, and it runs out of
    travel sooner the higher the arm is working - a few mm of lost reach down
    at the table, tens of mm up at carrying height. In close, the arm runs out
    of room to fold up. So probe the real solve. Note the limit is a radius,
    never an x: a work off to one side can be well out of reach while its x
    still looks perfectly fine.

    margin_deg shrinks the travel checked here below the servo's actual limit.
    move_linear interpolates x/y/z/R1/R2/R3 in a straight line, and the set of
    poses a joint-limited arm can hold is not convex - so even when both ends
    of a move are individually fine, a pose partway along the straight line
    between them can dip slightly outside a joint's travel. Landing the target
    a few degrees clear of the true edge instead of right on it keeps that
    partway dip from actually crossing it."""
    travel = [ (lo + margin_deg, hi - margin_deg) for lo, hi in joint_travel() ]
    bearing = math.atan2(y, x)

    def at(r):
        return r * math.cos(bearing), r * math.sin(bearing)

    def holds(r):
        px, py = at(r)
        return holds_pose([px, py, z, 0.0, r2, 0.0], travel)

    r0 = math.hypot(x, y)
    if min_radius <= r0 <= max_reach and holds(r0):
        return x, y

    # Walk outwards from the target radius in both directions at once, so the
    # nearest radius that works is the one taken, whether the work was too far
    # out or too far in.
    for i in range(1, int(max_reach / step) + 1):
        for r in (r0 - i * step, r0 + i * step):
            if min_radius <= r <= max_reach and holds(r):
                return at(r)

    return x, y     # nothing on this bearing works; let the caller's IK say so

def max_radius(z, r2):
    """Largest sqrt(x*x + y*y) the TCP can reach at height z holding the tool
    at pitch r2 (radians). Holding the tool down costs reach - the wrist centre
    then sits most of L5 above the TCP, and the shoulder can only put it
    L1 + LF away - so how far out the arm gets depends on the height it is
    working at. Joint limits are not considered, only the geometry."""
    reach = L5 * math.cos(r2) + L4 * math.sin(r2)
    rise  = L4 * math.cos(r2) - L5 * math.sin(r2)

    dz = (z - rise) - L0
    span = L1 + LF

    if span * span <= dz * dz:
        return 0.0

    return math.sqrt(span * span - dz * dz) + reach

def inverse_kinematics(pose, quiet=False, prefer=None):
    """prefer, if given, is a set of 6 joint radians (e.g. wherever the arm
    currently is) to pick the closest solution to, whenever more than one
    branch reaches this pose. Without it, forward-reach is preferred, same as
    always - which is fine for a one-shot solve, but dangerous for anything
    jogging a *sequence* of nearby poses: a target that moves only slightly
    can still cross the boundary where the forward-reach branch stops being
    valid, and the fallback branch can require the base to swing round ~180
    degrees to the *other* side - confirmed to happen for a plain +/-5 degree
    step in R2 alone, at plenty of ordinary positions, not just extreme ones.
    Passing the current joints avoids that: nothing here needs to be reached
    "the normal way" for its own sake, only continuously."""
    x, y, z, r1, r2, r3 = pose

    # ---- Where the TCP sits relative to the wrist centre, in the base-yaw
    # frame. The tool axis is rot_y(r2) @ rot_z(r3) applied to the radial
    # direction, and the L4 offset is the same rotation applied to the
    # vertical, so all three components come straight out of the pose.
    off_x = L5 * math.cos(r2) * math.cos(r3) + L4 * math.sin(r2)
    off_y = L5 * math.sin(r3)
    off_z = L4 * math.cos(r2) - L5 * math.sin(r2) * math.cos(r3)

    # ---- Base yaw. off_y is how far the tool reaches out of the plane the arm
    # swings in, and it does not depend on the base angle, so the TCP's
    # distance from the base axis pins the in-plane component and hence j0.
    # With r3 = 0 this is just j0 = atan2(y, x), i.e. what the arm did before
    # J4 existed.
    disc = x * x + y * y - off_y * off_y
    if disc < 0:
        # r3 tilts the tool further out of the arm's plane than the TCP is far
        # from the base axis, so no base angle can line this up.
        if not quiet:
            print(f'IK unreachable tilt r3:{degree(r3):.1f} x:{x:.1f} y:{y:.1f} off_y:{off_y:.1f}')
        return None

    root = math.sqrt(disc)

    # Both roots put the TCP at (x, y) - one reaching forwards, one with the
    # base swung round to reach backwards. Every branch is tried for an exact
    # fit before any is allowed to stretch to reach, so a target that only
    # just fits does not get answered with the wrong branch pulled straight.
    #
    # Both roots are collected (not just the first that works) so `prefer`
    # can choose between them; without it, forward-reach still wins, same
    # order as before.
    for tolerant in (False, True):
        candidates = []
        for in_plane in (root, -root):
            rest = solve_arm(in_plane - off_x, z - off_z, r1, r2, r3, tolerant, quiet)
            if rest is not None:
                j0 = math.atan2(y, x) - math.atan2(off_y, in_plane)
                candidates.append([ normalize_radian(t) for t in [j0] + rest ])

        if candidates:
            if prefer is None:
                return candidates[0]

            def dist(cand):
                return max(abs(normalize_radian(a - b)) for a, b in zip(cand, prefer))

            return min(candidates, key=dist)

    return None
