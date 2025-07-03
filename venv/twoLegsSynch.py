import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ─────────────── CONSTANTS (metres) ───────────────
radiusUpper   = 38.0  / 1000
radiusLower   = 30.0  / 1000
link1_length  = 31   / 1000        # red
link2_length  = 104  / 1000        # blue
bar_length    = 124  / 1000        # orange
attach_dist   = 48.0 / 1000        # green (measured from orange top)
joint_ratio   = 0.24               # fraction along blue bar

OFFSET_X      = 180.0 / 1000       # ← 180 mm leg-to-leg spacing

# ─────────────── FIRST-LEG ANCHORS ────────────────
motor1_A  = np.array([0.0, 0.0])                     # upper motor
motor2_A  = motor1_A + np.array([0.0386, -0.0268])   # lower motor
right_B   = motor2_A  + np.array([4.0, 11.3]) / 1000 # blue-bar right anchor

# ─────────────── SECOND-LEG ANCHORS ───────────────
motor2_A2 = motor2_A + np.array([OFFSET_X, 0.0])     # lower motor (leg-2)
motor1_A2 = motor2_A2 - (motor2_A - motor1_A)        # upper motor keeps same offset
right_B2  = motor2_A2 + np.array([4.0, 11.3]) / 1000

# ─────────────── FIGURE SET-UP ────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)
ax.set_aspect('equal')
ax.set_xlim(-0.10, OFFSET_X + 0.22)
ax.set_ylim(-0.22, 0.15)
ax.set_title('Two-Leg 2-Link System with Motor Swivels')
ax.axis('off')

# anchor dots
ax.plot(*motor1_A , 'ko')
ax.plot(*motor2_A , 'ko')
ax.plot(*right_B  , 'mo')
ax.plot(*motor1_A2, 'ko')
ax.plot(*motor2_A2, 'ko')
ax.plot(*right_B2 , 'mo')

# first-leg artists
l1, = ax.plot([], [], 'r-',  lw=3)          # red
l2, = ax.plot([], [], 'b-',  lw=3)          # blue
l3, = ax.plot([], [], 'g-',  lw=3)          # green
l4, = ax.plot([], [], color='orange', lw=3) # orange
tip1, = ax.plot([], [], 'go')
tip2, = ax.plot([], [], 'go')

# second-leg artists (slightly lighter colours for clarity)
l1b, = ax.plot([], [], 'r--',  lw=3)
l2b, = ax.plot([], [], 'b--',  lw=3)
l3b, = ax.plot([], [], 'g--',  lw=3)
l4b, = ax.plot([], [], color='orange', ls='--', lw=3)
tip1b, = ax.plot([], [], 'go')
tip2b, = ax.plot([], [], 'go')

# sliders
ax_s1 = plt.axes([0.20, 0.10, 0.65, 0.03])
ax_s2 = plt.axes([0.20, 0.05, 0.65, 0.03])
s1 = Slider(ax_s1, 'Motor 1 Angle', 0, 40, valinit=0)
s2 = Slider(ax_s2, 'Motor 2 Angle', 0, 40, valinit=0)

# ─────────────── KINEMATIC SOLVER ─────────────────
def solve_leg(m1, m2, right_anchor, theta1_deg, theta2_deg):
    """Return all key points (tip1,joint1,left_joint,orange_bot,tip2)."""
    t1 = np.radians(135 + theta1_deg)
    t2 = np.radians(-45 - theta2_deg)
    tip1 = m1 + radiusUpper * np.array([np.cos(t1), np.sin(t1)])
    tip2 = m2 + radiusLower * np.array([np.cos(t2), np.sin(t2)])

    # ---- red & blue linkage ---------------------------------------
    r, d = link1_length, link2_length * (1 - joint_ratio)
    D     = right_anchor - tip1
    dist  = np.linalg.norm(D)
    if dist > r + d or dist < abs(r - d) or (dist == 0 and r == d):
        return None                     # unreachable pose

    a = (r**2 - d**2 + dist**2) / (2 * dist)
    h = np.sqrt(abs(r**2 - a**2))
    midpoint = tip1 + a * D / dist
    perp     = np.array([-D[1], D[0]]) / dist
    joint1   = midpoint - h * perp                    # red-blue hinge

    full_vec   = (right_anchor - joint1) / (1 - joint_ratio)
    full_vec   = full_vec / np.linalg.norm(full_vec) * link2_length
    left_joint = right_anchor - full_vec              # orange top

    # ---- orange bar bottom via circle-circle ----------------------
    vec_lo = tip2 - left_joint
    d_lo   = np.linalg.norm(vec_lo)
    R = bar_length
    if d_lo == 0 or d_lo > 2 * R:
        return None
    a2   = d_lo / 2
    h2   = np.sqrt(R**2 - a2**2)
    base = left_joint + a2 * vec_lo / d_lo
    perp2 = np.array([-vec_lo[1], vec_lo[0]]) / d_lo
    cand1, cand2 = base + h2 * perp2, base - h2 * perp2
    orange_bottom = cand1 if cand1[1] < cand2[1] else cand2

    # ---- green link (foot) ---------------------------------------
    axis_len   = np.linalg.norm(orange_bottom - left_joint)
    if axis_len == 0 or attach_dist > axis_len:
        return None
    dir_orange = (orange_bottom - left_joint) / axis_len
    green_start = left_joint + dir_orange * attach_dist  # fixed point on orange

    return tip1, joint1, left_joint, orange_bottom, tip2, green_start

# ─────────────── UPDATE FUNCTION ─────────────────
def update(_):
    # first (left-side) leg
    pts = solve_leg(motor1_A, motor2_A, right_B, s1.val, s2.val)
    if pts:
        tipA, J1, LJ, OB, tipB, GS = pts
        l1.set_data([tipA[0], J1[0]], [tipA[1], J1[1]])
        l2.set_data([LJ[0],  right_B[0]], [LJ[1],  right_B[1]])
        l3.set_data([GS[0],  tipB[0]],   [GS[1],  tipB[1]])
        l4.set_data([LJ[0],  OB[0]],     [LJ[1],  OB[1]])
        tip1.set_data([tipA[0]], [tipA[1]])
        tip2.set_data([tipB[0]], [tipB[1]])

    # second (right-side) leg
    pts2 = solve_leg(motor1_A2, motor2_A2, right_B2, s1.val, s2.val)
    if pts2:
        tipA2, J12, LJ2, OB2, tipB2, GS2 = pts2
        l1b.set_data([tipA2[0], J12[0]], [tipA2[1], J12[1]])
        l2b.set_data([LJ2[0],  right_B2[0]], [LJ2[1], right_B2[1]])
        l3b.set_data([GS2[0],  tipB2[0]],    [GS2[1], tipB2[1]])
        l4b.set_data([LJ2[0],  OB2[0]],      [LJ2[1], OB2[1]])
        tip1b.set_data([tipA2[0]], [tipA2[1]])
        tip2b.set_data([tipB2[0]], [tipB2[1]])

    fig.canvas.draw_idle()

s1.on_changed(update)
s2.on_changed(update)
update(None)
plt.show()
